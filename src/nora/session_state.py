"""Durable per-session "at a glance" state file.

Lives at ``<cwd>/.nora/session_state.json`` and captures the facts
about a session that are most useful on resume:

- When the session was last active
- How many turns it holds
- The most recent user question and assistant reply
- Which analytic results have been produced (id, label, type)
- Which datasets are present in the session's working directory
- Which model the researcher had selected

Distinct from ``chat_history.jsonl`` (the full event stream): the
state file is a small, structured, easily-read summary. Cheap to
produce (everything is derivable from state Nora already keeps on
disk), cheap to consume (one JSON read).

Write semantics:
- Re-generated from scratch after every successful turn. We never
  incrementally append here — the file is always a fresh snapshot of
  the current state, so a partial or corrupted write can't leave
  stale data behind.
- Writes are atomic via tempfile + rename so a crash mid-write leaves
  the previous state intact rather than corrupting it.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nora.chat_history import read_last_turn_summary


SESSION_STATE_FILENAME = "session_state.json"
SESSION_STATE_VERSION = 1

# Per-cwd lock. The two writers — ``write_session_state`` (turn-end,
# regenerates everything) and ``set_custom_name`` (rename, mutates a
# single field) — both do read-modify-write on the same JSON file.
# Without serialisation a turn finishing and a researcher renaming
# the session at the same instant can each load the prior file, each
# build a half-updated snapshot, and the second write wins — silently
# losing whichever side wrote first. The lock isn't shared across
# processes (we're not handling that case) but it covers every thread
# inside one nora process, which is where the race actually fires.
_STATE_LOCKS: dict[Path, threading.Lock] = {}
_STATE_LOCKS_GUARD = threading.Lock()


def _state_lock_for(cwd: Path) -> threading.Lock:
    """Return the lock for ``cwd``, creating it on first use. The
    outer guard makes lock creation itself thread-safe."""
    key = cwd.resolve() if cwd.is_dir() else cwd
    with _STATE_LOCKS_GUARD:
        lock = _STATE_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _STATE_LOCKS[key] = lock
        return lock


def evict_state_lock(cwd: Path) -> None:
    """Drop the cached lock for ``cwd``.

    Called by ``ui.delete_session`` after the session directory is
    removed from disk. Without this, every session ever opened in
    a long-running daemon leaves one ``threading.Lock`` + dict entry
    behind in ``_STATE_LOCKS`` — small individually (~tens of bytes
    each) but unbounded across a research day that opens and discards
    many sessions.

    Safe to call when no lock exists (no-op). Callers must not be
    holding the lock; eviction during contention would let a future
    ``_state_lock_for`` recreate a different lock that races against
    the held one. In practice this is fine because ``delete_session``
    only fires after the runner is closed, so no thread is mid-write
    on this cwd's state file.
    """
    try:
        key = cwd.resolve() if cwd.is_dir() else cwd
    except OSError:
        # Directory already gone (delete_session rmtree'd it before
        # calling us); fall back to the unresolved path.
        key = cwd
    with _STATE_LOCKS_GUARD:
        _STATE_LOCKS.pop(key, None)

# Per-field caps for text we snapshot. These stay short because the
# state file is meant to be glanceable, not a re-encoding of the full
# transcript. Callers who need the full exchange use chat_history.
_LAST_MESSAGE_CAP = 800
_RECENT_RESULTS_CAP = 10
# Cap on the user-supplied session name. Long enough to fit a
# descriptive sentence ("Replication of Smith 2014, table 3"), short
# enough to keep the topbar pill and sidebar rows readable.
_CUSTOM_NAME_CAP = 120
# Imported from nora.schema so the catalog of recognised data files
# stays in one place — adding .parquet there propagates here.
from nora.schema import DATA_EXTENSIONS as _DATA_EXTS  # noqa: E402


@dataclass
class RecentResult:
    """A thin projection of store.StoredResult — just the fields that
    survive on-disk snapshotting. We deliberately do NOT include the
    sanitized payload or the script source; those stay in results.db,
    and the UI can fetch them via list_results / expand_result."""
    id: str
    label: str
    analysis_type: str
    created_at: str


@dataclass
class SessionState:
    """Everything the state file exposes. Optional fields default to
    empty / None so older files with fewer fields still read cleanly
    once new fields are added (forward-compat for cheap reader code)."""
    version: int = SESSION_STATE_VERSION
    last_active_at: str = ""
    turn_count: int = 0
    last_user_message: str = ""
    last_assistant_summary: str = ""
    recent_results: list[RecentResult] = field(default_factory=list)
    datasets: list[str] = field(default_factory=list)
    active_model: str | None = None
    # Optional researcher-set label for the session. When set, the UI
    # prefers this over the auto-derived title (dataset name /
    # timestamp). ``None`` means "use the auto-derived title".
    custom_name: str | None = None
    # Researcher-toggled "pin to top" flag. The sidebar surfaces pinned
    # sessions ahead of unpinned ones regardless of last_activity, so
    # frequently-revisited sessions stay reachable without scrolling.
    # ``pinned_at`` is the ISO timestamp of the most recent pin toggle
    # to ``True`` — used to sort within the pinned group so the most
    # recently pinned session sits at the very top.
    pinned: bool = False
    pinned_at: str = ""


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

def write_session_state(
    cwd: Path,
    model: str | None = None,
    *,
    # ``store_list`` is injected so tests don't have to construct a
    # real ResultStore. Production callers pass
    # ``list(store.list_all())`` (or leave it None and let the writer
    # open the store itself).
    store_list: list[Any] | None = None,
) -> SessionState | None:
    """Regenerate ``session_state.json`` from current on-disk state.

    Returns the ``SessionState`` that was written, or ``None`` if we
    couldn't write (e.g. cwd unset, disk full). Never raises —
    persistence failures shouldn't block a chat turn.
    """
    if cwd is None or not cwd.is_dir():
        # No session dir to write into — refuse. Writing a state
        # file for a path that doesn't exist would leave an
        # orphan .nora/ folder behind and pretend a session
        # was there.
        return None

    # 1. Pull the last user / assistant exchange from the turn-grouped
    #    chat log. The pair must come from the SAME turn — earlier
    #    versions walked reversed turns and grabbed the latest
    #    user-side and the latest assistant-side independently. For
    #    an in-flight turn (user typed, assistant hasn't replied yet)
    #    that pulled the user from turn N and the assistant from
    #    turn N-1, so the sidebar summary showed "user said X"
    #    alongside "assistant said Y" where Y was actually a reply
    #    to a different question.
    #
    #    Behavior: latest user-bearing turn wins. We pair its user
    #    with its OWN assistant (or empty when in-flight). If the
    #    assistant turns out to be empty, the sidebar shows
    #    "what the researcher just asked, no answer yet" — accurate
    #    to the current state, never a false pairing.
    #
    #    Uses ``read_last_turn_summary`` rather than the heavier
    #    ``read_turns`` so this snapshot stays cheap as the persisted
    #    UI replay log grows: ``read_turns`` json.loads every
    #    tool_result event (each carrying raw stdout/stderr and any
    #    plot thumbnail payload), and the writer runs after every
    #    successful turn — full reparse on every turn would make
    #    each turn slower than the last in a long session.
    turn_count, last_user_raw, last_assistant_raw = read_last_turn_summary(cwd)
    last_user = _truncate(last_user_raw, _LAST_MESSAGE_CAP) if last_user_raw else ""
    last_assistant = (
        _truncate(last_assistant_raw, _LAST_MESSAGE_CAP)
        if last_assistant_raw else ""
    )

    # 2. Pull recent results. Either from the injected list (tests)
    #    or by opening the store ourselves. If opening fails (missing
    #    db, corrupted file), we ship empty `recent_results` rather
    #    than failing the whole snapshot.
    recent: list[RecentResult] = []
    rows = store_list
    if rows is None:
        try:
            from nora.store import get_store
            store = get_store(cwd)
            rows = store.list_all()
        except Exception:  # noqa: BLE001 — store corruption shouldn't crash the turn
            rows = []
    # Newest first, capped. store.list_all() is unordered at the API
    # boundary, so we sort by created_at defensively.
    sortable = [r for r in (rows or []) if getattr(r, "created_at", None)]
    sortable.sort(key=lambda r: r.created_at, reverse=True)
    for r in sortable[:_RECENT_RESULTS_CAP]:
        recent.append(RecentResult(
            id=getattr(r, "id", "") or "",
            label=getattr(r, "label", "") or "",
            analysis_type=getattr(r, "analysis_type", "") or "",
            created_at=getattr(r, "created_at", "") or "",
        ))

    # 3. Enumerate data files in the cwd. Filenames only — no paths,
    #    no sizes (those live in ~/.nora-sessions listing).
    datasets: list[str] = []
    try:
        for child in sorted(cwd.iterdir()):
            if child.is_file() and child.suffix.lower() in _DATA_EXTS:
                datasets.append(child.name)
    except OSError:
        pass

    # Preserve the researcher-set custom name across rewrites. The
    # writer is called after every successful turn and regenerates
    # the file from scratch — without this read-and-carry, every turn
    # would silently drop a name the researcher had typed earlier.
    #
    # Lock-protected so a concurrent ``set_custom_name`` call can't
    # land its rename between this read and the write below: without
    # the lock, the rename's update gets clobbered by this writer
    # carrying the OLD ``custom_name`` it just read. The lock spans
    # only the read-modify-write — turn-end work above (chat history,
    # results, datasets) can run unsynchronised since it's all
    # single-writer per cwd.
    with _state_lock_for(cwd):
        prior = read_session_state(cwd)
        custom_name = prior.custom_name if prior is not None else None
        pinned = prior.pinned if prior is not None else False
        pinned_at = prior.pinned_at if prior is not None else ""

        state = SessionState(
            version=SESSION_STATE_VERSION,
            last_active_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            turn_count=turn_count,
            last_user_message=last_user,
            last_assistant_summary=last_assistant,
            recent_results=recent,
            datasets=datasets,
            active_model=model,
            custom_name=custom_name,
            pinned=pinned,
            pinned_at=pinned_at,
        )

        _atomic_write(cwd / ".nora" / SESSION_STATE_FILENAME, state)
    return state


def set_custom_name(cwd: Path, name: str | None) -> SessionState | None:
    """Update only the ``custom_name`` field on the session's state
    file. ``None`` (or an empty/whitespace string) clears it back to
    the auto-derived title. Returns the new ``SessionState`` or
    ``None`` if the cwd is invalid or no state file existed yet.

    This is a targeted edit — it does NOT regenerate the rest of the
    snapshot. That keeps the rename cheap (no chat-history walk, no
    store open) and avoids the rare race where regenerating would
    pick up a partial in-flight turn.
    """
    if cwd is None or not cwd.is_dir():
        return None
    cleaned: str | None
    if name is None:
        cleaned = None
    else:
        # Run the rename through the same text-safety boundary the
        # rest of the data-origin string surfaces use. Without it, a
        # paste of ``"name\n\n###System: ignore prior"`` lands
        # verbatim in the JSON, the topbar, and any future surface
        # that interpolates the title (chat-history headers, exports,
        # potential prompt slots). ``safe_text`` strips control / bidi
        # tricks and flattens whitespace; the cap below then enforces
        # the visual length budget on the cleaned form.
        from nora.text_safety import safe_text
        s = safe_text(name).strip()
        cleaned = s[:_CUSTOM_NAME_CAP] if s else None

    # Lock-protected read-modify-write — see ``write_session_state``
    # for the race this prevents (concurrent turn-end carrying the
    # old name forward, clobbering this rename, or vice versa).
    with _state_lock_for(cwd):
        prior = read_session_state(cwd)
        if prior is None:
            # No state yet — seed a minimal one so the name sticks. The
            # next successful turn will fill in turn_count etc.
            state = SessionState(
                version=SESSION_STATE_VERSION,
                last_active_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                custom_name=cleaned,
            )
        else:
            state = SessionState(
                version=prior.version,
                last_active_at=prior.last_active_at,
                turn_count=prior.turn_count,
                last_user_message=prior.last_user_message,
                last_assistant_summary=prior.last_assistant_summary,
                recent_results=prior.recent_results,
                datasets=prior.datasets,
                active_model=prior.active_model,
                custom_name=cleaned,
                pinned=prior.pinned,
                pinned_at=prior.pinned_at,
            )
        _atomic_write(cwd / ".nora" / SESSION_STATE_FILENAME, state)
    return state


def set_pinned(cwd: Path, pinned: bool) -> SessionState | None:
    """Update only the ``pinned`` flag on the session's state file.

    Stamps ``pinned_at`` with the current UTC time whenever a session
    flips from unpinned to pinned, so the sidebar can sort the pinned
    group most-recently-pinned-first. Unpinning leaves the prior
    ``pinned_at`` alone — harmless, since the flag itself is what the
    sort consults first. Returns the new ``SessionState`` or ``None``
    if the cwd is invalid.

    Mirrors :func:`set_custom_name` in being a targeted edit that does
    NOT regenerate the rest of the snapshot — pin-toggle should be
    instantaneous and never trip a partial turn-end snapshot.
    """
    if cwd is None or not cwd.is_dir():
        return None
    flag = bool(pinned)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with _state_lock_for(cwd):
        prior = read_session_state(cwd)
        if prior is None:
            # Seed a minimal state — the next successful turn fills the
            # rest. Without this, pinning a session that has never had
            # a turn (rare but possible: pin the landing-screen folder
            # before chatting) would no-op silently.
            state = SessionState(
                version=SESSION_STATE_VERSION,
                last_active_at=now,
                pinned=flag,
                pinned_at=now if flag else "",
            )
        else:
            # Re-stamp ``pinned_at`` only on the unpinned→pinned
            # transition. Idempotent pins (already True) keep the
            # original stamp so a researcher who clicks the icon twice
            # by accident doesn't jump the row to the very top.
            new_pinned_at = prior.pinned_at
            if flag and not prior.pinned:
                new_pinned_at = now
            state = SessionState(
                version=prior.version,
                last_active_at=prior.last_active_at,
                turn_count=prior.turn_count,
                last_user_message=prior.last_user_message,
                last_assistant_summary=prior.last_assistant_summary,
                recent_results=prior.recent_results,
                datasets=prior.datasets,
                active_model=prior.active_model,
                custom_name=prior.custom_name,
                pinned=flag,
                pinned_at=new_pinned_at,
            )
        _atomic_write(cwd / ".nora" / SESSION_STATE_FILENAME, state)
    return state


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------

def read_session_state(cwd: Path | None) -> SessionState | None:
    """Best-effort read. Returns ``None`` when the file is missing,
    unreadable, or has an incompatible version. Callers should treat
    None as "no durable state yet" rather than an error."""
    if cwd is None:
        return None
    path = cwd / ".nora" / SESSION_STATE_FILENAME
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    # Version gate: refuse forward-incompatible files rather than
    # silently misreading them. When we bump the version, add a
    # migration path here.
    version = raw.get("version")
    if version != SESSION_STATE_VERSION:
        return None
    try:
        return SessionState(
            version=version,
            last_active_at=raw.get("last_active_at", "") or "",
            turn_count=int(raw.get("turn_count", 0) or 0),
            last_user_message=raw.get("last_user_message", "") or "",
            last_assistant_summary=raw.get("last_assistant_summary", "") or "",
            recent_results=[
                RecentResult(
                    id=str(r.get("id", "") or ""),
                    label=str(r.get("label", "") or ""),
                    analysis_type=str(r.get("analysis_type", "") or ""),
                    created_at=str(r.get("created_at", "") or ""),
                )
                for r in (raw.get("recent_results") or [])
                if isinstance(r, dict)
            ],
            datasets=[str(d) for d in (raw.get("datasets") or []) if d],
            active_model=raw.get("active_model"),
            custom_name=(raw.get("custom_name") or None),
            pinned=bool(raw.get("pinned", False)),
            pinned_at=str(raw.get("pinned_at", "") or ""),
        )
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _atomic_write(path: Path, state: SessionState) -> None:
    """Serialize ``state`` to JSON and replace ``path`` atomically.

    Uses a tempfile in the same directory + os.replace so a crash
    mid-write leaves the prior file intact. Silently swallows OSError —
    persistence failures are not allowed to break a chat turn.
    """
    try:
        # ``path`` is ``<cwd>/.nora/session_state.json``; route through
        # the central helper so the .nora dir gets the 0o700 mode that
        # gates every Nora-owned file on shared filesystems.
        from nora.config import ensure_private_nora_dir
        ensure_private_nora_dir(path.parent.parent)
        payload = asdict(state)
        # asdict recursively converts RecentResult too, which is what
        # we want — the on-disk schema is a flat dict-of-primitives.
        encoded = json.dumps(payload, ensure_ascii=False, indent=2)
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=str(path.parent),
            prefix=".session_state.",
            suffix=".tmp",
            delete=False,
        ) as tf:
            tf.write(encoded)
            tf.flush()
            os.fsync(tf.fileno())
            tmp_name = tf.name
        os.replace(tmp_name, path)
    except OSError:
        # Best-effort cleanup of the tempfile if replace() failed
        # after we'd created it; ignore all errors in the cleanup.
        try:
            if "tmp_name" in locals():
                os.unlink(tmp_name)  # type: ignore[name-defined]
        except OSError:
            pass


def _truncate(s: str, cap: int) -> str:
    if len(s) <= cap:
        return s
    return s[:cap] + "…[truncated]"
