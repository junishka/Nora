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
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nora.chat_history import read_turns


SESSION_STATE_FILENAME = "session_state.json"
SESSION_STATE_VERSION = 1

# Per-field caps for text we snapshot. These stay short because the
# state file is meant to be glanceable, not a re-encoding of the full
# transcript. Callers who need the full exchange use chat_history.
_LAST_MESSAGE_CAP = 800
_RECENT_RESULTS_CAP = 10
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
    turns = read_turns(cwd)
    last_user = ""
    last_assistant = ""
    for t in reversed(turns):
        if t.user:
            last_user = _truncate(t.user, _LAST_MESSAGE_CAP)
            if t.assistant:
                last_assistant = _truncate(t.assistant, _LAST_MESSAGE_CAP)
            break

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

    state = SessionState(
        version=SESSION_STATE_VERSION,
        last_active_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        turn_count=len(turns),
        last_user_message=last_user,
        last_assistant_summary=last_assistant,
        recent_results=recent,
        datasets=datasets,
        active_model=model,
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
        path.parent.mkdir(parents=True, exist_ok=True)
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
