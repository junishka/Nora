"""Researcher-staged file manifest for the per-session cwd.

The script sandbox at ``executor.py`` intentionally allows writes to
the session cwd — ``saveRDS`` / ``df.to_csv`` / ``save "panel.dta"``
are part of the normal R/Stata/Python workflow. That same surface,
however, lets a model-authored script write raw row values into a
file with a script extension (``data_dump.R``, ``smuggled.py``) and
then later recall the bytes through ``read_attached_file`` /
``submit_script_file``, bypassing the SDC sanitizer that gates every
``submit_script`` result payload.

This module is the closure for that gap. The manifest at
``<cwd>/.nora/staged_files.json`` records the basename of every cwd
top-level file Nora considers researcher-known:

  * Files present in cwd at session-open time (the initial snapshot
    — the researcher staged them before opening Nora).
  * Files added through the bridge's file-staging endpoints
    (``add_files`` native picker, ``add_files_from_blobs`` paste/
    drop, ``upload_files`` landing-page drop). These also originate
    from explicit researcher action.

Any cwd top-level file NOT in the manifest is presumed sandbox-
output. ``read_attached_file`` and ``submit_script_file`` consult
``is_known`` on every cwd-resolved candidate and refuse with a
clear reason when the basename is unknown.

The manifest lives under ``<cwd>/.nora/`` which the analysis sandbox
already deny-reads/writes — a model script can't read or tamper
with it. Writes are atomic via tempfile + ``os.replace`` so a crash
mid-write leaves either the prior snapshot intact or the new one
fully written, never a half-truncated JSON.

Append-only by design: we never ``remove`` a name. A file the
researcher staged once stays known even after they delete it from
disk and re-stage a different one with the same name (the bridge's
own ``already_attached`` short-circuit handles the duplicate-name
case before reaching the manifest). The append-only shape also
makes the file degenerate gracefully across upgrades — a future
schema change can add fields without invalidating the existing
``names`` list.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from collections.abc import Iterable
from pathlib import Path


MANIFEST_FILENAME = "staged_files.json"
MANIFEST_VERSION = 1

# Per-cwd lock so the bridge's file-staging endpoints (which can fire
# concurrently when the researcher drops a folder of files) don't
# race on the read-modify-write cycle. Same shape as the lock in
# ``session_state.py``; not shared because the two manifests live in
# separate files and locking them together would serialise unrelated
# writes.
_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def _lock_for(cwd: Path) -> threading.Lock:
    """Return the per-cwd lock, creating it on first call."""
    key = str(cwd.resolve())
    with _LOCKS_GUARD:
        lock = _LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _LOCKS[key] = lock
        return lock


def _manifest_path(cwd: Path) -> Path:
    return cwd / ".nora" / MANIFEST_FILENAME


def _read_names(path: Path) -> set[str]:
    """Read the manifest. Missing / unreadable / malformed manifests
    return ``set()`` — callers treat that as "nothing is staged yet"
    rather than crashing the read. The atomic-write path means a
    half-written manifest cannot be observed; corruption only
    surfaces if the file was edited externally.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return set()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return set()
    names = data.get("names") if isinstance(data, dict) else None
    if not isinstance(names, list):
        return set()
    return {n for n in names if isinstance(n, str) and n}


def _write_names(path: Path, names: set[str]) -> None:
    """Atomic write: tmpfile in the same directory, then ``os.replace``.

    Same posture as ``policy.save_policy`` — direct ``write_text`` on
    a manifest the bridge re-writes on every staging event would
    leave half-written JSON observable to a concurrent read after a
    crash, and the read would silently start over with an empty
    set. ``os.replace`` is a true atomic rename within one
    filesystem.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        {"version": MANIFEST_VERSION, "names": sorted(names)},
        indent=2, sort_keys=True,
    ) + "\n"
    fd, tmp_name = tempfile.mkstemp(
        prefix=".staged_files.json.", suffix=".tmp", dir=path.parent,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _enumerate_cwd_top_level(cwd: Path) -> set[str]:
    """Return basenames of all regular files at cwd top level. Skips
    directories, symlinks (they could point outside cwd; the bridge
    stages by basename only, so a symlink target couldn't go through
    a legitimate add path), and dotfiles (Nora's own state lives
    under ``.nora/`` and any other dotfiles are researcher tooling
    that isn't part of the analysis surface).
    """
    out: set[str] = set()
    try:
        for child in cwd.iterdir():
            if not child.is_file() or child.is_symlink():
                continue
            if child.name.startswith("."):
                continue
            out.add(child.name)
    except OSError:
        pass
    return out


def initialize(cwd: Path) -> set[str]:
    """Snapshot cwd top-level files into the manifest at FIRST session-
    open only.

    On the very first open the cwd snapshot is presumed researcher-
    staged: the researcher dropped those files there before opening
    Nora. Once a manifest exists, subsequent re-opens MUST NOT
    re-snapshot — between sessions, the analysis sandbox may have
    written its own files into cwd (``df.to_csv("out.csv")`` is
    legitimate; ``open("smuggled.py", "w").write(...)`` from a
    model-authored script is the gap). Merging those in on reopen
    would silently promote sandbox output to "researcher-staged"
    and let ``read_attached_file`` / ``submit_script_file`` /
    ``search_in_session_files`` return their bytes — the same SDC
    bypass the manifest exists to prevent. The provenance guard
    must be effective across app restarts, not just within one
    live session.

    Backwards compatibility for sessions that pre-date this manifest:
    when the manifest file does not yet exist, we snapshot once and
    write it (the upgrade path). After that the manifest is the sole
    authority; new files added via the bridge's staging endpoints
    (``add_files`` / ``add_files_from_blobs`` / ``upload_files``)
    extend it through ``mark_known``.

    Returns the resulting name set so callers can log it.
    """
    path = _manifest_path(cwd)
    with _lock_for(cwd):
        # ``read_names`` returns ``set()`` for missing OR corrupt
        # manifests. We need to distinguish those: missing -> seed,
        # corrupt -> leave alone (don't silently seed an empty
        # manifest on top of a corrupt one and resnapshot whatever
        # is in cwd right now). ``path.exists()`` is the gate.
        if path.exists():
            return _read_names(path)
        snapshot = _enumerate_cwd_top_level(cwd)
        _write_names(path, snapshot)
        return snapshot


def mark_known(cwd: Path, names: Iterable[str]) -> set[str]:
    """Add the given basenames to the manifest. Returns the resulting
    full set so the caller can log the new entries if it wants.

    ``Path(name).name`` is used to defensively basename the input —
    callers should already be passing basenames, but a stray
    ``/foo/bar.csv`` wouldn't smuggle a path-shaped key in.
    """
    cleaned = {Path(n).name for n in names if n}
    if not cleaned:
        return _read_names(_manifest_path(cwd))
    path = _manifest_path(cwd)
    with _lock_for(cwd):
        merged = _read_names(path) | cleaned
        _write_names(path, merged)
        return merged


def is_known(cwd: Path, name: str) -> bool:
    """Whether ``name`` (basename) is in the manifest.

    Refuses path-traversal-shaped inputs by basenaming first; a
    caller that passed ``foo/../bar.csv`` gets the same answer as if
    they passed ``bar.csv``.
    """
    safe = Path(name).name
    if not safe:
        return False
    # No lock required for reads — the writer's atomic ``os.replace``
    # means readers see either the old manifest or the new one, never
    # a partial state.
    return safe in _read_names(_manifest_path(cwd))


def known_names(cwd: Path) -> set[str]:
    """Return the full manifest set (test/diagnostic helper)."""
    return _read_names(_manifest_path(cwd))
