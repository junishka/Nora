"""Persistent registry of folder-backed sessions.

``choose_folder`` lets the researcher open an arbitrary directory as
a session, bypassing the staging step that lands every other session
under ``~/.nora-sessions/``. Without a registry those folder-backed
sessions disappear from the sidebar the moment the researcher
switches away — ``list_sessions`` only enumerates direct children of
``SESSIONS_ROOT``, and ``switch_session`` refuses any path whose
parent isn't ``SESSIONS_ROOT``. After an app restart the path is lost
entirely.

This module keeps a tiny JSON file alongside the sessions root that
records each folder the researcher has opened via the picker, with a
timestamp. The list is capped at the most-recent N entries so it
doesn't grow unbounded; pruning is by registration time, not folder
mtime, because the goal is "remember the last few project dirs I
worked in" rather than "track everything I ever touched."

Entries whose path no longer exists on disk are filtered out at read
time so a deleted project directory doesn't haunt the sidebar.
Registration is idempotent: re-registering the same path bumps its
timestamp to the front of the list.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

# File name lives at ``<sessions_root>/.external_sessions.json``. The
# leading dot keeps it out of casual sidebar enumeration (the sessions-
# root iterdir loop filters by ``is_dir()`` so this never appears as a
# phantom session even without a name check; the dot just keeps the
# top of the directory tidy for researchers who ``ls`` it).
_REGISTRY_FILENAME = ".external_sessions.json"

# Maximum entries retained. Folder-backed sessions are typically project
# directories — a researcher doesn't usually have hundreds of active
# projects. 50 covers heavy power-users without unbounded growth on the
# sidebar.
_MAX_ENTRIES = 50


def _registry_path(sessions_root: Path) -> Path:
    return sessions_root / _REGISTRY_FILENAME


def _read_raw(sessions_root: Path) -> list[dict[str, Any]]:
    """Read the on-disk registry; return [] on any failure or missing file.

    Format on disk:
        {"sessions": [{"path": "/abs/path", "registered_at": 1234.0}, ...]}

    Any malformed entry is dropped silently rather than refusing the
    whole list — a corrupted file shouldn't take out the sidebar.
    """
    path = _registry_path(sessions_root)
    if not path.is_file():
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(raw, dict):
        return []
    sessions = raw.get("sessions")
    if not isinstance(sessions, list):
        return []
    out: list[dict[str, Any]] = []
    for entry in sessions:
        if not isinstance(entry, dict):
            continue
        p = entry.get("path")
        ts = entry.get("registered_at")
        if not isinstance(p, str) or not isinstance(ts, (int, float)):
            continue
        out.append({"path": p, "registered_at": float(ts)})
    return out


def _write_raw(sessions_root: Path, entries: list[dict[str, Any]]) -> None:
    """Atomic write of the registry. Best-effort: a write failure is
    swallowed because losing one registration is preferable to taking
    out a working session by raising up the call chain.

    Atomic via ``os.replace`` on a same-directory tempfile so a
    crash mid-write can't leave a partial JSON on disk.
    """
    try:
        sessions_root.mkdir(parents=True, exist_ok=True)
    except OSError:
        return
    payload = json.dumps({"sessions": entries}, separators=(",", ":"))
    try:
        # ``delete=False`` because we close + os.replace ourselves; the
        # ``delete_on_close`` flag is 3.12+. Same-directory tempfile so
        # ``os.replace`` is atomic on POSIX (cross-fs replace is not).
        fd, tmp_path = tempfile.mkstemp(
            prefix=_REGISTRY_FILENAME + ".",
            suffix=".tmp",
            dir=str(sessions_root),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(payload)
            os.replace(tmp_path, _registry_path(sessions_root))
        except OSError:
            # Clean up the tempfile if replace failed mid-flight.
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    except OSError:
        pass


def _normalize(path: Path) -> str:
    """Return the canonical absolute string used as the registry key.

    ``resolve(strict=False)`` so we can register a path that exists
    today; existence is re-checked at read time so a later delete
    naturally evicts the entry.
    """
    return str(path.expanduser().resolve())


def register(sessions_root: Path, folder: Path) -> None:
    """Record ``folder`` as a folder-backed session.

    Idempotent: if the folder is already in the registry, its
    ``registered_at`` is bumped so it sorts to the most-recent slot.
    Old entries past ``_MAX_ENTRIES`` are pruned.
    """
    key = _normalize(folder)
    entries = _read_raw(sessions_root)
    # Drop any prior entry for the same path; we'll re-prepend with a
    # fresh timestamp.
    entries = [e for e in entries if e["path"] != key]
    entries.insert(0, {"path": key, "registered_at": time.time()})
    if len(entries) > _MAX_ENTRIES:
        entries = entries[:_MAX_ENTRIES]
    _write_raw(sessions_root, entries)


def forget(sessions_root: Path, folder: Path) -> None:
    """Remove ``folder`` from the registry. No-op if absent."""
    key = _normalize(folder)
    entries = _read_raw(sessions_root)
    new_entries = [e for e in entries if e["path"] != key]
    if len(new_entries) != len(entries):
        _write_raw(sessions_root, new_entries)


def list_entries(sessions_root: Path) -> list[dict[str, Any]]:
    """Return registered folder-backed sessions whose paths still exist.

    Each entry: ``{"path": str, "registered_at": float}``. Ordering
    is preserved as on disk (most-recent first). Stale entries (path
    no longer a directory) are filtered but NOT yet pruned from disk
    — pruning happens on the next ``register`` or ``forget``, so
    transient unmounts (a USB drive popped out) don't immediately
    forget every project on that drive.
    """
    out: list[dict[str, Any]] = []
    for entry in _read_raw(sessions_root):
        try:
            p = Path(entry["path"])
            if p.is_dir():
                out.append(entry)
        except OSError:
            continue
    return out


def is_registered(sessions_root: Path, folder: Path) -> bool:
    """Return True if ``folder`` is in the registry (regardless of
    whether the path currently exists on disk)."""
    key = _normalize(folder)
    return any(e["path"] == key for e in _read_raw(sessions_root))
