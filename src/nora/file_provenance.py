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
``<cwd>/.nora/staged_files.json`` records, for every cwd top-level
file Nora considers researcher-known, a content fingerprint (SHA-256
+ byte size) computed at stage time:

  * Files present in cwd at session-open time (the initial snapshot
    — the researcher staged them before opening Nora).
  * Files added through the bridge's file-staging endpoints
    (``add_files`` native picker, ``add_files_from_blobs`` paste/
    drop, ``upload_files`` landing-page drop). These also originate
    from explicit researcher action.

``is_known`` verifies that the file at ``cwd/name`` still matches
the staged hash. A model-authored script that overwrites
``analysis.py`` with raw row bytes will change the hash, so a
subsequent ``read_attached_file("analysis.py")`` is rejected even
though the basename is still in the manifest. The earlier basename-
only check let that overwrite-then-recall path slip through.

The manifest lives under ``<cwd>/.nora/`` which the analysis sandbox
already deny-reads/writes, so a model script can't read or tamper
with it. Writes are atomic via tempfile + ``os.replace`` so a crash
mid-write leaves either the prior snapshot intact or the new one
fully written, never a half-truncated JSON.

Append-only by intent: we never ``remove`` a name on the legitimate
path. A file the researcher staged once stays known even after they
delete it from disk and re-stage a different file with the same
name (the bridge's own ``already_attached`` short-circuit handles
the duplicate-name case before reaching the manifest). If the
researcher does need to refresh a stale fingerprint after replacing
the on-disk file with new content, they re-stage through the same
bridge endpoints; ``mark_known`` overwrites the prior fingerprint
with the new one.

Schema versioning: ``version: 2`` carries fingerprints; ``version:
1`` carried a flat list of names. v1 manifests are read for backward
compatibility (sessions that pre-date the fingerprint binding) and
opportunistically upgraded to v2 on first read by re-hashing the
files that still exist; entries whose files have disappeared are
dropped at that point since they can't be verified anyway.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from collections.abc import Iterable
from pathlib import Path
from typing import Any


MANIFEST_FILENAME = "staged_files.json"
MANIFEST_VERSION = 2

# Hashing chunk size. 64 KB matches the ``shutil.copyfileobj`` default
# and is large enough to amortise syscall overhead on the bulk path
# while staying small enough that hashing a multi-GB dataset doesn't
# allocate proportionally to file size.
_HASH_CHUNK_BYTES = 64 * 1024


def _is_recallable_basename(name: str) -> bool:
    """Whether a file with this basename could have its bytes
    returned to the model through any recall surface.

    Only such files need a content fingerprint. The provenance
    manifest exists to defend against "script overwrites a staged
    file with raw rows, then recalls it as bytes." Data files
    (``.csv``, ``.dta``, ``.rds``, ``.parquet``, ``.jsonl``, ...)
    are rejected by ``read_attached_file``, ``submit_script_file``,
    and ``search_in_session_files`` regardless of provenance state
    — they cannot be recalled as bytes at all. Hashing them at
    session-open does no security work and reads multi-GB datasets
    end-to-end before the UI is usable.

    The recallable set is the union of script / log / graph
    extensions from ``session_files`` — that's the single source
    of truth for what the recall tools surface. If a new
    extension becomes recallable there, it picks up provenance
    binding automatically through this gate.
    """
    # Import lazily so this module stays light-weight and avoids
    # circulars with ``session_files`` if it ever grows imports
    # from here.
    from nora.session_files import GRAPH_EXTS, LOG_EXTS, SCRIPT_EXTS
    ext = Path(name).suffix.lower()
    return ext in SCRIPT_EXTS or ext in LOG_EXTS or ext in GRAPH_EXTS

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


def _fingerprint(path: Path) -> dict[str, Any] | None:
    """Return ``{"sha256": <hex>, "size": <bytes>}`` for ``path``,
    or ``None`` if the file can't be read. Streams in 64 KB chunks
    so a multi-GB dataset doesn't pin its whole content in memory.
    """
    h = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as fh:
            while True:
                chunk = fh.read(_HASH_CHUNK_BYTES)
                if not chunk:
                    break
                h.update(chunk)
                size += len(chunk)
    except OSError:
        return None
    return {"sha256": h.hexdigest(), "size": size}


def _read_manifest(path: Path) -> dict[str, dict[str, Any]]:
    """Read the manifest and return a ``{name: fingerprint}`` dict.

    Handles both schema versions:

    - v2: ``{"version": 2, "files": {name: {sha256, size}}}`` is
      returned directly.
    - v1: ``{"version": 1, "names": [...]}`` is read as a set of
      basenames with no fingerprint info; entries get a sentinel
      ``{"sha256": None, "size": None}`` so the caller knows to
      upgrade them.

    Missing / unreadable / malformed manifests return ``{}`` so
    callers treat that as "nothing is staged yet" rather than
    crashing the read.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}

    version = data.get("version")
    if version == 2:
        files = data.get("files")
        if not isinstance(files, dict):
            return {}
        out: dict[str, dict[str, Any]] = {}
        for name, fp in files.items():
            if not isinstance(name, str) or not name:
                continue
            if not isinstance(fp, dict):
                continue
            sha = fp.get("sha256")
            sz = fp.get("size")
            if not isinstance(sha, str) or not isinstance(sz, int):
                continue
            out[name] = {"sha256": sha, "size": sz}
        return out

    # v1 fallback: names-only list, no fingerprints. Mark each entry
    # with a None sentinel so the caller can re-hash on upgrade.
    names = data.get("names")
    if isinstance(names, list):
        return {
            n: {"sha256": None, "size": None}
            for n in names
            if isinstance(n, str) and n
        }
    return {}


def _write_manifest(path: Path, files: dict[str, dict[str, Any]]) -> None:
    """Atomic write: tmpfile in the same directory, then ``os.replace``.

    Same posture as ``policy.save_policy``: a direct ``write_text``
    on a manifest the bridge re-writes on every staging event would
    leave half-written JSON observable to a concurrent read after a
    crash, and the read would silently start over with an empty
    set. ``os.replace`` is a true atomic rename within one
    filesystem.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        name: {"sha256": fp["sha256"], "size": fp["size"]}
        for name, fp in files.items()
        if fp.get("sha256") is not None and fp.get("size") is not None
    }
    payload = json.dumps(
        {"version": MANIFEST_VERSION, "files": serializable},
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


def _upgrade_v1_in_place(
    cwd: Path, path: Path, entries: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Re-hash any v1 entries whose files still exist, drop the rest,
    and rewrite the manifest as v2. Returns the resulting full
    fingerprint dict. Called under the per-cwd lock.

    The legitimate-on-disk assumption here matches the original
    first-open snapshot: at v1-write time these names were trusted
    as researcher-staged, and the on-disk content at upgrade time
    is the best fingerprint we can capture without re-asking the
    researcher. After this upgrade any *subsequent* overwrite is
    caught — which is the whole point.
    """
    upgraded: dict[str, dict[str, Any]] = {}
    for name, fp in entries.items():
        if fp.get("sha256") is not None:
            upgraded[name] = fp
            continue
        if not _is_recallable_basename(name):
            # Non-recallable extension (data files, etc.): no recall
            # surface returns their bytes, so the provenance gate
            # doesn't apply. Drop the entry rather than spending I/O
            # to hash a multi-GB dataset for nothing.
            continue
        target = cwd / Path(name).name
        if not target.is_file() or target.is_symlink():
            # Entry's file is gone (or replaced by a symlink we
            # refuse to follow). Drop it; an unverifiable name in
            # the manifest is worse than a missing name, because
            # ``is_known`` would have to fail-open to honor it.
            continue
        fingerprint = _fingerprint(target)
        if fingerprint is None:
            continue
        upgraded[name] = fingerprint
    try:
        _write_manifest(path, upgraded)
    except OSError:
        # Best-effort upgrade. If the write fails we still return
        # the in-memory upgraded view so the current call gets
        # correct behavior; the next call will re-attempt.
        pass
    return upgraded


def _read_and_upgrade(cwd: Path, path: Path) -> dict[str, dict[str, Any]]:
    """Read the manifest. If it's v1, upgrade in place to v2 and
    return the upgraded view. Always called under the per-cwd lock.
    """
    entries = _read_manifest(path)
    has_v1 = any(fp.get("sha256") is None for fp in entries.values())
    if has_v1:
        entries = _upgrade_v1_in_place(cwd, path, entries)
    return entries


def initialize(cwd: Path) -> set[str]:
    """Snapshot cwd top-level files into the manifest at FIRST session-
    open only.

    On the very first open the cwd snapshot is presumed researcher-
    staged: the researcher dropped those files there before opening
    Nora. Each file is hashed at this point so subsequent
    ``is_known`` checks can verify content. Once a manifest exists,
    subsequent re-opens MUST NOT re-snapshot — between sessions,
    the analysis sandbox may have written its own files into cwd
    (``df.to_csv("out.csv")`` is legitimate; ``open("smuggled.py",
    "w").write(...)`` from a model-authored script is the gap).
    Merging those in on reopen would silently promote sandbox
    output to "researcher-staged" and let ``read_attached_file`` /
    ``submit_script_file`` / ``search_in_session_files`` return
    their bytes — the same SDC bypass the manifest exists to
    prevent. The provenance guard must be effective across app
    restarts, not just within one live session.

    Backwards compatibility for sessions that pre-date this manifest
    or pre-date the v2 fingerprint schema: on read, v1 entries are
    re-hashed against current on-disk content and rewritten as v2.

    Returns the set of staged names.
    """
    path = _manifest_path(cwd)
    with _lock_for(cwd):
        # ``_read_manifest`` returns ``{}`` for missing OR corrupt
        # manifests. We need to distinguish those: missing -> seed,
        # corrupt -> leave alone (don't silently seed an empty
        # manifest on top of a corrupt one and resnapshot whatever
        # is in cwd right now). ``path.exists()`` is the gate.
        if path.exists():
            return set(_read_and_upgrade(cwd, path).keys())
        snapshot: dict[str, dict[str, Any]] = {}
        for name in _enumerate_cwd_top_level(cwd):
            # Skip non-recallable extensions: data files (``.csv``,
            # ``.dta``, ``.parquet``, ...) can't have their bytes
            # returned by any recall surface, so a content
            # fingerprint does no security work and would force a
            # full-file read of every staged dataset before the
            # session is usable. A 3 GB ``.dta`` would block
            # folder-open for seconds on SSD and much longer on
            # network mounts.
            if not _is_recallable_basename(name):
                continue
            fingerprint = _fingerprint(cwd / name)
            if fingerprint is None:
                continue
            snapshot[name] = fingerprint
        _write_manifest(path, snapshot)
        return set(snapshot.keys())


def mark_known(cwd: Path, names: Iterable[str]) -> set[str]:
    """Hash the named files (resolved against ``cwd``) and record
    their fingerprints in the manifest. Returns the resulting set of
    staged names.

    Re-staging a name overwrites its prior fingerprint, which is the
    correct behavior when the researcher legitimately replaces a
    staged file's content through the bridge: the new content's
    hash is now authoritative.

    ``Path(name).name`` is used to defensively basename the input —
    callers should already be passing basenames, but a stray
    ``/foo/bar.csv`` wouldn't smuggle a path-shaped key in.

    Files that don't exist (or can't be read) are skipped silently.
    The bridge endpoints always copy bytes into ``cwd`` before
    calling here, so a missing file is a logic error elsewhere; we
    don't crash on it because provenance is best-effort wrapped at
    every call site.
    """
    cleaned: list[str] = []
    for n in names:
        if not n:
            continue
        basename = Path(n).name
        if basename:
            cleaned.append(basename)
    if not cleaned:
        return set(_read_and_upgrade_locked(cwd).keys())

    path = _manifest_path(cwd)
    with _lock_for(cwd):
        merged = _read_and_upgrade(cwd, path) if path.exists() else {}
        for basename in cleaned:
            # Skip non-recallable extensions for the same reason as
            # ``initialize``: hashing a multi-GB dataset gains nothing
            # because no recall surface returns its bytes.
            if not _is_recallable_basename(basename):
                continue
            fingerprint = _fingerprint(cwd / basename)
            if fingerprint is None:
                # File not present or unreadable — skip silently
                # (see docstring).
                continue
            merged[basename] = fingerprint
        _write_manifest(path, merged)
        return set(merged.keys())


def _read_and_upgrade_locked(cwd: Path) -> dict[str, dict[str, Any]]:
    """``_read_and_upgrade`` for callers that need a no-op-when-empty
    path without entering the write branch."""
    path = _manifest_path(cwd)
    with _lock_for(cwd):
        if not path.exists():
            return {}
        return _read_and_upgrade(cwd, path)


def is_known(cwd: Path, name: str) -> bool:
    """Whether the file at ``cwd/name`` is the same content that was
    staged. Returns True only if:

    1. ``name`` is in the manifest, AND
    2. the on-disk file at ``cwd / Path(name).name`` is a regular
       file (not a symlink, not a directory), AND
    3. its size matches the staged size, AND
    4. its SHA-256 matches the staged hash.

    Step 3 is a cheap short-circuit before the expensive hash:
    overwrite attacks that change file length never pass it. Steps
    2 and 4 cover the case where size happens to be equal but
    bytes differ, and the symlink-rejection bit prevents an
    attacker from substituting a symlink to a sensitive file under
    a name that was originally staged as a regular file.

    Refuses path-traversal-shaped inputs by basenaming first; a
    caller that passed ``foo/../bar.csv`` gets the same answer as if
    they passed ``bar.csv``.

    No lock needed for reads — the writer's atomic ``os.replace``
    means readers see either the old manifest or the new one,
    never a partial state.
    """
    safe = Path(name).name
    if not safe:
        return False
    path = _manifest_path(cwd)
    entries = _read_manifest(path)
    fp = entries.get(safe)
    if fp is None:
        return False
    expected_hash = fp.get("sha256")
    expected_size = fp.get("size")
    if expected_hash is None or expected_size is None:
        # v1 entry that hasn't been upgraded yet (read happened
        # before any write-triggering call). Fall through to
        # current-content hashing: re-hash and compare to the
        # stored hash, which is None here, so we cannot verify.
        # Trigger an upgrade by reading-with-upgrade under the
        # lock so the next call sees a v2 fingerprint.
        with _lock_for(cwd):
            upgraded = _read_and_upgrade(cwd, path)
        fp = upgraded.get(safe)
        if fp is None:
            return False
        expected_hash = fp.get("sha256")
        expected_size = fp.get("size")
        if expected_hash is None or expected_size is None:
            return False
    target = cwd / safe
    if target.is_symlink() or not target.is_file():
        return False
    try:
        actual_size = target.stat().st_size
    except OSError:
        return False
    if actual_size != expected_size:
        return False
    actual = _fingerprint(target)
    if actual is None:
        return False
    return actual["sha256"] == expected_hash


def known_names(cwd: Path) -> set[str]:
    """Return the set of names in the manifest (test/diagnostic
    helper). Does NOT re-verify on-disk content."""
    return set(_read_manifest(_manifest_path(cwd)).keys())
