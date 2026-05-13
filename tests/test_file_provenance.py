"""Unit tests for the per-session researcher-staged-files manifest.

Pins the contract ``read_attached_file`` and ``submit_script_file``
rely on:

* ``initialize`` snapshots the cwd's top-level files into the
  manifest at FIRST session-open time only. Subsequent re-opens
  must NOT re-snapshot — otherwise sandbox-written files
  accumulated between sessions silently become "researcher-
  staged" and bypass the SDC guard.
* ``mark_known`` records a content fingerprint (SHA-256 + size) for
  each named file. Re-staging the same name with the same content
  is a no-op shape; re-staging with new content overwrites the
  fingerprint (legitimate replace flow).
* ``is_known`` verifies the current on-disk content matches the
  staged fingerprint — overwriting a staged file with different
  bytes (the model-script SDC bypass attack) is detected and
  rejected.
* Path-traversal-shaped inputs (``foo/../bar.csv``) are basenamed
  defensively so they can't smuggle path-shaped keys into the
  manifest or ``is_known`` lookup.
* The manifest survives an ``os.replace`` mid-write (atomic write
  contract) — readers never see a half-written file.
* Missing or malformed manifests degrade to "nothing is staged"
  rather than crashing the read.
* v1 manifests (legacy names-only schema) are read for backward
  compatibility and opportunistically upgraded to v2 on first
  read.
"""

from __future__ import annotations

import json
from pathlib import Path

from nora.file_provenance import (
    MANIFEST_FILENAME,
    initialize,
    is_known,
    known_names,
    mark_known,
)


def test_initialize_snapshots_cwd_top_level(tmp_path: Path) -> None:
    """A fresh cwd with recallable top-level files turns into a
    manifest listing them. Subsequent ``is_known`` calls return True.

    Only files whose extension matches a recall surface (scripts,
    logs, graphs) are fingerprinted: data files can't have their
    bytes returned by any recall path, so spending I/O to hash a
    multi-GB dataset at session-open is wasteful and never gains
    security."""
    (tmp_path / "analysis.R").write_text("# r script\n")
    (tmp_path / "screenshot.png").write_bytes(b"")
    (tmp_path / "diagnostics.log").write_text("ok\n")

    names = initialize(tmp_path)
    assert names == {"analysis.R", "screenshot.png", "diagnostics.log"}
    for n in names:
        assert is_known(tmp_path, n)


def test_initialize_skips_dotfiles_and_directories(tmp_path: Path) -> None:
    """Dotfiles (Nora's own ``.nora/`` tree, plus any researcher tooling)
    and directories must not enter the manifest. Only regular files
    at top level — symlinks too are skipped because the bridge stages
    by basename and a symlink target couldn't go through a legitimate
    add path."""
    (tmp_path / "analysis.py").write_text("import pandas\n")
    (tmp_path / ".env").write_text("SECRET=x\n")
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "nested.py").write_text("import pandas\n")

    names = initialize(tmp_path)
    assert names == {"analysis.py"}
    assert not is_known(tmp_path, ".env")
    assert not is_known(tmp_path, "subdir")
    assert not is_known(tmp_path, "nested.py")


def test_initialize_does_not_resnapshot_on_reopen(tmp_path: Path) -> None:
    """Re-running initialize on a session that already has a manifest
    must NOT merge in newly-appeared top-level files. Those files
    could be sandbox output from a prior session; the analysis
    sandbox is allowed to write to cwd, so a model script could
    drop ``smuggled.py`` and rely on the re-snapshot to promote
    that file to "researcher-staged" on the next app start.
    The manifest must stay authoritative across reopens; new
    researcher additions arrive through the bridge's staging
    endpoints (``mark_known``)."""
    (tmp_path / "old_analysis.R").write_text("# r\n")
    initialize(tmp_path)
    # Researcher adds a recallable file via the bridge after open.
    (tmp_path / "staged_via_bridge.do").write_text("* stata\n")
    mark_known(tmp_path, ["staged_via_bridge.do"])
    # Between sessions, a sandbox-written script lands in cwd.
    # That file MUST NOT become trusted by the reopen.
    (tmp_path / "smuggled.py").write_text("# raw rows here\n")
    names = initialize(tmp_path)
    assert names == {
        "old_analysis.R",
        "staged_via_bridge.do",
    }
    assert not is_known(tmp_path, "smuggled.py")


def test_mark_known_records_fingerprint(tmp_path: Path) -> None:
    """``mark_known`` requires the file to exist at ``cwd/name`` so
    it can hash. Re-staging the same name with the same content is
    a no-op shape (same fingerprint stored). Names whose files do
    not exist are skipped silently (the bridge endpoints always
    copy bytes into cwd before calling here)."""
    initialize(tmp_path)
    (tmp_path / "x.py").write_text("# 1\n")
    (tmp_path / "y.py").write_text("# 2\n")
    mark_known(tmp_path, ["x.py"])
    mark_known(tmp_path, ["x.py"])  # idempotent for same content
    mark_known(tmp_path, ["x.py", "y.py"])
    assert known_names(tmp_path) == {"x.py", "y.py"}
    # ``z.py`` was never written to disk — mark_known skips it.
    mark_known(tmp_path, ["z.py"])
    assert known_names(tmp_path) == {"x.py", "y.py"}


def test_mark_known_basenames_path_traversal_input(tmp_path: Path) -> None:
    """Inputs that look like paths (``../escape.py``) get basenamed
    before storage. ``is_known`` answers the same way regardless of
    how the caller spelt the lookup."""
    initialize(tmp_path)
    (tmp_path / "escape.py").write_text("a\n")
    (tmp_path / "bar.py").write_text("b\n")
    mark_known(tmp_path, ["../../escape.py", "foo/bar.py"])
    assert is_known(tmp_path, "escape.py")
    assert is_known(tmp_path, "bar.py")
    # Path-shaped lookups also get basenamed.
    assert is_known(tmp_path, "/abs/escape.py")


def test_is_known_on_empty_manifest_returns_false(tmp_path: Path) -> None:
    """A cwd with no manifest yet — i.e. ``initialize`` was never
    called — answers False for every name. Callers treat this as
    "presumed sandbox-output" and refuse the read; the legitimate
    path runs ``initialize`` at session-open."""
    assert not is_known(tmp_path, "anything.csv")


def test_malformed_manifest_falls_back_to_empty(tmp_path: Path) -> None:
    """An externally-edited manifest that no longer parses as JSON
    must not crash the read — the gate degrades to ``False`` so the
    user-facing surface still returns a clean rejection.

    ``initialize`` over a malformed manifest must NOT overwrite the
    on-disk file with a fresh snapshot. The cwd at corrupt-time may
    contain sandbox output from before the corruption, and silently
    re-seeding would promote that output to researcher-staged. The
    safe behavior is to leave the corrupt manifest alone and return
    empty so callers see "nothing is staged".
    """
    nora_dir = tmp_path / ".nora"
    nora_dir.mkdir()
    (nora_dir / MANIFEST_FILENAME).write_text(
        "{this is not valid json", encoding="utf-8",
    )
    assert not is_known(tmp_path, "x.csv")
    (tmp_path / "data.py").write_text("a\n")
    names = initialize(tmp_path)
    assert names == set()
    assert not is_known(tmp_path, "data.py")


def test_manifest_round_trips_through_disk(tmp_path: Path) -> None:
    """The manifest persists on disk in the documented v2 shape —
    a second process / fresh import would observe the same set of
    names with content fingerprints."""
    (tmp_path / "a.py").write_text("# py\n")
    (tmp_path / "b.r").write_text("# r\n")
    initialize(tmp_path)
    raw = (tmp_path / ".nora" / MANIFEST_FILENAME).read_text(encoding="utf-8")
    data = json.loads(raw)
    assert data["version"] == 2
    assert set(data["files"].keys()) == {"a.py", "b.r"}
    for fp in data["files"].values():
        assert isinstance(fp["sha256"], str) and len(fp["sha256"]) == 64
        assert isinstance(fp["size"], int) and fp["size"] >= 0


# ---------------------------------------------------------------------------
# Content-binding: the security fix this module exists for
# ---------------------------------------------------------------------------


def test_overwrite_after_stage_breaks_is_known(tmp_path: Path) -> None:
    """The core attack the manifest exists to defend against: a
    model-authored script overwrites a staged file (eg
    ``analysis.py``) with raw row bytes, then asks
    ``read_attached_file`` for it. Under the prior basename-only
    check the read succeeded; under content-binding the size/hash
    mismatch is caught and ``is_known`` returns False, so the
    consumer's recall gate refuses."""
    legitimate = (tmp_path / "analysis.py")
    legitimate.write_text("import pandas as pd\n")
    initialize(tmp_path)
    assert is_known(tmp_path, "analysis.py")
    # Script overwrites the staged file with different content.
    legitimate.write_text("RAW_ROW_DATA = [(1, 'alice', 42000), ...]\n")
    assert not is_known(tmp_path, "analysis.py")


def test_overwrite_with_same_size_different_bytes_is_caught(
    tmp_path: Path,
) -> None:
    """Length-equal overwrite must also be rejected — the cheap
    size short-circuit is correct but not sufficient. SHA-256 is
    the authority. (Same-size collisions are easy if the attacker
    has read access to the original; SHA-256 collisions are not.)"""
    (tmp_path / "a.py").write_bytes(b"X = 42\nY = 0\n")
    initialize(tmp_path)
    assert is_known(tmp_path, "a.py")
    (tmp_path / "a.py").write_bytes(b"X = 99\nY = 1\n")  # same length
    assert not is_known(tmp_path, "a.py")


def test_legitimate_restage_updates_fingerprint(tmp_path: Path) -> None:
    """When the researcher legitimately re-stages a same-named file
    with new content through a bridge endpoint, ``mark_known``
    overwrites the prior fingerprint with the new one. The next
    ``is_known`` accepts the new content."""
    (tmp_path / "x.py").write_text("v1\n")
    initialize(tmp_path)
    assert is_known(tmp_path, "x.py")
    # Researcher edits the file and re-stages.
    (tmp_path / "x.py").write_text("v2 with more rows\n")
    assert not is_known(tmp_path, "x.py")  # stale fingerprint
    mark_known(tmp_path, ["x.py"])  # bridge updates fingerprint
    assert is_known(tmp_path, "x.py")


def test_symlink_substitution_is_rejected(tmp_path: Path) -> None:
    """If a name was staged as a regular file but someone replaces
    the on-disk entry with a symlink (eg pointing at a sensitive
    file outside cwd), ``is_known`` rejects it. The manifest
    fingerprint is meaningless against a symlink target the
    attacker controls."""
    target = (tmp_path / "report.log")
    target.write_text("data\n")
    initialize(tmp_path)
    assert is_known(tmp_path, "report.log")
    secret = tmp_path / "secret.log"
    secret.write_text("secret\n")
    target.unlink()
    target.symlink_to(secret)
    assert not is_known(tmp_path, "report.log")


# ---------------------------------------------------------------------------
# Backward compat: v1 (names-only) manifests upgrade to v2 on read
# ---------------------------------------------------------------------------


def test_v1_manifest_is_upgraded_to_v2_on_read(tmp_path: Path) -> None:
    """Sessions that pre-date the fingerprint schema have a v1
    manifest (``{"version": 1, "names": [...]}``). On first read we
    re-hash the still-present files and rewrite the manifest as v2;
    entries whose files have disappeared, or whose extensions
    aren't recallable (no recall surface returns their bytes), are
    dropped — an unverifiable name in the manifest is worse than a
    missing name because ``is_known`` would have to fail-open or
    always-False to honor it.
    """
    (tmp_path / ".nora").mkdir()
    (tmp_path / "data.csv").write_text("a,b\n1,2\n")
    (tmp_path / "script.R").write_text("# r\n")
    # File listed in v1 manifest but no longer on disk.
    legacy = {
        "version": 1,
        "names": sorted(["data.csv", "script.R", "deleted.txt"]),
    }
    (tmp_path / ".nora" / MANIFEST_FILENAME).write_text(
        json.dumps(legacy), encoding="utf-8",
    )

    # First lookup triggers the upgrade.
    assert is_known(tmp_path, "script.R")
    # Non-recallable data file is dropped on upgrade — no recall
    # surface would return its bytes anyway, so the manifest
    # doesn't need to track it.
    assert not is_known(tmp_path, "data.csv")
    # The deleted file is also dropped on upgrade (no content to
    # verify even if it were recallable).
    assert not is_known(tmp_path, "deleted.txt")
    # On-disk manifest is now v2 and contains only the recallable
    # entry.
    raw = (tmp_path / ".nora" / MANIFEST_FILENAME).read_text(encoding="utf-8")
    data = json.loads(raw)
    assert data["version"] == 2
    assert set(data["files"].keys()) == {"script.R"}


# ---------------------------------------------------------------------------
# Recallable-only fingerprinting: skip data files at first open
# ---------------------------------------------------------------------------


def test_initialize_skips_non_recallable_extensions(tmp_path: Path) -> None:
    """Data files cannot have their bytes returned through any
    recall surface (``read_attached_file`` /
    ``submit_script_file`` / ``search_in_session_files`` all
    reject data extensions before reaching the provenance check).
    Hashing them at first-open does no security work and would
    force a full-file read of every staged dataset before the UI
    is usable — a 3 GB ``.dta`` would block folder-open.

    The recallable set is the union of script / log / graph
    extensions from ``session_files``."""
    (tmp_path / "data.csv").write_text("a,b\n1,2\n")
    (tmp_path / "panel.dta").write_bytes(b"\x00" * 16)
    (tmp_path / "results.parquet").write_bytes(b"\x00" * 16)
    (tmp_path / "analysis.py").write_text("import pandas\n")
    (tmp_path / "fit.log").write_text("ok\n")
    (tmp_path / "plot.png").write_bytes(b"\x89PNG\r\n")

    names = initialize(tmp_path)
    # Only recallable extensions enter the manifest.
    assert names == {"analysis.py", "fit.log", "plot.png"}
    assert not is_known(tmp_path, "data.csv")
    assert not is_known(tmp_path, "panel.dta")
    assert not is_known(tmp_path, "results.parquet")


def test_mark_known_skips_non_recallable_extensions(tmp_path: Path) -> None:
    """The bridge endpoints call ``mark_known`` after every staging
    event; a researcher dropping a folder of mixed files should
    not pay the cost of hashing each dataset. Non-recallable
    extensions are skipped silently — the file stays usable for
    analysis but the manifest doesn't grow."""
    initialize(tmp_path)
    (tmp_path / "data.csv").write_text("a\n")
    (tmp_path / "panel.dta").write_bytes(b"\x00")
    (tmp_path / "analysis.R").write_text("# r\n")

    mark_known(tmp_path, ["data.csv", "panel.dta", "analysis.R"])
    assert known_names(tmp_path) == {"analysis.R"}


def test_initialize_does_not_read_large_data_files(tmp_path: Path) -> None:
    """End-to-end behavioral check: hashing a multi-GB data file
    is what makes session-open slow. We don't actually need a
    huge file to verify the fix; we monkey-patch the fingerprint
    helper to fail loudly if it's called for a data extension and
    confirm initialize completes without invoking it."""
    import nora.file_provenance as fp_mod

    (tmp_path / "huge.csv").write_text("x\n")
    (tmp_path / "small.py").write_text("# script\n")

    calls: list[str] = []
    original = fp_mod._fingerprint

    def _track(path: Path):
        calls.append(path.name)
        return original(path)

    fp_mod._fingerprint = _track  # type: ignore[assignment]
    try:
        initialize(tmp_path)
    finally:
        fp_mod._fingerprint = original  # type: ignore[assignment]

    # ``huge.csv`` is skipped entirely, no read.
    assert "huge.csv" not in calls
    assert "small.py" in calls


def test_v1_upgrade_detects_post_upgrade_overwrite(tmp_path: Path) -> None:
    """After v1->v2 upgrade, the fingerprints are bound to current
    content at upgrade time. A subsequent overwrite is caught the
    same as a freshly-v2 manifest. (The upgrade can't time-travel
    to detect bytes that were already changed before the upgrade
    happened — that's the existing-data trust posture — but it
    closes the boundary going forward.)"""
    (tmp_path / ".nora").mkdir()
    (tmp_path / "analysis.py").write_text("import pandas\n")
    legacy = {"version": 1, "names": ["analysis.py"]}
    (tmp_path / ".nora" / MANIFEST_FILENAME).write_text(
        json.dumps(legacy), encoding="utf-8",
    )
    assert is_known(tmp_path, "analysis.py")  # triggers upgrade
    # Script overwrites after upgrade — detected.
    (tmp_path / "analysis.py").write_text("rows leaked\n")
    assert not is_known(tmp_path, "analysis.py")
