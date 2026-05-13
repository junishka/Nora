"""Unit tests for the per-session researcher-staged-files manifest.

Pins the contract ``read_attached_file`` and ``submit_script_file``
rely on:

* ``initialize`` snapshots the cwd's top-level files into the
  manifest at FIRST session-open time only. Subsequent re-opens
  must NOT re-snapshot — otherwise sandbox-written files
  accumulated between sessions silently become "researcher-
  staged" and bypass the SDC guard.
* ``initialize`` and ``mark_known`` only fingerprint recallable
  extensions (scripts, logs, graphs). Data files (``.csv`` /
  ``.dta`` / ``.parquet`` / ...) cannot have their bytes returned
  by any recall path, so spending I/O to hash a multi-GB dataset
  at session-open is wasteful and never gains security.
* ``mark_known`` records a content fingerprint (sha256 + size) for
  each named file. Re-staging the same name with identical content
  is a no-op (dedup on (name, sha256)); re-staging with new content
  appends a fresh fingerprint alongside the existing entries
  (audit trail preserved, both are valid for ``is_known``).
* ``is_known`` verifies the current on-disk content matches a
  recorded fingerprint — overwriting a staged file with different
  bytes (the model-script SDC bypass attack) is detected and
  rejected. v1 (name-only) entries fail closed: there's no
  fingerprint to verify against.
* Path-traversal-shaped inputs (``foo/../bar.csv``) are basenamed
  defensively so they can't smuggle path-shaped keys into the
  manifest or ``is_known`` lookup.
* The manifest survives an ``os.replace`` mid-write (atomic write
  contract) — readers never see a half-written file.
* Missing or malformed manifests degrade to "nothing is staged"
  rather than crashing the read.
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
    """``mark_known`` fingerprints the files that exist at
    ``cwd/name``. Identical re-stages dedup on (name, sha256).
    Names whose files don't exist are recorded without a
    fingerprint and fail closed in ``is_known``."""
    initialize(tmp_path)
    (tmp_path / "x.py").write_text("# 1\n")
    (tmp_path / "y.py").write_text("# 2\n")
    mark_known(tmp_path, ["x.py"])
    mark_known(tmp_path, ["x.py"])  # idempotent for same content
    mark_known(tmp_path, ["x.py", "y.py"])
    assert known_names(tmp_path) == {"x.py", "y.py"}


def test_mark_known_basenames_path_traversal_input(tmp_path: Path) -> None:
    """Inputs that look like paths (``../escape.py``) get basenamed
    before storage. ``known_names`` answers under the basename, and
    ``is_known`` answers under the basename when the file also
    exists at ``cwd/<basename>`` with matching fingerprint.

    Note: this test pre-creates the files at the basenamed paths
    so the fingerprint check passes — that's the only realistic
    way path-shaped inputs ever appear at the bridge anyway (they
    don't, but the defensive basenaming code path stays in case).
    """
    initialize(tmp_path)
    # Pre-create the files at their BASENAMED locations so that
    # mark_known can fingerprint them. Use recallable extensions
    # so the entries survive the recallable-only filter.
    (tmp_path / "escape.py").write_text("escape\n")
    (tmp_path / "bar.py").write_text("bar\n")
    mark_known(tmp_path, ["../../escape.py", "foo/bar.py"])
    assert "escape.py" in known_names(tmp_path)
    assert "bar.py" in known_names(tmp_path)
    # Files exist with matching fingerprints — gate passes.
    assert is_known(tmp_path, "escape.py")
    assert is_known(tmp_path, "bar.py")
    # Path-shaped lookups also get basenamed before the gate check.
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


# ---------------------------------------------------------------------------
# Content-fingerprint enforcement
# ---------------------------------------------------------------------------
#
# These tests close the basename-collision gap that name-only
# tracking left open: a model-authored script can write to cwd
# (the executor's sandbox profile permits writes there), and
# could overwrite ``analysis.py`` (or any previously-staged
# filename) with raw row bytes. The v2 manifest records a sha256
# at stage time; ``is_known`` returns False when the current file's
# sha256 doesn't match the recorded set.


def test_overwriting_a_staged_file_revokes_is_known(tmp_path: Path) -> None:
    """The attack: researcher stages ``analysis.py``. Manifest records
    its sha256. Later, a sandbox script overwrites ``analysis.py``
    with raw rows. ``is_known`` must reject the read.

    Without the fingerprint check, the basename-only manifest would
    still answer True here — and ``read_attached_file`` /
    ``submit_script_file`` would return the raw bytes, bypassing the
    SDC sanitizer. The fingerprint is the closure for that gap.
    """
    (tmp_path / "analysis.py").write_text("# legit analysis script\n")
    initialize(tmp_path)
    assert is_known(tmp_path, "analysis.py")
    # Simulate a sandbox script overwriting the file in-place.
    (tmp_path / "analysis.py").write_text("name,ssn\nAlice,000-00-0000\n")
    assert not is_known(tmp_path, "analysis.py"), (
        "fingerprint mismatch must revoke is_known: a script-shaped "
        "file overwritten with raw rows still has the staged basename "
        "in the manifest"
    )
    # The name is still listed (audit trail intact) — ``known_names``
    # is the diagnostic view that ignores fingerprints.
    assert "analysis.py" in known_names(tmp_path)


def test_overwrite_with_same_size_different_bytes_is_caught(
    tmp_path: Path,
) -> None:
    """Length-equal overwrite must also be rejected — a cheap
    size short-circuit alone wouldn't be enough. SHA-256 is the
    authority. (Same-size collisions are easy if the attacker has
    read access to the original; SHA-256 collisions are not.)"""
    (tmp_path / "a.py").write_bytes(b"X = 42\nY = 0\n")
    initialize(tmp_path)
    assert is_known(tmp_path, "a.py")
    (tmp_path / "a.py").write_bytes(b"X = 99\nY = 1\n")  # same length
    assert not is_known(tmp_path, "a.py")


def test_re_staging_with_updated_content_authorises_new_content(tmp_path: Path) -> None:
    """The legitimate path: researcher edits a script outside Nora
    and drops it back into the chat. The bridge calls ``mark_known``
    with the updated file in place — both fingerprints (old and
    new) are recorded so the audit trail isn't lost, and the
    current on-disk content (which matches the new fingerprint)
    passes ``is_known``.
    """
    (tmp_path / "analysis.py").write_text("v1\n")
    initialize(tmp_path)
    assert is_known(tmp_path, "analysis.py")
    # Researcher edits the file externally and re-stages via the
    # bridge composer drop path.
    (tmp_path / "analysis.py").write_text("v2 updated\n")
    mark_known(tmp_path, ["analysis.py"])
    assert is_known(tmp_path, "analysis.py")
    # Both fingerprints persist for audit (paranoid but cheap).
    raw = (tmp_path / ".nora" / MANIFEST_FILENAME).read_text(encoding="utf-8")
    data = json.loads(raw)
    matching_entries = [e for e in data["entries"] if e["name"] == "analysis.py"]
    assert len(matching_entries) == 2


def test_re_staging_with_identical_content_is_noop(tmp_path: Path) -> None:
    """Re-staging the SAME content under the same name doesn't grow
    the manifest. The dedup key is (name, sha256), so identical
    re-stages collapse to a single entry."""
    (tmp_path / "x.py").write_text("# 1\n")
    initialize(tmp_path)
    mark_known(tmp_path, ["x.py"])
    mark_known(tmp_path, ["x.py"])
    raw = (tmp_path / ".nora" / MANIFEST_FILENAME).read_text(encoding="utf-8")
    data = json.loads(raw)
    matching_entries = [e for e in data["entries"] if e["name"] == "x.py"]
    assert len(matching_entries) == 1


def test_v1_legacy_manifest_fails_closed(tmp_path: Path) -> None:
    """Sessions opened before the fingerprint upgrade carry a v1
    manifest (names-only). v1 entries are READ for ``known_names``
    so the researcher can see what was staged, but ``is_known``
    rejects them — there's no fingerprint to verify against, and
    silently fingerprinting whatever is currently on disk would
    promote sandbox output to "trusted" if the script had already
    overwritten the file. The path forward is researcher
    re-stages via the bridge."""
    (tmp_path / "old_staged.py").write_text("# legacy content\n")
    (tmp_path / ".nora").mkdir()
    # Hand-write a v1 manifest the way the pre-fix code did.
    (tmp_path / ".nora" / MANIFEST_FILENAME).write_text(
        json.dumps({"version": 1, "names": ["old_staged.py"]}),
        encoding="utf-8",
    )
    # The name is visible to diagnostics.
    assert "old_staged.py" in known_names(tmp_path)
    # But not authorized for reads — no fingerprint to verify.
    assert not is_known(tmp_path, "old_staged.py")
    # Re-staging upgrades it: a fresh mark_known computes the
    # fingerprint and writes a v2 entry alongside the legacy one.
    mark_known(tmp_path, ["old_staged.py"])
    assert is_known(tmp_path, "old_staged.py")


def test_missing_file_returns_false(tmp_path: Path) -> None:
    """Name in manifest but the file isn't currently on disk: the
    fingerprint check can't run, so fail closed. Different from
    "name never staged" (also False) — both produce the same
    answer, which is what the gate needs."""
    (tmp_path / "x.py").write_text("# script\n")
    initialize(tmp_path)
    assert is_known(tmp_path, "x.py")
    (tmp_path / "x.py").unlink()
    assert not is_known(tmp_path, "x.py")


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


def test_manifest_round_trips_through_disk(tmp_path: Path) -> None:
    """The manifest persists on disk in the documented v2 shape —
    a second process / fresh import would observe the same entries
    with the same fingerprints."""
    (tmp_path / "a.py").write_text("# py\n")
    (tmp_path / "b.r").write_text("# r script\n")
    initialize(tmp_path)
    raw = (tmp_path / ".nora" / MANIFEST_FILENAME).read_text(encoding="utf-8")
    data = json.loads(raw)
    assert data["version"] == 2
    names_in_entries = sorted(e["name"] for e in data["entries"])
    assert names_in_entries == ["a.py", "b.r"]
    # Each entry carries a sha256 + size — the load-bearing v2
    # additions over v1.
    for entry in data["entries"]:
        assert isinstance(entry.get("sha256"), str)
        assert len(entry["sha256"]) == 64  # sha256 hex
        assert isinstance(entry.get("size_bytes"), int)
        assert entry["size_bytes"] >= 0


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
