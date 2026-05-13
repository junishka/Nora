"""Unit tests for the per-session researcher-staged-files manifest.

Pins the contract ``read_attached_file`` and ``submit_script_file``
rely on:

* ``initialize`` snapshots the cwd's top-level files into the
  manifest at FIRST session-open time only. Subsequent re-opens
  must NOT re-snapshot — otherwise sandbox-written files
  accumulated between sessions silently become "researcher-
  staged" and bypass the SDC guard.
* ``mark_known`` is append-only; re-marking the same name is a no-op.
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
    """A fresh cwd with three top-level files turns into a manifest
    listing those three. Subsequent ``is_known`` calls return True
    for them."""
    (tmp_path / "data.csv").write_text("a,b\n1,2\n")
    (tmp_path / "analysis.R").write_text("# r script\n")
    (tmp_path / "screenshot.png").write_bytes(b"")

    names = initialize(tmp_path)
    assert names == {"data.csv", "analysis.R", "screenshot.png"}
    for n in names:
        assert is_known(tmp_path, n)


def test_initialize_skips_dotfiles_and_directories(tmp_path: Path) -> None:
    """Dotfiles (Nora's own ``.nora/`` tree, plus any researcher tooling)
    and directories must not enter the manifest. Only regular files
    at top level — symlinks too are skipped because the bridge stages
    by basename and a symlink target couldn't go through a legitimate
    add path."""
    (tmp_path / "data.csv").write_text("a\n")
    (tmp_path / ".env").write_text("SECRET=x\n")
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "nested.csv").write_text("a\n")

    names = initialize(tmp_path)
    assert names == {"data.csv"}
    assert not is_known(tmp_path, ".env")
    assert not is_known(tmp_path, "subdir")
    assert not is_known(tmp_path, "nested.csv")


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
    (tmp_path / "old_data.csv").write_text("a\n")
    initialize(tmp_path)
    # Researcher adds a file via the bridge after open.
    mark_known(tmp_path, ["staged_via_bridge.dta"])
    # Between sessions, a sandbox-written script lands in cwd.
    # That file MUST NOT become trusted by the reopen.
    (tmp_path / "smuggled.py").write_text("# raw rows here\n")
    names = initialize(tmp_path)
    assert names == {
        "old_data.csv",
        "staged_via_bridge.dta",
    }
    assert not is_known(tmp_path, "smuggled.py")


def test_mark_known_is_append_only(tmp_path: Path) -> None:
    """Marking the same name twice is a no-op. The manifest doesn't
    grow on repeat events; it doesn't shrink ever."""
    initialize(tmp_path)
    mark_known(tmp_path, ["x.csv"])
    mark_known(tmp_path, ["x.csv"])
    mark_known(tmp_path, ["x.csv", "y.csv"])
    assert known_names(tmp_path) == {"x.csv", "y.csv"}


def test_mark_known_basenames_path_traversal_input(tmp_path: Path) -> None:
    """Inputs that look like paths (``../escape.csv``) get basenamed
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
    # mark_known can fingerprint them.
    (tmp_path / "escape.csv").write_text("escape\n")
    (tmp_path / "bar.csv").write_text("bar\n")
    mark_known(tmp_path, ["../../escape.csv", "foo/bar.csv"])
    assert "escape.csv" in known_names(tmp_path)
    assert "bar.csv" in known_names(tmp_path)
    # Files exist with matching fingerprints — gate passes.
    assert is_known(tmp_path, "escape.csv")
    assert is_known(tmp_path, "bar.csv")
    # Path-shaped lookups also get basenamed before the gate check.
    assert is_known(tmp_path, "/abs/escape.csv")


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
    (tmp_path / "data.csv").write_text("a\n")
    names = initialize(tmp_path)
    assert names == set()
    assert not is_known(tmp_path, "data.csv")


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
    (tmp_path / "x.csv").write_text("a,b\n1,2\n")
    initialize(tmp_path)
    mark_known(tmp_path, ["x.csv"])
    mark_known(tmp_path, ["x.csv"])
    raw = (tmp_path / ".nora" / MANIFEST_FILENAME).read_text(encoding="utf-8")
    data = json.loads(raw)
    matching_entries = [e for e in data["entries"] if e["name"] == "x.csv"]
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
    (tmp_path / "x.csv").write_text("a\n")
    initialize(tmp_path)
    assert is_known(tmp_path, "x.csv")
    (tmp_path / "x.csv").unlink()
    assert not is_known(tmp_path, "x.csv")


def test_manifest_round_trips_through_disk(tmp_path: Path) -> None:
    """The manifest persists on disk in the documented v2 shape —
    a second process / fresh import would observe the same entries
    with the same fingerprints."""
    (tmp_path / "a.csv").write_text("a,b\n1,2\n")
    (tmp_path / "b.r").write_text("# r script\n")
    initialize(tmp_path)
    raw = (tmp_path / ".nora" / MANIFEST_FILENAME).read_text(encoding="utf-8")
    data = json.loads(raw)
    assert data["version"] == 2
    names_in_entries = sorted(e["name"] for e in data["entries"])
    assert names_in_entries == ["a.csv", "b.r"]
    # Each entry carries a sha256 + size — the load-bearing v2
    # additions over v1.
    for entry in data["entries"]:
        assert isinstance(entry.get("sha256"), str)
        assert len(entry["sha256"]) == 64  # sha256 hex
        assert isinstance(entry.get("size_bytes"), int)
        assert entry["size_bytes"] >= 0
