"""Unit tests for the per-session researcher-staged-files manifest.

Pins the contract ``read_attached_file`` and ``submit_script_file``
rely on:

* ``initialize`` snapshots the cwd's top-level files into the
  manifest at session-open time AND merges with any existing manifest
  (the upgrade path for sessions that pre-date this feature).
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


def test_initialize_is_idempotent_and_merges_existing(tmp_path: Path) -> None:
    """Re-running initialize on a session that already has a manifest
    AND has new top-level files merges the two — the upgrade path
    for sessions opened before this feature shipped."""
    (tmp_path / "old_data.csv").write_text("a\n")
    initialize(tmp_path)
    # Researcher adds a file via the bridge after open.
    mark_known(tmp_path, ["staged_via_bridge.dta"])
    # Then a session is reopened with one new file appearing in
    # cwd top-level.
    (tmp_path / "new_data.parquet").write_text("")
    names = initialize(tmp_path)
    assert names == {
        "old_data.csv",
        "staged_via_bridge.dta",
        "new_data.parquet",
    }


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
    before storage. ``is_known`` answers the same way regardless of
    how the caller spelt the lookup."""
    initialize(tmp_path)
    mark_known(tmp_path, ["../../escape.csv", "foo/bar.csv"])
    assert is_known(tmp_path, "escape.csv")
    assert is_known(tmp_path, "bar.csv")
    # Path-shaped lookups also get basenamed.
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
    user-facing surface still returns a clean rejection."""
    nora_dir = tmp_path / ".nora"
    nora_dir.mkdir()
    (nora_dir / MANIFEST_FILENAME).write_text(
        "{this is not valid json", encoding="utf-8",
    )
    assert not is_known(tmp_path, "x.csv")
    # ``initialize`` over a malformed manifest builds a fresh one
    # from the cwd snapshot rather than crashing.
    (tmp_path / "data.csv").write_text("a\n")
    names = initialize(tmp_path)
    assert names == {"data.csv"}


def test_manifest_round_trips_through_disk(tmp_path: Path) -> None:
    """The manifest persists on disk in the documented shape — a
    second process / fresh import would observe the same set."""
    initialize(tmp_path)
    mark_known(tmp_path, ["a.csv", "b.r"])
    raw = (tmp_path / ".nora" / MANIFEST_FILENAME).read_text(encoding="utf-8")
    data = json.loads(raw)
    assert data["version"] == 1
    assert sorted(data["names"]) == ["a.csv", "b.r"]
