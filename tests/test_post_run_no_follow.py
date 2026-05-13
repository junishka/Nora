"""Regression: post-run host writes must refuse to follow symlinks.

A sandboxed analysis script CAN create a symlink inside its run_dir
(macOS sandbox-exec's ``(allow file-write* (subpath run_dir))``
does not exclude the symlink syscall — verified empirically against
the canonical-path profile shape Nora uses). If the host's
post-run writes (cwd_writes.json, stdout.log, stderr.log, the plot
manifest rewrite) follow a planted symlink, the script gets an
arbitrary-file overwrite primitive against anything the unsandboxed
Nora host process can write.

These tests plant a symlink at each post-run write path and verify
the host refuses to follow it. The "victim" file the symlink points
to must remain untouched.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from nora.executor import (
    _write_text_no_follow,
    _write_cwd_writes_manifest,
    _snapshot_cwd_top_level,
    _filter_plot_manifest,
)


def test_no_follow_helper_refuses_symlink(tmp_path: Path) -> None:
    """The helper opens with ``O_NOFOLLOW``; an existing symlink at
    the target path causes the open to fail and the victim file
    stays untouched."""
    victim = tmp_path / "victim.txt"
    victim.write_text("ORIGINAL", encoding="utf-8")

    target = tmp_path / "evil_target"
    os.symlink(victim, target)

    ok = _write_text_no_follow(target, "OVERWRITE ATTEMPT")
    assert ok is False, "helper should refuse to follow symlinks"
    assert victim.read_text(encoding="utf-8") == "ORIGINAL", (
        "victim got overwritten — symlink was followed"
    )


def test_no_follow_helper_writes_regular_file(tmp_path: Path) -> None:
    """Happy path: when the target doesn't exist, the helper
    creates it and writes the content."""
    target = tmp_path / "regular.txt"
    ok = _write_text_no_follow(target, "content")
    assert ok is True
    assert target.read_text(encoding="utf-8") == "content"


def test_no_follow_helper_overwrites_existing_regular_file(
    tmp_path: Path,
) -> None:
    """When the target exists as a regular file (the rewrite case),
    the helper truncates and overwrites it."""
    target = tmp_path / "regular.txt"
    target.write_text("old", encoding="utf-8")
    ok = _write_text_no_follow(target, "new")
    assert ok is True
    assert target.read_text(encoding="utf-8") == "new"


def test_cwd_writes_manifest_does_not_follow_symlink(
    tmp_path: Path,
) -> None:
    """End-to-end: a 'malicious script' plants a symlink at the
    cwd_writes manifest path before the host writes. The host's
    write must not reach the symlink target."""
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    run_dir = cwd / ".nora" / "runs" / "r1"
    run_dir.mkdir(parents=True)

    victim = tmp_path / "victim.txt"
    victim.write_text("INNOCENT BYSTANDER", encoding="utf-8")

    # Pretend the script planted this symlink.
    os.symlink(victim, run_dir / "cwd_writes.json")

    # Snapshot before, modify cwd, then call the manifest writer.
    pre = _snapshot_cwd_top_level(cwd)
    (cwd / "scratch.csv").write_text("x\n", encoding="utf-8")
    _write_cwd_writes_manifest(cwd, run_dir, pre)

    # Victim file MUST be untouched.
    assert victim.read_text(encoding="utf-8") == "INNOCENT BYSTANDER", (
        "host followed the script's planted symlink and overwrote "
        "an arbitrary user file"
    )
    # The symlink itself is still in place (we refused to follow it,
    # we didn't unlink it — leaving it for forensic review).
    assert (run_dir / "cwd_writes.json").is_symlink()


def test_plot_manifest_rewrite_does_not_follow_symlink(
    tmp_path: Path,
) -> None:
    """The plot-manifest rewrite path also refuses to follow a
    planted symlink. With the rewrite refused, the runner's
    per-entry token validation is what enforces authenticity
    (see test_plot_manifest_token_validation)."""
    run_dir = tmp_path / "run_dir"
    run_dir.mkdir()
    plots_dir = run_dir / "_nora_plots"
    plots_dir.mkdir()

    victim = tmp_path / "victim.txt"
    victim.write_text("PROTECTED", encoding="utf-8")

    # Plant the symlink AT the manifest path so the rewrite's open
    # would follow it without O_NOFOLLOW.
    os.symlink(victim, plots_dir / "manifest.jsonl")

    # Build a manifest the rewrite would normally consume. We can't
    # since the path is a symlink — but the rewrite must refuse the
    # write regardless of what content it tries to produce.
    # _filter_plot_manifest reads the manifest (follows the symlink
    # for the read, which is just info disclosure), validates, and
    # tries to rewrite. The rewrite must NOT follow.
    _filter_plot_manifest(run_dir, "some-token")

    assert victim.read_text(encoding="utf-8") == "PROTECTED", (
        "host followed the planted symlink and overwrote the victim"
    )
