"""Regression: forged plot manifest entries must NOT reach model
vision, even when the executor's on-disk rewrite of the manifest
fails.

The plot manifest lives inside the script-writable run directory.
The executor's ``_filter_plot_manifest`` validates each entry's
``_token`` against the per-run token and rewrites the manifest with
the validated subset. But the rewrite is best-effort — a script
can chmod the manifest read-only (or block the host's write some
other way) so the original (forged) entries remain on disk.

The runner's ``_capture_plots`` re-validates the token directly,
closing the gap. With no token registered, every entry is dropped
(fail closed).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


def _make_manifest(run_dir: Path, entries: list[dict[str, Any]]) -> None:
    plots_dir = run_dir / "_nora_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(e) + "\n" for e in entries)
    (plots_dir / "manifest.jsonl").write_text(text, encoding="utf-8")


def _make_plot_png(run_dir: Path, name: str) -> None:
    """Write a minimal valid PNG (the captor reads bytes + caps size,
    not the format — but the file must exist)."""
    plots_dir = run_dir / "_nora_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    (plots_dir / name).write_bytes(
        b"\x89PNG\r\n\x1a\n" + b"\x00" * 200
    )


def _make_runner() -> Any:
    """Build a minimal SessionRunner-ish object that just has the
    fields ``_capture_plots`` reads/writes — we don't need a real
    runner, just an instance with the method bound.
    """
    from nora.runner import SessionRunner

    # Construct without invoking __init__'s provider machinery —
    # _capture_plots only touches ``pending_plot_images`` and the
    # ``_log_pdf_conversion_failure`` helper, neither of which need
    # a live provider.
    runner = SessionRunner.__new__(SessionRunner)
    runner.pending_plot_images = []  # type: ignore[attr-defined]
    return runner


def test_forged_entries_dropped_when_no_token_registered(
    tmp_path: Path,
) -> None:
    """No entry in the registry → drop everything. Replay / re-attach
    paths land here and must not surface stale on-disk plots."""
    run_dir = tmp_path / "runs" / "r1"
    run_dir.mkdir(parents=True)
    _make_plot_png(run_dir, "coef.png")
    # Forged entry: looks legitimate (kind on allowlist) but no
    # token (or token doesn't match anything registered).
    _make_manifest(run_dir, [
        {"kind": "coefficients", "file": "coef.png", "label": "forged"},
    ])

    runner = _make_runner()
    runner._capture_plots(run_dir)
    assert runner.pending_plot_images == [], (
        "no token registered = no trust = no plots staged"
    )


def test_legitimate_entries_pass_when_token_registered(
    tmp_path: Path,
) -> None:
    """When the executor registers a token and the manifest entry's
    ``_token`` matches, the entry rides through normally. This is
    the happy path: covers the case where the on-disk rewrite
    didn't run (or didn't strip tokens) but the entry is legit."""
    from nora.executor import register_run_token, RESULT_TOKEN_FIELD

    run_dir = tmp_path / "runs" / "r2"
    run_dir.mkdir(parents=True)
    _make_plot_png(run_dir, "coef.png")
    token = "test-token-abc123"
    _make_manifest(run_dir, [
        {
            "kind": "coefficients", "file": "coef.png",
            "label": "ok", RESULT_TOKEN_FIELD: token,
        },
    ])
    register_run_token(run_dir, token)

    runner = _make_runner()
    runner._capture_plots(run_dir)
    assert len(runner.pending_plot_images) == 1
    staged = runner.pending_plot_images[0]
    assert staged["kind"] == "coefficients"
    # Token must be stripped from the in-memory copy.
    assert RESULT_TOKEN_FIELD not in staged


def test_forged_token_rejected_when_token_registered(
    tmp_path: Path,
) -> None:
    """A forged entry that carries a WRONG ``_token`` (e.g., the
    script guessed) must be dropped. The token is short-string
    compared via ``secrets.compare_digest`` so a near-match doesn't
    leak timing info, but the rejection is what we test here."""
    from nora.executor import register_run_token, RESULT_TOKEN_FIELD

    run_dir = tmp_path / "runs" / "r3"
    run_dir.mkdir(parents=True)
    _make_plot_png(run_dir, "coef.png")
    _make_plot_png(run_dir, "forged.png")
    real_token = "real-token-xyz789"
    _make_manifest(run_dir, [
        # legitimate
        {
            "kind": "coefficients", "file": "coef.png",
            "label": "real", RESULT_TOKEN_FIELD: real_token,
        },
        # forged: wrong token, raw-data plot mislabeled as coefficients
        {
            "kind": "coefficients", "file": "forged.png",
            "label": "evil", RESULT_TOKEN_FIELD: "wrong-token-guess",
        },
    ])
    register_run_token(run_dir, real_token)

    runner = _make_runner()
    runner._capture_plots(run_dir)

    staged_labels = [p["label"] for p in runner.pending_plot_images]
    assert "real" in staged_labels
    assert "evil" not in staged_labels


def test_tokenless_entries_dropped(tmp_path: Path) -> None:
    """The executor's rewrite now KEEPS tokens in the validated
    manifest so the runner can re-validate every entry. An entry
    with no ``_token`` field is treated as forged (the helper
    libraries always stamp the token; missing it means hand-crafted
    bypass) and dropped, regardless of what token is registered."""
    from nora.executor import register_run_token

    run_dir = tmp_path / "runs" / "r4"
    run_dir.mkdir(parents=True)
    _make_plot_png(run_dir, "coef.png")
    _make_manifest(run_dir, [
        # No _token field at all.
        {"kind": "coefficients", "file": "coef.png", "label": "no-token"},
    ])
    register_run_token(run_dir, "any-token")

    runner = _make_runner()
    runner._capture_plots(run_dir)
    labels = [p["label"] for p in runner.pending_plot_images]
    assert "no-token" not in labels, (
        "tokenless entries must be dropped — the helper libraries "
        "always stamp a token, and missing it is the forged-entry "
        "signature"
    )


def test_chmod_blocked_rewrite_still_blocks_forged_entries(
    tmp_path: Path,
) -> None:
    """End-to-end demonstration of the bug 4 fix: a script chmods
    the manifest read-only so the executor's rewrite fails. The
    original (forged + valid) manifest stays on disk. The runner
    re-validates and only stages the legit entry."""
    from nora.executor import (
        register_run_token, RESULT_TOKEN_FIELD, _filter_plot_manifest,
    )
    import os

    run_dir = tmp_path / "runs" / "r5"
    run_dir.mkdir(parents=True)
    _make_plot_png(run_dir, "real.png")
    _make_plot_png(run_dir, "forged.png")
    real_token = "real-token-12345"
    _make_manifest(run_dir, [
        {
            "kind": "coefficients", "file": "real.png",
            "label": "real", RESULT_TOKEN_FIELD: real_token,
        },
        {
            "kind": "coefficients", "file": "forged.png",
            "label": "evil",  # no _token
        },
    ])

    # chmod the manifest read-only to block the executor's rewrite.
    manifest_path = run_dir / "_nora_plots" / "manifest.jsonl"
    os.chmod(manifest_path, 0o444)
    try:
        _filter_plot_manifest(run_dir, real_token)
    finally:
        os.chmod(manifest_path, 0o644)

    # The original manifest is still on disk with the forged entry.
    on_disk = manifest_path.read_text(encoding="utf-8")
    assert "evil" in on_disk, (
        "test setup precondition: forged entry survived the "
        "rewrite (because chmod blocked the host's write)"
    )

    register_run_token(run_dir, real_token)
    runner = _make_runner()
    runner._capture_plots(run_dir)

    labels = [p["label"] for p in runner.pending_plot_images]
    assert "real" in labels
    assert "evil" not in labels, (
        "runner must drop the forged entry even when the "
        "executor's on-disk rewrite was blocked"
    )


def test_get_run_token_is_non_destructive(tmp_path: Path) -> None:
    """Both the runner's ``_capture_plots`` and the tool's
    ``_summarize_plot_helpers`` need to look up the same token
    within a single run, so the registry returns it
    non-destructively. Long-session memory growth is bounded by
    the cap-based LRU eviction inside the executor."""
    from nora.executor import register_run_token, get_run_token

    run_dir = tmp_path / "runs" / "r6"
    run_dir.mkdir(parents=True)
    register_run_token(run_dir, "shared")
    assert get_run_token(run_dir) == "shared"
    # Second call must still return the same token — both
    # consumers need to see it.
    assert get_run_token(run_dir) == "shared"


def test_summarize_plot_helpers_drops_forged_entries(
    tmp_path: Path,
) -> None:
    """The tool-layer summary that the MODEL sees must also drop
    forged entries. Otherwise the model would believe its forged
    plot succeeded ('I plotted evil.png as coefficients'), even
    though the runner refused to attach the image. Defense in
    depth across both surfaces that read the manifest."""
    from nora.executor import register_run_token, RESULT_TOKEN_FIELD
    from nora.tools import _summarize_plot_helpers

    run_dir = tmp_path / "runs" / "r7"
    run_dir.mkdir(parents=True)
    _make_plot_png(run_dir, "real.png")
    _make_plot_png(run_dir, "forged.png")
    real_token = "real-token-summary"
    _make_manifest(run_dir, [
        {
            "kind": "coefficients", "file": "real.png",
            "label": "real", RESULT_TOKEN_FIELD: real_token,
        },
        {
            # No token at all.
            "kind": "coefficients", "file": "forged.png",
            "label": "evil",
        },
    ])
    register_run_token(run_dir, real_token)

    summary = _summarize_plot_helpers(run_dir)
    assert summary is not None
    succeeded_labels = [s["label"] for s in summary.get("succeeded", [])]
    assert "real" in succeeded_labels
    assert "evil" not in succeeded_labels, (
        "the model-visible summary must not list forged plots"
    )
