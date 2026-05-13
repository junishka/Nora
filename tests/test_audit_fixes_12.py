"""Regression tests for the twelfth batch of reviewer-flagged fixes.

The behavior pinned here:

1. ``_filter_plot_manifest`` keeps the per-run ``_token`` on each
   validated entry so downstream consumers can re-validate the
   token against the per-run registry. The validated rewrite is
   fail-soft — if the no-follow write is blocked, the original
   manifest stays on disk and the downstream re-validation is the
   load-bearing protection. (An earlier iteration also added an
   unlink/rename cascade with a ``PlotManifestUnsanitizable``
   exception, but that proved redundant with the consumer-side
   re-validation and conflicted with the contract that the
   original file stays put when the rewrite is blocked.)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _make_manifest(run_dir: Path, lines: list[dict[str, Any]]) -> Path:
    plots_dir = run_dir / "_nora_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    manifest = plots_dir / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in lines),
        encoding="utf-8",
    )
    return manifest


def test_filter_keeps_token_on_validated_entries(tmp_path: Path) -> None:
    """Validated entries pass through unchanged, including their
    ``_token`` field. Downstream consumers re-validate the token
    against the per-run registry (defense in depth); leaving the
    token on disk lets every consumer make the same authenticity
    decision instead of implicitly trusting the rewrite to have
    happened."""
    from nora.executor import _filter_plot_manifest

    token = "deadbeef"
    manifest = _make_manifest(tmp_path, [
        {"file": "ok.png", "kind": "coefficients",
         "label": "spec 1", "_token": token},
    ])

    dropped = _filter_plot_manifest(tmp_path, token)
    assert dropped == 0
    after = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert after == [
        {"file": "ok.png", "kind": "coefficients",
         "label": "spec 1", "_token": token},
    ]


def test_filter_drops_forged_entry_on_happy_path(tmp_path: Path) -> None:
    """A forged entry with a valid ``kind`` but missing ``_token``
    is dropped in the normal rewrite path; the legit entry survives
    with its token still attached for downstream re-validation."""
    from nora.executor import _filter_plot_manifest

    token = "deadbeef"
    manifest = _make_manifest(tmp_path, [
        {"file": "real.png", "kind": "coefficients", "_token": token},
        {"file": "forged.png", "kind": "coefficients"},  # no _token
    ])

    dropped = _filter_plot_manifest(tmp_path, token)
    assert dropped == 1
    after = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert after == [
        {"file": "real.png", "kind": "coefficients", "_token": token},
    ]
