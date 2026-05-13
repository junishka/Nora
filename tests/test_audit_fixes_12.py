"""Regression tests for the twelfth batch of reviewer-flagged fixes.

The behavior pinned here:

1. ``_filter_plot_manifest`` is fail-closed when the sanitized
   rewrite is blocked. The prior implementation caught ``OSError``
   from ``write_text`` and returned silently, leaving the original
   manifest on disk. Downstream consumers (``_capture_plots``, the
   recall-path allowlist check, ``_summarize_plot_helpers``) gate
   on ``kind`` not ``_token``, so a forged ``kind="coefficients"``
   line that survived the bypassed rewrite would ride the next
   turn as an image. A model-authored script can engineer the
   write failure in one line (``os.chmod(manifest_path, 0o444)``),
   so "rare case" framing in the prior code was wrong.

   The fix cascades unlink, rename-file, rename-parent-dir
   fallbacks when the rewrite fails. The on-disk manifest is
   always neutralized in the realistic attack (file-level chmod or
   chflags), and a ``PlotManifestUnsanitizable`` exception fires
   only when every neutralization avenue is blocked too. The
   ``run_script`` call site catches it and demotes the run to
   ``ok=False`` so payloads do not reach the model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


def _make_manifest(run_dir: Path, lines: list[dict[str, Any]]) -> Path:
    plots_dir = run_dir / "_nora_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    manifest = plots_dir / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in lines),
        encoding="utf-8",
    )
    return manifest


def test_filter_strips_token_on_happy_path(tmp_path: Path) -> None:
    """A legit entry has its ``_token`` stripped; nothing else
    changes. Establishes the baseline before the fail-closed paths."""
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
        {"file": "ok.png", "kind": "coefficients", "label": "spec 1"},
    ]


def test_filter_drops_forged_entry_on_happy_path(tmp_path: Path) -> None:
    """A forged entry with a valid ``kind`` but missing ``_token``
    is dropped in the normal rewrite path."""
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
    assert after == [{"file": "real.png", "kind": "coefficients"}]


def test_filter_unlinks_when_rewrite_blocked(tmp_path: Path) -> None:
    """Realistic attack: the script chmods the manifest read-only
    before the executor regains control. ``write_text`` raises
    ``PermissionError`` (an ``OSError``). The fail-closed unlink
    fallback removes the manifest entirely so neither the forged
    entry nor any legit entry survives. Trading legitimate plots
    for a closed forgery path is the correct posture: the script
    engineered the write failure."""
    from nora.executor import _filter_plot_manifest

    token = "deadbeef"
    manifest = _make_manifest(tmp_path, [
        {"file": "real.png", "kind": "coefficients", "_token": token},
        {"file": "forged.png", "kind": "coefficients"},
    ])
    # chmod 0o444 blocks ``open(path, 'w')`` even for the owner on
    # POSIX. unlink uses parent-dir write, so it still works.
    manifest.chmod(0o444)
    try:
        dropped = _filter_plot_manifest(tmp_path, token)
    finally:
        # Restore mode so tmp_path cleanup can proceed if unlink
        # didn't actually happen (the assertion below would catch
        # that, but be defensive).
        if manifest.exists():
            manifest.chmod(0o644)

    # Both the forged entry and the legit entry are counted as
    # dropped: the function couldn't sanitize selectively, so it
    # neutralized the whole file.
    assert dropped == 2
    assert not manifest.exists()


def test_filter_renames_file_when_unlink_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If ``unlink`` is also blocked (eg ``chflags uchg`` on the
    file makes it immutable on macOS), the next fallback renames
    the file out of the way. Consumers' ``manifest.jsonl``
    ``is_file()`` check then returns False."""
    from nora.executor import _filter_plot_manifest

    token = "deadbeef"
    manifest = _make_manifest(tmp_path, [
        {"file": "forged.png", "kind": "coefficients"},
    ])

    real_write = Path.write_text
    real_unlink = Path.unlink

    def fail_write(self: Path, *a: object, **kw: object) -> int:
        if self.name == "manifest.jsonl":
            raise PermissionError("simulated write block")
        return real_write(self, *a, **kw)  # type: ignore[arg-type]

    def fail_unlink(self: Path, *a: object, **kw: object) -> None:
        if self.name == "manifest.jsonl":
            raise PermissionError("simulated unlink block")
        return real_unlink(self, *a, **kw)

    monkeypatch.setattr(Path, "write_text", fail_write)
    monkeypatch.setattr(Path, "unlink", fail_unlink)

    dropped = _filter_plot_manifest(tmp_path, token)
    assert dropped == 1
    assert not manifest.exists()
    assert (manifest.parent / "manifest.jsonl.unsanitized").exists()


def test_filter_renames_plots_dir_when_file_rename_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When write, unlink, AND file-rename are blocked (eg
    ``chflags uchg`` set such that even the parent-dir entry
    update is rejected), the function falls back to renaming the
    entire ``_nora_plots/`` directory. That modifies the run-dir
    entry, not the immutable file, so it succeeds when the file
    itself is locked."""
    from nora.executor import _filter_plot_manifest

    token = "deadbeef"
    manifest = _make_manifest(tmp_path, [
        {"file": "forged.png", "kind": "coefficients"},
    ])

    real_write = Path.write_text
    real_unlink = Path.unlink
    real_rename = Path.rename

    def fail_write(self: Path, *a: object, **kw: object) -> int:
        if self.name == "manifest.jsonl":
            raise PermissionError("simulated write block")
        return real_write(self, *a, **kw)  # type: ignore[arg-type]

    def fail_unlink(self: Path, *a: object, **kw: object) -> None:
        if self.name == "manifest.jsonl":
            raise PermissionError("simulated unlink block")
        return real_unlink(self, *a, **kw)

    def fail_file_rename(self: Path, target: Any) -> Path:
        if self.name == "manifest.jsonl":
            raise PermissionError("simulated file-rename block")
        return real_rename(self, target)

    monkeypatch.setattr(Path, "write_text", fail_write)
    monkeypatch.setattr(Path, "unlink", fail_unlink)
    monkeypatch.setattr(Path, "rename", fail_file_rename)

    dropped = _filter_plot_manifest(tmp_path, token)
    assert dropped == 1
    assert not manifest.parent.exists()
    assert (tmp_path / "_nora_plots.unsanitized").exists()


def test_filter_raises_when_every_fallback_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Absolute last resort. Write, unlink, file-rename, AND
    dir-rename all fail (the script has fully locked the run dir
    against any modification). The function raises
    ``PlotManifestUnsanitizable`` so the executor can mark the run
    failed rather than letting forged entries flow on by
    inertia."""
    from nora.executor import (
        PlotManifestUnsanitizable, _filter_plot_manifest,
    )

    token = "deadbeef"
    _make_manifest(tmp_path, [
        {"file": "forged.png", "kind": "coefficients"},
    ])

    def always_fail(*a: object, **kw: object) -> Any:
        raise PermissionError("simulated total lockdown")

    # Block write, unlink, AND rename on every Path.
    monkeypatch.setattr(Path, "write_text", always_fail)
    monkeypatch.setattr(Path, "unlink", always_fail)
    monkeypatch.setattr(Path, "rename", always_fail)

    with pytest.raises(PlotManifestUnsanitizable):
        _filter_plot_manifest(tmp_path, token)


def test_run_script_demotes_to_error_on_unsanitizable_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The executor catches ``PlotManifestUnsanitizable`` and
    surfaces it as ``ok=False`` + ``error`` rather than letting
    the exception propagate. The call site is part of the trust
    boundary: a raise here would surface to the runner as a
    Python crash; demoting to a structured failure keeps the rest
    of the response shape intact while ensuring payloads are not
    returned."""
    import nora.executor as executor_mod

    def boom(run_dir: Path, run_token: str) -> int:
        raise executor_mod.PlotManifestUnsanitizable("forced for test")

    monkeypatch.setattr(executor_mod, "_filter_plot_manifest", boom)

    # Minimal script that emits one payload so the normal happy
    # path would set ok=True. The unsanitizable manifest must
    # override that.
    code = (
        "import nora\n"
        "nora.nora_result(payload={'x': 1})\n"
    )
    result = executor_mod.run_script(
        "Python", code, tmp_path, timeout_seconds=30,
    )
    assert result.ok is False
    assert result.error is not None
    assert "forced for test" in result.error
    assert result.result_payloads == []
