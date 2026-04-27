"""Plot vision: only manifest-listed model-output plots cross to the model.

The privacy line: ``nora$plot_residuals`` / ``plot_interaction`` and
their Python siblings write a PNG into ``<run_dir>/_nora_plots/``
AND append a JSONL entry to ``manifest.jsonl``. The runner reads
ONLY the manifest. Anything else dropped into the directory by raw
``ggsave`` / ``plt.savefig`` (or by an attacker who somehow wrote a
file there) stays invisible to the model.

These tests pin the boundary at the runner-capture layer where it
matters most. Helper-function correctness (R and Python writing the
right PNGs given a real fit) is left to manual smoke testing — it
needs Rscript / scipy / matplotlib at runtime which CI doesn't
guarantee.
"""

from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
from typing import Any, AsyncIterator
from unittest.mock import MagicMock

import pytest

from nora.provider import (
    AssistantText,
    ToolCallResult,
    TurnDone,
)
from nora.runner import (
    _PLOT_KIND_ALLOWLIST,
    _PLOT_MAX_BYTES,
    _PLOT_MAX_PER_TURN,
    SessionRunner,
)


def _png_bytes(size: int = 200) -> bytes:
    """Tiny valid-ish PNG. The runner only checks the .png suffix +
    byte cap, so we don't need a real image — just bytes that fit
    the cap."""
    return b"\x89PNG\r\n\x1a\n" + b"\x00" * size


def _seed_run(
    cwd: Path, manifest_lines: list[dict[str, Any]],
    extra_files: dict[str, bytes] | None = None,
) -> Path:
    """Stand-in for a finished script run: write a run dir
    structure with a manifest and plot files."""
    run_dir = cwd / ".nora" / "runs" / "r0001"
    plots = run_dir / "_nora_plots"
    plots.mkdir(parents=True, exist_ok=True)
    for entry in manifest_lines:
        f = entry.get("file")
        if isinstance(f, str):
            (plots / f).write_bytes(_png_bytes())
    (plots / "manifest.jsonl").write_text(
        "\n".join(json.dumps(e) for e in manifest_lines) + "\n",
        encoding="utf-8",
    )
    for name, content in (extra_files or {}).items():
        (plots / name).write_bytes(content)
    return run_dir


def _runner(cwd: Path) -> SessionRunner:
    return SessionRunner(
        cwd=cwd, provider="anthropic", model="claude-sonnet-4-6[1m]"
    )


# ---------------------------------------------------------------------------
# Manifest-only allowlist
# ---------------------------------------------------------------------------

def test_capture_reads_only_manifest_entries(tmp_path: Path) -> None:
    """A residual.png in the dir but not in the manifest is
    invisible. A raw histogram.png the researcher wrote with
    ggsave/savefig must NOT cross."""
    run_dir = _seed_run(
        tmp_path,
        manifest_lines=[
            {"file": "residuals.png", "kind": "residuals", "label": "Residual diagnostics"},
        ],
        extra_files={
            # Researcher's raw-data plot — written via plt.savefig,
            # bypassing the helpers. Must stay invisible.
            "raw_histogram.png": _png_bytes(),
            # An attacker-style file in a wrong format — also rejected.
            "secret.json": b'{"data": "leak"}',
        },
    )

    runner = _runner(tmp_path)
    runner._capture_plots(run_dir)

    names = [p["name"] for p in runner.pending_plot_images]
    assert names == ["residuals.png"], (
        f"only manifest-listed files should cross; got {names}"
    )


def test_capture_rejects_off_allowlist_kinds(tmp_path: Path) -> None:
    """A manifest entry whose ``kind`` isn't in the runner's
    allowlist (e.g. ``raw_histogram``) is silently dropped. This
    is the second layer of the gate: even if a future runtime
    helper were modified to register a forbidden kind, the runner
    refuses to surface it."""
    run_dir = _seed_run(
        tmp_path,
        manifest_lines=[
            {"file": "residuals.png", "kind": "residuals", "label": "ok"},
            {"file": "raw_histogram.png", "kind": "raw_histogram",
             "label": "should not cross"},
            {"file": "scatter.png", "kind": "scatter_raw",
             "label": "should not cross"},
        ],
    )
    runner = _runner(tmp_path)
    runner._capture_plots(run_dir)

    kinds = [p["kind"] for p in runner.pending_plot_images]
    assert kinds == ["residuals"]
    # And every kind that does cross is in the published allowlist.
    assert all(k in _PLOT_KIND_ALLOWLIST for k in kinds)


def test_capture_no_manifest_is_silent(tmp_path: Path) -> None:
    """When no manifest exists (the script didn't call any helper),
    capture is a clean no-op — no errors, no images staged. PNGs
    in the dir are ignored."""
    run_dir = tmp_path / ".nora" / "runs" / "r0002"
    plots = run_dir / "_nora_plots"
    plots.mkdir(parents=True)
    (plots / "stray.png").write_bytes(_png_bytes())

    runner = _runner(tmp_path)
    runner._capture_plots(run_dir)
    assert runner.pending_plot_images == []


# ---------------------------------------------------------------------------
# Path safety
# ---------------------------------------------------------------------------

def test_capture_refuses_path_traversal(tmp_path: Path) -> None:
    """A manifest entry with ``file: "../secret.png"`` must be
    refused even if the resolved path exists. The plot file has to
    live inside ``_nora_plots/``."""
    run_dir = _seed_run(
        tmp_path, manifest_lines=[
            {"file": "residuals.png", "kind": "residuals", "label": "ok"},
        ],
    )
    # Drop a file the manifest entry would point at if traversal worked.
    (run_dir / "secret.png").write_bytes(_png_bytes())

    # Add a malicious manifest entry pointing outside _nora_plots.
    plots = run_dir / "_nora_plots"
    with (plots / "manifest.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "file": "../secret.png", "kind": "residuals",
            "label": "evil",
        }) + "\n")

    runner = _runner(tmp_path)
    runner._capture_plots(run_dir)
    names = [p["name"] for p in runner.pending_plot_images]
    assert "secret.png" not in names
    assert names == ["residuals.png"]


def test_capture_refuses_non_png(tmp_path: Path) -> None:
    """A manifest pointing at ``.svg`` / ``.pdf`` / ``.json`` is
    rejected. PNG is the only image format the bridge sends to the
    model — and the executor uses PNG too."""
    run_dir = _seed_run(
        tmp_path, manifest_lines=[
            {"file": "result.svg", "kind": "residuals", "label": "x"},
            {"file": "report.pdf", "kind": "residuals", "label": "x"},
            {"file": "data.json", "kind": "residuals", "label": "x"},
        ],
        extra_files={
            "result.svg": b"<svg/>",
            "report.pdf": b"%PDF-1.4",
            "data.json": b"{}",
        },
    )
    runner = _runner(tmp_path)
    runner._capture_plots(run_dir)
    assert runner.pending_plot_images == []


# ---------------------------------------------------------------------------
# Caps
# ---------------------------------------------------------------------------

def test_capture_drops_oversized_plots(tmp_path: Path) -> None:
    """Plots beyond the 2 MB byte cap are dropped (the on-disk
    file stays for the researcher to view)."""
    run_dir = _seed_run(
        tmp_path, manifest_lines=[
            {"file": "small.png", "kind": "residuals", "label": "ok"},
            {"file": "huge.png", "kind": "residuals", "label": "too big"},
        ],
    )
    # Overwrite huge.png with bytes above the cap.
    (run_dir / "_nora_plots" / "huge.png").write_bytes(b"\x89PNG" + b"x" * (_PLOT_MAX_BYTES + 100))

    runner = _runner(tmp_path)
    runner._capture_plots(run_dir)
    names = [p["name"] for p in runner.pending_plot_images]
    assert names == ["small.png"]


def test_capture_caps_count_per_turn(tmp_path: Path) -> None:
    """If the script writes more plots than the per-turn cap, only
    the most recent N are surfaced (manifest order)."""
    n_total = _PLOT_MAX_PER_TURN + 3
    entries = [
        {"file": f"residual_{i}.png", "kind": "residuals", "label": f"r{i}"}
        for i in range(n_total)
    ]
    run_dir = _seed_run(tmp_path, manifest_lines=entries)
    runner = _runner(tmp_path)
    runner._capture_plots(run_dir)
    assert len(runner.pending_plot_images) == _PLOT_MAX_PER_TURN
    # Most-recent semantics: the last N entries of the manifest.
    expected_names = [
        e["file"] for e in entries[-_PLOT_MAX_PER_TURN:]
    ]
    got_names = [p["name"] for p in runner.pending_plot_images]
    assert got_names == expected_names


# ---------------------------------------------------------------------------
# End-to-end: tool result event triggers capture, next turn merges images
# ---------------------------------------------------------------------------

class _RecordingSession:
    """Stand-in provider session: yields one assistant_text + a
    terminal event, and records the ``images`` arg passed to send()
    so the test can verify the runner's merge logic."""

    def __init__(self) -> None:
        self.last_images: list[dict[str, Any]] | None = None
        self.send_calls = 0

    async def open(self) -> None: ...
    async def close(self) -> None: ...

    async def send(self, prompt: str, images: Any = None) -> AsyncIterator[Any]:
        self.send_calls += 1
        self.last_images = list(images) if images else None
        yield AssistantText(text="ok")
        yield TurnDone()


def test_tool_result_with_run_dir_triggers_capture_and_next_turn_attaches(
    tmp_path: Path,
) -> None:
    """End-to-end: a ToolCallResult event with run_dir pointing at a
    seeded plot dir → runner captures the plot. Next turn's send()
    receives the plot in its images list."""
    run_dir = _seed_run(
        tmp_path, manifest_lines=[
            {"file": "residuals.png", "kind": "residuals",
             "label": "Residual diagnostics"},
        ],
    )

    runner = _runner(tmp_path)

    # Stub session that emits a ToolCallResult mid-stream so the
    # runner's capture branch fires.
    class _PlotEmittingSession:
        async def open(self) -> None: ...
        async def close(self) -> None: ...

        async def send(self, prompt: str, images: Any = None):
            yield ToolCallResult(
                call_id="c1", text='{"type":"linear_regression","n":100}',
                is_error=False, run_dir=str(run_dir), language="R",
            )
            yield AssistantText(text="done")
            yield TurnDone()

    runner._session = _PlotEmittingSession()
    asyncio.run(runner.run_turn(
        "run a regression", images=None,
        on_event=lambda e: None,
        build_context_prefix=lambda cwd: "",
        build_script_prefix=lambda atts, cwd: "",
    ))
    assert len(runner.pending_plot_images) == 1
    assert runner.pending_plot_images[0]["name"] == "residuals.png"

    # Next turn: pending plots should ride along in `images`.
    second = _RecordingSession()
    runner._session = second
    asyncio.run(runner.run_turn(
        "what's the residual fit like?", images=None,
        on_event=lambda e: None,
        build_context_prefix=lambda cwd: "",
        build_script_prefix=lambda atts, cwd: "",
    ))
    assert second.last_images is not None
    assert len(second.last_images) == 1
    assert second.last_images[0]["mime"] == "image/png"
    assert second.last_images[0]["name"] == "residuals.png"
    # Decode round-trip — the bytes are the seeded PNG.
    decoded = base64.b64decode(second.last_images[0]["data"])
    assert decoded.startswith(b"\x89PNG")
    # And the staging is cleared after consumption.
    assert runner.pending_plot_images == []


def test_user_supplied_images_merge_with_pending_plots(tmp_path: Path) -> None:
    """When the researcher attaches their own image AND plots are
    staged, both reach the next send. Pending plots come FIRST so
    the model reads the result-context plots before the
    researcher's screenshot."""
    runner = _runner(tmp_path)
    runner.pending_plot_images = [{
        "data": "Zm9v", "mime": "image/png",
        "name": "residuals.png", "kind": "residuals", "label": "x",
    }]

    rec = _RecordingSession()
    runner._session = rec
    user_image = {"data": "YmFy", "mime": "image/png"}
    asyncio.run(runner.run_turn(
        "see attached", images=[user_image],
        on_event=lambda e: None,
        build_context_prefix=lambda cwd: "",
        build_script_prefix=lambda atts, cwd: "",
    ))
    assert rec.last_images is not None
    assert len(rec.last_images) == 2
    assert rec.last_images[0]["name"] == "residuals.png"  # pending first
    assert rec.last_images[1] is user_image                # user image after


def test_cancel_during_send_restores_pending_plots(tmp_path: Path) -> None:
    """If the next turn is cancelled while sending, the staged
    plots come back to ``pending_plot_images`` so the next attempt
    still carries them. Mirrors how prefix / attachments are
    restored on cancel."""
    runner = _runner(tmp_path)
    runner.pending_plot_images = [{
        "data": "Zm9v", "mime": "image/png",
        "name": "residuals.png", "kind": "residuals", "label": "x",
    }]

    class _CancelMidSend:
        async def open(self) -> None: ...
        async def close(self) -> None: ...

        async def send(self, prompt: str, images: Any = None):
            # Yield once, then raise CancelledError on the next iteration.
            yield AssistantText(text="...")
            raise asyncio.CancelledError()

    runner._session = _CancelMidSend()
    asyncio.run(runner.run_turn(
        "go", images=None,
        on_event=lambda e: None,
        build_context_prefix=lambda cwd: "",
        build_script_prefix=lambda atts, cwd: "",
    ))
    # Restored — same image is still staged.
    assert len(runner.pending_plot_images) == 1
    assert runner.pending_plot_images[0]["name"] == "residuals.png"
