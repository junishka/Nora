"""Tests for the ``read_attached_file`` tool - the model-callable
recall path for files the researcher attached earlier in the session.

Covers:
  - Script return shape (text content, language hint, size, truncation
    marker on >64 KB files).
  - Image return shape (MCP image content block + text descriptor).
  - PDF / EPS rasterisation via the existing sips sidecar (skipped
    when the converter isn't available).
  - Datasets and arbitrary other extensions are refused with a clear
    "use get_schema" hint.
  - Path-traversal attempts are refused.
  - Files in a helper-plot subdir resolve correctly (so the model can
    recall a manifest plot like ``residuals_lm1.png``).
"""

from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path

import pytest

from nora.config import set_cwd
from nora.tools import read_attached_file


# A 1×1 transparent PNG - same fixture as the @-mention tests use.
_TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNgYAAAAAMAAS"
    "sJTYQAAAAASUVORK5CYII="
)


def _call(name: str) -> dict:
    """Run the @tool-decorated handler and return its content payload."""
    return asyncio.run(read_attached_file.handler({"name": name}))


def _text_payload(result: dict) -> dict:
    """Pull the JSON payload out of an MCP text-content response."""
    text_block = next(
        b for b in result["content"] if b.get("type") == "text"
    )
    return json.loads(text_block["text"])


# ---------------------------------------------------------------------------
# Scripts
# ---------------------------------------------------------------------------

def test_script_text_returned_inline(tmp_path: Path) -> None:
    set_cwd(tmp_path)
    (tmp_path / "regression.py").write_text(
        "import pandas as pd\nols(...)\n", encoding="utf-8"
    )
    payload = _text_payload(_call("regression.py"))
    assert payload["status"] == "ok"
    assert payload["kind"] == "script"
    assert payload["language"] == "Python"
    assert payload["truncated"] is False
    assert "import pandas" in payload["content"]


@pytest.mark.parametrize("ext,expected_lang", [
    (".py", "Python"),
    (".do", "Stata"),
    (".r", "R"),
    (".rmd", "R Markdown"),
])
def test_script_language_hint_per_extension(
    tmp_path: Path, ext: str, expected_lang: str,
) -> None:
    """The language hint travels in the result so the model can
    pass the right ``language`` argument to ``submit_script`` if
    it decides to re-run the recalled script."""
    set_cwd(tmp_path)
    name = f"analysis{ext}"
    (tmp_path / name).write_text("# placeholder\n", encoding="utf-8")
    payload = _text_payload(_call(name))
    assert payload["status"] == "ok"
    assert payload["language"] == expected_lang
    assert payload["ext"] == ext


def test_script_oversize_is_head_and_tail_truncated_with_marker(
    tmp_path: Path,
) -> None:
    """Scripts larger than the 96 KB cap come back head+tail-truncated:
    the first half of the byte budget is the start of the file, the
    second half is the end, and an elision marker names the gap.

    The tail is the load-bearing property — it's where save calls
    (``df.to_parquet``, ``write_dta``, ``saveRDS``) live, and the
    answer to "did this script write the dataset out" is invisible
    under head-only truncation.
    """
    set_cwd(tmp_path)
    head_marker = "# HEAD_LINE_DO_NOT_DROP\n"
    tail_marker = "df.to_parquet('out.parquet')\n# TAIL_LINE_DO_NOT_DROP\n"
    middle = "x = 1\n" * 30_000  # ~180 KB
    big = head_marker + middle + tail_marker
    (tmp_path / "big.py").write_text(big, encoding="utf-8")

    payload = _text_payload(_call("big.py"))

    assert payload["status"] == "ok"
    assert payload["truncated"] is True
    assert payload["size"] == len(big.encode("utf-8"))
    content = payload["content"]
    # Content is bounded by cap + the elision marker overhead.
    assert len(content.encode("utf-8")) <= 96 * 1024 + 256
    # Both ends survive — neither head-only nor tail-only truncation.
    assert head_marker.strip() in content
    assert "df.to_parquet" in content
    assert tail_marker.strip().splitlines()[-1] in content
    # The elision marker names the gap so the model knows truncation
    # happened in the middle, not at the edges.
    assert "elided" in content


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------

def test_image_returned_as_mcp_image_block(tmp_path: Path) -> None:
    """A PNG mention should come back with an MCP image content
    block alongside a text descriptor. The Anthropic provider
    forwards the image to the model; the descriptor keeps the
    response from being empty on text-only providers."""
    set_cwd(tmp_path)
    (tmp_path / "residuals.png").write_bytes(_TINY_PNG)

    result = _call("residuals.png")
    blocks = result["content"]
    assert any(b.get("type") == "image" for b in blocks), (
        "image content block missing from result - model wouldn't "
        "see the plot"
    )

    image_block = next(b for b in blocks if b.get("type") == "image")
    assert image_block["mimeType"] == "image/png"
    assert base64.b64decode(image_block["data"]) == _TINY_PNG

    text = _text_payload(result)
    assert text["status"] == "ok"
    assert text["kind"] == "image"
    assert text["name"] == "residuals.png"


def test_image_oversize_refused(tmp_path: Path) -> None:
    set_cwd(tmp_path)
    huge = tmp_path / "huge.png"
    huge.write_bytes(b"\0" * (6 * 1024 * 1024))  # 6 MB
    payload = _text_payload(_call("huge.png"))
    assert payload["status"] == "error"
    assert "5 MB" in payload["reason"]


# ---------------------------------------------------------------------------
# Datasets / unrelated extensions
# ---------------------------------------------------------------------------

def test_dataset_extensions_refused_with_get_schema_hint(tmp_path: Path) -> None:
    """Datasets must NOT be retrievable through this tool - that's
    the SDC line. The error should point the model at get_schema."""
    set_cwd(tmp_path)
    (tmp_path / "panel.dta").write_bytes(b"<stata bytes>")
    payload = _text_payload(_call("panel.dta"))
    assert payload["status"] == "rejected"
    assert "get_schema" in payload["reason"]


def test_log_file_refused(tmp_path: Path) -> None:
    """Logs aren't on the recall allowlist either - the model
    should call expand_result for stored sanitized payloads
    instead."""
    set_cwd(tmp_path)
    (tmp_path / "session.log").write_text("noise", encoding="utf-8")
    payload = _text_payload(_call("session.log"))
    assert payload["status"] == "rejected"


# ---------------------------------------------------------------------------
# Path safety
# ---------------------------------------------------------------------------

def test_path_traversal_basenamed_then_not_found(tmp_path: Path) -> None:
    """A traversal attempt like ``../secret.py`` must be reduced to
    ``secret.py`` (basename) and looked up in cwd. The file is
    NOT in cwd, so the result is "not_found" - no leak of the
    parent-dir file."""
    set_cwd(tmp_path)
    parent_secret = tmp_path.parent / "secret.py"
    parent_secret.write_text("# stolen\n", encoding="utf-8")

    payload = _text_payload(_call("../secret.py"))
    assert payload["status"] == "not_found"


def test_missing_name_arg_returns_error(tmp_path: Path) -> None:
    set_cwd(tmp_path)
    payload = _text_payload(asyncio.run(read_attached_file.handler({})))
    assert payload["status"] == "error"
    assert "name argument" in payload["reason"]


# ---------------------------------------------------------------------------
# Helper-plot subdir resolution
# ---------------------------------------------------------------------------

def test_resolves_files_under_helper_plot_dir(tmp_path: Path) -> None:
    """Plots produced by ``nora.plot_*`` helpers live under
    ``.nora/runs/<id>/_nora_plots/``. The Files panel exposes them;
    so the recall tool should resolve them too - otherwise the
    model can see them in a tool result, but can't fetch them by
    name later."""
    set_cwd(tmp_path)
    plots_dir = tmp_path / ".nora" / "runs" / "run-001" / "_nora_plots"
    plots_dir.mkdir(parents=True)
    (plots_dir / "residuals_lm1.png").write_bytes(_TINY_PNG)

    result = _call("residuals_lm1.png")
    text = _text_payload(result)
    assert text["status"] == "ok"
    assert text["kind"] == "image"
