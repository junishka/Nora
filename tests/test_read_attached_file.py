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


# ---------------------------------------------------------------------------
# Run-dir script resolution — the recovery path after a rewind clears
# the chat history. Scripts Nora wrote on prior submit_script calls
# live at ``<cwd>/.nora/runs/<id>/script.{do,R,py}`` and surface in
# the Files panel under labeled or fallback names; ``read_attached_file``
# must resolve those same names.
# ---------------------------------------------------------------------------

def test_resolves_run_dir_script_by_label(tmp_path: Path) -> None:
    """A script Nora wrote with ``submit_script(label="H1a Path A...")``
    is on disk at ``<run_dir>/script.do``. The Files panel shows it as
    ``H1a Path A: op margin, FP-only.do`` (the cleaned label). The
    model must be able to pass that same name to ``read_attached_file``
    and get the script back — otherwise a rewound conversation leaves
    the script visible but unfetchable."""
    set_cwd(tmp_path)
    run_dir = tmp_path / ".nora" / "runs" / "20260507T120000Z_aaaaaaaa"
    run_dir.mkdir(parents=True)
    (run_dir / "script.do").write_text(
        "use \"data.dta\", clear\nregress y x\n", encoding="utf-8",
    )

    from nora.store import get_store
    get_store(tmp_path).insert(
        label="H1a Path A: op margin, FP-only",
        analysis_type="linear_regression",
        sanitized_payload={"type": "linear_regression"},
        language="Stata",
        script_code="use \"data.dta\", clear\nregress y x\n",
        transformations=[],
        raw_log_path=str(run_dir),
        script_run_id="run-aaaaaaaa",
    )

    payload = _text_payload(_call("H1a Path A: op margin, FP-only.do"))
    assert payload["status"] == "ok"
    assert payload["kind"] == "script"
    assert payload["language"] == "Stata"
    assert "use \"data.dta\", clear" in payload["content"]


def test_resolves_run_dir_script_by_short_id_fallback(
    tmp_path: Path,
) -> None:
    """When the model omitted ``label`` on ``submit_script``, the
    panel surfaces the script as ``script_<short_id>.do``. The recall
    path must accept the same name."""
    set_cwd(tmp_path)
    run_dir = tmp_path / ".nora" / "runs" / "20260507T120100Z_bbbbbbbb"
    run_dir.mkdir(parents=True)
    (run_dir / "script.do").write_text(
        "use \"data.dta\", clear\n", encoding="utf-8",
    )
    # No store row inserted — simulates the script-crashed-before-any-
    # helper-fired path. Display name falls back to script_<short_id>.

    payload = _text_payload(_call("script_bbbbbbbb.do"))
    assert payload["status"] == "ok"
    assert payload["kind"] == "script"
    assert "use \"data.dta\"" in payload["content"]


def test_run_dir_script_lookup_blocked_after_rewind(
    tmp_path: Path,
) -> None:
    """A rewind hides results in the store; the on-disk run dir
    remains. Earlier behaviour let the model still fetch the
    script via ``read_attached_file`` (and discover it via
    ``list_session_files``), defeating the rewind: the model could
    re-fetch the discarded branch's analysis verbatim by name. The
    model-facing path now filters run-dir lookups against the
    visible (non-hidden) result set, so a hidden script is
    not_found from the model's perspective. The Files panel
    intentionally still shows it so the researcher can decide
    whether to delete it."""
    set_cwd(tmp_path)
    run_dir = tmp_path / ".nora" / "runs" / "20260507T120200Z_cccccccc"
    run_dir.mkdir(parents=True)
    (run_dir / "script.do").write_text(
        "regress y x\n", encoding="utf-8",
    )

    from nora.store import get_store
    store = get_store(tmp_path)
    row = store.insert(
        label="M27-M38 base spec",
        analysis_type="linear_regression",
        sanitized_payload={"type": "linear_regression"},
        language="Stata",
        script_code="regress y x\n",
        transformations=[],
        raw_log_path=str(run_dir),
        script_run_id="run-cccccccc",
    )
    # Simulate a rewind: hide the row.
    store.hide_results_not_in(set(), reason="rewind")

    payload = _text_payload(_call("M27-M38 base spec.do"))
    assert payload["status"] == "not_found"
    # And confirm hide actually fired (otherwise the test is trivial).
    assert store.get(row.id) is None  # default include_hidden=False


# ---------------------------------------------------------------------------
# Display-name round-trip — the model only ever sees ``safe_text(name)``
# from list_session_files. When the on-disk name is long enough (or
# carries embedded whitespace / control chars) that ``safe_text`` rewrites
# it, the displayed name no longer equals the basename and a direct
# ``Path`` lookup fails. The recall path must scan with the same
# sanitisation so the model can fetch what it saw.
# ---------------------------------------------------------------------------

def test_resolves_truncated_long_filename_via_display_name(
    tmp_path: Path,
) -> None:
    """A 200-char autogenerated filename hits ``safe_text``'s 120-char
    cap → the listing surfaces it with a ``[TRUNCATED]`` marker. The
    model passes that marker-bearing name back to ``read_attached_file``;
    direct path resolution will fail (no on-disk file with that name).
    The fallback must scan cwd and match by ``safe_text(child.name)``."""
    from nora.text_safety import safe_text

    set_cwd(tmp_path)
    long_stem = "lm_diagnostics_run_" + ("x" * 200)
    on_disk = tmp_path / f"{long_stem}.py"
    on_disk.write_text("import pandas as pd\n", encoding="utf-8")

    # The display name the model sees in list_session_files.
    displayed = safe_text(on_disk.name)
    assert displayed != on_disk.name  # would otherwise be a no-op test
    assert "[TRUNCATED]" in displayed

    payload = _text_payload(_call(displayed))
    assert payload["status"] == "ok", payload
    assert payload["kind"] == "script"
    assert "import pandas" in payload["content"]


def test_resolves_whitespace_normalised_filename(tmp_path: Path) -> None:
    """A filename with embedded literal newlines (rare but legal on
    macOS / Linux) is surfaced through ``safe_text`` with whitespace
    flattened to single spaces. The recall path must round-trip back
    to the on-disk file when the model passes the flattened form."""
    from nora.text_safety import safe_text

    set_cwd(tmp_path)
    on_disk = tmp_path / "ok\nmid\tend.py"
    on_disk.write_text("x = 1\n", encoding="utf-8")

    displayed = safe_text(on_disk.name)
    assert "\n" not in displayed
    assert displayed != on_disk.name

    payload = _text_payload(_call(displayed))
    assert payload["status"] == "ok", payload
    assert payload["kind"] == "script"


def test_resolves_plot_with_truncated_long_filename(tmp_path: Path) -> None:
    """Same round-trip as the cwd case but for plot files under
    ``.nora/runs/<id>/_nora_plots/``. A model that recalls a plot from
    a run with autogenerated long filenames should still resolve via
    the displayed name."""
    from nora.text_safety import safe_text

    set_cwd(tmp_path)
    plots_dir = tmp_path / ".nora" / "runs" / "run-002" / "_nora_plots"
    plots_dir.mkdir(parents=True)
    long_stem = "residuals_lm_fit_" + ("z" * 200)
    on_disk = plots_dir / f"{long_stem}.png"
    on_disk.write_bytes(_TINY_PNG)

    displayed = safe_text(on_disk.name)
    assert "[TRUNCATED]" in displayed

    result = _call(displayed)
    text = _text_payload(result)
    assert text["status"] == "ok", text
    assert text["kind"] == "image"
