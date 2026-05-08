"""Regression tests for the third batch of reviewer-flagged fixes.

Builds on ``test_audit_fixes_2.py`` and ``test_audit_fixes_3.py``; the
issues this file pins:

1. Residual diagnostics were on the model-vision allowlist. The plot
   helpers build them from per-observation residuals + fitted values
   — exactly the row-level fields the JSON sanitizer refuses to
   surface — so the image side channel let row data ride into the
   model around SDC. Residuals are now researcher-only; the summary
   carries a ``researcher_only: true`` marker so the model still
   knows the plot was made.

2. ``_summarize_plot_helpers`` forwarded helper-error ``message``
   (and ``fix``) text raw from ``helper_errors.jsonl``. Those bodies
   come from exceptions raised inside user-authored scripts, so a
   script could leak raw cell values via the exception body. The
   message field is now an allowlist-only forward (well-known
   import / dependency patterns); everything else is replaced with a
   redacted placeholder.

3. ``count_next_context`` stat()'d the whole ``chat_history.jsonl``,
   but the file mixes model-facing records with UI-only enrichments
   (``raw_stdout``, ``raw_stderr``, base64 plot thumbnails). The
   chip overcounted dramatically on plot- or script-heavy sessions.
   The new ``_model_facing_history_chars`` strips UI-only fields
   from each line before summing.

4. The edit-and-rerun path silently dropped attachments / images:
   ``rewind_to`` truncated the original turn's persisted record and
   the rerun called ``send_message(newText)`` with no payload. The
   JS handler now warns the researcher with a clear message before
   proceeding.

5. ``submit_script_file`` resolved only top-level cwd basenames.
   Run-dir scripts (Nora-written ``script.{do,R,py}`` surfaced under
   labeled display names by ``list_session_files``) hit "not_found"
   even though the panel listed them. Now falls back through
   ``find_run_dir_script_by_name`` like ``read_attached_file``.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any


def _mcp_text(payload: dict) -> dict:
    text_block = next(
        b for b in payload["content"] if b.get("type") == "text"
    )
    return json.loads(text_block["text"])


# ---------------------------------------------------------------------------
# 1. Residuals are not visible to the model
# ---------------------------------------------------------------------------

def test_capture_drops_residuals_kind(tmp_path: Path) -> None:
    """A manifest entry with ``kind: residuals`` must NOT cross to
    the model — the per-observation scatter exposes row-level
    residuals + fitted values that the JSON sanitizer would
    refuse. The on-disk file stays for the researcher; only the
    image side channel is closed."""
    from nora.runner import (
        SessionRunner,
        _PLOT_KIND_ALLOWLIST,
        _PLOT_KIND_RESEARCHER_ONLY,
    )

    # Sanity check the allowlists themselves so a future edit that
    # restores residuals to ``_PLOT_KIND_ALLOWLIST`` trips this
    # test loudly rather than silently re-opening the side channel.
    assert "residuals" not in _PLOT_KIND_ALLOWLIST, (
        "residuals must stay off the model-vision allowlist; "
        "see _PLOT_KIND_ALLOWLIST docstring"
    )
    assert "residuals" in _PLOT_KIND_RESEARCHER_ONLY

    run = tmp_path / ".nora" / "runs" / "r1"
    plots = run / "_nora_plots"
    plots.mkdir(parents=True)
    (plots / "residuals.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
    (plots / "manifest.jsonl").write_text(
        json.dumps({
            "file": "residuals.png",
            "kind": "residuals",
            "label": "Residual diagnostics",
        }) + "\n",
        encoding="utf-8",
    )

    runner = SessionRunner(
        cwd=tmp_path, provider="anthropic", model="claude-sonnet-4-6[1m]",
    )
    runner._capture_plots(run)

    # Image NOT staged — the model never sees it.
    assert runner.pending_plot_images == [], (
        "residuals image must not be staged for model vision"
    )


def test_summarize_plot_helpers_marks_residuals_researcher_only(
    tmp_path: Path,
) -> None:
    """The text summary still surfaces the fact that a residuals
    plot was made — otherwise the model loops calling
    ``plot_residuals`` with no acknowledgement — but flags it
    ``researcher_only`` so the model knows it can't see the image."""
    from nora.tools import _summarize_plot_helpers

    plots_dir = tmp_path / "_nora_plots"
    plots_dir.mkdir()
    (plots_dir / "manifest.jsonl").write_text(
        json.dumps({
            "file": "residuals.png",
            "kind": "residuals",
            "label": "Residual diagnostics",
        }) + "\n",
        encoding="utf-8",
    )
    summary = _summarize_plot_helpers(tmp_path)
    assert summary is not None
    succeeded = summary["succeeded"]
    assert len(succeeded) == 1
    assert succeeded[0]["kind"] == "residuals"
    assert succeeded[0].get("researcher_only") is True


def test_summarize_plot_helpers_does_not_mark_visible_kinds_researcher_only(
    tmp_path: Path,
) -> None:
    """Negative regression: kinds that DO cross to the model
    (coefficients, interaction, marginal_effects) must not get the
    researcher_only flag — they're model-visible."""
    from nora.tools import _summarize_plot_helpers

    plots_dir = tmp_path / "_nora_plots"
    plots_dir.mkdir()
    (plots_dir / "manifest.jsonl").write_text(
        json.dumps({
            "file": "coef.png",
            "kind": "coefficients",
            "label": "ok",
        }) + "\n",
        encoding="utf-8",
    )
    summary = _summarize_plot_helpers(tmp_path)
    assert summary is not None
    assert "researcher_only" not in summary["succeeded"][0]


# ---------------------------------------------------------------------------
# 2. Helper error messages — allowlist gate
# ---------------------------------------------------------------------------

def test_summarize_plot_helpers_forwards_known_import_error(
    tmp_path: Path,
) -> None:
    """``No module named 'matplotlib'`` is the canonical case the
    model needs verbatim — it derives ``pip install matplotlib``
    from it. The allowlist must keep this path open."""
    from nora.tools import _summarize_plot_helpers

    plots_dir = tmp_path / "_nora_plots"
    plots_dir.mkdir()
    (plots_dir / "helper_errors.jsonl").write_text(
        json.dumps({
            "helper": "plot_coefficients",
            "error": "ModuleNotFoundError",
            "message": "No module named 'matplotlib'",
            "fix": "pip install matplotlib",
        }) + "\n",
        encoding="utf-8",
    )
    summary = _summarize_plot_helpers(tmp_path)
    assert summary is not None
    fail = summary["failed"][0]
    # Verbatim: the substring 'matplotlib' makes it through the
    # allowlist gate.
    assert "matplotlib" in fail["message"]
    # And the fix hint also rides through (gated on the message
    # being kept verbatim).
    assert fail["fix"] == "pip install matplotlib"


def test_summarize_plot_helpers_redacts_unknown_message(
    tmp_path: Path,
) -> None:
    """An exception body that looks like a row excerpt — pandas
    formatters and user-authored ``raise ValueError(f"bad: {row}")``
    statements both produce strings like this — must be replaced
    with a redacted placeholder. Otherwise raw cell values can
    leak through ``plots.failed`` even when the analysis payload
    sanitizer blocks the JSON path."""
    from nora.tools import _summarize_plot_helpers

    plots_dir = tmp_path / "_nora_plots"
    plots_dir.mkdir()
    (plots_dir / "helper_errors.jsonl").write_text(
        json.dumps({
            "helper": "plot_coefficients",
            "error": "ValueError",
            "message": "bad row: {'patient_id': 12345, 'age': 47, 'income': 89000.0}",
            "fix": "examine the offending row",
        }) + "\n",
        encoding="utf-8",
    )
    summary = _summarize_plot_helpers(tmp_path)
    assert summary is not None
    fail = summary["failed"][0]
    assert "patient_id" not in fail["message"]
    assert "12345" not in fail["message"]
    assert "redacted" in fail["message"].lower()
    # When the message was redacted, the ``fix`` hint is also
    # dropped — its semantic depends on the message, and the
    # exception body could have been an injection vector for it
    # too.
    assert "fix" not in fail


def test_summarize_plot_helpers_message_cap_is_tight(
    tmp_path: Path,
) -> None:
    """Even a known-pattern message that's overlong gets trimmed
    well below the prior 400-char cap — bound is the gate, the
    allowlist match alone is not a license to forward arbitrarily
    long bodies."""
    from nora.tools import (
        _summarize_plot_helpers,
        _PLOT_HELPER_MESSAGE_MAX_LEN,
    )

    plots_dir = tmp_path / "_nora_plots"
    plots_dir.mkdir()
    long_message = "No module named 'matplotlib' " + ("X" * 2000)
    (plots_dir / "helper_errors.jsonl").write_text(
        json.dumps({
            "helper": "plot_coefficients",
            "error": "ModuleNotFoundError",
            "message": long_message,
        }) + "\n",
        encoding="utf-8",
    )
    summary = _summarize_plot_helpers(tmp_path)
    assert summary is not None
    fail = summary["failed"][0]
    assert len(fail["message"]) <= _PLOT_HELPER_MESSAGE_MAX_LEN


# ---------------------------------------------------------------------------
# 3. Context counter projects to model-facing fields only
# ---------------------------------------------------------------------------

def test_model_facing_history_chars_strips_ui_only_fields(
    tmp_path: Path,
) -> None:
    """The persisted log mixes model-facing record bodies with UI
    enrichments (raw_stdout, raw_stderr, base64 plot thumbs). The
    counter must strip those before summing — otherwise the chip
    badly overcounts on plot- or script-heavy sessions."""
    from nora.context_count import _model_facing_history_chars

    log = tmp_path / "chat_history.jsonl"
    # One model-facing record + one tool_result enriched with UI
    # fields that the model never sees.
    big_blob = "X" * 2_000_000  # ~2 MB pretending to be base64 plot data
    big_stdout = "Y" * 30_000   # ~30 KB raw stdout
    records = [
        {"type": "user_message", "text": "fit a regression"},
        {
            "type": "tool_result",
            "call_id": "c1",
            "text": '{"type":"linear_regression","n":100}',
            "raw_stdout": big_stdout,
            "raw_stderr": "",
            "plots": [{"name": "p.png", "data": big_blob, "mime": "image/png"}],
            "plot_diagnostic": "ok",
        },
    ]
    log.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )
    raw_size = log.stat().st_size
    projected = _model_facing_history_chars(log)
    # Must be VASTLY smaller — the 2 MB plot blob alone dwarfs the
    # legitimate model-facing content.
    assert projected < raw_size // 10, (
        f"projected={projected} should be <<{raw_size}; UI-only "
        f"fields must be stripped"
    )
    # But not zero — model-facing fields are still counted.
    assert projected > 0


def test_count_next_context_uses_model_facing_projection(
    tmp_path: Path,
) -> None:
    """End-to-end through the bridge: a session whose history
    contains big UI-only payloads must NOT inflate the chip's
    token count proportional to those UI bytes. Compare to a stat-
    based estimate to make the reduction measurable."""
    from nora.context_count import count_next_context

    nora_dir = tmp_path / ".nora"
    nora_dir.mkdir()
    log = nora_dir / "chat_history.jsonl"
    big_blob = "X" * 1_500_000
    records = [
        {"type": "user_message", "text": "hello"},
        {
            "type": "tool_result",
            "call_id": "c1",
            "text": '{"type":"linear_regression","n":100}',
            "raw_stdout": "Y" * 20_000,
            "plots": [{"name": "p.png", "data": big_blob}],
        },
    ]
    log.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )

    counted = count_next_context(
        cwd=tmp_path,
        system_prompt_chars=10_000,
        tool_schema_chars=20_000,
        ceiling=1_000_000,
        request_id=1,
    )

    raw_size = log.stat().st_size
    # If the chip used stat() we'd get ~raw_size/3.5 ≈ 430k tokens
    # just from history. With projection, history contributes only
    # the model-facing part.
    naive_token_estimate = raw_size // 3.5
    assert counted.tokens < naive_token_estimate / 2, (
        f"chip shouldn't reflect UI-only fields; got {counted.tokens} "
        f"vs naive {naive_token_estimate}"
    )


# ---------------------------------------------------------------------------
# 5. submit_script_file run-dir fallback
# ---------------------------------------------------------------------------

def test_submit_script_file_resolves_run_dir_display_name(
    tmp_path: Path,
) -> None:
    """The mention dropdown lists Nora-written run-dir scripts under
    label-derived display names. Without the run-dir fallback,
    submit_script_file refused them with ``not_found`` even though
    they appeared in the panel — defeating the purpose-built
    run-this-file tool. Mirrors the third fallback
    ``read_attached_file`` already does."""
    from nora.config import use_cwd
    from nora.tools import HANDLERS

    run_dir = tmp_path / ".nora" / "runs" / "20260101_120000_abcd1234"
    run_dir.mkdir(parents=True)
    (run_dir / "script.do").write_text("regress y x\n", encoding="utf-8")
    (run_dir / "meta.json").write_text(
        json.dumps({"label": "Linear Regression Run"}),
        encoding="utf-8",
    )

    # Sanity: the listing surfaces the display name.
    from nora.run_files import enumerate_run_dir_scripts
    entries = enumerate_run_dir_scripts(tmp_path)
    assert len(entries) == 1
    display = entries[0].display_name
    assert display.endswith(".do")

    # Pre-fix: this errored before reaching submit_script. Now it
    # falls back through find_run_dir_script_by_name and either
    # runs (if executor deps available) or fails downstream — we
    # only care that it RESOLVED the file.
    with use_cwd(tmp_path):
        result = asyncio.run(HANDLERS["submit_script_file"]({
            "name": display,
        }))
    body = _mcp_text(result)
    # The run-dir resolution succeeded if the body's status is NOT
    # the resolution-failure status. Downstream ``submit_script``
    # may still error on missing executor deps, but that's a
    # different code path — the bug we're pinning is the lookup
    # never finding the file in the first place.
    assert body.get("status") != "not_found", (
        f"run-dir display name {display!r} should resolve via the "
        f"new fallback; got {body}"
    )


def test_submit_script_file_still_refuses_missing_names(
    tmp_path: Path,
) -> None:
    """Negative regression: the new fallback must NOT admit names
    that don't exist anywhere — neither cwd top-level nor any run
    dir."""
    from nora.config import use_cwd
    from nora.tools import HANDLERS

    with use_cwd(tmp_path):
        result = asyncio.run(HANDLERS["submit_script_file"]({
            "name": "definitely-not-real.do",
        }))
    body = _mcp_text(result)
    assert body.get("status") == "not_found"
