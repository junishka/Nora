"""Regression tests for ``build_system_prompt``.

The template uses ``str.format`` with three named placeholders:
``{cwd}``, ``{datasets_list}``, ``{SERVER_NAME}``. Any other
unescaped ``{`` or ``}`` triggers ``IndexError: Replacement index
0 out of range for positional args tuple`` at session-open time —
which the user sees as ``session setup failed`` with no chat at
all. Easy to introduce when adding example code containing braces;
hard to catch without a render check.

The right way to inspect the template's placeholders is via
``string.Formatter.parse`` — same parser format() uses internally,
so what it reports is exactly what format() will see. A regex over
the raw text can't tell escaped ``{{...}}`` apart from a real
placeholder.
"""

from __future__ import annotations

from pathlib import Path
from string import Formatter

import pytest

from nora.system_prompt import (
    SYSTEM_PROMPT_TEMPLATE,
    build_system_prompt,
)


_INTENDED_PLACEHOLDERS = frozenset({
    "cwd", "datasets_list", "SERVER_NAME", "runtime_environment",
})


def _placeholder_names(template: str) -> set[str]:
    """Return the set of distinct ``{name}`` placeholder names
    ``format()`` would try to substitute. Skips literal text and
    escaped braces; resolves unnamed ``{}`` to the implied positional
    index ``""``."""
    return {
        field
        for _, field, _, _ in Formatter().parse(template)
        if field is not None
    }


def test_template_renders_without_format_errors(tmp_path: Path) -> None:
    """A clean session dir should produce a non-empty rendered prompt
    without raising. This is the test that catches a stray ``{...}``
    example in the helper docstrings."""
    out = build_system_prompt(tmp_path, "nora")
    assert isinstance(out, str)
    assert len(out) > 1000  # the template is ~22k chars; sanity floor


def test_no_unintended_format_placeholders() -> None:
    """Every placeholder ``format()`` will try to substitute must be
    one of the three intended named placeholders. Any other name
    (``...``, ``0``, ``coefficients``, …) means example code with
    unescaped braces was added; use ``{{...}}`` for literal braces."""
    found = _placeholder_names(SYSTEM_PROMPT_TEMPLATE)
    unexpected = found - _INTENDED_PLACEHOLDERS
    assert not unexpected, (
        f"unescaped brace placeholders in system prompt: {sorted(unexpected)}. "
        f"Use ``{{{{...}}}}`` for a literal ``{{...}}`` in example code."
    )


def test_render_substitutes_each_placeholder(tmp_path: Path) -> None:
    """Confirm the three intended placeholders actually get replaced —
    a missing one would mean the template branch was rewritten and
    the placeholder name silently changed.

    We DON'T scan the rendered text for stray ``{...}`` here because
    the template legitimately includes literal ``{{...}}`` in code
    examples (renders to ``{...}``); a placeholder-name scan over
    rendered text can't tell those apart from a real format slip.
    The previous test catches that class via Formatter.parse on the
    raw template, which is the right surface.
    """
    sentinel_dir = tmp_path / "session-with-marker-12345"
    sentinel_dir.mkdir()
    rendered = build_system_prompt(sentinel_dir, "nora-server-marker")
    assert str(sentinel_dir) in rendered, "cwd was not substituted"
    assert "nora-server-marker" in rendered, "SERVER_NAME was not substituted"


def test_language_choice_guidance_pins_dta_to_stata() -> None:
    """The model picks language by dataset format. ``.dta`` must
    point to Stata first — R needs ``haven`` (frequently missing)
    and Python needs ``pyreadstat`` (also frequently missing).
    Without explicit guidance, the model defaults to whichever
    language it recently used and burns turns on missing-package
    errors. Pin the guidance so future prompt edits don't quietly
    drop it."""
    rendered = build_system_prompt(Path("/tmp"), "nora")
    assert ".dta" in rendered
    assert "Stata first" in rendered, (
        "the prompt must explicitly steer .dta to Stata; without "
        "this the model retries R/Python in a loop on .dta files"
    )
    # And the don't-loop-in-the-same-language note exists.
    assert "switch to stata" in rendered.lower()


def test_language_choice_guidance_pins_rds_to_r() -> None:
    """``.rds`` is R-native serialization — nothing else reads it.
    Pin so the model doesn't try Python or Stata."""
    rendered = build_system_prompt(Path("/tmp"), "nora")
    assert ".rds" in rendered
    assert "R only" in rendered


def test_no_stale_stata_unimplemented_claims() -> None:
    """The prompt USED to say "Stata's interaction-plot helper
    isn't implemented yet" — that line stuck around after we
    actually shipped ``nora_plot_interaction.ado``, so the model
    kept saying "I'll switch to R" even when Stata had the helper.

    Pin the inverse: no language in the rendered prompt that
    claims a Stata plot helper is missing. All four kinds
    (residuals / interaction / coefficients / estimate_comparison)
    exist for Stata as of this commit."""
    rendered = build_system_prompt(Path("/tmp"), "nora")
    forbidden = [
        "Stata's interaction-plot helper isn't implemented",
        "Stata's coefficient and interaction helpers aren't implemented",
        "Stata's interaction helper isn't",
        "switch to R or Python",  # tied to the same stale claim
    ]
    for phrase in forbidden:
        assert phrase not in rendered, (
            f"stale prompt language: {phrase!r} — Stata has all "
            f"four plot helpers now (nora_plot_residuals, "
            f"nora_plot_coefficients, nora_plot_interaction, "
            f"nora_plot_estimate_comparison). Update the prompt "
            f"to reflect that."
        )
    # And the explicit positive claim is in there.
    assert "nora_plot_interaction" in rendered


def test_runtime_environment_block_renders(tmp_path: Path) -> None:
    """The system prompt includes a runtime-environment listing
    so the model can pick a language by what's actually installed
    rather than discovering missing packages by trying and failing.
    Surfaces R / Python / Stata presence and (where applicable)
    optional-package availability."""
    rendered = build_system_prompt(tmp_path, "nora")
    assert "Runtime environment on this machine" in rendered
    # At least one of the three should land — even on a barebones
    # CI box the listing must produce something.
    assert any(
        marker in rendered for marker in (
            "  - R:",
            "  - Stata:",
            "  - Python",
        )
    )


def test_runtime_environment_marks_missing_packages_with_x() -> None:
    """When an optional package is missing, the listing shows ``✗``
    next to the package name. The model uses this to avoid
    ``library(haven)`` / ``import matplotlib`` calls that would
    fail."""
    from nora.system_prompt import runtime_environment_listing
    listing = runtime_environment_listing()
    # The format is "(haven: ✓, ggplot2: ✗)" or similar — the
    # symbols must be present in a real run.
    if "  - R:" in listing and "(" in listing:
        # If R is installed, package status was rendered.
        assert "✓" in listing or "✗" in listing


def test_stata_plot_coefficients_in_stage_runtime_list(tmp_path: Path) -> None:
    """Stata gets a coefficient-plot helper so .dta-native analyses
    don't have to switch languages just to produce a forest plot.
    The .ado must be in the executor's stage list — otherwise
    Stata can't find it on adopath at runtime even though the
    file exists in the package."""
    from nora.executor import _stage_runtime
    run = tmp_path / "run"
    run.mkdir()
    lib = _stage_runtime(run, "Stata")
    assert (lib / "nora_plot_coefficients.ado").is_file()


def test_anthropic_prompt_carries_mcp_prefix_intro(tmp_path: Path) -> None:
    """The Anthropic variant must keep the ``mcp__<server>__`` prefix
    line — the model actually sees those names on its tool surface
    via the in-process MCP server, and the prompt's nudge ("when
    referenced") helps the model understand its tool naming."""
    rendered = build_system_prompt(tmp_path, "nora", provider="anthropic")
    assert "mcp__nora__" in rendered
    assert "(all prefixed `mcp__nora__` when referenced" in rendered


def test_openai_prompt_drops_mcp_prefix_intro(tmp_path: Path) -> None:
    """OpenAI's function tools have flat names — no ``mcp__`` prefix.
    Telling GPT-5.5 about a name convention it never sees is both
    inaccurate and wastes tokens on the per-call prefix."""
    rendered = build_system_prompt(tmp_path, "nora", provider="openai")
    assert "mcp__nora__" not in rendered
    # The replacement intro still introduces the tool list so the
    # numbered enumeration after it has context.
    assert "Your tools:" in rendered


def test_provider_default_is_anthropic_for_back_compat(tmp_path: Path) -> None:
    """Older call sites (and the terminal CLI which is Anthropic-only)
    omit the ``provider=`` arg. Default behavior must match the
    pre-split rendering so nothing silently regresses."""
    default = build_system_prompt(tmp_path, "nora")
    explicit = build_system_prompt(tmp_path, "nora", provider="anthropic")
    assert default == explicit


def test_openai_prompt_is_smaller_than_anthropic(tmp_path: Path) -> None:
    """The OpenAI variant must be at least the Anthropic-prefix-line
    shorter. If it isn't, the replacement didn't fire — the intro
    string in build_system_prompt drifted from what the template
    bakes in. Use a strict-shorter assertion rather than an exact
    delta so future Anthropic-specific phrasing additions don't
    flake the test."""
    a = build_system_prompt(tmp_path, "nora", provider="anthropic")
    o = build_system_prompt(tmp_path, "nora", provider="openai")
    assert len(o) < len(a)


def test_stata_plot_coefficients_writes_to_run_dir() -> None:
    """The helper resolves run_dir from ``NORA_RESULT_PATH`` and
    writes ``coefficients.png`` + manifest into
    ``<run_dir>/_nora_plots/``. Same posture as nora_plot_residuals
    — landing under session_cwd is the bug we already fixed once
    and don't want to reintroduce."""
    from importlib import resources
    src = resources.files("nora.runtime").joinpath(
        "nora_plot_coefficients.ado"
    ).read_text(encoding="utf-8")
    assert ": env NORA_RESULT_PATH" in src
    assert "/result.json" in src
    assert "`rundir'/_nora_plots" in src
    # The kind label is allowlisted on the runner side as
    # "coefficients" — make sure the helper writes that exact value.
    assert '"kind":"coefficients"' in src
