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
    "tool_count",
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


def test_tool_count_matches_allowed_tool_names(tmp_path: Path) -> None:
    """The prompt's two ``"the {N} tools below/above"`` lines must
    reflect the actual tool registry. Hardcoding "thirteen" twice
    drifted in the past — adding ``compose_results`` /
    ``list_session_files`` / ``search_in_session_files`` left the
    prose claiming the wrong count for a window. Now the count is
    derived from ``ALLOWED_TOOL_NAMES`` at render time and this
    test pins the link so the two can't desync.
    """
    from nora.tools import ALLOWED_TOOL_NAMES

    rendered = build_system_prompt(tmp_path, "nora")
    n = len(ALLOWED_TOOL_NAMES)
    assert f"the {n} tools below" in rendered
    assert f"Only the {n} above" in rendered


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
    # The .dta-specific "don't reach for the other languages first"
    # steer. Was previously phrased as "switch to stata"; the
    # current prompt expresses the same idea against the .dta
    # case directly.
    assert "Don't reach for R/Python on a .dta" in rendered, (
        "the prompt must tell the model NOT to default to R/Python "
        "for a .dta file; without this it picks whichever language "
        "it used last and loops on missing-package errors"
    )
    # And the generic don't-loop-in-the-same-language note: when a
    # chosen language fails to import the package it needs, switch
    # to the format's native language rather than working around
    # the import.
    assert "switch to the format-native language" in rendered


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


def test_plot_residuals_marked_researcher_only_in_prompt() -> None:
    """``plot_residuals`` is NOT on ``runner._PLOT_KIND_ALLOWLIST``;
    the runner produces the image on disk for the researcher but
    deliberately withholds it from the model's vision. The prompt
    therefore must NOT list it alongside the model-visible
    helpers, and SHOULD surface the researcher-only nature so the
    model doesn't plan around inspecting the image.

    Regression test for the prompt-vs-implementation disagreement
    where ``plot_residuals`` sat under the "model-visible plots
    ONLY" header — the model would expect to see residual plots,
    none would arrive, and the model would either wait, retry, or
    misreport what it had access to."""
    rendered = build_system_prompt(Path("/tmp"), "nora")
    # Find the model-visible block (split by language prefix is
    # noisy — anchor on the two section markers we wrote).
    visible_start = rendered.find("Model-visible helpers")
    researcher_start = rendered.find("Researcher-only helpers")
    assert visible_start != -1, (
        "prompt is missing the 'Model-visible helpers' section "
        "header — split between visible and researcher-only must "
        "stay explicit"
    )
    assert researcher_start != -1, (
        "prompt is missing the 'Researcher-only helpers' section "
        "header"
    )
    assert researcher_start > visible_start, (
        "researcher-only section must come AFTER the model-visible "
        "section (ordering matters for the 'You see only sanctioned "
        "model-visible helper plots' rule below)"
    )
    visible_block = rendered[visible_start:researcher_start]
    # The model-visible block must NOT contain plot_residuals.
    # Check all three language prefixes for completeness.
    for spelling in (
        "nora$plot_residuals",
        "nora_plot_residuals",
        "nora.plot_residuals",
    ):
        assert spelling not in visible_block, (
            f"{spelling!r} is in the model-visible helpers block "
            f"but ``_PLOT_KIND_ALLOWLIST`` excludes 'residuals' — "
            f"the prompt would mislead the model into expecting "
            f"to see residual plots"
        )
    # And the researcher-only block DOES contain them. Cut at the
    # next section header so a downstream mention of plot_residuals
    # (e.g. in a future rules block) doesn't accidentally satisfy
    # this assertion.
    researcher_block = rendered[researcher_start:]
    next_section = researcher_block.find("\nPlot rules:")
    if next_section != -1:
        researcher_block = researcher_block[:next_section]
    for spelling in (
        "nora$plot_residuals",
        "nora_plot_residuals",
        "nora.plot_residuals",
    ):
        assert spelling in researcher_block, (
            f"researcher-only block is missing {spelling!r} — "
            f"the model needs to know the helper exists and can be "
            f"called, just that the image is researcher-visible only"
        )


def test_plot_residuals_consistent_with_runner_allowlist() -> None:
    """Defence-in-depth: the prompt's split between model-visible
    and researcher-only helpers must agree with the runtime gate
    at ``runner._PLOT_KIND_ALLOWLIST``. If a future contributor
    moves ``residuals`` onto the runtime allowlist OR moves any
    of the visible kinds off it, this test surfaces the prompt
    that needs updating."""
    from nora.runner import _PLOT_KIND_ALLOWLIST, _PLOT_KIND_RESEARCHER_ONLY
    # Every kind tracked by the runner is on exactly one of the
    # two sets — sanity check, not specific to this fix.
    assert _PLOT_KIND_ALLOWLIST.isdisjoint(_PLOT_KIND_RESEARCHER_ONLY)
    # The fix's invariant: residuals are researcher-only at runtime.
    assert "residuals" in _PLOT_KIND_RESEARCHER_ONLY
    assert "residuals" not in _PLOT_KIND_ALLOWLIST


def test_formatting_rules_sit_at_end_of_prompt(tmp_path: Path) -> None:
    """Formatting rules drift after long contexts — by the time the
    model is generating a multi-result analytical response, the
    instructions need to be the LAST thing it read, not buried in
    the middle. The "Think hard" closer references the formatting
    block above it; if these get reordered with operational notes
    after them, output regresses to bold sentence-leaders and
    multi-paragraph prose blocks. Pin the structural ordering."""
    rendered = build_system_prompt(tmp_path, "nora")
    fmt_pos = rendered.find("Bold sentence-leaders")
    think_pos = rendered.find("Think hard and thoroughly")
    tool_use_pos = rendered.find("Tool use notes:")
    assert fmt_pos > 0 and think_pos > 0 and tool_use_pos > 0
    # Tool use notes come BEFORE formatting rules.
    assert tool_use_pos < fmt_pos, (
        "Tool use notes must precede formatting rules so formatting "
        "is the last instruction block before the 'Think hard' anchor"
    )
    # Formatting rules come BEFORE the 'Think hard' closer.
    assert fmt_pos < think_pos
    # The 'Think hard' line is the last line before the prompt ends.
    tail = rendered[think_pos:]
    assert len(tail) < 400, (
        f"'Think hard' should be near the very end; trailing "
        f"content is {len(tail)} chars"
    )


def test_no_bold_sentence_leaders_rule_is_imperative(tmp_path: Path) -> None:
    """The model kept reverting to bold sentence-leaders ("**The big
    picture.**", "**Pre-trends clean.**") on long analytical
    responses. The rule needs an explicit anti-pattern name so it
    binds to the failure mode. Inline emphasis bold is allowed —
    the earlier blanket "DO NOT bold words inside prose" overshot
    the failure mode and made the model also drop legitimate
    emphasis."""
    rendered = build_system_prompt(tmp_path, "nora")
    assert "Bold sentence-leaders" in rendered
    assert "are forbidden" in rendered


def test_composite_table_rule_pins_pvalue_in_brackets(tmp_path: Path) -> None:
    """For wide composite spec × outcome matrices, cells must carry
    the p-value in square brackets next to coefficient + SE.
    Significance stars are the old convention and forbidden — explicit
    p-values supersede them. Pin the format so the rule doesn't
    silently revert."""
    rendered = build_system_prompt(tmp_path, "nora")
    assert "Composite cell-format table" in rendered
    assert "[0.002]" in rendered  # the canonical example
    assert "p-value in square brackets" in rendered
    assert "Do NOT use significance stars" in rendered


def test_inline_backtick_rule_present(tmp_path: Path) -> None:
    """Two independent failure modes the prompt must guard:
    (1) the model dropping backticks on data-identifier tokens
    (variable names, column refs) so ``ln_govt_grants`` and
    ``has_np`` render as plain prose — the rule must explicitly
    name variable names / column identifiers so the model uses
    backticks consistently;
    (2) the Stata local-macro syntax landmine (leading backtick +
    trailing apostrophe) which a markdown parser sees as an opening
    code fence and renders broken.
    Pin both halves."""
    rendered = build_system_prompt(tmp_path, "nora")
    assert "Inline backticks for variable names" in rendered
    assert "column identifiers" in rendered
    assert "Stata local-macro syntax" in rendered


def test_loop_directive_for_parameterized_batches_present(
    tmp_path: Path,
) -> None:
    """The model defaults to writing a loop in one script when running
    N parameterized variants (specs, subgroups, sweeps). Without this
    directive, prior single-result habits push toward N separate
    scripts even though the architecture now supports multi-result
    in one call. Pin the directive so a future prompt trim doesn't
    silently drop it."""
    rendered = build_system_prompt(tmp_path, "nora")
    assert "For parameterized batches" in rendered
    assert "write ONE script with a loop" in rendered
    assert "Do NOT submit N separate scripts" in rendered


def test_partial_failure_semantics_documented(tmp_path: Path) -> None:
    """When a script aborts mid-loop, the model receives the helpers
    that emitted before the abort plus the abort cause. The prompt
    must document this so the model knows to read partials and not
    assume "abort" means "no results"."""
    rendered = build_system_prompt(tmp_path, "nora")
    assert "execution_failed_partial" in rendered
    assert "On partial failure" in rendered
    # The load-bearing line: partials in the response are real
    # results, not retry candidates. Phrasing was previously
    # "Do NOT re-run the helpers that already succeeded"; the
    # current prompt expresses the same constraint as a positive
    # framing ("treat partials as ordinary results") plus the
    # imperative.
    assert "Treat partials as ordinary results; don't re-run them" in rendered


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


def test_stata_correlation_helper_is_staged(tmp_path: Path) -> None:
    """``nora_result_correlation`` emits the correlation_matrix payload
    (Pearson / Spearman / Kendall dispatch) for Stata. The .ado must
    reach the run's adopath at runtime — without it, scripts that
    call ``nora_result_correlation`` get ``command not found`` even
    though the file exists in the package."""
    from nora.executor import _stage_runtime
    run = tmp_path / "run"
    run.mkdir()
    lib = _stage_runtime(run, "Stata")
    assert (lib / "nora_result_correlation.ado").is_file()


def test_stata_self_contained_ttest_helper_is_staged(tmp_path: Path) -> None:
    """``nora_ttest`` is the self-contained ttest helper that runs
    the appropriate ``ttest`` form itself, eliminating the r()-
    clobbering foot-gun the legacy ``nora_result_ttest`` had. It
    must reach Stata's adopath at runtime; without staging the
    .ado file the helper isn't found and scripts get a confusing
    "command not found" instead of the expected ttest table."""
    from nora.executor import _stage_runtime
    run = tmp_path / "run"
    run.mkdir()
    lib = _stage_runtime(run, "Stata")
    assert (lib / "nora_ttest.ado").is_file()
    body = (lib / "nora_ttest.ado").read_text(encoding="utf-8")
    # Pin behavioural keywords so a future trim doesn't silently
    # drop the self-contained property: each shape's ttest call,
    # the r() capture before any subsequent r-class operation, and
    # the mutually-exclusive validation.
    assert "ttest `vname' == `paired'" in body
    assert "ttest `vname' `if', by(`by') unequal" in body
    assert "ttest `vname' `if' == `against'" in body
    assert "only one of against(...), paired(...), or by(...)" in body


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
    """Older call sites that omit the ``provider=`` arg should still
    render the Anthropic prompt, since that's the historical default.
    Default behavior must match the pre-split rendering so nothing
    silently regresses."""
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
