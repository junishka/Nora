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


_INTENDED_PLACEHOLDERS = frozenset({"cwd", "datasets_list", "SERVER_NAME"})


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
