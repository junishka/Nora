"""Pin the web-UI markdown renderer's italic-emphasis behaviour.

Field report (2026-04-28): when Claude wrote prose that referenced
identifiers like ``fp_dur_resolved`` / ``age_at_arrival`` / ``webal_new``
or used a bare ``*`` as a multiplication operator
(``max(charity_age * (etime_any == 0))``), the renderer's old italic
regexes treated the underscores / asterisks as emphasis markers and
either italicised the middle chunk of an identifier or - worse -
opened an emphasis that ran across multiple sentences until it found
a closing marker.

The fix is a CommonMark-flavoured word-boundary rule: emphasis
markers must sit at a word boundary AND must not be adjacent to
whitespace inside the emphasis run.

The renderer is JS-only (web UI) so we exercise it through ``node``.
``node`` is on every dev box; if it ever isn't, the tests skip
rather than wedge the suite.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

NODE = shutil.which("node")

_REPO_ROOT = Path(__file__).resolve().parent.parent
_RENDERER = _REPO_ROOT / "src" / "nora" / "web" / "markdown.js"


pytestmark = pytest.mark.skipif(
    NODE is None or not _RENDERER.is_file(),
    reason="node not installed or markdown.js missing",
)


def _render(text: str) -> str:
    """Run the in-tree markdown renderer over ``text`` and return
    the raw HTML it produces. Uses a single ``node -e`` invocation
    per call - fast enough for a few dozen tests."""
    js = (
        f"const fs = require('fs');"
        f"const code = fs.readFileSync({json.dumps(str(_RENDERER))}, 'utf8');"
        # The module wraps itself in an IIFE that attaches to
        # window.NoraMarkdown - emulate that surface.
        f"const window = {{}};"
        f"eval(code);"
        f"const input = JSON.parse(process.argv[1]);"
        f"process.stdout.write(window.NoraMarkdown.render(input));"
    )
    proc = subprocess.run(
        [NODE, "-e", js, "--", json.dumps(text)],
        capture_output=True,
        check=False,
        text=True,
        timeout=10,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"renderer crashed:\nstderr={proc.stderr!r}"
            f"\nstdout={proc.stdout!r}"
        )
    return proc.stdout


# ---------------------------------------------------------------------------
# Intra-word underscores (the headline regression)
# ---------------------------------------------------------------------------

def test_intra_word_underscores_are_not_emphasis() -> None:
    """``fp_dur_resolved`` in prose must render literally - no
    ``<em>dur</em>`` slipping into the middle. This is the rule
    CommonMark codifies explicitly, and the bug that motivated the
    word-boundary regex rewrite."""
    out = _render("The variable fp_dur_resolved is the treatment.")
    assert "<em>" not in out
    assert "fp_dur_resolved" in out


@pytest.mark.parametrize("identifier", [
    "fp_dur_resolved",
    "age_at_arrival",
    "ceo_start_year",
    "webal_new",
    "officer_name_clean",
    "etime_any",
    "charity_age",
])
def test_common_underscore_identifiers_pass_through(identifier: str) -> None:
    out = _render(f"prose mentioning {identifier} mid-sentence.")
    assert "<em>" not in out, f"{identifier} got partially italicised"
    assert identifier in out


# ---------------------------------------------------------------------------
# Bare `*` as math/code operator
# ---------------------------------------------------------------------------

def test_bare_asterisk_does_not_open_cross_sentence_emphasis() -> None:
    """Claude often writes Stata / pandas snippets in prose with a
    bare ``*`` for multiplication. Without the word-boundary rule,
    the first ``*`` opened an emphasis that ran to the next ``*``
    and italicised every sentence in between."""
    text = (
        "any / thr / bin / pure / hybrid * interactions. "
        "This is the panel that produced H1 and H2. "
        "Section 2 uses max(charity_age * 0.5) instead."
    )
    out = _render(text)
    assert "<em>" not in out, (
        "a bare `*` used as a multiplication operator opened an "
        "emphasis that should not have triggered - the word-boundary "
        "rule is back to being permissive"
    )


def test_asterisk_with_inner_whitespace_is_not_emphasis() -> None:
    """``"* x"`` and ``"x *"`` are not emphasis. The CommonMark
    flanking rule says markers must not have whitespace immediately
    inside the emphasis run."""
    out = _render("3 * 4 = 12")
    assert "<em>" not in out


# ---------------------------------------------------------------------------
# Real italic still renders
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("This is _italic_ now.", "<em>italic</em>"),
    ("This is *italic* now.", "<em>italic</em>"),
    ("Section 1 _(Steps 0-13)_ is the panel build.", "<em>(Steps 0-13)</em>"),
    ("She said _hello there_ today.", "<em>hello there</em>"),
])
def test_legitimate_italic_still_renders(text: str, expected: str) -> None:
    out = _render(text)
    assert expected in out, f"legit italic broken on {text!r}: got {out!r}"


def test_bold_still_renders() -> None:
    out = _render("**both** pipelines")
    assert "<strong>both</strong>" in out


def test_bold_inside_otherwise_normal_text() -> None:
    """Regression for the field screenshot: ``**both**`` rendered
    bold even though earlier failures had also italicised the
    surrounding sentence. With the fix, only the bold should render
    - no extraneous emphasis."""
    text = "The H3 mature_org moderator is in **both** pipelines."
    out = _render(text)
    assert "<strong>both</strong>" in out
    assert "<em>" not in out


# ---------------------------------------------------------------------------
# Code spans protect their content
# ---------------------------------------------------------------------------

def test_underscores_inside_code_span_stay_literal() -> None:
    """``mature_org`` wrapped in backticks is a code span - the
    inline emphasis rules never run inside it. The rendered <code>
    block must contain the full identifier, not a partial italic."""
    out = _render("The H3 `mature_org` moderator")
    assert "<code>mature_org</code>" in out
    assert "<em>" not in out


def test_asterisks_at_code_span_boundary_stay_literal() -> None:
    """``*x*`` wrapped in backticks must NOT have the inner ``x``
    italicised. The earlier renderInline did substitution-then-
    bold/italic, so the resulting ``<code>*x*</code>`` HTML had its
    asterisks at boundaries (``>`` before, ``<`` after) — the italic
    regex matched right across the closing tag. Placeholder pattern
    fixes it.
    """
    out = _render("The Stata glob `*y` matches every variable")
    assert "<code>*y</code>" in out
    assert "<em>" not in out

    # ``a * b`` inside backticks: bare asterisk used as multiplication.
    out2 = _render("Compute `max(charity_age * 0.5)` per row")
    assert "<code>max(charity_age * 0.5)</code>" in out2
    assert "<em>" not in out2


def test_underscores_at_code_span_boundary_stay_literal() -> None:
    """Mirror of the asterisk test for ``_``. ``_x_`` between
    backticks must not become ``<code><em>x</em></code>``.
    """
    out = _render("The token `_cons` is Stata's intercept")
    assert "<code>_cons</code>" in out
    assert "<em>" not in out
