"""The model popup must stay on screen at any window size.

The native window is ``resizable=True`` with no minimum size, so the
stylesheet cannot assume the default 960x720. The first anchoring fix
(right-anchored to the chip wrap with a viewport-clamped max-width)
only held at normal sizes — measured against the real layout in a
browser, the popup's top sat at -12px at 600x400 and -92px at
480x320, its left at -28px at 320px wide, and with
``overflow: visible`` the clipped options were unreachable. Two
structural causes: the shared ``.policy-popup`` ``min-width: 320px``
beats any max-width clamp (CSS min-width wins), and a wrap-relative
anchor inherits the wrap's variable inset from the window edge
(footer padding plus, mid-session, the context chip), which no
static ``calc(100vw - N)`` can account for.

The fix positions ``#model-popup`` against the viewport
(``position: fixed``) so every clamp is exact, clamps ``min-width``
to the viewport too, and restores ``max-height`` + ``overflow-y:
auto`` scrolling on short windows. Re-measured after the fix at
960x720 / 600x400 / 480x320 / 400x300 / 340x500 / 320x568, with the
sidebar expanded and collapsed and the context chip hidden and
visible: top >= 12 and left >= 12 in all 24 scenarios, every option
reachable, and the 960x720 position within a few pixels of the old
anchor.

These are source assertions (no JS/CSS runner in this repo): they
pin the structural properties the browser measurement validated.
"""

from __future__ import annotations

import re
from pathlib import Path


def _style_css_without_comments() -> str:
    src = (
        Path(__file__).resolve().parent.parent
        / "src" / "nora" / "web" / "style.css"
    ).read_text(encoding="utf-8")
    return re.sub(r"/\*.*?\*/", "", src, flags=re.DOTALL)


def _model_popup_blocks() -> list[str]:
    """Every ``#model-popup { ... }`` declaration block (the base rule
    and any media-query copies), comments stripped."""
    code = _style_css_without_comments()
    return [
        m.group(1)
        for m in re.finditer(r"#model-popup\s*\{([^}]*)\}", code)
    ]


def test_model_popup_is_viewport_positioned() -> None:
    """A wrap-relative anchor inherits the wrap's inset from the
    window edge — footer padding plus the context chip — so only the
    viewport as containing block makes the width/height clamps exact."""
    blocks = _model_popup_blocks()
    assert any("position: fixed" in b for b in blocks), (
        "#model-popup must be position: fixed; any chip-relative "
        "anchor puts it off screen at small window sizes"
    )


def test_model_popup_overrides_the_shared_percent_bottom() -> None:
    """The shared .policy-popup rule sets ``bottom: calc(100% + 8px)``
    — correct for an absolutely positioned popup (100% = the chip
    wrap's height) but catastrophic under position: fixed, where 100%
    resolves against the viewport height and flings the popup fully
    off the top of the window. The fixed popup must restate bottom
    as an absolute offset."""
    blocks = _model_popup_blocks()
    fixed_block = next(b for b in blocks if "position: fixed" in b)
    m = re.search(r"bottom:\s*([^;]+);", fixed_block)
    assert m is not None, "#model-popup must set its own bottom offset"
    assert "%" not in m.group(1), (
        "#model-popup's bottom must not be percentage-based once the "
        "viewport is its containing block"
    )


def test_model_popup_width_floor_yields_to_the_viewport() -> None:
    """CSS min-width beats max-width, so the shared 320px floor
    reintroduced the horizontal overflow the viewport-clamped
    max-width was added to prevent (left -28px at a 320px-wide
    window). Both bounds must carry the viewport clamp."""
    blocks = _model_popup_blocks()
    joined = "\n".join(blocks)
    assert "max-width: min(560px, calc(100vw - 24px))" in joined
    assert "min-width: min(320px, calc(100vw - 24px))" in joined, (
        "the shared .policy-popup min-width: 320px wins over "
        "max-width and pushes the popup off screen below ~350px "
        "window widths unless it is viewport-clamped too"
    )


def test_model_popup_scrolls_on_short_windows() -> None:
    """The popup is ~364px tall and its overflow: visible override
    (kept for the hover tooltips) means anything past the window top
    is unreachable — there is no scrollbar to bring it back. Short
    windows must cap the height to the viewport and restore
    overflow-y: auto."""
    code = _style_css_without_comments()
    media = re.search(
        r"@media\s*\(max-height:[^)]+\)\s*\{(.*?)\n\}", code, re.DOTALL,
    )
    assert media is not None, (
        "style.css needs a max-height media query that lets the model "
        "popup scroll on short windows"
    )
    body = media.group(1)
    assert "#model-popup" in body
    assert "overflow-y: auto" in body
    assert re.search(r"max-height:\s*calc\(100vh[^;]*\);", body), (
        "the short-window cap must track the viewport height, not a "
        "fixed pixel value"
    )
