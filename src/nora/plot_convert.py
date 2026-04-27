"""Plot-format conversion: PDF → PNG via macOS ``sips``.

Stata's PNG export depends on the ``Graph2png`` translator, which is
absent on many macOS installs (``translator Graph2png not found``).
The Stata plot helpers fall back to PDF (Stata 17+ writes natively;
Stata 15-16 via ``Graph2pdf``) when PNG fails. PDF is great for the
researcher (Preview opens it natively) but the model's vision API
only accepts raster image bytes — we must convert to PNG before
attaching as image content on the next turn.

This module wraps the conversion in one place so both the runner
(model-vision path) and the bridge (researcher-thumbnail path) use
the same logic + caching.

macOS-only: ``/usr/bin/sips`` is the system tool for image format
conversion. It handles PDF → PNG, JPEG → PNG, etc. without any
extra install. On non-macOS platforms (Nora is macOS-only today)
the conversion silently fails and the caller falls back to "no
PNG available" — same posture as ``Graph2png`` not being installed.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


# Output sidecars from this module land at ``<original>.nora.png``
# so a successive conversion attempt can detect "already done" by
# mtime, and so the researcher's "Open in Preview" workflow on
# the original PDF stays unaffected.
_SIDECAR_SUFFIX = ".nora.png"


# Formats sips can rasterize directly to PNG. PDF works via PDFKit
# (always); EPS works on most macOS via the system's PostScript-to-
# PDF pipeline. ``.gph`` is genuinely Stata-only — there is no
# pure-macOS path that doesn't require re-running Stata.
_CONVERTIBLE_SUFFIXES: frozenset[str] = frozenset({".pdf", ".eps"})


def png_for(path: Path) -> Path | None:
    """Return a PNG path for ``path``, converting if needed.

    - ``.png`` / ``.jpg`` / ``.jpeg``: returned as-is (already raster).
    - ``.pdf`` / ``.eps``: converted via ``sips`` to a sibling
      ``<basename>.nora.png`` and that path returned. Cached: if
      the sidecar exists and is newer than the source, conversion
      is skipped. EPS conversion succeeds on most macOS versions
      via the system PostScript-to-PDF pipeline; falls through to
      None if it doesn't.
    - Anything else (``.gph``, ``.svg``, …): returns ``None``.
      ``.gph`` in particular has no conversion path that doesn't
      require re-running Stata; the caller treats these as
      researcher-only.

    Returns ``None`` on any conversion failure. "No PNG" is a
    normal degraded state, not a programming bug.
    """
    if not path.is_file():
        return None
    suffix = path.suffix.lower()
    if suffix in (".png", ".jpg", ".jpeg"):
        return path
    if suffix not in _CONVERTIBLE_SUFFIXES:
        # .gph / .svg / unknown — outside what sips can convert
        # without extra tooling. Researcher opens them via "Show
        # folder" / "Open in Stata".
        return None

    sidecar = path.with_name(path.stem + _SIDECAR_SUFFIX)
    try:
        src_mtime = path.stat().st_mtime
    except OSError:
        return None
    if sidecar.is_file():
        try:
            if sidecar.stat().st_mtime >= src_mtime:
                return sidecar
        except OSError:
            pass

    sips = _find_sips()
    if sips is None:
        return None

    try:
        # ``sips -s format png -Z 1600 in.<ext> --out out.png``: rasterizes
        # the first page to PNG, capping the longest side at 1600px.
        # Without ``-Z``, sips defaults to the source's native DPI (often
        # 72), producing a tiny image that looks pixelated even when the
        # researcher clicks to enlarge it. 1600px matches the Stata
        # helper's PNG width so PDF/PNG paths produce visually similar
        # output. Stata graphs are always single-page so multi-page
        # PDFs aren't a concern.
        result = subprocess.run(
            [sips, "-s", "format", "png", "-Z", "1600",
             str(path), "--out", str(sidecar)],
            capture_output=True, text=True, timeout=15, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0 or not sidecar.is_file():
        return None
    return sidecar


def _find_sips() -> str | None:
    """Locate ``sips`` on macOS. Returns the absolute path or None."""
    # Stable system path on macOS — try directly first to avoid
    # depending on PATH state at subprocess time.
    direct = "/usr/bin/sips"
    if Path(direct).is_file():
        return direct
    return shutil.which("sips")
