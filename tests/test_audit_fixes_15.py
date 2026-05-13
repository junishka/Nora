"""Regression tests for the fifteenth batch of reviewer-flagged
fixes.

The behaviors pinned here:

1. ``enumerate_run_dir_scripts`` disambiguates display names by
   counting duplicates over the FULL candidate set, not over the
   ``max_count`` slice. Listing (``max_count=12``) and lookup
   (``max_count=64``) now agree on which entries need a
   ``(short_id)`` suffix, so any name the model sees in the
   listing round-trips through ``find_run_dir_script_by_name``.

2. Plot-manifest consumers refuse to read a manifest that still
   contains entries carrying ``_token``. The executor's
   ``_filter_plot_manifest`` strips that field from every entry
   it validates; observing it on the consumer side means the
   filter didn't complete its sweep, so the manifest is
   untrusted as a whole. Closes the residual gap in the absolute-
   last-resort ``PlotManifestUnsanitizable`` path, and also
   defends against any future regression in the filter that lets
   a token-bearing entry slip through.

3. Python's ``plot_interaction`` runs categorical tick labels
   through ``safe_text`` before rendering. R's
   ``nora$plot_interaction`` does the same via a new
   ``nora$.safe_tick_label`` helper. Without this, raw category
   labels (control chars, bidi overrides, zero-width characters,
   prompt-like text) ride into the model-visible image,
   bypassing the JSON/text path's safety gate. Stata is immune
   by design (numeric-only).

4. Stata's ``nora_result_regress`` gates both ``condition_number``
   and ``vif`` on classical OLS (``e(cmd) == "regress"`` AND
   ``e(vce) in ("", "ols")``). Under robust or clustered VCE,
   ``e(V)`` is the sandwich/cluster covariance, not σ²·(X'X)^-1,
   so the eigenvalue-ratio and ``SE²·TSS/σ²`` formulas no longer
   recover Belsley-Kuh-Welsch condition index or VIF. Publishing
   them in that regime would silently report wrong numbers.

5. ``search_in_session_files`` enforces both the global match
   cap and the global char cap as HARD caps. The prior pre-loop
   checks allowed up to ``max_matches_per_file - 1`` extra
   matches past the cap once a file's matches were appended in
   full. The new per-match accounting refuses to add a match
   that would push either budget past its limit.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# 1. Run-dir disambiguation is stable across listing and lookup windows
# ---------------------------------------------------------------------------


def _make_run_dir_with_label(
    cwd: Path, run_basename: str, label: str, ext: str = ".do",
) -> Path:
    """Create a run dir under ``cwd/.nora/runs/`` whose label maps to
    ``label`` and whose script file is at ``script<ext>``. Returns
    the run-dir path."""
    runs_root = cwd / ".nora" / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    run_dir = runs_root / run_basename
    run_dir.mkdir()
    (run_dir / f"script{ext}").write_text(f"* {label}\n", encoding="utf-8")
    # Mirror the run-store's label persistence: ``label.txt`` next
    # to ``script.{ext}`` is what ``_labels_by_run_basename`` reads
    # first (umbrella label, vs the per-helper label fallback in
    # results.db).
    (run_dir / "label.txt").write_text(label, encoding="utf-8")
    return run_dir


def test_run_dir_disambiguation_stable_across_windows(tmp_path: Path) -> None:
    """Two runs with the same label, 15 distinct ones in between.

    Listing (``max_count=12``) sees one ``H1.do`` in its window
    (the newest occurrence); under the old code it would
    therefore omit the ``(short_id)`` suffix. Lookup
    (``max_count=64``) sees BOTH copies and would add the suffix
    to each. The fix tallies counts over the full candidate set,
    so both windows arrive at the same display name."""
    from nora.run_files import (
        enumerate_run_dir_scripts,
        find_run_dir_script_by_name,
    )
    import os
    import time

    base = time.time() - 1000
    # Older H1 (will be position ~17 by mtime).
    older = _make_run_dir_with_label(
        tmp_path, "run_old_aaaaaaaa", "H1",
    )
    os.utime(older / "script.do", (base, base))
    # 15 unique-named runs between them (still inside the top-64
    # window but outside the top-12).
    for i in range(15):
        d = _make_run_dir_with_label(
            tmp_path, f"run_mid_{i:08d}", f"unique_{i:02d}",
        )
        os.utime(d / "script.do", (base + i + 1, base + i + 1))
    # Newer H1 (will be position 1 by mtime — the only ``H1.do``
    # the listing window sees under the old code).
    newer = _make_run_dir_with_label(
        tmp_path, "run_new_bbbbbbbb", "H1",
    )
    os.utime(newer / "script.do", (base + 100, base + 100))

    # Listing window: model sees these names.
    listing = enumerate_run_dir_scripts(tmp_path, max_count=12)
    listed_names = [e.display_name for e in listing]

    # Find the H1 entry in the listing. Under the fix it must
    # carry a ``(short_id)`` suffix because the FULL candidate
    # set has two H1s, even though only one fits the top-12 slice.
    h1_in_listing = [n for n in listed_names if "H1" in n]
    assert len(h1_in_listing) == 1, (
        f"expected exactly one H1 entry in the top-12 listing; "
        f"got {h1_in_listing}"
    )
    h1_listed_name = h1_in_listing[0]
    assert h1_listed_name != "H1.do", (
        "listing window omitted the disambiguation suffix; lookup "
        "with max_count=64 would compute a different name and the "
        "round-trip would break"
    )

    # Round-trip: the name the model saw must resolve.
    resolved = find_run_dir_script_by_name(tmp_path, h1_listed_name)
    assert resolved is not None, (
        f"name {h1_listed_name!r} from listing window did not "
        f"resolve in lookup window"
    )
    # And it should be the newer H1 (the one within the top-12).
    assert resolved.parent == newer


# ---------------------------------------------------------------------------
# 2. Plot-manifest consumers validate _token against the run-token registry
# ---------------------------------------------------------------------------
#
# Defense-in-depth update: rather than rejecting the manifest when
# any entry still carries ``_token`` (the original batch-15 design),
# the consumer-side gate now validates each entry's token against
# the per-run token registered in ``executor._RUN_TOKEN_REGISTRY``.
# Validated entries have their token stripped from the in-memory
# copy; forged / missing-token / no-registry entries are dropped.
# Same shape across ``_capture_plots`` / ``_summarize_plot_helpers``
# / ``_manifest_allowed_plot_kinds``.


def test_capture_plots_drops_entries_with_wrong_token(
    tmp_path: Path,
) -> None:
    """Build a run dir whose ``_nora_plots/manifest.jsonl`` carries
    a forged entry (wrong ``_token``) and a legit entry (matching
    ``_token``). With a token registered, only the legit entry
    survives; the forged one is silently dropped."""
    from nora.executor import register_run_token
    from nora.runner import SessionRunner
    runner = SessionRunner.__new__(SessionRunner)
    runner.pending_plot_images = []  # type: ignore[attr-defined]

    plots = tmp_path / "_nora_plots"
    plots.mkdir()
    (plots / "real.png").write_bytes(
        b"\x89PNG\r\n\x1a\n" + b"\x00" * 64,
    )
    (plots / "forged.png").write_bytes(
        b"\x89PNG\r\n\x1a\n" + b"\x00" * 64,
    )
    manifest = plots / "manifest.jsonl"
    real_token = "deadbeef" * 4
    manifest.write_text(
        json.dumps({
            "file": "real.png", "kind": "coefficients",
            "_token": real_token,
        }) + "\n" +
        json.dumps({
            "file": "forged.png", "kind": "coefficients",
            "_token": "wrong-token",
        }) + "\n",
        encoding="utf-8",
    )
    register_run_token(tmp_path, real_token)

    runner._capture_plots(tmp_path)  # type: ignore[attr-defined]
    staged_names = [
        img.get("name")
        for img in runner.pending_plot_images  # type: ignore[attr-defined]
    ]
    assert "real.png" in staged_names
    assert "forged.png" not in staged_names


def test_capture_plots_drops_everything_without_registered_token(
    tmp_path: Path,
) -> None:
    """If no token is registered for the run (replay / re-attach /
    crashed before registration), ``_capture_plots`` fails closed:
    every entry is dropped. The risk it guards against is a stale
    or attacker-supplied manifest from an earlier run being trusted
    by ``kind`` alone."""
    from nora.runner import SessionRunner
    runner = SessionRunner.__new__(SessionRunner)
    runner.pending_plot_images = []  # type: ignore[attr-defined]

    plots = tmp_path / "_nora_plots"
    plots.mkdir()
    (plots / "fig.png").write_bytes(
        b"\x89PNG\r\n\x1a\n" + b"\x00" * 64,
    )
    manifest = plots / "manifest.jsonl"
    manifest.write_text(
        json.dumps({
            "file": "fig.png", "kind": "coefficients",
            "_token": "anything",
        }) + "\n",
        encoding="utf-8",
    )

    runner._capture_plots(tmp_path)  # type: ignore[attr-defined]
    assert runner.pending_plot_images == []  # type: ignore[attr-defined]


def test_manifest_allowed_plot_kinds_trusts_sanitized_file(
    tmp_path: Path,
) -> None:
    """Recall path trusts the executor's filter to have sanitized
    the on-disk manifest by the time we read it. The filter either
    (a) rewrote with validated entries (forged ones already
    dropped), (b) neutralized the file via unlink/rename so
    ``is_file()`` returns False, or (c) raised
    PlotManifestUnsanitizable so the run was marked failed.

    Under (a), every entry on disk passed the token check; we
    just match by (file, kind). The token field IS kept in the
    entries but isn't re-validated here — that's the in-session
    paths' job."""
    from nora.tools import _manifest_allowed_plot_kinds

    plots = tmp_path / "_nora_plots"
    plots.mkdir()
    manifest = plots / "manifest.jsonl"
    # Post-filter shape: only validated entries remain on disk.
    manifest.write_text(
        json.dumps({
            "file": "real.png", "kind": "coefficients",
            "_token": "any-validated-token",
        }) + "\n",
        encoding="utf-8",
    )

    assert _manifest_allowed_plot_kinds(plots, "real.png") == "coefficients"


def test_manifest_allowed_plot_kinds_returns_none_when_neutralized(
    tmp_path: Path,
) -> None:
    """When the executor's filter could not rewrite and fell
    through to unlink/rename, the manifest file is gone from
    disk. ``_manifest_allowed_plot_kinds`` returns None because
    ``manifest.is_file()`` fails — the neutralization is the
    protection at the recall path."""
    from nora.tools import _manifest_allowed_plot_kinds

    plots = tmp_path / "_nora_plots"
    plots.mkdir()
    # No manifest.jsonl on disk (neutralized path).
    assert _manifest_allowed_plot_kinds(plots, "fig.png") is None


# ---------------------------------------------------------------------------
# 3. Interaction plot tick labels go through safe_text
# ---------------------------------------------------------------------------


def test_plot_interaction_python_safe_tick_label_source() -> None:
    """The Python helper's categorical branch must call into
    ``safe_text`` for every tick label before passing them to
    matplotlib. Verified structurally: the call site uses the
    ``safe_text`` primitive, and the prior bare ``_trim`` call
    is gone."""
    py = (
        Path(__file__).resolve().parents[1]
        / "src" / "nora" / "runtime" / "nora.py"
    )
    src = py.read_text(encoding="utf-8")

    fn_open = src.find("def plot_interaction(")
    assert fn_open != -1, "plot_interaction not found"
    fn_close = src.find("\ndef ", fn_open + 1)
    body = src[fn_open:fn_close if fn_close != -1 else None]

    # The categorical branch must import safe_text and call it
    # for every tick label. The exact local helper name doesn't
    # matter, but the call has to be there.
    assert "from nora.text_safety import safe_text" in body, (
        "plot_interaction (categorical branch) must use safe_text "
        "from nora.text_safety on tick labels"
    )
    # The earlier code wrapped only with ``_trim``; we accept
    # that helper still existing as long as safe_text wraps it.
    # Reject the specific old call shape so a future refactor
    # cannot silently drop safe_text.
    assert not re.search(
        r"set_xticklabels\(\[_trim\(g\)\s+for\s+g\s+in\s+grid\]",
        body,
    ), "raw _trim-only tick labels still present"


def test_plot_interaction_r_safe_tick_label_helper_present() -> None:
    """The R helper must define ``nora$.safe_tick_label`` and use
    it for the categorical-x branch. The helper strips control
    chars / bidi / zero-width and caps length."""
    r_path = (
        Path(__file__).resolve().parents[1]
        / "src" / "nora" / "runtime" / "nora.R"
    )
    src = r_path.read_text(encoding="utf-8")

    assert "nora$.safe_tick_label <- function" in src, (
        "missing nora$.safe_tick_label helper in nora.R"
    )
    # The categorical-x branch must call it instead of using
    # ``as.character(grid)`` directly.
    barplot_block_open = src.find("graphics::barplot(")
    assert barplot_block_open != -1
    barplot_block = src[barplot_block_open - 800:barplot_block_open + 200]
    assert "nora$.safe_tick_label" in barplot_block, (
        "barplot branch of plot_interaction must run grid labels "
        "through nora$.safe_tick_label before passing them in"
    )


# ---------------------------------------------------------------------------
# 4. Stata regress diagnostics gated on classical OLS
# ---------------------------------------------------------------------------


def test_stata_regress_diagnostics_gated_on_classical_ols() -> None:
    """The ``condition_number`` and ``vif`` blocks must both gate
    on classical OLS — ``e(cmd) == "regress"`` AND
    ``e(vce) in ("", "ols")``. Under robust / clustered VCE
    ``e(V)`` is the sandwich covariance and the BKW / VIF formulas
    no longer recover the design-matrix quantities; under a GLM
    ``e(V)`` comes from the score Hessian and the same arguments
    don't apply."""
    ado = (
        Path(__file__).resolve().parents[1]
        / "src" / "nora" / "runtime" / "nora_result_regress.ado"
    )
    src = ado.read_text(encoding="utf-8")

    # The combined predicate must exist somewhere in the file.
    assert re.search(
        r'"`e\(cmd\)\'"\s*==\s*"regress".+?"`e\(vce\)\'"\s*==\s*""[^&]*'
        r'\|\|?\s*"`e\(vce\)\'"\s*==\s*"ols"',
        src, re.DOTALL,
    ) is not None or re.search(
        r'_classical_ols\s*=\s*\("`e\(cmd\)\'"\s*==\s*"regress"\)',
        src,
    ) is not None, (
        "classical-OLS predicate (regress + vce in {empty, ols}) "
        "not found"
    )

    # condition_number must be inside the gate. The
    # ``file write `fh' `","condition_number"`` line is what
    # publishes it; that line must sit inside an
    # ``if _classical_ols`` block.
    cn_pos = src.find('"condition_number"')
    assert cn_pos != -1
    preceding = src[:cn_pos]
    # Walk back to find the nearest enclosing ``if``. The closest
    # ``if`` before the publish must reference the classical-OLS
    # gate.
    if_open = preceding.rfind("\n    if ")
    assert if_open != -1
    if_line = preceding[if_open:if_open + 200]
    assert "_classical_ols" in if_line, (
        f"condition_number publish is not gated on classical OLS; "
        f"nearest enclosing if-line: {if_line!r}"
    )

    # VIF: the ``"vif":{`` write must also be inside an
    # ``if _classical_ols`` gate.
    vif_pos = src.find('"vif":{')
    assert vif_pos != -1
    preceding = src[:vif_pos]
    if_open = preceding.rfind("\n    if ")
    assert if_open != -1
    if_line = preceding[if_open:if_open + 200]
    assert "_classical_ols" in if_line, (
        f"vif publish is not gated on classical OLS; nearest "
        f"enclosing if-line: {if_line!r}"
    )


# ---------------------------------------------------------------------------
# 5. search_in_session_files caps are hard caps
# ---------------------------------------------------------------------------


def test_search_files_match_cap_is_hard(tmp_path: Path) -> None:
    """The 200-match global cap must not be exceeded. The prior
    pre-loop check let a file's full per-file batch land before
    re-testing the budget; with ``max_matches_per_file=50`` and a
    sequence of 50-match files, the response could carry 249
    matches against a documented cap of 200.

    Setup: stage 6 script files, each with 50 lines containing
    the query. Use ``max_matches_per_file=50`` to maximise the
    overshoot window. Confirm the response's ``total_matches`` is
    at most 200 and that ``truncated`` is set."""
    from nora.config import set_cwd
    from nora.file_provenance import initialize
    from nora.tools import (
        HANDLERS, _SEARCH_FILES_TOTAL_MATCHES_CAP,
    )
    set_cwd(tmp_path)

    # 6 files × 50 matches each = 300 candidates, well over the
    # 200-match cap.
    for i in range(6):
        body = "\n".join(["TARGET line"] * 50)
        (tmp_path / f"file_{i:02d}.py").write_text(body, encoding="utf-8")
    initialize(tmp_path)

    payload = asyncio.run(
        HANDLERS["search_in_session_files"]({
            "query": "TARGET", "max_matches_per_file": 50,
        }),
    )
    body = json.loads(payload["content"][0]["text"])
    assert body["status"] == "ok"
    assert body["total_matches"] <= _SEARCH_FILES_TOTAL_MATCHES_CAP, (
        f"total_matches={body['total_matches']} exceeds the "
        f"{_SEARCH_FILES_TOTAL_MATCHES_CAP}-match cap"
    )
    # And the response must flag that it was truncated.
    assert body["truncated"], (
        "response did not flag truncation despite hitting the cap"
    )


def test_search_files_char_cap_is_hard(tmp_path: Path) -> None:
    """Same shape for the character cap. Each match excerpt
    carries a 240-byte line excerpt + ~12 bytes of structural
    cost, so a file of ~5 over-cap-length matches can add ~1.25 KB
    of rendered chars. Stack enough of those to cross the
    60_000-char global cap and confirm the response stays
    within budget."""
    from nora.config import set_cwd
    from nora.file_provenance import initialize
    from nora.tools import (
        HANDLERS, _SEARCH_FILES_LINE_EXCERPT_CAP,
        _SEARCH_FILES_TOTAL_CHARS_CAP,
    )
    set_cwd(tmp_path)

    # 50 matches per file, each at the excerpt cap (240 chars).
    # 50 matches × (12 + 240) = 12_600 chars per file. Six files
    # give 75_600 chars, well over the 60_000 cap.
    long_line = "X" * (_SEARCH_FILES_LINE_EXCERPT_CAP + 50) + " TARGET"
    body = "\n".join([long_line] * 50)
    for i in range(6):
        (tmp_path / f"big_{i:02d}.py").write_text(body, encoding="utf-8")
    initialize(tmp_path)

    payload = asyncio.run(
        HANDLERS["search_in_session_files"]({
            "query": "TARGET", "max_matches_per_file": 50,
        }),
    )
    body = json.loads(payload["content"][0]["text"])
    assert body["status"] == "ok"
    # Approximate the same char accounting the tool uses.
    rendered = 0
    for row in body["results"]:
        for m in row["matches"]:
            rendered += 12 + len(m.get("text", ""))
    assert rendered <= _SEARCH_FILES_TOTAL_CHARS_CAP, (
        f"rendered_chars={rendered} exceeds the "
        f"{_SEARCH_FILES_TOTAL_CHARS_CAP}-char cap"
    )
    assert body["truncated"]
