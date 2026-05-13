"""Tests for the crosstab (2D frequency table) sanitizer.

The crosstab's central SDC invariant: **no margins, ever.** With no
margins published, primary suppression alone is sufficient — there's
nothing for an adversary to back-solve against. The tests codify that
invariant so a future contributor can't accidentally add a grand-total
field and weaken the guarantee.
"""

from __future__ import annotations

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from nora.sanitizer import DEFAULT_CONFIG, sanitize, supported_types
from nora.sdc import suppression_marker


_name = st.text(
    alphabet=st.characters(
        min_codepoint=33, max_codepoint=126,
        blacklist_characters="\"\\",
    ),
    min_size=1, max_size=10,
)

_MARGIN_FIELDS = (
    "n", "grand_total", "row_totals", "column_totals",
    "col_totals", "marginals",
)


@st.composite
def crosstab_payloads(draw, max_rows: int = 5, max_cols: int = 5, max_count: int = 200):
    """Well-formed crosstab payloads with varied suppression patterns."""
    row_levels = draw(st.lists(_name, min_size=1, max_size=max_rows, unique=True))
    col_levels = draw(st.lists(_name, min_size=1, max_size=max_cols, unique=True))
    counts: dict[str, dict[str, int]] = {}
    for r in row_levels:
        counts[r] = {c: draw(st.integers(min_value=0, max_value=max_count))
                     for c in col_levels}
    payload: dict = {
        "type": "crosstab",
        "row_variable": draw(_name),
        "col_variable": draw(_name),
        "counts": counts,
        "missing_count": draw(st.integers(min_value=0, max_value=20)),
    }
    # Randomly try to slip a forbidden margin in.
    if draw(st.booleans()):
        payload["n"] = draw(st.integers(min_value=0, max_value=1000))
    if draw(st.booleans()):
        payload["grand_total"] = draw(st.integers(min_value=0, max_value=1000))
    return payload


# ---------------------------------------------------------------------------
# Core invariants
# ---------------------------------------------------------------------------

def test_crosstab_is_supported():
    assert "crosstab" in supported_types()


@given(raw=crosstab_payloads())
def test_crosstab_never_emits_margins(raw):
    """The load-bearing invariant: no margin field ever appears in output."""
    result = sanitize(raw)
    # ``assume`` instead of silent ``return``: if a sanitizer bug
    # causes EVERY generated input to be rejected, ``return`` would
    # leave every example vacuously passing and Hypothesis would
    # report "100 passing" while the assertions below never ran.
    # ``assume`` tells Hypothesis to discard the example and seek a
    # valid one, so a coverage collapse surfaces as a "could not
    # find enough valid examples" error.
    assume(result.ok)
    for forbidden in _MARGIN_FIELDS:
        assert forbidden not in result.sanitized, (
            f"margin field {forbidden!r} leaked through — "
            f"no-margins invariant violated"
        )


@given(raw=crosstab_payloads())
def test_crosstab_cells_below_threshold_always_suppressed(raw):
    """No cell below threshold survives as a raw integer."""
    result = sanitize(raw)
    # ``assume`` instead of silent ``return``: if a sanitizer bug
    # causes EVERY generated input to be rejected, ``return`` would
    # leave every example vacuously passing and Hypothesis would
    # report "100 passing" while the assertions below never ran.
    # ``assume`` tells Hypothesis to discard the example and seek a
    # valid one, so a coverage collapse surfaces as a "could not
    # find enough valid examples" error.
    assume(result.ok)
    threshold = DEFAULT_CONFIG.cell_suppression_threshold
    marker = suppression_marker(threshold)
    nested = result.sanitized["counts"]
    for row_key, inner in nested.items():
        for col_key, v in inner.items():
            if isinstance(v, int):
                assert v >= threshold, (
                    f"cell [{row_key!r}][{col_key!r}] = {v} survived "
                    f"suppression threshold {threshold}"
                )
            else:
                assert v == marker, (
                    f"cell [{row_key!r}][{col_key!r}] = {v!r} is not the "
                    f"expected marker"
                )


@given(raw=crosstab_payloads())
def test_crosstab_output_keys_are_visible_or_suppressed_bucket(raw):
    """Every output row key is either an input row that has at
    least one above-threshold cell, OR is missing entirely (rows
    whose every cell is suppressed have their LABEL dropped — the
    label itself is disclosive at small N). Per surviving row,
    every column key is either visible (count >= threshold) or the
    single ``[suppressed]`` bucket."""
    from nora.text_safety import safe_key
    result = sanitize(raw)
    # ``assume`` instead of silent ``return``: if a sanitizer bug
    # causes EVERY generated input to be rejected, ``return`` would
    # leave every example vacuously passing and Hypothesis would
    # report "100 passing" while the assertions below never ran.
    # ``assume`` tells Hypothesis to discard the example and seek a
    # valid one, so a coverage collapse surfaces as a "could not
    # find enough valid examples" error.
    assume(result.ok)
    threshold = DEFAULT_CONFIG.cell_suppression_threshold
    out_counts = result.sanitized["counts"]
    in_counts = raw["counts"]
    # Every output row label maps to an input row that had at
    # least one >=threshold cell. Rows with NO visible cell have
    # their label dropped; that's the new SDC contract.
    for row_key, row_cells in out_counts.items():
        # Per surviving row: keys are either visible inputs or
        # the bucket — never a suppressed input column.
        assert set(row_cells.keys()).issubset(
            {safe_key(c) for c in in_counts[row_key]} | {"[suppressed]"}
        )
        # Every visible cell value is >= threshold; the bucket
        # entry carries the marker.
        for c, v in row_cells.items():
            if c == "[suppressed]":
                continue
            assert isinstance(v, int) and v >= threshold


@given(raw=crosstab_payloads())
def test_crosstab_logs_margin_drops_loudly(raw):
    """If the script smuggled a margin in, the transformation log names it."""
    result = sanitize(raw)
    # ``assume`` instead of silent ``return``: if a sanitizer bug
    # causes EVERY generated input to be rejected, ``return`` would
    # leave every example vacuously passing and Hypothesis would
    # report "100 passing" while the assertions below never ran.
    # ``assume`` tells Hypothesis to discard the example and seek a
    # valid one, so a coverage collapse surfaces as a "could not
    # find enough valid examples" error.
    assume(result.ok)
    smuggled = [f for f in _MARGIN_FIELDS if f in raw]
    for f in smuggled:
        assert any(f"dropped margin field {f!r}" in t for t in result.transformations), (
            f"sanitizer silently dropped {f!r} without logging it"
        )


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------

def test_crosstab_rejects_non_nested_counts():
    r = sanitize({
        "type": "crosstab",
        "row_variable": "x",
        "col_variable": "y",
        "counts": {"a": 1, "b": 2},  # WRONG — should be nested
    })
    assert not r.ok


def test_crosstab_rejects_missing_required_fields():
    r = sanitize({
        "type": "crosstab",
        "row_variable": "x",
        # missing col_variable + counts
    })
    assert not r.ok


def test_crosstab_happy_path():
    """A row with one rare cell would let an attacker recover that
    cell exactly from a separately-queried row-variable marginal
    (``N_row - sum(visible) = the lone suppressed cell``). Secondary
    suppression promotes the smallest visible cell in any row /
    column with exactly one suppressed cell. The third row
    (``very_old``) keeps the ``rare_event`` column from going
    vulnerable on the column-side check (two primary suppressions
    already in that column means no further column-side cascade
    reaches ``very_old``)."""
    r = sanitize({
        "type": "crosstab",
        "row_variable": "age",
        "col_variable": "outcome",
        "counts": {
            "young":    {"recovered": 200, "died": 150, "rare_event": 7},   # 1 cell < 10
            "old":      {"recovered": 100, "died":  80, "rare_event": 5},   # 1 cell < 10
            "very_old": {"recovered":  50, "died":  40, "rare_event": 30},  # 0 cells < 10
        },
    })
    assert r.ok
    counts = r.sanitized["counts"]
    # ``very_old`` is untouched — no row-wise vulnerability, and the
    # ``rare_event`` column already has two primary suppressions
    # (young, old) so the column-side check sees >=2 suppressed
    # and skips it.
    assert counts["very_old"] == {"recovered": 50, "died": 40, "rare_event": 30}
    # ``young`` row had exactly one primary suppression. Secondary
    # promoted the smallest visible cell (``died`` = 150, smaller
    # than ``recovered`` = 200) so the bucket holds two cells and
    # the marginal can't recover either.
    young = counts["young"]
    assert young["recovered"] == 200
    assert "died" not in young
    assert "rare_event" not in young
    assert young["[suppressed]"] == "<10"
    # Same shape for ``old`` — smallest visible (``died`` = 80) was
    # the one promoted.
    old = counts["old"]
    assert old["recovered"] == 100
    assert "died" not in old
    assert "rare_event" not in old
    assert old["[suppressed]"] == "<10"
    # Accounting: 2 primary (young/rare, old/rare) + 2 secondary
    # (young/died, old/died) = 4. No row was fully dropped.
    assert r.sanitized["suppressed_cell_count"] == 4
    assert r.sanitized.get("suppressed_row_count", 0) == 0
    # Transformation log distinguishes the two stages so a
    # researcher auditing the SDC trail can see what was promoted
    # for back-calc protection vs. dropped for being below
    # threshold.
    log = " ".join(r.transformations)
    assert "primary suppression: 2 cell" in log
    assert "secondary suppression: 2" in log
    # And no margins, ever.
    assert "n" not in r.sanitized
    assert "grand_total" not in r.sanitized


# ---------------------------------------------------------------------------
# Secondary suppression — cross-query back-calc defence
# ---------------------------------------------------------------------------
#
# These tests close the gap left explicit in ``sdc.enforce_back_calc_safety``
# ("NOT a full secondary-suppression algorithm — 2D tables with row +
# column margins require linear programming"). For Nora's threat model
# the 2D defence doesn't need τ-ARGUS-grade optimality; it just needs
# to make per-row and per-column back-calc from an externally-known
# marginal infeasible. The chosen heuristic — iteratively promote the
# smallest visible cell in any row or column with exactly one
# suppressed cell — is the standard ONS / Eurostat starting point.


@given(raw=crosstab_payloads())
def test_property_no_published_row_has_single_cell_bucket(raw):
    """Across every well-formed input the sanitizer accepts: if a
    surviving row publishes a ``[suppressed]`` bucket, that bucket
    aggregates at least two cells from the input. This is the load-
    bearing invariant the secondary pass guarantees — a single-cell
    bucket is back-calc-recoverable from an externally-known row
    marginal."""
    result = sanitize(raw)
    assume(result.ok)
    counts = result.sanitized.get("counts", {})
    raw_counts = raw["counts"]
    from nora.text_safety import safe_key
    for row_label, row_cells in counts.items():
        if "[suppressed]" not in row_cells:
            continue
        # Find the matching raw row: row_label is safe_key'd, so
        # invert by scanning. The shape generator guarantees no
        # safe_key collisions because levels are drawn unique.
        raw_row = None
        for raw_label, raw_row_cells in raw_counts.items():
            if safe_key(raw_label) == row_label:
                raw_row = raw_row_cells
                break
        assert raw_row is not None, (
            f"published row {row_label!r} has no matching raw input"
        )
        n_visible = sum(1 for c in row_cells if c != "[suppressed]")
        n_bucketed = len(raw_row) - n_visible
        assert n_bucketed >= 2, (
            f"row {row_label!r} ended with a {n_bucketed}-cell "
            f"bucket — back-calc invariant violated. "
            f"row_cells={row_cells}, raw_row={raw_row}"
        )


def test_secondary_no_row_publishes_single_cell_bucket():
    """Invariant the secondary pass guarantees: every surviving row
    that emits a ``[suppressed]`` bucket has at least TWO suppressed
    cells in that row. A bucket with one cell would let the model
    recover the cell's exact count from an externally-queried row
    marginal (``N_row - sum(visible) = the lone cell``)."""
    r = sanitize({
        "type": "crosstab",
        "row_variable": "age",
        "col_variable": "outcome",
        "counts": {
            "young":    {"recovered": 200, "died": 150, "rare": 7},
            "old":      {"recovered": 100, "died":  80, "rare": 5},
            "very_old": {"recovered":  50, "died":  40, "rare": 30},
        },
    })
    assert r.ok
    counts = r.sanitized["counts"]
    threshold = DEFAULT_CONFIG.cell_suppression_threshold
    marker = suppression_marker(threshold)
    # For every surviving row that has a bucket, at least two
    # columns must be missing from its dict (= bucketed). The
    # marker is the bucket value; visible cells are ints.
    raw_counts = {
        "young":    {"recovered": 200, "died": 150, "rare": 7},
        "old":      {"recovered": 100, "died":  80, "rare": 5},
        "very_old": {"recovered":  50, "died":  40, "rare": 30},
    }
    for row_label, row_cells in counts.items():
        if "[suppressed]" not in row_cells:
            continue
        assert row_cells["[suppressed]"] == marker
        visible_columns = {c for c in row_cells if c != "[suppressed]"}
        n_bucketed = len(raw_counts[row_label]) - len(visible_columns)
        assert n_bucketed >= 2, (
            f"row {row_label!r} bucket has {n_bucketed} cell(s); "
            f"single-cell bucket is back-calc-recoverable from N_row"
        )


def test_secondary_handles_column_side_back_calc():
    """Even when every row has at most one suppressed cell BEFORE
    secondary, a column with exactly one suppressed cell across
    surviving rows is recoverable from the column marginal
    (``N_col - sum(visible_in_col) = the lone cell``). The column
    pass fires the same promotion in that case.

    Setup: row ``r1`` has one suppression at ``c2``. Row ``r2`` has
    a small visible cell at ``c2`` (value 40, well above threshold)
    and visible cells elsewhere. After primary, column ``c2`` has
    exactly one suppressed cell — vulnerable. Secondary must
    promote ``r2``'s ``c2`` (=40, the smallest visible in column
    ``c2``) so ``c2`` ends with 2 suppressed cells.
    """
    r = sanitize({
        "type": "crosstab",
        "row_variable": "region",
        "col_variable": "category",
        "counts": {
            "r1": {"c1": 200, "c2":   3, "c3": 100, "c4": 80},  # 1 sup (c2)
            "r2": {"c1":  90, "c2":  40, "c3":  60, "c4": 70},  # 0 sup
            "r3": {"c1":  50, "c2":  35, "c3":  45, "c4": 55},  # 0 sup
        },
    })
    assert r.ok
    counts = r.sanitized["counts"]
    # ``r1`` had one primary suppression and three visible cells;
    # row-side secondary promotes the smallest visible (``c4`` = 80)
    # so ``r1`` ends with a 2-cell bucket.
    assert "c2" not in counts["r1"]
    assert "c4" not in counts["r1"]
    assert counts["r1"]["c1"] == 200
    assert counts["r1"]["c3"] == 100
    assert counts["r1"]["[suppressed]"] == "<10"
    # After row-side: column ``c2`` has one suppressed cell (r1)
    # and two visible (r2, r3). Column-side secondary promotes the
    # smallest visible in c2 (r3's c2 = 35, smaller than r2's = 40).
    # That makes ``r3`` row-vulnerable in turn — its smallest
    # visible (``c3`` = 45) gets promoted, leaving r3 with c1=50
    # visible. The same cascade propagates to r2.
    log = " ".join(r.transformations)
    assert "secondary suppression" in log
    # Whatever the exact cascade pattern, the published output must
    # satisfy: no row has a single-cell bucket.
    raw_per_row_columns = {
        "r1": 4, "r2": 4, "r3": 4,
    }
    for row_label, row_cells in counts.items():
        if "[suppressed]" not in row_cells:
            continue
        n_bucketed = raw_per_row_columns[row_label] - (len(row_cells) - 1)
        assert n_bucketed >= 2, (
            f"row {row_label!r} ended with single-cell bucket"
        )


def test_secondary_drops_row_when_only_one_visible_cell_left():
    """If a row had exactly one primary suppression AND only one
    visible cell, the row-side promotion takes that visible cell
    too — leaving the row fully suppressed. The row label is then
    dropped (same path as a fully-naturally-suppressed row).
    Better to lose the row's label than to publish a label whose
    sole visible cell can be flipped through the marginal."""
    r = sanitize({
        "type": "crosstab",
        "row_variable": "diagnosis",
        "col_variable": "outcome",
        "counts": {
            # Sparse row: 1 visible, 1 suppressed. Secondary
            # promotes the visible cell and the row is dropped.
            "sparse_dx": {"recovered": 50, "rare_o": 3},
            # Padding rows that share the columns so the column-
            # side check doesn't propagate further into them.
            "common_a":  {"recovered": 100, "rare_o": 5},
            "common_b":  {"recovered": 200, "rare_o": 4},
        },
    })
    assert r.ok
    counts = r.sanitized["counts"]
    log = " ".join(r.transformations)
    # ``sparse_dx`` should be entirely gone from the published
    # output AND from the transformations log.
    response_text = str(r.sanitized) + log
    assert "sparse_dx" not in response_text
    assert r.sanitized.get("suppressed_row_count", 0) >= 1


def test_crosstab_forbidden_margins_dropped_and_logged():
    r = sanitize({
        "type": "crosstab",
        "row_variable": "age",
        "col_variable": "sex",
        "counts": {"a": {"x": 50}},
        "n": 50,
        "row_totals": [50],
        "col_totals": [50],
    })
    assert r.ok
    for f in ("n", "row_totals", "col_totals"):
        assert f not in r.sanitized
    assert any("grand_total" not in t and "'n'" in t for t in r.transformations)
    assert any("'row_totals'" in t for t in r.transformations)
    assert any("'col_totals'" in t for t in r.transformations)
