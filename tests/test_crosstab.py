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
    r = sanitize({
        "type": "crosstab",
        "row_variable": "age",
        "col_variable": "sex",
        "counts": {
            "young": {"M": 50, "F": 40},
            "old":   {"M": 30, "F": 5},
        },
    })
    assert r.ok
    # 'old' had an above-threshold cell (M=30) so the row label
    # survives; the suppressed F cell is bucketed under the row's
    # ``[suppressed]`` entry. The 'F' column label is hidden because
    # leaving it would tell the model "old/F is the rare cell".
    old = r.sanitized["counts"]["old"]
    assert old["M"] == 30
    assert "F" not in old
    assert old["[suppressed]"] == "<10"
    # Visible cells unchanged.
    assert r.sanitized["counts"]["young"]["M"] == 50
    assert r.sanitized["counts"]["young"]["F"] == 40
    assert "n" not in r.sanitized
    assert "grand_total" not in r.sanitized


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
