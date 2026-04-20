"""Tests for the crosstab (2D frequency table) sanitizer.

The crosstab's central SDC invariant: **no margins, ever.** With no
margins published, primary suppression alone is sufficient — there's
nothing for an adversary to back-solve against. The tests codify that
invariant so a future contributor can't accidentally add a grand-total
field and weaken the guarantee.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from builder.sanitizer import DEFAULT_CONFIG, sanitize, supported_types
from builder.sdc import suppression_marker


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
    if not result.ok:
        return
    for forbidden in _MARGIN_FIELDS:
        assert forbidden not in result.sanitized, (
            f"margin field {forbidden!r} leaked through — "
            f"no-margins invariant violated"
        )


@given(raw=crosstab_payloads())
def test_crosstab_cells_below_threshold_always_suppressed(raw):
    """No cell below threshold survives as a raw integer."""
    result = sanitize(raw)
    if not result.ok:
        return
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
def test_crosstab_preserves_row_col_keys(raw):
    """Suppression replaces values, never adds or removes row/col keys."""
    result = sanitize(raw)
    if not result.ok:
        return
    out_counts = result.sanitized["counts"]
    in_counts = raw["counts"]
    assert set(out_counts.keys()) == set(in_counts.keys())
    for row_key in in_counts:
        assert set(out_counts[row_key].keys()) == set(in_counts[row_key].keys())


@given(raw=crosstab_payloads())
def test_crosstab_logs_margin_drops_loudly(raw):
    """If the script smuggled a margin in, the transformation log names it."""
    result = sanitize(raw)
    if not result.ok:
        return
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
    assert r.sanitized["counts"]["old"]["F"] == "<10"
    assert r.sanitized["counts"]["old"]["M"] == 30
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
