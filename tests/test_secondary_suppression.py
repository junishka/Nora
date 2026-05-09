"""Property tests for secondary cell suppression.

The invariant we need to prove: when the sanitizer emits a 1D
frequency table with the total N, no single suppressed cell is
back-calculable from the margin. Concretely, at most one equation of
the form ``x_i = N - sum(knowns)`` can have a unique solution.

Phrased as a property: if the output has N present, then either
**zero** cells are suppressed OR **at least two** cells are suppressed.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from nora.sanitizer import DEFAULT_CONFIG, sanitize
from nora.sdc import suppression_marker


_name = st.text(
    alphabet=st.characters(
        min_codepoint=33, max_codepoint=126,
        blacklist_characters="\"\\",
    ),
    min_size=1, max_size=10,
)


@st.composite
def freq_table_with_total(draw, max_cells: int = 15, max_count: int = 500):
    """Generate freq_table payloads where n is always the sum of counts."""
    levels = draw(st.lists(_name, min_size=1, max_size=max_cells, unique=True))
    counts = {lv: draw(st.integers(min_value=0, max_value=max_count)) for lv in levels}
    missing = draw(st.integers(min_value=0, max_value=50))
    n = sum(counts.values()) + missing
    return {
        "type": "frequency_table",
        "variable": draw(_name),
        "counts": counts,
        "n": n,
        "missing_count": missing,
    }


@given(raw=freq_table_with_total())
def test_never_exactly_one_suppressed_when_n_present(raw):
    """Core SDC invariant: 0 or >=2 suppressed DISTINCT cells, never
    exactly one. Suppressed cells are now bucketed under a single
    ``[suppressed]`` key in the output, so the count of distinct
    suppressed cells lives in the ``suppressed_cell_count`` field."""
    result = sanitize(raw)
    if not result.ok:
        return
    sanitized = result.sanitized
    if "n" not in sanitized:
        return  # if N isn't published, back-calc isn't a concern
    distinct_suppressed = sanitized.get("suppressed_cell_count", 0)
    assert distinct_suppressed != 1, (
        f"exactly one cell suppressed with total n present — back-calc "
        f"violation. suppressed_cell_count={distinct_suppressed}, "
        f"counts={sanitized['counts']}"
    )


@given(raw=freq_table_with_total())
def test_secondary_only_adds_when_necessary(raw):
    """If >=2 cells are already primary-suppressed, the log shouldn't mention secondary."""
    result = sanitize(raw)
    if not result.ok:
        return
    distinct_suppressed = result.sanitized.get("suppressed_cell_count", 0)
    secondary_logged = any(
        "secondary suppression" in t for t in result.transformations
    )
    # Secondary should fire iff exactly one cell was primary-suppressed AND N present.
    # Reverse-engineer "primary suppressed" from the input.
    primary_candidates = sum(
        1 for v in raw["counts"].values()
        if v < DEFAULT_CONFIG.cell_suppression_threshold
    )
    if primary_candidates == 1 and "n" in result.sanitized:
        assert secondary_logged, (
            "expected secondary suppression but it didn't fire"
        )
        assert distinct_suppressed == 2, (
            f"secondary should have added exactly one cell; got "
            f"suppressed_cell_count={distinct_suppressed}"
        )
    else:
        assert not secondary_logged, (
            "secondary fired when it shouldn't have; "
            f"primary_candidates={primary_candidates}, raw={raw}"
        )


@given(raw=freq_table_with_total())
def test_all_suppressed_cells_use_marker(raw):
    """Visible cells are ints; the single ``[suppressed]`` bucket
    carries the marker. No other shapes."""
    result = sanitize(raw)
    if not result.ok:
        return
    marker = suppression_marker(DEFAULT_CONFIG.cell_suppression_threshold)
    for k, v in result.sanitized["counts"].items():
        if k == "[suppressed]":
            assert v == marker
        else:
            assert isinstance(v, int)


def test_sample_back_calc_scenario():
    """Concrete regression test for the exact case the invariant protects.

    With bucketing, both ``small`` and ``tiny`` get rolled into a
    single ``[suppressed]`` entry; the back-calc invariant is now
    enforced via the count of DISTINCT suppressed cells (which must
    be 0 or >=2 when n is published)."""
    r = sanitize({
        "type": "frequency_table",
        "variable": "state",
        "counts": {"big": 500, "medium": 100, "small": 50, "tiny": 3},
        "n": 653,
        "missing_count": 0,
    })
    assert r.ok
    counts = r.sanitized["counts"]
    marker = suppression_marker(10)
    assert counts["big"] == 500
    assert counts["medium"] == 100
    # ``small`` and ``tiny`` are bucketed; their original labels
    # don't appear in the output dict.
    assert "small" not in counts
    assert "tiny" not in counts
    assert counts["[suppressed]"] == marker
    # Exactly two distinct cells were suppressed → back-calc safe.
    assert r.sanitized["suppressed_cell_count"] == 2
    # n stays published since 2 cells were suppressed (not exactly 1).
    assert "n" in r.sanitized
