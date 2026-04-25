"""Property tests for SDC primitives.

The sanitizer leans on these — if precision rounding or cell suppression
misbehave on any input, the whole data-boundary guarantee degrades. Test
them hard.
"""

from __future__ import annotations

import math

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from nora.sdc import (
    MinimumNViolation,
    clamp_precision,
    require_minimum_n,
    round_to_sigfigs,
    sigfigs_for_n,
    suppress_cells_below,
    suppression_marker,
)


# ---------------------------------------------------------------------------
# sigfigs_for_n
# ---------------------------------------------------------------------------

@given(n=st.integers(min_value=0, max_value=10**9))
def test_sigfigs_for_n_in_range(n):
    """Sig-fig output must stay inside the [floor, cap] range for all inputs."""
    s = sigfigs_for_n(n)
    assert 3 <= s <= 8


@given(n=st.integers(min_value=1, max_value=10**9))
def test_sigfigs_for_n_monotonic_by_magnitude(n):
    """Larger magnitudes never give fewer sig figs."""
    # Doubling N stays in the same or next bucket; 100x guarantees >= .
    assert sigfigs_for_n(n * 100) >= sigfigs_for_n(n)


# ---------------------------------------------------------------------------
# round_to_sigfigs
# ---------------------------------------------------------------------------

# Finite floats excluding exactly-zero (which short-circuits in the impl).
_nonzero_finite = st.floats(
    allow_nan=False, allow_infinity=False, min_value=-1e15, max_value=1e15,
).filter(lambda x: x != 0)


@given(x=_nonzero_finite, sigfigs=st.integers(min_value=1, max_value=8))
def test_round_has_at_most_sigfigs(x, sigfigs):
    """The rounded value should have at most `sigfigs` significant digits.

    Measured by: |round(x) - x| <= 0.5 * 10^(floor(log10(|x|)) - sigfigs + 1).
    Equivalent statement: the rounding error is bounded by half of the
    smallest unit at the `sigfigs`-th position.
    """
    r = round_to_sigfigs(x, sigfigs)
    # The rounded value is finite.
    assert math.isfinite(r)
    # Bound the rounding error.
    magnitude = math.floor(math.log10(abs(x)))
    max_error = 0.5 * (10 ** (magnitude - sigfigs + 1))
    # Small tolerance to account for float imprecision in the bound itself.
    assert abs(r - x) <= max_error * 1.000001


@given(sigfigs=st.integers(min_value=1, max_value=8))
def test_round_preserves_sentinels(sigfigs):
    """Zero, inf, and NaN pass through unchanged; rounding them is meaningless."""
    assert round_to_sigfigs(0.0, sigfigs) == 0.0
    assert math.isinf(round_to_sigfigs(float("inf"), sigfigs))
    assert math.isnan(round_to_sigfigs(float("nan"), sigfigs))


@given(x=_nonzero_finite, sigfigs=st.integers(min_value=1, max_value=8))
def test_round_sign_preserved(x, sigfigs):
    """Rounding never flips the sign of a nonzero number."""
    r = round_to_sigfigs(x, sigfigs)
    # r can be 0 if |x| << 1 unit at that sigfig, but sign shouldn't flip.
    if r != 0:
        assert (r > 0) == (x > 0)


@given(x=_nonzero_finite, sigfigs=st.integers(min_value=1, max_value=8))
def test_round_idempotent(x, sigfigs):
    """Rounding to N sig figs twice equals rounding once."""
    r1 = round_to_sigfigs(x, sigfigs)
    r2 = round_to_sigfigs(r1, sigfigs)
    # Tiny floating-point noise possible — allow tiny relative error.
    if r1 == 0:
        assert r2 == 0
    else:
        assert abs(r2 - r1) / abs(r1) < 1e-10


def test_round_rejects_nonpositive_sigfigs():
    with pytest.raises(ValueError):
        round_to_sigfigs(1.5, 0)
    with pytest.raises(ValueError):
        round_to_sigfigs(1.5, -3)


# ---------------------------------------------------------------------------
# clamp_precision (sigfigs_for_n ∘ round_to_sigfigs)
# ---------------------------------------------------------------------------

@given(
    x=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e12, max_value=1e12),
    n=st.integers(min_value=1, max_value=10**7),
)
def test_clamp_precision_finite(x, n):
    """Clamping a finite input always produces a finite output."""
    assert math.isfinite(clamp_precision(x, n))


# ---------------------------------------------------------------------------
# suppress_cells_below
# ---------------------------------------------------------------------------

_count_dict = st.dictionaries(
    keys=st.text(min_size=1, max_size=20),
    values=st.integers(min_value=0, max_value=10**6),
    min_size=0,
    max_size=20,
)


@given(counts=_count_dict, threshold=st.integers(min_value=0, max_value=100))
def test_suppression_removes_all_below_threshold(counts, threshold):
    """No integer cell below threshold survives suppression."""
    result = suppress_cells_below(counts, threshold)
    marker = suppression_marker(threshold)
    for k, v in result.counts.items():
        assert v == marker or (isinstance(v, int) and v >= threshold), (
            f"cell {k!r}={v!r} violated the suppression rule"
        )


@given(counts=_count_dict, threshold=st.integers(min_value=0, max_value=100))
def test_suppression_preserves_keys(counts, threshold):
    """Suppression never adds or removes keys — only replaces values."""
    result = suppress_cells_below(counts, threshold)
    assert set(result.counts.keys()) == set(counts.keys())


@given(counts=_count_dict, threshold=st.integers(min_value=0, max_value=100))
def test_suppression_total_unchanged(counts, threshold):
    """The reported `total_original` matches the sum of all input counts."""
    result = suppress_cells_below(counts, threshold)
    assert result.total_original == sum(counts.values())


def test_suppression_rejects_negatives():
    with pytest.raises(ValueError):
        suppress_cells_below({"A": -1}, threshold=5)


def test_suppression_rejects_nonint():
    with pytest.raises(TypeError):
        suppress_cells_below({"A": 1.5}, threshold=5)  # type: ignore[dict-item]


# ---------------------------------------------------------------------------
# require_minimum_n
# ---------------------------------------------------------------------------

@given(
    n=st.integers(min_value=0, max_value=10_000),
    threshold=st.integers(min_value=0, max_value=10_000),
)
def test_minimum_n_gate(n, threshold):
    """Gate is exactly `n >= threshold`."""
    if n >= threshold:
        require_minimum_n(n, threshold)  # no raise
    else:
        with pytest.raises(MinimumNViolation) as ei:
            require_minimum_n(n, threshold)
        assert ei.value.actual == n
        assert ei.value.required == threshold
