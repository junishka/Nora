"""Regression tests for the sixteenth batch of reviewer-flagged
fixes.

The behaviors pinned here:

1. The descriptive sanitizer drops ``min_value`` / ``max_value``
   unconditionally. The earlier per-variable opt-in via
   ``SDCConfig.non_disclosive_variables`` was unsafe under the
   threat model: ``source_dataset``, ``variable``, and the values
   themselves are all model/script-controlled, and the sanitizer
   cannot bind the reported values to the named variable's
   actual column. A typed-helper marker doesn't close it (helpers
   accept caller-supplied min/max), and refactoring the helper to
   take a Series doesn't either (the caller still chooses the
   Series and label independently). The channel is closed; the
   policy field stays for documented intent.

2. ``request_data`` ``quartiles`` blends at integer-position
   percentiles. pandas' default linear quantile returns the exact
   sorted observation when ``q*(n-1)`` is an integer (n=33, 37,
   41, ...). After 2-sigfig rounding, an exact-individual value
   can pass through unchanged. The helper now midpoint-blends
   with the adjacent sorted observation so every published
   quartile is the average of two distinct values.

3. The linear-regression sanitizer validates vcov invariants: a
   real σ²·(X'X)^-1 matrix is symmetric and its diagonals are
   SE². A hostile payload emitted through generic ``result(
   type="linear_regression", vcov=...)`` could otherwise carry
   up to N² attacker-shaped numeric cells; the per-cell key /
   finiteness checks don't catch that. The whole vcov field is
   dropped on invariant violation.

4. The correlation_matrix sanitizer rejects payloads that fail
   correlation-matrix invariants: completeness (every declared
   variable has a row covering every other declared variable),
   symmetric within tolerance, and diagonals = 1. Real
   ``df.corr()`` matrices satisfy these; an attacker-crafted
   payload smuggling encoded numeric values does not.
"""

from __future__ import annotations

import pytest

from nora.sanitizer import sanitize


# ---------------------------------------------------------------------------
# 1. Descriptive min/max passthrough is closed
# ---------------------------------------------------------------------------


def test_descriptive_min_max_dropped_with_no_opt_in() -> None:
    """Baseline: without any opt-in, min/max have always been
    dropped. Pin the baseline so the closure is visible."""
    payload = {
        "type": "descriptive",
        "variable": "income",
        "n": 1000,
        "mean": 50000.0,
        "sd": 12000.0,
        "missing_count": 5,
        "min_value": 1.0,
        "max_value": 1500000.0,
    }
    r = sanitize(payload)
    assert r.ok
    assert "min_value" not in r.sanitized
    assert "max_value" not in r.sanitized


def test_descriptive_min_max_dropped_even_with_explicit_opt_in() -> None:
    """The closure: even when the variable is on the opt-in list,
    min/max are dropped. The sanitizer cannot bind the reported
    values to the named variable's actual column, so the opt-in
    field is inert here."""
    from dataclasses import replace as dc_replace

    from nora.sanitizer import DEFAULT_CONFIG

    cfg = dc_replace(
        DEFAULT_CONFIG,
        non_disclosive_variables=frozenset({"age", "education_years"}),
    )
    payload = {
        "type": "descriptive",
        "variable": "age",
        "n": 1000,
        "mean": 42.5,
        "sd": 12.3,
        "missing_count": 0,
        "min_value": 18,
        "max_value": 89,
    }
    r = sanitize(payload, cfg)
    assert r.ok
    assert "min_value" not in r.sanitized
    assert "max_value" not in r.sanitized


def test_from_summarize_no_longer_accepts_min_max_kwargs() -> None:
    """The Python helper's signature no longer carries
    ``min_value`` / ``max_value`` parameters. Passing them as
    keyword arguments raises TypeError so a researcher can't
    silently send fields the sanitizer would drop anyway.

    Source-level check rather than importing the runtime module:
    ``nora.runtime.nora`` enforces ``NORA_RUN_TOKEN`` at import
    time, so we can't pull it into the test process directly.
    Verifying the signature in source is enough — Python's parser
    enforces the rest."""
    import re
    from pathlib import Path
    src = (
        Path(__file__).resolve().parents[1]
        / "src" / "nora" / "runtime" / "nora.py"
    ).read_text(encoding="utf-8")
    fn_open = src.find("def from_summarize(")
    assert fn_open != -1, "from_summarize not found"
    # Find the function-signature close (the ``):``).
    sig_close = src.find(") -> None:", fn_open)
    assert sig_close != -1
    signature = src[fn_open:sig_close]
    assert "min_value" not in signature, (
        f"from_summarize still accepts min_value as a parameter: "
        f"{signature!r}"
    )
    assert "max_value" not in signature, (
        f"from_summarize still accepts max_value as a parameter: "
        f"{signature!r}"
    )


# ---------------------------------------------------------------------------
# 2. Quartiles midpoint-blend at integer positions
# ---------------------------------------------------------------------------


def test_quartiles_never_returns_exact_sorted_observation() -> None:
    """For n in {33, 37, 41, ...} the linear-method position
    ``(n-1)*0.25`` is integer, so pandas' default quantile
    returns ``sorted[k]`` exactly. The fix forces a midpoint
    blend with the neighbour so the published value lands
    between two distinct observations.

    Use a step-shaped distribution so the midpoint is
    obviously different from either endpoint even after 2-sigfig
    rounding. Under the old (buggy) code the q25 would equal
    sorted[8] exactly."""
    pd = pytest.importorskip("pandas")
    from nora.data_request import _quartiles

    # n=33: (33-1)*0.25 = 8 (integer). sorted[8]=10, sorted[9]=100.
    # Old buggy code: q25 = 10. Fixed code: q25 = (10+100)/2 = 55.
    # Even after 2-sigfig rounding the answers are distinguishable.
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]  # sorted 0..7
    values += [10.0, 100.0]                            # sorted[8]=10, sorted[9]=100
    values += [float(v) for v in range(101, 124)]      # sorted[10..32]
    assert len(values) == 33
    from nora.sanitizer import DEFAULT_CONFIG
    res = _quartiles(pd.Series(values), len(values), DEFAULT_CONFIG)
    assert res.status == "granted"
    q25 = res.answer["percentile_25"]
    # Reject the old buggy value AND any other exact-sorted match
    # at the integer position.
    assert q25 != 10.0, (
        f"q25={q25} equals sorted[8]=10; the integer-position case "
        f"should have produced a midpoint blend"
    )
    # Midpoint of 10 and 100 is 55. Rounded to 2 sigfigs is 55.
    assert q25 == 55.0


def test_quartiles_n_37_blends_at_integer_position() -> None:
    """n=37 is another integer-position case: (37-1)*0.25 = 9 is
    integer. Confirms the fix works for multiple n values, not
    just n=33."""
    pd = pytest.importorskip("pandas")
    from nora.data_request import _quartiles

    # n=37: (37-1)*0.25 = 9 (integer). sorted[9]=10, sorted[10]=100.
    # Old buggy code: q25=10. Fixed code: midpoint=55.
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]   # 0..8
    values += [10.0, 100.0]                                   # 9, 10
    values += [float(v) for v in range(101, 127)]             # 11..36
    assert len(values) == 37
    from nora.sanitizer import DEFAULT_CONFIG
    res = _quartiles(pd.Series(values), len(values), DEFAULT_CONFIG)
    assert res.status == "granted"
    assert res.answer["percentile_25"] != 10.0
    assert res.answer["percentile_25"] == 55.0


def test_quartiles_interpolated_path_unchanged() -> None:
    """For non-integer positions (eg n=34) the helper interpolates
    between the bracketing sorted observations. Confirms the fix
    doesn't regress the common-case behaviour — non-integer
    positions don't enter the midpoint-blend branch."""
    pd = pytest.importorskip("pandas")
    from nora.data_request import _quartiles

    # n=34: (34-1) * 0.25 = 8.25 (non-integer). Position is 8.25
    # between sorted[8]=9 and sorted[9]=10. Linear: 0.75*9 +
    # 0.25*10 = 9.25. Rounded to 2 sigfigs uses banker's rounding,
    # so 9.25 -> 9.2 (round half to even).
    values = [float(i) for i in range(1, 35)]
    from nora.sanitizer import DEFAULT_CONFIG
    res = _quartiles(pd.Series(values), len(values), DEFAULT_CONFIG)
    assert res.status == "granted"
    # Accept either 9.2 or 9.3 — the rounding convention matters
    # less than the fact that we land between sorted[8]=9 and
    # sorted[9]=10 rather than on either of them.
    q25 = res.answer["percentile_25"]
    assert 9.2 <= q25 <= 9.3


# ---------------------------------------------------------------------------
# 3. vcov invariant validation
# ---------------------------------------------------------------------------


def _ols_base(**overrides):
    base = {
        "type": "linear_regression",
        "n": 100,
        "response_variable": "y",
        "predictor_variables": ["x1", "x2"],
        "coefficients": {"x1": 1.0, "x2": 2.0},
        "standard_errors": {"x1": 0.1, "x2": 0.2},
    }
    base.update(overrides)
    return base


def test_vcov_symmetric_with_matching_diagonals_passes() -> None:
    """A real cov matrix is symmetric and its diagonals are SE².
    A payload that satisfies both invariants must still pass and
    surface vcov in the sanitized output."""
    payload = _ols_base(vcov={
        "x1": {"x1": 0.01, "x2": 0.005},   # diag = 0.1² = 0.01
        "x2": {"x1": 0.005, "x2": 0.04},   # diag = 0.2² = 0.04
    })
    r = sanitize(payload)
    assert r.ok, f"rejected unexpectedly: {r.rejection_reason}"
    assert "vcov" in r.sanitized


def test_vcov_asymmetric_off_diagonal_is_dropped() -> None:
    """A payload that violates symmetry — eg ``vcov[x1][x2]=0.5``
    but ``vcov[x2][x1]=0.001`` — has the entire vcov field
    dropped. The other regression fields (coefficients, SEs,
    etc.) still flow through."""
    payload = _ols_base(vcov={
        "x1": {"x1": 0.01, "x2": 0.5},     # asymmetric
        "x2": {"x1": 0.001, "x2": 0.04},
    })
    r = sanitize(payload)
    assert r.ok
    assert "vcov" not in r.sanitized
    assert "coefficients" in r.sanitized
    # The transformation log should record the drop.
    assert any("vcov" in t for t in r.transformations)


def test_vcov_diagonal_mismatch_with_se_is_dropped() -> None:
    """Diagonals must match SE². A diagonal of 9999.0 when the
    declared SE is 0.1 (so SE² = 0.01) is dropped."""
    payload = _ols_base(vcov={
        "x1": {"x1": 9999.0, "x2": 0.0},
        "x2": {"x1": 0.0, "x2": 0.04},
    })
    r = sanitize(payload)
    assert r.ok
    assert "vcov" not in r.sanitized


def test_vcov_negative_variance_is_dropped() -> None:
    """A negative diagonal entry isn't a variance — real
    covariance matrices are positive semi-definite. Drop the
    whole field rather than letting a negative cell through."""
    payload = _ols_base(vcov={
        "x1": {"x1": -0.01, "x2": 0.0},
        "x2": {"x1": 0.0, "x2": 0.04},
    })
    r = sanitize(payload)
    assert r.ok
    assert "vcov" not in r.sanitized


# ---------------------------------------------------------------------------
# 4. correlation_matrix invariant validation
# ---------------------------------------------------------------------------


def _corr_base(**overrides):
    base = {
        "type": "correlation_matrix",
        "n": 200,
        "missing_count": 0,
        "variables": ["x", "y", "z"],
        "method": "pearson",
        "correlations": {
            "x": {"x": 1.0, "y": 0.5, "z": 0.3},
            "y": {"x": 0.5, "y": 1.0, "z": 0.1},
            "z": {"x": 0.3, "y": 0.1, "z": 1.0},
        },
    }
    base.update(overrides)
    return base


def test_correlation_matrix_real_shape_passes() -> None:
    """A symmetric, complete, 1-on-diagonal matrix passes the
    new invariant check."""
    r = sanitize(_corr_base())
    assert r.ok, f"rejected: {r.rejection_reason}"
    assert "correlations" in r.sanitized
    # Diagonal preserved.
    assert r.sanitized["correlations"]["x"]["x"] == 1.0


def test_correlation_matrix_asymmetric_is_rejected() -> None:
    """Off-diagonal asymmetry: ``corr[x][y]=0.5`` but
    ``corr[y][x]=0.9`` violates the invariant. Rejected
    outright."""
    payload = _corr_base(correlations={
        "x": {"x": 1.0, "y": 0.5, "z": 0.3},
        "y": {"x": 0.9, "y": 1.0, "z": 0.1},   # asymmetric on x
        "z": {"x": 0.3, "y": 0.1, "z": 1.0},
    })
    r = sanitize(payload)
    assert not r.ok
    assert "asymmetric" in (r.rejection_reason or "").lower()


def test_correlation_matrix_non_unit_diagonal_is_rejected() -> None:
    """Diagonals must equal 1.0. ``corr[x][x]=0.42`` is a hand-
    crafted matrix, not a real ``df.corr()`` output."""
    payload = _corr_base(correlations={
        "x": {"x": 0.42, "y": 0.5, "z": 0.3},   # non-unit diagonal
        "y": {"x": 0.5, "y": 1.0, "z": 0.1},
        "z": {"x": 0.3, "y": 0.1, "z": 1.0},
    })
    r = sanitize(payload)
    assert not r.ok
    assert "diagonal" in (r.rejection_reason or "").lower()


def test_correlation_matrix_partial_row_is_rejected() -> None:
    """Every declared variable must have its row cover every
    other declared variable. A row that's missing a column
    (so the attacker is sneaking free cells past the variable
    count cap) is rejected."""
    payload = _corr_base(correlations={
        # 'z' row is missing the 'y' column.
        "x": {"x": 1.0, "y": 0.5, "z": 0.3},
        "y": {"x": 0.5, "y": 1.0, "z": 0.1},
        "z": {"x": 0.3, "z": 1.0},
    })
    r = sanitize(payload)
    assert not r.ok
    assert "missing column" in (r.rejection_reason or "").lower()


def test_correlation_matrix_missing_variable_row_is_rejected() -> None:
    """Every declared variable must have a row. Skipping a row
    is the row-axis variant of the cell-completeness check."""
    payload = _corr_base(correlations={
        "x": {"x": 1.0, "y": 0.5, "z": 0.3},
        "y": {"x": 0.5, "y": 1.0, "z": 0.1},
        # 'z' row missing entirely.
    })
    r = sanitize(payload)
    assert not r.ok
    assert "missing row" in (r.rejection_reason or "").lower()
