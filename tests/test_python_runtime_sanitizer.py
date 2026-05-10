"""End-to-end Python runtime tests through the sanitizer.

The previous Python smoke tests exercised the executor + sandbox up
to the point of token validation but didn't run the emitted payload
through ``nora.sanitizer.sanitize`` — exactly the production step
that decides whether a result reaches the model. That gap let the
runtime helpers ship with field-name drift against the sanitizer
contract (e.g. ``subtype`` vs ``test_type``, ``group_variable`` vs
``row_variable``).

Each test here:
  1. Sets ``NORA_RESULT_PATH`` to a tmp file and ``NORA_RUN_TOKEN``
     to a known value, then imports ``nora.runtime.nora`` (the
     runtime library reads both env vars at import time).
  2. Calls one of the ``nora.from_*`` / ``nora.result`` helpers
     with realistic inputs.
  3. Reads the JSON payload back from ``NORA_RESULT_PATH``, strips
     the per-run authenticity token (the executor does this in
     production), and runs ``sanitize()`` on the result.
  4. Asserts ``ok=True`` and that the emitted payload type matches.

A failure here means a researcher who follows the advertised helper
API will see ``rejected_by_sanitizer`` at runtime — which is the
worst possible UX because the script ran fine and produced
something, but Nora silently throws it away.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nora.sanitizer import sanitize


_TEST_TOKEN = "deadbeef" * 8  # 64 hex chars, matches secrets.token_hex(32)


@pytest.fixture
def runtime(tmp_path: Path):
    """Provide a fresh ``nora.runtime.nora`` module with its env vars
    pointing at a tmp result file. The module reads ``NORA_RUN_TOKEN``
    and ``NORA_RESULT_PATH`` at import time and ``pop()``s the token,
    so we reload it under monkeypatched env each time.

    Yields ``(module, result_path)`` so the test can call helpers and
    read what they wrote.
    """
    result_path = tmp_path / "result.json"
    prev_token = os.environ.get("NORA_RUN_TOKEN")
    prev_path = os.environ.get("NORA_RESULT_PATH")
    os.environ["NORA_RUN_TOKEN"] = _TEST_TOKEN
    os.environ["NORA_RESULT_PATH"] = str(result_path)
    # Drop any stale module so the import re-reads env.
    sys.modules.pop("nora.runtime.nora", None)
    try:
        mod = importlib.import_module("nora.runtime.nora")
        yield mod, result_path
    finally:
        sys.modules.pop("nora.runtime.nora", None)
        # Restore the previous env so other tests aren't poisoned.
        if prev_token is None:
            os.environ.pop("NORA_RUN_TOKEN", None)
        else:
            os.environ["NORA_RUN_TOKEN"] = prev_token
        if prev_path is None:
            os.environ.pop("NORA_RESULT_PATH", None)
        else:
            os.environ["NORA_RESULT_PATH"] = prev_path


def _read_payload_strip_token(result_path: Path) -> dict:
    """Read the JSON the runtime wrote and strip the authenticity
    token the way ``executor._validate_and_strip_token`` does in
    production."""
    raw = json.loads(result_path.read_text(encoding="utf-8"))
    assert raw.get("_token") == _TEST_TOKEN, (
        "runtime payload must carry the per-run token"
    )
    return {k: v for k, v in raw.items() if k != "_token"}


# ---------------------------------------------------------------------------
# from_summarize
# ---------------------------------------------------------------------------

def test_from_summarize_through_sanitizer(runtime) -> None:
    mod, path = runtime
    mod.from_summarize("salary", n=523, mean=85000.0, sd=12000.0,
                       missing_count=4)
    payload = _read_payload_strip_token(path)
    res = sanitize(payload)
    assert res.ok, f"sanitizer rejected: {res.rejection_reason}"
    assert res.analysis_type == "descriptive"
    assert res.sanitized["variable"] == "salary"
    assert res.sanitized["n"] == 523


# ---------------------------------------------------------------------------
# from_table (frequency_table)
# ---------------------------------------------------------------------------

def test_from_table_through_sanitizer(runtime) -> None:
    mod, path = runtime
    counts = {"A": 312, "B": 189, "C": 105, "D": 73}
    mod.from_table("treatment", counts, missing_count=2)
    payload = _read_payload_strip_token(path)
    res = sanitize(payload)
    assert res.ok, f"sanitizer rejected: {res.rejection_reason}"
    assert res.analysis_type == "frequency_table"


# ---------------------------------------------------------------------------
# from_t_test
# ---------------------------------------------------------------------------

class _FakeTTestResult:
    """Stand-in for ``scipy.stats._stats_py.TtestResult`` — no scipy
    dep needed for this test, just the attributes ``from_t_test``
    reads."""
    statistic = 2.31
    pvalue = 0.022
    df = 198.5


def test_from_t_test_through_sanitizer(runtime) -> None:
    mod, path = runtime
    mod.from_t_test(
        _FakeTTestResult(),
        n1=100, n2=100,
        mean1=4.2, mean2=3.8,
    )
    payload = _read_payload_strip_token(path)
    res = sanitize(payload)
    assert res.ok, (
        f"sanitizer rejected from_t_test: {res.rejection_reason}. "
        "Likely cause: helper emits a field name the sanitizer "
        "doesn't expect (e.g. 'subtype' vs 'test_type')."
    )
    assert res.analysis_type == "t_test"


# ---------------------------------------------------------------------------
# from_lm
# ---------------------------------------------------------------------------

class _FakeFitInner:
    """The ``model.model`` attribute statsmodels exposes — carries
    endog/exog names. ``from_lm`` reads from here for the
    response/predictor variable names."""
    endog_names = "outcome"
    exog_names = ("Intercept", "treatment", "age")


class _FakeFit:
    """Stand-in for a fitted statsmodels result. Only the attributes
    ``from_lm`` reads are populated; everything else would TypeError
    in real usage but never gets touched here."""
    model = _FakeFitInner()
    nobs = 200.0
    df_resid = 197
    rsquared = 0.34
    rsquared_adj = 0.33
    fvalue = 51.0
    f_pvalue = 1e-18
    scale = 1.21  # squared residual SE; from_lm takes sqrt

    def __init__(self) -> None:
        idx = ["Intercept", "treatment", "age"]
        self.params = pd.Series([5.0, 2.5, -0.1], index=idx)
        self.bse = pd.Series([0.4, 0.3, 0.02], index=idx)
        self.tvalues = pd.Series([12.5, 8.3, -5.0], index=idx)
        self.pvalues = pd.Series([1e-30, 1e-15, 1e-7], index=idx)

    def summary(self) -> str:
        return "(fake summary)"


def test_from_lm_through_sanitizer(runtime) -> None:
    mod, path = runtime
    mod.from_lm(_FakeFit())
    payload = _read_payload_strip_token(path)
    res = sanitize(payload)
    assert res.ok, (
        f"sanitizer rejected from_lm: {res.rejection_reason}"
    )
    assert res.analysis_type == "linear_regression"
    assert "treatment" in res.sanitized.get("predictor_variables", [])


def test_from_lm_emits_vif_and_condition_number_when_design_available(runtime) -> None:
    """When the fitted result exposes ``model.exog`` (the design
    matrix), the helper computes VIF and condition number natively
    and the sanitizer accepts them.

    ``_FakeFit`` historically didn't carry ``exog`` so the diagnostic
    fields were silently omitted. This test extends the fake to a
    real numpy design and pins that VIF lands keyed on declared
    predictors only, condition_number is a finite scalar."""
    rng = np.random.default_rng(7)
    n = 200
    treatment = rng.integers(0, 2, size=n).astype(float)
    age = rng.normal(40, 12, size=n)
    intercept = np.ones(n)
    exog = np.column_stack([intercept, treatment, age])

    class _ModelWithDesign:
        endog_names = "outcome"
        exog_names = ("Intercept", "treatment", "age")

    _ModelWithDesign.exog = exog  # set at outer scope so the closure resolves

    class _FitWithDesign(_FakeFit):
        model = _ModelWithDesign()

    mod, path = runtime
    mod.from_lm(_FitWithDesign())
    payload = _read_payload_strip_token(path)
    res = sanitize(payload)
    assert res.ok, (
        f"sanitizer rejected diagnostic-extended from_lm: "
        f"{res.rejection_reason}"
    )
    sanitized = res.sanitized
    # VIF must be present, keyed on declared predictors (intercept
    # excluded by construction in the helper).
    assert "vif" in sanitized
    assert sorted(sanitized["vif"].keys()) == ["age", "treatment"]
    for v in sanitized["vif"].values():
        assert isinstance(v, (int, float)) and v >= 1.0
    # Condition number is a finite positive scalar.
    assert "condition_number" in sanitized
    cond = sanitized["condition_number"]
    assert isinstance(cond, (int, float)) and cond > 0


def test_from_lm_emits_vcov_when_cov_params_available(runtime) -> None:
    """When the fit exposes ``cov_params()``, the helper emits the
    full variance-covariance matrix (dict-of-dict keyed on
    coefficient names). Diagonals must equal SE^2 to within
    precision-clamp; off-diagonals carry the covariances Wald
    tests need."""
    rng = np.random.default_rng(11)
    n = 200
    treatment = rng.integers(0, 2, size=n).astype(float)
    age = rng.normal(40, 12, size=n)
    intercept = np.ones(n)
    exog = np.column_stack([intercept, treatment, age])

    cov_df = pd.DataFrame(
        [[0.50, 0.02, 0.01], [0.02, 0.30, 0.005], [0.01, 0.005, 0.10]],
        index=["Intercept", "treatment", "age"],
        columns=["Intercept", "treatment", "age"],
    )

    class _ModelWithDesign:
        endog_names = "outcome"
        exog_names = ("Intercept", "treatment", "age")

    _ModelWithDesign.exog = exog

    class _FitWithVcov(_FakeFit):
        model = _ModelWithDesign()

        def cov_params(self) -> pd.DataFrame:
            return cov_df

    mod, path = runtime
    mod.from_lm(_FitWithVcov())
    payload = _read_payload_strip_token(path)
    res = sanitize(payload)
    assert res.ok, (
        f"sanitizer rejected vcov-bearing payload: {res.rejection_reason}"
    )
    sanitized = res.sanitized
    assert "vcov" in sanitized
    # All declared predictors plus the intercept survive.
    assert sorted(sanitized["vcov"].keys()) == [
        "Intercept", "age", "treatment",
    ]
    # Diagonal exists and matches the input within sigfig clamp.
    treatment_var = sanitized["vcov"]["treatment"]["treatment"]
    assert 0.29 <= treatment_var <= 0.31
    # Off-diagonal symmetry preserved.
    cov_age_treat = sanitized["vcov"]["age"]["treatment"]
    cov_treat_age = sanitized["vcov"]["treatment"]["age"]
    assert abs(cov_age_treat - cov_treat_age) < 1e-9


def test_from_lm_omits_vcov_when_cov_params_missing(runtime) -> None:
    """A fit object that doesn't expose ``cov_params`` (sklearn-
    shaped, custom estimator) shouldn't emit vcov. The helper omits
    the field rather than crashing or writing null — caller drops
    the diagnostic gracefully."""
    mod, path = runtime
    mod.from_lm(_FakeFit())  # no cov_params attribute
    payload = _read_payload_strip_token(path)
    res = sanitize(payload)
    assert res.ok
    assert "vcov" not in res.sanitized


def test_from_lm_omits_diagnostics_when_design_missing(runtime) -> None:
    """When the result doesn't expose a design matrix (the legacy
    ``_FakeFit`` shape), the helper omits VIF / condition_number
    rather than emitting null or crashing the run. This pins the
    backward-compat path for sklearn-shaped results that go through
    the helper before the user switches to ``nora.result(...)``
    directly."""
    mod, path = runtime
    mod.from_lm(_FakeFit())  # no exog
    payload = _read_payload_strip_token(path)
    res = sanitize(payload)
    assert res.ok
    assert "vif" not in res.sanitized
    assert "condition_number" not in res.sanitized


# ---------------------------------------------------------------------------
# from_crosstab
# ---------------------------------------------------------------------------

def test_from_crosstab_through_sanitizer(runtime) -> None:
    mod, path = runtime
    df = pd.DataFrame({
        "group": ["A"] * 200 + ["B"] * 200,
        "outcome": ["yes"] * 130 + ["no"] * 70 + ["yes"] * 90 + ["no"] * 110,
    })
    table = pd.crosstab(df["group"], df["outcome"])
    mod.from_crosstab(table)
    payload = _read_payload_strip_token(path)
    res = sanitize(payload)
    assert res.ok, (
        f"sanitizer rejected from_crosstab: {res.rejection_reason}"
    )
    assert res.analysis_type == "crosstab"


# ---------------------------------------------------------------------------
# from_magnitude_table
# ---------------------------------------------------------------------------

def test_from_magnitude_table_through_sanitizer(runtime) -> None:
    """Pass a balanced sum-by-group table through. The dominance metric
    must be a share in [0, 1] (max(abs(vals)) / sum(abs(vals))) per the
    sanitizer contract — anything else gets rejected with 'max_share
    must be a finite number'-style errors or silently treated as a
    dominance failure."""
    mod, path = runtime
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "region": (["west"] * 80) + (["east"] * 80) + (["south"] * 80),
        "revenue": np.concatenate([
            rng.uniform(100, 200, 80),
            rng.uniform(120, 180, 80),
            rng.uniform(90,  220, 80),
        ]),
    })
    mod.from_magnitude_table(df, "region", "revenue", aggregation="sum")
    payload = _read_payload_strip_token(path)
    res = sanitize(payload)
    assert res.ok, (
        f"sanitizer rejected from_magnitude_table: "
        f"{res.rejection_reason}. Likely cause: emitted "
        f"'group_variable' instead of 'row_variable', or per-cell "
        f"dominance field is not in [0, 1]."
    )
    assert res.analysis_type == "magnitude_table"


def test_generic_result_strips_helper_provenance_marker(runtime) -> None:
    """``nora.result()`` is the generic emit. A script that tries to
    publish a magnitude_table through this path with a forged
    ``_via_helper`` marker — bypassing the typed helper that
    computes max_share from data — must NOT slip past the sanitizer.
    The runtime strips the marker from caller-passed fields here;
    the sanitizer rejects payloads without it. Together they block
    the trivial ``nora.result(type="magnitude_table",
    _via_helper="from_magnitude_table", cells={..., max_share: 0})``
    bypass.
    """
    mod, path = runtime
    # Forge the marker through the generic API.
    mod.result(
        type="magnitude_table",
        row_variable="industry",
        value_variable="revenue",
        aggregation="sum",
        # Looks innocent: max_share=0. Reality (in the attacker
        # scenario) would be one company contributing 99%.
        cells={"tech": {"value": 1e9, "n": 50, "max_share": 0.0}},
        _via_helper="from_magnitude_table",
    )
    payload = _read_payload_strip_token(path)
    # The generic ``result()`` stripped the forged marker before
    # writing — so the on-disk JSON has no ``_via_helper`` at all.
    assert "_via_helper" not in payload, (
        "nora.result() must strip _via_helper from caller-passed fields"
    )
    # Without the marker the sanitizer rejects.
    res = sanitize(payload)
    assert not res.ok
    assert "typed runtime helper" in (res.rejection_reason or "")


def test_typed_helper_marker_survives_to_sanitizer(runtime) -> None:
    """Companion to the strip test: the typed helper bypasses
    ``result()`` and writes through ``_write_result`` directly so
    the marker reaches the sanitizer. Without this round-trip the
    sanitizer would reject every legitimate magnitude_table."""
    mod, path = runtime
    df = pd.DataFrame({
        "g": ["a"] * 50 + ["b"] * 50,
        "v": list(range(50)) + list(range(100, 150)),
    })
    mod.from_magnitude_table(df, "g", "v", aggregation="sum")
    payload = _read_payload_strip_token(path)
    assert payload.get("_via_helper") == "from_magnitude_table"


# ---------------------------------------------------------------------------
# from_correlation
# ---------------------------------------------------------------------------

def test_from_correlation_through_sanitizer(runtime) -> None:
    """A pandas DataFrame correlated via the helper must produce a
    correlation_matrix payload the sanitizer accepts. Validates the
    shape contract end-to-end: top-level fields (n / variables /
    method / correlations / missing_count), inner dict keys are
    declared variable names, all values are in [-1, 1]."""
    mod, path = runtime
    rng = np.random.default_rng(42)
    n = 200
    age = rng.normal(40, 12, n)
    # Build income correlated with age plus noise, education weakly
    # correlated with both.
    income = age * 1500 + rng.normal(0, 5000, n)
    edu = age * 0.05 + rng.normal(0, 1.5, n)
    df = pd.DataFrame({"age": age, "income": income, "education": edu})
    mod.from_correlation(df, method="pearson")
    payload = _read_payload_strip_token(path)
    res = sanitize(payload)
    assert res.ok, (
        f"sanitizer rejected from_correlation: {res.rejection_reason}"
    )
    assert res.analysis_type == "correlation_matrix"
    assert res.sanitized["n"] == n
    assert sorted(res.sanitized["variables"]) == ["age", "education", "income"]
    assert res.sanitized["method"] == "pearson"
    # Diagonal is exactly 1.0 (after clamp + clip).
    for v in res.sanitized["variables"]:
        assert res.sanitized["correlations"][v][v] == 1.0
    # All off-diagonal values within [-1, 1].
    for row in res.sanitized["correlations"].values():
        for val in row.values():
            assert -1.0 <= val <= 1.0


def test_from_correlation_drops_complete_case_rows(runtime) -> None:
    """Helper computes correlation on rows complete over the chosen
    variables. ``n`` must reflect the COMPLETE sample size, not the
    raw row count — pairwise N would make off-diagonals draw from
    different samples and joint inference dishonest. Use a missing
    count above the disclosure-threshold (10) so the orthogonal
    rare-missingness coarsening doesn't shadow the assertion below
    — that gate has its own test in ``test_sanitizer``."""
    mod, path = runtime
    # 13 complete + 12 incomplete = 25 raw rows, missing_count=12.
    df = pd.DataFrame({
        "age": [20, 25, 30, 35, 40, 45, 50, 55, 60, 65,
                70, 75, 80,
                np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        "income": [30, 35, 40, 50, 55, 60, 70, 80, 85, 90,
                   95, 100, 110,
                   50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105],
    })
    mod.from_correlation(df)
    payload = _read_payload_strip_token(path)
    res = sanitize(payload)
    assert res.ok
    assert res.sanitized["n"] == 13
    assert res.sanitized["missing_count"] == 12


def test_from_correlation_invalid_method_raises(runtime) -> None:
    """Unsupported method names raise inside the helper rather than
    emitting a payload the sanitizer would reject. The helper is the
    point where method names should be validated; surfacing here is
    closer to the bug than the sanitizer rejection."""
    mod, _ = runtime
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [1.0, 2.0, 4.0]})
    with pytest.raises(ValueError, match="method must be"):
        mod.from_correlation(df, method="bogus")


# ---------------------------------------------------------------------------
# Generic result() escape hatch
# ---------------------------------------------------------------------------

def test_generic_result_round_trip(runtime) -> None:
    mod, path = runtime
    mod.result(
        type="descriptive",
        variable="x",
        n=50,
        mean=1.0,
        sd=0.1,
        missing_count=0,
    )
    payload = _read_payload_strip_token(path)
    res = sanitize(payload)
    assert res.ok
    assert res.analysis_type == "descriptive"
