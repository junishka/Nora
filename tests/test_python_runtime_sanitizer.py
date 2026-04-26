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
