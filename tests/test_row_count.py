"""Tests for row-count change detection.

Two layers:
1. Unit tests on ``_effective_n`` — verify per-type extraction of
   "rows used in the analysis" across all five analysis types.
2. Integration tests on ``_check_row_count`` — given a dataset and a
   sanitized payload, does the check correctly flag / not flag?

Live integration through ``submit_script`` is exercised in the
end-to-end live test, not here (it needs Rscript installed).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from nora.config import set_cwd
from nora.tools import (
    _check_row_count,
    _effective_n,
    _resolve_source_row_count,
)


# ---------------------------------------------------------------------------
# _effective_n — per-type extraction
# ---------------------------------------------------------------------------

def test_effective_n_linear_regression():
    assert _effective_n({"type": "linear_regression", "n": 500}) == 500
    assert _effective_n({"type": "linear_regression"}) is None


def test_effective_n_t_test_two_sample():
    assert _effective_n({"type": "t_test", "n1": 100, "n2": 80}) == 180


def test_effective_n_t_test_one_sample():
    # No n2 → treat as one-sample; return n1.
    assert _effective_n({"type": "t_test", "n1": 50}) == 50


def test_effective_n_descriptive():
    # Descriptive's n is non-missing; total is n + missing_count.
    assert _effective_n({"type": "descriptive", "n": 90, "missing_count": 10}) == 100


def test_effective_n_frequency_table():
    assert _effective_n({"type": "frequency_table", "n": 300}) == 300


def test_effective_n_crosstab_fully_visible():
    payload = {
        "type": "crosstab",
        "counts": {
            "young": {"M": 50, "F": 40},
            "old":   {"M": 30, "F": 20},
        },
        "missing_count": 10,
    }
    assert _effective_n(payload) == 150


def test_effective_n_crosstab_with_suppression_returns_none():
    """Can't confidently compute N if any cell is suppressed (string)."""
    payload = {
        "type": "crosstab",
        "counts": {
            "young": {"M": 50, "F": 40},
            "old":   {"M": 30, "F": "<10"},
        },
        "missing_count": 0,
    }
    assert _effective_n(payload) is None


def test_effective_n_magnitude_table_fully_visible():
    payload = {
        "type": "magnitude_table",
        "cells": {
            "A": {"value": 1000, "n": 50},
            "B": {"value": 500, "n": 30},
        },
    }
    assert _effective_n(payload) == 80


def test_effective_n_magnitude_table_with_suppression_returns_none():
    payload = {
        "type": "magnitude_table",
        "cells": {
            "A": {"value": 1000, "n": 50},
            "B": {"value": "<10", "n": "<10"},
        },
    }
    assert _effective_n(payload) is None


def test_effective_n_unknown_type():
    assert _effective_n({"type": "covariance_matrix"}) is None


# ---------------------------------------------------------------------------
# _check_row_count — integration with real CSV
# ---------------------------------------------------------------------------

@pytest.fixture
def dataset_100(tmp_path: Path) -> Path:
    set_cwd(tmp_path)
    df = pd.DataFrame({"x": range(100), "y": range(100, 200)})
    p = tmp_path / "data.csv"
    df.to_csv(p, index=False)
    return p


def test_check_returns_none_when_no_source_given(dataset_100):
    # No source_dataset → no check, returns None.
    msg = _check_row_count({"type": "linear_regression", "n": 50}, None, 100)
    assert msg is None
    msg = _check_row_count({"type": "linear_regression", "n": 50}, "", 100)
    assert msg is None


def test_check_returns_none_when_source_n_unknown(dataset_100):
    """If the resolver couldn't load a row count (unsupported format,
    transient error), the per-payload check passes through without
    flagging — same posture as before, just expressed through a
    None ``source_n`` instead of a silent load failure inside the
    check."""
    msg = _check_row_count(
        {"type": "linear_regression", "n": 50}, "data.csv", None,
    )
    assert msg is None


def test_check_returns_none_when_n_matches_source(dataset_100):
    msg = _check_row_count(
        {"type": "linear_regression", "n": 100}, "data.csv", 100,
    )
    assert msg is None


def test_check_flags_shortfall(dataset_100):
    msg = _check_row_count(
        {"type": "linear_regression", "n": 80}, "data.csv", 100,
    )
    assert msg is not None
    assert "ROW COUNT CHANGE" in msg
    assert "n=80" in msg
    assert "100" in msg
    assert "20" in msg  # the difference
    assert "20.0%" in msg  # percentage


def test_check_flags_anomaly_when_analysis_n_too_big(dataset_100):
    msg = _check_row_count(
        {"type": "linear_regression", "n": 150}, "data.csv", 100,
    )
    assert msg is not None
    assert "ROW COUNT ANOMALY" in msg


def test_resolve_silent_on_nonexistent_source(dataset_100):
    """Path resolution lives in ``_resolve_source_row_count`` now;
    bad paths return None instead of raising. submit_script then
    skips the per-payload check."""
    assert _resolve_source_row_count("nonexistent.csv") is None


def test_resolve_silent_on_path_escape(dataset_100):
    assert _resolve_source_row_count("../../etc/passwd") is None


def test_resolve_returns_n_for_real_dataset(dataset_100):
    n = _resolve_source_row_count("data.csv")
    assert n == 100


def test_check_silent_when_effective_n_unknown(dataset_100):
    """Types we can't extract N from shouldn't false-flag."""
    # unknown type
    msg = _check_row_count({"type": "weird"}, "data.csv", 100)
    assert msg is None
    # suppressed crosstab
    msg = _check_row_count(
        {"type": "crosstab", "counts": {"a": {"x": "<10"}}},
        "data.csv", 100,
    )
    assert msg is None


def test_check_applies_to_t_test(dataset_100):
    # n1 + n2 = 40, source N = 100 → flag
    msg = _check_row_count(
        {"type": "t_test", "n1": 25, "n2": 15, "test_type": "two_sample"},
        "data.csv", 100,
    )
    assert msg is not None
    assert "n=40" in msg


def test_check_applies_to_magnitude_table(dataset_100):
    payload = {
        "type": "magnitude_table",
        "cells": {
            "A": {"value": 1.0, "n": 20},
            "B": {"value": 2.0, "n": 30},
        },
    }
    msg = _check_row_count(payload, "data.csv", 100)
    assert msg is not None
    assert "n=50" in msg
