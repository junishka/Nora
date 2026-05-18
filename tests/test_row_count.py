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


def test_effective_n_correlation_matrix():
    # Same shape as ``descriptive``: n is complete-cases count,
    # missing_count is rows dropped by listwise NA-removal; the sum
    # is what the DataFrame had at correlation time.
    assert _effective_n({
        "type": "correlation_matrix", "n": 90, "missing_count": 10,
    }) == 100
    # Missing ``missing_count`` blocks the audit (no false positives
    # from a partial payload).
    assert _effective_n({"type": "correlation_matrix", "n": 90}) is None


def test_effective_n_marginal_effects():
    assert _effective_n({"type": "marginal_effects", "n": 500}) == 500
    assert _effective_n({"type": "marginal_effects"}) is None


def test_effective_n_kaplan_meier():
    assert _effective_n({"type": "kaplan_meier", "n_subjects": 300}) == 300
    assert _effective_n({"type": "kaplan_meier"}) is None


def test_effective_n_factor_decomposition():
    assert _effective_n({
        "type": "factor_decomposition", "n_observations": 200,
    }) == 200
    assert _effective_n({"type": "factor_decomposition"}) is None


def test_effective_n_cluster_analysis():
    assert _effective_n({
        "type": "cluster_analysis", "n_observations": 150,
    }) == 150
    assert _effective_n({"type": "cluster_analysis"}) is None


def test_effective_n_rdd_deliberately_skipped():
    """RDD's ``effective_n_*`` fields are bandwidth-restricted by
    construction — narrowing IS the analysis, not silent filtering.
    The audit deliberately does not extract an N from these payloads;
    if a future revision changes that, this test should be updated
    alongside the new branch in ``_effective_n``."""
    assert _effective_n({
        "type": "rdd", "effective_n_left": 50, "effective_n_right": 50,
        "effective_n_total": 100,
    }) is None


def test_effective_n_did_event_study_deliberately_skipped():
    """DiD event study's only N-like field is ``n_treated_per_group``
    (per-cohort treated-unit counts) — units, not rows, and not even
    all units (untreated controls aren't counted). Comparing to
    source dataset rows is a units-vs-rows mismatch, so the audit
    deliberately skips it."""
    assert _effective_n({
        "type": "did_event_study",
        "n_treated_per_group": {"2010": 40, "2012": 30},
    }) is None


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


# ---------------------------------------------------------------------------
# Sanitizer → row-count audit integration. The hand-built payloads
# above don't exercise the real handoff: the helpers emit
# ``coefficient_table_with_fit_stats`` (the canonical name) but
# earlier ``_effective_n`` only matched the legacy ``linear_regression``
# alias, so the audit silently skipped every regression result that
# went through the sanitizer. This block runs payloads through
# ``sanitize()`` and confirms the audit fires on the sanitized
# output regardless of which alias the raw payload carries.
# ---------------------------------------------------------------------------

def _raw_regression_payload(type_name: str, n: int) -> dict:
    """Minimal raw regression payload that passes ``_sanitize_linear_regression``.

    Covers the structural required set (``_OLS_REQUIRED``) and nothing
    more; intercept-only fit so no coefficient-name cross-field
    validation comes into play.
    """
    return {
        "type": type_name,
        "n": n,
        "response_variable": "y",
        "predictor_variables": [],
        "coefficients": {"(Intercept)": 1.0},
        "standard_errors": {"(Intercept)": 0.1},
    }


def test_sanitizer_to_audit_canonical_type_fires(dataset_100):
    """Real-pipeline regression: helper emits the canonical name,
    sanitizer keeps it, audit must see n=80 < source N=100 and flag.
    Previously this silently passed because ``_effective_n`` only
    matched the legacy alias."""
    from nora.sanitizer import sanitize

    raw = _raw_regression_payload("coefficient_table_with_fit_stats", 80)
    sanitized = sanitize(raw)
    assert sanitized.ok, sanitized.rejection_reason
    msg = _check_row_count(sanitized.sanitized or {}, "data.csv", 100)
    assert msg is not None
    assert "ROW COUNT CHANGE" in msg
    assert "n=80" in msg


def test_sanitizer_to_audit_legacy_alias_fires(dataset_100):
    """Same path with the legacy alias — older stored payloads and
    scripts that hand-craft ``result(type='linear_regression', ...)``
    must keep working."""
    from nora.sanitizer import sanitize

    raw = _raw_regression_payload("linear_regression", 80)
    sanitized = sanitize(raw)
    assert sanitized.ok, sanitized.rejection_reason
    msg = _check_row_count(sanitized.sanitized or {}, "data.csv", 100)
    assert msg is not None
    assert "ROW COUNT CHANGE" in msg
    assert "n=80" in msg


# ---------------------------------------------------------------------------
# Sanitizer → audit handoff for the other 0.10 shapes. Same
# motivation as the regression-bucket test above: hand-built payloads
# don't exercise the real ``sanitize() → _check_row_count`` flow, so
# silent gaps in ``_effective_n``'s type dispatch slip past unit
# tests.
# ---------------------------------------------------------------------------

def test_sanitizer_to_audit_correlation_matrix_fires(dataset_100):
    from nora.sanitizer import sanitize

    raw = {
        "type": "correlation_matrix",
        "n": 70,
        "missing_count": 10,
        "variables": ["x", "y"],
        "correlations": {"x": {"x": 1.0, "y": 0.5}, "y": {"x": 0.5, "y": 1.0}},
    }
    sanitized = sanitize(raw)
    assert sanitized.ok, sanitized.rejection_reason
    msg = _check_row_count(sanitized.sanitized or {}, "data.csv", 100)
    assert msg is not None
    assert "ROW COUNT CHANGE" in msg
    # n + missing_count = 80, audit reports n=80 vs source 100
    assert "n=80" in msg


def test_sanitizer_to_audit_marginal_effects_fires(dataset_100):
    from nora.sanitizer import sanitize

    raw = {
        "type": "marginal_effects",
        "n": 80,
        "method": "ame",
        "variables": ["age"],
        "effects": {"age": 0.012},
    }
    sanitized = sanitize(raw)
    assert sanitized.ok, sanitized.rejection_reason
    msg = _check_row_count(sanitized.sanitized or {}, "data.csv", 100)
    assert msg is not None
    assert "n=80" in msg


def test_sanitizer_to_audit_kaplan_meier_fires(dataset_100):
    from nora.sanitizer import sanitize

    raw = {
        "type": "kaplan_meier",
        "time_variable": "t",
        "event_variable": "ev",
        "n_subjects": 80,
        "n_failures": 30,
    }
    sanitized = sanitize(raw)
    assert sanitized.ok, sanitized.rejection_reason
    msg = _check_row_count(sanitized.sanitized or {}, "data.csv", 100)
    assert msg is not None
    assert "n=80" in msg


def test_sanitizer_to_audit_factor_decomposition_fires(dataset_100):
    from nora.sanitizer import sanitize

    raw = {
        "type": "factor_decomposition",
        "method": "pca",
        "n_observations": 80,
        "n_variables": 2,
        "n_components": 1,
        "variables": ["x", "y"],
        "components": ["PC1"],
        "loadings": {"x": {"PC1": 0.7}, "y": {"PC1": 0.7}},
    }
    sanitized = sanitize(raw)
    assert sanitized.ok, sanitized.rejection_reason
    msg = _check_row_count(sanitized.sanitized or {}, "data.csv", 100)
    assert msg is not None
    assert "n=80" in msg


def test_sanitizer_to_audit_cluster_analysis_fires(dataset_100):
    from nora.sanitizer import sanitize

    raw = {
        "type": "cluster_analysis",
        "method": "kmeans",
        "n_observations": 80,
        "n_clusters": 2,
        "n_features": 2,
        "variables": ["x", "y"],
        "cluster_labels": ["c1", "c2"],
        "cluster_sizes": {"c1": 40, "c2": 40},
        "centroids": {
            "c1": {"x": 0.0, "y": 0.0},
            "c2": {"x": 1.0, "y": 1.0},
        },
    }
    sanitized = sanitize(raw)
    assert sanitized.ok, sanitized.rejection_reason
    msg = _check_row_count(sanitized.sanitized or {}, "data.csv", 100)
    assert msg is not None
    assert "n=80" in msg


# ---------------------------------------------------------------------------
# row_count — header-row detection on .csv / .tsv
# ---------------------------------------------------------------------------

def test_row_count_csv_with_header(tmp_path: Path) -> None:
    """A typical CSV with string column names: count is line count
    minus 1 for the header. The pre-fix unconditional minus-1 also
    handled this case correctly."""
    from nora.schema import row_count

    p = tmp_path / "with_header.csv"
    p.write_text("col_a,col_b,col_c\n1,2,3\n4,5,6\n")
    assert row_count(p) == 2


def test_row_count_csv_without_header(tmp_path: Path) -> None:
    """A headerless CSV (raw instrument dump, anonymous panel data,
    log file renamed to .csv) — every line is data. The pre-fix
    code subtracted 1 for a header that didn't exist, producing an
    audit count off by one and false-flagging scripts that
    correctly counted N rows."""
    from nora.schema import row_count

    p = tmp_path / "no_header.csv"
    p.write_text("1,2,3\n4,5,6\n7,8,9\n")
    assert row_count(p) == 3


def test_row_count_tsv_with_header(tmp_path: Path) -> None:
    from nora.schema import row_count

    p = tmp_path / "panel.tsv"
    p.write_text("id\tyear\tvalue\n1\t2020\t3.14\n2\t2021\t2.71\n")
    assert row_count(p) == 2


def test_row_count_empty_file(tmp_path: Path) -> None:
    from nora.schema import row_count

    p = tmp_path / "empty.csv"
    p.write_text("")
    assert row_count(p) == 0


def test_row_count_header_only_file(tmp_path: Path) -> None:
    """A file with only a header line — zero data rows. Distinguishes
    from a single-data-row headerless file (which should report 1)."""
    from nora.schema import row_count

    p = tmp_path / "only_header.csv"
    p.write_text("a,b,c\n")
    assert row_count(p) == 0


def test_row_count_csv_with_quoted_multiline_field(tmp_path: Path) -> None:
    """RFC 4180 lets a CSV field contain newlines inside quotes — one
    logical record can span multiple physical lines. The pre-fix
    byte-streamed line counter treated every physical ``\\n`` as a row
    boundary, so:

        id,note
        1,"hello
        world"
        2,"another
        multi
        line"

    counted as 6 lines → 5 rows after header offset, even though the
    file has 2 data rows. That false count then false-flagged the
    submit_script row-count audit with a ``ROW COUNT CHANGE`` warning
    whenever the analysis (correctly) saw N=2. The fix routes through
    ``csv.reader`` which honours quoting and treats embedded newlines
    as part of the field, not a record terminator.
    """
    from nora.schema import row_count

    p = tmp_path / "multiline.csv"
    p.write_text('id,note\n1,"hello\nworld"\n2,"another\nmulti\nline"\n')
    assert row_count(p) == 2, (
        "embedded \\n inside quoted CSV fields must not bump the row "
        "count — that would false-flag the submit_script audit"
    )


# ---------------------------------------------------------------------------
# names_only fast path must agree with row_count on header detection
# ---------------------------------------------------------------------------

def test_names_only_headerless_csv_does_not_consume_data_as_header(
    tmp_path: Path,
) -> None:
    """A headerless numeric CSV used to come back through the
    ``names_only`` fast path with variable names ``["1","2","3"]``
    (pandas' default ``header='infer'`` consumed row 1 as a header)
    AND ``observation_count=2`` (from ``row_count``, which correctly
    detected the lack of a header).

    Two surfaces of the schema response disagreed about the same
    file: the variable list said row 1 was a column name, the row
    count said it was data. Share ``_csv_has_header`` between the two
    so they agree: variables are auto-generated placeholders
    (``0,1,2…``) and observation_count covers every line.
    """
    from nora.schema import extract

    # Use values that don't overlap with pandas' integer-placeholder
    # names (0, 1, 2…) so the "names look like placeholders, not data"
    # assertion is unambiguous.
    p = tmp_path / "no_header.csv"
    p.write_text("100,200,300\n400,500,600\n")

    res = extract(p, depth="names_only")
    assert res["status"] == "ok"
    # Variables must NOT be ``["100","200","300"]`` — that would mean
    # row 1 got silently promoted to column names against the header
    # heuristic's verdict.
    names = [v["name"] for v in res["variables"]]
    for data_value in ("100", "200", "300"):
        assert data_value not in names, (
            f"headerless CSV must not surface its first data row as column "
            f"names; got {names}"
        )
    # Pandas ``header=None`` produces integer placeholders 0..n-1,
    # which ``safe_key`` stringifies. Confirm the response carries
    # exactly those placeholders for a 3-column file.
    assert names == ["0", "1", "2"], (
        f"headerless CSV should fall back to pandas' integer "
        f"placeholders; got {names}"
    )
    assert res["observation_count"] == 2, (
        f"both data rows must be counted; got "
        f"{res['observation_count']}"
    )


def test_names_only_headered_csv_still_uses_first_row_as_names(
    tmp_path: Path,
) -> None:
    """Don't regress the headered case — the fast path must still
    detect ``id,name,value`` style headers and surface them as the
    variable list. Mixed-type first rows are the common case for
    research CSVs; the headerless branch only applies to all-numeric
    first records."""
    from nora.schema import extract

    p = tmp_path / "with_header.csv"
    p.write_text("id,score,year\n1,3.14,2020\n2,2.71,2021\n3,1.41,2022\n")
    res = extract(p, depth="names_only")
    assert res["status"] == "ok"
    names = [v["name"] for v in res["variables"]]
    assert names == ["id", "score", "year"]
    assert res["observation_count"] == 3


def test_names_only_headerless_tsv_matches_row_count(tmp_path: Path) -> None:
    """Mirror of the CSV test for TSV — the fast path shares the
    same heuristic, so headerless TSVs must not have row 1 consumed
    as a header either."""
    from nora.schema import extract, row_count

    # Same numeric-overlap caveat as the CSV test — pick values that
    # can't collide with pandas' 0-indexed integer placeholders.
    p = tmp_path / "no_header.tsv"
    p.write_text("100\t200\t300\n400\t500\t600\n700\t800\t900\n")

    rc = row_count(p)
    res = extract(p, depth="names_only")
    assert res["observation_count"] == rc, (
        f"names_only fast path and row_count must agree on the same "
        f"file: fast path says {res['observation_count']}, "
        f"row_count says {rc}"
    )
    names = [v["name"] for v in res["variables"]]
    for data_value in ("100", "200", "300"):
        assert data_value not in names
