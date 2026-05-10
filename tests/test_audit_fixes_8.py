"""Regression tests for the seventh batch of reviewer-flagged fixes.

Findings addressed:

1. Sanitizer rejection paths in frequency_table / crosstab /
   magnitude_table no longer echo data-derived cell / row / column
   labels via ``rejection_reason``. Pre-fix, malformed cell shapes
   embedded a ``safe_key``-clamped attacker-controlled label
   (40 chars per rejection) into the model-visible message —
   submit_script forwarded that into both the inline result and the
   persisted diagnostic row.

2. ``_collect_allowed`` no longer names dropped unknown / forbidden
   top-level fields, and no longer names per-entry malformed inner
   keys inside dict_numeric. Pre-fix, a script could emit up to
   50 raw-row strings as JSON field names and read them back via
   ``transformations``.

3. ``request_data`` redacts library exception bodies on dataset
   load: only the exception class name reaches the model, matching
   the posture ``get_schema`` already used for the same readers.

4. ``_na_count`` is symmetric: a small ``na_count`` denies just like
   a small ``non_na_count`` did. Pre-fix, a single missing
   observation could be reported exactly through this path even
   though the schema summary suppresses the same channel.

5. ``_collect_allowed`` dict_numeric, magnitude_table cells, and
   correlation row/col axes detect ``safe_key`` collisions instead
   of silently overwriting. The vcov path already had this check;
   these are the remaining shapes.

6. Required-field validation runs AFTER ``_collect_allowed`` filters
   by type. Pre-fix, a wrong-typed ``coefficients`` /
   ``standard_errors`` / ``mean1`` / etc. survived to ``ok=True``
   with the field absent because ``_require_fields`` only checked
   key presence in the raw payload.
"""

from __future__ import annotations

from nora.sanitizer import sanitize


# --- Finding 1: rejection_reason no longer echoes data-derived labels ----


def test_freq_table_count_rejection_does_not_echo_level_name() -> None:
    """A non-int count value rejects the payload — but the level name
    that the script chose for that cell must not appear in the
    rejection_reason. The level name is data-derived and crosses to
    the model via submit_script."""
    secret = "patient_42_ssn_123_45_6789_dob_1980"
    r = sanitize({
        "type": "frequency_table",
        "variable": "x",
        "counts": {secret: "not_an_int"},
        "n": 100,
        "missing_count": 0,
    })
    assert not r.ok
    assert secret not in (r.rejection_reason or "")
    assert "withheld" in (r.rejection_reason or "").lower()


def test_crosstab_row_rejection_does_not_echo_row_label() -> None:
    secret_row = "secret_row_payload_xxxxxxxxxxxxxxxxxxxx"
    r = sanitize({
        "type": "crosstab",
        "row_variable": "r",
        "col_variable": "c",
        "counts": {secret_row: "not_a_dict"},
    })
    assert not r.ok
    assert secret_row not in (r.rejection_reason or "")


def test_crosstab_cell_rejection_does_not_echo_labels() -> None:
    secret_row = "row_secret_42"
    secret_col = "col_secret_99"
    r = sanitize({
        "type": "crosstab",
        "row_variable": "r",
        "col_variable": "c",
        "counts": {secret_row: {secret_col: "not_an_int"}},
    })
    assert not r.ok
    assert secret_row not in (r.rejection_reason or "")
    assert secret_col not in (r.rejection_reason or "")


def test_magnitude_table_rejection_does_not_echo_group_label() -> None:
    secret_group = "secret_group_payload_for_exfil"
    r = sanitize({
        "type": "magnitude_table",
        "row_variable": "g",
        "value_variable": "v",
        "aggregation": "sum",
        "cells": {secret_group: "not_a_dict"},
    })
    assert not r.ok
    assert secret_group not in (r.rejection_reason or "")


# --- Finding 2: unknown fields counted, not named ------------------------


def test_collect_allowed_unknown_fields_not_named_in_transformations() -> None:
    """Unknown top-level field names are data-derived under an
    adversarial script (raw cell values encoded as JSON keys). The
    transformations log emits a count, never the names."""
    secret_a = "fake_field_carrying_secret_payload_AAAA"
    secret_b = "fake_field_carrying_secret_payload_BBBB"
    r = sanitize({
        "type": "descriptive",
        "variable": "x",
        "n": 100,
        "mean": 1.0,
        "sd": 0.5,
        "missing_count": 0,
        secret_a: 42,
        secret_b: "anything",
    })
    assert r.ok, r.rejection_reason
    joined = "\n".join(r.transformations)
    assert secret_a not in joined
    assert secret_b not in joined
    assert "withheld" in joined.lower()


def test_collect_allowed_dict_inner_drops_not_named_in_transformations() -> None:
    """Non-finite values or non-string keys inside a dict_numeric
    field were previously logged with the inner key name. They are
    now collapsed into a per-parent count."""
    secret_inner = "secret_coefficient_name_xxxxxxxxxxxxxxxx"
    r = sanitize({
        "type": "linear_regression",
        "n": 100,
        "response_variable": "y",
        "predictor_variables": ["x1", "x2"],
        "coefficients": {"x1": 1.0, "x2": 2.0, secret_inner: float("nan")},
        "standard_errors": {"x1": 0.1, "x2": 0.2},
    })
    assert r.ok, r.rejection_reason
    joined = "\n".join(r.transformations)
    assert secret_inner not in joined


# --- Finding 4: symmetric na_count suppression ---------------------------


def test_na_count_denies_when_missing_count_is_small(tmp_path) -> None:
    """The schema summary suppresses missingness counts on either
    side (``schema._suppress_rare_count``); this path must match."""
    import pandas as pd
    from nora.data_request import handle as data_request_handle

    # 100 observations, 1 missing — non-NA side is huge, NA side is 1.
    df = pd.DataFrame({"x": [1.0] * 99 + [None]})
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)
    r = data_request_handle(csv_path, "na_count", "x")
    assert r.status == "denied", r
    assert "rarer" in (r.reason or "").lower() or "too small" in (r.reason or "").lower()


# --- Finding 5: collision detection in dict_numeric / magnitude / correlation


def test_dict_numeric_collision_drops_duplicate_silently_with_count() -> None:
    """Two coefficient names that ``safe_key`` collapses to the same
    form must NOT silently overwrite. The current behavior keeps the
    first and drops the duplicate, with a count-only transformation."""
    from nora.text_safety import safe_key
    raw_a = "x" * 40 + "_first"
    raw_b = "x" * 40 + "_second"
    assert safe_key(raw_a) == safe_key(raw_b)
    r = sanitize({
        "type": "linear_regression",
        "n": 100,
        "response_variable": "y",
        "predictor_variables": [raw_a],
        "coefficients": {raw_a: 1.0, raw_b: 99.0},
        "standard_errors": {raw_a: 0.1},
    })
    assert r.ok, r.rejection_reason
    coefs = r.sanitized["coefficients"]
    # First write wins; duplicate is dropped, not overwritten.
    assert len(coefs) == 1
    assert next(iter(coefs.values())) == 1.0
    joined = "\n".join(r.transformations)
    assert "duplicate" in joined.lower()
    # Names of the colliding keys are withheld.
    assert raw_a not in joined
    assert raw_b not in joined


def test_magnitude_table_label_collision_rejects() -> None:
    from nora.text_safety import safe_key
    raw_a = "x" * 40 + "_one"
    raw_b = "x" * 40 + "_two"
    assert safe_key(raw_a) == safe_key(raw_b)
    r = sanitize({
        "type": "magnitude_table",
        "row_variable": "g",
        "value_variable": "v",
        "aggregation": "sum",
        "cells": {
            raw_a: {"value": 100.0, "n": 50, "max_share": 0.1},
            raw_b: {"value": 200.0, "n": 50, "max_share": 0.1},
        },
    })
    assert not r.ok
    assert "collision" in (r.rejection_reason or "").lower()
    assert raw_a not in (r.rejection_reason or "")
    assert raw_b not in (r.rejection_reason or "")


def test_correlation_row_collision_drops_duplicate() -> None:
    from nora.text_safety import safe_key
    raw_a = "x" * 40 + "_one"
    raw_b = "x" * 40 + "_two"
    assert safe_key(raw_a) == safe_key(raw_b)
    r = sanitize({
        "type": "correlation_matrix",
        "n": 100,
        "method": "pearson",
        "variables": [raw_a, "y"],
        "correlations": {
            raw_a: {raw_a: 1.0, "y": 0.5},
            raw_b: {raw_a: 1.0, "y": 0.99},
            "y": {raw_a: 0.5, "y": 1.0},
        },
    })
    assert r.ok, r.rejection_reason
    joined = "\n".join(r.transformations)
    assert "duplicate" in joined.lower()
    assert raw_a not in joined
    assert raw_b not in joined


# --- Finding 6: required fields revalidated after type filtering --------


def test_ols_wrong_typed_coefficients_rejected_after_filter() -> None:
    """``_collect_allowed`` drops a non-dict ``coefficients`` field
    silently. Without the post-filter recheck, the response was
    ``ok=True`` with no coefficients at all — a "successful" but
    structurally-empty regression."""
    r = sanitize({
        "type": "linear_regression",
        "n": 100,
        "response_variable": "y",
        "predictor_variables": ["x1"],
        "coefficients": "not_a_dict",
        "standard_errors": {"x1": 0.1},
    })
    assert not r.ok
    assert "coefficients" in (r.rejection_reason or "")


def test_ttest_wrong_typed_t_statistic_rejected_after_filter() -> None:
    r = sanitize({
        "type": "t_test",
        "test_type": "one_sample",
        "n1": 100,
        "mean1": 1.0,
        "t_statistic": "not_a_number",
        "p_value": 0.05,
    })
    assert not r.ok
    assert "t_statistic" in (r.rejection_reason or "")


def test_descriptive_wrong_typed_mean_rejected_after_filter() -> None:
    r = sanitize({
        "type": "descriptive",
        "variable": "x",
        "n": 100,
        "mean": "not_a_number",
        "sd": 0.5,
        "missing_count": 0,
    })
    assert not r.ok
    assert "mean" in (r.rejection_reason or "")
