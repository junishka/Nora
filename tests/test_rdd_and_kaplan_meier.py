"""Tests for the new ``rdd`` and ``kaplan_meier`` shapes, including
the load-bearing privacy carve-outs.

For RDD the carve-out is *structural*: the McCrary density curve
and binscatter bin coordinates are not in the allowlist, so even a
hand-crafted payload routed through generic ``result(type="rdd",
...)`` cannot smuggle them past the sanitizer. The choice is the
same shape as the plot-vision policy — researcher-only by
construction, not by opt-in policy.

For Kaplan-Meier the carve-out is **safe form only**: the full step
function and per-event-time S(t) series are not in the allowlist.
The shape ships median survival + S(t) at preset horizons (1y / 3y
/ 5y / 10y) each gated by per-horizon ``n_at_risk_h``. Drop a
horizon, and its S(h) plus CI bounds drop together — partial
publication would leak n_at_risk through "this horizon survives,
that one doesn't".
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nora.sanitizer import sanitize, supported_types  # noqa: E402


# ---------------------------------------------------------------------------
# RDD
# ---------------------------------------------------------------------------


def _good_rdd() -> dict:
    return {
        "type": "rdd",
        "estimator": "local_polynomial",
        "running_variable": "income",
        "outcome_variable": "voted",
        "cutoff": 50000,
        "tau_conventional": 0.052,
        "tau_bias_corrected": 0.048,
        "tau_robust": 0.047,
        "se_conventional": 0.012,
        "se_bias_corrected": 0.013,
        "se_robust": 0.015,
        "p_conventional": 0.0001,
        "p_bias_corrected": 0.0002,
        "p_robust": 0.0017,
        "ci_lower_robust": 0.018,
        "ci_upper_robust": 0.076,
        "bandwidth_left": 8000,
        "bandwidth_right": 8000,
        "kernel": "triangular",
        "polynomial_order": 1,
        "effective_n_left": 850,
        "effective_n_right": 920,
    }


def test_rdd_is_a_supported_type() -> None:
    assert "rdd" in supported_types()


def test_rdd_well_formed_payload_sanitizes() -> None:
    res = sanitize(_good_rdd())
    assert res.ok, res.rejection_reason
    s = res.sanitized
    assert s["estimator"] == "local_polynomial"
    assert s["kernel"] == "triangular"
    assert s["polynomial_order"] == 1
    # tau / se / p / CI all preserved
    for k in ("tau_conventional", "tau_bias_corrected", "tau_robust",
              "se_conventional", "se_bias_corrected", "se_robust"):
        assert k in s


def test_rdd_per_side_min_n_gate() -> None:
    """The per-side effective-N gate guards against a 50/2 split
    that totals OK but has one side under-identified."""
    p = _good_rdd()
    p["effective_n_left"] = 3
    res = sanitize(p)
    assert not res.ok
    assert "effective_n_left" in res.rejection_reason


def test_rdd_effective_n_total_filled_when_absent() -> None:
    p = _good_rdd()
    res = sanitize(p)
    assert res.sanitized["effective_n_total"] == (
        p["effective_n_left"] + p["effective_n_right"]
    )


def test_rdd_mccrary_curve_structurally_excluded() -> None:
    """Privacy carve-out: McCrary's *density curve* (running-variable
    density evaluated at a grid of points around the cutoff) is the
    distribution of the most sensitive variable on the most
    identifying slice. The bare scalar test statistic also enables
    a cutoff-scan attack (placebo cutoffs c±δ map the density). We
    exclude the diagnostic entirely from the shape.

    Pin the structural exclusion: even when smuggled through generic
    ``result(type="rdd", ...)``, the density curve does NOT reach
    the model."""
    p = _good_rdd()
    p["mccrary_density_curve"] = [
        [48000, 0.00010], [48500, 0.00012], [49000, 0.00014],
        [49500, 0.00018], [50000, 0.00025], [50500, 0.00031],
    ]
    p["mccrary_z_statistic"] = -1.32
    p["mccrary_p_value"] = 0.187
    res = sanitize(p)
    assert res.ok
    s = res.sanitized
    assert "mccrary_density_curve" not in s
    assert "mccrary_z_statistic" not in s
    assert "mccrary_p_value" not in s


def test_rdd_binscatter_bins_structurally_excluded() -> None:
    """Privacy carve-out, binscatter near cutoff: bin widths shrink
    by construction at the cutoff to make the discontinuity visible,
    creating small-N cells over the most sensitive slice. Excluded
    structurally — not in the allowlist."""
    p = _good_rdd()
    p["binscatter_bins"] = [
        [49000, 0.40, 12], [49500, 0.42, 8], [50000, 0.55, 6],
        [50500, 0.58, 9], [51000, 0.61, 15],
    ]
    res = sanitize(p)
    assert res.ok
    assert "binscatter_bins" not in res.sanitized


def test_rdd_invalid_kernel_dropped() -> None:
    p = _good_rdd()
    p["kernel"] = "made_up_kernel"
    res = sanitize(p)
    assert res.ok
    assert "kernel" not in res.sanitized


def test_rdd_polynomial_order_capped() -> None:
    p = _good_rdd()
    p["polynomial_order"] = 7
    res = sanitize(p)
    assert res.ok
    assert "polynomial_order" not in res.sanitized


# ---------------------------------------------------------------------------
# Kaplan-Meier (safe form)
# ---------------------------------------------------------------------------


def _good_km() -> dict:
    return {
        "type": "kaplan_meier",
        "time_variable": "t_obs",
        "event_variable": "cens",
        "group_variable": "arm",
        "n_subjects": 300,
        "n_failures": 178,
        "median_survival_time": 36.5,
        "median_survival_ci_lower": 30.1,
        "median_survival_ci_upper": 42.8,
        "survival_at_1y": 0.85, "n_at_risk_1y": 270,
        "survival_at_3y": 0.62, "n_at_risk_3y": 150,
        "survival_at_5y": 0.48, "n_at_risk_5y": 50,
        "survival_at_1y_ci_lower": 0.81, "survival_at_1y_ci_upper": 0.89,
        "logrank_chi_squared": 5.2,
        "logrank_p_value": 0.022,
        "n_groups": 2,
    }


def test_kaplan_meier_is_a_supported_type() -> None:
    assert "kaplan_meier" in supported_types()


def test_km_well_formed_sanitizes() -> None:
    res = sanitize(_good_km())
    assert res.ok, res.rejection_reason
    s = res.sanitized
    assert s["n_subjects"] == 300
    assert s["n_failures"] == 178
    assert "median_survival_time" in s


def test_km_under_gated_horizon_dropped_with_its_ci_bounds() -> None:
    """The per-horizon gate suppresses S(h) AND its CI bounds
    together. Partial publication of just-the-CIs would leak
    n_at_risk through 'horizon survives by way of its bounds'."""
    p = _good_km()
    p["survival_at_10y"] = 0.20
    p["n_at_risk_10y"] = 3  # below min_n
    p["survival_at_10y_ci_lower"] = 0.05
    p["survival_at_10y_ci_upper"] = 0.40
    res = sanitize(p)
    assert res.ok
    s = res.sanitized
    assert "survival_at_10y" not in s
    assert "survival_at_10y_ci_lower" not in s
    assert "survival_at_10y_ci_upper" not in s


def test_km_horizon_without_at_risk_field_dropped() -> None:
    """Forgetting to supply ``n_at_risk_h`` for a horizon is treated
    the same as failing the gate — drop the horizon. The gate has
    no input without that field; allowing the S(h) value through
    would mean trusting the script."""
    p = _good_km()
    p["survival_at_10y"] = 0.20
    # NO n_at_risk_10y at all
    res = sanitize(p)
    assert res.ok
    assert "survival_at_10y" not in res.sanitized


def test_km_curve_data_structurally_excluded() -> None:
    """The full step function (per-event-time S(t) series) is the
    *unsafe form*. It's not in the allowlist; even hand-crafted
    payloads through generic ``result(type="kaplan_meier", ...)``
    can't smuggle it."""
    p = _good_km()
    p["survival_curve"] = [
        [0.1, 0.99], [0.5, 0.95], [1.0, 0.85], [2.0, 0.71],
    ]
    p["event_times"] = [0.1, 0.5, 1.0, 2.0]
    p["at_risk_curve"] = [300, 280, 270, 220]
    res = sanitize(p)
    assert res.ok
    s = res.sanitized
    assert "survival_curve" not in s
    assert "event_times" not in s
    assert "at_risk_curve" not in s


def test_km_n_failures_cannot_exceed_n_subjects() -> None:
    p = _good_km()
    p["n_failures"] = 9999
    res = sanitize(p)
    assert not res.ok
    assert "cannot exceed" in res.rejection_reason


def test_km_min_n_gate_on_subjects() -> None:
    p = _good_km()
    p["n_subjects"] = 3
    res = sanitize(p)
    assert not res.ok


def test_km_logrank_chi_preserved_when_grouped() -> None:
    res = sanitize(_good_km())
    s = res.sanitized
    assert "logrank_chi_squared" in s
    assert "logrank_p_value" in s
    assert "n_groups" in s


def test_km_n_failures_below_threshold_is_suppressed() -> None:
    """KM-on-rare-events: a survival curve with 3 deaths discloses
    those 3 individuals the same way Cox with the same n_failures
    does. The Cox path already coarsens this via
    ``_coarsen_small_cox_counts``; KM previously kept the exact
    count, leaving the same disclosure surface unguarded. The fix
    routes KM through the same helper so the two estimators apply
    the same rule to identically-shaped data."""
    p = _good_km()
    p["n_failures"] = 3
    res = sanitize(p)
    assert res.ok, res.rejection_reason
    assert res.sanitized["n_failures"] == "<10", (
        f"n_failures=3 must be coarsened (matching the Cox path), "
        f"got {res.sanitized['n_failures']!r}"
    )
    assert any("n_failures" in t for t in res.transformations)


def test_km_zero_failures_left_as_zero() -> None:
    """Mirrors ``test_ols_cox_zero_failures_left_as_zero``: the
    suppression rule fires for ``0 < n < threshold``. Zero events
    means there is no individual to identify, so the value passes
    through unchanged. Without this gate, a study where nothing
    happened would be over-suppressed."""
    p = _good_km()
    p["n_failures"] = 0
    res = sanitize(p)
    assert res.ok, res.rejection_reason
    assert res.sanitized["n_failures"] == 0
