"""Property tests for the ``factor_decomposition`` sanitizer shape.

Real-fit helper tests live in ``tests/test_from_pca_real_fits.py``.
This module exercises the shape on hand-crafted payloads:
structural caps, cross-field validation, structural exclusion of
factor scores (the privacy carve-out).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nora.sanitizer import sanitize, supported_types  # noqa: E402


def _good_payload() -> dict:
    return {
        "type": "factor_decomposition",
        "method": "pca",
        "rotation": "none",
        "n_observations": 200,
        "n_variables": 5,
        "n_components": 3,
        "variables": ["v1", "v2", "v3", "v4", "v5"],
        "components": ["PC1", "PC2", "PC3"],
        "loadings": {
            "v1": {"PC1": 0.85, "PC2": 0.12, "PC3": -0.03},
            "v2": {"PC1": 0.78, "PC2": 0.31, "PC3": 0.05},
            "v3": {"PC1": -0.42, "PC2": 0.70, "PC3": 0.15},
            "v4": {"PC1": 0.30, "PC2": -0.55, "PC3": 0.62},
            "v5": {"PC1": 0.10, "PC2": 0.20, "PC3": -0.80},
        },
        "explained_variance": {"PC1": 1.31, "PC2": 1.02, "PC3": 0.87},
        "explained_variance_ratio": {"PC1": 0.28, "PC2": 0.22, "PC3": 0.19},
        "cumulative_variance": {"PC1": 0.28, "PC2": 0.50, "PC3": 0.69},
        "eigenvalues": {"PC1": 1.31, "PC2": 1.02, "PC3": 0.87},
        "communalities": {"v1": 0.73, "v2": 0.71, "v3": 0.69, "v4": 0.74, "v5": 0.69},
    }


def test_factor_decomposition_is_a_supported_type() -> None:
    assert "factor_decomposition" in supported_types()


def test_well_formed_payload_sanitizes() -> None:
    res = sanitize(_good_payload())
    assert res.ok, res.rejection_reason
    s = res.sanitized
    assert s["type"] == "factor_decomposition"
    assert s["method"] == "pca"
    assert s["rotation"] == "none"
    assert set(s["variables"]) == {"v1", "v2", "v3", "v4", "v5"}
    assert s["components"] == ["PC1", "PC2", "PC3"]


def test_required_fields_enforced() -> None:
    for missing in (
        "type", "method", "n_observations", "n_variables", "n_components",
        "variables", "loadings",
    ):
        p = _good_payload()
        del p[missing]
        res = sanitize(p)
        assert not res.ok, f"missing {missing!r} should reject"


def test_method_must_be_in_valid_set() -> None:
    p = _good_payload()
    p["method"] = "bogus_method"
    res = sanitize(p)
    assert not res.ok
    assert "method" in res.rejection_reason


def test_factor_scores_structurally_excluded() -> None:
    """Privacy carve-out: per-observation factor scores are NOT in
    the allowlist. Even hand-crafted payloads through
    ``nora.result(type="factor_decomposition", ...)`` can't smuggle
    them — same pattern as RDD's McCrary density curve and KM's
    step function."""
    p = _good_payload()
    p["factor_scores"] = [[0.1, 0.2, 0.3], [0.5, 0.6, 0.7]]
    p["transformed_data"] = [[1.1, 2.2], [3.3, 4.4]]
    res = sanitize(p)
    assert res.ok
    assert "factor_scores" not in res.sanitized
    assert "transformed_data" not in res.sanitized


def test_n_variables_claim_must_match_variables_list() -> None:
    p = _good_payload()
    p["n_variables"] = 99  # lie
    res = sanitize(p)
    assert not res.ok
    assert "n_variables" in res.rejection_reason


def test_n_components_claim_must_match_components_list() -> None:
    p = _good_payload()
    p["n_components"] = 99  # lie
    res = sanitize(p)
    assert not res.ok


def test_structural_cap_on_n_variables() -> None:
    p = _good_payload()
    big = [f"v{i}" for i in range(150)]
    p["variables"] = big
    p["n_variables"] = 150
    p["loadings"] = {v: {"PC1": 0.1, "PC2": 0.1, "PC3": 0.1} for v in big}
    res = sanitize(p)
    assert not res.ok
    assert "structural cap" in res.rejection_reason


def test_structural_cap_on_n_components() -> None:
    p = _good_payload()
    p["n_components"] = 100
    p["components"] = [f"PC{i}" for i in range(100)]
    res = sanitize(p)
    assert not res.ok


def test_loadings_outer_keys_must_be_declared_variables() -> None:
    p = _good_payload()
    p["loadings"]["UNDECLARED"] = {"PC1": 0.99, "PC2": 0.01, "PC3": 0.01}
    res = sanitize(p)
    assert res.ok
    assert "UNDECLARED" not in res.sanitized["loadings"]


def test_loadings_inner_keys_must_be_declared_components() -> None:
    p = _good_payload()
    p["loadings"]["v1"]["PC99"] = 0.5
    res = sanitize(p)
    assert res.ok
    assert "PC99" not in res.sanitized["loadings"]["v1"]


def test_per_component_dict_keys_must_be_declared() -> None:
    p = _good_payload()
    p["eigenvalues"]["BOGUS_PC"] = 0.5
    res = sanitize(p)
    assert res.ok
    assert "BOGUS_PC" not in res.sanitized["eigenvalues"]


def test_per_variable_dict_keys_must_be_declared() -> None:
    p = _good_payload()
    p["communalities"]["UNDECLARED"] = 0.99
    res = sanitize(p)
    assert res.ok
    assert "UNDECLARED" not in res.sanitized["communalities"]


def test_min_n_observations_gate() -> None:
    p = _good_payload()
    p["n_observations"] = 3
    res = sanitize(p)
    assert not res.ok


def test_invalid_rotation_dropped() -> None:
    p = _good_payload()
    p["rotation"] = "made_up_rotation"
    res = sanitize(p)
    assert res.ok
    assert "rotation" not in res.sanitized


def test_safe_key_collision_on_variables_rejects() -> None:
    p = _good_payload()
    p["variables"] = ["ab\tcd", "ab cd", "v3", "v4", "v5"]
    p["n_variables"] = 5
    p["loadings"]["ab\tcd"] = {"PC1": 0.1, "PC2": 0.1, "PC3": 0.1}
    p["loadings"]["ab cd"] = {"PC1": 0.1, "PC2": 0.1, "PC3": 0.1}
    del p["loadings"]["v1"]; del p["loadings"]["v2"]
    res = sanitize(p)
    assert not res.ok
    assert "collision" in res.rejection_reason.lower()
