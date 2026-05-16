"""Property tests for the ``cluster_analysis`` sanitizer shape.

Pins the two new SDC primitives:
  1. Whole-cluster suppression — clusters with size < min_n drop
     entirely (cluster_sizes entry, centroid row, every per-cluster
     dict). Partial publication would leak size through which
     clusters survived.
  2. Per-cluster precision clamping — each centroid value is
     clamped using THAT cluster's N, not the global n_observations.
     A 12-person cluster's centroid carries ~3 sigfigs; a 12,000-
     person cluster's centroid carries more.

Plus the structural exclusion of per-observation assignments
(``labels_``, ``cluster``, ``cluster_membership``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nora.sanitizer import sanitize, supported_types  # noqa: E402


def _good_payload() -> dict:
    return {
        "type": "cluster_analysis",
        "method": "kmeans",
        "distance_metric": "euclidean",
        "n_observations": 200,
        "n_clusters": 4,
        "n_features": 3,
        "variables": ["age", "income", "tenure"],
        "cluster_labels": ["c1", "c2", "c3", "tinyZ"],
        "cluster_sizes": {"c1": 60, "c2": 47, "c3": 88, "tinyZ": 5},
        "centroids": {
            "c1": {"age": 34.5, "income": 52123.4, "tenure": 4.5},
            "c2": {"age": 28.1, "income": 38567.1, "tenure": 2.3},
            "c3": {"age": 45.9, "income": 71234.5, "tenure": 8.9},
            "tinyZ": {"age": 31.7, "income": 89999.0, "tenure": 0.5},
        },
        "within_cluster_ss": {"c1": 1234.5, "c2": 567.8, "c3": 2345.6, "tinyZ": 45.0},
        "total_within_ss": 4192.9,
        "between_cluster_ss": 12345.6,
        "total_ss": 16538.5,
        "ss_ratio": 0.746,
        "inertia": 4192.9,
        "n_iterations": 12,
    }


def test_cluster_analysis_is_a_supported_type() -> None:
    assert "cluster_analysis" in supported_types()


def test_well_formed_payload_sanitizes() -> None:
    res = sanitize(_good_payload())
    assert res.ok, res.rejection_reason
    s = res.sanitized
    assert s["type"] == "cluster_analysis"
    assert s["method"] == "kmeans"
    assert s["distance_metric"] == "euclidean"


def test_small_cluster_dropped_whole() -> None:
    """The new SDC primitive: clusters below ``min_n_descriptive``
    are suppressed entirely. Their entry in cluster_sizes,
    centroid row, within_cluster_ss — everything drops together.
    Partial publication would leak size through which clusters
    survived."""
    res = sanitize(_good_payload())
    s = res.sanitized
    # small (n=5) suppressed.
    assert "tinyZ" not in s["cluster_sizes"]
    assert "tinyZ" not in s["centroids"]
    assert "tinyZ" not in s.get("within_cluster_ss", {})
    assert "tinyZ" not in s["cluster_labels"]
    # n_clusters reduced.
    assert s["n_clusters"] == 3
    # Surviving clusters retain all their fields.
    for cl in ("c1", "c2", "c3"):
        assert cl in s["cluster_sizes"]
        assert cl in s["centroids"]
        assert set(s["centroids"][cl].keys()) == {"age", "income", "tenure"}


def test_small_cluster_label_never_echoed() -> None:
    """Privacy: the suppressed cluster's label is data-derived
    (synthetic but tied to which clustering result fired). Never
    appears in transformations log or rejection_reason."""
    res = sanitize(_good_payload())
    for t in res.transformations:
        assert "tinyZ" not in t


def test_per_cluster_precision_clamping_on_centroids() -> None:
    """Each centroid value gets clamped using its OWN cluster's N,
    not the global n_observations. Compare what happens when a
    cluster has 12 members vs 12,000 members on the same raw value."""
    p = _good_payload()
    p["n_observations"] = 12012
    p["cluster_sizes"] = {"c1": 12, "c2": 12000, "c3": 88, "tinyZ": 5}
    p["centroids"]["c1"]["income"] = 52123.456789
    p["centroids"]["c2"]["income"] = 52123.456789
    p["centroids"]["c3"]["income"] = 52123.456789
    res = sanitize(p)
    assert res.ok
    s = res.sanitized
    c1_val = s["centroids"]["c1"]["income"]   # n=12 → fewer sigfigs
    c2_val = s["centroids"]["c2"]["income"]   # n=12000 → more sigfigs
    # The 12-person cluster centroid loses precision relative to the
    # 12,000-person one. Both should differ from each other after
    # per-cluster clamping (same raw input, different N).
    assert c1_val != c2_val, (
        f"per-cluster clamp didn't fire: c1 (n=12) and c2 (n=12000) "
        f"both rounded to {c1_val}"
    )
    # The 12-person centroid is rounded more coarsely.
    assert abs(c1_val - 52123.456789) > abs(c2_val - 52123.456789)


def test_per_observation_assignments_structurally_excluded() -> None:
    """The privacy carve-out: per-observation cluster assignments
    (``labels_``, ``cluster``, ``cluster_membership``) are NOT in
    the allowlist. Even hand-crafted payloads through
    ``nora.result(type="cluster_analysis", ...)`` can't smuggle
    them. Same structural-absence pattern as RDD's McCrary density
    and PCA's factor_scores."""
    p = _good_payload()
    p["labels_"] = [0, 1, 2, 3, 0, 1, 2, 3]
    p["cluster"] = [0, 1, 2]
    p["cluster_membership"] = [0, 1, 0, 1]
    p["assignments"] = list(range(200))
    p["row_to_cluster"] = {"row_42": "c1", "row_99": "c2"}
    res = sanitize(p)
    assert res.ok
    s = res.sanitized
    for forbidden in (
        "labels_", "cluster", "cluster_membership",
        "assignments", "row_to_cluster",
    ):
        assert forbidden not in s, f"{forbidden!r} leaked through"


def test_method_must_be_in_valid_set() -> None:
    p = _good_payload()
    p["method"] = "made_up_clustering"
    res = sanitize(p)
    assert not res.ok


def test_required_fields_enforced() -> None:
    for missing in (
        "type", "method", "n_observations", "n_clusters", "n_features",
        "variables", "cluster_labels", "cluster_sizes", "centroids",
    ):
        p = _good_payload()
        del p[missing]
        res = sanitize(p)
        assert not res.ok, f"missing {missing!r} should reject"


def test_all_clusters_suppressed_rejects() -> None:
    p = _good_payload()
    p["cluster_sizes"] = {"c1": 3, "c2": 4, "c3": 2, "tinyZ": 1}
    res = sanitize(p)
    assert not res.ok
    assert "cluster-size gate" in res.rejection_reason


def test_n_clusters_claim_mismatch_with_labels() -> None:
    p = _good_payload()
    p["n_clusters"] = 99  # doesn't match cluster_labels length
    res = sanitize(p)
    assert not res.ok


def test_n_features_claim_mismatch() -> None:
    p = _good_payload()
    p["n_features"] = 99
    res = sanitize(p)
    assert not res.ok


def test_structural_caps() -> None:
    # n_clusters cap.
    p = _good_payload()
    big_labels = [f"cluster_{i}" for i in range(60)]
    p["cluster_labels"] = big_labels
    p["n_clusters"] = 60
    p["cluster_sizes"] = {c: 100 for c in big_labels}
    p["centroids"] = {c: {"age": 1, "income": 1, "tenure": 1} for c in big_labels}
    res = sanitize(p)
    assert not res.ok and "structural cap" in res.rejection_reason
    # n_features cap.
    p = _good_payload()
    big_vars = [f"v{i}" for i in range(110)]
    p["variables"] = big_vars
    p["n_features"] = 110
    res = sanitize(p)
    assert not res.ok and "structural cap" in res.rejection_reason


def test_cluster_size_must_be_non_negative_int() -> None:
    p = _good_payload()
    p["cluster_sizes"]["c1"] = -5
    res = sanitize(p)
    assert not res.ok


def test_missing_cluster_size_entry_rejects() -> None:
    """A declared cluster_label that's missing from cluster_sizes
    bypasses the gate. Reject rather than guess."""
    p = _good_payload()
    del p["cluster_sizes"]["c2"]
    res = sanitize(p)
    assert not res.ok
    assert "cluster_sizes entry" in res.rejection_reason
    # Don't echo the missing label.
    assert "c2" not in res.rejection_reason


def test_centroids_outer_keys_must_be_surviving_clusters() -> None:
    p = _good_payload()
    p["centroids"]["UNDECLARED"] = {"age": 1, "income": 1, "tenure": 1}
    res = sanitize(p)
    assert res.ok
    assert "UNDECLARED" not in res.sanitized["centroids"]


def test_centroids_inner_keys_must_be_declared_variables() -> None:
    p = _good_payload()
    p["centroids"]["c1"]["BOGUS_VAR"] = 42.0
    res = sanitize(p)
    assert res.ok
    assert "BOGUS_VAR" not in res.sanitized["centroids"]["c1"]


def test_invalid_linkage_dropped() -> None:
    p = _good_payload()
    p["linkage"] = "made_up_linkage"
    res = sanitize(p)
    assert res.ok
    assert "linkage" not in res.sanitized


def test_invalid_distance_metric_dropped() -> None:
    p = _good_payload()
    p["distance_metric"] = "made_up_distance"
    res = sanitize(p)
    assert res.ok
    assert "distance_metric" not in res.sanitized


def test_within_cluster_ss_uses_per_cluster_clamp() -> None:
    """``within_cluster_ss`` is an aggregate over each cluster's
    members; same logic as centroids — precision should scale with
    the cluster's OWN N, not the global N. Pin the per-key clamp
    primitive fires on this field too.

    Compare a 12-member cluster's SS against a 12,000-member
    cluster's SS, both with the same raw input: the small cluster's
    value loses precision, the large cluster's keeps more sigfigs."""
    p = _good_payload()
    p["n_observations"] = 12012
    p["cluster_sizes"] = {"c1": 12, "c2": 12000, "c3": 88, "tinyZ": 5}
    raw_val = 1234.567891
    p["within_cluster_ss"] = {
        "c1": raw_val, "c2": raw_val, "c3": raw_val, "tinyZ": raw_val,
    }
    # Adjust centroids to match the new cluster_sizes set.
    p["centroids"] = {
        "c1": {"age": 30, "income": 50000, "tenure": 5},
        "c2": {"age": 30, "income": 50000, "tenure": 5},
        "c3": {"age": 30, "income": 50000, "tenure": 5},
        "tinyZ": {"age": 30, "income": 50000, "tenure": 5},
    }
    res = sanitize(p)
    assert res.ok
    s = res.sanitized
    # tinyZ suppressed entirely (size < min_n).
    assert "tinyZ" not in s["within_cluster_ss"]
    c1_val = s["within_cluster_ss"]["c1"]   # n=12 → fewer sigfigs
    c2_val = s["within_cluster_ss"]["c2"]   # n=12000 → more sigfigs
    assert c1_val != c2_val, (
        "per-key clamp didn't fire on within_cluster_ss"
    )


def test_linkage_matrix_and_merge_heights_structurally_excluded() -> None:
    """Hierarchical clusterings produce a linkage matrix and merge-
    height series — per-merge records over the data. These have no
    slot in the allowlist; even hand-crafted payloads through
    ``nora.result(type="cluster_analysis", ...)`` can't smuggle
    them. Same construction as the per-observation assignments
    exclusion above."""
    p = _good_payload()
    p["method"] = "hierarchical"
    p["linkage"] = "ward"
    p["linkage_matrix"] = [[0, 1, 0.5], [2, 3, 1.2], [4, 5, 2.1]]
    p["merge_heights"] = [0.5, 1.2, 2.1, 4.5]
    p["dendrogram"] = "(((1:0.5,2:0.5):1.2,(3:0.5,4:0.5):1.2):2.1);"
    p["fit_children_"] = [[0, 1], [2, 3]]
    res = sanitize(p)
    assert res.ok
    s = res.sanitized
    for forbidden in (
        "linkage_matrix", "merge_heights", "dendrogram", "fit_children_",
    ):
        assert forbidden not in s, f"{forbidden!r} leaked through"


def test_dbscan_payload_with_no_centroids_accepted() -> None:
    """DBSCAN has no centroids by construction. The shape accepts
    the payload without ``centroids``; cluster sizes + noise count
    cross."""
    p = {
        "type": "cluster_analysis",
        "method": "dbscan",
        "n_observations": 200,
        "n_clusters": 3,
        "n_features": 2,
        "variables": ["x", "y"],
        "cluster_labels": ["c1", "c2", "c3"],
        "cluster_sizes": {"c1": 60, "c2": 47, "c3": 88},
        "n_noise_points": 5,
        # NO centroids field — should be OK for dbscan.
    }
    res = sanitize(p)
    assert res.ok
    assert "centroids" not in res.sanitized
    assert res.sanitized["n_noise_points"] == 5


def test_dbscan_payload_with_centroids_still_validates() -> None:
    """If a DBSCAN payload includes centroids anyway (caller computed
    them post-hoc), the structure is validated against the same
    cross-field rules as any other clustering. No bypass."""
    p = {
        "type": "cluster_analysis",
        "method": "dbscan",
        "n_observations": 200,
        "n_clusters": 2,
        "n_features": 2,
        "variables": ["x", "y"],
        "cluster_labels": ["c1", "c2"],
        "cluster_sizes": {"c1": 60, "c2": 50},
        "n_noise_points": 12,
        "centroids": {
            "c1": {"x": 1.0, "y": 2.0, "UNDECLARED": 99.0},
            "c2": {"x": 3.0, "y": 4.0},
            "UNDECLARED_CLUSTER": {"x": 99, "y": 99},
        },
    }
    res = sanitize(p)
    assert res.ok
    s = res.sanitized
    # Undeclared inner key dropped despite dbscan path being lenient
    # about field presence — structure validation still fires.
    assert "UNDECLARED" not in s["centroids"]["c1"]
    assert "UNDECLARED_CLUSTER" not in s["centroids"]


def test_non_dbscan_method_still_requires_centroids() -> None:
    """The conditional centroids logic must not let a kmeans payload
    slip through without centroids — that would be a malformed
    payload, not a methodologically-empty one."""
    p = _good_payload()
    del p["centroids"]
    res = sanitize(p)
    assert not res.ok
    assert "centroids is required" in res.rejection_reason


def test_f_statistic_per_variable_round_trips() -> None:
    """Per-variable F-statistic dict (variable discrimination) goes
    through the new per-variable numeric dict slot."""
    p = _good_payload()
    p["f_statistic_per_variable"] = {
        "age": 12.345, "income": 23.456, "tenure": 7.89,
        "UNDECLARED_VAR": 99.0,
    }
    res = sanitize(p)
    assert res.ok
    s = res.sanitized
    assert "f_statistic_per_variable" in s
    assert set(s["f_statistic_per_variable"].keys()) == {"age", "income", "tenure"}


def test_cut_height_scalar_round_trips() -> None:
    """Hierarchical fits emit a single ``cut_height`` scalar — the
    dendrogram threshold above which the desired ``n_clusters``
    remain. Aggregate scalar; the full height series is excluded."""
    p = _good_payload()
    p["method"] = "hierarchical"
    p["linkage"] = "ward"
    p["cut_height"] = 6.4789
    res = sanitize(p)
    assert res.ok
    assert "cut_height" in res.sanitized


def test_n_noise_points_round_trips() -> None:
    p = _good_payload()
    p["method"] = "dbscan"
    del p["centroids"]  # absent OK for dbscan
    p["n_noise_points"] = 17
    res = sanitize(p)
    assert res.ok
    assert res.sanitized["n_noise_points"] == 17


def test_pam_and_dbscan_in_method_enum() -> None:
    """Verify the two enum additions are accepted."""
    for method in ("pam", "dbscan"):
        p = _good_payload()
        p["method"] = method
        if method == "dbscan":
            del p["centroids"]
            p["n_noise_points"] = 0
        res = sanitize(p)
        assert res.ok, f"{method}: {res.rejection_reason}"


def test_safe_key_collision_on_clusters_rejects() -> None:
    p = _good_payload()
    p["cluster_labels"] = ["ab\tcd", "ab cd", "c3", "c4"]
    p["cluster_sizes"] = {"ab\tcd": 30, "ab cd": 30, "c3": 50, "c4": 60}
    p["centroids"] = {
        "ab\tcd": {"age": 1, "income": 1, "tenure": 1},
        "ab cd": {"age": 2, "income": 2, "tenure": 2},
        "c3": {"age": 3, "income": 3, "tenure": 3},
        "c4": {"age": 4, "income": 4, "tenure": 4},
    }
    res = sanitize(p)
    assert not res.ok
    assert "collision" in res.rejection_reason.lower()
