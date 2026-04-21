"""Tests for the schema-depth policy module.

The policy is the researcher's consent mechanism: a ceiling on what
schema information Claude can see about each dataset. These tests
lock in:

- Conservative default behavior when no policy file exists.
- Round-trip load→save→load.
- Graceful fallback on malformed / wrong-version / corrupted files
  (a broken policy file must not lock the researcher out).
- Correct ceiling comparison in ``depth_allowed``.
- ``has_explicit_policy`` distinguishes set-by-researcher vs.
  inherited-from-default.
"""

from __future__ import annotations

import json
from pathlib import Path

from builder.policy import (
    DEFAULT_MAX_DEPTH,
    VALID_DEPTHS,
    BuilderPolicy,
    DatasetPolicy,
    depth_allowed,
    get_max_depth,
    has_explicit_policy,
    load_policy,
    policy_path,
    save_policy,
)


# ---------------------------------------------------------------------------
# Load / default behavior
# ---------------------------------------------------------------------------

def test_load_missing_file_returns_default(tmp_path: Path):
    """No policy file → default conservative policy. Never raises."""
    p = load_policy(tmp_path)
    assert p.version == 1
    assert p.default_max_depth == DEFAULT_MAX_DEPTH
    assert p.datasets == {}


def test_load_corrupted_json_falls_back_to_default(tmp_path: Path):
    """A broken JSON file must not lock the researcher out of Builder.
    Default policy applies silently — the safe fallback is the
    conservative one."""
    policy_path(tmp_path).parent.mkdir()
    policy_path(tmp_path).write_text("{ this is not valid JSON }")
    p = load_policy(tmp_path)
    assert p.default_max_depth == DEFAULT_MAX_DEPTH
    assert p.datasets == {}


def test_load_unknown_version_falls_back_to_default(tmp_path: Path):
    """Future versions should have a migration path; until one exists,
    bail to default rather than risk misinterpreting."""
    policy_path(tmp_path).parent.mkdir()
    policy_path(tmp_path).write_text(
        json.dumps({"version": 99, "default_max_depth": "names_types_labels"})
    )
    p = load_policy(tmp_path)
    assert p.default_max_depth == DEFAULT_MAX_DEPTH


def test_load_unknown_depth_in_default_falls_back(tmp_path: Path):
    """An invalid depth name in the policy file must not be forwarded
    — silently correct to the conservative default."""
    policy_path(tmp_path).parent.mkdir()
    policy_path(tmp_path).write_text(
        json.dumps({
            "version": 1,
            "default_max_depth": "names_types_labels_summary_extra_unreal",
        })
    )
    p = load_policy(tmp_path)
    assert p.default_max_depth == DEFAULT_MAX_DEPTH


def test_load_unknown_depth_in_per_dataset_falls_back(tmp_path: Path):
    """Same for per-dataset entries — unknown depth → conservative."""
    policy_path(tmp_path).parent.mkdir()
    policy_path(tmp_path).write_text(
        json.dumps({
            "version": 1,
            "datasets": {
                "survey.csv": {"max_depth": "bogus_tier"},
            },
        })
    )
    p = load_policy(tmp_path)
    assert p.datasets["survey.csv"].max_depth == DEFAULT_MAX_DEPTH


def test_load_valid_policy(tmp_path: Path):
    """A well-formed policy file loads into the expected dataclasses."""
    policy_path(tmp_path).parent.mkdir()
    policy_path(tmp_path).write_text(
        json.dumps({
            "version": 1,
            "default_max_depth": "names_types",
            "datasets": {
                "survey.csv": {
                    "max_depth": "names_types_labels",
                    "set_at": "2026-04-21T14:20:00+00:00",
                },
                "demographics.dta": {
                    "max_depth": "names_types_labels_summary",
                    "set_at": "2026-04-21T14:25:00+00:00",
                },
            },
        })
    )
    p = load_policy(tmp_path)
    assert p.default_max_depth == "names_types"
    assert p.datasets["survey.csv"].max_depth == "names_types_labels"
    assert (
        p.datasets["demographics.dta"].max_depth
        == "names_types_labels_summary"
    )


# ---------------------------------------------------------------------------
# Save + round-trip
# ---------------------------------------------------------------------------

def test_save_creates_dot_builder_dir(tmp_path: Path):
    """`.builder/` directory is created on save if it doesn't exist."""
    policy = BuilderPolicy(datasets={
        "a.csv": DatasetPolicy(max_depth="names_types_labels", set_at="t"),
    })
    save_policy(tmp_path, policy)
    assert policy_path(tmp_path).is_file()


def test_round_trip(tmp_path: Path):
    original = BuilderPolicy(
        default_max_depth="names_types",
        datasets={
            "a.csv": DatasetPolicy(
                max_depth="names_types_labels", set_at="2026-04-21T00:00:00+00:00"
            ),
            "b.dta": DatasetPolicy(
                max_depth="names_types_labels_summary", set_at="2026-04-21T00:01:00+00:00"
            ),
        },
    )
    save_policy(tmp_path, original)
    loaded = load_policy(tmp_path)
    assert loaded.default_max_depth == original.default_max_depth
    assert loaded.datasets == original.datasets


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

def test_get_max_depth_falls_back_to_default():
    policy = BuilderPolicy(default_max_depth="names_types")
    assert get_max_depth(policy, "missing.csv") == "names_types"


def test_get_max_depth_returns_explicit_when_set():
    policy = BuilderPolicy(
        default_max_depth="names_types",
        datasets={"x.csv": DatasetPolicy(max_depth="names_types_labels")},
    )
    assert get_max_depth(policy, "x.csv") == "names_types_labels"


def test_has_explicit_policy_true_for_set():
    policy = BuilderPolicy(
        datasets={"x.csv": DatasetPolicy(max_depth="names_types")}
    )
    assert has_explicit_policy(policy, "x.csv")


def test_has_explicit_policy_false_for_inherited():
    policy = BuilderPolicy()
    assert not has_explicit_policy(policy, "x.csv")


# ---------------------------------------------------------------------------
# depth_allowed — the ceiling comparison that gates `get_schema`
# ---------------------------------------------------------------------------

def test_depth_allowed_at_ceiling():
    assert depth_allowed("names_types_labels", "names_types_labels")


def test_depth_allowed_below_ceiling():
    assert depth_allowed("names_only", "names_types_labels_summary")
    assert depth_allowed("names_types", "names_types_labels")


def test_depth_allowed_above_ceiling():
    assert not depth_allowed("names_types_labels_summary", "names_types")
    assert not depth_allowed("names_types_labels", "names_types")


def test_depth_allowed_unknown_rejected():
    """Unknown depth names reject, not silently accept."""
    assert not depth_allowed("bogus", "names_types_labels")
    assert not depth_allowed("names_types", "bogus")


def test_all_valid_depths_orderable():
    """Every depth in VALID_DEPTHS must compare correctly against
    every other. Lock in the total ordering."""
    for i, lower in enumerate(VALID_DEPTHS):
        for j, upper in enumerate(VALID_DEPTHS):
            if i <= j:
                assert depth_allowed(lower, upper), (
                    f"{lower!r} should be allowed under ceiling {upper!r}"
                )
            else:
                assert not depth_allowed(lower, upper), (
                    f"{lower!r} should NOT be allowed under ceiling {upper!r}"
                )
