"""Tests for the result store.

Scope: insert round-trips, ID sequencing, list/get retrieval, persistence
across connections. Kept focused — the store is simple enough that
exhaustive property testing isn't warranted; the sanitizer is where the
real guarantee lives.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from builder.store import ResultStore


@pytest.fixture
def store(tmp_path: Path) -> ResultStore:
    return ResultStore(tmp_path / ".builder" / "results.db")


def _sample_payload() -> dict:
    return {
        "type": "linear_regression",
        "n": 1000,
        "coefficients": {"x": 0.5},
        "standard_errors": {"x": 0.05},
        "r_squared": 0.42,
        "response_variable": "y",
        "predictor_variables": ["x"],
    }


def test_empty_store_has_zero_count(store: ResultStore):
    assert store.count() == 0
    assert store.list_all() == []


def test_insert_returns_sequential_ids(store: ResultStore):
    r1 = store.insert(
        label="first", analysis_type="linear_regression",
        sanitized_payload=_sample_payload(),
        language="R", script_code="x <- 1", transformations=[],
    )
    r2 = store.insert(
        label="second", analysis_type="linear_regression",
        sanitized_payload=_sample_payload(),
        language="R", script_code="x <- 2", transformations=[],
    )
    r3 = store.insert(
        label="third", analysis_type="linear_regression",
        sanitized_payload=_sample_payload(),
        language="Stata", script_code="reg y x", transformations=[],
    )
    assert r1.id == "M1"
    assert r2.id == "M2"
    assert r3.id == "M3"
    assert store.count() == 3


def test_get_returns_inserted_row(store: ResultStore):
    inserted = store.insert(
        label="my regression",
        analysis_type="linear_regression",
        sanitized_payload=_sample_payload(),
        language="R",
        script_code="library(stats); lm(y ~ x)",
        transformations=["clamped coefficients to 4 sig figs"],
    )
    fetched = store.get(inserted.id)
    assert fetched is not None
    assert fetched.id == inserted.id
    assert fetched.label == "my regression"
    assert fetched.analysis_type == "linear_regression"
    assert fetched.sanitized_payload == _sample_payload()
    assert fetched.language == "R"
    assert fetched.script_code == "library(stats); lm(y ~ x)"
    assert fetched.transformations == ["clamped coefficients to 4 sig figs"]


def test_get_missing_returns_none(store: ResultStore):
    assert store.get("M999") is None
    assert store.get("") is None


def test_list_all_orders_by_creation(store: ResultStore):
    for i in range(5):
        store.insert(
            label=f"label-{i}",
            analysis_type="descriptive",
            sanitized_payload={"type": "descriptive", "variable": f"v{i}"},
            language="R",
            script_code=f"# script {i}",
            transformations=[],
        )
    rows = store.list_all()
    assert [r.label for r in rows] == [f"label-{i}" for i in range(5)]


def test_persistence_across_connections(tmp_path: Path):
    db = tmp_path / ".builder" / "results.db"
    s1 = ResultStore(db)
    s1.insert(
        label="persistent",
        analysis_type="linear_regression",
        sanitized_payload=_sample_payload(),
        language="R",
        script_code="x",
        transformations=["a"],
    )
    s1.close()

    s2 = ResultStore(db)
    assert s2.count() == 1
    row = s2.get("M1")
    assert row is not None
    assert row.label == "persistent"
    assert row.transformations == ["a"]
    s2.close()


def test_unicode_roundtrip(store: ResultStore):
    """Labels, code, and transformations with non-ASCII survive the JSON round-trip."""
    store.insert(
        label="régression — n=200",
        analysis_type="linear_regression",
        sanitized_payload=_sample_payload(),
        language="R",
        script_code="# résidus… émission",
        transformations=["clamped — 4 sig figs"],
    )
    row = store.get("M1")
    assert row is not None
    assert row.label == "régression — n=200"
    assert row.script_code == "# résidus… émission"
    assert row.transformations == ["clamped — 4 sig figs"]
