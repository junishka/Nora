"""Tests for the ``search_schema`` tool.

Wide research datasets (hundreds of variables) make ``get_schema``
expensive: the full schema lands in context just to answer "which
columns are about salary." ``search_schema`` filters by a case-
insensitive substring against name + label and returns just the
matches with a ``total_matches`` count so the model can refine.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pandas as pd
import pytest

from nora.config import set_cwd
from nora.tools import HANDLERS


def _mcp_text(payload: dict) -> dict:
    return json.loads(payload["content"][0]["text"])


def _call(args: dict) -> dict:
    return _mcp_text(asyncio.run(HANDLERS["search_schema"](args)))


@pytest.fixture
def wide_csv(tmp_path: Path) -> Path:
    """Synthetic wide dataset with salary-related and tenure-related
    columns plus a bunch of unrelated ones, so the matcher has both
    hits and non-hits to work with."""
    set_cwd(tmp_path)
    df = pd.DataFrame({
        "salary_2020": [100.0, 110.0, 120.0],
        "salary_2021": [105.0, 115.0, 125.0],
        "log_salary_2021": [4.6, 4.7, 4.8],
        "ceo_tenure_years": [3, 7, 10],
        "tenure_band": ["a", "b", "c"],
        "age": [30, 40, 50],
        "gender": ["F", "M", "F"],
        "region_id": [1, 2, 3],
    })
    p = tmp_path / "wide.csv"
    df.to_csv(p, index=False)
    return p


def test_search_schema_finds_substring_matches(wide_csv: Path) -> None:
    body = _call({"dataset": "wide.csv", "query": "salary"})
    assert body["status"] == "ok"
    names = sorted(v["name"] for v in body["variables"])
    assert names == ["log_salary_2021", "salary_2020", "salary_2021"]
    assert body["total_matches"] == 3
    assert body["truncated"] is False
    assert body["query"] == "salary"


def test_search_schema_is_case_insensitive(wide_csv: Path) -> None:
    body_lower = _call({"dataset": "wide.csv", "query": "salary"})
    body_upper = _call({"dataset": "wide.csv", "query": "SALARY"})
    body_mixed = _call({"dataset": "wide.csv", "query": "SaLaRy"})
    a = sorted(v["name"] for v in body_lower["variables"])
    b = sorted(v["name"] for v in body_upper["variables"])
    c = sorted(v["name"] for v in body_mixed["variables"])
    assert a == b == c


def test_search_schema_no_matches_returns_empty_with_zero_total(
    wide_csv: Path,
) -> None:
    body = _call({"dataset": "wide.csv", "query": "nonexistent_xyz"})
    assert body["status"] == "ok"
    assert body["variables"] == []
    assert body["total_matches"] == 0
    assert body["truncated"] is False


def test_search_schema_respects_limit_and_reports_truncation(
    wide_csv: Path,
) -> None:
    body = _call({"dataset": "wide.csv", "query": "a", "limit": 2})
    # The query 'a' matches multiple columns; we cap at 2.
    assert body["status"] == "ok"
    assert len(body["variables"]) == 2
    assert body["total_matches"] >= 3, body
    assert body["truncated"] is True
    assert body["limit"] == 2


def test_search_schema_rejects_empty_query(wide_csv: Path) -> None:
    body = _call({"dataset": "wide.csv", "query": ""})
    assert body["status"] == "error"
    assert "query" in body["reason"].lower()
    assert "get_schema" in body["reason"]


def test_search_schema_rejects_missing_dataset() -> None:
    body = _call({"dataset": "", "query": "salary"})
    assert body["status"] == "error"
    assert "dataset" in body["reason"]


def test_search_schema_rejects_path_outside_cwd(tmp_path: Path) -> None:
    set_cwd(tmp_path)
    body = _call({"dataset": "../escape.csv", "query": "x"})
    assert body["status"] == "denied"


def test_search_schema_file_not_found(tmp_path: Path) -> None:
    set_cwd(tmp_path)
    body = _call({"dataset": "missing.csv", "query": "x"})
    assert body["status"] == "error"
    assert "not found" in body["reason"].lower()


def test_search_schema_response_carries_dataset_metadata(
    wide_csv: Path,
) -> None:
    """The response should be self-describing so the model can act on
    it without round-tripping back to get_schema for context."""
    body = _call({"dataset": "wide.csv", "query": "salary"})
    assert body["status"] == "ok"
    assert body["dataset"] == "wide.csv"
    assert body["depth"] == "names_types_labels"
    assert body["observation_count"] == 3
    assert body["variable_count"] == 8
    assert body["policy_max_depth"]
