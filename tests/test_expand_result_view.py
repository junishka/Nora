"""Tests for the ``view`` option on ``expand_result``.

The default behavior returns the full stored payload. ``view="full"``
is equivalent. ``view="coefficients"`` is a regression-specific
shorthand that drops the variance-covariance matrix (``vcov``) and
per-predictor VIF table — the two largest fields on a wide regression
result. The model uses this when it only needs the headline
coefficient pattern; ``view="full"`` is still available when the
diagnostics actually matter.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from nora.config import use_cwd
from nora.store import StoredResult, get_store, reset_store_for_tests
from nora.tools import HANDLERS


def _mcp_text(payload: dict) -> dict:
    return json.loads(payload["content"][0]["text"])


@pytest.fixture(autouse=True)
def _clear_caches():
    reset_store_for_tests()
    yield
    reset_store_for_tests()


def _insert_regression(cwd: Path) -> StoredResult:
    """Plant a stored linear_regression payload with vcov + vif so
    the view trim has something to remove."""
    store = get_store(cwd)
    payload = {
        "type": "linear_regression",
        "n": 100,
        "coefficients": {"x1": 0.42, "x2": -0.13},
        "standard_errors": {"x1": 0.05, "x2": 0.04},
        "p_values": {"x1": 0.001, "x2": 0.06},
        "r_squared": 0.31,
        "condition_number": 4.2,
        "vif": {"x1": 1.05, "x2": 1.05},
        "vcov": {
            "x1": {"x1": 0.0025, "x2": 0.0001},
            "x2": {"x1": 0.0001, "x2": 0.0016},
        },
    }
    return store.insert(
        label="regress y on x1+x2",
        analysis_type="linear_regression",
        sanitized_payload=payload,
        language="R",
        script_code="lm(y ~ x1 + x2, data=df)",
        transformations=[],
    )


def test_default_view_returns_full_payload(tmp_path: Path) -> None:
    cwd = tmp_path / "session"
    cwd.mkdir()
    row = _insert_regression(cwd)
    with use_cwd(cwd):
        res = asyncio.run(HANDLERS["expand_result"]({"result_id": row.id}))
    body = _mcp_text(res)
    assert body["status"] == "ok"
    assert "vcov" in body["payload"]
    assert "vif" in body["payload"]
    assert "view" not in body  # absent when default
    assert "view_dropped_fields" not in body


def test_view_full_is_equivalent_to_default(tmp_path: Path) -> None:
    cwd = tmp_path / "session"
    cwd.mkdir()
    row = _insert_regression(cwd)
    with use_cwd(cwd):
        res = asyncio.run(HANDLERS["expand_result"]({
            "result_id": row.id, "view": "full",
        }))
    body = _mcp_text(res)
    assert body["status"] == "ok"
    assert "vcov" in body["payload"]
    assert "vif" in body["payload"]
    assert body.get("view") == "full"


def test_view_coefficients_drops_vcov_and_vif(tmp_path: Path) -> None:
    """The ``coefficients`` view is a token-saver: drops the two
    largest collinearity-diagnostic fields, keeps the headline
    coefficient pattern."""
    cwd = tmp_path / "session"
    cwd.mkdir()
    row = _insert_regression(cwd)
    with use_cwd(cwd):
        res = asyncio.run(HANDLERS["expand_result"]({
            "result_id": row.id, "view": "coefficients",
        }))
    body = _mcp_text(res)
    assert body["status"] == "ok"
    # Trimmed.
    assert "vcov" not in body["payload"]
    assert "vif" not in body["payload"]
    # Kept (headline pattern).
    assert body["payload"]["coefficients"] == {"x1": 0.42, "x2": -0.13}
    assert body["payload"]["standard_errors"] == {"x1": 0.05, "x2": 0.04}
    assert body["payload"]["r_squared"] == 0.31
    assert body["payload"]["condition_number"] == 4.2
    # Discoverability: the response names what was dropped so the
    # model knows it can call view="full" if it needs them back.
    assert body.get("view") == "coefficients"
    assert sorted(body.get("view_dropped_fields") or []) == ["vcov", "vif"]


def test_view_coefficients_on_non_regression_is_passthrough(
    tmp_path: Path,
) -> None:
    """The coefficients view is regression-specific. Apply it to a
    descriptive payload and the payload comes back unchanged — no
    fields dropped, no error."""
    cwd = tmp_path / "session"
    cwd.mkdir()
    store = get_store(cwd)
    row = store.insert(
        label="summary of x",
        analysis_type="descriptive",
        sanitized_payload={
            "type": "descriptive", "variable": "x",
            "n": 50, "mean": 1.0, "sd": 0.2, "missing_count": 0,
        },
        language="Python",
        script_code="",
        transformations=[],
    )
    with use_cwd(cwd):
        res = asyncio.run(HANDLERS["expand_result"]({
            "result_id": row.id, "view": "coefficients",
        }))
    body = _mcp_text(res)
    assert body["status"] == "ok"
    assert body["payload"]["type"] == "descriptive"
    assert body["payload"]["mean"] == 1.0
    # No fields dropped because the trim only applies to regressions.
    assert "view_dropped_fields" not in body


def test_unknown_view_is_rejected(tmp_path: Path) -> None:
    cwd = tmp_path / "session"
    cwd.mkdir()
    row = _insert_regression(cwd)
    with use_cwd(cwd):
        res = asyncio.run(HANDLERS["expand_result"]({
            "result_id": row.id, "view": "garbage",
        }))
    body = _mcp_text(res)
    assert body["status"] == "error"
    assert "view must be" in body["reason"]
