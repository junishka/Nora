"""Tests for ``executor._resolve_default_timeout``.

The function resolves the per-script wall-clock cap from the
``NORA_SCRIPT_TIMEOUT_SECONDS`` environment variable, falling back
to 300s. Three behaviours are pinned:

  - bad values fall back without crashing (and emit a warning so a
    typo isn't silent),
  - non-positive values fall back,
  - extreme values are clamped at a 24h ceiling so a runaway script
    can't hang the runner for years.
"""

from __future__ import annotations

import logging

import pytest

from nora import executor


def _resolve(monkeypatch: pytest.MonkeyPatch, value: str | None) -> int:
    if value is None:
        monkeypatch.delenv("NORA_SCRIPT_TIMEOUT_SECONDS", raising=False)
    else:
        monkeypatch.setenv("NORA_SCRIPT_TIMEOUT_SECONDS", value)
    return executor._resolve_default_timeout()


def test_unset_returns_default(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _resolve(monkeypatch, None) == 300


def test_explicit_value_within_range_is_honored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _resolve(monkeypatch, "600") == 600


def test_unparseable_value_falls_back_with_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    """A typo like ``5min`` must not crash the bridge — fall back to
    the default — but the user has clearly TRIED to set a value, so a
    silent fallback hides their typo. Warn so they see it."""
    with caplog.at_level(logging.WARNING, logger="nora.executor"):
        result = _resolve(monkeypatch, "5min")
    assert result == 300
    assert any(
        "not an integer" in rec.getMessage() for rec in caplog.records
    ), [rec.getMessage() for rec in caplog.records]


def test_zero_or_negative_falls_back_with_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    """``0`` would disable the timeout entirely; negative values are
    nonsensical. Both fall back to the default with a warning."""
    with caplog.at_level(logging.WARNING, logger="nora.executor"):
        result = _resolve(monkeypatch, "0")
    assert result == 300
    assert any("below floor" in rec.getMessage() for rec in caplog.records)


def test_extreme_value_is_clamped_at_24h_ceiling(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    """``2147483647`` (max int32) is clamped to 24h so a runaway
    script can't hang the runner for years. Real workloads (long
    bootstraps, simulations) fit well under 24h; anything beyond
    that is "should be a batch job", not interactive analysis."""
    with caplog.at_level(logging.WARNING, logger="nora.executor"):
        result = _resolve(monkeypatch, "2147483647")
    assert result == 24 * 60 * 60
    assert any("ceiling" in rec.getMessage() for rec in caplog.records)


def test_value_just_below_ceiling_passes_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A genuinely large but bounded value (e.g. 12h for a long
    Monte Carlo) must NOT be clamped — only values exceeding the
    24h ceiling are."""
    twelve_hours = 12 * 60 * 60
    assert _resolve(monkeypatch, str(twelve_hours)) == twelve_hours
