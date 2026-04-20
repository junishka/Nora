"""Tests for the magnitude_table sanitizer and the dominance rule.

Core invariants:
1. ``max_share`` is never in the sanitized output, under any input.
2. A cell that fails either the primary n-threshold OR the dominance
   threshold is fully suppressed (both value and n become markers).
3. A cell that passes both is precision-clamped based on its n.
4. The dominance rule's definition matches the SDC literature: any
   max_share in [0, threshold] passes; anything above or outside
   [0, 1] fails.
"""

from __future__ import annotations

import math

from hypothesis import given
from hypothesis import strategies as st

from builder.sanitizer import DEFAULT_CONFIG, SDCConfig, sanitize, supported_types
from builder.sdc import (
    DOMINANCE_THRESHOLD_DEFAULT,
    dominance_fails,
    suppression_marker,
)


# ---------------------------------------------------------------------------
# dominance_fails — primitive
# ---------------------------------------------------------------------------

def test_dominance_passes_well_distributed():
    assert not dominance_fails(0.0)
    assert not dominance_fails(0.5)
    assert not dominance_fails(0.85)  # exactly at threshold → passes


def test_dominance_fails_over_threshold():
    assert dominance_fails(0.86)
    assert dominance_fails(0.99)
    assert dominance_fails(1.0)


def test_dominance_rejects_out_of_range():
    assert dominance_fails(-0.1)
    assert dominance_fails(1.5)


def test_dominance_rejects_non_finite():
    assert dominance_fails(float("nan"))
    assert dominance_fails(float("inf"))


def test_dominance_custom_threshold():
    strict = 0.5
    assert dominance_fails(0.6, threshold=strict)
    assert not dominance_fails(0.4, threshold=strict)


# ---------------------------------------------------------------------------
# Sanitizer strategy + invariants
# ---------------------------------------------------------------------------

_name = st.text(
    alphabet=st.characters(
        min_codepoint=33, max_codepoint=126,
        blacklist_characters="\"\\",
    ),
    min_size=1, max_size=8,
)

_finite_float = st.floats(
    allow_nan=False, allow_infinity=False,
    min_value=-1e9, max_value=1e9,
)


@st.composite
def magnitude_payloads(draw, max_groups: int = 6):
    groups = draw(st.lists(_name, min_size=1, max_size=max_groups, unique=True))
    cells = {}
    for g in groups:
        n = draw(st.integers(min_value=0, max_value=500))
        value = draw(_finite_float)
        max_share = draw(st.floats(min_value=0.0, max_value=1.0,
                                   allow_nan=False, allow_infinity=False))
        cells[g] = {"value": value, "n": n, "max_share": max_share}
    return {
        "type": "magnitude_table",
        "row_variable": draw(_name),
        "value_variable": draw(_name),
        "aggregation": draw(st.sampled_from(["sum", "mean"])),
        "cells": cells,
    }


def test_magnitude_table_is_supported():
    assert "magnitude_table" in supported_types()


@given(raw=magnitude_payloads())
def test_max_share_never_emitted(raw):
    """The load-bearing privacy property."""
    r = sanitize(raw)
    if not r.ok:
        return
    for cell in r.sanitized["cells"].values():
        assert "max_share" not in cell, (
            "max_share leaked through — dominance-metric-only invariant "
            "violated"
        )


@given(raw=magnitude_payloads())
def test_suppressed_cells_have_both_markers(raw):
    """Suppression is atomic: if value is suppressed, n is too, and vice versa."""
    r = sanitize(raw)
    if not r.ok:
        return
    marker = suppression_marker(DEFAULT_CONFIG.cell_suppression_threshold)
    for cell in r.sanitized["cells"].values():
        v_is_mark = cell.get("value") == marker
        n_is_mark = cell.get("n") == marker
        assert v_is_mark == n_is_mark, (
            f"partial suppression leaked structure: {cell!r}"
        )


@given(raw=magnitude_payloads())
def test_small_n_cells_always_suppressed(raw):
    """Any raw cell with n below threshold must be suppressed in output."""
    r = sanitize(raw)
    if not r.ok:
        return
    threshold = DEFAULT_CONFIG.cell_suppression_threshold
    marker = suppression_marker(threshold)
    for raw_key, raw_cell in raw["cells"].items():
        if raw_cell["n"] < threshold:
            # Find the key in the sanitized output (the sanitizer
            # applies safe_key which may rewrite, but for our ASCII
            # strategy the key is unchanged).
            out_cell = r.sanitized["cells"].get(raw_key)
            assert out_cell is not None
            assert out_cell["value"] == marker


@given(raw=magnitude_payloads())
def test_dominant_cells_always_suppressed(raw):
    """Any raw cell with max_share > dominance threshold must be suppressed."""
    r = sanitize(raw)
    if not r.ok:
        return
    threshold_dom = DEFAULT_CONFIG.dominance_threshold
    threshold_n = DEFAULT_CONFIG.cell_suppression_threshold
    marker = suppression_marker(threshold_n)
    for raw_key, raw_cell in raw["cells"].items():
        # Only check cells that pass the n threshold — so we isolate
        # the dominance rule's contribution.
        if raw_cell["n"] >= threshold_n and raw_cell["max_share"] > threshold_dom:
            out_cell = r.sanitized["cells"].get(raw_key)
            assert out_cell is not None
            assert out_cell["value"] == marker, (
                f"cell {raw_key!r} had max_share={raw_cell['max_share']} "
                f"> {threshold_dom} but was not suppressed"
            )


# ---------------------------------------------------------------------------
# Shape tests — readable, exact
# ---------------------------------------------------------------------------

def test_happy_path_well_distributed():
    r = sanitize({
        "type": "magnitude_table",
        "row_variable": "state",
        "value_variable": "income",
        "aggregation": "sum",
        "cells": {
            "CA": {"value": 12500000, "n": 125, "max_share": 0.03},
            "NY": {"value": 9800000, "n": 98, "max_share": 0.04},
        },
    })
    assert r.ok
    # Both cells published, max_share absent.
    for state in ("CA", "NY"):
        cell = r.sanitized["cells"][state]
        assert "max_share" not in cell
        assert isinstance(cell["value"], float)
        assert isinstance(cell["n"], int)


def test_dominance_fires_exactly_over_threshold():
    """Threshold is inclusive — 0.85 passes, 0.851 fails."""
    r = sanitize({
        "type": "magnitude_table",
        "row_variable": "g",
        "value_variable": "v",
        "aggregation": "sum",
        "cells": {
            "edge":  {"value": 1000, "n": 50, "max_share": 0.85},
            "over":  {"value": 1000, "n": 50, "max_share": 0.851},
        },
    })
    assert r.ok
    marker = suppression_marker(DEFAULT_CONFIG.cell_suppression_threshold)
    assert r.sanitized["cells"]["edge"]["value"] == 1000.0  # passes
    assert r.sanitized["cells"]["over"]["value"] == marker  # fails


def test_custom_dominance_threshold():
    strict = SDCConfig(dominance_threshold=0.5)
    r = sanitize({
        "type": "magnitude_table",
        "row_variable": "g",
        "value_variable": "v",
        "aggregation": "sum",
        "cells": {"A": {"value": 1000, "n": 50, "max_share": 0.6}},
    }, config=strict)
    assert r.ok
    marker = suppression_marker(10)
    assert r.sanitized["cells"]["A"]["value"] == marker


def test_rejects_non_sum_mean_aggregation():
    r = sanitize({
        "type": "magnitude_table",
        "row_variable": "g",
        "value_variable": "v",
        "aggregation": "median",
        "cells": {"A": {"value": 1, "n": 1, "max_share": 0.0}},
    })
    assert not r.ok
    assert "aggregation" in (r.rejection_reason or "")


def test_rejects_cell_missing_max_share():
    r = sanitize({
        "type": "magnitude_table",
        "row_variable": "g",
        "value_variable": "v",
        "aggregation": "sum",
        "cells": {"A": {"value": 1000, "n": 50}},  # no max_share
    })
    assert not r.ok


def test_transformations_name_suppression_reasons():
    """The log distinguishes n-failures from dominance-failures."""
    r = sanitize({
        "type": "magnitude_table",
        "row_variable": "g",
        "value_variable": "v",
        "aggregation": "sum",
        "cells": {
            "tiny": {"value": 100, "n": 3, "max_share": 0.1},     # primary
            "dom":  {"value": 1000, "n": 50, "max_share": 0.95},  # dominance
            "ok":   {"value": 500, "n": 50, "max_share": 0.1},
        },
    })
    assert r.ok
    logged = " ".join(r.transformations)
    assert "primary suppression" in logged
    assert "dominance suppression" in logged
    assert "'tiny'" in logged
    assert "'dom'" in logged
