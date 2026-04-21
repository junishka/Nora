"""Integration tests for the get_schema MCP tool's policy enforcement.

These tests exercise the full tool-handler path:
``get_schema`` loads the policy from ``<cwd>/.builder/policy.json``,
compares the requested depth against the per-dataset ceiling, and
denies requests that exceed it. Successful responses carry a
``policy_max_depth`` field so Claude knows what the ceiling is
without needing to hit a denial first.

The policy module itself is unit-tested in ``test_policy.py``; this
file covers the wiring between the tool layer and the policy.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from builder.config import set_cwd
from builder.policy import (
    BuilderPolicy,
    DatasetPolicy,
    save_policy,
)
from builder.tools import get_schema


def _call_get_schema(args: dict) -> dict:
    """Call the @tool-wrapped ``get_schema`` and return the decoded
    JSON payload (unwrapped from the MCP content envelope).
    """
    envelope = asyncio.run(get_schema.handler(args))
    # MCP response shape: {"content": [{"type": "text", "text": "..."}], ...}
    text_block = envelope["content"][0]["text"]
    return json.loads(text_block)


# ---------------------------------------------------------------------------
# Happy path — no policy file → conservative default applies
# ---------------------------------------------------------------------------

def test_default_policy_allows_names_types(tmp_path: Path):
    """No policy file → default ceiling is names_types. A request at
    that ceiling succeeds and the response advertises the ceiling."""
    set_cwd(tmp_path)
    csv = tmp_path / "d.csv"
    csv.write_text("x,y\n1,2\n3,4\n5,6\n")

    resp = _call_get_schema({"dataset": "d.csv", "depth": "names_types"})
    assert resp["status"] == "ok"
    assert resp["depth"] == "names_types"
    assert resp["policy_max_depth"] == "names_types"


def test_default_policy_allows_names_only(tmp_path: Path):
    """A narrower-than-ceiling request also succeeds (policy is a
    ceiling, not a fixed value)."""
    set_cwd(tmp_path)
    csv = tmp_path / "d.csv"
    csv.write_text("x,y\n1,2\n3,4\n5,6\n")

    resp = _call_get_schema({"dataset": "d.csv", "depth": "names_only"})
    assert resp["status"] == "ok"
    assert resp["depth"] == "names_only"


def test_default_policy_denies_names_types_labels(tmp_path: Path):
    """Conservative default ceiling is names_types; asking for labels
    must be denied unless the researcher has opted in per-dataset."""
    set_cwd(tmp_path)
    csv = tmp_path / "d.csv"
    csv.write_text("x,y\n1,2\n3,4\n5,6\n")

    resp = _call_get_schema({"dataset": "d.csv", "depth": "names_types_labels"})
    assert resp["status"] == "denied"
    assert resp["requested_depth"] == "names_types_labels"
    assert resp["policy_max_depth"] == "names_types"
    # The reason should mention the ceiling so Claude can explain to
    # the researcher what needs to change to unlock.
    assert "default" in resp["reason"].lower()


def test_default_policy_denies_summary(tmp_path: Path):
    """Even further above the ceiling — still denied."""
    set_cwd(tmp_path)
    csv = tmp_path / "d.csv"
    csv.write_text("x,y\n1,2\n3,4\n5,6\n")

    resp = _call_get_schema({
        "dataset": "d.csv",
        "depth": "names_types_labels_summary",
    })
    assert resp["status"] == "denied"


# ---------------------------------------------------------------------------
# Explicit policy — researcher opts into richer schema for a dataset
# ---------------------------------------------------------------------------

def test_explicit_policy_raises_ceiling(tmp_path: Path):
    """When the researcher has set `max_depth: names_types_labels` for
    this dataset in policy.json, that depth is now allowed."""
    set_cwd(tmp_path)
    csv = tmp_path / "d.csv"
    csv.write_text("x,y\n1,2\n3,4\n5,6\n")

    save_policy(tmp_path, BuilderPolicy(
        datasets={"d.csv": DatasetPolicy(
            max_depth="names_types_labels",
            set_at="2026-04-21T00:00:00+00:00",
        )},
    ))

    resp = _call_get_schema({"dataset": "d.csv", "depth": "names_types_labels"})
    assert resp["status"] == "ok"
    assert resp["depth"] == "names_types_labels"
    assert resp["policy_max_depth"] == "names_types_labels"


def test_explicit_policy_denial_mentions_explicit(tmp_path: Path):
    """Denial reason distinguishes an explicit ceiling from the
    default — helps the researcher understand whether they need to
    raise the ceiling or whether it's just the default applying."""
    set_cwd(tmp_path)
    csv = tmp_path / "d.csv"
    csv.write_text("x,y\n1,2\n3,4\n5,6\n")

    save_policy(tmp_path, BuilderPolicy(
        datasets={"d.csv": DatasetPolicy(
            max_depth="names_only",
            set_at="2026-04-21T00:00:00+00:00",
        )},
    ))

    resp = _call_get_schema({"dataset": "d.csv", "depth": "names_types"})
    assert resp["status"] == "denied"
    assert "explicit" in resp["reason"].lower()
    assert resp["policy_max_depth"] == "names_only"


def test_policy_applies_per_dataset(tmp_path: Path):
    """Two datasets, two different ceilings — each is enforced
    independently."""
    set_cwd(tmp_path)
    csv_a = tmp_path / "public.csv"
    csv_a.write_text("x,y\n1,2\n3,4\n")
    csv_b = tmp_path / "sensitive.csv"
    csv_b.write_text("x,y\n1,2\n3,4\n")

    save_policy(tmp_path, BuilderPolicy(
        datasets={
            "public.csv": DatasetPolicy(max_depth="names_types_labels_summary"),
            "sensitive.csv": DatasetPolicy(max_depth="names_only"),
        },
    ))

    # The permissive dataset allows the richest depth.
    resp_a = _call_get_schema({
        "dataset": "public.csv",
        "depth": "names_types_labels_summary",
    })
    assert resp_a["status"] == "ok"

    # The restrictive dataset denies even names_types.
    resp_b = _call_get_schema({"dataset": "sensitive.csv", "depth": "names_types"})
    assert resp_b["status"] == "denied"
    assert resp_b["policy_max_depth"] == "names_only"


# ---------------------------------------------------------------------------
# Ceiling is always reported on success
# ---------------------------------------------------------------------------

def test_ok_response_always_includes_policy_max_depth(tmp_path: Path):
    """Every successful response carries policy_max_depth so Claude can
    inform future calls without needing to probe for a denial."""
    set_cwd(tmp_path)
    csv = tmp_path / "d.csv"
    csv.write_text("x,y\n1,2\n3,4\n5,6\n")

    for d in ("names_only", "names_types"):
        resp = _call_get_schema({"dataset": "d.csv", "depth": d})
        assert resp["status"] == "ok"
        assert "policy_max_depth" in resp
