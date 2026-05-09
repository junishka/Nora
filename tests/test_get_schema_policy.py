"""Integration tests for the get_schema MCP tool's policy enforcement.

These tests exercise the full tool-handler path:
``get_schema`` loads the policy from ``<cwd>/.nora/policy.json``,
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

from nora.config import set_cwd
from nora.policy import (
    DEFAULT_MAX_DEPTH,
    NoraPolicy,
    DatasetPolicy,
    save_policy,
)
from nora.tools import get_schema


def _call_get_schema(args: dict) -> dict:
    """Call the @tool-wrapped ``get_schema`` and return the decoded
    JSON payload (unwrapped from the MCP content envelope).
    """
    envelope = asyncio.run(get_schema.handler(args))
    # MCP response shape: {"content": [{"type": "text", "text": "..."}], ...}
    text_block = envelope["content"][0]["text"]
    return json.loads(text_block)


# ---------------------------------------------------------------------------
# Happy path — no policy file → permissive default applies
# ---------------------------------------------------------------------------
#
# The default used to be ``names_types`` (conservative). It's now
# ``names_types_labels_summary`` (NA count + distinct count tier) —
# researchers want Claude reasoning about the richest non-leaky
# metadata by default and can dial down per-dataset via the Permission
# chip. These tests lock in the new default and the fact that every
# depth on the ladder succeeds without a policy file.


def test_default_is_names_types_labels_summary():
    """Guard: if someone flips the default again (e.g. back to a
    conservative tier), this test catches it so the prompt copy,
    Permission chip labels, and downstream test expectations can
    all be updated together."""
    assert DEFAULT_MAX_DEPTH == "names_types_labels_summary"


def test_default_policy_allows_names_types(tmp_path: Path):
    """No policy file → default ceiling is the permissive tier.
    A narrower request (names_types) succeeds and the response
    advertises the default ceiling."""
    set_cwd(tmp_path)
    csv = tmp_path / "d.csv"
    csv.write_text("x,y\n1,2\n3,4\n5,6\n")

    resp = _call_get_schema({"dataset": "d.csv", "depth": "names_types"})
    assert resp["status"] == "ok"
    assert resp["depth"] == "names_types"
    assert resp["policy_max_depth"] == DEFAULT_MAX_DEPTH


def test_default_policy_allows_names_only(tmp_path: Path):
    """A narrower-than-ceiling request also succeeds (policy is a
    ceiling, not a fixed value)."""
    set_cwd(tmp_path)
    csv = tmp_path / "d.csv"
    csv.write_text("x,y\n1,2\n3,4\n5,6\n")

    resp = _call_get_schema({"dataset": "d.csv", "depth": "names_only"})
    assert resp["status"] == "ok"
    assert resp["depth"] == "names_only"


def test_default_policy_allows_names_types_labels(tmp_path: Path):
    """Labels-tier is below the new permissive default and is allowed
    without an explicit policy."""
    set_cwd(tmp_path)
    csv = tmp_path / "d.csv"
    csv.write_text("x,y\n1,2\n3,4\n5,6\n")

    resp = _call_get_schema({"dataset": "d.csv", "depth": "names_types_labels"})
    assert resp["status"] == "ok"
    assert resp["depth"] == "names_types_labels"
    assert resp["policy_max_depth"] == DEFAULT_MAX_DEPTH


def test_default_policy_allows_summary(tmp_path: Path):
    """NA-count/distinct-count is the new default ceiling itself —
    allowed without any explicit policy."""
    set_cwd(tmp_path)
    csv = tmp_path / "d.csv"
    csv.write_text("x,y\n1,2\n3,4\n5,6\n")

    resp = _call_get_schema({
        "dataset": "d.csv",
        "depth": "names_types_labels_summary",
    })
    assert resp["status"] == "ok"
    assert resp["depth"] == "names_types_labels_summary"
    assert resp["policy_max_depth"] == "names_types_labels_summary"


def test_explicit_lower_policy_denies_above_ceiling(tmp_path: Path):
    """When the researcher explicitly lowers the ceiling for a
    dataset below the default, requests above that ceiling must be
    denied — this is the core privacy guarantee of the policy layer.
    (Replaces two earlier 'default denies X' tests that became
    irrelevant when the default was raised.)"""
    set_cwd(tmp_path)
    csv = tmp_path / "d.csv"
    csv.write_text("x,y\n1,2\n3,4\n5,6\n")

    save_policy(tmp_path, NoraPolicy(
        datasets={"d.csv": DatasetPolicy(
            max_depth="names_types",
            set_at="2026-04-21T00:00:00+00:00",
        )},
    ))

    resp = _call_get_schema({"dataset": "d.csv", "depth": "names_types_labels"})
    assert resp["status"] == "denied"
    assert resp["requested_depth"] == "names_types_labels"
    assert resp["policy_max_depth"] == "names_types"
    # Reason should mention it's an explicit ceiling (not the default)
    # so Claude can tell the researcher their own setting is what's
    # blocking, not Nora's baseline.
    assert "explicit" in resp["reason"].lower()


# ---------------------------------------------------------------------------
# Explicit policy — researcher opts into richer schema for a dataset
# ---------------------------------------------------------------------------

def test_explicit_policy_raises_ceiling(tmp_path: Path):
    """When the researcher has set `max_depth: names_types_labels` for
    this dataset in policy.json, that depth is now allowed."""
    set_cwd(tmp_path)
    csv = tmp_path / "d.csv"
    csv.write_text("x,y\n1,2\n3,4\n5,6\n")

    save_policy(tmp_path, NoraPolicy(
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

    save_policy(tmp_path, NoraPolicy(
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

    save_policy(tmp_path, NoraPolicy(
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


# ---------------------------------------------------------------------------
# Schema parse failures must not surface raw row content to the model.
# Pandas / pyreadstat parse-error messages can quote the offending row
# verbatim — forwarding that contradicts the schema tool's promise of
# never returning individual observation values.
# ---------------------------------------------------------------------------

def test_schema_parse_error_does_not_echo_row_content(tmp_path: Path) -> None:
    """A malformed CSV should yield a generic ``failed to read`` reason
    that names the exception class but not its message body."""
    set_cwd(tmp_path)
    # Pandas C-engine ParserError quotes the offending line in its
    # message. We craft a CSV whose header has 2 columns but one row
    # has a 5-column overflow including a recognizable secret value.
    secret_value = "PII_PATIENT_42_SSN_123_45_6789"
    bad = tmp_path / "broken.csv"
    bad.write_text(
        f"a,b\n1,2\n3,4,5,6,{secret_value}\n",
        encoding="utf-8",
    )
    resp = _call_get_schema({"dataset": "broken.csv", "depth": "names_types"})
    assert resp["status"] == "error"
    reason = resp.get("reason", "")
    # The exception class name CAN be in the response (helps the
    # model decide between a malformed-CSV vs malformed-Stata fix);
    # the row content MUST NOT be.
    assert secret_value not in reason
    # Generic phrasing tells the model the file is malformed without
    # quoting any data.
    assert "malformed" in reason.lower() or "corrupted" in reason.lower()
