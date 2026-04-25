"""Nora — researcher consent policy for schema exposure.

Schema depth — how much structural information Claude sees about a
dataset — was historically a code default (``names_types``). This
module makes it an explicit researcher policy, stored in
``<cwd>/.nora/policy.json``:

    {
      "version": 1,
      "default_max_depth": "names_types",
      "datasets": {
        "survey.csv": {
          "max_depth": "names_types_labels",
          "set_at": "2026-04-21T14:20:00+00:00"
        }
      }
    }

The policy sets a **ceiling** on what Claude can request for a given
dataset. ``get_schema(dataset, depth=...)`` consults the policy and
denies requests that exceed the ceiling. If a dataset has no entry,
``default_max_depth`` applies (conservative by default).

Depths, from most-private to most-permissive (each tier includes
everything above it):

- ``names_only`` — just the list of variable names.
- ``names_types`` — + a coarse type per variable (default).
- ``names_types_labels`` — + variable labels and value labels
  (the categorical-level dictionaries in `.dta` files).
- ``names_types_labels_summary`` — + per-variable NA count and
  distinct-value count for categoricals.

Never at any depth: raw observation values, min, max, median,
quantiles, or any frequency distributions. Those belong to
``request_data`` (with its own SDC rules) or ``submit_script``
(sanitized through the result pipeline).

Design notes:

- The policy is a **ceiling**, not a fixed value. Claude is free to
  request a lower depth than the ceiling — e.g., a dataset with a
  ``names_types_labels`` ceiling can still be queried at
  ``names_only`` if Claude doesn't need the labels for the task at
  hand. Nora enforces "at most", not "exactly".

- Interactive policy editing is wired up in both frontends:
  the terminal ``/policy`` slash-command (see
  ``app.py:_run_policy_wizard``) opens a dataset picker + depth
  menu; the web UI exposes a compact "Policy" chip next to the
  composer (see ``web/app.js:updatePolicyChip``) that unfurls a
  per-dataset dropdown. The JSON file is still the single source
  of truth — both UIs just read and write it — so a researcher
  comfortable editing it directly can still do that.

- Unknown or malformed entries fall back to the conservative
  default rather than raising. A broken policy file should not
  prevent the researcher from using Nora — the default is safe.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


# Depth identifiers. Must match the ``depth`` values in ``schema.py``
# and the documented enumeration in the ``get_schema`` tool help.
VALID_DEPTHS: tuple[str, ...] = (
    "names_only",
    "names_types",
    "names_types_labels",
    "names_types_labels_summary",
)

# Default schema depth for new datasets. NA count + distinct count is
# the richest non-leaky tier (below that, Claude is reasoning about
# variables with almost no metadata). Researchers can still dial it
# down per dataset via the Permission chip.
DEFAULT_MAX_DEPTH = "names_types_labels_summary"

# Map depth → rank so we can compare "is requested at most the ceiling".
_DEPTH_RANK: dict[str, int] = {d: i for i, d in enumerate(VALID_DEPTHS)}

POLICY_FILE = Path(".nora") / "policy.json"


@dataclass(frozen=True)
class DatasetPolicy:
    """Policy for a single dataset.

    ``set_at`` is an ISO-8601 UTC string marking when the researcher
    wrote this entry. Empty means the entry is inherited (default)
    rather than explicitly set.
    """
    max_depth: str = DEFAULT_MAX_DEPTH
    set_at: str = ""


@dataclass(frozen=True)
class NoraPolicy:
    """Top-level policy document.

    ``datasets`` keys are dataset filenames (not full paths) — the
    policy lives inside ``<cwd>/.nora/policy.json`` so paths are
    already relative to cwd.
    """
    version: int = 1
    default_max_depth: str = DEFAULT_MAX_DEPTH
    datasets: dict[str, DatasetPolicy] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Load / save
# ---------------------------------------------------------------------------

def policy_path(cwd: Path) -> Path:
    """Return the canonical path to the policy file for ``cwd``."""
    return cwd / POLICY_FILE


def load_policy(cwd: Path) -> NoraPolicy:
    """Load the policy for ``cwd``, or return a conservative default.

    Never raises. Malformed files (JSON errors, wrong shape, unknown
    depths) fall back to the default — a broken policy should not
    lock the researcher out of using Nora, and the fallback is
    safe.
    """
    p = policy_path(cwd)
    if not p.is_file():
        return NoraPolicy()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return NoraPolicy()
    if not isinstance(data, dict):
        return NoraPolicy()
    if data.get("version") != 1:
        # Future versions should have a migration path, but until one
        # exists, bail to default rather than misinterpret.
        return NoraPolicy()

    default_max = data.get("default_max_depth", DEFAULT_MAX_DEPTH)
    if default_max not in VALID_DEPTHS:
        default_max = DEFAULT_MAX_DEPTH

    datasets: dict[str, DatasetPolicy] = {}
    raw = data.get("datasets")
    if isinstance(raw, dict):
        for name, entry in raw.items():
            if not isinstance(name, str) or not isinstance(entry, dict):
                continue
            max_depth = entry.get("max_depth", DEFAULT_MAX_DEPTH)
            if max_depth not in VALID_DEPTHS:
                max_depth = DEFAULT_MAX_DEPTH
            set_at = entry.get("set_at", "")
            if not isinstance(set_at, str):
                set_at = ""
            datasets[name] = DatasetPolicy(max_depth=max_depth, set_at=set_at)

    return NoraPolicy(
        version=1, default_max_depth=default_max, datasets=datasets
    )


def save_policy(cwd: Path, policy: NoraPolicy) -> None:
    """Persist ``policy`` to ``<cwd>/.nora/policy.json``.

    Creates the ``.nora`` directory if it doesn't already exist.
    """
    p = policy_path(cwd)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "version": policy.version,
        "default_max_depth": policy.default_max_depth,
        "datasets": {
            name: {"max_depth": dp.max_depth, "set_at": dp.set_at}
            for name, dp in policy.datasets.items()
        },
    }
    p.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

def get_max_depth(policy: NoraPolicy, dataset_name: str) -> str:
    """Return the ceiling depth for ``dataset_name`` under ``policy``.

    Falls back to ``policy.default_max_depth`` if the dataset has no
    explicit entry.
    """
    entry = policy.datasets.get(dataset_name)
    if entry is None:
        return policy.default_max_depth
    return entry.max_depth


def depth_allowed(requested: str, ceiling: str) -> bool:
    """Return ``True`` iff ``requested`` depth is at or below ``ceiling``.

    Unknown depths are rejected (not the caller's bug to silently
    accept — callers should have validated against ``VALID_DEPTHS``
    before consulting the policy).
    """
    if requested not in _DEPTH_RANK or ceiling not in _DEPTH_RANK:
        return False
    return _DEPTH_RANK[requested] <= _DEPTH_RANK[ceiling]


def has_explicit_policy(policy: NoraPolicy, dataset_name: str) -> bool:
    """Return ``True`` iff ``policy`` has an explicit entry for the
    dataset (not inherited from ``default_max_depth``).
    """
    return dataset_name in policy.datasets
