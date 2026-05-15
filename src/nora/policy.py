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

- Interactive policy editing is wired up in both frontends through
  the same backend method (see ``ui.py:set_dataset_policy``); the
  web UI exposes a compact "Policy" chip next to the composer (see
  ``web/app.js:updatePolicyChip``) that unfurls a per-dataset
  dropdown. The JSON file is still the single source of truth —
  both UIs just read and write it — so a researcher comfortable
  editing it directly can still do that.

- A malformed policy file (truncated JSON, wrong shape, future
  schema version) fails closed: the in-memory policy returned has
  ``default_max_depth = "names_only"`` so schema access is denied
  until the file is repaired. A broken consent file is not a
  fresh-session signal — it's most often a partial write or editor
  mishap, and silently reverting to the rich default would expose
  metadata the researcher had previously restricted. Per-entry
  malformations clamp to the strictest tier on the same reasoning.
  Loading never raises.
"""

from __future__ import annotations

import json
import os
import tempfile
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

# Fail-closed depth used when the policy file exists but is broken
# (truncated JSON, wrong shape, future schema version, malformed
# entry). The premise: a researcher who wrote a policy.json had
# opinions about their data, and the most likely reason their file
# is unreadable is a partial write or an editor mishap — NOT a fresh
# session. Defaulting to the richest tier in that state would silently
# expose metadata they had previously restricted. Falling back to the
# strictest tier instead means schema queries get denied until the
# file is repaired, which is the right loudness for "your consent
# policy is unreadable."
FAIL_CLOSED_MAX_DEPTH = "names_only"

# Map depth → rank so we can compare "is requested at most the ceiling".
_DEPTH_RANK: dict[str, int] = {d: i for i, d in enumerate(VALID_DEPTHS)}

POLICY_FILE = Path(".nora") / "policy.json"


@dataclass(frozen=True)
class DatasetPolicy:
    """Policy for a single dataset.

    ``set_at`` is an ISO-8601 UTC string marking when the researcher
    wrote this entry. Empty means the entry is inherited (default)
    rather than explicitly set.

    ``non_disclosive_variables`` is the per-variable opt-in list:
    variables the researcher has explicitly judged safe to expose
    raw min / max / median for in descriptive results. Default empty
    — the conservative posture is "every variable's min/max could
    identify outlier individuals". Researchers add a variable here
    only after checking that its extremes don't single anyone out
    (e.g., ``age`` in years, ``year_of_birth``, ``education_years``;
    NOT ``salary`` or ``rare_diagnosis_code``).
    """
    max_depth: str = DEFAULT_MAX_DEPTH
    set_at: str = ""
    non_disclosive_variables: tuple[str, ...] = ()


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


def _fail_closed_policy() -> NoraPolicy:
    """Return the policy used when ``policy.json`` exists but is unreadable.

    Set ``default_max_depth`` to the strictest tier so a researcher
    who tightened their consent policy doesn't see it silently revert
    to the richest tier just because the JSON file got a stray
    character. Schema requests for unrestricted datasets will be
    denied until the file is repaired — the correct loudness for "we
    can't read your policy file."
    """
    return NoraPolicy(default_max_depth=FAIL_CLOSED_MAX_DEPTH)


def load_policy(cwd: Path) -> NoraPolicy:
    """Load the policy for ``cwd``.

    Never raises. Behavior in three regimes:

    - File absent: return the permissive in-memory default
      (``DEFAULT_MAX_DEPTH``). A fresh session has no expressed
      researcher opinion to honor, so the rich-by-default tier is
      correct — they can dial it down per-dataset later.

    - File present but unparseable / wrong shape / future version:
      fail closed. Return ``_fail_closed_policy()`` so schema access
      defaults to ``names_only`` until the file is repaired. The
      previous behavior fell back to the rich default here, which
      meant a partial write could silently expose metadata that the
      researcher had explicitly restricted.

    - File present and parseable: honor the document. Per-entry
      malformations (unknown ``max_depth``, missing fields) clamp the
      offending entry to the strictest tier rather than to the rich
      default, on the same fail-closed reasoning.
    """
    p = policy_path(cwd)
    if not p.is_file():
        return NoraPolicy()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return _fail_closed_policy()
    if not isinstance(data, dict):
        return _fail_closed_policy()
    if data.get("version") != 1:
        # Future versions should have a migration path, but until one
        # exists, fail closed rather than misinterpret. A version-skewed
        # file wasn't written for this code path; treating its absent
        # entries as "no opinion" would silently re-open access the
        # newer version may have tightened.
        return _fail_closed_policy()

    default_max = data.get("default_max_depth", DEFAULT_MAX_DEPTH)
    if default_max not in VALID_DEPTHS:
        default_max = FAIL_CLOSED_MAX_DEPTH

    datasets: dict[str, DatasetPolicy] = {}
    raw = data.get("datasets")
    if isinstance(raw, dict):
        for name, entry in raw.items():
            if not isinstance(name, str):
                # JSON keys are always strings, but be defensive.
                continue
            if not isinstance(entry, dict):
                # Malformed entry shape (e.g. ``"survey.csv": "names_only"``
                # written as shorthand without the wrapping dict).
                # Skipping the entry would silently fall back to
                # ``default_max_depth``, contradicting the "per-entry
                # malformations clamp to the strictest tier" rule in the
                # module docstring. Record a fail-closed entry so the
                # researcher's apparent intent — they wrote a key with
                # this dataset name — is honoured at the strictest tier.
                datasets[name] = DatasetPolicy(max_depth=FAIL_CLOSED_MAX_DEPTH)
                continue
            max_depth = entry.get("max_depth", FAIL_CLOSED_MAX_DEPTH)
            if max_depth not in VALID_DEPTHS:
                # The entry exists — researcher had an opinion — but
                # the depth string is unrecognised. Clamp to the
                # strictest tier rather than letting the file-wide
                # default take over (which could be more permissive
                # than what the researcher intended).
                max_depth = FAIL_CLOSED_MAX_DEPTH
            set_at = entry.get("set_at", "")
            if not isinstance(set_at, str):
                set_at = ""
            ndv_raw = entry.get("non_disclosive_variables", [])
            if isinstance(ndv_raw, list):
                non_disclosive = tuple(
                    str(v) for v in ndv_raw if isinstance(v, str) and v
                )
            else:
                non_disclosive = ()
            datasets[name] = DatasetPolicy(
                max_depth=max_depth,
                set_at=set_at,
                non_disclosive_variables=non_disclosive,
            )

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
            name: {
                "max_depth": dp.max_depth,
                "set_at": dp.set_at,
                # Only emit the field when populated — keeps the
                # default policy file tidy for datasets that don't
                # use the opt-in.
                **(
                    {"non_disclosive_variables": list(dp.non_disclosive_variables)}
                    if dp.non_disclosive_variables else {}
                ),
            }
            for name, dp in policy.datasets.items()
        },
    }
    # Write atomically via a sibling tmp file + ``os.replace``. Direct
    # ``write_text`` truncates ``policy.json`` and then streams bytes;
    # a crash mid-write (or a second writer that wins the race) leaves
    # a half-written file on disk, which the next ``load_policy`` reads
    # as malformed JSON and silently falls back to defaults — silently
    # widening every dataset's max_depth ceiling. Two known concurrent-
    # writer paths exist today: the web UI's policy editor and the
    # researcher TUI both call ``save_policy``, and the researcher can
    # have both open. ``NamedTemporaryFile(dir=p.parent)`` puts the tmp
    # file on the same filesystem so ``os.replace`` is a true atomic
    # rename (cross-fs ``os.replace`` falls back to copy-then-unlink,
    # which loses atomicity).
    payload = json.dumps(data, indent=2, sort_keys=True) + "\n"
    fd, tmp_name = tempfile.mkstemp(
        prefix=".policy.json.", suffix=".tmp", dir=p.parent,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, p)
    except Exception:
        # Best-effort cleanup of the orphan tmp file. ``os.replace``
        # consumes the source on success, so this only matters when
        # the write or rename failed.
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


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


def non_disclosive_for(
    policy: NoraPolicy, dataset_name: str
) -> frozenset[str]:
    """Return the set of variable names the researcher has explicitly
    marked as non-disclosive for ``dataset_name``.

    Empty set when the dataset has no explicit entry — the default
    posture is "no variable is opted-in to min/max disclosure".
    """
    entry = policy.datasets.get(dataset_name)
    if entry is None:
        return frozenset()
    return frozenset(entry.non_disclosive_variables)
