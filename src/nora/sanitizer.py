"""Nora — sanitizer + runtime-library contract.

This module is the data boundary's second gate. The researcher's script
runs in R/Stata (soon, step 4 executor), emits a structured payload via
the Nora runtime library, and Nora passes that payload through
`sanitize()` before anything reaches Claude. The payload Claude sees is
the sanitizer's output — nothing more.

This module is *also* the spec. The dataclasses + allowlists here ARE the
runtime-library contract. `nora$result(...)` in R and `nora_result_*`
in Stata must emit payloads whose shape matches one of these types.
Anything outside the allowlist is dropped; anything violating a hard SDC
rule causes the whole payload to be rejected.

**Coverage: six v0 types.**
- `linear_regression` — OLS and its variants (robust SE, clustered SE).
- `t_test` — one-sample, two-sample, paired, Welch.
- `descriptive` — per-variable n, mean, sd, missing count (never min/max/
  median/quantiles at v0; those leak individual observations).
- `frequency_table` — 1D level-count tables with primary + secondary
  suppression.
- `crosstab` — 2D contingency tables, no margins emitted (avoids the
  τ-ARGUS-class secondary-suppression problem).
- `magnitude_table` — sum/mean by group, with (1, k)-dominance.

**Design principles baked in here:**
1. **Allowlist fields, don't blocklist.** If a field isn't listed as
   allowed for a type, it's dropped. This is safer than trying to
   enumerate forbidden fields — the attacker's attack surface is the set
   of field names Nora has NOT thought about, and there are infinitely
   many such names.
2. **Hard vs soft rules.** Hard (minimum-N violations, structural size
   cap overflows) reject the whole payload. Soft (precision clamping,
   cell suppression, undeclared-key drops) transform in place and log.
3. **Transformations are logged.** Every modification the sanitizer makes
   is recorded in `SanitizerResult.transformations`, so the researcher
   can audit what Claude actually saw vs what the script produced.
4. **Structural size caps.** Each allowed dict / list field has an
   entry-count cap on top of the per-entry character cap enforced by
   `safe_key` / `safe_text`. The two limits together bound how much
   data a prompt-injected script can smuggle through an allowed field.
   See `_OLS_MAX_PREDICTORS`, `_FREQ_MAX_CELLS`, `_XTAB_MAX_CELLS`,
   `_MAGTAB_MAX_CELLS`.

**Known gaps / residual risks — documented so a maintainer doesn't have
to rediscover them.**

1. **`predictor_variables` has no upstream data authority.** The OLS
   coefficient-dict filter uses this list as the allowlist for inner
   keys (fix: dropped undeclared keys with a transformation log).
   But nothing ties `predictor_variables` itself back to the source
   dataset's columns — the script declares it, and the sanitizer
   trusts it. A prompt-injected script that emits both a fake
   `predictor_variables` and matching fake coefficient keys survives
   the filter. Closing this gap would require the sanitizer to read
   the source dataset's schema, which breaks its data-isolation
   invariant. A better fix lives in the runtime library: require a
   model object (not free-form args) and derive predictor_variables
   from the model's `xlevels`.

2. **`nora$result(...)` generic escape hatch.** The R and Stata
   runtime libraries expose a generic constructor that lets a script
   emit any supported-type payload with hand-crafted fields. Legit
   use case: bootstraps, custom statistics. But this is the path
   that bypasses gap #1 — without it, only `nora$from_lm(model)`
   would be available, and the model object would authoritatively
   define the variable names. Removing the escape hatch is a
   research-workflow trade-off and stays out of scope for the
   security pass.

3. **Data-derived names are still a channel.** Category / level names
   in frequency tables, crosstabs, and magnitude tables originate in
   the researcher's data (reading dataset values). A prompt-injected
   script can fabricate category names to encode bits — each name
   capped at 40 chars by `safe_key`, total cell count capped by the
   structural caps. Bandwidth is bounded (≈8 KB per payload) but not
   zero. To eliminate entirely, the runtime library would need to
   verify category names come from the actual data (same constraint
   as #1 — requires data access during payload construction).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from nora.sdc import (
    DOMINANCE_THRESHOLD_DEFAULT,
    MinimumNViolation,
    clamp_precision,
    clamp_precision_dict,
    dominance_fails,
    enforce_back_calc_safety,
    require_minimum_n,
    sigfigs_for_n,
    suppress_cells_below,
    suppression_marker,
)
from nora.text_safety import safe_key, safe_text


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SDCConfig:
    """Thresholds and policy knobs the sanitizer consults.

    Defaults are conservative; production use with sensitive data should
    probably tighten them. Step 5's SDC review should validate each
    choice against published guidance.
    """
    # Minimum total N for any regression / descriptive result. Below this,
    # the whole payload is rejected — no precision clamp makes it safe.
    min_n_regression: int = 10
    min_n_descriptive: int = 10
    # Minimum group size for t-tests (applied to both groups for
    # two-sample and to the single group for one-sample).
    min_n_ttest_group: int = 10
    # Cell-size threshold for frequency-table primary suppression.
    cell_suppression_threshold: int = 10
    # Dominance threshold for magnitude tables. If any single contributor
    # in a group accounts for more than this fraction of the cell's
    # total, the cell's value is suppressed. See sdc.py.
    dominance_threshold: float = DOMINANCE_THRESHOLD_DEFAULT
    # Per-variable opt-in for min / max in descriptive payloads.
    # Default empty: every variable's min/max is suppressed because
    # extremes can identify outlier individuals (one $1.5M salary,
    # one rare-disease respondent). The researcher can populate this
    # set via the per-dataset policy file (
    # ``.nora/policy.json`` ``non_disclosive_variables``) for
    # variables they've judged safe to expose raw — typical
    # examples: ``age`` in years, ``year_of_birth``, ``education_years``.
    non_disclosive_variables: frozenset[str] = field(
        default_factory=frozenset,
    )


DEFAULT_CONFIG = SDCConfig()


# ---------------------------------------------------------------------------
# Result envelope
# ---------------------------------------------------------------------------

@dataclass
class SanitizerResult:
    """Outcome of running a raw payload through the sanitizer.

    - `ok=True, sanitized={...}` on success; Claude is shown `sanitized`.
    - `ok=False, rejection_reason=str` on hard rule violation; the raw
      payload is discarded.
    - `transformations` is a list of human-readable strings describing
      every soft-rule transformation applied. The researcher's TUI shows
      these so the gap between "raw" and "what Claude saw" is auditable.
    """
    ok: bool
    analysis_type: str | None
    sanitized: dict[str, Any] | None = None
    rejection_reason: str | None = None
    transformations: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Per-type schemas — the runtime-library contract
# ---------------------------------------------------------------------------

# Each schema entry lists fields by disposition. "Allowed" keys pass
# through (possibly transformed by SDC); everything else is dropped.

# --- Linear regression ------------------------------------------------------

# Required fields — if missing, the payload is rejected as malformed.
# Required fields. ``r_squared`` is intentionally NOT required: the
# ``linear_regression`` bucket spans every regression-shape model the
# runtime emits via ``nora_result_regress`` / ``from_lm`` (OLS, logit,
# probit, Poisson, Cox PH, etc.), and many of those don't have an R²:
#
#   - Cox PH (``stcox``): partial-likelihood model, no R² at all.
#     Fit is reported via log-likelihood, LR chi-squared, and
#     concordance (Harrell's C, computed via ``estat concordance``).
#   - Logit / probit / Poisson (``logit``, ``probit``, ``poisson``):
#     populate ``e(r2_p)`` (McFadden pseudo-R²), not ``e(r2)``. Stata
#     helper now emits this as ``pseudo_r_squared``.
#
# The previous required-set demanded ``r_squared`` and rejected every
# Cox PH payload as malformed — researcher could fit a tenure-survival
# model but couldn't read it back. Coefficients + SEs + n + variable
# names are the structural minimum that makes a regression payload
# meaningful and verifiable; fit metrics are model-family-specific
# and pass through when present (see the allowed-numeric set below).
_OLS_REQUIRED: frozenset[str] = frozenset(
    ("type", "n", "coefficients", "standard_errors",
     "response_variable", "predictor_variables")
)

# Fields we allow through after SDC. Anything else is dropped silently
# and logged as a transformation. Notably absent: residuals, fitted,
# leverage, influence, cook_distance, data — all indexed by observation
# and therefore disclosive by construction.
_OLS_ALLOWED_NUMERIC_FIELDS: frozenset[str] = frozenset((
    "r_squared", "adj_r_squared", "f_statistic", "f_p_value",
    "residual_std_error",
    # Non-OLS fit metrics. All aggregate scalars derived from the
    # likelihood or design matrix; no per-observation leak. Allowed
    # so logit / probit / Poisson / Cox PH payloads carry their
    # natural fit indicators through to the model:
    #   - pseudo_r_squared: McFadden's R² for logit / probit / Poisson
    #     (``e(r2_p)`` in Stata; statsmodels ``.prsquared``).
    #   - log_likelihood: final log-likelihood. Standard for any MLE.
    #   - aic / bic: information criteria for model comparison.
    #   - chi_squared / chi_squared_p_value: LR or Wald omnibus test
    #     (``e(chi2)`` / ``e(p)`` in Stata MLE commands).
    #   - concordance: Harrell's C-index for survival models. Only
    #     populated after ``estat concordance`` (stcox doesn't put it
    #     in e() automatically).
    "pseudo_r_squared", "log_likelihood",
    "aic", "bic",
    "chi_squared", "chi_squared_p_value",
    "concordance",
    # Aggregate diagnostics. All scalars derived from the design
    # matrix or residual sum-of-squares — no per-observation leak.
    # ``condition_number`` is kappa(X), the ratio of largest to
    # smallest singular value of the design; high values flag
    # numerical instability and hidden collinearity.
    "condition_number",
))
_OLS_ALLOWED_INT_FIELDS: frozenset[str] = frozenset((
    "n", "degrees_of_freedom",
    # Survival-specific sample metadata. ``n`` for stcox is the number
    # of records (post-stset, can include split episodes per subject);
    # ``n_subjects`` and ``n_failures`` are what the researcher
    # actually reads off a Cox table — "324 subjects, 178 events" — so
    # they need to reach the model alongside the coefficients.
    "n_subjects", "n_failures",
))
_OLS_ALLOWED_STRING_FIELDS: frozenset[str] = frozenset((
    "type", "response_variable", "robust_se_type", "cluster_variable",
))
_OLS_ALLOWED_DICT_NUMERIC: frozenset[str] = frozenset((
    "coefficients", "standard_errors", "t_statistics", "p_values",
    # Variance-inflation factors, one per predictor. Cross-field
    # key validation (further down in _sanitize_linear_regression)
    # restricts the keys to declared predictor names + the
    # intercept aliases, so this dict can't be used to smuggle
    # arbitrary numeric channels.
    "vif",
))
_OLS_ALLOWED_LIST_STRING: frozenset[str] = frozenset((
    "predictor_variables",
))


# --- t-test ----------------------------------------------------------------

_TTEST_REQUIRED: frozenset[str] = frozenset(
    ("type", "test_type", "n1", "mean1", "t_statistic", "p_value")
)

_TTEST_VALID_SUBTYPES: frozenset[str] = frozenset(
    ("one_sample", "two_sample", "paired", "welch")
)

_TTEST_ALLOWED_NUMERIC_FIELDS: frozenset[str] = frozenset((
    "mean1", "mean2", "sd1", "sd2", "mean_difference",
    "t_statistic", "p_value", "degrees_of_freedom",
))
_TTEST_ALLOWED_INT_FIELDS: frozenset[str] = frozenset(("n1", "n2"))
_TTEST_ALLOWED_STRING_FIELDS: frozenset[str] = frozenset(
    ("type", "test_type", "alternative")
)
_TTEST_ALLOWED_LIST_NUMERIC: frozenset[str] = frozenset(("confidence_interval",))


# --- Descriptive statistics ------------------------------------------------

_DESC_REQUIRED: frozenset[str] = frozenset(
    ("type", "variable", "n", "mean", "sd", "missing_count")
)

# Default-allowed numerics: mean and sd — pure aggregates, never
# disclosive at row level. min / max are individual observations and
# are gated by ``SDCConfig.non_disclosive_variables`` (researcher-side
# per-variable opt-in via ``.nora/policy.json``); when the variable
# is in that set, ``min_value`` and ``max_value`` are also accepted.
# Median / quartiles remain forbidden in this payload type — use
# ``request_data`` ``quartiles`` (which omits the median exactly
# because at odd N it IS an individual observation).
_DESC_ALLOWED_NUMERIC_FIELDS: frozenset[str] = frozenset(("mean", "sd"))
_DESC_OPTIONAL_NUMERIC_FIELDS: frozenset[str] = frozenset(
    ("min_value", "max_value")
)
_DESC_ALLOWED_INT_FIELDS: frozenset[str] = frozenset(
    ("n", "missing_count", "distinct_count")
)
_DESC_ALLOWED_STRING_FIELDS: frozenset[str] = frozenset(("type", "variable"))


# --- Frequency table (1D only at v0) ---------------------------------------

_FREQ_REQUIRED: frozenset[str] = frozenset(
    ("type", "variable", "counts", "n", "missing_count")
)
_FREQ_ALLOWED_INT_FIELDS: frozenset[str] = frozenset(("n", "missing_count"))
_FREQ_ALLOWED_STRING_FIELDS: frozenset[str] = frozenset(("type", "variable"))


# --- Magnitude table (sum / mean of a numeric variable by group) -----------

# Required fields. Each cell is itself a dict with `value` (the
# aggregate), `n` (group size), and `max_share` (the dominance metric
# the runtime library computed on raw values — used internally for
# suppression, NEVER emitted to Claude).
_MAGTAB_REQUIRED: frozenset[str] = frozenset(
    ("type", "row_variable", "value_variable", "aggregation", "cells")
)
_MAGTAB_ALLOWED_STRING_FIELDS: frozenset[str] = frozenset(
    ("type", "row_variable", "value_variable", "aggregation")
)
# Aggregation kinds we understand. The runtime library should only emit
# these; anything else is rejected as a schema violation.
_MAGTAB_VALID_AGGREGATIONS: frozenset[str] = frozenset(("sum", "mean"))

# Helper-provenance marker. Typed runtime helpers (Python's
# ``from_magnitude_table``, R's ``nora$from_magnitude_table``, Stata's
# ``nora_result_magnitude``) stamp this on payloads they emit. The
# generic runtime ``result()`` API strips the field from caller-passed
# kwargs, so a script can't forge it through the public entry point.
# Required for ``magnitude_table`` because cell-level ``max_share`` is
# consulted-only and stripped: without proof the metric came from
# raw-data computation, a script could publish a dominance-violating
# value with a forged ``max_share=0`` and skip the (1, k)-dominance
# gate. The token gate alone doesn't catch this — token validation
# proves the line passed through *some* runtime path (including the
# generic ``result()`` API), not specifically the typed helper. Same
# "raise the bar, not absolute guarantee" posture as the token: a
# script that hand-writes JSON to NORA_RESULT_PATH (after reading
# nora._RUN_TOKEN) can still forge the marker, but trivial misuse
# of ``nora.result(type="magnitude_table", cells={..., "max_share":
# 0})`` is rejected.
_HELPER_PROVENANCE_FIELD = "_via_helper"
_MAGTAB_HELPER_VALUE = "from_magnitude_table"


# --- Structural size caps --------------------------------------------------
# Hard limits on the number of top-level entries each payload type can
# carry. Primarily a defense-in-depth measure: the per-entry key cap
# (40 chars via safe_key) plus these entry-count caps bound how much
# data a prompt-injected script can smuggle through an allowed field.
#
# Numbers picked to comfortably accommodate legitimate research
# output and reject anything that looks engineered — a regression
# with 60 predictors isn't interpretable statistics, a frequency
# table with 300 distinct levels isn't a useful summary.
_OLS_MAX_PREDICTORS = 50
_FREQ_MAX_CELLS = 200
_XTAB_MAX_CELLS = 2500          # allows up to ~50 × 50
_MAGTAB_MAX_CELLS = 200
_CORR_MAX_VARIABLES = 30        # NxN ⇒ up to 900 entries before clamping


# --- Correlation matrix ----------------------------------------------------

# Pairwise correlations among a list of numeric variables. Pure
# aggregate (sums of products / N), no per-row leak — but reject at
# low N where a near-perfect correlation is just "the three points
# are collinear" rather than a population property.
_CORR_REQUIRED: frozenset[str] = frozenset(
    ("type", "n", "variables", "correlations")
)
_CORR_ALLOWED_INT_FIELDS: frozenset[str] = frozenset(("n", "missing_count"))
_CORR_ALLOWED_STRING_FIELDS: frozenset[str] = frozenset(
    ("type", "method", "label")
)
_CORR_ALLOWED_LIST_STRING: frozenset[str] = frozenset(("variables",))
# Allowed correlation types — Pearson is the linear default; Spearman /
# Kendall handle rank-based and ordinal data. Anything else is
# rejected as a schema violation rather than silently coerced.
_CORR_VALID_METHODS: frozenset[str] = frozenset(
    ("pearson", "spearman", "kendall")
)


# --- Crosstab (2D frequency table, no margins emitted) ---------------------

# Required fields. Note the deliberate absence of `n` — crosstabs do NOT
# expose a grand total at v0. Without margins (row totals, column totals,
# grand total), primary cell suppression alone is sufficient to prevent
# back-calculation; with margins, 2D requires LP-based secondary
# suppression (τ-ARGUS territory, deferred).
_XTAB_REQUIRED: frozenset[str] = frozenset(
    ("type", "row_variable", "col_variable", "counts")
)
# Margin-ish fields forbidden by name. `_collect_allowed` drops anything
# not on the allowlist, but naming these here is the documentation
# anchor: these are the fields that break the "no margins" invariant.
_XTAB_FORBIDDEN_MARGIN_FIELDS: frozenset[str] = frozenset((
    "n", "grand_total", "row_totals", "column_totals", "col_totals",
    "marginals",
))
_XTAB_ALLOWED_STRING_FIELDS: frozenset[str] = frozenset(
    ("type", "row_variable", "col_variable")
)
# `missing_count` is kept optional — it's not a disclosive margin (it
# refers to observations with a missing value on one or both axes, which
# is a pipeline characteristic, not a cell-identifying quantity). The
# counts themselves are handled specially below.
_XTAB_ALLOWED_INT_FIELDS: frozenset[str] = frozenset(("missing_count",))


# ---------------------------------------------------------------------------
# Top-level dispatch
# ---------------------------------------------------------------------------

def sanitize(raw: dict[str, Any], config: SDCConfig = DEFAULT_CONFIG) -> SanitizerResult:
    """Validate a raw analysis payload and apply SDC rules.

    Dispatches on `raw["type"]`. Unknown types are rejected as malformed.
    All rejections carry a machine-readable reason; all transformations
    are logged.
    """
    if not isinstance(raw, dict):
        return SanitizerResult(
            ok=False, analysis_type=None,
            rejection_reason=f"payload must be a dict, got {type(raw).__name__}",
        )

    analysis_type = raw.get("type")
    if not isinstance(analysis_type, str):
        return SanitizerResult(
            ok=False, analysis_type=None,
            rejection_reason="payload missing 'type' field or type is not a string",
        )

    handler = _HANDLERS.get(analysis_type)
    if handler is None:
        # The script controls ``raw["type"]``; an adversarial payload
        # could set it to a raw cell value (a row, a cell, a JSON
        # blob) and trigger this branch to smuggle the value out
        # through ``rejection_reason``, which submit_script forwards
        # into both the inline result and the persisted diagnostic
        # row. Echo the type only after ``safe_key`` (40-char cap,
        # control-char strip) so the leak channel is bounded to a
        # short token, and store the same sanitized form in
        # ``analysis_type`` so the diagnostic row never carries the
        # raw value either.
        safe_type = safe_key(analysis_type)
        return SanitizerResult(
            ok=False, analysis_type=safe_type,
            rejection_reason=(
                f"unknown analysis type {safe_type!r}. Supported in v0: "
                f"{sorted(_HANDLERS.keys())}"
            ),
        )
    return handler(raw, config)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _require_fields(
    raw: dict[str, Any], required: frozenset[str], analysis_type: str
) -> str | None:
    """Return a rejection reason if required fields are missing, else None."""
    missing = required - raw.keys()
    if missing:
        return (
            f"{analysis_type} payload missing required fields: "
            f"{sorted(missing)}"
        )
    return None


def _require_after_filter(
    out: dict[str, Any],
    required: frozenset[str],
    analysis_type: str,
    *,
    pre_validated: frozenset[str] = frozenset(),
) -> str | None:
    """Re-check required fields after ``_collect_allowed`` runs.

    ``_require_fields`` only checks that required keys are present in
    the raw payload; ``_collect_allowed`` then DROPS any field whose
    type doesn't match the schema (e.g. ``coefficients`` shipped as a
    string). The result is an ``ok=True`` payload missing structural
    fields — the model thinks the analysis succeeded with garbage. So
    callers re-check required fields against ``out`` after collection.

    ``pre_validated`` lists fields the caller already gates explicitly
    BEFORE ``_collect_allowed`` (e.g. ``n`` for OLS, ``cells`` for
    magnitude_table) — those are never inserted into ``out`` by
    ``_collect_allowed`` itself, so excluding them avoids spurious
    rejection. The handler is expected to assemble those fields
    elsewhere in ``out`` if they survive their pre-validation.
    """
    needs_check = required - pre_validated - {"type"}
    missing = needs_check - out.keys()
    if missing:
        return (
            f"{analysis_type} payload required field(s) had wrong "
            f"type and were dropped during sanitization: "
            f"{sorted(missing)}"
        )
    return None


def _is_finite_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


# Hard cap on transformation-log entries a single ``_collect_allowed``
# invocation may emit. The transformations list rides into the
# tool_result payload and is read by the model, so any loop that
# appends one entry per dropped key is a model-visible channel whose
# size is controlled by the payload author. Two such loops exist in
# this function: the outer ``for k in raw.items()`` (one entry per
# unknown field) and the inner ``for kk in v.items()`` inside the
# ``dict_numeric`` branch (one entry per malformed nested key).
# Without a cap, a payload with thousands of arbitrary keys would
# fill the model's context with megabytes of "dropped …" lines.
# 50 is enough that realistic shape-mismatch debugging is still
# legible; the trailing summary tells the model how many drops it
# isn't seeing in detail.
_COLLECT_ALLOWED_LOG_CAP = 50


def _collect_allowed(
    raw: dict[str, Any],
    *,
    numeric: frozenset[str] = frozenset(),
    integer: frozenset[str] = frozenset(),
    string: frozenset[str] = frozenset(),
    dict_numeric: frozenset[str] = frozenset(),
    list_string: frozenset[str] = frozenset(),
    list_numeric: frozenset[str] = frozenset(),
    transformations: list[str] | None = None,
) -> dict[str, Any]:
    """Filter a raw payload to just the allowed fields, validating shapes.

    Fields with wrong types are dropped with a log entry (not an error)
    — we treat malformed values as indistinguishable from untrusted
    input. Unknown fields are dropped silently (they're explicitly not
    in the allowlist, so logging every unknown field would be noisy).

    ``transformations`` is appended to in-place if passed. Field names
    and dict keys that originate in the researcher's data are passed
    through ``safe_key`` before being interpolated into log messages or
    returned as output keys — otherwise a maliciously-named variable
    could inject text into Claude's context through the transformations
    log or through a coefficient dict key.

    Per-invocation cap: at most ``_COLLECT_ALLOWED_LOG_CAP`` drop
    entries land in ``transformations``; surplus drops are summarised
    in a single tail line. See ``_COLLECT_ALLOWED_LOG_CAP`` for why.
    """
    t = transformations if transformations is not None else []
    out: dict[str, Any] = {}
    allowed = numeric | integer | string | dict_numeric | list_string | list_numeric

    # Track how many entries this call has emitted, separately from
    # ``len(t)`` — the caller may have prefilled ``t`` with notes from
    # earlier sanitiser stages, and we only want to bound THIS call's
    # contribution. ``surplus`` is appended once at the end as a
    # human-readable tail.
    emitted = 0
    surplus = 0

    def _log(msg: str) -> None:
        nonlocal emitted, surplus
        if emitted < _COLLECT_ALLOWED_LOG_CAP:
            t.append(msg)
            emitted += 1
        else:
            surplus += 1

    # Count rather than name unknown fields. Field names in ``raw``
    # but outside ``allowed`` are data-derived (an attacker-authored
    # script can encode raw row values as JSON field names and read
    # them back through ``transformations``). The single summary line
    # below emits the count only — the per-row store keeps the raw
    # payload for researcher audit, so naming the fields here was
    # exfil-without-benefit.
    unknown_field_count = 0
    # Same pattern for dict_numeric inner keys: non-string keys and
    # non-finite values inside an allowed dict-of-numeric field both
    # carry caller-controlled bytes if echoed. We collapse them into
    # one summary line per parent field.
    dict_drops: dict[str, int] = {}

    for k, v in raw.items():
        if k not in allowed:
            unknown_field_count += 1
            continue
        if k in integer:
            if not isinstance(v, int) or isinstance(v, bool):
                _log(f"dropped {k!r}: expected int, got {type(v).__name__}")
                continue
            out[k] = v
        elif k in numeric:
            if not _is_finite_number(v):
                _log(f"dropped {k!r}: not a finite number")
                continue
            out[k] = float(v)
        elif k in string:
            if not isinstance(v, str):
                _log(f"dropped {k!r}: expected str, got {type(v).__name__}")
                continue
            cleaned = safe_text(v)
            if cleaned != v:
                _log(f"sanitized scalar string field {k!r}")
            out[k] = cleaned
        elif k in dict_numeric:
            if not isinstance(v, dict):
                _log(f"dropped {k!r}: expected dict, got {type(v).__name__}")
                continue
            clean: dict[str, float] = {}
            inner_drops = 0
            inner_collisions = 0
            for kk, vv in v.items():
                if not isinstance(kk, str):
                    inner_drops += 1
                    continue
                if not _is_finite_number(vv):
                    inner_drops += 1
                    continue
                # safe_key on the key — e.g. coefficient names, which
                # originate in the data's variable names, cross to
                # Claude. Two raw keys that ``safe_key`` collapses to
                # the same form (newline → space, 40-char prefix
                # share) MUST NOT silently overwrite — the second
                # value would replace the first and the model would
                # see one value labelled by an ambiguous key. Drop
                # the duplicate; track a count for the transformation
                # log so the researcher can audit. The vcov path in
                # _sanitize_linear_regression detects collisions in
                # the same shape.
                safe_kk = safe_key(kk)
                if safe_kk in clean:
                    inner_collisions += 1
                    continue
                clean[safe_kk] = float(vv)
            if inner_drops:
                dict_drops[k] = inner_drops
            if inner_collisions:
                # Separate counter so the cause is auditable. Names
                # withheld — collisions identify pairs of data-derived
                # raw keys.
                _log(
                    f"dropped {inner_collisions} duplicate inner "
                    f"key(s) from {k!r} after sanitization "
                    f"(colliding names withheld)"
                )
            out[k] = clean
        elif k in list_string:
            if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
                _log(f"dropped {k!r}: not a list[str]")
                continue
            # Each string element originates in the data — sanitize all.
            out[k] = [safe_key(x) for x in v]
        elif k in list_numeric:
            if not isinstance(v, list) or not all(_is_finite_number(x) for x in v):
                _log(f"dropped {k!r}: not a list of finite numbers")
                continue
            out[k] = [float(x) for x in v]

    if unknown_field_count:
        # Aggregate count only — the field names themselves were
        # data-derived and are deliberately not echoed. Researchers
        # who need to audit the dropped names can read the raw payload
        # from the per-row store; the model only sees the count.
        _log(
            f"dropped {unknown_field_count} unknown/forbidden "
            f"top-level field(s) (names withheld)"
        )
    if dict_drops:
        # Per-parent count for malformed dict-of-numeric inner entries.
        # Both non-string keys and non-finite values are collapsed —
        # the inner key is data-derived (a coefficient / VIF / vcov
        # row name) and a non-string key would otherwise be coerced to
        # str and echoed back.
        for parent, n_dropped in sorted(dict_drops.items()):
            _log(
                f"dropped {n_dropped} malformed entry(ies) from "
                f"{parent!r} (inner keys/values withheld)"
            )
    if surplus:
        # Single line that bounds the total log size at
        # ``_COLLECT_ALLOWED_LOG_CAP + 1`` regardless of payload size.
        t.append(
            f"… and {surplus} more drops omitted from this payload's log "
            f"(cap {_COLLECT_ALLOWED_LOG_CAP})"
        )
    return out


def _coarsen_small_missing_count(
    out: dict[str, Any],
    transformations: list[str],
    config: SDCConfig,
) -> None:
    """Replace ``missing_count`` with the suppression marker when its
    exact value is itself disclosive (``0 < missing_count < threshold``).

    An exact small missingness count identifies the few records whose
    value on this variable is missing — combined with other variables
    it supports re-identification ("the one patient who declined to
    answer income"). Same threshold as cell suppression so the rule
    is uniform across payload kinds. Zero is left as 0 (no
    missingness, nothing to suppress).

    Mutates ``out`` in place. Appends one log line if coarsening
    fired. The schema-side ``request_data(na_count)`` path already
    enforces this gate symmetrically; this helper closes the gap on
    the stored-result path (descriptive / frequency_table /
    correlation_matrix / crosstab) where ``missing_count`` is
    allowlisted but was previously forwarded verbatim.
    """
    threshold = config.cell_suppression_threshold
    miss_raw = out.get("missing_count")
    if isinstance(miss_raw, int) and 0 < miss_raw < threshold:
        out["missing_count"] = suppression_marker(threshold)
        transformations.append(
            f"coarsened missing_count to {suppression_marker(threshold)} "
            f"(exact small missingness counts are themselves disclosive)"
        )


def _coarsen_small_cox_counts(
    out: dict[str, Any],
    transformations: list[str],
    config: SDCConfig,
) -> None:
    """Replace ``n_failures`` / ``n_subjects`` with the suppression
    marker when their exact values fall below ``cell_suppression_threshold``.

    Survival-specific Cox fits commonly report "n records / n subjects /
    n failures" together. ``n`` (top-level) is already gated by
    ``require_minimum_n(config.min_n_regression)`` upstream — typically
    a much higher floor than the cell-suppression threshold — so a
    Cox payload that survives to this point has at least
    ``min_n_regression`` records. But ``n_failures`` is a different
    quantity: it counts events (deaths, conversions, churn) and on a
    rare-outcome study can be tiny even when ``n`` is in the thousands.
    "324 subjects, 3 events" identifies those 3 specific individuals
    just as surely as a frequency_table cell with count 3 would.

    Apply the same threshold as cell suppression / missing_count for
    a uniform disclosure rule. Zero is left as 0 (no events — no
    individual to identify; same posture as ``_coarsen_small_missing_count``).
    ``n_subjects`` is coarsened for symmetry: in survival data it CAN
    differ from ``n`` (records can split into multiple per-subject
    episodes via ``stset``) and an analyst studying a panel of e.g.
    very rare patient subgroups could have a small ``n_subjects``
    even with many records.
    """
    threshold = config.cell_suppression_threshold
    for field in ("n_failures", "n_subjects"):
        raw = out.get(field)
        if isinstance(raw, int) and 0 < raw < threshold:
            out[field] = suppression_marker(threshold)
            transformations.append(
                f"coarsened {field} to {suppression_marker(threshold)} "
                f"(exact small Cox-style event/subject counts are "
                f"themselves disclosive)"
            )


# ---------------------------------------------------------------------------
# Linear regression sanitizer
# ---------------------------------------------------------------------------

def _sanitize_linear_regression(
    raw: dict[str, Any], config: SDCConfig
) -> SanitizerResult:
    missing_reason = _require_fields(raw, _OLS_REQUIRED, "linear_regression")
    if missing_reason:
        return SanitizerResult(
            ok=False, analysis_type="linear_regression",
            rejection_reason=missing_reason,
        )

    n_raw = raw.get("n")
    if not isinstance(n_raw, int) or isinstance(n_raw, bool) or n_raw < 0:
        # Don't echo ``n_raw`` itself — a malicious script could set
        # ``n`` to a raw cell value to smuggle it out via this
        # rejection_reason (which submit_script forwards into both
        # the inline result and the persisted diagnostic row). The
        # type name leaks zero bits of payload content.
        return SanitizerResult(
            ok=False, analysis_type="linear_regression",
            rejection_reason=(
                f"n must be a non-negative int, got {type(n_raw).__name__}"
            ),
        )

    try:
        require_minimum_n(n_raw, config.min_n_regression, "n")
    except MinimumNViolation as e:
        return SanitizerResult(
            ok=False, analysis_type="linear_regression",
            rejection_reason=str(e),
        )

    # Structural size cap on predictor_variables. Each predictor name
    # goes through safe_key (40-char cap) already; this check bounds
    # the total number of names, so the data channel available through
    # this list is bounded in both dimensions. Also catches
    # accidentally-huge models that wouldn't be interpretable research
    # output anyway.
    raw_predictors = raw.get("predictor_variables")
    if isinstance(raw_predictors, list) and len(raw_predictors) > _OLS_MAX_PREDICTORS:
        return SanitizerResult(
            ok=False, analysis_type="linear_regression",
            rejection_reason=(
                f"predictor_variables has {len(raw_predictors)} entries; "
                f"the structural cap is {_OLS_MAX_PREDICTORS}. A regression "
                f"with that many predictors isn't interpretable output — "
                f"rejected as probable adversarial payload."
            ),
        )

    transformations: list[str] = []
    out = _collect_allowed(
        raw,
        numeric=_OLS_ALLOWED_NUMERIC_FIELDS,
        integer=_OLS_ALLOWED_INT_FIELDS,
        string=_OLS_ALLOWED_STRING_FIELDS,
        dict_numeric=_OLS_ALLOWED_DICT_NUMERIC,
        list_string=_OLS_ALLOWED_LIST_STRING,
        transformations=transformations,
    )

    # Re-check required fields after type filtering. ``_require_fields``
    # only verifies key presence in ``raw``; ``_collect_allowed`` then
    # silently drops any required field whose type doesn't match
    # (e.g. ``coefficients`` shipped as a string). Without this gate,
    # a wrong-typed ``coefficients`` / ``standard_errors`` /
    # ``response_variable`` survives to ``ok=True`` with the field
    # absent. ``n`` is already pre-validated above.
    missing_after_filter = _require_after_filter(
        out, _OLS_REQUIRED, "linear_regression",
        pre_validated=frozenset(("n",)),
    )
    if missing_after_filter:
        return SanitizerResult(
            ok=False, analysis_type="linear_regression",
            rejection_reason=missing_after_filter,
        )

    # Disclosure-control gate: refuse formula-categorical coefficient
    # names. statsmodels / patsy formula fits encode raw categorical
    # levels into coefficient names using contrast markers — e.g.
    # ``C(diagnosis)[T.diabetes]`` for treatment contrasts, with
    # ``[Sum.`` / ``[Diff.`` / ``[Helmert.`` for other coding schemes.
    # ``safe_key`` only enforces prompt-injection bounds (length,
    # control chars); it doesn't recognise the level value as data,
    # so a script can ``ols('y ~ C(secret)', data=df).fit()`` and
    # leak each unique level of ``secret`` through the regression
    # coefficient / SE / p-value keys (and through the predictor
    # list itself).
    #
    # Force the script to expand dummies explicitly via
    # ``pd.get_dummies(...)`` and pass them as named columns. That
    # moves the level-naming responsibility into the script proper
    # (where it's visible in the code the researcher reviews) and
    # the resulting predictor names go through ``predictor_variables``
    # like any other column name — same disclosure surface as a
    # normal regression on already-encoded data.
    _CATEGORICAL_CONTRAST_RE = re.compile(r"\[[A-Za-z]+\.")
    suspicious_keys: set[str] = set()
    for name in out.get("predictor_variables") or []:
        if isinstance(name, str) and _CATEGORICAL_CONTRAST_RE.search(name):
            suspicious_keys.add(name)
    for dict_field in _OLS_ALLOWED_DICT_NUMERIC:
        d = out.get(dict_field)
        if not isinstance(d, dict):
            continue
        for k in d:
            if isinstance(k, str) and _CATEGORICAL_CONTRAST_RE.search(k):
                suspicious_keys.add(k)
    if suspicious_keys:
        return SanitizerResult(
            ok=False, analysis_type="linear_regression",
            rejection_reason=(
                "regression payload contains formula-categorical "
                "coefficient name(s) — patsy / statsmodels formula "
                "fits embed raw categorical level values into "
                "coefficient names (e.g. ``C(var)[T.level]``), "
                "which would surface those level values without "
                "going through the frequency-table cell suppression "
                "policy. Expand categorical predictors into named "
                "indicator columns before fitting (pandas: "
                "``pd.get_dummies(df, columns=[...], drop_first=True)``; "
                "R: build a model matrix with ``model.matrix`` then "
                "fit on the resulting numeric columns) so the "
                "predictor names you emit are plain identifiers."
            ),
        )

    # Cross-field integrity: each coefficient-dict key must name a
    # declared predictor OR the intercept. Without this, a prompt-
    # injected Claude can smuggle arbitrary numbers out by emitting
    # e.g. ``coefficients: {leak_bit_0: 0.001, leak_bit_1: 0.002, …}``
    # — the inner dict accepts any well-formed key through
    # ``_collect_allowed``, and precision-clamping just rounds the
    # smuggled values, it doesn't reject them.
    #
    # R reports the intercept as ``(Intercept)``; Stata as ``_cons``.
    # We accept both plus a permissive ``intercept`` form in case
    # the runtime library normalizes. Any other key not declared in
    # ``predictor_variables`` is dropped with a transformation log
    # entry so the researcher can see what got stripped.
    declared_predictors = set(out.get("predictor_variables") or [])
    # Intercept aliases each runtime emits. R's lm() reports
    # "(Intercept)"; statsmodels formula fits report "Intercept";
    # statsmodels ``add_constant(X)`` reports "const"; Stata reports
    # "_cons". The lowercase "intercept" form is a permissive fallback
    # in case a future runtime normalizes naming.
    allowed_coefficient_keys = declared_predictors | {
        "(Intercept)", "_cons", "intercept", "Intercept", "const",
    }
    for dict_field in _OLS_ALLOWED_DICT_NUMERIC:
        if dict_field not in out:
            continue
        d = out[dict_field]
        if not isinstance(d, dict):
            continue
        kept: dict[str, float] = {}
        dropped: list[str] = []
        for k, v in d.items():
            if k in allowed_coefficient_keys:
                kept[k] = v
            else:
                dropped.append(k)
        if dropped:
            # Names withheld by design: ``dropped`` contains keys
            # the caller-authored script put into the result dict
            # (coefficient / SE / t / p / vif keys originate from
            # the regression's design matrix column names, which
            # the script chooses freely). Echoing those names back
            # gives a script that ran on raw data ~30 strings ×
            # ~40 chars per submit_script call of attacker-chosen
            # content through the transformations log — a covert
            # channel for small high-value values (numeric IDs,
            # ZIP codes, salaries) that's both well-bounded enough
            # to fit in the safe_key cap and far easier than the
            # legitimate model-context channels. ``_collect_allowed``
            # uses this same "names withheld" treatment for unknown
            # top-level fields and for malformed inner dict values;
            # this matches it.
            transformations.append(
                f"dropped {len(dropped)} undeclared key(s) from "
                f"{dict_field!r} (names withheld — keys are caller-"
                f"controlled and could carry raw data bytes)"
            )
        out[dict_field] = kept

    # Variance-covariance matrix (vcov). Optional, dict-of-dict-of-
    # numeric keyed on the same coefficient names. Pure aggregate from
    # the design (sigma^2 * (X'X)^-1); the diagonals are SE^2 and the
    # off-diagonals enable Wald tests / joint hypothesis testing /
    # linear-combination CIs the model can compute itself. Each row
    # AND column key must reference a declared predictor (or
    # intercept alias); alien keys are dropped with the same defense
    # used on coefficients above.
    #
    # Sanitize each row/col key through ``safe_key`` BEFORE validation.
    # ``allowed_coefficient_keys`` was derived from already-sanitized
    # coefficient names (``out["predictor_variables"]`` was passed
    # through ``safe_key`` in ``_collect_allowed``). The raw vcov
    # keys haven't been sanitized yet, so any coefficient name that
    # changes under ``safe_key`` (length > 40, embedded control
    # chars, newlines) would compare unequal to its sanitized
    # counterpart and the entire row/col would be dropped as
    # "undeclared" — losing the matrix while keeping the coefficient
    # / SE entries intact (those went through the dict_numeric
    # branch in ``_collect_allowed``, which sanitizes keys). Apply
    # ``safe_key`` here so the comparison is apples-to-apples, and
    # detect post-sanitization collisions (two raw keys that map to
    # the same cleaned form) explicitly so a sanitized matrix
    # doesn't silently overwrite cells.
    raw_vcov = raw.get("vcov")
    if isinstance(raw_vcov, dict):
        sanitized_vcov: dict[str, dict[str, float]] = {}
        dropped_vcov: list[str] = []
        collisions: list[str] = []
        for row_key, row_value in raw_vcov.items():
            if not isinstance(row_key, str):
                dropped_vcov.append(f"row {row_key!r} (non-string key)")
                continue
            safe_row = safe_key(row_key)
            if safe_row not in allowed_coefficient_keys:
                dropped_vcov.append(f"row {safe_row!r}")
                continue
            if not isinstance(row_value, dict):
                dropped_vcov.append(f"row {safe_row!r} (non-dict)")
                continue
            sanitized_row: dict[str, float] = {}
            for col_key, val in row_value.items():
                if not isinstance(col_key, str):
                    dropped_vcov.append(
                        f"{safe_row}.{col_key!r} (non-string key)"
                    )
                    continue
                safe_col = safe_key(col_key)
                if safe_col not in allowed_coefficient_keys:
                    dropped_vcov.append(f"{safe_row}.{safe_col}")
                    continue
                if not _is_finite_number(val):
                    continue
                if safe_col in sanitized_row:
                    # Two raw column keys cleaned to the same name.
                    # Don't silently overwrite the earlier value;
                    # log the collision and skip the duplicate so
                    # the matrix degrades safely (the model sees the
                    # transformation log entry and can decide
                    # whether to re-fit with disambiguated names).
                    collisions.append(f"{safe_row}.{safe_col}")
                    continue
                sanitized_row[safe_col] = float(val)
            if sanitized_row:
                if safe_row in sanitized_vcov:
                    collisions.append(f"row {safe_row}")
                    continue
                sanitized_vcov[safe_row] = sanitized_row
        if dropped_vcov:
            # Names withheld for the same reason as the
            # ``dict_numeric`` log entry above — vcov row / col
            # keys originate in the regression's predictor names
            # and are caller-controlled bytes.
            transformations.append(
                f"dropped {len(dropped_vcov)} undeclared key(s) from "
                f"'vcov' (names withheld — keys are caller-controlled "
                f"and could carry raw data bytes)"
            )
        if collisions:
            # Collision labels are data-derived (two raw names that
            # both safe_key-cleaned to the same string). Withhold
            # them too: a script could craft colliding names whose
            # collision pattern itself encodes a payload.
            transformations.append(
                f"dropped {len(collisions)} 'vcov' cell(s) whose "
                f"sanitized keys collided (names withheld)"
            )
        if sanitized_vcov:
            out["vcov"] = sanitized_vcov

    # Precision clamp every numeric field and every dict-of-numeric
    # field. Clamp AFTER the cross-field key filter above so we only
    # pay the rounding cost on keys that survive the filter.
    n = out["n"]
    sigfigs = sigfigs_for_n(n)
    for key in _OLS_ALLOWED_NUMERIC_FIELDS:
        if key in out:
            out[key] = clamp_precision(out[key], n)
    for key in _OLS_ALLOWED_DICT_NUMERIC:
        if key in out:
            out[key] = clamp_precision_dict(out[key], n)
    # vcov is dict-of-dict; clamp each inner dict's values.
    if "vcov" in out:
        out["vcov"] = {
            row: clamp_precision_dict(inner, n)
            for row, inner in out["vcov"].items()
        }
    transformations.append(
        f"clamped all numeric fields to {sigfigs} significant figures (n={n})"
    )

    # R² sanity — clamp to [0, 1] after rounding since rounding could push
    # a value right at the boundary out of range.
    if "r_squared" in out:
        out["r_squared"] = max(0.0, min(1.0, out["r_squared"]))
    if "adj_r_squared" in out:
        # adj_r_squared can be mildly negative; clamp only the upper bound.
        out["adj_r_squared"] = min(1.0, out["adj_r_squared"])

    # Cox-style survival counts (``n_failures`` / ``n_subjects``) ride
    # in via the same payload type as OLS but aren't gated by
    # ``min_n_regression``. ``n_failures`` is the event count and is
    # commonly small on rare-outcome studies — "n=2000 records,
    # 3 deaths" identifies those 3 individuals. ``n_subjects`` can
    # also fall below the gate when records are split-episode rows
    # (stset can multiply rows per subject). The shared helper
    # ``_coarsen_small_cox_counts`` applies the same
    # cell-suppression rule we use for ``missing_count`` so the
    # disclosure floor is uniform across surfaces.
    _coarsen_small_cox_counts(out, transformations, config)

    return SanitizerResult(
        ok=True, analysis_type="linear_regression",
        sanitized=out, transformations=transformations,
    )


# ---------------------------------------------------------------------------
# t-test sanitizer
# ---------------------------------------------------------------------------

def _sanitize_t_test(raw: dict[str, Any], config: SDCConfig) -> SanitizerResult:
    missing_reason = _require_fields(raw, _TTEST_REQUIRED, "t_test")
    if missing_reason:
        return SanitizerResult(
            ok=False, analysis_type="t_test", rejection_reason=missing_reason,
        )

    subtype = raw.get("test_type")
    if subtype not in _TTEST_VALID_SUBTYPES:
        # Same exfiltration concern as the dispatcher's unknown-type
        # branch: a script could set ``test_type`` to a raw cell value
        # to smuggle it through this rejection_reason. Bound the leak
        # to the type name (or, for strings, a 40-char ``safe_key``
        # which strips control chars and caps length).
        return SanitizerResult(
            ok=False, analysis_type="t_test",
            rejection_reason=(
                f"test_type must be one of {sorted(_TTEST_VALID_SUBTYPES)}, "
                f"got {type(subtype).__name__}"
            ),
        )

    n1 = raw.get("n1")
    if not isinstance(n1, int) or isinstance(n1, bool) or n1 < 0:
        return SanitizerResult(
            ok=False, analysis_type="t_test",
            rejection_reason=(
                f"n1 must be a non-negative int, got {type(n1).__name__}"
            ),
        )

    # For two-sample / welch, n2 is required; for paired, n1 is the number
    # of pairs (single effective sample size); for one-sample, n1 is N.
    n2 = raw.get("n2")
    needs_n2 = subtype in ("two_sample", "welch")
    if needs_n2:
        if not isinstance(n2, int) or isinstance(n2, bool) or n2 < 0:
            return SanitizerResult(
                ok=False, analysis_type="t_test",
                rejection_reason=(
                    f"{subtype} requires integer n2 >= 0, got "
                    f"{type(n2).__name__}"
                ),
            )

    try:
        require_minimum_n(n1, config.min_n_ttest_group, "n1")
        if needs_n2:
            require_minimum_n(n2, config.min_n_ttest_group, "n2")
    except MinimumNViolation as e:
        return SanitizerResult(
            ok=False, analysis_type="t_test", rejection_reason=str(e),
        )

    transformations: list[str] = []
    out = _collect_allowed(
        raw,
        numeric=_TTEST_ALLOWED_NUMERIC_FIELDS,
        integer=_TTEST_ALLOWED_INT_FIELDS,
        string=_TTEST_ALLOWED_STRING_FIELDS,
        list_numeric=_TTEST_ALLOWED_LIST_NUMERIC,
        transformations=transformations,
    )

    # Re-check required fields after type filtering. ``test_type`` and
    # ``n1`` are pre-validated above; ``mean1`` / ``t_statistic`` /
    # ``p_value`` would otherwise survive to ``ok=True`` if shipped
    # with a non-numeric type.
    missing_after_filter = _require_after_filter(
        out, _TTEST_REQUIRED, "t_test",
        pre_validated=frozenset(("n1", "test_type")),
    )
    if missing_after_filter:
        return SanitizerResult(
            ok=False, analysis_type="t_test",
            rejection_reason=missing_after_filter,
        )

    # ``confidence_interval`` must be a 2-element [lower, upper]
    # list. The generic list_numeric filter accepts any length, so
    # without this check a 3+ element list would survive and could
    # smuggle arbitrary numbers out — one real bound plus arbitrary
    # extras. Drop any length != 2 with a transformation note.
    if "confidence_interval" in out:
        ci = out["confidence_interval"]
        if not isinstance(ci, list) or len(ci) != 2:
            transformations.append(
                f"dropped 'confidence_interval': expected a 2-element "
                f"[lower, upper] list, got "
                f"{len(ci) if isinstance(ci, list) else type(ci).__name__}"
            )
            del out["confidence_interval"]

    # Use the smallest group for conservative sig-fig scaling. If only
    # one n is present (one_sample, paired), use that.
    n_for_precision = n1 if not needs_n2 else min(n1, n2)  # type: ignore[arg-type]
    sigfigs = sigfigs_for_n(n_for_precision)
    for key in _TTEST_ALLOWED_NUMERIC_FIELDS:
        if key in out:
            out[key] = clamp_precision(out[key], n_for_precision)
    if "confidence_interval" in out:
        # Length was already verified as 2 above.
        out["confidence_interval"] = [
            clamp_precision(x, n_for_precision)
            for x in out["confidence_interval"]
        ]
    transformations.append(
        f"clamped numeric fields to {sigfigs} significant figures "
        f"(smallest-group n={n_for_precision})"
    )

    return SanitizerResult(
        ok=True, analysis_type="t_test",
        sanitized=out, transformations=transformations,
    )


# ---------------------------------------------------------------------------
# Descriptive statistics sanitizer
# ---------------------------------------------------------------------------

def _sanitize_descriptive(
    raw: dict[str, Any], config: SDCConfig
) -> SanitizerResult:
    missing_reason = _require_fields(raw, _DESC_REQUIRED, "descriptive")
    if missing_reason:
        return SanitizerResult(
            ok=False, analysis_type="descriptive",
            rejection_reason=missing_reason,
        )

    n_raw = raw.get("n")
    if not isinstance(n_raw, int) or isinstance(n_raw, bool) or n_raw < 0:
        return SanitizerResult(
            ok=False, analysis_type="descriptive",
            rejection_reason=(
                f"n must be a non-negative int, got {type(n_raw).__name__}"
            ),
        )

    try:
        require_minimum_n(n_raw, config.min_n_descriptive, "n")
    except MinimumNViolation as e:
        return SanitizerResult(
            ok=False, analysis_type="descriptive", rejection_reason=str(e),
        )

    transformations: list[str] = []
    # Per-variable opt-in for min / max. When the researcher has
    # added this variable to the dataset's ``non_disclosive_variables``
    # list (via .nora/policy.json), ``min_value`` and ``max_value``
    # join the numeric allowlist for THIS payload only. Default
    # empty set → behaves exactly as before.
    variable_name = raw.get("variable")
    extra_numeric: frozenset[str] = frozenset()
    if (
        isinstance(variable_name, str)
        and variable_name in config.non_disclosive_variables
    ):
        extra_numeric = _DESC_OPTIONAL_NUMERIC_FIELDS

    numeric_allowlist = _DESC_ALLOWED_NUMERIC_FIELDS | extra_numeric
    out = _collect_allowed(
        raw,
        numeric=numeric_allowlist,
        integer=_DESC_ALLOWED_INT_FIELDS,
        string=_DESC_ALLOWED_STRING_FIELDS,
        transformations=transformations,
    )

    # Re-check required fields after type filtering — ``mean`` / ``sd``
    # / ``missing_count`` / ``variable`` would otherwise be silently
    # dropped on type mismatch and the response would still be
    # ``ok=True``.
    missing_after_filter = _require_after_filter(
        out, _DESC_REQUIRED, "descriptive",
        pre_validated=frozenset(("n",)),
    )
    if missing_after_filter:
        return SanitizerResult(
            ok=False, analysis_type="descriptive",
            rejection_reason=missing_after_filter,
        )

    n = out["n"]
    for key in numeric_allowlist:
        if key in out:
            out[key] = clamp_precision(out[key], n)
    if extra_numeric and any(k in out for k in extra_numeric):
        transformations.append(
            f"min_value / max_value passed through (variable "
            f"{variable_name!r} is on the dataset's "
            f"non_disclosive_variables opt-in list)"
        )
    _coarsen_small_missing_count(out, transformations, config)
    transformations.append(
        f"clamped numeric fields to {sigfigs_for_n(n)} significant "
        f"figures (n={n})"
    )

    return SanitizerResult(
        ok=True, analysis_type="descriptive",
        sanitized=out, transformations=transformations,
    )


# ---------------------------------------------------------------------------
# Frequency-table sanitizer (1D only at v0)
# ---------------------------------------------------------------------------

def _sanitize_frequency_table(
    raw: dict[str, Any], config: SDCConfig
) -> SanitizerResult:
    missing_reason = _require_fields(raw, _FREQ_REQUIRED, "frequency_table")
    if missing_reason:
        return SanitizerResult(
            ok=False, analysis_type="frequency_table",
            rejection_reason=missing_reason,
        )

    raw_counts = raw.get("counts")
    if not isinstance(raw_counts, dict) or not raw_counts:
        return SanitizerResult(
            ok=False, analysis_type="frequency_table",
            rejection_reason=(
                "counts must be a non-empty dict of level→count"
            ),
        )
    if len(raw_counts) > _FREQ_MAX_CELLS:
        # Structural cap: 200 distinct levels is already more than any
        # readable frequency table. Bounds the data channel available
        # through level-name strings.
        return SanitizerResult(
            ok=False, analysis_type="frequency_table",
            rejection_reason=(
                f"counts has {len(raw_counts)} distinct levels; "
                f"the structural cap is {_FREQ_MAX_CELLS}. Collapse "
                f"rare levels, use a different summary, or rejected "
                f"as probable adversarial payload."
            ),
        )
    # Count values must be non-negative ints. Keys are level *names* — they
    # originate in the researcher's data (e.g. category strings in the
    # original CSV), so they're an injection surface.
    clean_counts: dict[str, int] = {}
    for k, v in raw_counts.items():
        if not isinstance(k, str):
            return SanitizerResult(
                ok=False, analysis_type="frequency_table",
                rejection_reason=(
                    f"count keys must be strings, got {type(k).__name__}"
                ),
            )
        if not isinstance(v, int) or isinstance(v, bool) or v < 0:
            # Don't echo the level name (it's data-derived; even after
            # ``safe_key`` it carries up to 40 chars of attacker-
            # controlled bytes through ``rejection_reason``, which
            # ``submit_script`` forwards back to the model).
            return SanitizerResult(
                ok=False, analysis_type="frequency_table",
                rejection_reason=(
                    f"a count value is not a non-negative int "
                    f"(got {type(v).__name__}); level name withheld"
                ),
            )
        # safe_key neutralizes control chars / length / newline injections
        # in level names before they cross to Claude. But the same
        # normalisation also creates a collision surface: ``"A\nB"`` and
        # ``"A B"`` both sanitize to ``"A B"``, and two long labels
        # sharing the same 40-char prefix collapse to the same key. If
        # we silently overwrote, a small (suppressible) cell could be
        # hidden inside an aggregated total — defeating cell
        # suppression, since the post-merge count would be above
        # threshold even though one component was below it. Reject
        # the payload outright so the script has to disambiguate
        # before crossing the boundary.
        clean_key = safe_key(k)
        if clean_key in clean_counts:
            # The colliding key is data-derived; don't echo it back to
            # the model (each rejection would ship 40 chars of attacker-
            # controlled bytes through ``rejection_reason``).
            return SanitizerResult(
                ok=False, analysis_type="frequency_table",
                rejection_reason=(
                    "two distinct level names sanitize to the same "
                    "key (e.g. embedded newlines or shared 40-char "
                    "prefix). Collisions are rejected because "
                    "aggregating the counts would defeat cell "
                    "suppression on the smaller component. "
                    "Disambiguate the levels in the source data; "
                    "the colliding key is withheld."
                ),
            )
        clean_counts[clean_key] = v

    transformations: list[str] = []
    # `counts` is handled separately below (it gets SDC suppression). Strip
    # it from the raw dict before _collect_allowed so we don't spuriously
    # log "dropped 'counts'" — it isn't dropped, just routed specially.
    raw_minus_counts = {k: v for k, v in raw.items() if k != "counts"}
    out = _collect_allowed(
        raw_minus_counts,
        integer=_FREQ_ALLOWED_INT_FIELDS,
        string=_FREQ_ALLOWED_STRING_FIELDS,
        transformations=transformations,
    )

    # Re-check required fields after type filtering. ``counts`` is
    # validated above and routed in separately, so it sits in
    # ``pre_validated``. ``variable`` / ``n`` / ``missing_count``
    # would otherwise survive a type mismatch with ``ok=True``.
    missing_after_filter = _require_after_filter(
        out, _FREQ_REQUIRED, "frequency_table",
        pre_validated=frozenset(("counts",)),
    )
    if missing_after_filter:
        return SanitizerResult(
            ok=False, analysis_type="frequency_table",
            rejection_reason=missing_after_filter,
        )

    # Primary cell suppression.
    primary = suppress_cells_below(
        clean_counts, config.cell_suppression_threshold
    )
    primary_suppressed = len(primary.suppressed_keys)
    if primary_suppressed:
        # Log the COUNT of suppressed cells, not their names. The level
        # names of suppressed cells are themselves disclosive (knowing
        # ``rare_diagnosis_X`` exists in this dataset identifies anyone
        # with that diagnosis), so they never leave this sanitizer.
        transformations.append(
            f"primary suppression: {primary_suppressed} cell(s) with "
            f"count < {config.cell_suppression_threshold} "
            f"(level names withheld — see [suppressed] bucket below)"
        )

    # Secondary suppression: when publishing `n`, exactly one primary-
    # suppressed cell is trivially back-calculable from the margin. The
    # fix is to also suppress the next-smallest cell so there are at
    # least two unknowns.
    has_total = "n" in out
    after_secondary = enforce_back_calc_safety(primary, total_n_present=has_total)
    secondary_added_count = (
        len(after_secondary.suppressed_keys) - len(primary.suppressed_keys)
    )
    if secondary_added_count:
        transformations.append(
            f"secondary suppression: also suppressed "
            f"{secondary_added_count} cell(s) because only one "
            f"primary-suppressed cell was back-calculable from the "
            f"total n (level name withheld)"
        )

    # Degenerate case: exactly one suppressed cell remains AND no other
    # cell was available for secondary (e.g. all cells < threshold, or
    # single-cell table). Without a sacrificial cell, the only way to
    # prevent back-calculation from the margin is to remove the margin
    # itself — drop `n` and `missing_count`.
    total_suppressed_distinct = len(after_secondary.suppressed_keys)
    if total_suppressed_distinct == 1 and has_total:
        # No secondary was added and we still have a single suppressed
        # cell + a published total. Strip the total. Note: this check
        # MUST run on the per-cell suppression result, before bucketing,
        # because bucketing collapses N suppressed cells into a single
        # entry — afterwards the dict no longer carries the count.
        for margin_field in ("n", "missing_count"):
            out.pop(margin_field, None)
        transformations.append(
            "stripped total n and missing_count: a single cell was "
            "suppressed and no secondary cell was available, so the "
            "margin would have made it back-calculable"
        )

    # Bucket every suppressed entry under a single ``[suppressed]``
    # key. The level names themselves are an SDC violation — knowing
    # ``rare_disease_X`` exists in the dataset identifies someone with
    # that diagnosis, regardless of whether the count is masked. The
    # bucket carries the suppression marker as its value (``<10``);
    # callers can read ``suppressed_cell_count`` for the count of
    # distinct levels collapsed here. The bucket aggregate is
    # back-calculable from ``n`` minus the visible cells, but only as
    # a SUM across all bucketed levels — no individual level's count
    # is recoverable.
    bucketed_counts: dict[str, int | str] = {
        k: v
        for k, v in after_secondary.counts.items()
        if isinstance(v, int)
    }
    if total_suppressed_distinct > 0:
        bucketed_counts["[suppressed]"] = suppression_marker(
            config.cell_suppression_threshold
        )
        out["suppressed_cell_count"] = total_suppressed_distinct
    out["counts"] = bucketed_counts

    # Coarsen any rare ``missing_count`` that survived the back-calc
    # strip above. ``submit_script`` can publish a frequency_table
    # with ``missing_count=1`` even when the cell suppression rule
    # otherwise fires cleanly — the schema-side ``request_data
    # (na_count)`` path already suppresses the same disclosure on
    # the discovery side, this closes the gap on the stored-result
    # side.
    _coarsen_small_missing_count(out, transformations, config)

    return SanitizerResult(
        ok=True, analysis_type="frequency_table",
        sanitized=out, transformations=transformations,
    )


# ---------------------------------------------------------------------------
# Crosstab sanitizer (2D, no margins)
# ---------------------------------------------------------------------------

def _sanitize_crosstab(
    raw: dict[str, Any], config: SDCConfig
) -> SanitizerResult:
    """Primary-suppress cells in a 2D contingency table.

    Structure: ``counts`` is a dict-of-dicts, ``counts[row_level][col_level]
    = int``. Cells below threshold are replaced with the suppression
    marker. No margins (row totals, column totals, grand total) are
    ever emitted; attempting to include them via any named field on the
    allowlist is impossible by construction, but we additionally log a
    loud drop message if the researcher's script tried to pass a known
    margin-field name like ``n`` or ``row_totals``.
    """
    missing_reason = _require_fields(raw, _XTAB_REQUIRED, "crosstab")
    if missing_reason:
        return SanitizerResult(
            ok=False, analysis_type="crosstab",
            rejection_reason=missing_reason,
        )

    raw_counts = raw.get("counts")
    if not isinstance(raw_counts, dict) or not raw_counts:
        return SanitizerResult(
            ok=False, analysis_type="crosstab",
            rejection_reason=(
                "counts must be a non-empty dict-of-dicts "
                "(row_level → col_level → int)"
            ),
        )
    # Structural cap on the total cell count (sum over rows of inner
    # dict sizes). A 50×50 crosstab is already dense; anything bigger
    # isn't readable output.
    total_cells = 0
    for inner in raw_counts.values():
        if isinstance(inner, dict):
            total_cells += len(inner)
    if total_cells > _XTAB_MAX_CELLS:
        return SanitizerResult(
            ok=False, analysis_type="crosstab",
            rejection_reason=(
                f"counts contains {total_cells} cells; the structural "
                f"cap is {_XTAB_MAX_CELLS}. Pre-aggregate the table, "
                f"or rejected as probable adversarial payload."
            ),
        )

    # Validate nested shape and collect a flattened view for suppression.
    # Row + col keys are level *names* from the data — sanitize before
    # use. A prior version did ``clean_counts[(safe_row, safe_col)] = v``
    # unconditionally, which let two raw labels that sanitize to the
    # same safe_key SILENTLY OVERWRITE each other in the dict. Concrete
    # leak: raw row ``"A\nB"`` with count 2 (suppressible) overwritten
    # by raw row ``"A B"`` with count 100 (visible) leaves the model
    # seeing the visible count under a label that's actually ambiguous
    # — and worse, secondary suppression accounting now operates on
    # the wrong value.
    #
    # Detection: a duplicate (safe_row, safe_col) tuple in the build
    # loop means either (a) two raw row keys sanitized to the same
    # safe_row, or (b) two raw col keys within a row sanitized to
    # the same safe_col. Either is genuinely ambiguous; the SDC
    # posture is to deny rather than guess. The fix matches the
    # equivalent gate in ``data_request._resolve_variable``.
    clean_counts: dict[tuple[str, str], int] = {}
    col_levels: set[str] = set()
    for row_key, inner in raw_counts.items():
        if not isinstance(row_key, str):
            return SanitizerResult(
                ok=False, analysis_type="crosstab",
                rejection_reason=(
                    f"row keys must be strings; got {type(row_key).__name__}"
                ),
            )
        if not isinstance(inner, dict):
            # Row labels are data-derived; redact them from rejection
            # messages so the model can't trigger this branch with a
            # crafted label and read it back via ``rejection_reason``.
            return SanitizerResult(
                ok=False, analysis_type="crosstab",
                rejection_reason=(
                    f"a counts row is not a dict "
                    f"(col_level → int); got {type(inner).__name__}; "
                    f"row label withheld"
                ),
            )
        safe_row = safe_key(row_key)
        for col_key, v in inner.items():
            if not isinstance(col_key, str):
                return SanitizerResult(
                    ok=False, analysis_type="crosstab",
                    rejection_reason=(
                        f"col keys must be strings; got "
                        f"{type(col_key).__name__}"
                    ),
                )
            if not isinstance(v, int) or isinstance(v, bool) or v < 0:
                # Don't echo row/col labels — they are data-derived.
                return SanitizerResult(
                    ok=False, analysis_type="crosstab",
                    rejection_reason=(
                        f"a counts cell is not a non-negative int "
                        f"(got {type(v).__name__}); row/col labels "
                        f"withheld"
                    ),
                )
            safe_col = safe_key(col_key)
            if (safe_row, safe_col) in clean_counts:
                # The colliding labels are data-derived; redact.
                return SanitizerResult(
                    ok=False, analysis_type="crosstab",
                    rejection_reason=(
                        "label collision after sanitization in "
                        "crosstab: two distinct raw row/col labels "
                        "sanitize to the same safe_key, which would "
                        "silently overwrite counts (and break "
                        "suppression accounting). Rename the "
                        "colliding levels in the source script — e.g. "
                        "strip embedded whitespace / control "
                        "characters before the crosstab — and re-run. "
                        "The colliding labels are withheld."
                    ),
                )
            clean_counts[(safe_row, safe_col)] = v
            col_levels.add(safe_col)

    transformations: list[str] = []

    # Strip known margin-ish fields loudly before _collect_allowed drops
    # unknown fields quietly. This makes violations of the no-margins
    # invariant visible in the transformation log.
    raw_pruned = dict(raw)
    raw_pruned.pop("counts", None)
    for field in _XTAB_FORBIDDEN_MARGIN_FIELDS:
        if field in raw_pruned:
            transformations.append(
                f"dropped margin field {field!r}: crosstabs do not emit "
                f"totals (no margins → no back-calc)"
            )
            raw_pruned.pop(field, None)

    out = _collect_allowed(
        raw_pruned,
        integer=_XTAB_ALLOWED_INT_FIELDS,
        string=_XTAB_ALLOWED_STRING_FIELDS,
        transformations=transformations,
    )

    # Re-check required fields after type filtering. ``counts`` is
    # validated above and reattached separately.
    missing_after_filter = _require_after_filter(
        out, _XTAB_REQUIRED, "crosstab",
        pre_validated=frozenset(("counts",)),
    )
    if missing_after_filter:
        return SanitizerResult(
            ok=False, analysis_type="crosstab",
            rejection_reason=missing_after_filter,
        )

    # Primary suppression on the flat view, then reshape back to
    # nested with bucketing. Suppressed (row, col) labels themselves
    # are disclosive — a row named ``rare_diagnosis`` whose only
    # column counts are below threshold leaks the existence of that
    # diagnosis even when its numbers are masked. We:
    #
    #   * Per surviving row: collapse all of its suppressed columns
    #     into a single ``[suppressed]`` entry.
    #   * Drop rows that have NO surviving (visible) cells entirely
    #     — including their row label — and account for them in a
    #     top-level ``suppressed_row_count`` field.
    #
    # The structural cap on table size keeps this loop cheap.
    threshold = config.cell_suppression_threshold
    marker = suppression_marker(threshold)

    # Primary status per cell: True == suppressed (below threshold).
    suppressed_status: dict[tuple[str, str], bool] = {
        key: v < threshold for key, v in clean_counts.items()
    }
    primary_count = sum(1 for s in suppressed_status.values() if s)

    # Secondary suppression to defend against per-row / per-column
    # back-calc when the model has an externally-known marginal.
    #
    # The attack: the model issues a separate ``request_data``
    # frequency_table on the row (or column) variable. That table
    # publishes per-level counts for visible levels — i.e. the row
    # / column marginal N_R (or N_C) of this crosstab. Then:
    #
    #   * If a surviving row R has exactly ONE suppressed cell, the
    #     row publishes a ``[suppressed]`` bucket whose sum equals
    #     ``N_R - sum(visible in R)`` — recovering the lone cell
    #     exactly.
    #   * Symmetrically for columns: the model sums visible cells
    #     in column C across the output and computes ``N_C -
    #     sum(visible in C)``. If exactly one cell in column C is
    #     hidden (suppressed in a surviving row, since dropped rows
    #     also have their column-C cell suppressed), that cell is
    #     recovered.
    #
    # Remedy is the standard SDC choice (ONS / Eurostat guidance):
    # promote additional visible cells to suppressed until every row
    # and every column with any suppression has either 0 or >=2
    # suppressed cells. Iterate to a fixed point — a row-side fix
    # can create a column-side violation and vice versa. The loop
    # is bounded by the total cell count.
    #
    # Victim choice: the smallest visible cell. This is the standard
    # data-utility-minimising choice — losing the smallest value
    # costs the least information to legitimate downstream
    # analysis. Ties broken by key for determinism (tests need
    # reproducibility).
    sorted_rows = sorted({r for (r, _) in clean_counts})
    sorted_cols = sorted({c for (_, c) in clean_counts})
    secondary_count = 0
    while True:
        target: tuple[str, str] | None = None
        # Row pass first.
        for r in sorted_rows:
            in_row = [c for c in sorted_cols if (r, c) in clean_counts]
            if not in_row:
                continue
            n_supp = sum(1 for c in in_row if suppressed_status[(r, c)])
            if n_supp != 1:
                continue
            visible = [
                (c, clean_counts[(r, c)]) for c in in_row
                if not suppressed_status[(r, c)]
            ]
            if not visible:
                continue
            victim_c, _ = min(visible, key=lambda cv: (cv[1], cv[0]))
            target = (r, victim_c)
            break
        # Column pass if row pass found nothing.
        if target is None:
            for c in sorted_cols:
                in_col = [r for r in sorted_rows if (r, c) in clean_counts]
                if not in_col:
                    continue
                n_supp = sum(1 for r in in_col if suppressed_status[(r, c)])
                if n_supp != 1:
                    continue
                visible = [
                    (r, clean_counts[(r, c)]) for r in in_col
                    if not suppressed_status[(r, c)]
                ]
                if not visible:
                    continue
                victim_r, _ = min(visible, key=lambda rv: (rv[1], rv[0]))
                target = (victim_r, c)
                break
        if target is None:
            break
        suppressed_status[target] = True
        secondary_count += 1

    nested_raw: dict[str, dict[str, int | str]] = {}
    suppressed_cell_count = 0
    for (r, c), v in clean_counts.items():
        if r not in nested_raw:
            nested_raw[r] = {}
        if suppressed_status[(r, c)]:
            nested_raw[r][c] = marker
            suppressed_cell_count += 1
        else:
            nested_raw[r][c] = v

    nested: dict[str, dict[str, int | str]] = {}
    suppressed_row_count = 0
    for row_label, row_cells in nested_raw.items():
        visible_cols = {
            c: v for c, v in row_cells.items() if isinstance(v, int)
        }
        if not visible_cols:
            # Every cell in this row was suppressed — drop the row
            # label too. Knowing the row exists (and is rare) is the
            # leak we're closing here.
            suppressed_row_count += 1
            continue
        n_suppressed_in_row = len(row_cells) - len(visible_cols)
        if n_suppressed_in_row:
            visible_cols["[suppressed]"] = marker
        nested[row_label] = visible_cols

    if primary_count:
        transformations.append(
            f"primary suppression: {primary_count} cell(s) "
            f"with count < {threshold} (cell labels withheld — "
            f"bucketed under '[suppressed]')"
        )
    if secondary_count:
        # Secondary cells were >= threshold originally but got
        # promoted to suppressed to defend against per-row /
        # per-column back-calc from an externally-known marginal
        # (a separate request_data on the row or column variable
        # publishes its level totals). The published bucket marker
        # stays ``<threshold`` for compactness; this log line is
        # the authoritative statement that some bucket entries do
        # NOT actually fall below threshold.
        transformations.append(
            f"secondary suppression: {secondary_count} additional "
            f"cell(s) promoted to '[suppressed]' to prevent "
            f"per-row/column back-calc when an external marginal "
            f"is known (smallest visible cells chosen)"
        )
    if suppressed_row_count:
        transformations.append(
            f"row suppression: {suppressed_row_count} row(s) had every "
            f"cell below threshold; row labels withheld since their "
            f"existence at this rarity is itself disclosive"
        )
        out["suppressed_row_count"] = suppressed_row_count
    if suppressed_cell_count:
        out["suppressed_cell_count"] = suppressed_cell_count

    # Strip ``missing_count`` when the cross-query back-calc is trivial.
    # ``n`` is already in ``_XTAB_FORBIDDEN_MARGIN_FIELDS`` so the
    # crosstab payload alone never exposes the grand total — but the
    # model can derive ``N`` from a separate descriptive query, then
    # compute ``sum(suppressed) = (N - missing_count) - sum(visible)``.
    # The unsafe configuration is "exactly one cell suppressed AND no
    # row was fully dropped": the surviving row's ``[suppressed]``
    # bucket then contains exactly that cell's count, the bucket sum
    # equals ``(N - missing_count) - sum(visible)`` exactly, and the
    # row label is published, so a single arithmetic step recovers the
    # cell. Mirrors the freq-table guard at ``_sanitize_frequency_table``
    # which drops both ``n`` and ``missing_count`` for the analogous
    # in-payload case. With a fully-dropped row in the mix, the dropped
    # row's cells contribute to the same sum but can't be separated, so
    # the cleanly-recoverable case requires ``suppressed_row_count == 0``.
    if (
        suppressed_cell_count == 1
        and suppressed_row_count == 0
        and "missing_count" in out
    ):
        out.pop("missing_count", None)
        transformations.append(
            "stripped missing_count: exactly one cell was suppressed "
            "and no row was dropped, so the published "
            "'[suppressed]' bucket would have been back-calculable "
            "from missing_count plus an externally-known N"
        )

    # Coarsen any rare ``missing_count`` that survived the back-calc
    # strip above. The exact small-missingness disclosure ("the one
    # row missing on either dimension") is independent of the
    # back-calc concern handled by the strip — the strip protects
    # the suppressed-cell bucket sum, this protects the missingness
    # cell itself.
    _coarsen_small_missing_count(out, transformations, config)

    out["counts"] = nested

    return SanitizerResult(
        ok=True, analysis_type="crosstab",
        sanitized=out, transformations=transformations,
    )


# ---------------------------------------------------------------------------
# Magnitude-table sanitizer (sum/mean by group, with dominance rule)
# ---------------------------------------------------------------------------

def _sanitize_magnitude_table(
    raw: dict[str, Any], config: SDCConfig
) -> SanitizerResult:
    """Suppress cells that fail either primary (n) or dominance (max_share).

    Unlike frequency tables, where the disclosure risk is "a cell of
    size 1 identifies someone", a magnitude cell can have n=100 and
    still be disclosive if one of those 100 contributors dominates —
    their value is effectively revealed by the cell's sum. That's what
    the (1, k)-dominance rule handles.

    Two suppression triggers per cell:
    - ``n < cell_suppression_threshold``: primary. Suppress.
    - ``dominance_fails(max_share, dominance_threshold)``: dominance.
      Suppress.

    The ``max_share`` field is computed by the runtime library on raw
    values (since the sanitizer has no access to them), consulted here
    for the suppression decision, and then **stripped from the output**.
    Emitting it would tell Claude "this cell has a dominant contributor"
    — information we don't need to publish.
    """
    missing_reason = _require_fields(
        raw, _MAGTAB_REQUIRED, "magnitude_table"
    )
    if missing_reason:
        return SanitizerResult(
            ok=False, analysis_type="magnitude_table",
            rejection_reason=missing_reason,
        )

    # Helper-provenance gate. ``max_share`` is caller-supplied and
    # consulted-only — the dominance gate trusts it. A script that
    # bypasses the typed helper (e.g. via the generic ``result()``)
    # could publish a dominance-violating value with a forged
    # ``max_share=0`` and skip the gate. Require the marker the typed
    # helper stamps. See the constant's docstring for threat-model
    # detail and the limits of this defense.
    if raw.get(_HELPER_PROVENANCE_FIELD) != _MAGTAB_HELPER_VALUE:
        return SanitizerResult(
            ok=False, analysis_type="magnitude_table",
            rejection_reason=(
                "magnitude_table payloads must come through the "
                "typed runtime helper (Python: nora.from_magnitude_table; "
                "R: nora$from_magnitude_table; Stata: "
                "nora_result_magnitude). The generic nora.result() API "
                "is rejected for this type because cell-level max_share "
                "is consulted-only and a hand-crafted payload could "
                "publish a dominance-violating value with max_share=0 "
                "to skip the dominance gate."
            ),
        )

    aggregation = raw.get("aggregation")
    if aggregation not in _MAGTAB_VALID_AGGREGATIONS:
        return SanitizerResult(
            ok=False, analysis_type="magnitude_table",
            rejection_reason=(
                f"aggregation must be one of "
                f"{sorted(_MAGTAB_VALID_AGGREGATIONS)}, got "
                f"{type(aggregation).__name__}"
            ),
        )

    raw_cells = raw.get("cells")
    if not isinstance(raw_cells, dict) or not raw_cells:
        return SanitizerResult(
            ok=False, analysis_type="magnitude_table",
            rejection_reason=(
                "cells must be a non-empty dict of group_level → "
                "{value, n, max_share}"
            ),
        )
    if len(raw_cells) > _MAGTAB_MAX_CELLS:
        # Same rationale as the other table caps — bound the
        # data-channel bandwidth through group-name strings.
        return SanitizerResult(
            ok=False, analysis_type="magnitude_table",
            rejection_reason=(
                f"cells has {len(raw_cells)} groups; the structural "
                f"cap is {_MAGTAB_MAX_CELLS}. Aggregate to fewer "
                f"groups, or rejected as probable adversarial payload."
            ),
        )

    transformations: list[str] = []

    # Strip the raw cells dict before _collect_allowed — we handle it
    # specially. Without this, the log spuriously says "dropped cells".
    # Also strip the helper-provenance marker so it doesn't appear in
    # the transformation log as a "dropped unknown field" — the marker
    # is internal to the runtime-library/sanitizer boundary and the
    # model has no business seeing its name.
    raw_pruned = {
        k: v for k, v in raw.items()
        if k != "cells" and k != _HELPER_PROVENANCE_FIELD
    }
    out = _collect_allowed(
        raw_pruned,
        string=_MAGTAB_ALLOWED_STRING_FIELDS,
        transformations=transformations,
    )

    # Re-check required fields after type filtering. ``cells`` and
    # ``aggregation`` are pre-validated above.
    missing_after_filter = _require_after_filter(
        out, _MAGTAB_REQUIRED, "magnitude_table",
        pre_validated=frozenset(("cells", "aggregation")),
    )
    if missing_after_filter:
        return SanitizerResult(
            ok=False, analysis_type="magnitude_table",
            rejection_reason=missing_after_filter,
        )

    threshold_n = config.cell_suppression_threshold
    dom_threshold = config.dominance_threshold
    marker_n = suppression_marker(threshold_n)

    cleaned_cells: dict[str, Any] = {}
    # Counts only — never the group labels. See the
    # "suppressed cells leak labels" SDC fix: a cell whose label is
    # ``rare_industry`` is disclosive even when the count and value
    # are masked, since it tells the model that level exists in the
    # data at small N. We track aggregate counts for the
    # transformation log and bucket all suppressed groups under a
    # single ``[suppressed]`` entry below.
    n_suppressed_total = 0
    n_suppressed_by_n = 0
    n_suppressed_by_dominance = 0

    for raw_group, cell in raw_cells.items():
        if not isinstance(raw_group, str):
            return SanitizerResult(
                ok=False, analysis_type="magnitude_table",
                rejection_reason=(
                    f"cells keys must be strings; got {type(raw_group).__name__}"
                ),
            )
        if not isinstance(cell, dict):
            # Group labels are data-derived; redact from
            # ``rejection_reason`` (which crosses to the model via
            # ``submit_script``). ``cell`` here is the offending
            # dict-or-not from the raw payload, distinct from
            # ``cells`` (the parent dict) — no collision with the
            # parent name in the sanitizer itself.
            return SanitizerResult(
                ok=False, analysis_type="magnitude_table",
                rejection_reason=(
                    f"a cells entry is not a dict with keys "
                    f"value, n, max_share; got {type(cell).__name__}; "
                    f"group label withheld"
                ),
            )
        value = cell.get("value")
        n = cell.get("n")
        max_share = cell.get("max_share")

        if not _is_finite_number(value):
            return SanitizerResult(
                ok=False, analysis_type="magnitude_table",
                rejection_reason=(
                    f"a cells entry's 'value' is not a finite "
                    f"number; got {type(value).__name__}; "
                    f"group label withheld"
                ),
            )
        if not isinstance(n, int) or isinstance(n, bool) or n < 0:
            return SanitizerResult(
                ok=False, analysis_type="magnitude_table",
                rejection_reason=(
                    f"a cells entry's 'n' is not a non-negative "
                    f"int; got {type(n).__name__}; "
                    f"group label withheld"
                ),
            )
        if not _is_finite_number(max_share):
            return SanitizerResult(
                ok=False, analysis_type="magnitude_table",
                rejection_reason=(
                    f"a cells entry's 'max_share' is not a "
                    f"finite number; got {type(max_share).__name__}; "
                    f"group label withheld"
                ),
            )

        safe_group = safe_key(raw_group)

        fails_n = n < threshold_n
        fails_dominance = dominance_fails(float(max_share), threshold=dom_threshold)

        if fails_n or fails_dominance:
            # Don't emit a per-group entry at all — the group label
            # is itself disclosive (``rare_industry`` exists with
            # n < threshold identifies its members). Track counts
            # only.
            n_suppressed_total += 1
            if fails_n:
                n_suppressed_by_n += 1
            if fails_dominance:
                n_suppressed_by_dominance += 1
            continue
        # Reject ``safe_key`` collisions outright (mirrors crosstab /
        # frequency_table). Two raw group labels that sanitize to the
        # same form would silently overwrite — a small (suppressible)
        # cell could be replaced by a visible cell, or vice versa,
        # and the suppression accounting would never see the dropped
        # entry. Group labels are data-derived; rejection_reason
        # withholds them.
        if safe_group in cleaned_cells:
            return SanitizerResult(
                ok=False, analysis_type="magnitude_table",
                rejection_reason=(
                    "label collision after sanitization in "
                    "magnitude_table cells: two distinct raw group "
                    "labels sanitize to the same safe_key, which "
                    "would silently overwrite values (and break "
                    "dominance / primary suppression accounting). "
                    "Disambiguate the group labels in the source "
                    "script — e.g. strip embedded whitespace / "
                    "control characters before grouping — and "
                    "re-run. The colliding labels are withheld."
                ),
            )
        # Precision-clamp the value at sigfigs appropriate for n.
        cleaned_cells[safe_group] = {
            "value": clamp_precision(float(value), n),
            "n": n,
        }
        # NEVER emit max_share. It's only used internally above.

    if n_suppressed_by_n:
        transformations.append(
            f"primary suppression: {n_suppressed_by_n} cell(s) with "
            f"n < {threshold_n} (group labels withheld — bucketed "
            f"under '[suppressed]')"
        )
    if n_suppressed_by_dominance:
        transformations.append(
            f"dominance suppression: {n_suppressed_by_dominance} cell(s) "
            f"where one contributor exceeded {dom_threshold:.0%} of "
            f"the total (group labels withheld)"
        )
    transformations.append(
        "max_share stripped from every cell: dominance metric is internal "
        "to the sanitizer and never forwarded"
    )

    if n_suppressed_total:
        # Single bucketed entry for every suppressed group. The
        # marker tells the model these cells exist but their labels
        # and per-group n / value are deliberately withheld.
        cleaned_cells["[suppressed]"] = {
            "value": marker_n,
            "n": marker_n,
        }
        out["suppressed_cell_count"] = n_suppressed_total

    out["cells"] = cleaned_cells

    return SanitizerResult(
        ok=True, analysis_type="magnitude_table",
        sanitized=out, transformations=transformations,
    )


# ---------------------------------------------------------------------------
# Correlation matrix sanitizer
# ---------------------------------------------------------------------------


def _sanitize_correlation_matrix(
    raw: dict[str, Any], config: SDCConfig
) -> SanitizerResult:
    """Pairwise correlation matrix (Pearson / Spearman / Kendall).

    Privacy rationale: the matrix is a sums-of-products aggregate, so
    no per-row data crosses back. Three guardrails on top:

    1. Minimum N (``min_n_descriptive``) — at very low N a near-perfect
       correlation is just "the three points are collinear" and could
       imply individual coordinates, so reject below threshold.
    2. Variable-count cap — limits how much can be smuggled through
       even-well-formed payloads, mirroring the OLS predictor cap.
    3. Cross-field key validation — every row/column key in the
       correlations dict must be a declared variable. Without this,
       a prompt-injected script could smuggle channels via spurious
       keys like ``leak_bit_0`` carrying engineered values.

    Each correlation is precision-clamped (sigfigs scale with N), then
    clipped to [-1, 1] in case rounding pushed it past the boundary.
    """
    missing_reason = _require_fields(raw, _CORR_REQUIRED, "correlation_matrix")
    if missing_reason:
        return SanitizerResult(
            ok=False, analysis_type="correlation_matrix",
            rejection_reason=missing_reason,
        )

    n_raw = raw.get("n")
    if not isinstance(n_raw, int) or isinstance(n_raw, bool) or n_raw < 0:
        return SanitizerResult(
            ok=False, analysis_type="correlation_matrix",
            rejection_reason=(
                f"n must be a non-negative int, got {type(n_raw).__name__}"
            ),
        )

    try:
        require_minimum_n(n_raw, config.min_n_descriptive, "n")
    except MinimumNViolation as e:
        return SanitizerResult(
            ok=False, analysis_type="correlation_matrix",
            rejection_reason=str(e),
        )

    raw_vars = raw.get("variables")
    if not isinstance(raw_vars, list) or not raw_vars:
        return SanitizerResult(
            ok=False, analysis_type="correlation_matrix",
            rejection_reason="variables must be a non-empty list of strings",
        )
    if len(raw_vars) > _CORR_MAX_VARIABLES:
        return SanitizerResult(
            ok=False, analysis_type="correlation_matrix",
            rejection_reason=(
                f"variables has {len(raw_vars)} entries; the structural cap "
                f"is {_CORR_MAX_VARIABLES}. A correlation matrix that wide "
                f"isn't interpretable output — rejected as probable "
                f"adversarial payload."
            ),
        )

    # Method, if provided, must be one we recognise.
    method = raw.get("method")
    if method is not None and method not in _CORR_VALID_METHODS:
        return SanitizerResult(
            ok=False, analysis_type="correlation_matrix",
            rejection_reason=(
                f"method must be one of {sorted(_CORR_VALID_METHODS)} or "
                f"omitted, got {type(method).__name__}"
            ),
        )

    correlations = raw.get("correlations")
    if not isinstance(correlations, dict) or not correlations:
        return SanitizerResult(
            ok=False, analysis_type="correlation_matrix",
            rejection_reason="correlations must be a non-empty dict of dicts",
        )

    transformations: list[str] = []
    out = _collect_allowed(
        raw,
        integer=_CORR_ALLOWED_INT_FIELDS,
        string=_CORR_ALLOWED_STRING_FIELDS,
        list_string=_CORR_ALLOWED_LIST_STRING,
        transformations=transformations,
    )

    # Re-check required fields after type filtering. ``n``,
    # ``variables``, and ``correlations`` are pre-validated above
    # (``correlations`` is reattached after sanitization further down).
    missing_after_filter = _require_after_filter(
        out, _CORR_REQUIRED, "correlation_matrix",
        pre_validated=frozenset(("n", "variables", "correlations")),
    )
    if missing_after_filter:
        return SanitizerResult(
            ok=False, analysis_type="correlation_matrix",
            rejection_reason=missing_after_filter,
        )

    # ``out["variables"]`` is the safe_key-transformed list (each
    # element passed through safe_key in _collect_allowed). The raw
    # ``correlations`` dict keys are not yet transformed. Compare on
    # safe_key both sides so a long or otherwise-transformed name
    # doesn't get spuriously "dropped as undeclared" simply because
    # the variables list shows the truncated form. Without this, a
    # legitimate matrix with a 50-char variable name returned with
    # ``correlations: {}`` and ``ok=True`` — silent empty success.
    sanitized_vars = list(out.get("variables") or [])
    # Reject sanitized-name collisions in the declared variables list.
    # Two raw names like ``"A B"`` / ``"A\nB"`` collapse to the same
    # ``safe_key`` and the previous ``set(...)`` silently merged
    # them — leaving a matrix where one declared label represents
    # two source variables (and the corresponding rows / columns
    # were dropped or merged in the per-key collision counters
    # below). The result was an ``ok=True`` matrix that was
    # ambiguous from the model's seat. Reject loudly so the script
    # has to disambiguate at the source. Same posture as the
    # frequency_table collision check above.
    if len(sanitized_vars) != len(set(sanitized_vars)):
        return SanitizerResult(
            ok=False, analysis_type="correlation_matrix",
            rejection_reason=(
                "two or more variable names sanitize to the same "
                "key (e.g. embedded newlines or shared 40-char "
                "prefix). The declared variables list is ambiguous; "
                "rename the source variables to disambiguate. The "
                "colliding names are withheld — they're data-derived."
            ),
        )
    declared = set(sanitized_vars)

    sanitized_corr: dict[str, dict[str, float]] = {}
    # Counters only — row/col keys are data-derived (variable names),
    # so the per-key sample previously emitted in this transformation
    # leaked names back to the model. The per-row store keeps the raw
    # payload for researcher audit; the model sees totals.
    dropped_row_count = 0
    dropped_col_count = 0
    collision_row_count = 0
    collision_col_count = 0
    n = out["n"]
    for raw_row_key, row_value in correlations.items():
        if not isinstance(raw_row_key, str):
            dropped_row_count += 1
            continue
        row_key = safe_key(raw_row_key)
        if row_key not in declared:
            dropped_row_count += 1
            continue
        if not isinstance(row_value, dict):
            dropped_row_count += 1
            continue
        # Reject row-key collisions outright — two raw row keys that
        # ``safe_key`` collapses to the same form would silently
        # overwrite, replacing an earlier correlation row with a
        # later one and reporting it under an ambiguous label.
        if row_key in sanitized_corr:
            collision_row_count += 1
            continue
        kept_row: dict[str, float] = {}
        for raw_col_key, val in row_value.items():
            if not isinstance(raw_col_key, str):
                dropped_col_count += 1
                continue
            col_key = safe_key(raw_col_key)
            if col_key not in declared:
                dropped_col_count += 1
                continue
            if not _is_finite_number(val):
                continue
            # Same collision check on the column axis: skip duplicates
            # rather than overwrite. Counter only, no name echo.
            if col_key in kept_row:
                collision_col_count += 1
                continue
            clamped = clamp_precision(float(val), n)
            # Clip to [-1, 1] in case precision-clamping nudged a near-
            # boundary value past it.
            clamped = max(-1.0, min(1.0, clamped))
            kept_row[col_key] = clamped
        if kept_row:
            sanitized_corr[row_key] = kept_row
    if dropped_row_count or dropped_col_count:
        transformations.append(
            f"dropped {dropped_row_count} undeclared row(s) and "
            f"{dropped_col_count} undeclared column entry(ies) from "
            f"correlations (names withheld)"
        )
    if collision_row_count or collision_col_count:
        transformations.append(
            f"dropped {collision_row_count} duplicate row key(s) and "
            f"{collision_col_count} duplicate column key(s) from "
            f"correlations after sanitization (colliding names "
            f"withheld)"
        )
    if not sanitized_corr:
        # Every entry got dropped. Returning ok=True with an empty
        # matrix is misleading — the model would think "the analysis
        # ran but produced no correlations" when the truth is "the
        # payload's keys didn't line up with the declared variables."
        return SanitizerResult(
            ok=False, analysis_type="correlation_matrix",
            rejection_reason=(
                "correlations dict is empty after sanitization — every "
                "row/column key was either not in the declared "
                "``variables`` list or had a non-finite value. The "
                "payload likely has a variables/correlations mismatch."
            ),
        )
    out["correlations"] = sanitized_corr

    sigfigs = sigfigs_for_n(n)
    transformations.append(
        f"clamped correlation values to {sigfigs} significant figures (n={n})"
    )

    # Coarsen rare ``missing_count``. The complete-case correlation
    # path can publish a single-row-missing count exactly, which
    # identifies the one observation that's incomplete on at least
    # one of the variables in the matrix — same disclosure shape
    # the schema-side ``request_data(na_count)`` gate already
    # closes.
    _coarsen_small_missing_count(out, transformations, config)

    return SanitizerResult(
        ok=True, analysis_type="correlation_matrix",
        sanitized=out, transformations=transformations,
    )


# ---------------------------------------------------------------------------
# Handler registry
# ---------------------------------------------------------------------------

_HANDlerFn = Callable[[dict[str, Any], SDCConfig], SanitizerResult]

_HANDLERS: dict[str, _HANDlerFn] = {
    "linear_regression": _sanitize_linear_regression,
    "t_test": _sanitize_t_test,
    "descriptive": _sanitize_descriptive,
    "frequency_table": _sanitize_frequency_table,
    "crosstab": _sanitize_crosstab,
    "magnitude_table": _sanitize_magnitude_table,
    "correlation_matrix": _sanitize_correlation_matrix,
}


def supported_types() -> list[str]:
    """Return the list of analysis types the sanitizer currently accepts."""
    return sorted(_HANDLERS.keys())
