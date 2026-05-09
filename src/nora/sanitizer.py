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

    for k, v in raw.items():
        if k not in allowed:
            # safe_key on the field name before it's echoed back to Claude.
            _log(f"dropped unknown/forbidden field {safe_key(str(k))!r}")
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
            for kk, vv in v.items():
                if not isinstance(kk, str):
                    _log(f"dropped {k!r}[{safe_key(str(kk))!r}]: key not a string")
                    continue
                if not _is_finite_number(vv):
                    _log(f"dropped {k!r}[{safe_key(kk)!r}]: not a finite number")
                    continue
                # safe_key on the key — e.g. coefficient names, which
                # originate in the data's variable names, cross to Claude.
                clean[safe_key(kk)] = float(vv)
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

    if surplus:
        # Single line that bounds the total log size at
        # ``_COLLECT_ALLOWED_LOG_CAP + 1`` regardless of payload size.
        t.append(
            f"… and {surplus} more drops omitted from this payload's log "
            f"(cap {_COLLECT_ALLOWED_LOG_CAP})"
        )
    return out


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
        return SanitizerResult(
            ok=False, analysis_type="linear_regression",
            rejection_reason=f"n must be a non-negative int, got {n_raw!r}",
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
            transformations.append(
                f"dropped {len(dropped)} undeclared key(s) from "
                f"{dict_field!r}: {sorted(dropped)[:5]}"
                + (" …" if len(dropped) > 5 else "")
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
    raw_vcov = raw.get("vcov")
    if isinstance(raw_vcov, dict):
        sanitized_vcov: dict[str, dict[str, float]] = {}
        dropped_vcov: list[str] = []
        for row_key, row_value in raw_vcov.items():
            if row_key not in allowed_coefficient_keys:
                dropped_vcov.append(f"row {row_key!r}")
                continue
            if not isinstance(row_value, dict):
                dropped_vcov.append(f"row {row_key!r} (non-dict)")
                continue
            sanitized_row: dict[str, float] = {}
            for col_key, val in row_value.items():
                if col_key not in allowed_coefficient_keys:
                    dropped_vcov.append(f"{row_key}.{col_key}")
                    continue
                if not _is_finite_number(val):
                    continue
                sanitized_row[col_key] = float(val)
            if sanitized_row:
                sanitized_vcov[row_key] = sanitized_row
        if dropped_vcov:
            transformations.append(
                f"dropped {len(dropped_vcov)} undeclared key(s) from "
                f"'vcov': {sorted(dropped_vcov)[:5]}"
                + (" …" if len(dropped_vcov) > 5 else "")
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
        return SanitizerResult(
            ok=False, analysis_type="t_test",
            rejection_reason=(
                f"test_type must be one of {sorted(_TTEST_VALID_SUBTYPES)}, "
                f"got {subtype!r}"
            ),
        )

    n1 = raw.get("n1")
    if not isinstance(n1, int) or isinstance(n1, bool) or n1 < 0:
        return SanitizerResult(
            ok=False, analysis_type="t_test",
            rejection_reason=f"n1 must be a non-negative int, got {n1!r}",
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
                    f"{subtype} requires integer n2 >= 0, got {n2!r}"
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
            rejection_reason=f"n must be a non-negative int, got {n_raw!r}",
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
            return SanitizerResult(
                ok=False, analysis_type="frequency_table",
                rejection_reason=(
                    f"count value for {safe_key(k)!r} must be "
                    f"non-negative int, got {v!r}"
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
            return SanitizerResult(
                ok=False, analysis_type="frequency_table",
                rejection_reason=(
                    f"two distinct level names sanitize to the same "
                    f"key {clean_key!r} (e.g. embedded newlines or "
                    f"shared 40-char prefix). Collisions are rejected "
                    f"because aggregating the counts would defeat "
                    f"cell suppression on the smaller component. "
                    f"Disambiguate the levels in the source data."
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
    # use. Do this BEFORE the dict is built so collisions (if any) fold
    # together safely rather than bypassing the cleaner.
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
            return SanitizerResult(
                ok=False, analysis_type="crosstab",
                rejection_reason=(
                    f"counts[{safe_key(row_key)!r}] must be a dict "
                    f"(col_level → int); got {type(inner).__name__}"
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
                return SanitizerResult(
                    ok=False, analysis_type="crosstab",
                    rejection_reason=(
                        f"counts[{safe_row!r}][{safe_key(col_key)!r}] must "
                        f"be a non-negative int; got {v!r}"
                    ),
                )
            safe_col = safe_key(col_key)
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
    nested_raw: dict[str, dict[str, int | str]] = {}
    suppressed_cell_count = 0
    for (r, c), v in clean_counts.items():
        if r not in nested_raw:
            nested_raw[r] = {}
        if v < threshold:
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

    if suppressed_cell_count:
        transformations.append(
            f"primary suppression: {suppressed_cell_count} cell(s) "
            f"with count < {threshold} (cell labels withheld — "
            f"bucketed under '[suppressed]')"
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

    aggregation = raw.get("aggregation")
    if aggregation not in _MAGTAB_VALID_AGGREGATIONS:
        return SanitizerResult(
            ok=False, analysis_type="magnitude_table",
            rejection_reason=(
                f"aggregation must be one of "
                f"{sorted(_MAGTAB_VALID_AGGREGATIONS)}, got {aggregation!r}"
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
    raw_pruned = {k: v for k, v in raw.items() if k != "cells"}
    out = _collect_allowed(
        raw_pruned,
        string=_MAGTAB_ALLOWED_STRING_FIELDS,
        transformations=transformations,
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
            return SanitizerResult(
                ok=False, analysis_type="magnitude_table",
                rejection_reason=(
                    f"cells[{safe_key(raw_group)!r}] must be a dict with "
                    f"keys value, n, max_share; got {type(cell).__name__}"
                ),
            )
        value = cell.get("value")
        n = cell.get("n")
        max_share = cell.get("max_share")

        if not _is_finite_number(value):
            return SanitizerResult(
                ok=False, analysis_type="magnitude_table",
                rejection_reason=(
                    f"cells[{safe_key(raw_group)!r}].value must be a finite "
                    f"number; got {value!r}"
                ),
            )
        if not isinstance(n, int) or isinstance(n, bool) or n < 0:
            return SanitizerResult(
                ok=False, analysis_type="magnitude_table",
                rejection_reason=(
                    f"cells[{safe_key(raw_group)!r}].n must be a non-negative "
                    f"int; got {n!r}"
                ),
            )
        if not _is_finite_number(max_share):
            return SanitizerResult(
                ok=False, analysis_type="magnitude_table",
                rejection_reason=(
                    f"cells[{safe_key(raw_group)!r}].max_share must be a "
                    f"finite number; got {max_share!r}"
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
            rejection_reason=f"n must be a non-negative int, got {n_raw!r}",
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
                f"omitted, got {method!r}"
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

    # ``out["variables"]`` is the safe_key-transformed list (each
    # element passed through safe_key in _collect_allowed). The raw
    # ``correlations`` dict keys are not yet transformed. Compare on
    # safe_key both sides so a long or otherwise-transformed name
    # doesn't get spuriously "dropped as undeclared" simply because
    # the variables list shows the truncated form. Without this, a
    # legitimate matrix with a 50-char variable name returned with
    # ``correlations: {}`` and ``ok=True`` — silent empty success.
    declared = set(out.get("variables") or [])

    sanitized_corr: dict[str, dict[str, float]] = {}
    dropped: list[str] = []
    n = out["n"]
    for raw_row_key, row_value in correlations.items():
        if not isinstance(raw_row_key, str):
            dropped.append(f"row {raw_row_key!r} (non-string)")
            continue
        row_key = safe_key(raw_row_key)
        if row_key not in declared:
            dropped.append(f"row {safe_key(raw_row_key)!r}")
            continue
        if not isinstance(row_value, dict):
            dropped.append(f"row {row_key!r} (non-dict)")
            continue
        kept_row: dict[str, float] = {}
        for raw_col_key, val in row_value.items():
            if not isinstance(raw_col_key, str):
                dropped.append(f"{row_key}.{raw_col_key!r} (non-string)")
                continue
            col_key = safe_key(raw_col_key)
            if col_key not in declared:
                dropped.append(f"{row_key}.{col_key}")
                continue
            if not _is_finite_number(val):
                continue
            clamped = clamp_precision(float(val), n)
            # Clip to [-1, 1] in case precision-clamping nudged a near-
            # boundary value past it.
            clamped = max(-1.0, min(1.0, clamped))
            kept_row[col_key] = clamped
        if kept_row:
            sanitized_corr[row_key] = kept_row
    if dropped:
        transformations.append(
            f"dropped {len(dropped)} undeclared key(s) from "
            f"correlations: {sorted(dropped)[:5]}"
            + (" …" if len(dropped) > 5 else "")
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
