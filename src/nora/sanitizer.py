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

   **Partial mitigation in place.** Every variable-name-bearing field
   (`response_variable`, `cluster_variable`, `predictor_variables[*]`,
   `variable`, `row_variable`, `col_variable`, `value_variable`,
   correlation `variables[*]`) is gated by an identifier-shape regex
   (`_NAME_IDENT_RE`) before it reaches the model. Values that survive
   `safe_text` / `safe_key` (control-char strip, whitespace flatten,
   length cap) but don't match the column-name / coefficient-name
   character class — spaces, quotes, commas, semicolons, brackets,
   braces, equals, ampersand, slashes, dollar — are replaced with the
   empty string (scalars) or filtered out of the list (list-valued
   fields). This narrows the channel from "any 120-char arbitrary text"
   to "identifier-alphabet only", which blocks the dominant raw-data
   shapes (CSV rows, JSON dumps, error-message bodies). It does not
   close the gap for adversarial column names that already match the
   identifier shape — the runtime-library fix above is still the
   right long-term answer.

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
    clamp_dict_by_per_key_n,
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
    # Treated-cohort size threshold for DiD event-study suppression.
    # The new SDC primitive Callaway-Sant'Anna / de Chaisemartin /
    # Sun-Abraham introduce: min-N gate on the *treated-cohort size*
    # (carried in ``n_treated_per_group``), not on the cell count of
    # the ATT panel. A balanced panel can make cell counts look
    # comfortable (4 firms × 8 quarters = 32 cells) while the actual
    # disclosure unit is the 4 firms whose entire outcome trajectories
    # are summarized by the cohort's ATT series. Cohorts below this
    # threshold are dropped *whole* — partial-cell publication would
    # leak the cohort size through which cells survived.
    min_n_did_cohort: int = 10
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
    # IV / 2SLS / GMM diagnostics. All bounded scalars; none reach
    # into per-observation data. Decision made in
    # ``docs/direction.md`` "IV as regression-bucket extension":
    # 2SLS produces a single structural coefficient table plus a
    # handful of diagnostic scalars; the *first-stage* coefficient
    # table is rarely what the model needs (it just needs to know
    # whether the instruments are strong). Composite-shape territory
    # is reserved for genuine multi-stage estimators (3SLS,
    # mediation with separate exposure→mediator and mediator→outcome
    # regressions, control-function corrections).
    #   - first_stage_f: minimum F-statistic across endogenous
    #     variables, testing joint significance of excluded
    #     instruments in the first stage. Stock-Yogo rule of thumb
    #     flags weak instruments below ~10.
    #   - weak_instrument_p: p-value associated with first_stage_f
    #     (when computed) — sometimes more useful than the F itself
    #     for non-standard sample sizes.
    #   - hansen_j / hansen_j_p: overidentification test statistic
    #     and its p-value. Hansen J under heteroskedasticity;
    #     Sargan under homoskedasticity. Only defined when the
    #     number of instruments exceeds the number of endogenous
    #     regressors (overidentified case).
    #   - endogeneity_p: Wu-Hausman / Durbin-Wu-Hausman test
    #     p-value — whether OLS and IV estimates diverge enough to
    #     justify using IV at all.
    "first_stage_f", "weak_instrument_p",
    "hansen_j", "hansen_j_p",
    "endogeneity_p",
    # Intraclass correlation — fraction of total variance attributable
    # to the random-effects group. For a one-level model:
    # icc = sigma_u² / (sigma_u² + sigma_e²). Bounded [0, 1], no
    # disclosure surface beyond what ``random_effects_variance`` already
    # carries; emitted for inference adequacy so the model can cite
    # "ρ = 0.42, school explains 42% of variance" without the
    # researcher computing it post-hoc.
    "icc",
))
_OLS_ALLOWED_INT_FIELDS: frozenset[str] = frozenset((
    "n", "degrees_of_freedom",
    # Survival-specific sample metadata. ``n`` for stcox is the number
    # of records (post-stset, can include split episodes per subject);
    # ``n_subjects`` and ``n_failures`` are what the researcher
    # actually reads off a Cox table — "324 subjects, 178 events" — so
    # they need to reach the model alongside the coefficients.
    "n_subjects", "n_failures",
    # IV: count of instruments and count of endogenous regressors.
    # Both are dimension cardinalities of the design (overidentification
    # = ``n_instruments > n_endogenous``), not data-derived quantities.
    "n_instruments", "n_endogenous",
))
_OLS_ALLOWED_STRING_FIELDS: frozenset[str] = frozenset((
    # ``cluster_variable`` (singular) kept as back-compat — older
    # payloads in the SQLite store carry it. New helpers emit the
    # plural list form via ``cluster_variables`` below so multi-way
    # clustering (Cameron-Gelbach-Miller two-way and beyond) round-
    # trips natively without a schema change.
    "type", "response_variable", "robust_se_type", "cluster_variable",
    # Estimation method, primarily for mixed-effects: ``REML`` or
    # ``ML``. Also useful on GLM family fits if the caller wants to
    # surface the link function or scale-estimator choice. Free-text
    # is bounded to ~40 chars via ``safe_text``; for known enum
    # values the model interprets directly, others are still safe
    # because they go through text-safety.
    "fit_method",
))
_OLS_ALLOWED_DICT_NUMERIC: frozenset[str] = frozenset((
    "coefficients", "standard_errors", "t_statistics", "p_values",
    # Variance-inflation factors, one per predictor. Cross-field
    # key validation (further down in _sanitize_linear_regression)
    # restricts the keys to declared predictor names + the
    # intercept aliases, so this dict can't be used to smuggle
    # arbitrary numeric channels.
    "vif",
    # Absorbed fixed-effects cardinality, one entry per FE dimension.
    # Keys are FE-variable names (dataset column names); values are
    # the count of distinct levels in that dimension. The cardinality
    # is the disclosure-relevant quantity ("firm FE absorbed, 1,247
    # levels"); the level identities themselves are NOT in this dict
    # by construction — fixest emits sizes, not labels. Excluded from
    # the coefficient-name cross-field check below because FE-var keys
    # are exactly the names NOT in predictor_variables (they're
    # absorbed, not regressors of interest).
    "fixed_effects",
    # Cluster-robust SE cardinality, one entry per clustering
    # dimension. Same shape and disclosure profile as
    # ``fixed_effects`` — keys are clustering-variable names (dataset
    # columns the model already saw in the schema), values are
    # cluster counts ("clustered at firm, 1,247 clusters"; for two-
    # way clustering, both entries are present). Decision codified
    # here per the previous turn's "modifier vs new sub-shape" rule:
    # bounded aggregate scalars and counts go in the existing OLS
    # allowlist, structured shapes get their own type. Clustering
    # cardinalities are bounded counts → allowlist, not a new shape.
    "n_clusters",
    # Mixed-effects variance components, one entry per random-effect
    # group (and one ``residual`` entry for the residual variance).
    # Keys are random-effects-factor names (dataset column names);
    # values are variance components from the random-effects
    # covariance matrix. Random-slope models contribute multiple
    # entries per group keyed like ``school.x`` (slope on x within
    # school). The intercept-slope covariance is NOT emitted in this
    # field — keep the disclosure surface to variances only; the
    # full random-effects covariance is reachable through ``vcov``
    # if a researcher genuinely needs it. Same skip-coef-key-check
    # treatment as fixed_effects — keys are NOT predictor names.
    "random_effects_variance",
    # Per-level group counts for mixed-effects models — dataset
    # column names → number of distinct groups in that level. Two-
    # level model ``y ~ x + (1|school) + (1|classroom)`` emits
    # ``{school: 30, classroom: 60}``. Same disclosure profile and
    # treatment as ``fixed_effects`` and ``n_clusters``.
    "n_groups_per_level",
))
# Dict-numeric fields whose keys are NOT coefficient names, so the
# coefficient-name cross-field validation below must skip them.
# ``fixed_effects`` keys = absorbed-FE-var names; ``n_clusters`` keys
# = clustering-var names. Both are dataset column names that are
# deliberately NOT coefficients of interest. VIF, coefficients,
# standard_errors, t_statistics, p_values all use coefficient names
# and must remain inside the cross-field check.
_OLS_DICT_FIELDS_SKIP_COEF_KEY_CHECK: frozenset[str] = frozenset((
    "fixed_effects", "n_clusters",
    # Random-effects entries are keyed by RE-factor name (possibly
    # with a ``.term`` suffix for random slopes); never by coefficient
    # name. Same exclusion as fixed_effects.
    "random_effects_variance", "n_groups_per_level",
))
# Dict-numeric fields holding integer COUNTS (cardinalities) rather
# than data-precision measurements. Skipped from the sigfigs clamp
# (1247 firms shouldn't round to 1250 at sigfigs=3) and coerced to
# int rather than left as float. Same rule for FE level counts,
# cluster counts, and mixed-effects per-level group counts.
_OLS_DICT_FIELDS_INT_COUNTS: frozenset[str] = frozenset((
    "fixed_effects", "n_clusters", "n_groups_per_level",
))
# Canonical name of the regression-bucket payload type, plus its
# legacy alias. The bucket holds OLS / logit / probit / Poisson /
# negative binomial / Cox PH / fixest / 2SLS — anything that emits a
# coefficient table with associated fit statistics. The original
# name ``linear_regression`` misled both readers and the model into
# thinking the scope was OLS-only (see the audit arc that found
# Cox hard-failing through the helper and GLMs shipping no fit
# metrics). The descriptive name ``coefficient_table_with_fit_stats``
# is the new canonical; ``linear_regression`` is kept as an alias
# so payloads from older sessions, older helpers, and the existing
# SQLite stores on researcher disks still sanitize and render.
_REGRESSION_TYPE_CANONICAL: str = "coefficient_table_with_fit_stats"
_REGRESSION_TYPE_LEGACY: str = "linear_regression"
_REGRESSION_TYPE_ALIASES: frozenset[str] = frozenset((
    _REGRESSION_TYPE_CANONICAL, _REGRESSION_TYPE_LEGACY,
))


def _emitted_regression_type(raw: dict[str, Any]) -> str:
    """Return the type string to stamp on the sanitized output for a
    regression-bucket payload.

    Round-trips the input's ``type`` field if it's one of the
    recognised aliases. Defaults to the legacy name so the value
    never appears unset — but a well-formed payload always carries
    one of the aliases here because the dispatch table only routes
    those two strings to this sanitizer.
    """
    raw_type = raw.get("type")
    if isinstance(raw_type, str) and raw_type in _REGRESSION_TYPE_ALIASES:
        return raw_type
    return _REGRESSION_TYPE_LEGACY


_OLS_ALLOWED_LIST_STRING: frozenset[str] = frozenset((
    "predictor_variables",
    # IV / 2SLS supplementary identifiers. ``instrument_variables``
    # names the excluded instruments (their cardinality goes through
    # ``n_instruments``); ``endogenous_variables`` names which of
    # the predictors are treated as endogenous. Both are bounded
    # name lists — each entry goes through ``safe_key`` (40-char
    # cap, control-char stripping) — so they live in the same
    # disclosure-budget envelope as ``predictor_variables``.
    "instrument_variables", "endogenous_variables",
    # Plural cluster-variable list — multi-way clustering shows up
    # by emitting a list rather than the singular string field.
    # Helpers should emit this going forward; ``cluster_variable``
    # singular is kept for back-compat with stored payloads.
    "cluster_variables",
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
# are NEVER accepted in this payload type. The opt-in path that
# previously let an explicitly-listed variable's min/max pass through
# was unsafe under the threat model: ``source_dataset``, ``variable``,
# and the values themselves are all model/script-controlled, and the
# sanitizer cannot prove that a payload labeled ``variable="age"``
# carries age's min/max rather than (eg) salary's. A typed helper
# stamping a provenance marker doesn't close it either — the marker
# only proves the helper was called; the caller still chooses which
# column to summarize and what to label it. Closing the channel
# requires a Nora-owned dataset-load path, which is out of scope here.
# The ``non_disclosive_variables`` policy field stays for documented
# intent and forward compatibility, but is inert in this sanitizer.
# Median / quartiles remain forbidden too — use ``request_data``
# ``quartiles`` for those (which is Nora-owned and IQR-clamped).
_DESC_ALLOWED_NUMERIC_FIELDS: frozenset[str] = frozenset(("mean", "sd"))
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


def _correlation_invariants_hold(
    corr: dict[str, dict[str, float]],
    declared: set[str],
) -> tuple[bool, str]:
    """Verify the aggregate invariants of a real correlation matrix.

    A correlation matrix from ``df.corr()`` is square, symmetric,
    has 1s on the main diagonal, and (Pearson) is bounded in
    [-1, 1]. The per-cell checks already clip to [-1, 1]; this
    function adds:

    * **Completeness**: every declared variable has a row whose
      columns cover every other declared variable. A partial matrix
      could otherwise be used to thread free cells past the
      declared-variables cap.
    * **Diagonals = 1**: a real Pearson / Spearman / Kendall
      diagonal is exactly 1 (or, with float round-trip noise,
      within a small epsilon of 1). Any other value indicates a
      hand-crafted matrix.
    * **Symmetry**: ``corr[i][j] == corr[j][i]`` within a small
      tolerance. Halves the bandwidth attacker-engineered values
      could otherwise use.

    Returns ``(ok, reason)``. ``reason`` is suitable for the
    sanitizer's ``rejection_reason``; it does not echo any specific
    correlation value back to the model (just structural facts).
    """
    REL_TOL = 1e-3
    ABS_TOL = 1e-9
    # Completeness: every declared variable must have its own row,
    # and that row must cover every other declared variable.
    for var in declared:
        row = corr.get(var)
        if row is None:
            return (False, f"missing row for declared variable {var!r}")
        for other in declared:
            if other not in row:
                return (
                    False,
                    f"row {var!r} missing column {other!r} "
                    f"(declared but unreported correlation)"
                )
    # Diagonals = 1.
    for var in declared:
        diag = corr[var][var]
        if abs(diag - 1.0) > ABS_TOL + REL_TOL:
            return (
                False,
                f"diagonal not 1.0: corr[{var!r}][{var!r}]={diag!r}"
            )
    # Symmetry.
    for row_var in declared:
        row = corr[row_var]
        for col_var in declared:
            if col_var == row_var:
                continue
            val = row[col_var]
            partner = corr[col_var][row_var]
            diff = abs(val - partner)
            scale = max(abs(val), abs(partner), 1.0)
            if diff > ABS_TOL + REL_TOL * scale:
                return (
                    False,
                    f"asymmetric: corr[{row_var!r}][{col_var!r}]={val!r} "
                    f"!= corr[{col_var!r}][{row_var!r}]={partner!r}"
                )
    return (True, "")


def _vcov_invariants_hold(
    vcov: dict[str, dict[str, float]],
    standard_errors: dict[str, float],
) -> tuple[bool, str]:
    """Verify the aggregate invariants of a variance-covariance matrix.

    Returns ``(ok, reason)``. A real cov matrix from σ²·(X'X)^-1 is
    symmetric (``vcov[i][j] == vcov[j][i]``) and its diagonals are
    the squared standard errors. Anything emitted through the
    generic ``result(type="linear_regression", vcov=...)`` escape
    hatch can carry arbitrary numeric cells; without these checks a
    hostile payload smuggles up to N² scalar values to the model
    through ``expand_result``. The checks reject the whole matrix
    rather than per-cell so an attacker can't slip a small number
    of inconsistent cells through.

    Tolerance: symmetry is checked against a small relative epsilon
    that accommodates float round-trip noise from JSON. Diagonals
    compare ``vcov[i][i]`` against ``standard_errors[i]**2`` with a
    similarly relaxed bound so legitimately-fit models with
    ill-conditioned designs still pass.
    """
    REL_TOL = 1e-3
    ABS_TOL = 1e-9
    # Symmetry: every off-diagonal cell must have a partner across
    # the main diagonal with a near-equal value.
    for row, inner in vcov.items():
        for col, val in inner.items():
            if row == col:
                continue
            partner_inner = vcov.get(col)
            if partner_inner is None or row not in partner_inner:
                return (
                    False,
                    f"asymmetric: vcov[{row!r}][{col!r}] present but "
                    f"vcov[{col!r}][{row!r}] missing"
                )
            partner = partner_inner[row]
            diff = abs(val - partner)
            scale = max(abs(val), abs(partner), 1.0)
            if diff > ABS_TOL + REL_TOL * scale:
                return (
                    False,
                    f"asymmetric: vcov[{row!r}][{col!r}]={val!r} "
                    f"!= vcov[{col!r}][{row!r}]={partner!r}"
                )
    # Diagonals match SE². Intercept aliases in ``standard_errors``
    # may use a different name than the vcov diagonal key (the
    # sanitizer accepts a few intercept aliases); skip the check
    # when no matching SE entry exists rather than over-reject.
    for row, inner in vcov.items():
        diag = inner.get(row)
        if diag is None:
            continue
        if diag < 0:
            return (
                False,
                f"negative variance: vcov[{row!r}][{row!r}]={diag!r}"
            )
        se = standard_errors.get(row)
        if se is None:
            continue
        expected = float(se) ** 2
        diff = abs(diag - expected)
        scale = max(abs(diag), abs(expected), 1.0)
        if diff > ABS_TOL + REL_TOL * scale:
            return (
                False,
                f"diagonal mismatch: vcov[{row!r}][{row!r}]={diag!r} "
                f"!= standard_errors[{row!r}]**2={expected!r}"
            )
    return (True, "")


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
# Identifier-shape gate for variable-name fields
# ---------------------------------------------------------------------------
#
# Background. Allowlisted string fields (``response_variable``,
# ``predictor_variables``, ``variable``, ``row_variable``,
# ``value_variable``, ``col_variable``, correlation ``variables``) are
# supposed to carry COLUMN NAMES — short identifiers chosen by the
# researcher in their data file. ``safe_text`` / ``safe_key`` neutralise
# prompt-injection text (control chars, newlines, length) but place no
# constraint on the character class. After whitespace-flattening, a
# value like ``"x SYSTEM: ignore previous"``, a CSV row
# ``'"John Smith",25,"Boston, MA",50000'``, or a JSON dump
# ``'{"id": 123, "ssn": "..."}'`` all survive ``safe_text`` and reach
# the model verbatim through the allowlist.
#
# Documented "Known gap #1" at the top of this file is exactly this:
# nothing ties these fields back to the dataset schema. The ideal fix
# lives in the runtime library (require a model object, derive names
# from ``xlevels``), but a partial sanitizer-side defence is cheap:
# require these fields to MATCH AN IDENTIFIER SHAPE before they're
# echoed back. The shape admits every character a legitimate column /
# coefficient name could plausibly contain (letters, digits,
# underscore, period for SQL/R/Python convention; parens, colon, hash,
# caret for R / Stata formula operators like ``factor(x)Asia``,
# ``I(age^2)``, ``age:sex``, ``c.age#c.sex``) and EXCLUDES the
# characters that raw CSV rows / JSON dumps / error-message bodies
# would carry (spaces, quotes, commas, semicolons, brackets, braces,
# equals, ampersand, slashes, dollar, asterisk).
#
# Threat narrowed, not eliminated: a script that already controls the
# dataset's column names (e.g. dataset prepared by the same hostile
# upstream) can still encode bits in shapes that pass — but bandwidth
# drops by an order of magnitude (40-char identifier alphabet vs.
# 120-char arbitrary-text alphabet), and the most common raw-data
# shapes (CSV rows, JSON, error bodies, secrets with ``=`` or ``-``
# separators) are filtered out.
#
# When a value fails the gate, it's replaced with the empty string and
# a transformation log entry records the drop. For LIST-valued fields
# (``predictor_variables``, correlation ``variables``) non-conforming
# entries are removed from the list — coefficient-dict keys are
# already filtered through the resulting list elsewhere in this
# module, so dropping a predictor here automatically drops its
# coefficient / SE / t / p / vif entries.
_NAME_IDENT_RE = re.compile(r"^[A-Za-z0-9_.(][A-Za-z0-9_.():^#]*$")

# ``safe_text`` / ``safe_key`` append this marker when an input
# exceeds the cap. A legitimate over-length identifier (rare but
# valid — long column names in user datasets) lands here as
# ``"<prefix>[TRUNCATED]"``; the bracket chars aren't in the regex
# character class above, so we strip the suffix before matching.
_TRUNCATION_TAIL = "[TRUNCATED]"


def _is_identifier_shape(value: str) -> bool:
    """True iff ``value`` matches the column-name / coefficient-name shape.

    Empty string is treated as non-identifier (callers that legitimately
    have already dropped a value should not re-enter this gate). The
    trailing ``[TRUNCATED]`` marker emitted by ``safe_text`` / ``safe_key``
    on over-length inputs is tolerated — the marker chars are not in
    the identifier alphabet, but their presence on the suffix is a
    sanitizer-controlled signal, not caller-controlled bytes.
    """
    if not value:
        return False
    body = (
        value[: -len(_TRUNCATION_TAIL)]
        if value.endswith(_TRUNCATION_TAIL)
        else value
    )
    return bool(_NAME_IDENT_RE.fullmatch(body))


def _enforce_identifier_string_fields(
    out: dict[str, Any],
    fields: frozenset[str],
    transformations: list[str],
    *,
    type_label: str,
) -> None:
    """Replace non-identifier-shape string fields with the empty string.

    Only fields actually present in ``out`` are checked. Mutates ``out``
    in place. Logs one transformation line per dropped field, naming
    the FIELD (which is sanitizer-controlled, not data-derived) and
    withholding the rejected VALUE (which is caller-controlled).
    """
    for field in fields:
        v = out.get(field)
        if not isinstance(v, str) or not v:
            continue
        if not _is_identifier_shape(v):
            out[field] = ""
            transformations.append(
                f"dropped {field!r} from {type_label} payload: value did "
                f"not match the column-name / coefficient-name identifier "
                f"shape (value withheld — caller-controlled)"
            )


def _enforce_identifier_list_field(
    out: dict[str, Any],
    field: str,
    transformations: list[str],
    *,
    type_label: str,
) -> None:
    """Filter a list-of-strings field to entries matching the identifier shape.

    Mutates ``out[field]`` in place. Logs a single transformation line
    with the COUNT of dropped entries (names withheld — entries are
    caller-controlled). If every entry is dropped the field becomes
    an empty list; downstream ``_require_after_filter`` may then
    reject the payload, which is the desired hard-fail behaviour.
    """
    raw_list = out.get(field)
    if not isinstance(raw_list, list):
        return
    kept = [x for x in raw_list if isinstance(x, str) and _is_identifier_shape(x)]
    dropped = len(raw_list) - len(kept)
    if dropped:
        out[field] = kept
        transformations.append(
            f"dropped {dropped} non-identifier-shape entry(ies) from "
            f"{field!r} in {type_label} payload (names withheld — "
            f"caller-controlled)"
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
            ok=False, analysis_type=_emitted_regression_type(raw),
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
            ok=False, analysis_type=_emitted_regression_type(raw),
            rejection_reason=(
                f"n must be a non-negative int, got {type(n_raw).__name__}"
            ),
        )

    try:
        require_minimum_n(n_raw, config.min_n_regression, "n")
    except MinimumNViolation as e:
        return SanitizerResult(
            ok=False, analysis_type=_emitted_regression_type(raw),
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
            ok=False, analysis_type=_emitted_regression_type(raw),
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
            ok=False, analysis_type=_emitted_regression_type(raw),
            rejection_reason=missing_after_filter,
        )

    # Identifier-shape gate. ``response_variable`` and
    # ``cluster_variable`` carry single column names; each entry of
    # ``predictor_variables`` carries a coefficient name. See
    # ``_enforce_identifier_*`` above for the threat model. This gate
    # runs BEFORE the categorical-contrast / cross-field-key checks so
    # those downstream passes see the cleaned predictor list — a
    # predictor dropped here automatically loses its coefficient / SE /
    # t / p / vif entries because ``declared_predictors`` is recomputed
    # from ``out["predictor_variables"]`` below.
    _enforce_identifier_string_fields(
        out,
        frozenset(("response_variable", "cluster_variable")),
        transformations,
        type_label="linear_regression",
    )
    _enforce_identifier_list_field(
        out, "predictor_variables", transformations,
        type_label="linear_regression",
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
            ok=False, analysis_type=_emitted_regression_type(raw),
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
        # ``fixed_effects`` (and any future dict-numeric field whose
        # keys aren't coefficient names) lives outside this check by
        # design — see _OLS_DICT_FIELDS_SKIP_COEF_KEY_CHECK.
        if dict_field in _OLS_DICT_FIELDS_SKIP_COEF_KEY_CHECK:
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
            # Aggregate-consistency check. A real variance-covariance
            # matrix from σ²·(X'X)^-1 is symmetric and its diagonals
            # are SE². Generic ``result(type="linear_regression",
            # vcov={...})`` bypasses the typed helper and can carry
            # arbitrary numeric cells; the key + finiteness checks
            # above don't catch that. Reject the whole vcov if either
            # invariant fails — a real model never produces such a
            # matrix, and accepting it would let the script smuggle
            # up to N² cells of attacker-shaped numeric data through
            # to the model via expand_result.
            vcov_ok, reject_reason = _vcov_invariants_hold(
                sanitized_vcov, out.get("standard_errors") or {},
            )
            if vcov_ok:
                out["vcov"] = sanitized_vcov
            else:
                transformations.append(
                    f"dropped vcov entirely: {reject_reason}"
                )

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
            # Cardinality dicts (FE level counts, cluster counts)
            # carry integer counts describing dataset structure, not
            # data-derived measurements that scale precision with N.
            # Round to int rather than running through the sigfig
            # clamp, which would distort small counts (1247 → 1250
            # at sigfigs=3). Same rule applies to any future
            # cardinality-dict field — extend the set, not the branch.
            if key in _OLS_DICT_FIELDS_INT_COUNTS:
                out[key] = {
                    k: int(round(v)) for k, v in out[key].items()
                    if isinstance(v, (int, float)) and v >= 0
                }
            else:
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
        ok=True, analysis_type=_emitted_regression_type(raw),
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
    # min_value / max_value are NEVER passed through here — see the
    # comment on ``_DESC_ALLOWED_NUMERIC_FIELDS`` for why. The opt-in
    # mechanism the prior code implemented (per-variable allowance
    # via ``config.non_disclosive_variables``) was unsafe because
    # nothing in the payload binds the reported values to the named
    # variable's actual column. Researchers who need a variable's
    # range should use a Nora-owned path (eg ``request_data``).
    numeric_allowlist = _DESC_ALLOWED_NUMERIC_FIELDS
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

    # Identifier-shape gate on ``variable`` (the single column name
    # this descriptive stat describes). Non-conforming values are
    # replaced with the empty string; see ``_enforce_identifier_*``
    # for the threat model.
    _enforce_identifier_string_fields(
        out, frozenset(("variable",)), transformations,
        type_label="descriptive",
    )

    n = out["n"]
    for key in numeric_allowlist:
        if key in out:
            out[key] = clamp_precision(out[key], n)
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

    # Identifier-shape gate on ``variable`` (the column this table
    # tabulates). LEVEL names in ``counts`` keys are data values, not
    # identifiers, and remain governed by ``safe_key`` + the structural
    # cell cap.
    _enforce_identifier_string_fields(
        out, frozenset(("variable",)), transformations,
        type_label="frequency_table",
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

    # Identifier-shape gate on the row/col VARIABLE NAMES (column
    # names of the two factors being crosstabbed). Cell LEVEL names
    # in ``counts`` are data values and stay governed by ``safe_key``
    # + the structural cell cap.
    _enforce_identifier_string_fields(
        out, frozenset(("row_variable", "col_variable")), transformations,
        type_label="crosstab",
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

    # Identifier-shape gate on the row/value VARIABLE NAMES. Cell
    # GROUP labels (``cells`` keys) are data values and remain
    # governed by ``safe_key`` + structural cell cap.
    _enforce_identifier_string_fields(
        out,
        frozenset(("row_variable", "value_variable")),
        transformations,
        type_label="magnitude_table",
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

    # Identifier-shape gate on each entry of ``variables`` (column
    # names being correlated). Entries that fail are dropped before
    # ``declared`` is built — corresponding rows/cols in the
    # ``correlations`` dict will then be dropped as "undeclared" by
    # the cross-field validation below, with the same counters.
    _enforce_identifier_list_field(
        out, "variables", transformations,
        type_label="correlation_matrix",
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

    # Aggregate invariants for a real correlation matrix. The
    # per-cell checks above (declared keys, finiteness, clip to
    # [-1, 1]) don't catch a matrix that's asymmetric, has
    # off-diagonal cells without their transpose partner, or has
    # diagonals != 1. A real ``df.corr()`` always produces these
    # invariants; an attacker emitting through generic
    # ``result(type="correlation_matrix", ...)`` to smuggle numeric
    # values is the only realistic origin of a violation.
    #
    # Reject the whole payload rather than the matrix alone: unlike
    # vcov (which sits alongside coefficients/SE/etc.), the
    # correlations field IS the payload, and a correlation_matrix
    # without correlations is meaningless.
    invariants_ok, reject_reason = _correlation_invariants_hold(
        sanitized_corr, declared,
    )
    if not invariants_ok:
        return SanitizerResult(
            ok=False, analysis_type="correlation_matrix",
            rejection_reason=(
                f"correlation matrix failed aggregate-invariant check: "
                f"{reject_reason}. Real correlation matrices are "
                f"symmetric with 1s on the diagonal; a violation here "
                f"means the payload didn't come from a ``df.corr()``-"
                f"shaped computation, which is required by the SDC "
                f"posture for this result type."
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
# DiD event study (Callaway-Sant'Anna / de Chaisemartin-D'Haultfœuille /
# Sun-Abraham / TWFE event study)
# ---------------------------------------------------------------------------
#
# The modern-DiD literature has moved decisively to heterogeneous-
# treatment estimators that decompose into ATT(g, t) — average
# treatment effect on the treated, indexed by treatment cohort g and
# event time t (calendar time relative to treatment). Callaway-
# Sant'Anna (the ``did`` R package), de Chaisemartin-
# D'Haultfœuille (``DIDmultiplegt``), and Sun-Abraham (the
# ``fixest::sunab`` interaction-weighted estimator) all produce
# this shape, plus the older TWFE event-study with leads-and-lags
# coefficients indexed by event time.
#
# **The new SDC primitive this shape introduces** is min-N gated by
# the treated-cohort size, NOT by the cell count of the ATT panel.
# Concrete reason: in strategy / finance / applied micro, treated
# cohorts of 3-10 firms are normal (mergers, IPOs, regulatory
# events). A balanced panel can make the cell count of ATT(g, t)
# look comfortable (4 firms × 8 quarters = 32 "observations") while
# the actual disclosure unit is those 4 firms whose outcome
# trajectories are summarized by the ATT series for cohort g.
# Combined with knowledge that the cohort was treated at calendar
# time T (often public), the ATT series leaks firm-level outcome
# changes if the cohort is small.
#
# Suppression rule: any cohort g with ``n_treated_per_group[g] <
# min_n_did`` gets ALL its cells dropped from ``att`` and the
# per-cell SE / p / CI dicts. Whole-cohort suppression is
# mandatory — partial-cell publication would leak the cohort size
# through *which* cells survived. The cohort label itself is also
# withheld (the marker key is ``[suppressed]``), since the
# cohort label is data-derived (it's typically the treatment date
# / cohort id and identifies the cohort directly).
#
# Cross-field validation: ``att`` is a nested {group: {event_time:
# value}} dict. Every outer key must be in ``groups``; every inner
# key must be in ``event_times``. ``standard_errors`` / ``p_values``
# / ``ci_lower`` / ``ci_upper`` mirror that shape and validate the
# same way. ``n_treated_per_group`` outer keys must equal ``groups``.

# Required structural fields. ``att`` and ``n_treated_per_group``
# are load-bearing — without the latter the cohort-N gate has no
# input and SDC degenerates to "trust the script". The aggregate
# ATT block is optional (a study might only report the matrix).
_DID_EVENT_REQUIRED: frozenset[str] = frozenset((
    "type", "groups", "event_times", "att", "n_treated_per_group",
))
_DID_EVENT_ALLOWED_NUMERIC_FIELDS: frozenset[str] = frozenset((
    "aggregate_att", "aggregate_se", "aggregate_p_value",
    "aggregate_ci_lower", "aggregate_ci_upper",
    "pre_trends_chi_squared", "pre_trends_p_value",
))
_DID_EVENT_ALLOWED_INT_FIELDS: frozenset[str] = frozenset((
    # Anticipation window the estimator was told to assume. Callaway-
    # Sant'Anna's ``anticipation`` arg specifies how many periods
    # before treatment the treatment effect may "leak in" — shifting
    # which pre-periods are usable as controls. A scalar count, no
    # disclosure risk; surfaced so the model can report "...assuming
    # zero anticipation" or call out a non-default value.
    "anticipation_periods",
    "n_pre_treatment_periods", "n_post_treatment_periods",
))
_DID_EVENT_ALLOWED_STRING_FIELDS: frozenset[str] = frozenset((
    "type", "estimator", "outcome_variable", "treatment_variable",
    "aggregation_method",
    # Which units serve as the control group during the differencing
    # step. CS / dCdH let you pick:
    #   * ``nevertreated`` — only units that NEVER receive treatment.
    #     Stricter but loses data when the never-treated cohort is
    #     small or absent.
    #   * ``notyettreated`` — units that will be treated later are
    #     also valid controls until their own treatment date.
    # Pinned by ``_DID_VALID_COMPARISON_GROUP`` below.
    "comparison_group",
    # CS / dCdH ``base_period`` rule: which pre-treatment period
    # serves as the reference for each cohort. ``varying`` (the R
    # ``did`` package default) re-bases for each (g, t) pair to the
    # immediately-pre-treatment period. ``universal`` fixes the base
    # period across all (g, t) pairs to a single period.
    "base_period",
))
_DID_VALID_COMPARISON_GROUP: frozenset[str] = frozenset((
    "nevertreated", "notyettreated",
    # snake_case variants — accept both so the helper doesn't have
    # to choose between the R package's no-underscore form and the
    # more readable Python idiom.
    "never_treated", "not_yet_treated",
))
_DID_VALID_BASE_PERIOD: frozenset[str] = frozenset((
    "varying", "universal",
))
_DID_EVENT_ALLOWED_LIST_STRING: frozenset[str] = frozenset((
    "groups",
))
_DID_EVENT_ALLOWED_LIST_NUMERIC: frozenset[str] = frozenset((
    "event_times",
))
# Nested-dict (group → event_time → value) fields. Each gets the
# same cohort suppression and cross-field validation pass.
_DID_EVENT_NESTED_DICT_FIELDS: frozenset[str] = frozenset((
    "att", "standard_errors", "p_values", "ci_lower", "ci_upper",
))
# Flat per-group dicts. ``n_treated_per_group`` is the SDC-relevant
# one (drives the cohort gate); ``n_control_per_cell`` is optional
# secondary metadata if the script computed cell-level control N.
_DID_EVENT_PER_GROUP_INT_FIELDS: frozenset[str] = frozenset((
    "n_treated_per_group",
))
# Structural caps on the panel dimensions. A real Callaway-Sant'Anna
# study reports a handful of cohorts (treatment-year cohorts in a
# DiD design) over a window of event times (typically ±5 to ±10).
# A 50-cohort × 30-event-time panel is already 1500 cells of
# disclosure surface; bigger numbers are almost always a sign the
# script is shipping disaggregated data through this channel.
_DID_EVENT_MAX_GROUPS: int = 50
_DID_EVENT_MAX_EVENT_TIMES: int = 30
_DID_EVENT_VALID_AGGREGATION: frozenset[str] = frozenset((
    "overall", "by_group", "by_event_time", "simple",
    "dynamic", "calendar",  # Callaway-Sant'Anna aggregator names
    # ``event`` is the Python ``differences`` package's name for the
    # same event-time aggregation R's ``did`` calls ``dynamic``.
    # Accept both so the helper doesn't have to normalize.
    "event", "group",
))
_DID_EVENT_VALID_ESTIMATOR: frozenset[str] = frozenset((
    "callaway_santanna", "de_chaisemartin", "sun_abraham",
    "twfe_event_study", "twfe",
))


def _sanitize_did_event_study(
    raw: dict[str, Any], config: SDCConfig
) -> SanitizerResult:
    missing_reason = _require_fields(raw, _DID_EVENT_REQUIRED, "did_event_study")
    if missing_reason:
        return SanitizerResult(
            ok=False, analysis_type="did_event_study",
            rejection_reason=missing_reason,
        )

    # Validate groups list shape and cap before anything else — the
    # cohort identifiers gate the cross-field key validation below.
    groups_raw = raw.get("groups")
    if not isinstance(groups_raw, list) or not all(
        isinstance(x, str) for x in groups_raw
    ):
        return SanitizerResult(
            ok=False, analysis_type="did_event_study",
            rejection_reason=(
                "groups must be a list of strings (cohort identifiers); "
                f"got {type(groups_raw).__name__}"
            ),
        )
    if len(groups_raw) > _DID_EVENT_MAX_GROUPS:
        return SanitizerResult(
            ok=False, analysis_type="did_event_study",
            rejection_reason=(
                f"groups has {len(groups_raw)} cohorts; the structural "
                f"cap is {_DID_EVENT_MAX_GROUPS}. A real Callaway-Sant'Anna "
                f"/ event-study analysis ships a handful of cohorts; "
                f"larger payloads are rejected as probable adversarial."
            ),
        )
    if len(groups_raw) == 0:
        return SanitizerResult(
            ok=False, analysis_type="did_event_study",
            rejection_reason="groups is empty",
        )

    # Validate event_times — list of finite numbers (typically ints
    # but allow floats; sanitize to int when integer-valued for nice
    # JSON, otherwise keep float).
    event_times_raw = raw.get("event_times")
    if not isinstance(event_times_raw, list) or not all(
        _is_finite_number(x) for x in event_times_raw
    ):
        return SanitizerResult(
            ok=False, analysis_type="did_event_study",
            rejection_reason=(
                "event_times must be a list of finite numbers; "
                f"got {type(event_times_raw).__name__}"
            ),
        )
    if len(event_times_raw) > _DID_EVENT_MAX_EVENT_TIMES:
        return SanitizerResult(
            ok=False, analysis_type="did_event_study",
            rejection_reason=(
                f"event_times has {len(event_times_raw)} entries; the "
                f"structural cap is {_DID_EVENT_MAX_EVENT_TIMES}."
            ),
        )

    # n_treated_per_group: required, dict[str, int]. This is the
    # SDC primitive's input — every cohort must declare its treated
    # size or the cohort-N gate can't run.
    n_treated_raw = raw.get("n_treated_per_group")
    if not isinstance(n_treated_raw, dict):
        return SanitizerResult(
            ok=False, analysis_type="did_event_study",
            rejection_reason=(
                "n_treated_per_group must be a dict mapping cohort id "
                "to treated-unit count; "
                f"got {type(n_treated_raw).__name__}"
            ),
        )

    transformations: list[str] = []

    # Sanitize group labels and build the declared-cohort set after
    # safe_key normalization. Reject safe_key collisions outright
    # mirrors the magnitude_table / crosstab pattern.
    safe_groups: list[str] = []
    safe_groups_set: set[str] = set()
    for raw_g in groups_raw:
        sg = safe_key(raw_g)
        if sg in safe_groups_set:
            return SanitizerResult(
                ok=False, analysis_type="did_event_study",
                rejection_reason=(
                    "cohort label collision after sanitization in "
                    "did_event_study.groups: two distinct raw labels "
                    "sanitize to the same safe_key, which would silently "
                    "overwrite ATT entries. Disambiguate labels in the "
                    "source script. Colliding labels withheld."
                ),
            )
        safe_groups_set.add(sg)
        safe_groups.append(sg)

    # Sanitize event_time labels. Use a string form (so JSON keys are
    # stable: "-3", "-2", ...). Strip non-finite; coerce ints to int.
    safe_event_times: list[Any] = []
    safe_event_times_str_set: set[str] = set()
    for raw_t in event_times_raw:
        t = float(raw_t)
        if not math.isfinite(t):
            continue
        if t == int(t):
            t_norm: Any = int(t)
        else:
            t_norm = t
        safe_event_times.append(t_norm)
        safe_event_times_str_set.add(str(t_norm))

    # Apply the cohort-N gate. Map safe-group → treated count; drop
    # the entire cohort when count < threshold (using the same
    # threshold as descriptive's min_n for consistency; the SDC
    # config has a single ``min_n`` that drives all suppression).
    cohort_min_n = config.min_n_did_cohort
    suppressed_cohorts: set[str] = set()
    cleaned_n_treated: dict[str, int] = {}

    for raw_g_key, count in n_treated_raw.items():
        if not isinstance(raw_g_key, str):
            continue
        sg = safe_key(raw_g_key)
        if sg not in safe_groups_set:
            # Cohort key not in declared groups — drop silently,
            # don't name it (would echo a data-derived label).
            continue
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            return SanitizerResult(
                ok=False, analysis_type="did_event_study",
                rejection_reason=(
                    "n_treated_per_group values must be non-negative "
                    f"ints; got {type(count).__name__} for one entry "
                    "(cohort label withheld)"
                ),
            )
        if count < cohort_min_n:
            suppressed_cohorts.add(sg)
        else:
            cleaned_n_treated[sg] = count

    # Every declared cohort must have a count entry. A cohort named
    # in ``groups`` but missing from ``n_treated_per_group`` would
    # bypass the gate — reject the payload rather than guess.
    declared_safe_in_n = {
        safe_key(k) for k in n_treated_raw.keys()
        if isinstance(k, str) and safe_key(k) in safe_groups_set
    }
    missing_n = [g for g in safe_groups if g not in declared_safe_in_n]
    if missing_n:
        return SanitizerResult(
            ok=False, analysis_type="did_event_study",
            rejection_reason=(
                f"{len(missing_n)} declared cohort(s) have no "
                f"n_treated_per_group entry. The cohort-N gate cannot "
                f"run without per-cohort sizes. Cohort labels withheld."
            ),
        )

    surviving_cohorts: set[str] = safe_groups_set - suppressed_cohorts
    if suppressed_cohorts:
        transformations.append(
            f"cohort suppression: {len(suppressed_cohorts)} cohort(s) "
            f"with n_treated < {cohort_min_n} dropped entirely (labels "
            f"withheld — cohort identities are disclosive)"
        )

    if not surviving_cohorts:
        return SanitizerResult(
            ok=False, analysis_type="did_event_study",
            rejection_reason=(
                f"all cohorts have n_treated < {cohort_min_n}; nothing "
                f"survives the cohort-N gate. No ATT panel published."
            ),
        )

    # Build the output. Top-level allowed fields first.
    out: dict[str, Any] = _collect_allowed(
        raw,
        numeric=_DID_EVENT_ALLOWED_NUMERIC_FIELDS,
        integer=_DID_EVENT_ALLOWED_INT_FIELDS,
        string=_DID_EVENT_ALLOWED_STRING_FIELDS,
        list_string=_DID_EVENT_ALLOWED_LIST_STRING,
        list_numeric=_DID_EVENT_ALLOWED_LIST_NUMERIC,
        transformations=transformations,
    )

    # Validate string-enum fields (aggregation_method, estimator,
    # comparison_group, base_period).
    aggm = out.get("aggregation_method")
    if aggm is not None and aggm not in _DID_EVENT_VALID_AGGREGATION:
        transformations.append(
            f"dropped 'aggregation_method' value (not in valid set)"
        )
        del out["aggregation_method"]
    est = out.get("estimator")
    if est is not None and est not in _DID_EVENT_VALID_ESTIMATOR:
        transformations.append(
            f"dropped 'estimator' value (not in valid set)"
        )
        del out["estimator"]
    cmp_group = out.get("comparison_group")
    if cmp_group is not None and cmp_group not in _DID_VALID_COMPARISON_GROUP:
        transformations.append(
            "dropped 'comparison_group' value (must be one of "
            "nevertreated / notyettreated / never_treated / not_yet_treated)"
        )
        del out["comparison_group"]
    bperiod = out.get("base_period")
    if bperiod is not None and bperiod not in _DID_VALID_BASE_PERIOD:
        transformations.append(
            "dropped 'base_period' value (must be 'varying' or 'universal')"
        )
        del out["base_period"]

    # Re-place the sanitized identifier lists (use safe forms).
    out["groups"] = sorted(surviving_cohorts)
    out["event_times"] = safe_event_times
    out["n_treated_per_group"] = cleaned_n_treated

    # Process each nested dict-of-dict field (att / SE / p / CI
    # lower+upper). Apply: (a) outer key must be a surviving cohort,
    # (b) inner key must be in event_times, (c) values finite, then
    # precision-clamp by aggregate treated N.
    total_treated_n = sum(cleaned_n_treated.values())
    sigfigs_n = total_treated_n if total_treated_n > 0 else cohort_min_n

    for field in _DID_EVENT_NESTED_DICT_FIELDS:
        v = raw.get(field)
        if v is None:
            continue
        if not isinstance(v, dict):
            transformations.append(
                f"dropped {field!r}: expected nested dict, "
                f"got {type(v).__name__}"
            )
            continue
        cleaned: dict[str, dict[str, float]] = {}
        dropped_outer = 0
        dropped_inner = 0
        for outer_k, inner_v in v.items():
            if not isinstance(outer_k, str):
                dropped_outer += 1
                continue
            sg = safe_key(outer_k)
            if sg not in surviving_cohorts:
                # Either the cohort was suppressed or it's not in
                # ``groups`` at all. Either way: drop, don't name.
                dropped_outer += 1
                continue
            if not isinstance(inner_v, dict):
                dropped_outer += 1
                continue
            cleaned_inner: dict[str, float] = {}
            for inner_k, cell_v in inner_v.items():
                # Inner key may be str or int (event times); normalize
                # to str-form to match safe_event_times_str_set.
                k_str = str(inner_k)
                if k_str not in safe_event_times_str_set:
                    dropped_inner += 1
                    continue
                if not _is_finite_number(cell_v):
                    dropped_inner += 1
                    continue
                cleaned_inner[k_str] = clamp_precision(
                    float(cell_v), sigfigs_n
                )
            if cleaned_inner:
                cleaned[sg] = cleaned_inner
        if dropped_outer:
            transformations.append(
                f"dropped {dropped_outer} undeclared/suppressed outer "
                f"key(s) from {field!r} (cohort labels withheld)"
            )
        if dropped_inner:
            transformations.append(
                f"dropped {dropped_inner} undeclared event-time key(s) "
                f"from {field!r}"
            )
        out[field] = cleaned

    # Aggregate-att scalars: precision-clamp at total-treated-N.
    for key in _DID_EVENT_ALLOWED_NUMERIC_FIELDS:
        if key in out:
            out[key] = clamp_precision(out[key], sigfigs_n)

    transformations.append(
        f"clamped numeric fields to precision matching total treated "
        f"n={sigfigs_n}"
    )

    return SanitizerResult(
        ok=True, analysis_type="did_event_study",
        sanitized=out, transformations=transformations,
    )


# ---------------------------------------------------------------------------
# Regression discontinuity design (RDD)
# ---------------------------------------------------------------------------
#
# The RDD shape ships the local-polynomial point estimate(s) plus
# bandwidth and effective-N diagnostics — what the model needs to
# evaluate an RDD design. The Calonico-Cattaneo-Titiunik (rdrobust)
# convention reports three flavors of τ at one fit: the conventional
# local-polynomial estimate, the bias-corrected variant, and the
# robust variant whose standard error accounts for bias-correction
# noise. The model sees all three so it can report the standard
# table conventional / bc / robust users expect.
#
# **Privacy carve-out, made structural via the allowlist:**
# Two RDD diagnostics are deliberately excluded from the shape:
#
#   1.  The **McCrary density test**'s estimated density curve.
#       The test statistic itself is a single scalar (log-discontinuity
#       in density at the cutoff); the *curve* — density evaluated at
#       a grid of points around the cutoff — is essentially a
#       histogram of the running variable in the most identifying
#       region (a few bandwidths either side of c). The running
#       variable in an RDD is by construction sensitive: income at a
#       tax-credit cutoff, test score at an admissions threshold,
#       date-of-birth at a school-entry cutoff. Surfacing the density
#       curve to the model would invert the privacy claim ("the
#       model never sees a raw cell value") on the exact slice where
#       individual identification is most likely. Even the bare
#       statistic at the cutoff has a cutoff-scan attack: re-run at
#       placebo cutoffs c±δ and the sequence of statistics maps the
#       density. We exclude the test ENTIRELY from this shape.
#
#   2.  **Binscatter near the cutoff** — by construction, bins shrink
#       toward the cutoff to make the discontinuity visible. Small
#       bins mean cells of small N over the most sensitive variable
#       slice. Same disclosure surface as McCrary's curve.
#
# Both are researcher-only by construction — they have no field in
# the ``rdd`` allowlist below. A script can still produce them
# visually for the researcher (the executor's raw-log panel and the
# helper-error JSONL surface stay intact), but no path through the
# sanitizer carries them to the model. This matches the helper-
# allowlist precedent set for plot vision (the only paths to the
# model run through ``plot_residuals`` / ``plot_interaction`` /
# ``plot_coefficients`` / ``plot_estimate_comparison`` — bespoke
# plots stay researcher-only).
#
# Binscatter AWAY from cutoffs, with min-N-per-bin guarantees, fits
# the existing ``magnitude_table`` shape and can ship through that
# channel. The exclusion here is specifically for the cutoff-
# proximity case where binwidths shrink by construction.

_RDD_REQUIRED: frozenset[str] = frozenset((
    "type", "running_variable", "cutoff",
    "tau_robust", "se_robust",
    "effective_n_left", "effective_n_right",
))
_RDD_ALLOWED_NUMERIC_FIELDS: frozenset[str] = frozenset((
    # CCT three-flavor estimates
    "tau_conventional", "tau_bias_corrected", "tau_robust",
    "se_conventional", "se_bias_corrected", "se_robust",
    "p_conventional", "p_bias_corrected", "p_robust",
    "ci_lower_conventional", "ci_upper_conventional",
    "ci_lower_bias_corrected", "ci_upper_bias_corrected",
    "ci_lower_robust", "ci_upper_robust",
    # Cutoff (the threshold value) — a researcher-chosen constant,
    # not a data-derived quantity. Surfaced so the model can echo
    # "discontinuity at age 65 = ..." rather than guessing.
    "cutoff",
    # Bandwidth(s). Left/right bandwidth differ when the optimal
    # MSE-minimizing bandwidth is computed separately on each side
    # (rdrobust's default).
    "bandwidth_left", "bandwidth_right",
    "bandwidth_bias_correction_left", "bandwidth_bias_correction_right",
    # Fuzzy-RDD diagnostic: first-stage F-statistic for joint
    # significance of the cutoff dummy in the first-stage regression
    # of the endogenous-treatment indicator on the running variable.
    # Below ~10 flags a weak first-stage and renders the Wald-ratio
    # τ unstable. Same primitive as IV's ``first_stage_f``; the
    # field name is duplicated here so RDD payloads don't have to
    # route through the regression-bucket schema. Whether the fit
    # is sharp or fuzzy is communicated via the ``estimator`` enum
    # below (``fuzzy_2sls`` vs ``local_polynomial``).
    "first_stage_f",
))
_RDD_ALLOWED_INT_FIELDS: frozenset[str] = frozenset((
    # Effective N inside the bandwidth window — the SDC-relevant
    # quantity. RDD inference is local; the effective sample sizes
    # are what bound how tightly the local fit can identify
    # individuals. Required, so the min-N gate has its inputs.
    "effective_n_left", "effective_n_right", "effective_n_total",
    "polynomial_order",
))
_RDD_ALLOWED_STRING_FIELDS: frozenset[str] = frozenset((
    "type", "estimator", "running_variable", "outcome_variable",
    "kernel",
    # rdrobust's bandwidth-selection rule. Each is a documented CCT /
    # CER selector with different optimality criteria:
    #   * ``mserd``: single MSE-optimal bandwidth, same on both sides.
    #   * ``msetwo``: MSE-optimal bandwidth selected separately per side.
    #   * ``msesum`` / ``msecomb1`` / ``msecomb2``: MSE variants.
    #   * ``cerrd`` / ``certwo`` / ``cercomb1`` / ``cercomb2``: coverage-
    #     error-rate (CER) optimal — narrower bandwidth, lower bias,
    #     wider CIs.
    #   * ``manual``: caller-supplied bandwidth (no automatic selection).
    # The validation set below pins these as the only legal values; the
    # field passes ``safe_text`` regardless, but limiting to known
    # selectors prevents a script from smuggling free-text into the
    # model via this slot.
    "bandwidth_selector",
))
_RDD_VALID_ESTIMATOR: frozenset[str] = frozenset((
    "local_polynomial", "sharp_parametric", "fuzzy_2sls",
    "rdrobust", "rdlocrand",
))
_RDD_VALID_KERNEL: frozenset[str] = frozenset((
    "triangular", "uniform", "epanechnikov",
))
_RDD_VALID_BANDWIDTH_SELECTOR: frozenset[str] = frozenset((
    "mserd", "msetwo", "msesum", "msecomb1", "msecomb2",
    "cerrd", "certwo", "cercomb1", "cercomb2",
    "manual",
))


def _sanitize_rdd(
    raw: dict[str, Any], config: SDCConfig
) -> SanitizerResult:
    missing_reason = _require_fields(raw, _RDD_REQUIRED, "rdd")
    if missing_reason:
        return SanitizerResult(
            ok=False, analysis_type="rdd",
            rejection_reason=missing_reason,
        )

    # Effective-N gate. Local-polynomial RDD identifies τ from
    # observations inside the bandwidth on each side of the cutoff;
    # the local sample sizes must each pass the min-N threshold or
    # the inference isn't trustworthy. Apply per-side, not just to
    # the total — a 50/2 split can hit total ≥ threshold while one
    # side has unbounded uncertainty.
    for side_field in ("effective_n_left", "effective_n_right"):
        v = raw.get(side_field)
        if not isinstance(v, int) or isinstance(v, bool) or v < 0:
            return SanitizerResult(
                ok=False, analysis_type="rdd",
                rejection_reason=(
                    f"{side_field} must be a non-negative int; "
                    f"got {type(v).__name__}"
                ),
            )
        try:
            require_minimum_n(v, config.min_n_regression, side_field)
        except MinimumNViolation as e:
            return SanitizerResult(
                ok=False, analysis_type="rdd",
                rejection_reason=str(e),
            )

    transformations: list[str] = []
    out = _collect_allowed(
        raw,
        numeric=_RDD_ALLOWED_NUMERIC_FIELDS,
        integer=_RDD_ALLOWED_INT_FIELDS,
        string=_RDD_ALLOWED_STRING_FIELDS,
        transformations=transformations,
    )

    # Validate enum string fields.
    est = out.get("estimator")
    if est is not None and est not in _RDD_VALID_ESTIMATOR:
        transformations.append(
            "dropped 'estimator' value (not in valid set)"
        )
        del out["estimator"]
    krn = out.get("kernel")
    if krn is not None and krn not in _RDD_VALID_KERNEL:
        transformations.append(
            "dropped 'kernel' value (not in valid set)"
        )
        del out["kernel"]
    bwsel = out.get("bandwidth_selector")
    if bwsel is not None and bwsel not in _RDD_VALID_BANDWIDTH_SELECTOR:
        transformations.append(
            "dropped 'bandwidth_selector' value (not in valid set — "
            "must be one of mserd / msetwo / msesum / msecomb1 / "
            "msecomb2 / cerrd / certwo / cercomb1 / cercomb2 / manual)"
        )
        del out["bandwidth_selector"]
    po = out.get("polynomial_order")
    if po is not None:
        if po < 0 or po > 4:
            transformations.append(
                "dropped 'polynomial_order' value (must be 0..4; "
                "local-polynomial RDD with degree > 4 is suspect)"
            )
            del out["polynomial_order"]

    # ``running_variable`` is identifier-shape gated. ``rejection_reason``
    # withholds bad names — the running variable is data-derived (the
    # column name the researcher chose).
    _enforce_identifier_string_fields(
        out, frozenset(("running_variable", "outcome_variable")),
        transformations, type_label="rdd",
    )

    # Re-check required after type filtering.
    missing_after = _require_after_filter(
        out, _RDD_REQUIRED, "rdd",
        pre_validated=frozenset(("effective_n_left", "effective_n_right")),
    )
    if missing_after:
        return SanitizerResult(
            ok=False, analysis_type="rdd",
            rejection_reason=missing_after,
        )

    # Precision clamp by total effective N (sum of sides if total
    # absent). Total drives the precision of τ; per-side N drives
    # the per-side gate already enforced above.
    n_total = out.get("effective_n_total")
    if n_total is None:
        n_total = out["effective_n_left"] + out["effective_n_right"]
        out["effective_n_total"] = n_total
    for key in _RDD_ALLOWED_NUMERIC_FIELDS:
        if key in out:
            out[key] = clamp_precision(out[key], n_total)

    transformations.append(
        f"clamped numeric fields to precision matching effective n={n_total}"
    )

    return SanitizerResult(
        ok=True, analysis_type="rdd",
        sanitized=out, transformations=transformations,
    )


# ---------------------------------------------------------------------------
# Kaplan-Meier (safe-form: scalars at preset horizons, no curve)
# ---------------------------------------------------------------------------
#
# The full KM step function — survival probability at every observed
# event time — is too granular near small risk sets. With small
# n_at_risk, the survival drop from a single event identifies that
# event's timing and (combined with covariate distribution) the
# individual. The shape published here is the *safe form* described
# in ``docs/direction.md``: median survival with CI, plus survival
# at a small set of preset horizons (e.g., 1y, 3y, 5y) each gated by
# its own n_at_risk threshold. Dedicated shape, not a coefficient
# table sub-type, because the cross-field invariant is different
# (per-horizon N gate, not per-coefficient name match).
#
# The curve itself (per-event-time S(t) values, the Greenwood SE
# series, KM-by-group log-rank chi²) is researcher-only by
# construction — it has no field in this allowlist. Same exclusion
# pattern as McCrary in the RDD shape: structurally absent from the
# allowlist means it can't be smuggled through the generic
# ``result(type="kaplan_meier", ...)`` path.

_KM_REQUIRED: frozenset[str] = frozenset((
    "type", "time_variable", "event_variable",
    "n_subjects", "n_failures",
))
_KM_ALLOWED_NUMERIC_FIELDS: frozenset[str] = frozenset((
    "median_survival_time", "median_survival_ci_lower",
    "median_survival_ci_upper",
    # Pre-specified horizon survival probabilities. We allowlist a
    # small fixed set — enough to cover the conventional 1y / 3y /
    # 5y reporting plus a couple extra — rather than letting the
    # caller name arbitrary horizons (which would be a covert-channel
    # surface for raw time values).
    "survival_at_1y", "survival_at_3y", "survival_at_5y",
    "survival_at_10y",
    "survival_at_1y_ci_lower", "survival_at_1y_ci_upper",
    "survival_at_3y_ci_lower", "survival_at_3y_ci_upper",
    "survival_at_5y_ci_lower", "survival_at_5y_ci_upper",
    "survival_at_10y_ci_lower", "survival_at_10y_ci_upper",
    # Log-rank omnibus across groups (when KM-by-group requested).
    "logrank_chi_squared", "logrank_p_value",
))
_KM_ALLOWED_INT_FIELDS: frozenset[str] = frozenset((
    "n_subjects", "n_failures",
    # n_at_risk at each pre-specified horizon — drives the per-
    # horizon gate. Each horizon needs ≥ min_n_regression at-risk
    # subjects or its S(t) is dropped.
    "n_at_risk_1y", "n_at_risk_3y", "n_at_risk_5y", "n_at_risk_10y",
    # Group count for log-rank-by-group setups.
    "n_groups",
))
_KM_ALLOWED_STRING_FIELDS: frozenset[str] = frozenset((
    "type", "time_variable", "event_variable", "group_variable",
))


def _sanitize_kaplan_meier(
    raw: dict[str, Any], config: SDCConfig
) -> SanitizerResult:
    missing_reason = _require_fields(raw, _KM_REQUIRED, "kaplan_meier")
    if missing_reason:
        return SanitizerResult(
            ok=False, analysis_type="kaplan_meier",
            rejection_reason=missing_reason,
        )

    n_sub = raw.get("n_subjects")
    if not isinstance(n_sub, int) or isinstance(n_sub, bool) or n_sub < 0:
        return SanitizerResult(
            ok=False, analysis_type="kaplan_meier",
            rejection_reason=(
                f"n_subjects must be a non-negative int; "
                f"got {type(n_sub).__name__}"
            ),
        )
    try:
        require_minimum_n(n_sub, config.min_n_regression, "n_subjects")
    except MinimumNViolation as e:
        return SanitizerResult(
            ok=False, analysis_type="kaplan_meier",
            rejection_reason=str(e),
        )

    n_fail = raw.get("n_failures")
    if not isinstance(n_fail, int) or isinstance(n_fail, bool) or n_fail < 0:
        return SanitizerResult(
            ok=False, analysis_type="kaplan_meier",
            rejection_reason=(
                f"n_failures must be a non-negative int; "
                f"got {type(n_fail).__name__}"
            ),
        )
    if n_fail > n_sub:
        return SanitizerResult(
            ok=False, analysis_type="kaplan_meier",
            rejection_reason=(
                f"n_failures ({n_fail}) cannot exceed n_subjects ({n_sub})"
            ),
        )

    transformations: list[str] = []
    out = _collect_allowed(
        raw,
        numeric=_KM_ALLOWED_NUMERIC_FIELDS,
        integer=_KM_ALLOWED_INT_FIELDS,
        string=_KM_ALLOWED_STRING_FIELDS,
        transformations=transformations,
    )

    _enforce_identifier_string_fields(
        out, frozenset(("time_variable", "event_variable", "group_variable")),
        transformations, type_label="kaplan_meier",
    )

    missing_after = _require_after_filter(
        out, _KM_REQUIRED, "kaplan_meier",
        pre_validated=frozenset(("n_subjects", "n_failures")),
    )
    if missing_after:
        return SanitizerResult(
            ok=False, analysis_type="kaplan_meier",
            rejection_reason=missing_after,
        )

    # Per-horizon n_at_risk gate. For each horizon h whose S(h) field
    # is populated, the corresponding n_at_risk_h must be present
    # and pass min_n_regression. If the gate fails, drop the S(h)
    # AND its CI bounds — partial publication leaks at-risk count
    # through "this horizon survives, that one doesn't".
    horizons = ("1y", "3y", "5y", "10y")
    for h in horizons:
        s_field = f"survival_at_{h}"
        if s_field not in out:
            continue
        n_risk_field = f"n_at_risk_{h}"
        n_risk = out.get(n_risk_field)
        if (n_risk is None
            or not isinstance(n_risk, int)
            or n_risk < config.min_n_regression):
            # Drop this horizon's S(h) and CI bounds together.
            dropped_fields = [s_field]
            for suffix in ("_ci_lower", "_ci_upper"):
                key = s_field + suffix
                if key in out:
                    dropped_fields.append(key)
                    del out[key]
            del out[s_field]
            transformations.append(
                f"dropped horizon {h}: n_at_risk_{h} below "
                f"min_n_regression ({config.min_n_regression}) "
                f"or absent"
            )

    # Precision clamp by total n_subjects.
    for key in _KM_ALLOWED_NUMERIC_FIELDS:
        if key in out:
            out[key] = clamp_precision(out[key], n_sub)
    transformations.append(
        f"clamped numeric fields to precision matching n_subjects={n_sub}"
    )

    return SanitizerResult(
        ok=True, analysis_type="kaplan_meier",
        sanitized=out, transformations=transformations,
    )


# ---------------------------------------------------------------------------
# Factor decomposition — PCA + factor analysis as one shape
# ---------------------------------------------------------------------------
#
# Covers principal-components analysis (PCA), classical factor
# analysis (factanal / sklearn.decomposition.FactorAnalyzer), and
# maximum-likelihood factor analysis. The disclosure-relevant
# quantities are all aggregates over the full sample:
#   * Loadings (variable × component matrix) — eigenvectors of the
#     correlation / covariance matrix. Bounded roughly [-1, 1] for
#     standardized inputs. The variable names are dataset columns
#     the model has already seen; component names ("PC1", "factor1",
#     etc.) are synthetic.
#   * Eigenvalues / explained variance / cumulative variance — one
#     scalar per component. Aggregate scalars.
#   * Communalities / uniqueness — one scalar per variable.
#   * Goodness-of-fit (KMO, Bartlett, chi²) — aggregate test stats.
#
# Privacy carve-out, structural: factor SCORES (per-observation
# projections onto the components) are NOT in this allowlist. Scores
# are essentially raw observations transformed; emitting them would
# undo the privacy claim on the exact axis PCA/FA defines. Stays
# researcher-only by construction.
#
# The same shape carries PCA, classical FA, and ML-FA outputs;
# ``method`` distinguishes. Helpers per method × language; the
# sanitizer doesn't dispatch on it.

_FACTOR_REQUIRED: frozenset[str] = frozenset((
    "type", "method", "n_observations", "n_variables", "n_components",
    "variables", "loadings",
))
_FACTOR_ALLOWED_NUMERIC_FIELDS: frozenset[str] = frozenset((
    # Goodness-of-fit scalars (mostly for ML factor analysis):
    "kmo",                       # Kaiser-Meyer-Olkin sampling adequacy
    "bartlett_chi_squared",      # Bartlett's test of sphericity
    "bartlett_p_value",
    "chi_squared",               # ML-FA goodness-of-fit
    "chi_squared_p_value",
    "log_likelihood",
    "rmsea",                     # Root mean square error of approximation
    "tli",                       # Tucker-Lewis index
))
_FACTOR_ALLOWED_INT_FIELDS: frozenset[str] = frozenset((
    "n_observations", "n_variables", "n_components",
    "degrees_of_freedom",
))
_FACTOR_ALLOWED_STRING_FIELDS: frozenset[str] = frozenset((
    "type", "method", "rotation",
))
_FACTOR_ALLOWED_LIST_STRING: frozenset[str] = frozenset((
    # Variable names participating in the decomposition. Each goes
    # through ``safe_key``; the list is bounded by ``_FACTOR_MAX_VARIABLES``.
    "variables",
    # Component labels ("PC1", "PC2", ..., or "factor1", "factor2", ...).
    # Synthetic — generated by the helper rather than data-derived —
    # but allowlisted for consistency so the sanitizer's cross-field
    # check has the component keys to validate against.
    "components",
))
# Nested-dict-of-dict field: loadings is {variable: {component: value}}.
# Processed separately after the top-level filter, mirroring the
# ``did_event_study`` pattern.
_FACTOR_NESTED_DICT_FIELDS: frozenset[str] = frozenset((
    "loadings",
))
# Flat dict-numeric fields. Two key conventions live here:
#   * Component-keyed: ``explained_variance`` / ``explained_variance_ratio``
#     / ``cumulative_variance`` / ``eigenvalues``. Keys must match the
#     declared ``components`` list.
#   * Variable-keyed: ``communalities`` / ``uniqueness``. Keys must
#     match the declared ``variables`` list.
_FACTOR_PER_COMPONENT_DICTS: frozenset[str] = frozenset((
    "explained_variance", "explained_variance_ratio",
    "cumulative_variance", "eigenvalues",
))
_FACTOR_PER_VARIABLE_DICTS: frozenset[str] = frozenset((
    "communalities", "uniqueness",
))
_FACTOR_VALID_METHODS: frozenset[str] = frozenset((
    "pca",                          # principal components
    "factor_analysis",              # generic
    "principal_factor",             # principal-factor extraction
    "maximum_likelihood",           # ML factor analysis
    "minimum_residual",             # MinRes
))
_FACTOR_VALID_ROTATIONS: frozenset[str] = frozenset((
    "none", "varimax", "promax", "oblimin", "quartimax",
    "equamax", "geomin", "bentlerT", "bifactor",
))
# Structural caps. A real PCA / FA published in a paper uses ≤ ~50
# variables and ≤ ~20 components; bigger shapes are almost always
# data-shaped objects masquerading as aggregates.
_FACTOR_MAX_VARIABLES: int = 100
_FACTOR_MAX_COMPONENTS: int = 50


def _sanitize_factor_decomposition(
    raw: dict[str, Any], config: SDCConfig
) -> SanitizerResult:
    missing_reason = _require_fields(raw, _FACTOR_REQUIRED, "factor_decomposition")
    if missing_reason:
        return SanitizerResult(
            ok=False, analysis_type="factor_decomposition",
            rejection_reason=missing_reason,
        )

    # n_observations gates the precision clamp; ``min_n_descriptive``
    # is the same threshold used for descriptive payloads (PCA / FA on
    # tiny samples is statistically meaningless anyway).
    n_obs = raw.get("n_observations")
    if not isinstance(n_obs, int) or isinstance(n_obs, bool) or n_obs < 0:
        return SanitizerResult(
            ok=False, analysis_type="factor_decomposition",
            rejection_reason=(
                f"n_observations must be a non-negative int; "
                f"got {type(n_obs).__name__}"
            ),
        )
    try:
        require_minimum_n(n_obs, config.min_n_descriptive, "n_observations")
    except MinimumNViolation as e:
        return SanitizerResult(
            ok=False, analysis_type="factor_decomposition",
            rejection_reason=str(e),
        )

    # Method enum check before any further work.
    method = raw.get("method")
    if not isinstance(method, str) or method not in _FACTOR_VALID_METHODS:
        return SanitizerResult(
            ok=False, analysis_type="factor_decomposition",
            rejection_reason=(
                f"method must be one of {sorted(_FACTOR_VALID_METHODS)}; "
                f"got {method!r}"
            ),
        )

    # Variable list — required, bounded, names go through safe_key.
    raw_vars = raw.get("variables")
    if not isinstance(raw_vars, list) or not all(
        isinstance(v, str) for v in raw_vars
    ):
        return SanitizerResult(
            ok=False, analysis_type="factor_decomposition",
            rejection_reason=(
                "variables must be a list of strings (dataset column names)"
            ),
        )
    if len(raw_vars) == 0:
        return SanitizerResult(
            ok=False, analysis_type="factor_decomposition",
            rejection_reason="variables list is empty",
        )
    if len(raw_vars) > _FACTOR_MAX_VARIABLES:
        return SanitizerResult(
            ok=False, analysis_type="factor_decomposition",
            rejection_reason=(
                f"variables has {len(raw_vars)} entries; structural cap "
                f"is {_FACTOR_MAX_VARIABLES}"
            ),
        )

    # n_components claim must match the actual declared structure.
    n_comp_claim = raw.get("n_components")
    if not isinstance(n_comp_claim, int) or n_comp_claim <= 0:
        return SanitizerResult(
            ok=False, analysis_type="factor_decomposition",
            rejection_reason=(
                f"n_components must be a positive int; got {n_comp_claim!r}"
            ),
        )
    if n_comp_claim > _FACTOR_MAX_COMPONENTS:
        return SanitizerResult(
            ok=False, analysis_type="factor_decomposition",
            rejection_reason=(
                f"n_components is {n_comp_claim}; structural cap is "
                f"{_FACTOR_MAX_COMPONENTS}"
            ),
        )

    # n_variables claim must match the variables list length.
    n_var_claim = raw.get("n_variables")
    if not isinstance(n_var_claim, int) or n_var_claim != len(raw_vars):
        return SanitizerResult(
            ok=False, analysis_type="factor_decomposition",
            rejection_reason=(
                f"n_variables claim ({n_var_claim}) does not match "
                f"variables list length ({len(raw_vars)})"
            ),
        )

    transformations: list[str] = []

    # Sanitize variable + component label lists; reject safe_key
    # collisions outright (the same pattern as crosstab / DiD).
    safe_vars: list[str] = []
    safe_var_set: set[str] = set()
    for raw_v in raw_vars:
        sv = safe_key(raw_v)
        if sv in safe_var_set:
            return SanitizerResult(
                ok=False, analysis_type="factor_decomposition",
                rejection_reason=(
                    "variable label collision after sanitization "
                    "(two distinct raw labels sanitize to the same "
                    "safe_key; would silently overwrite loadings rows). "
                    "Labels withheld."
                ),
            )
        safe_var_set.add(sv)
        safe_vars.append(sv)

    raw_comps = raw.get("components") or [f"PC{i+1}" for i in range(n_comp_claim)]
    if not isinstance(raw_comps, list) or not all(isinstance(c, str) for c in raw_comps):
        return SanitizerResult(
            ok=False, analysis_type="factor_decomposition",
            rejection_reason="components must be a list of strings",
        )
    if len(raw_comps) != n_comp_claim:
        return SanitizerResult(
            ok=False, analysis_type="factor_decomposition",
            rejection_reason=(
                f"components list length ({len(raw_comps)}) does not "
                f"match n_components ({n_comp_claim})"
            ),
        )
    safe_comps: list[str] = []
    safe_comp_set: set[str] = set()
    for raw_c in raw_comps:
        sc = safe_key(raw_c)
        if sc in safe_comp_set:
            return SanitizerResult(
                ok=False, analysis_type="factor_decomposition",
                rejection_reason=(
                    "component label collision after sanitization"
                ),
            )
        safe_comp_set.add(sc)
        safe_comps.append(sc)

    out: dict[str, Any] = _collect_allowed(
        raw,
        numeric=_FACTOR_ALLOWED_NUMERIC_FIELDS,
        integer=_FACTOR_ALLOWED_INT_FIELDS,
        string=_FACTOR_ALLOWED_STRING_FIELDS,
        list_string=_FACTOR_ALLOWED_LIST_STRING,
        transformations=transformations,
    )
    out["type"] = "factor_decomposition"
    out["method"] = method
    out["variables"] = safe_vars
    out["components"] = safe_comps
    out["n_observations"] = n_obs
    out["n_variables"] = n_var_claim
    out["n_components"] = n_comp_claim

    # Validate rotation enum if supplied.
    rot = out.get("rotation")
    if rot is not None and rot not in _FACTOR_VALID_ROTATIONS:
        transformations.append(
            f"dropped 'rotation' value (must be one of "
            f"{sorted(_FACTOR_VALID_ROTATIONS)})"
        )
        del out["rotation"]

    # Loadings: nested {variable: {component: value}}. Outer keys
    # must be in safe_vars; inner keys must be in safe_comps.
    loadings_raw = raw.get("loadings")
    if not isinstance(loadings_raw, dict):
        return SanitizerResult(
            ok=False, analysis_type="factor_decomposition",
            rejection_reason=(
                f"loadings must be a nested dict {{variable: {{component: value}}}};"
                f" got {type(loadings_raw).__name__}"
            ),
        )
    cleaned_loadings: dict[str, dict[str, float]] = {}
    dropped_outer = 0
    dropped_inner = 0
    for outer_k, inner_v in loadings_raw.items():
        if not isinstance(outer_k, str):
            dropped_outer += 1
            continue
        sv = safe_key(outer_k)
        if sv not in safe_var_set:
            dropped_outer += 1
            continue
        if not isinstance(inner_v, dict):
            dropped_outer += 1
            continue
        cleaned_inner: dict[str, float] = {}
        for inner_k, val in inner_v.items():
            if not isinstance(inner_k, str):
                dropped_inner += 1
                continue
            sc = safe_key(inner_k)
            if sc not in safe_comp_set:
                dropped_inner += 1
                continue
            if not _is_finite_number(val):
                dropped_inner += 1
                continue
            cleaned_inner[sc] = clamp_precision(float(val), n_obs)
        if cleaned_inner:
            cleaned_loadings[sv] = cleaned_inner
    if dropped_outer:
        transformations.append(
            f"dropped {dropped_outer} undeclared variable(s) from loadings"
        )
    if dropped_inner:
        transformations.append(
            f"dropped {dropped_inner} undeclared component entry(ies) from loadings"
        )
    if not cleaned_loadings:
        return SanitizerResult(
            ok=False, analysis_type="factor_decomposition",
            rejection_reason=(
                "loadings dict empty after sanitization — keys didn't match "
                "the declared variables/components"
            ),
        )
    out["loadings"] = cleaned_loadings

    # Per-component dicts (explained_variance, eigenvalues, …) — outer
    # keys must be in safe_comps.
    for field in _FACTOR_PER_COMPONENT_DICTS:
        v = raw.get(field)
        if v is None:
            continue
        if not isinstance(v, dict):
            transformations.append(
                f"dropped {field!r}: expected dict, got {type(v).__name__}"
            )
            continue
        cleaned: dict[str, float] = {}
        for k, val in v.items():
            if not isinstance(k, str):
                continue
            sc = safe_key(k)
            if sc not in safe_comp_set:
                continue
            if not _is_finite_number(val):
                continue
            cleaned[sc] = clamp_precision(float(val), n_obs)
        if cleaned:
            out[field] = cleaned

    # Per-variable dicts (communalities, uniqueness) — outer keys
    # must be in safe_vars.
    for field in _FACTOR_PER_VARIABLE_DICTS:
        v = raw.get(field)
        if v is None:
            continue
        if not isinstance(v, dict):
            transformations.append(
                f"dropped {field!r}: expected dict, got {type(v).__name__}"
            )
            continue
        cleaned = {}
        for k, val in v.items():
            if not isinstance(k, str):
                continue
            sv = safe_key(k)
            if sv not in safe_var_set:
                continue
            if not _is_finite_number(val):
                continue
            cleaned[sv] = clamp_precision(float(val), n_obs)
        if cleaned:
            out[field] = cleaned

    # Precision-clamp scalar numeric fields.
    for key in _FACTOR_ALLOWED_NUMERIC_FIELDS:
        if key in out:
            out[key] = clamp_precision(out[key], n_obs)

    transformations.append(
        f"clamped numeric fields to precision matching n_observations={n_obs}"
    )

    return SanitizerResult(
        ok=True, analysis_type="factor_decomposition",
        sanitized=out, transformations=transformations,
    )


# ---------------------------------------------------------------------------
# Cluster analysis (k-means, k-medoids, hierarchical, …)
# ---------------------------------------------------------------------------
#
# The shape ships per-cluster centroids and quality metrics from a
# fitted clustering. Two new SDC primitives the existing shapes
# don't have:
#
#   1.  **Whole-cluster suppression by size.** Clusters below
#       ``min_n_cluster`` are dropped entirely — their entry in
#       ``cluster_sizes``, their centroid row, their within-cluster
#       SS, every per-cluster dict. Partial publication would leak
#       the cluster size through which clusters survived. Same
#       pattern as DiD's cohort gate.
#
#   2.  **Per-cluster precision clamping on centroids.** Centroids
#       are means over the cluster's members; their precision
#       scales with that cluster's N, not the global N. A centroid
#       of a 12-person cluster on income should be clamped to
#       ~3 sigfigs; a centroid of a 12,000-person cluster on income
#       can carry ~5. The existing shapes all clamp by global N
#       (``clamp_precision_dict(d, n_total)``); this shape
#       walks per-row and clamps with the row-specific N.
#
# Privacy carve-out, structural: per-observation cluster assignments
# (sklearn's ``labels_``, R kmeans's ``$cluster``) are NOT in this
# allowlist. Assignments are per-row data — emitting them would tell
# the model which row went where, which combined with the centroid
# is enough to identify individuals in small clusters. The
# researcher's local R / Python session sees them; the model
# doesn't.

_CLUSTER_REQUIRED: frozenset[str] = frozenset((
    "type", "method", "n_observations", "n_clusters", "n_features",
    "variables", "cluster_labels", "cluster_sizes",
    # ``centroids`` is conditionally required — required for every
    # method that has centroids by construction (kmeans / hierarchical
    # / pam / agglomerative), absent-OK for DBSCAN / HDBSCAN (density-
    # based; no centroids by design). The conditional check fires
    # below in ``_sanitize_cluster_analysis``. If a DBSCAN payload
    # includes centroids anyway (caller computed them post-hoc from
    # the labels array), the field is still validated against the
    # cross-field rules.
))
_CLUSTER_METHODS_WITHOUT_CENTROIDS: frozenset[str] = frozenset((
    "dbscan", "hdbscan",
))
_CLUSTER_ALLOWED_NUMERIC_FIELDS: frozenset[str] = frozenset((
    # Sum-of-squares decomposition. Scalars over the whole fit;
    # aggregate quantities, no per-row leak.
    "total_within_ss", "between_cluster_ss", "total_ss",
    "ss_ratio",                # between / total — fraction explained
    "inertia",                 # sklearn alias for total_within_ss
    # Cluster quality scalars.
    "silhouette_score",        # global mean silhouette
    "calinski_harabasz_score",
    "davies_bouldin_score",
    # Hierarchical-specific: the dendrogram cut height that produced
    # the n_clusters partition. A scalar over the data; the
    # dendrogram itself (linkage matrix / merge heights series) is
    # structurally absent from the allowlist.
    "cut_height",
))
_CLUSTER_ALLOWED_INT_FIELDS: frozenset[str] = frozenset((
    "n_observations", "n_clusters", "n_features", "n_iterations",
    # DBSCAN / HDBSCAN: count of points labeled noise (outside any
    # cluster). Aggregate scalar; the noise points' identities don't
    # cross — same disclosure profile as ``n_clusters``.
    "n_noise_points",
))
_CLUSTER_ALLOWED_STRING_FIELDS: frozenset[str] = frozenset((
    "type", "method", "distance_metric", "linkage",
))
_CLUSTER_ALLOWED_LIST_STRING: frozenset[str] = frozenset((
    "variables",
    "cluster_labels",          # synthetic identifiers like "cluster_1"
))
# Per-cluster flat dicts. ``cluster_sizes`` drives the suppression
# gate so it's required. The others are optional metrics; their keys
# must match the declared (surviving) cluster labels.
_CLUSTER_PER_CLUSTER_INT_DICTS: frozenset[str] = frozenset((
    "cluster_sizes",
))
_CLUSTER_PER_CLUSTER_NUMERIC_DICTS: frozenset[str] = frozenset((
    "within_cluster_ss", "silhouette_per_cluster",
))
# Per-variable numeric dict: ``f_statistic_per_variable`` carries
# the between-cluster F-statistic for each input variable —
# diagnostic for which variables most discriminate clusters.
# Aggregate scalar per variable; keys validated against the
# declared ``variables`` list (not cluster_sizes).
_CLUSTER_PER_VARIABLE_NUMERIC_DICTS: frozenset[str] = frozenset((
    "f_statistic_per_variable",
))
# Nested-dict (cluster × variable) — centroids. Per-cluster precision
# clamping fires on this field.
_CLUSTER_NESTED_DICT_FIELDS: frozenset[str] = frozenset((
    "centroids",
))
_CLUSTER_VALID_METHODS: frozenset[str] = frozenset((
    "kmeans",
    "hierarchical",      # use the ``linkage`` field for ward / complete /
                         # average / single / centroid / median
    "agglomerative",     # alias for hierarchical bottom-up; same payload
    "pam",               # partitioning around medoids — literature-standard
                         # name for k-medoids
    "kmedoids",          # legacy alias retained; ``pam`` is the canonical
    "dbscan",            # density-based — no centroids by construction;
                         # centroids field becomes optional below
    "hdbscan",           # hierarchical density-based; same payload shape
    "gaussian_mixture",  # accepted today; per-component variances + mixture
                         # weights aren't well-represented in this shape and
                         # the inference-adequacy story for GMM payloads is
                         # an open follow-up. Helper deferred.
    "spectral",
))
_CLUSTER_VALID_LINKAGE: frozenset[str] = frozenset((
    "ward", "complete", "average", "single", "centroid", "median",
))
_CLUSTER_VALID_DISTANCE: frozenset[str] = frozenset((
    "euclidean", "manhattan", "cosine", "mahalanobis", "chebyshev",
    "minkowski", "hamming", "jaccard",
))
_CLUSTER_MAX_CLUSTERS: int = 50
_CLUSTER_MAX_FEATURES: int = 100


def _sanitize_cluster_analysis(
    raw: dict[str, Any], config: SDCConfig
) -> SanitizerResult:
    missing_reason = _require_fields(raw, _CLUSTER_REQUIRED, "cluster_analysis")
    if missing_reason:
        return SanitizerResult(
            ok=False, analysis_type="cluster_analysis",
            rejection_reason=missing_reason,
        )

    n_obs = raw.get("n_observations")
    if not isinstance(n_obs, int) or isinstance(n_obs, bool) or n_obs < 0:
        return SanitizerResult(
            ok=False, analysis_type="cluster_analysis",
            rejection_reason=(
                f"n_observations must be a non-negative int; "
                f"got {type(n_obs).__name__}"
            ),
        )
    try:
        require_minimum_n(n_obs, config.min_n_descriptive, "n_observations")
    except MinimumNViolation as e:
        return SanitizerResult(
            ok=False, analysis_type="cluster_analysis",
            rejection_reason=str(e),
        )

    method = raw.get("method")
    if not isinstance(method, str) or method not in _CLUSTER_VALID_METHODS:
        return SanitizerResult(
            ok=False, analysis_type="cluster_analysis",
            rejection_reason=(
                f"method must be one of {sorted(_CLUSTER_VALID_METHODS)}; "
                f"got {method!r}"
            ),
        )

    raw_vars = raw.get("variables")
    if not isinstance(raw_vars, list) or not all(
        isinstance(v, str) for v in raw_vars
    ):
        return SanitizerResult(
            ok=False, analysis_type="cluster_analysis",
            rejection_reason="variables must be a list of strings",
        )
    if len(raw_vars) == 0:
        return SanitizerResult(
            ok=False, analysis_type="cluster_analysis",
            rejection_reason="variables list is empty",
        )
    if len(raw_vars) > _CLUSTER_MAX_FEATURES:
        return SanitizerResult(
            ok=False, analysis_type="cluster_analysis",
            rejection_reason=(
                f"variables has {len(raw_vars)} entries; structural cap "
                f"is {_CLUSTER_MAX_FEATURES}"
            ),
        )

    n_clusters_claim = raw.get("n_clusters")
    if not isinstance(n_clusters_claim, int) or n_clusters_claim <= 0:
        return SanitizerResult(
            ok=False, analysis_type="cluster_analysis",
            rejection_reason=(
                f"n_clusters must be a positive int; got {n_clusters_claim!r}"
            ),
        )
    if n_clusters_claim > _CLUSTER_MAX_CLUSTERS:
        return SanitizerResult(
            ok=False, analysis_type="cluster_analysis",
            rejection_reason=(
                f"n_clusters is {n_clusters_claim}; structural cap is "
                f"{_CLUSTER_MAX_CLUSTERS}"
            ),
        )

    n_features_claim = raw.get("n_features")
    if not isinstance(n_features_claim, int) or n_features_claim != len(raw_vars):
        return SanitizerResult(
            ok=False, analysis_type="cluster_analysis",
            rejection_reason=(
                f"n_features ({n_features_claim}) does not match "
                f"variables list length ({len(raw_vars)})"
            ),
        )

    raw_labels = raw.get("cluster_labels")
    if not isinstance(raw_labels, list) or not all(
        isinstance(c, str) for c in raw_labels
    ):
        return SanitizerResult(
            ok=False, analysis_type="cluster_analysis",
            rejection_reason="cluster_labels must be a list of strings",
        )
    if len(raw_labels) != n_clusters_claim:
        return SanitizerResult(
            ok=False, analysis_type="cluster_analysis",
            rejection_reason=(
                f"cluster_labels length ({len(raw_labels)}) does not "
                f"match n_clusters ({n_clusters_claim})"
            ),
        )

    transformations: list[str] = []

    # Sanitize variable + cluster labels.
    safe_vars: list[str] = []
    safe_var_set: set[str] = set()
    for raw_v in raw_vars:
        sv = safe_key(raw_v)
        if sv in safe_var_set:
            return SanitizerResult(
                ok=False, analysis_type="cluster_analysis",
                rejection_reason=(
                    "variable label collision after sanitization"
                ),
            )
        safe_var_set.add(sv)
        safe_vars.append(sv)

    safe_labels: list[str] = []
    safe_label_set: set[str] = set()
    for raw_c in raw_labels:
        sl = safe_key(raw_c)
        if sl in safe_label_set:
            return SanitizerResult(
                ok=False, analysis_type="cluster_analysis",
                rejection_reason=(
                    "cluster label collision after sanitization"
                ),
            )
        safe_label_set.add(sl)
        safe_labels.append(sl)

    # Validate cluster_sizes structure + run the whole-cluster
    # suppression gate. Clusters below ``min_n_descriptive`` are
    # dropped whole — their entry in cluster_sizes, their centroid
    # row, their within-cluster SS, every per-cluster entry.
    raw_sizes = raw.get("cluster_sizes")
    if not isinstance(raw_sizes, dict):
        return SanitizerResult(
            ok=False, analysis_type="cluster_analysis",
            rejection_reason="cluster_sizes must be a dict",
        )

    cluster_n: dict[str, int] = {}
    declared_sizes: set[str] = set()
    for raw_k, raw_n in raw_sizes.items():
        if not isinstance(raw_k, str):
            continue
        sl = safe_key(raw_k)
        if sl not in safe_label_set:
            continue
        declared_sizes.add(sl)
        if not isinstance(raw_n, int) or isinstance(raw_n, bool) or raw_n < 0:
            return SanitizerResult(
                ok=False, analysis_type="cluster_analysis",
                rejection_reason=(
                    "cluster_sizes values must be non-negative ints"
                ),
            )
        cluster_n[sl] = raw_n

    # Every declared cluster_label needs a size entry — the gate
    # can't run otherwise.
    missing_sizes = [c for c in safe_labels if c not in declared_sizes]
    if missing_sizes:
        return SanitizerResult(
            ok=False, analysis_type="cluster_analysis",
            rejection_reason=(
                f"{len(missing_sizes)} cluster_label(s) have no "
                f"cluster_sizes entry. Labels withheld."
            ),
        )

    min_n_cluster = config.min_n_descriptive
    suppressed: set[str] = set()
    surviving: list[str] = []
    cleaned_sizes: dict[str, int] = {}
    for sl in safe_labels:
        size = cluster_n[sl]
        if size < min_n_cluster:
            suppressed.add(sl)
        else:
            surviving.append(sl)
            cleaned_sizes[sl] = size

    if suppressed:
        transformations.append(
            f"cluster suppression: {len(suppressed)} cluster(s) with "
            f"size < {min_n_cluster} dropped entirely (labels withheld "
            f"— cluster identities are disclosive when small)"
        )

    if not surviving:
        return SanitizerResult(
            ok=False, analysis_type="cluster_analysis",
            rejection_reason=(
                f"all clusters have size < {min_n_cluster}; nothing "
                f"survives the cluster-size gate"
            ),
        )

    out: dict[str, Any] = _collect_allowed(
        raw,
        numeric=_CLUSTER_ALLOWED_NUMERIC_FIELDS,
        integer=_CLUSTER_ALLOWED_INT_FIELDS,
        string=_CLUSTER_ALLOWED_STRING_FIELDS,
        list_string=_CLUSTER_ALLOWED_LIST_STRING,
        transformations=transformations,
    )
    out["type"] = "cluster_analysis"
    out["method"] = method
    out["variables"] = safe_vars
    out["n_observations"] = n_obs
    out["n_variables"] = n_features_claim  # back-compat synonym
    out["n_features"] = n_features_claim
    # cluster_labels list is the surviving set (alphabetized to
    # match the per-dict keys' order).
    out["cluster_labels"] = sorted(surviving)
    out["n_clusters"] = len(surviving)
    if len(surviving) != n_clusters_claim:
        transformations.append(
            f"n_clusters reduced from {n_clusters_claim} to "
            f"{len(surviving)} after cluster-size suppression"
        )
    out["cluster_sizes"] = cleaned_sizes

    # Linkage / distance_metric enum validation.
    lk = out.get("linkage")
    if lk is not None and lk not in _CLUSTER_VALID_LINKAGE:
        transformations.append(
            f"dropped 'linkage' value (must be one of "
            f"{sorted(_CLUSTER_VALID_LINKAGE)})"
        )
        del out["linkage"]
    dm = out.get("distance_metric")
    if dm is not None and dm not in _CLUSTER_VALID_DISTANCE:
        transformations.append(
            f"dropped 'distance_metric' value (must be one of "
            f"{sorted(_CLUSTER_VALID_DISTANCE)})"
        )
        del out["distance_metric"]

    # Centroids: nested {cluster: {variable: value}}. Outer keys
    # must be surviving clusters; inner keys in safe_vars.
    #
    # Conditionally required: methods in
    # ``_CLUSTER_METHODS_WITHOUT_CENTROIDS`` (DBSCAN, HDBSCAN) have
    # no centroids by construction — absent-OK. If a DBSCAN payload
    # ships centroids anyway (caller computed them post-hoc from the
    # labels), the structure is still validated below: cross-field
    # rules apply identically so the field can't smuggle anything
    # past the gate.
    #
    # Per-cluster precision clamping fires on the surviving
    # centroids — each value clamped by the cluster's OWN N rather
    # than the global n_observations. The 12-member cluster gets
    # fewer sigfigs than the 12,000-member cluster.
    centroids_raw = raw.get("centroids")
    centroids_absent_ok = method in _CLUSTER_METHODS_WITHOUT_CENTROIDS

    if centroids_raw is None:
        if not centroids_absent_ok:
            return SanitizerResult(
                ok=False, analysis_type="cluster_analysis",
                rejection_reason=(
                    f"centroids is required for method={method!r}; "
                    f"only DBSCAN-family methods may omit centroids"
                ),
            )
        # DBSCAN-family with no centroids — skip the centroid block.
    else:
        if not isinstance(centroids_raw, dict):
            return SanitizerResult(
                ok=False, analysis_type="cluster_analysis",
                rejection_reason="centroids must be a nested dict",
            )
        cleaned_centroids: dict[str, dict[str, float]] = {}
        dropped_outer = 0
        dropped_inner = 0
        surviving_set = set(surviving)
        for outer_k, inner_v in centroids_raw.items():
            if not isinstance(outer_k, str):
                dropped_outer += 1
                continue
            sl = safe_key(outer_k)
            if sl not in surviving_set:
                dropped_outer += 1
                continue
            if not isinstance(inner_v, dict):
                dropped_outer += 1
                continue
            cleaned_inner: dict[str, float] = {}
            for inner_k, val in inner_v.items():
                if not isinstance(inner_k, str):
                    dropped_inner += 1
                    continue
                sv = safe_key(inner_k)
                if sv not in safe_var_set:
                    dropped_inner += 1
                    continue
                if not _is_finite_number(val):
                    dropped_inner += 1
                    continue
                cleaned_inner[sv] = float(val)
            if cleaned_inner:
                # Per-cluster clamp via the named primitive: each
                # variable's centroid value gets clamped by THIS
                # cluster's N rather than the global n_observations.
                cleaned_centroids[sl] = clamp_precision_dict(
                    cleaned_inner, cleaned_sizes[sl],
                )
        if dropped_outer:
            transformations.append(
                f"dropped {dropped_outer} undeclared/suppressed cluster(s) "
                f"from centroids (labels withheld)"
            )
        if dropped_inner:
            transformations.append(
                f"dropped {dropped_inner} undeclared variable entry(ies) "
                f"from centroids"
            )
        # For non-DBSCAN methods, centroids must be non-empty after
        # the gate; for DBSCAN, an empty centroids dict (e.g. all
        # centroid clusters were sub-min-N) is acceptable since
        # centroids weren't required.
        if not cleaned_centroids and not centroids_absent_ok:
            return SanitizerResult(
                ok=False, analysis_type="cluster_analysis",
                rejection_reason=(
                    "centroids dict empty after sanitization — keys did "
                    "not match declared (and surviving) "
                    "clusters/variables"
                ),
            )
        if cleaned_centroids:
            out["centroids"] = cleaned_centroids
            transformations.append(
                "centroid precision clamped per-cluster (each value's "
                "sigfigs scales with that cluster's size, not global N)"
            )

    # Per-cluster numeric dicts (within_cluster_ss, silhouette_per_cluster):
    # keys must be surviving clusters; values precision-clamped by
    # the cluster's OWN N via the named ``clamp_dict_by_per_key_n``
    # primitive. Each metric is an aggregate over the cluster's
    # members (within-SS is a sum over the cluster's points;
    # silhouette is a mean over them), so local N is the right
    # precision floor — same reasoning as the centroid clamp.
    for field in _CLUSTER_PER_CLUSTER_NUMERIC_DICTS:
        v = raw.get(field)
        if v is None:
            continue
        if not isinstance(v, dict):
            transformations.append(
                f"dropped {field!r}: expected dict, got {type(v).__name__}"
            )
            continue
        cleaned: dict[str, float] = {}
        for k, val in v.items():
            if not isinstance(k, str):
                continue
            sl = safe_key(k)
            if sl not in set(surviving):
                continue
            if not _is_finite_number(val):
                continue
            cleaned[sl] = float(val)
        if cleaned:
            # Per-cluster precision clamp via the named primitive.
            out[field] = clamp_dict_by_per_key_n(cleaned, cleaned_sizes)

    # Per-variable numeric dicts (f_statistic_per_variable): keys
    # must be declared variable names; values are aggregate scalars
    # (between-cluster F-stat per variable) — global-N clamp is the
    # right precision floor since the F is computed over the full
    # sample, not a subgroup.
    for field in _CLUSTER_PER_VARIABLE_NUMERIC_DICTS:
        v = raw.get(field)
        if v is None:
            continue
        if not isinstance(v, dict):
            transformations.append(
                f"dropped {field!r}: expected dict, got {type(v).__name__}"
            )
            continue
        cleaned_var: dict[str, float] = {}
        for k, val in v.items():
            if not isinstance(k, str):
                continue
            sv = safe_key(k)
            if sv not in safe_var_set:
                continue
            if not _is_finite_number(val):
                continue
            cleaned_var[sv] = clamp_precision(float(val), n_obs)
        if cleaned_var:
            out[field] = cleaned_var

    # Precision-clamp scalar numeric fields (total_within_ss, etc.)
    # by global N.
    for key in _CLUSTER_ALLOWED_NUMERIC_FIELDS:
        if key in out:
            out[key] = clamp_precision(out[key], n_obs)

    return SanitizerResult(
        ok=True, analysis_type="cluster_analysis",
        sanitized=out, transformations=transformations,
    )


# ---------------------------------------------------------------------------
# Handler registry
# ---------------------------------------------------------------------------

_HANDlerFn = Callable[[dict[str, Any], SDCConfig], SanitizerResult]

_HANDLERS: dict[str, _HANDlerFn] = {
    # Regression bucket — canonical name and legacy alias both
    # dispatch to the same sanitizer. The output's ``analysis_type``
    # mirrors whichever name the input used, so existing stored
    # payloads keep their old name on read and new emissions carry
    # the canonical name.
    _REGRESSION_TYPE_CANONICAL: _sanitize_linear_regression,
    _REGRESSION_TYPE_LEGACY:    _sanitize_linear_regression,
    "t_test": _sanitize_t_test,
    "descriptive": _sanitize_descriptive,
    "frequency_table": _sanitize_frequency_table,
    "crosstab": _sanitize_crosstab,
    "magnitude_table": _sanitize_magnitude_table,
    "correlation_matrix": _sanitize_correlation_matrix,
    "did_event_study": _sanitize_did_event_study,
    "rdd": _sanitize_rdd,
    "kaplan_meier": _sanitize_kaplan_meier,
    "factor_decomposition": _sanitize_factor_decomposition,
    "cluster_analysis": _sanitize_cluster_analysis,
}


def supported_types() -> list[str]:
    """Return the list of analysis types the sanitizer currently accepts."""
    return sorted(_HANDLERS.keys())
