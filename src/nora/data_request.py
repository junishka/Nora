"""Nora — ``request_data`` handler.

The MCP ``request_data`` tool lets Claude ask for a specific, bounded
fact about a variable — more than the schema gives, less than a script
would produce. Every response goes through SDC rules before Claude
sees it.

**Design principle:** each request type has a narrow, pre-approved
output shape. The set of request types is the entire "what Claude can
learn about the data beyond the schema" surface — if Claude needs
something outside the set, they must write a ``submit_script`` and
let the sanitizer there handle it.

**v0 request types:**

- ``categorical_levels`` — returns the list of level *names* whose
  counts meet the threshold. Low-count levels are *hidden entirely*
  (their names and counts are both suppressed), only a tally of how
  many levels were suppressed is revealed. Level names themselves
  can be disclosive (rare-disease codes, specific ethnicities).

- ``numeric_bounds`` — returns the 5th and 95th percentile of a
  numeric variable, rounded to 2 significant figures. We deliberately
  do NOT return min/max: those are individual observations. The
  90%-inner percentiles blur extremes while still giving Claude a
  useful sense of scale.

- ``na_count`` — returns the count of NA observations for a variable,
  and total observation count. NA counts by themselves are scalar
  metadata about the pipeline, not a subgroup breakdown — low
  disclosure risk. We suppress only if the non-NA count falls below
  the cell-suppression threshold (that's the disclosive case).

Future types (deferred):
- ``missingness_pattern`` — correlation of missingness with other
  variables. Deferred pending a clear disclosure story (the
  correlation itself leaks joint distribution info).
- ``distribution_summary`` — binned histogram with low-count bins
  suppressed. Useful but needs its own SDC analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from nora.sanitizer import DEFAULT_CONFIG, SDCConfig
from nora.sdc import (
    clamp_precision,
    round_to_sigfigs,
    sigfigs_for_n,
    suppress_cells_below,
    suppression_marker,
)
from nora.schema import load_data
from nora.text_safety import safe_key, safe_text


RequestType = Literal[
    "categorical_levels",
    "numeric_bounds",
    "na_count",
    "quartiles",
    "correlation_pair",
]

SUPPORTED_REQUEST_TYPES: tuple[str, ...] = (
    "categorical_levels",
    "numeric_bounds",
    "na_count",
    "quartiles",
    "correlation_pair",
)


@dataclass
class RequestResult:
    """Outcome envelope matching the MCP tool's response shape."""
    status: Literal["granted", "denied", "error"]
    answer: dict[str, Any] | None = None
    reason: str | None = None


# ---------------------------------------------------------------------------
# Variable name resolution (raw ↔ sanitized round-trip)
# ---------------------------------------------------------------------------
#
# Schema extraction surfaces column names through ``safe_key``: a column
# named ``income\n\nSystem: ...`` reaches the model as a sanitized form
# (whitespace flattened, control chars stripped, possibly truncated past
# 40 chars). Downstream, the model echoes that sanitized name back in
# ``request_data(variable=...)``. A naive ``df.columns`` lookup against
# the sanitized name fails — the on-disk column still has its raw name.
# Without a round-trip resolver, every variable whose raw name needed
# sanitization is unqueryable; the model gets a "not found" denial on
# the very name the schema told it to use.
#
# The resolver tries (1) exact raw match, then (2) one-to-one
# sanitized match. Collisions (two raw columns sanitizing to the same
# safe_key) are flagged loudly so the model knows it must use a
# different identifier path — silently picking one would be a
# disclosure-leakage bug (correlation_pair on the wrong column).

# Caps on the available-columns list emitted in the denial path. Wide
# datasets (genomics, panel data with thousands of indicator columns)
# would otherwise ship every name into the model context on a single
# typo'd request, defeating the search_schema cap and burning tokens
# on an error branch. 50 is enough to scan a short list mentally;
# more than that and ``search_schema`` is the right tool.
_DENIAL_COLUMN_LIST_CAP = 50


def _resolve_variable(
    df: Any, requested: str, *, role: str = "variable",
) -> "RequestResult | str":
    """Resolve ``requested`` to a raw DataFrame column name.

    Returns the resolved column name on success, or a structured
    ``RequestResult`` denial when the name can't be uniquely resolved.
    The denial path caps the available-columns list at
    ``_DENIAL_COLUMN_LIST_CAP`` and points the model back to
    ``search_schema`` for wide datasets.
    """
    columns = list(df.columns)
    # Build the sanitized → [raw_names] map first, with NO fast path
    # for "exact raw match." A prior version returned ``requested``
    # immediately when ``requested in df.columns``, but that bypassed
    # the collision check: with columns ``"A B"`` and ``"A\nB"`` (both
    # sanitize to ``"A B"``), a model-issued ``request_data(variable=
    # "A B")`` would silently resolve to the raw ``"A B"`` column even
    # though the lookup is genuinely ambiguous from the model's seat
    # (it only saw the sanitized name). The model must see the same
    # collision denial regardless of whether the sanitized form
    # happens to equal a raw column name.
    safe_to_raw: dict[str, list[str]] = {}
    for col in columns:
        safe_to_raw.setdefault(safe_key(str(col)), []).append(str(col))
    matches = safe_to_raw.get(requested, [])
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        # Collision: the sanitized name is ambiguous. Surface the
        # collision count rather than picking one — the model needs to
        # know to use a different path (e.g., write a script that
        # references the column by its raw bytes via a DataFrame
        # method, or rename the column upstream).
        return RequestResult(
            status="denied",
            reason=(
                f"{role} {safe_key(str(requested))!r} matches "
                f"{len(matches)} columns whose sanitized names collide. "
                f"The raw column names cannot be safely echoed back to "
                f"you (data-origin strings are an injection surface), "
                f"so this lookup is ambiguous. Rename one of the "
                f"colliding columns in the dataset, or run a script "
                f"that references the column by index instead."
            ),
        )
    # No match. Return a structured denial with a capped column
    # listing — wide datasets must not ship the full column list
    # through the error path.
    safe_requested = safe_key(str(requested))
    safe_columns = [safe_key(str(c)) for c in columns]
    total = len(safe_columns)
    truncated = total > _DENIAL_COLUMN_LIST_CAP
    listed = safe_columns[:_DENIAL_COLUMN_LIST_CAP]
    if truncated:
        suffix = (
            f" {total - _DENIAL_COLUMN_LIST_CAP} more column(s) elided. "
            f"Use search_schema(query=...) to find the right column "
            f"on a wide dataset rather than scanning the full list."
        )
    else:
        suffix = ""
    return RequestResult(
        status="denied",
        reason=(
            f"{role} {safe_requested!r} not found in dataset. "
            f"Available columns ({len(listed)} of {total}): "
            f"{listed!r}.{suffix}"
        ),
    )


# ---------------------------------------------------------------------------
# Top-level dispatch
# ---------------------------------------------------------------------------

def handle(
    dataset_path: Path,
    request_type: str,
    variable: str,
    config: SDCConfig = DEFAULT_CONFIG,
    *,
    variable2: str | None = None,
) -> RequestResult:
    """Compute the requested fact on real data and apply SDC rules.

    Returns the sanitized answer, or a structured denial / error. Never
    raises for normal failure modes (missing variable, unsupported
    type, etc.) — those become ``status=denied`` or ``status=error``
    with a ``reason`` the caller can forward to Claude.

    ``variable2`` is consumed only by the multi-variable request types
    (``correlation_pair``); single-variable types ignore it. Passing it
    to a single-variable type is silently OK rather than rejected so a
    caller composing requests dynamically doesn't need per-type
    branching just to set the field.
    """
    if request_type not in SUPPORTED_REQUEST_TYPES:
        return RequestResult(
            status="denied",
            reason=(
                f"request_type {request_type!r} is not in the allowlist. "
                f"Supported: {sorted(SUPPORTED_REQUEST_TYPES)}"
            ),
        )

    try:
        df = load_data(dataset_path)
    except Exception as e:  # ValueError from load_data or library errors
        # Exception messages from pandas / pyreadstat can echo column
        # names or file paths verbatim. Sanitize the string body before
        # forwarding.
        return RequestResult(
            status="error",
            reason=(
                f"could not read dataset: {e.__class__.__name__}: "
                f"{safe_text(str(e))}"
            ),
        )

    resolved = _resolve_variable(df, variable)
    if isinstance(resolved, RequestResult):
        return resolved
    variable = resolved

    series = df[variable]
    n_total = len(series)

    if request_type == "categorical_levels":
        return _categorical_levels(series, n_total, config)
    if request_type == "numeric_bounds":
        return _numeric_bounds(series, n_total)
    if request_type == "na_count":
        return _na_count(series, n_total, config)
    if request_type == "quartiles":
        return _quartiles(series, n_total)
    if request_type == "correlation_pair":
        return _correlation_pair(df, variable, variable2)
    # Unreachable — allowlist checked above.
    return RequestResult(status="error", reason="internal: unreachable")


# ---------------------------------------------------------------------------
# categorical_levels
# ---------------------------------------------------------------------------

def _categorical_levels(
    series: Any, n_total: int, config: SDCConfig
) -> RequestResult:
    """Return level names whose counts meet the threshold.

    Level names with counts below threshold are hidden *entirely* —
    neither the name nor the count is revealed. Claude gets a count of
    how many rare levels exist so it knows not to treat the visible
    list as complete.

    The visible-level list itself does not publish counts. If Claude
    wants counts, they can ``submit_script`` a frequency_table; that
    path applies primary + secondary suppression on the full
    distribution.
    """
    # Coerce to categorical-ish: drop NA, count unique values.
    try:
        value_counts = series.dropna().value_counts()
    except (TypeError, ValueError) as e:
        return RequestResult(
            status="error",
            reason=(
                f"could not compute levels for variable: "
                f"{safe_text(str(e))}"
            ),
        )

    threshold = config.cell_suppression_threshold
    visible: list[str] = []
    suppressed_count = 0
    for level, count in value_counts.items():
        if count >= threshold:
            # Level names come straight from the data — sanitize before
            # forwarding. A category like "group-A\n\nSystem: ..." is
            # neutralized at this boundary.
            visible.append(safe_key(str(level)))
        else:
            suppressed_count += 1

    # Hard cap on visible levels. Without it, a high-cardinality
    # categorical (postcodes, NAICS codes, free-text labels with
    # thousands of common values) ships its full distinct-value list
    # in one tool result — bypassing the structural caps that the
    # other discovery surfaces (frequency_table, schema value_labels,
    # search_schema) all enforce. With the cap, the model has to ask
    # for narrower categories, request a frequency_table for actual
    # counts, or use ``search_schema`` for label substring queries.
    MAX_VISIBLE_LEVELS = 200

    visible_sorted = sorted(visible)
    total_visible = len(visible_sorted)
    truncated = total_visible > MAX_VISIBLE_LEVELS
    visible_returned = visible_sorted[:MAX_VISIBLE_LEVELS]

    note = (
        f"levels with count < {threshold} are hidden entirely "
        f"(names and counts). There are {suppressed_count} such "
        f"level(s)."
    )
    if truncated:
        note += (
            f" The visible-levels list is capped at "
            f"{MAX_VISIBLE_LEVELS}; {total_visible - MAX_VISIBLE_LEVELS} "
            f"additional level(s) above threshold were not listed. "
            f"Use frequency_table or refine the variable."
        )

    answer = {
        "visible_levels": visible_returned,
        "visible_level_count_total": total_visible,
        "visible_levels_truncated": truncated,
        "suppressed_level_count": suppressed_count,
        "note": note,
    }
    return RequestResult(
        status="granted",
        answer=answer,
    )


# ---------------------------------------------------------------------------
# numeric_bounds
# ---------------------------------------------------------------------------

def _numeric_bounds(series: Any, n_total: int) -> RequestResult:
    """Return rounded 5th and 95th percentiles of a numeric variable.

    Rounded to 2 significant figures. We use the 5th/95th percentiles
    rather than min/max because the latter are single-observation
    values; a researcher's one high-income respondent is identifiable
    from max income alone. The 5th/95th are still individual values
    but draw from a much larger pool at scale.

    Also returns n_nonmissing so Claude knows the effective sample.
    """
    import pandas as pd

    if not pd.api.types.is_numeric_dtype(series):
        # series.dtype is a pandas/numpy dtype object whose repr is
        # effectively controlled (no injection risk), but sanitize for
        # defense in depth at the boundary.
        return RequestResult(
            status="denied",
            reason=(
                "numeric_bounds requires a numeric variable; "
                f"this variable has dtype {safe_key(str(series.dtype))!r}"
            ),
        )

    non_na = series.dropna()
    n_effective = int(len(non_na))
    # Tail percentiles (5th / 95th) at small N are interpolations
    # adjacent to the min and max — at N=10, the 5th percentile sits
    # between the 1st and 2nd order statistic and rounds, even at 2
    # sig figs, to a value that effectively reveals the bottom
    # outlier. We require N >= 30 so the percentile is averaged over
    # roughly 1.5 - 2.5 observations on either tail rather than
    # essentially echoing back the extremes. This is stricter than
    # cell_suppression_threshold (10) on purpose: the extra factor
    # of 3 buys real interpolation breadth.
    NUMERIC_BOUNDS_MIN_N = 30
    if n_effective < NUMERIC_BOUNDS_MIN_N:
        return RequestResult(
            status="denied",
            reason=(
                f"variable has only {n_effective} non-missing observations "
                f"— too few for tail-percentile bounds (need at least "
                f"{NUMERIC_BOUNDS_MIN_N}). At small N the 5th and 95th "
                f"percentiles interpolate close to the min and max and "
                f"would identify the tail individuals."
            ),
        )

    p5 = float(non_na.quantile(0.05))
    p95 = float(non_na.quantile(0.95))
    return RequestResult(
        status="granted",
        answer={
            "percentile_5": round_to_sigfigs(p5, 2),
            "percentile_95": round_to_sigfigs(p95, 2),
            "precision": "2 significant figures",
            "n_nonmissing": n_effective,
            "note": (
                "5th and 95th percentiles are returned instead of min/max "
                "to avoid revealing extreme individual observations."
            ),
        },
    )


# ---------------------------------------------------------------------------
# na_count
# ---------------------------------------------------------------------------

def _na_count(series: Any, n_total: int, config: SDCConfig) -> RequestResult:
    """Return the NA count for a variable.

    NA counts by themselves are pipeline metadata, not subgroup
    breakdowns — disclosure risk is low. We still suppress when the
    *non-NA* count is below threshold (the genuinely disclosive case:
    "only 3 people have a non-NA value for this variable" identifies
    those people by inverse).
    """
    na_count = int(series.isna().sum())
    non_na_count = n_total - na_count
    threshold = config.cell_suppression_threshold
    if non_na_count < threshold:
        return RequestResult(
            status="denied",
            reason=(
                f"only {non_na_count} non-missing observation(s) — below "
                f"the disclosure threshold of {threshold}. The count is "
                f"suppressed because the non-missing subgroup is too small."
            ),
        )
    return RequestResult(
        status="granted",
        answer={
            "na_count": na_count,
            "non_na_count": non_na_count,
            "total": n_total,
        },
    )


# ---------------------------------------------------------------------------
# quartiles
# ---------------------------------------------------------------------------

def _quartiles(series: Any, n_total: int) -> RequestResult:
    """Return rounded 25th and 75th percentiles of a numeric variable.

    Pairs with ``numeric_bounds`` (5th / 95th) to give the model an IQR-
    style sense of the distribution's middle. The 50th percentile
    (median) is deliberately NOT returned — for any odd-N variable it
    is exactly an individual observation, and the system prompt's
    forbidden-fields rule already names ``min/max/median`` as
    disclosive at the row level. Rounding to 2 sig figs is the same
    posture as ``numeric_bounds``.
    """
    import pandas as pd

    if not pd.api.types.is_numeric_dtype(series):
        return RequestResult(
            status="denied",
            reason=(
                "quartiles requires a numeric variable; "
                f"this variable has dtype {safe_key(str(series.dtype))!r}"
            ),
        )

    non_na = series.dropna()
    n_effective = int(len(non_na))
    if n_effective < 10:
        return RequestResult(
            status="denied",
            reason=(
                f"variable has only {n_effective} non-missing observations "
                f"— too few to publish quartiles without identifying "
                f"individuals."
            ),
        )

    q25 = float(non_na.quantile(0.25))
    q75 = float(non_na.quantile(0.75))
    # Compute the published IQR by SUBTRACTING the rounded quartiles
    # rather than independently rounding ``q75 - q25``. Independently
    # rounded triples over-determine the system: comparing
    # ``rounded(q75) - rounded(q25)`` against an independently-rounded
    # IQR recovers ~1 extra bit per quartile from the rounding-error
    # disagreement. Holding ``iqr == p75 - p25`` exactly removes that
    # channel — the model now sees three numbers that are mutually
    # consistent at the published precision, with no over-determined
    # constraint to invert.
    rounded_q25 = round_to_sigfigs(q25, 2)
    rounded_q75 = round_to_sigfigs(q75, 2)
    return RequestResult(
        status="granted",
        answer={
            "percentile_25": rounded_q25,
            "percentile_75": rounded_q75,
            "iqr": rounded_q75 - rounded_q25,
            "precision": "2 significant figures",
            "n_nonmissing": n_effective,
            "note": (
                "25th and 75th percentiles are returned. The 50th "
                "(median) is deliberately omitted: for any odd-N "
                "variable it is exactly an individual observation, "
                "which the SDC rules forbid at the row level."
            ),
        },
    )


# ---------------------------------------------------------------------------
# correlation_pair
# ---------------------------------------------------------------------------

def _correlation_pair(
    df: Any, var1: str, var2: str | None,
) -> RequestResult:
    """Pearson correlation between two numeric variables.

    Multi-variable correlation matrices have their own sanitizer type
    (``correlation_matrix`` via ``submit_script``) — this fast path is
    for the common "is X correlated with Y" question that doesn't
    warrant a full script.

    Returns the correlation coefficient (rounded), the complete-case N
    (rows with both variables observed), and the missing count. The
    correlation is a pure aggregate over sums-of-products; no per-row
    leak. We still gate on the same minimum N as ``numeric_bounds`` —
    at low N a near-perfect correlation is just "the three points are
    collinear" and could imply individual coordinates.
    """
    import pandas as pd

    if not var2:
        return RequestResult(
            status="denied",
            reason=(
                "correlation_pair requires both ``variable`` (the "
                "first variable) and ``variable2`` (the second). "
                "Pass both."
            ),
        )

    resolved = _resolve_variable(df, var2, role="variable2")
    if isinstance(resolved, RequestResult):
        return resolved
    var2 = resolved

    if var1 == var2:
        return RequestResult(
            status="denied",
            reason=(
                "correlation_pair: variable and variable2 must differ. "
                "A variable's correlation with itself is always 1; "
                "the request is structurally redundant."
            ),
        )

    s1 = df[var1]
    s2 = df[var2]
    if not pd.api.types.is_numeric_dtype(s1):
        return RequestResult(
            status="denied",
            reason=(
                f"correlation_pair: ``variable`` ({safe_key(str(var1))!r}) "
                f"has dtype {safe_key(str(s1.dtype))!r}, not numeric"
            ),
        )
    if not pd.api.types.is_numeric_dtype(s2):
        return RequestResult(
            status="denied",
            reason=(
                f"correlation_pair: ``variable2`` ({safe_key(str(var2))!r}) "
                f"has dtype {safe_key(str(s2.dtype))!r}, not numeric"
            ),
        )

    pair = pd.concat([s1, s2], axis=1).dropna()
    n_complete = int(len(pair))
    if n_complete < 10:
        return RequestResult(
            status="denied",
            reason=(
                f"only {n_complete} row(s) with both variables observed "
                f"— too few to publish a correlation without identifying "
                f"individuals (a near-perfect r at small N usually just "
                f"says 'these three points are collinear')."
            ),
        )

    r = float(pair[var1].corr(pair[var2]))
    # Pearson is undefined when either column has zero variance
    # (constant column, or a perfectly-imputed series). pandas returns
    # NaN there. Don't ship NaN as a "granted" numeric answer — it
    # serializes to a non-strict-JSON token and forces every consumer
    # to special-case the value. Reject with a reason that names the
    # constant column so the model knows what to fix.
    import math
    if not math.isfinite(r):
        zero_var: list[str] = []
        for name, series in ((var1, pair[var1]), (var2, pair[var2])):
            try:
                if float(series.std(ddof=0)) == 0.0:
                    zero_var.append(safe_key(str(name)))
            except (TypeError, ValueError):
                continue
        if zero_var:
            culprits = " and ".join(repr(v) for v in zero_var)
            reason = (
                f"correlation_pair: undefined because {culprits} "
                f"{'has' if len(zero_var) == 1 else 'have'} zero "
                f"variance on the complete-case rows. A constant "
                f"column has no correlation with anything."
            )
        else:
            reason = (
                "correlation_pair: result is not finite (NaN/Inf). "
                "This usually means one of the two variables has zero "
                "variance on the complete-case rows. Drop the constant "
                "column or restrict the sample."
            )
        return RequestResult(status="denied", reason=reason)

    sigfigs = sigfigs_for_n(n_complete)
    return RequestResult(
        status="granted",
        answer={
            "variable": safe_key(str(var1)),
            "variable2": safe_key(str(var2)),
            "correlation": round_to_sigfigs(r, sigfigs),
            "method": "pearson",
            "n_complete": n_complete,
            "missing_count": int(len(s1) - n_complete),
            "note": (
                "Pearson correlation between the two variables on rows "
                "where BOTH are observed. For a multi-variable matrix "
                "use submit_script + nora$from_correlation / "
                "nora.from_correlation."
            ),
        },
    )
