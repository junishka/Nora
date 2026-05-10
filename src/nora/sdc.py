"""Nora — Statistical Disclosure Control primitives.

These are the small, testable building blocks the per-analysis-type
sanitizers compose. Each operation is deliberately narrow; each has a
clear disclosive concern it's defending against:

- `round_to_sigfigs`: a regression coefficient reported to 15 decimal
  places on small-N data is a fingerprint for the exact dataset — the
  researcher's data can be matched against it. Rounding to N-appropriate
  significant figures neutralizes that channel.
- `sigfigs_for_n`: how many sig figs is safe scales roughly with sample
  size. This is a conservative default; step 5's proper SDC pass should
  refine it against the Eurostat / UK ONS guidance.
- `suppress_cells_below`: frequency-table cells below a threshold identify
  individuals. A cell of size 1 re-identifies one observation directly.
  This implements *primary* suppression only — *secondary* suppression
  (margin-back-solve protection via LP) is step 5 territory with τ-ARGUS.
- `MinimumNViolation`: a hard-reject sentinel for when a whole analysis
  result is too small to publish at any precision.

Everything here is pure — no I/O, no globals. The sanitizer composes
these into per-type policies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# Sentinel string for suppressed frequency cells. A few reasons to use a
# string rather than None or a dict:
# - JSON-serializable without special handling.
# - Visibly different from "absent" (None) which conflates "no data" with
#   "suppressed for disclosure".
# - The threshold is embedded so the reader of the result knows the rule
#   without having to look up a config.
def suppression_marker(threshold: int) -> str:
    """Return the string Nora uses to mark a cell suppressed below `threshold`."""
    return f"<{threshold}"


class MinimumNViolation(Exception):
    """Raised when a hard SDC rule (minimum N) would be violated.

    Carries a machine-readable reason so the sanitizer can forward it as a
    policy rejection rather than an exception trace.
    """

    def __init__(self, field: str, actual: int, required: int):
        self.field = field
        self.actual = actual
        self.required = required
        super().__init__(
            f"{field}={actual} is below the minimum threshold of {required} "
            f"required by SDC policy"
        )


# ---------------------------------------------------------------------------
# Precision clamping
# ---------------------------------------------------------------------------

# Hard upper bound on sig figs, regardless of N. Even at billion-row
# datasets, 8 sig figs is more precision than analysis needs and starts to
# leak information about fit quality on specific observations.
_SIGFIGS_CAP = 8

# Hard floor: below this, rounding becomes uselessly coarse.
_SIGFIGS_FLOOR = 3


def sigfigs_for_n(n: int, base: int = 3) -> int:
    """Return a safe number of significant figures given sample size `n`.

    Scales roughly with `log10(n) / 2`, so each doubling of order-of-
    magnitude in sample size buys one more sig fig. Clamped to
    [_SIGFIGS_FLOOR, _SIGFIGS_CAP]. At N=10, 3 sig figs. At N=10_000,
    5 sig figs. At N=10_000_000, 7 sig figs.

    This is a placeholder default. Step 5 should replace with a function
    whose choice is traceable to SDC literature.
    """
    if n <= 0:
        return _SIGFIGS_FLOOR
    proposed = base + (int(math.floor(math.log10(n))) // 2)
    return max(_SIGFIGS_FLOOR, min(_SIGFIGS_CAP, proposed))


def round_to_sigfigs(x: float, sigfigs: int) -> float:
    """Round `x` to `sigfigs` significant figures.

    Python's built-in `round` rounds to decimal places, not sig figs.
    This handles magnitude via log10 and falls back cleanly for 0, NaN,
    and Inf (which are passed through unchanged — rounding them is
    meaningless).
    """
    if sigfigs <= 0:
        raise ValueError(f"sigfigs must be positive, got {sigfigs}")
    if x == 0 or not math.isfinite(x):
        return x
    magnitude = math.floor(math.log10(abs(x)))
    decimals = sigfigs - 1 - int(magnitude)
    return round(x, decimals)


def clamp_precision(x: float, n: int) -> float:
    """Convenience: round `x` to the safe sig-fig count for sample size `n`."""
    return round_to_sigfigs(x, sigfigs_for_n(n))


def clamp_precision_dict(d: dict[str, float], n: int) -> dict[str, float]:
    """Apply clamp_precision to every value in a dict."""
    return {k: clamp_precision(v, n) for k, v in d.items()}


# ---------------------------------------------------------------------------
# Cell suppression (primary only — secondary suppression is step 5)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SuppressionResult:
    """Outcome of applying cell suppression to a count dict.

    `suppressed_keys` is the list of keys whose counts fell below the
    threshold and were replaced with the marker. `total_original` is the
    sum of all original counts (for sanity-checking that suppression
    didn't change the implied N).
    """
    counts: dict[str, int | str]
    suppressed_keys: list[str]
    threshold: int
    total_original: int


def suppress_cells_below(
    counts: dict[str, int], threshold: int
) -> SuppressionResult:
    """Replace any count < `threshold` with the suppression marker string.

    This is **primary** suppression only. If a 1D table has marginals
    (row/column totals) present alongside cells, a single suppressed cell
    can be back-calculated from the margin — the caller is responsible
    for ensuring the total isn't included in the returned payload, or for
    applying secondary suppression (step 5, via τ-ARGUS).

    Returns a `SuppressionResult` so callers can report which cells were
    suppressed (useful for transformation logs the researcher sees).
    """
    if threshold < 1:
        # `threshold = 0` would silently disable suppression (no count
        # `v >= 0` ever satisfies `v < 0`), defeating the purpose of
        # the routine. Require >= 1 so a misconfigured `SDCConfig`
        # surfaces as a hard error rather than a silent zero-cell-
        # suppressed result.
        raise ValueError(
            f"threshold must be at least 1, got {threshold}; a value "
            f"of 0 would silently disable suppression"
        )
    marker = suppression_marker(threshold)
    out: dict[str, int | str] = {}
    suppressed: list[str] = []
    total = 0
    for k, v in counts.items():
        if not isinstance(v, int):
            raise TypeError(
                f"cell count for key {k!r} must be int, got {type(v).__name__}"
            )
        if v < 0:
            raise ValueError(f"cell count for key {k!r} is negative: {v}")
        total += v
        if v < threshold:
            out[k] = marker
            suppressed.append(k)
        else:
            out[k] = v
    return SuppressionResult(
        counts=out,
        suppressed_keys=suppressed,
        threshold=threshold,
        total_original=total,
    )


# ---------------------------------------------------------------------------
# Secondary suppression (1D tables with total)
# ---------------------------------------------------------------------------

def enforce_back_calc_safety(
    suppression: "SuppressionResult", total_n_present: bool
) -> "SuppressionResult":
    """Secondary suppression for 1D frequency tables that publish a total.

    If the researcher / Claude sees ``n`` alongside the suppressed counts,
    exactly one primary-suppressed cell is trivially back-solvable:
    ``suppressed_value = n - sum(non_suppressed) - sum(other_suppressed)``.

    The fix, per standard SDC guidance (UK ONS / Eurostat), is to make
    sure at least two cells are suppressed so no single value is
    isolable from the margin. This helper finds the **smallest
    non-suppressed** cell and additionally suppresses it whenever:

    - A total is being published (``total_n_present=True``)
    - Exactly one cell was primary-suppressed
    - There is another cell available to sacrifice

    Returns a new ``SuppressionResult`` with the secondary suppression
    applied and its key appended to ``suppressed_keys``. No-ops when
    the conditions don't hold.

    This is NOT a full secondary-suppression algorithm — 2D tables
    with row + column margins require linear programming (τ-ARGUS).
    This handles the 1D case, which is what the v0 frequency_table
    schema produces.
    """
    if not total_n_present or len(suppression.suppressed_keys) != 1:
        return suppression
    # Find the smallest remaining integer cell.
    candidates = [
        (k, v) for k, v in suppression.counts.items()
        if isinstance(v, int)
    ]
    if not candidates:
        return suppression  # nothing left to suppress
    # Secondary target: smallest count. Ties broken by key order for
    # determinism (tests need reproducibility).
    victim_key, _ = min(candidates, key=lambda kv: (kv[1], kv[0]))
    new_counts = dict(suppression.counts)
    new_counts[victim_key] = suppression_marker(suppression.threshold)
    new_suppressed = list(suppression.suppressed_keys) + [victim_key]
    return SuppressionResult(
        counts=new_counts,
        suppressed_keys=new_suppressed,
        threshold=suppression.threshold,
        total_original=suppression.total_original,
    )


# ---------------------------------------------------------------------------
# Dominance rule (magnitude tables)
# ---------------------------------------------------------------------------

# Default (1, k)-dominance threshold: if any single contributor provides
# more than this fraction of a cell's total, suppress the cell. UK ONS
# typically uses 0.85 for a single-contributor rule; Eurostat has similar
# guidance for the (n, k)-dominance family. Conservative default here.
DOMINANCE_THRESHOLD_DEFAULT = 0.85


def dominance_fails(max_share: float, threshold: float = DOMINANCE_THRESHOLD_DEFAULT) -> bool:
    """True if a cell's largest-contributor share exceeds `threshold`.

    Callers compute `max_share` as ``|max(values)| / |sum(values)|`` on
    the raw values inside a group (done in the runtime library since the
    sanitizer never sees raw values). When this returns True, the
    caller must suppress the cell's value — the sum effectively
    reveals the dominant contributor's value even if n looks safe.

    Values outside [0, 1] get clamped conceptually: any negative or >1
    share is treated as disclosive (the math went sideways, e.g. mixed
    signs, so be safe).
    """
    if not math.isfinite(max_share):
        return True
    if max_share < 0 or max_share > 1:
        return True
    return max_share > threshold


# ---------------------------------------------------------------------------
# Minimum-N gate
# ---------------------------------------------------------------------------

def require_minimum_n(n: int, threshold: int, field: str = "n") -> None:
    """Raise `MinimumNViolation` if `n < threshold`, else no-op.

    Callers convert this into a top-level rejection of the whole payload.
    Used for analyses whose entire output is too disclosive below a
    sample-size threshold (e.g., regression on N=5 — no precision
    clamping makes the coefficients safe).
    """
    if n < threshold:
        raise MinimumNViolation(field=field, actual=n, required=threshold)
