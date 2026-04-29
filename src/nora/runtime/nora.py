"""Nora runtime library for Python.

Imported at the top of every Python script Nora runs. Provides the
single sanctioned I/O surface for emitting structured results:

    nora.result(type="linear_regression", ...)
    nora.from_lm(model)              # statsmodels OLS / GLM result
    nora.from_t_test(res, n1=..., n2=...)   # scipy.stats t-test
    nora.from_summarize(variable, n, mean, sd, missing_count)
    nora.from_table(variable, counts, n=..., missing_count=...)
    nora.from_crosstab(table)
    nora.from_magnitude_table(df, group_var, value_var, aggregation="sum")

The script writes structured payloads to ``$NORA_RESULT_PATH``. Raw
stdout / stderr are captured by the executor as the raw log the
researcher sees in the TUI; only the structured JSON reaches the
sanitizer (and from there, the model).

Hard requirements: ``pandas`` and ``numpy`` are needed to load the
module — the runtime ships a small numpy/pandas-aware JSON encoder
so floats / int64 / NaN serialise cleanly. ``statsmodels`` and
``scipy`` are needed only by the ``from_lm`` and ``from_t_test``
helpers; scripts that emit via the generic ``result(...)`` path or
the descriptive helpers don't need them.

NOTE on numeric precision: floats are emitted at full IEEE-754
precision. The Python sanitizer (``nora.sanitizer``) clamps
precision per-type using ``sigfigs_for_n`` after the payload comes
back. Mirrors the R / Stata libraries — language-of-origin doesn't
change what the model sees.
"""

from __future__ import annotations

import json
import math
import os
import sys
from typing import Any


# ---------------------------------------------------------------------------
# Per-run authenticity token
# ---------------------------------------------------------------------------
#
# Read the token once at import time, stash in module state, and
# clear the env var so user code that imports ``nora`` later can't
# read it via ``os.environ``. A determined script can still reach
# into ``nora._RUN_TOKEN`` directly — Python module state is
# inherently inspectable — but doing so requires code that obviously
# shows up in the script the researcher reviews. Same posture as the
# R library; see ``docs/direction.md`` "runtime-library contract"
# for the deliberate limits of this measure.

_RUN_TOKEN: str = os.environ.pop("NORA_RUN_TOKEN", "")
if not _RUN_TOKEN:
    raise RuntimeError(
        "NORA_RUN_TOKEN not set. This script must be run through the "
        "Nora executor; direct ``python`` invocation of user code that "
        "emits result payloads isn't supported."
    )

_RESULT_PATH: str = os.environ.get("NORA_RESULT_PATH", "")
if not _RESULT_PATH:
    raise RuntimeError(
        "NORA_RESULT_PATH not set. The Nora executor sets this; if you "
        "see this error in normal usage, the executor wiring is broken."
    )


# ---------------------------------------------------------------------------
# JSON encoder that knows about pandas / numpy types
# ---------------------------------------------------------------------------


class _NoraJSONEncoder(json.JSONEncoder):
    """Handle the numeric / pandas / numpy types stats scripts emit.

    - numpy scalars (np.int64, np.float64, np.bool_) -> Python equivalents
    - numpy arrays / pandas Series / pandas Index -> lists
    - non-finite floats (NaN, Inf) -> JSON null (matches the R library)
    - dataclass instances -> dict (best-effort)
    """

    def default(self, obj: Any) -> Any:  # noqa: D401
        # Lazy imports so the runtime works without numpy/pandas if a
        # script never emits one of their types (rare in practice).
        try:
            import numpy as np
        except ImportError:
            np = None  # type: ignore[assignment]
        try:
            import pandas as pd
        except ImportError:
            pd = None  # type: ignore[assignment]

        if np is not None:
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                f = float(obj)
                return f if math.isfinite(f) else None
            if isinstance(obj, np.bool_):
                return bool(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
        if pd is not None:
            if isinstance(obj, (pd.Series, pd.Index)):
                return obj.tolist()
            if isinstance(obj, pd.DataFrame):
                # DataFrames serialise as a list of row-dicts so the
                # sanitizer (which scans by field name) can still read
                # them. Researchers rarely want this — they should
                # build an explicit payload via ``result(...)`` —
                # but the fallback prevents an inscrutable
                # TypeError in the JSON pass.
                return obj.to_dict(orient="records")
        # Standard floats with non-finite values.
        if isinstance(obj, float) and not math.isfinite(obj):
            return None
        return super().default(obj)


def _to_json(payload: dict[str, Any]) -> str:
    """Serialise ``payload`` with the numpy/pandas-aware encoder.

    ``allow_nan=False`` would catch any float Inf/NaN we forgot to
    map, but we let the encoder's ``default`` handle them and emit
    ``null`` instead so a NaN coefficient (e.g. perfect collinearity)
    serialises as null rather than crashing the script post-fit.
    """
    return json.dumps(payload, cls=_NoraJSONEncoder, allow_nan=True)


# ---------------------------------------------------------------------------
# Core emit
# ---------------------------------------------------------------------------


def _write_result(payload: dict[str, Any]) -> None:
    """Embed the per-run token and write the JSON payload to disk.

    The executor validates the token and strips it before the
    payload reaches the sanitizer. A hand-crafted payload that
    bypasses this function and writes JSON directly to
    ``NORA_RESULT_PATH`` will be rejected (no token).
    """
    if not isinstance(payload, dict):
        raise TypeError(
            f"nora payload must be a dict, got {type(payload).__name__}"
        )
    payload = dict(payload)  # don't mutate caller's dict
    payload["_token"] = _RUN_TOKEN
    with open(_RESULT_PATH, "w", encoding="utf-8") as f:
        f.write(_to_json(payload))


def result(*, type: str, **fields: Any) -> None:  # noqa: A002 — match R API
    """Generic emit. Use one of the ``from_*`` helpers when there's
    a matching one — they pull standard fields out of common Python
    objects (statsmodels results, scipy ttest_result, pandas
    DataFrames) so the researcher doesn't have to assemble the dict
    by hand."""
    payload = {"type": type, **fields}
    _write_result(payload)


# ---------------------------------------------------------------------------
# Helpers — statsmodels / scipy / pandas convenience wrappers
# ---------------------------------------------------------------------------


def from_lm(model: Any, **extra: Any) -> None:
    """Emit a ``linear_regression`` payload from a fitted statsmodels
    result (e.g. ``sm.OLS(y, X).fit()`` or ``smf.ols(...).fit()``).

    Extracts coefficients, SEs, t-statistics, p-values, R^2,
    adjusted R^2, F, F p-value, residual SE, and degrees of
    freedom. The exact attribute names below are the statsmodels
    conventions — sklearn models don't expose them; for sklearn
    use ``result(type="linear_regression", ...)`` directly.

    Also prints ``model.summary()`` to stdout so the researcher
    sees the conventional regression table in the TUI's raw log
    panel. Stdout never reaches the sanitizer, so this is purely
    for the researcher.
    """
    try:
        print(model.summary())
    except Exception:  # noqa: BLE001 — never let printing block the emit
        pass

    coefs = _to_dict(getattr(model, "params"))
    ses = _to_dict(getattr(model, "bse"))
    tvals = _to_dict(getattr(model, "tvalues"))
    pvals = _to_dict(getattr(model, "pvalues"))

    # ``statsmodels`` exposes the design as ``model.model.exog_names``;
    # the response is ``model.model.endog_names``. The first column
    # is "Intercept" for formula-fit models and "const" for
    # ``add_constant(X)`` setups — we keep whichever name was used.
    inner = getattr(model, "model", None)
    response = getattr(inner, "endog_names", None) if inner is not None else None
    exog_names = list(getattr(inner, "exog_names", []) or []) if inner else []
    # predictor_variables = exog minus the intercept (sanitizer wants
    # the regressors of interest, not the intercept).
    predictors = [n for n in exog_names if n not in ("const", "Intercept")]

    n = _safe_int(getattr(model, "nobs", None))
    df_resid = _safe_int(getattr(model, "df_resid", None))
    r2 = _safe_float(getattr(model, "rsquared", None))
    adj_r2 = _safe_float(getattr(model, "rsquared_adj", None))
    f = _safe_float(getattr(model, "fvalue", None))
    f_p = _safe_float(getattr(model, "f_pvalue", None))
    sigma = _safe_float(
        # statsmodels names the residual SE differently across model
        # families; check both.
        getattr(model, "scale", None)
    )
    if sigma is not None:
        sigma = math.sqrt(sigma) if sigma >= 0 else None

    fields: dict[str, Any] = {
        "n": n,
        "response_variable": response,
        "predictor_variables": predictors,
        "coefficients": coefs,
        "standard_errors": ses,
        "t_statistics": tvals,
        "p_values": pvals,
        "r_squared": r2,
        "adj_r_squared": adj_r2,
        "f_statistic": f,
        "f_p_value": f_p,
        "degrees_of_freedom": df_resid,
        "residual_std_error": sigma,
    }
    # Aggregate diagnostics. These derive from the design matrix and
    # residual sums — pure aggregates, no per-observation leak. Add
    # only when computable; if numpy is missing or the model object
    # doesn't expose its design, omit silently rather than failing
    # the whole emit.
    vif = _compute_vif(model, predictors)
    if vif:
        fields["vif"] = vif
    cond = _compute_condition_number(model)
    if cond is not None:
        fields["condition_number"] = cond
    vcov = _compute_vcov(model)
    if vcov:
        fields["vcov"] = vcov
    fields.update(extra)
    result(type="linear_regression", **fields)


def _compute_vcov(model: Any) -> dict[str, dict[str, float]] | None:
    """Variance-covariance matrix of the coefficient estimates.

    statsmodels exposes ``model.cov_params()`` returning a labelled
    DataFrame whose row + column index are the coefficient names.
    Diagonals are the squared SEs (so ``standard_errors[name]`` =
    ``sqrt(vcov[name][name])``); off-diagonals carry the
    coefficient covariances that drive Wald tests, joint
    significance, and linear-combination CIs the model can compute
    on its own.

    Pure aggregate from sigma^2 * (X'X)^-1 — no per-observation
    information. Returns None when the model object doesn't expose
    a parameter covariance (sklearn-shaped, custom estimators,
    etc.); the caller drops the field rather than emitting null.
    """
    fn = getattr(model, "cov_params", None)
    if fn is None or not callable(fn):
        return None
    try:
        cov = fn()
    except Exception:  # noqa: BLE001
        return None
    # statsmodels returns a pandas DataFrame for formula fits and a
    # numpy array for raw OLS(y, X). Handle both.
    to_dict = getattr(cov, "to_dict", None)
    if callable(to_dict):
        try:
            raw = to_dict()
        except Exception:  # noqa: BLE001
            return None
        out: dict[str, dict[str, float]] = {}
        for row_key, row_dict in raw.items():
            if not isinstance(row_dict, dict):
                continue
            inner: dict[str, float] = {}
            for col_key, val in row_dict.items():
                try:
                    fval = float(val)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(fval):
                    inner[str(col_key)] = fval
            if inner:
                out[str(row_key)] = inner
        return out or None
    # numpy array path: pair with exog_names from the inner model.
    try:
        import numpy as np
    except Exception:  # noqa: BLE001
        return None
    arr = np.asarray(cov, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        return None
    inner_model = getattr(model, "model", None)
    names = list(
        getattr(inner_model, "exog_names", []) or []
    ) if inner_model is not None else []
    if len(names) != arr.shape[0]:
        return None
    out_arr: dict[str, dict[str, float]] = {}
    for i, row_name in enumerate(names):
        inner: dict[str, float] = {}
        for j, col_name in enumerate(names):
            v = float(arr[i, j])
            if math.isfinite(v):
                inner[col_name] = v
        if inner:
            out_arr[row_name] = inner
    return out_arr or None


def _compute_vif(model: Any, predictors: list[str]) -> dict[str, float] | None:
    """Variance inflation factor per predictor.

    For each predictor x_i, fit an auxiliary OLS of x_i on the OTHER
    predictors (intercept handled by the design matrix). Return
    1 / (1 - R^2_aux). Pure aggregate over the design columns; no
    per-row data crosses back.

    Skipped silently if numpy isn't installed, the model lacks an
    accessible design matrix, or any predictor is perfectly collinear
    with the rest (R^2_aux >= 1) — the caller treats absence as
    "diagnostic unavailable" rather than "no collinearity".
    """
    try:
        import numpy as np
    except Exception:  # noqa: BLE001 — numpy missing → quietly omit
        return None
    inner = getattr(model, "model", None)
    X = getattr(inner, "exog", None) if inner is not None else None
    if X is None:
        return None
    try:
        X = np.asarray(X, dtype=float)
    except Exception:  # noqa: BLE001
        return None
    if X.ndim != 2 or X.shape[0] < 2 or X.shape[1] < 2:
        return None
    exog_names = list(getattr(inner, "exog_names", []) or [])
    if len(exog_names) != X.shape[1]:
        return None

    out: dict[str, float] = {}
    for i, name in enumerate(exog_names):
        if name in ("const", "Intercept", "(Intercept)"):
            continue
        if predictors and name not in predictors:
            # Only emit VIF for declared predictors so the sanitizer's
            # cross-field key validation accepts the result.
            continue
        xi = X[:, i]
        X_others = np.delete(X, i, axis=1)
        try:
            beta, *_ = np.linalg.lstsq(X_others, xi, rcond=None)
            xi_hat = X_others @ beta
            ss_res = float(np.sum((xi - xi_hat) ** 2))
            ss_tot = float(np.sum((xi - np.mean(xi)) ** 2))
        except Exception:  # noqa: BLE001
            continue
        if ss_tot <= 0 or ss_res < 0:
            continue
        r2_aux = 1.0 - ss_res / ss_tot
        if r2_aux >= 1.0 or r2_aux < 0.0:
            continue
        out[name] = 1.0 / (1.0 - r2_aux)
    return out or None


def _compute_condition_number(model: Any) -> float | None:
    """``kappa(X)`` — ratio of the largest to smallest singular
    value of the design matrix. High values flag near-collinearity
    that VIF can miss when it's spread across many predictors.

    Returns ``None`` if numpy is missing or the design isn't
    reachable; the caller drops the field rather than emitting a
    confusing ``null``.
    """
    try:
        import numpy as np
    except Exception:  # noqa: BLE001
        return None
    inner = getattr(model, "model", None)
    X = getattr(inner, "exog", None) if inner is not None else None
    if X is None:
        return None
    try:
        return float(np.linalg.cond(np.asarray(X, dtype=float)))
    except Exception:  # noqa: BLE001
        return None


def from_t_test(res: Any, *, n1: int, n2: int | None = None,
                mean1: float | None = None, mean2: float | None = None,
                test_type: str | None = None, **extra: Any) -> None:
    """Emit a ``t_test`` payload from a SciPy ``ttest_ind`` /
    ``ttest_rel`` / ``ttest_1samp`` result.

    SciPy's result object carries ``statistic``, ``pvalue``, and
    ``df`` (on newer versions); it does NOT carry sample sizes or
    group means, so callers must pass them explicitly. The
    docstring lists this requirement loudly because forgetting it
    is the #1 way scripts produce a payload the sanitizer rejects.

    ``test_type`` defaults to ``"one_sample"`` when only ``n1`` is
    given, ``"two_sample"`` when ``n2`` is set. Pass ``"welch"`` /
    ``"paired"`` explicitly when the underlying scipy call used those
    variants — scipy's result object doesn't carry that info itself.
    The field name is ``test_type`` (not ``subtype``) to match the R
    and Stata helpers and the sanitizer's ``_TTEST_REQUIRED`` set —
    drift here means the sanitizer drops every emit.
    """
    try:
        print(repr(res))
    except Exception:  # noqa: BLE001
        pass

    statistic = _safe_float(getattr(res, "statistic", None))
    pvalue = _safe_float(getattr(res, "pvalue", None))
    df = _safe_float(getattr(res, "df", None))

    if test_type is None:
        test_type = "one_sample" if n2 is None else "two_sample"

    fields: dict[str, Any] = {
        "test_type": test_type,
        "t_statistic": statistic,
        "p_value": pvalue,
        "degrees_of_freedom": df,
        "n1": int(n1),
        "mean1": mean1,
    }
    if n2 is not None:
        fields["n2"] = int(n2)
    if mean2 is not None:
        fields["mean2"] = mean2
    fields.update(extra)
    result(type="t_test", **fields)


def from_summarize(variable: str, *, n: int, mean: float, sd: float,
                   missing_count: int = 0,
                   min_value: float | None = None,
                   max_value: float | None = None,
                   **extra: Any) -> None:
    """Emit a ``descriptive`` payload for a single numeric variable.
    Mirrors ``nora$from_summarize`` in the R library.

    ``min_value`` / ``max_value`` are passed through ONLY when the
    variable is on the dataset's ``non_disclosive_variables`` opt-in
    list in ``.nora/policy.json``. The sanitizer drops them silently
    for any variable not on that list — same posture as residuals /
    fitted values, gated by an explicit per-variable researcher
    judgment instead of a blanket ban. Pass them when you have them;
    they cost nothing and surface automatically if the researcher
    has opted the variable in.
    """
    fields: dict[str, Any] = {
        "variable": variable,
        "n": int(n),
        "mean": _safe_float(mean),
        "sd": _safe_float(sd),
        "missing_count": int(missing_count),
    }
    if min_value is not None:
        fields["min_value"] = _safe_float(min_value)
    if max_value is not None:
        fields["max_value"] = _safe_float(max_value)
    fields.update(extra)
    result(type="descriptive", **fields)


def from_table(variable: str, counts: Any, *, n: int | None = None,
               missing_count: int = 0, **extra: Any) -> None:
    """Emit a 1-D ``frequency_table`` payload.

    ``counts`` accepts a dict ``{"level": count}``, a pandas Series,
    or anything the encoder can normalise to that shape.
    """
    if hasattr(counts, "to_dict"):
        counts_dict = counts.to_dict()
    else:
        counts_dict = dict(counts)
    if n is None:
        n = sum(int(v) for v in counts_dict.values())
    fields = {
        "variable": variable,
        "counts": {str(k): int(v) for k, v in counts_dict.items()},
        "n": int(n),
        "missing_count": int(missing_count),
    }
    fields.update(extra)
    result(type="frequency_table", **fields)


def from_crosstab(table: Any, *, row_variable: str | None = None,
                  col_variable: str | None = None,
                  missing_count: int = 0, **extra: Any) -> None:
    """Emit a 2-D ``crosstab`` payload from a pandas DataFrame
    produced by ``pd.crosstab(...)`` (or any 2-D table-like)."""
    try:
        import pandas as pd
    except ImportError:
        pd = None  # type: ignore[assignment]

    if pd is not None and isinstance(table, pd.DataFrame):
        counts: dict[str, dict[str, int]] = {}
        for row_label, row in table.iterrows():
            counts[str(row_label)] = {
                str(col): int(row[col]) for col in table.columns
            }
        if row_variable is None:
            row_variable = str(table.index.name or "row")
        if col_variable is None:
            col_variable = str(table.columns.name or "column")
    else:
        # Caller passed a pre-built nested dict.
        counts = {
            str(rk): {str(ck): int(cv) for ck, cv in (rv or {}).items()}
            for rk, rv in dict(table).items()
        }
        row_variable = row_variable or "row"
        col_variable = col_variable or "column"

    fields = {
        "row_variable": row_variable,
        "col_variable": col_variable,
        "counts": counts,
        "missing_count": int(missing_count),
    }
    fields.update(extra)
    result(type="crosstab", **fields)


def from_magnitude_table(df: Any, group_var: str, value_var: str, *,
                         aggregation: str = "sum", **extra: Any) -> None:
    """Emit a ``magnitude_table`` payload (sum or mean of a numeric
    by group). Pre-aggregates here; the sanitizer applies the
    (1, 85%)-dominance rule on the per-cell ``max_share``.

    Field-name and dominance-metric contract MUST match the R / Stata
    helpers and the sanitizer's ``_MAGTAB_REQUIRED`` set:

    - top-level key is ``row_variable`` (not ``group_variable``).
    - per-cell ``max_share`` is the share-in-[0,1] dominance metric
      ``max(abs(values)) / sum(abs(values))`` — NOT the raw top
      absolute value. The sanitizer suppresses cells whose
      ``max_share`` exceeds the dominance threshold (default 0.85)
      and strips ``max_share`` from the visible payload before it
      reaches the model.

    Empty groups (no non-missing observations) emit ``{value: 0,
    n: 0, max_share: 0}`` so the sanitizer suppresses them on the
    n side rather than failing required-field validation. Same shape
    R uses.
    """
    if aggregation not in ("sum", "mean"):
        raise ValueError(
            f"aggregation must be 'sum' or 'mean', got {aggregation!r}"
        )
    grouped = df.groupby(group_var)[value_var]
    cells: dict[str, dict[str, Any]] = {}
    for key, group in grouped:
        clean = group.dropna()
        n_cell = int(len(clean))
        if n_cell == 0:
            cells[str(key)] = {"value": 0.0, "n": 0, "max_share": 0.0}
            continue
        agg_value = float(clean.sum()) if aggregation == "sum" else float(clean.mean())
        # Dominance metric: top contributor's share of the absolute
        # total. All-zero groups have undefined share — emit 0 (no
        # contributor dominates because there's no magnitude). Same
        # guard the R helper applies.
        abs_vals = clean.abs()
        total_abs = float(abs_vals.sum())
        max_share = (
            float(abs_vals.max()) / total_abs if total_abs > 0 else 0.0
        )
        cells[str(key)] = {
            "value": agg_value,
            "n": n_cell,
            "max_share": max_share,
        }
    fields = {
        "row_variable": group_var,
        "value_variable": value_var,
        "aggregation": aggregation,
        "cells": cells,
    }
    fields.update(extra)
    result(type="magnitude_table", **fields)


def from_correlation(
    df: Any,
    *,
    variables: list[str] | None = None,
    method: str = "pearson",
    **extra: Any,
) -> None:
    """Emit a ``correlation_matrix`` payload from a pandas DataFrame.

    By default correlates every numeric column; pass ``variables`` to
    restrict to a named subset. ``method`` is one of ``'pearson'``,
    ``'spearman'``, ``'kendall'`` — anything else is rejected by the
    sanitizer.

    Sample size N is the number of *complete* rows over the chosen
    variables (pandas ``.dropna()`` semantics). Sub-threshold N is
    rejected by the sanitizer with a clear reason — at very low N a
    correlation of 0.99 between two columns is just "the three points
    are collinear" and could imply individual coordinates.

    Also prints the correlation matrix to stdout so the researcher
    sees the conventional view in the raw log panel.
    """
    if method not in ("pearson", "spearman", "kendall"):
        raise ValueError(
            f"method must be 'pearson' / 'spearman' / 'kendall', "
            f"got {method!r}"
        )
    # Pick columns. Default to numeric columns if no list given;
    # respect the order the caller passed when they did.
    if variables is None:
        # Lazy: keep numeric + boolean (booleans correlate fine).
        try:
            import numpy as _np  # noqa: F401
        except ImportError:
            pass
        variables = [
            c for c in df.columns
            if str(df[c].dtype) not in ("object", "string", "category")
        ]
    if not variables:
        raise ValueError(
            "from_correlation: no numeric columns found and no "
            "``variables`` provided"
        )
    sub = df[variables]
    # Correlation matrix on rows where ALL chosen variables are
    # observed. Emitting N as `len(complete_rows)` is the honest
    # number — pairwise N-by-pair would be deceptive (each off-
    # diagonal would be a different sample).
    complete = sub.dropna()
    n = int(len(complete))
    missing_count = int(len(df) - n)
    corr = complete.corr(method=method)
    try:
        print(corr)
    except Exception:  # noqa: BLE001 — never let print block emit
        pass
    correlations: dict[str, dict[str, float]] = {}
    for row_var in variables:
        row_dict: dict[str, float] = {}
        for col_var in variables:
            try:
                v = float(corr.at[row_var, col_var])
                if math.isfinite(v):
                    row_dict[col_var] = v
            except Exception:  # noqa: BLE001
                continue
        if row_dict:
            correlations[row_var] = row_dict
    fields: dict[str, Any] = {
        "n": n,
        "variables": list(variables),
        "method": method,
        "correlations": correlations,
        "missing_count": missing_count,
    }
    fields.update(extra)
    result(type="correlation_matrix", **fields)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_dict(thing: Any) -> dict[str, Any]:
    """Normalise a pandas Series / dict-like to a plain ``{name: value}``
    dict the encoder can serialise without further coercion."""
    if hasattr(thing, "to_dict"):
        return {str(k): v for k, v in thing.to_dict().items()}
    return {str(k): v for k, v in dict(thing).items()}


def _safe_float(x: Any) -> float | None:
    """Coerce to float, returning None for None/NaN/Inf or
    non-coercible inputs. Keeps the payload JSON-clean."""
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _safe_int(x: Any) -> int | None:
    """Coerce to int, returning None for non-coercible / NaN inputs."""
    if x is None:
        return None
    try:
        f = float(x)
        if not math.isfinite(f):
            return None
        return int(f)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Plot helpers — model-output visualizations only
# ---------------------------------------------------------------------------
#
# Plots produced via these helpers are surfaced to the model on the
# next turn as image attachments. Raw-data plots (a histogram of an
# observed column, a scatter of all rows) are NOT covered on
# purpose — they would expose the data itself, which is the privacy
# line Nora is built to keep.
#
# Allowlist: only files written via these helpers (and registered
# in the manifest) are visible. ``plt.savefig(...)`` outside the
# helpers does NOT cross to the model — the file lands in the run
# dir for the researcher's eyes only.
#
# Mechanism mirrors the R library: write a PNG into
# ``<run_dir>/_nora_plots/`` and append a JSONL entry to
# ``manifest.jsonl``. The bridge reads only the manifest.


def _plots_dir() -> Any:
    """Return the ``_nora_plots`` directory beside the result file,
    creating it on first use. None when ``NORA_RESULT_PATH`` isn't
    set (caller didn't go through the executor)."""
    if not _RESULT_PATH:
        return None
    from pathlib import Path
    d = Path(_RESULT_PATH).parent / "_nora_plots"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _append_plot_manifest(file: str, kind: str, label: str | None) -> None:
    d = _plots_dir()
    if d is None:
        return
    entry: dict[str, Any] = {"file": file, "kind": kind}
    if label:
        entry["label"] = label
    try:
        with (d / "manifest.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        pass


def _append_plot_helper_error(helper: str, exc: BaseException) -> None:
    """Record a structured plot-helper failure so ``submit_script``
    can surface it in the tool result the MODEL receives. Without
    this, helper failures (matplotlib missing, etc.) only land in
    stderr, and the model says "thumbnail should be visible above"
    while the researcher sees nothing — the loop the user reported.

    The runner reads ``_nora_plots/helper_errors.jsonl`` after the
    run and includes a summary in the structured tool result so
    the model can react instead of guessing.
    """
    d = _plots_dir()
    if d is None:
        return
    error_kind = type(exc).__name__
    message = str(exc)
    fix: str | None = None
    lower = message.lower()
    if "matplotlib" in lower or "no module named 'matplotlib'" in lower:
        fix = "pip install matplotlib"
    elif "no module named 'scipy'" in lower:
        fix = "pip install scipy"
    elif "no module named 'statsmodels'" in lower:
        fix = "pip install statsmodels"
    entry: dict[str, Any] = {
        "helper": helper, "error": error_kind, "message": message,
    }
    if fix:
        entry["fix"] = fix
    try:
        with (d / "helper_errors.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        pass


def plot_residuals(fitted: Any, label: str | None = None) -> None:
    """Write the four standard residual diagnostic panels for a
    statsmodels fit and register them with the plot manifest.

    Errors inside this helper print to stderr but never raise — a
    broken plot helper must not break the analysis script.
    """
    try:
        d = _plots_dir()
        if d is None:
            return
        # Force a non-interactive backend before importing pyplot:
        # the executor runs scripts headless and any default GUI
        # backend would either crash (no display) or pop a window
        # the researcher didn't ask for.
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        resid = getattr(fitted, "resid", None)
        fitted_vals = getattr(fitted, "fittedvalues", None)
        if resid is None or fitted_vals is None:
            sys.stderr.write(
                "nora.plot_residuals: fitted object has no .resid / "
                ".fittedvalues; skipping\n"
            )
            return

        try:
            import numpy as _np
            resid_arr = _np.asarray(resid, dtype=float)
            fitted_arr = _np.asarray(fitted_vals, dtype=float)
        except ImportError:
            sys.stderr.write("nora.plot_residuals: numpy missing\n")
            return

        fig, axes = plt.subplots(2, 2, figsize=(9, 7))
        # Residuals vs fitted
        axes[0, 0].scatter(fitted_arr, resid_arr, alpha=0.5, s=12)
        axes[0, 0].axhline(0, color="gray", lw=0.8)
        axes[0, 0].set_xlabel("Fitted values")
        axes[0, 0].set_ylabel("Residuals")
        axes[0, 0].set_title("Residuals vs Fitted")
        # Normal Q-Q
        try:
            from scipy import stats as _stats
            _stats.probplot(resid_arr, dist="norm", plot=axes[0, 1])
            axes[0, 1].set_title("Normal Q-Q")
        except ImportError:
            axes[0, 1].text(0.5, 0.5, "scipy not installed",
                            ha="center", va="center")
            axes[0, 1].set_title("Normal Q-Q")
        # Scale-Location
        sd = float(resid_arr.std() or 1.0)
        std_resid = (resid_arr - resid_arr.mean()) / sd
        sqrt_abs = (abs(std_resid)) ** 0.5
        axes[1, 0].scatter(fitted_arr, sqrt_abs, alpha=0.5, s=12)
        axes[1, 0].set_xlabel("Fitted values")
        axes[1, 0].set_ylabel(r"$\sqrt{|standardized\ resid|}$")
        axes[1, 0].set_title("Scale-Location")
        # Residual distribution
        axes[1, 1].hist(resid_arr, bins=20)
        axes[1, 1].set_xlabel("Residual")
        axes[1, 1].set_ylabel("Count")
        axes[1, 1].set_title("Residual distribution")
        fig.tight_layout()

        fname = "residuals.png"
        fig.savefig(d / fname, dpi=110)
        plt.close(fig)
        _append_plot_manifest(
            fname, "residuals",
            label or "Residual diagnostics",
        )
    except Exception as e:  # noqa: BLE001 — never let plotting fail the script
        sys.stderr.write(f"nora.plot_residuals failed: {e}\n")
        _append_plot_helper_error("plot_residuals", e)


def plot_coefficients(fitted: Any, label: str | None = None) -> None:
    """Forest plot of coefficient point estimates with 95% CIs.

    Operates ONLY on the fit's ``params`` and ``conf_int()`` —
    pure functions of model output, never the raw data. The
    helper is the gate; there is no escape-hatch path that
    accepts an arbitrary file (that would let a histogram of
    raw rows pose as a coefficient plot — bypassing the
    privacy line the entire system rests on).

    Errors inside the helper print to stderr but never raise —
    a broken plot helper must not break the analysis around it.
    """
    try:
        d = _plots_dir()
        if d is None:
            return
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as _np

        params = getattr(fitted, "params", None)
        if params is None:
            sys.stderr.write(
                "nora.plot_coefficients: fitted object has no "
                ".params; need a statsmodels-style fit\n"
            )
            return
        try:
            ci = fitted.conf_int(alpha=0.05)
        except Exception as e:  # noqa: BLE001
            sys.stderr.write(f"nora.plot_coefficients: conf_int failed: {e}\n")
            return

        # Drop intercept by default — researchers almost never want
        # it on the same scale as predictors. (If they do, they can
        # call this on a model fit without an intercept term.)
        names = list(getattr(params, "index", range(len(params))))
        ests = _np.asarray(params, dtype=float)
        try:
            import pandas as _pd
            if isinstance(ci, _pd.DataFrame):
                lows = ci.iloc[:, 0].to_numpy(dtype=float)
                highs = ci.iloc[:, 1].to_numpy(dtype=float)
            else:
                ci_arr = _np.asarray(ci, dtype=float)
                lows, highs = ci_arr[:, 0], ci_arr[:, 1]
        except ImportError:
            ci_arr = _np.asarray(ci, dtype=float)
            lows, highs = ci_arr[:, 0], ci_arr[:, 1]

        keep = [
            i for i, n in enumerate(names)
            if str(n).lower() not in ("intercept", "const", "_cons")
        ]
        if not keep:
            sys.stderr.write(
                "nora.plot_coefficients: nothing to plot after "
                "dropping intercept term\n"
            )
            return
        names = [str(names[i]) for i in keep]
        ests = ests[keep]
        lows = lows[keep]
        highs = highs[keep]

        fig, ax = plt.subplots(figsize=(8, max(2.5, 0.4 * len(names) + 1)))
        y = _np.arange(len(names))
        ax.hlines(y, lows, highs, lw=2, color="#4C78A8")
        ax.scatter(ests, y, s=60, color="#4C78A8", zorder=3)
        ax.axvline(0, color="gray", lw=1, ls="--")
        ax.set_yticks(y)
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_xlabel("Coefficient (95% CI)")
        ax.set_title("Coefficients")
        fig.tight_layout()

        fname = "coefficients.png"
        fig.savefig(d / fname, dpi=110)
        plt.close(fig)
        _append_plot_manifest(
            fname, "coefficients",
            label or "Coefficient estimates with 95% CIs",
        )
    except Exception as e:  # noqa: BLE001
        sys.stderr.write(f"nora.plot_coefficients failed: {e}\n")
        _append_plot_helper_error("plot_coefficients", e)


def plot_estimate_comparison(
    models: Any,
    coef: str,
    label: str | None = None,
) -> None:
    """Forest plot comparing one coefficient across multiple fits.

    ``models`` is a dict mapping label → fitted model (the keys
    become y-axis labels). Each fit must expose ``params`` and
    ``cov_params()`` (statsmodels) or ``params``/``bse``. ``coef``
    is the coefficient name to extract from each.

    Use case: "female gap before/after controls" — two regressions,
    one plot, no language switching to compose them. Same posture
    as the other ``plot_*`` helpers: produces a model-output plot
    only (point estimates + CIs from each fit's covariance), never
    raw rows.
    """
    try:
        d = _plots_dir()
        if d is None:
            return
        if not isinstance(models, dict) or len(models) < 2:
            sys.stderr.write(
                "nora.plot_estimate_comparison: `models` must be a "
                "dict of at least 2 fits keyed by label\n"
            )
            return
        if not isinstance(coef, str) or not coef:
            sys.stderr.write(
                "nora.plot_estimate_comparison: `coef` must be a "
                "coefficient name string\n"
            )
            return
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as _np

        labels: list[str] = []
        ests: list[float] = []
        ses: list[float] = []
        for nm, fit in models.items():
            params = getattr(fit, "params", None)
            if params is None:
                sys.stderr.write(
                    f"nora.plot_estimate_comparison: model {nm!r} "
                    f"has no .params; need a statsmodels-style fit\n"
                )
                return
            try:
                idx = list(params.index)
            except AttributeError:
                idx = [str(i) for i in range(len(params))]
            if coef not in idx:
                sys.stderr.write(
                    f"nora.plot_estimate_comparison: coef "
                    f"{coef!r} not in model {nm!r}\n"
                )
                return
            est = float(params[coef])
            # SE: prefer .bse[coef]; fall back to sqrt of cov_params
            # diagonal. statsmodels exposes both.
            bse = getattr(fit, "bse", None)
            if bse is not None and coef in list(bse.index):
                se = float(bse[coef])
            else:
                cov = fit.cov_params()
                se = float(_np.sqrt(cov.loc[coef, coef]))
            labels.append(str(nm))
            ests.append(est)
            ses.append(se)

        ests_arr = _np.asarray(ests, dtype=float)
        ses_arr = _np.asarray(ses, dtype=float)
        lows = ests_arr - 1.96 * ses_arr
        highs = ests_arr + 1.96 * ses_arr

        n = len(labels)
        fig, ax = plt.subplots(figsize=(8.5, max(2.5, 0.5 * n + 1.5)))
        y = _np.arange(n)
        ax.hlines(y, lows, highs, lw=2, color="#4C78A8")
        ax.scatter(ests_arr, y, s=70, color="#4C78A8", zorder=3)
        ax.axvline(0, color="gray", lw=1, ls="--")
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel(f"{coef} (95% CI)")
        ax.set_title(f"Estimate comparison: {coef}")
        fig.tight_layout()

        fname = "estimate_comparison.png"
        fig.savefig(d / fname, dpi=110)
        plt.close(fig)
        _append_plot_manifest(
            fname, "coefficients",
            label or f"Estimate comparison: {coef}",
        )
    except Exception as e:  # noqa: BLE001
        sys.stderr.write(f"nora.plot_estimate_comparison failed: {e}\n")
        _append_plot_helper_error("plot_estimate_comparison", e)


def plot_interaction(
    fitted: Any,
    var: str,
    data: Any | None = None,
    label: str | None = None,
    xlab: str | None = None,
    ylab: str | None = None,
    title: str | None = None,
) -> None:
    """Predicted-response curve across one predictor with the others
    held at their means (numeric) or first level (categorical).
    Bands are 1.96 * SE of the predicted mean.

    ``fitted`` is a statsmodels results object. ``data`` is the
    DataFrame the model was fit on (statsmodels doesn't reliably
    expose this back through the results object once formulae are
    involved). Optional ``xlab`` / ``ylab`` / ``title`` override
    defaults that fall back to the variable name and a generic
    "Predicted response" label.

    Falls through quietly with a stderr note when shape can't be
    derived.
    """
    try:
        d = _plots_dir()
        if d is None:
            return
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as _np
        import pandas as _pd

        if data is None:
            data = getattr(fitted.model, "data", None)
            data = getattr(data, "frame", None) if data is not None else None
        if data is None or var not in getattr(data, "columns", []):
            sys.stderr.write(
                f"nora.plot_interaction: pass data=... that contains "
                f"column {var!r}; couldn't derive it from the fit\n"
            )
            return

        # Build the prediction grid and the held-at-means template.
        col = data[var]
        is_numeric = _pd.api.types.is_numeric_dtype(col)
        if is_numeric:
            grid = _np.linspace(float(col.min()), float(col.max()), 100)
        else:
            grid = list(col.dropna().unique())

        template = {}
        for c in data.columns:
            v = data[c]
            if _pd.api.types.is_numeric_dtype(v):
                template[c] = float(v.mean())
            else:
                template[c] = v.dropna().iloc[0] if not v.dropna().empty else None
        new_rows = []
        for g in grid:
            row = dict(template)
            row[var] = g
            new_rows.append(row)
        new = _pd.DataFrame(new_rows)

        # Statsmodels' get_prediction returns a PredictionResults
        # with .summary_frame() including 'mean' and 'mean_ci_lower/upper'.
        try:
            pred = fitted.get_prediction(new)
            sf = pred.summary_frame(alpha=0.05)
            mean = sf["mean"].to_numpy()
            lo = sf["mean_ci_lower"].to_numpy()
            hi = sf["mean_ci_upper"].to_numpy()
        except Exception:
            # Fallback: predict() alone (no CI available)
            mean = _np.asarray(fitted.predict(new))
            lo = mean
            hi = mean

        xtitle = xlab if xlab is not None else var
        ytitle = ylab if ylab is not None else "Predicted response"
        ptitle = title if title is not None else f"Predicted response by {var}"

        fig, ax = plt.subplots(figsize=(9, 5))
        if is_numeric:
            # Filled CI ribbon under a colored line — much more
            # readable than the prior dashed-line whiskers, which
            # the user called out as a "really shitty" rendering.
            ax.fill_between(grid, lo, hi, color="#4C78A8", alpha=0.20)
            ax.plot(grid, mean, lw=2, color="#1F4E79")
        else:
            xs = _np.arange(len(grid))
            ax.bar(xs, mean, yerr=[mean - lo, hi - mean],
                   color="#4C78A8", edgecolor="#1F4E79",
                   capsize=4)
            ax.set_xticks(xs)
            ax.set_xticklabels([str(g) for g in grid])
        ax.set_xlabel(xtitle)
        ax.set_ylabel(ytitle)
        ax.set_title(ptitle, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()

        # ``var`` can in principle contain weird chars; sanitize to
        # something filesystem-safe but still informative.
        safe_var = "".join(c if c.isalnum() or c in "-_" else "_" for c in var)
        fname = f"interaction_{safe_var}.png"
        fig.savefig(d / fname, dpi=110)
        plt.close(fig)
        _append_plot_manifest(
            fname, "interaction",
            label or f"Predicted response by {var}",
        )
    except Exception as e:  # noqa: BLE001
        sys.stderr.write(f"nora.plot_interaction failed: {e}\n")
        _append_plot_helper_error("plot_interaction", e)
