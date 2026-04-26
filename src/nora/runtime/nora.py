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
    fields.update(extra)
    result(type="linear_regression", **fields)


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
                   missing_count: int = 0, **extra: Any) -> None:
    """Emit a ``descriptive`` payload for a single numeric variable.
    Mirrors ``nora$from_summarize`` in the R library."""
    fields = {
        "variable": variable,
        "n": int(n),
        "mean": _safe_float(mean),
        "sd": _safe_float(sd),
        "missing_count": int(missing_count),
    }
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
