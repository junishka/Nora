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


def _scrub_non_finite(obj: Any) -> Any:
    """Replace non-finite *plain* Python floats with ``None``.

    The encoder's ``default()`` only fires for objects ``json.dumps``
    doesn't know how to serialise natively; ``float('nan')`` /
    ``float('inf')`` ARE natively serialisable, so the encoder's
    non-finite branch never sees them and ``allow_nan=True`` would
    emit RFC-8259-invalid ``NaN`` / ``Infinity`` tokens. Walk the
    payload first and substitute ``None`` so the wire stays
    strict-JSON, matching the R library's "non-finite -> null"
    contract.
    """
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj
    if isinstance(obj, dict):
        return {k: _scrub_non_finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_scrub_non_finite(v) for v in obj]
    return obj


def _to_json(payload: dict[str, Any]) -> str:
    """Serialise ``payload`` with the numpy/pandas-aware encoder.

    ``allow_nan=False`` ensures any float Inf/NaN the pre-pass missed
    raises a ``ValueError`` rather than silently emitting RFC-8259-
    invalid tokens. Plain Python non-finite floats are scrubbed to
    ``None`` up front so a NaN coefficient (e.g. perfect collinearity)
    serialises as null rather than crashing the script post-fit;
    numpy non-finite floats still go through the encoder's
    ``default()``.
    """
    return json.dumps(
        _scrub_non_finite(payload), cls=_NoraJSONEncoder, allow_nan=False,
    )


# ---------------------------------------------------------------------------
# Core emit
# ---------------------------------------------------------------------------


def _write_result(payload: dict[str, Any]) -> None:
    """Embed the per-run token and append the JSON payload to disk.

    The result file is JSONL: one object per line. Each helper call
    appends its own line, so a script that calls multiple helpers
    surfaces every payload back to the executor. The executor
    validates the token on each line and strips it before the
    payload reaches the sanitizer. A hand-crafted line that
    bypasses this function will be rejected (no token).
    """
    if not isinstance(payload, dict):
        raise TypeError(
            f"nora payload must be a dict, got {type(payload).__name__}"
        )
    payload = dict(payload)  # don't mutate caller's dict
    payload["_token"] = _RUN_TOKEN
    with open(_RESULT_PATH, "a", encoding="utf-8") as f:
        f.write(_to_json(payload))
        f.write("\n")


def result(*, type: str, **fields: Any) -> None:  # noqa: A002 — match R API
    """Generic emit. Use one of the ``from_*`` helpers when there's
    a matching one — they pull standard fields out of common Python
    objects (statsmodels results, scipy ttest_result, pandas
    DataFrames) so the researcher doesn't have to assemble the dict
    by hand.

    The ``_via_helper`` field is reserved as a sanitizer-side
    helper-provenance marker (typed helpers stamp it after computing
    disclosure metrics from raw data). Strip it here so a script
    can't bypass the typed helper for ``magnitude_table`` etc. by
    passing a forged marker through ``result()``.
    """
    fields = {k: v for k, v in fields.items() if k != "_via_helper"}
    payload = {"type": type, **fields}
    _write_result(payload)


# ---------------------------------------------------------------------------
# Helpers — statsmodels / scipy / pandas convenience wrappers
# ---------------------------------------------------------------------------


def from_lm(model: Any, **extra: Any) -> None:
    """Emit a ``linear_regression`` payload from a fitted statsmodels
    result.

    Supports the regression-shape estimators statsmodels exposes:
    ``OLS``, ``GLM`` (Binomial / Poisson / Gaussian / Gamma / …),
    ``Logit``, ``Probit``, ``Poisson``, ``NegativeBinomial``,
    ``PHReg`` (Cox proportional hazards), and ``IV2SLS``. Each class
    needs a slightly different attribute mix:

      * OLS exposes ``rsquared`` / ``rsquared_adj`` / ``fvalue`` /
        ``f_pvalue`` / ``scale``; ``llf`` / ``aic`` / ``bic`` are
        also present.
      * GLM-family results (Logit, Probit, Poisson, NegBin, GLM)
        expose ``prsquared`` (McFadden's R²), ``llf``, ``aic``,
        ``bic``; ``rsquared`` is **not present** — the old helper
        would emit it as ``null`` and ship every GLM payload missing
        all fit metrics.
      * PHReg exposes ``llf`` and the censoring status via
        ``model.status``; ``aic`` / ``bic`` are not on the result
        wrapper (omit cleanly). ``nobs`` is also **not present** on
        ``PHRegResults`` — derive ``n`` from ``model.endog`` shape
        so the sanitizer's required-field check accepts the payload.

    Sklearn models don't expose any of these conventions; for
    sklearn use ``result(type="linear_regression", ...)`` directly.

    Also prints ``model.summary()`` to stdout so the researcher
    sees the conventional regression table in the TUI's raw log
    panel. Stdout never reaches the sanitizer, so this is purely
    for the researcher.
    """
    try:
        print(model.summary())
    except Exception:  # noqa: BLE001 — never let printing block the emit
        pass

    # Per-class dispatch by capability probe rather than ``isinstance``
    # (avoids importing statsmodels at module load — the runtime is
    # imported on every script, and pulling in statsmodels there
    # would charge the import cost on descriptive-only scripts too).
    inner = _safe_attr(model, "model")
    cls_name = type(model).__name__
    # PHReg: result is ``PHRegResults`` (no -Wrapper suffix); the
    # ``status`` attribute on ``model.model`` is the censoring flag.
    is_cox = cls_name.startswith("PHReg") or (
        inner is not None and hasattr(inner, "status")
    )
    # MixedLM detection. ``MixedLMResultsWrapper`` exposes ``cov_re``
    # (the random-effects covariance matrix) — no other result class
    # does. Goes BEFORE the GLM check so a future ``MixedGLM`` shape
    # doesn't get misclassified.
    is_mixed = (not is_cox) and _safe_attr(model, "cov_re") is not None
    # GLM family. Two paths:
    #   * ``Logit`` / ``Probit`` / ``Poisson`` / ``NegativeBinomial``
    #     result wrappers ship ``prsquared`` (McFadden's R²).
    #   * ``smf.glm(... family=Binomial())`` returns ``GLMResultsWrapper``
    #     which does NOT expose ``prsquared`` but ships ``deviance``
    #     and ``null_deviance`` — compute pseudo-R² from those.
    has_prsq = _safe_float(_safe_attr(model, "prsquared")) is not None
    has_deviance_pair = (
        _safe_float(_safe_attr(model, "deviance")) is not None
        and _safe_float(_safe_attr(model, "null_deviance")) is not None
    )
    is_glm = (not is_cox) and (not is_mixed) and (has_prsq or has_deviance_pair)
    # OLS-shape: anything with finite ``rsquared`` that isn't the
    # above. IV2SLS lands here too — it exposes ``rsquared`` but
    # NotImplementedError on ``llf`` / ``aic`` / ``bic``; ``_safe_attr``
    # absorbs those so the helper still emits the OLS fields it can.
    is_ols = (
        (not is_cox) and (not is_glm) and (not is_mixed)
        and _safe_float(_safe_attr(model, "rsquared")) is not None
    )

    # ``statsmodels`` exposes the design as ``model.model.exog_names``;
    # the response is ``model.model.endog_names``. The first column
    # is "Intercept" for formula-fit models and "const" for
    # ``add_constant(X)`` setups — we keep whichever name was used.
    response = getattr(inner, "endog_names", None) if inner is not None else None
    exog_names = list(getattr(inner, "exog_names", []) or []) if inner else []
    # predictor_variables = exog minus the intercept (sanitizer wants
    # the regressors of interest, not the intercept).
    predictors = [n for n in exog_names if n not in ("const", "Intercept")]

    # Coefficient table. PHReg ships these as bare ndarrays — pair
    # with ``exog_names`` rather than letting ``dict(ndarray)`` raise
    # the helper into silent oblivion.
    coefs = _to_dict(_safe_attr(model, "params"), names=exog_names)
    ses   = _to_dict(_safe_attr(model, "bse"),    names=exog_names)
    tvals = _to_dict(_safe_attr(model, "tvalues"), names=exog_names)
    pvals = _to_dict(_safe_attr(model, "pvalues"), names=exog_names)

    # Sample size. ``nobs`` on the result wrapper works for OLS / GLM
    # but is absent on ``PHRegResults``. Fall back to ``endog`` shape
    # so Cox payloads carry ``n`` instead of failing the sanitizer's
    # ``n`` required-int check.
    n = _safe_int(_safe_attr(model, "nobs"))
    if n is None and inner is not None:
        endog = getattr(inner, "endog", None)
        if endog is not None:
            try:
                n = int(getattr(endog, "shape", (len(endog),))[0])
            except (TypeError, AttributeError):
                n = None
    df_resid = _safe_int(_safe_attr(model, "df_resid"))

    fields: dict[str, Any] = {
        "n": n,
        "response_variable": response,
        "predictor_variables": predictors,
        "coefficients": coefs,
        "standard_errors": ses,
        "t_statistics": tvals,
        "p_values": pvals,
        "degrees_of_freedom": df_resid,
    }

    # Class-specific fit metrics — only emit fields meaningful for
    # this estimator. Shipping ``r_squared: null`` from a GLM (the
    # old behaviour) made every GLM payload trigger a sanitizer
    # transformation "dropped 'r_squared': not a finite number",
    # while leaving the actual fit metrics absent.
    if is_ols:
        for src, dst in (
            ("rsquared",     "r_squared"),
            ("rsquared_adj", "adj_r_squared"),
            ("fvalue",       "f_statistic"),
            ("f_pvalue",     "f_p_value"),
            ("llf",          "log_likelihood"),
            ("aic",          "aic"),
            ("bic",          "bic"),
        ):
            v = _safe_float(_safe_attr(model, src))
            if v is not None:
                fields[dst] = v
        # ``scale`` is the residual variance; sanitizer's
        # ``residual_std_error`` slot expects the standard deviation.
        sigma_sq = _safe_float(_safe_attr(model, "scale"))
        if sigma_sq is not None and sigma_sq >= 0:
            fields["residual_std_error"] = math.sqrt(sigma_sq)

    if is_glm:
        # Prefer ``prsquared`` when present; fall back to McFadden-
        # equivalent computed from deviance ratio for ``GLMResultsWrapper``
        # (``smf.glm(family=Binomial())`` and friends), which doesn't
        # expose ``prsquared``.
        pr2 = _safe_float(_safe_attr(model, "prsquared"))
        if pr2 is None:
            dev = _safe_float(_safe_attr(model, "deviance"))
            null_dev = _safe_float(_safe_attr(model, "null_deviance"))
            if dev is not None and null_dev is not None and null_dev > 0:
                pr2 = 1.0 - dev / null_dev
        if pr2 is not None:
            fields["pseudo_r_squared"] = pr2
        for src, dst in (
            ("llf", "log_likelihood"), ("aic", "aic"), ("bic", "bic"),
        ):
            v = _safe_float(_safe_attr(model, src))
            if v is not None:
                fields[dst] = v
        # Chi-squared LR test vs. the null model. ``llnull`` is the
        # log-likelihood of the intercept-only model; chi² =
        # 2 · (llf − llnull). Two sources by class:
        #   * Logit / Poisson / NegBin result wrappers compute it
        #     automatically and expose ``llnull``.
        #   * ``GLMResultsWrapper`` exposes the same via ``llf`` and
        #     the deviance pair: 2 · (llf - llnull) = null_dev - dev.
        llf = _safe_float(_safe_attr(model, "llf"))
        llnull = _safe_float(_safe_attr(model, "llnull"))
        if llf is not None and llnull is not None and llf >= llnull:
            fields["chi_squared"] = 2.0 * (llf - llnull)
        elif "chi_squared" not in fields:
            dev = _safe_float(_safe_attr(model, "deviance"))
            null_dev = _safe_float(_safe_attr(model, "null_deviance"))
            if dev is not None and null_dev is not None and null_dev >= dev:
                fields["chi_squared"] = null_dev - dev

    if is_mixed:
        # statsmodels MixedLM. Fixed-effects coefficient table is
        # already extracted above (Estimate / SE / z / P>|z|, since
        # MixedLM uses z-tests like a GLM). Mixed-specific fields:
        # variance components, per-level group counts, fit method,
        # ICC for the one-level intercept-only common case.
        #
        # statsmodels' single-grouping MixedLM stashes the column
        # values of the grouping factor in ``model.groups`` (an
        # ndarray, no name). The original column name isn't reachable
        # from the result, so the caller passes ``group_variable``
        # via kwargs. If omitted, default to "group" — the model
        # still gets the cardinality, just keyed by a generic name.
        group_var_name = str(extra.pop("group_variable", None) or "group")
        cov_re_attr = _safe_attr(model, "cov_re")
        re_var: dict[str, float] = {}
        if cov_re_attr is not None:
            try:
                # cov_re is a labelled DataFrame; diagonal entries
                # are variances. For random-intercept-only (k_re=1)
                # there's one diagonal entry. Random-slope adds more.
                if hasattr(cov_re_attr, "iloc"):
                    cov_arr = cov_re_attr.values
                    re_names = list(cov_re_attr.index)
                else:
                    import numpy as _np
                    cov_arr = _np.asarray(cov_re_attr)
                    re_names = [f"re_{i+1}" for i in range(cov_arr.shape[0])]
                for i, rn in enumerate(re_names):
                    v = float(cov_arr[i, i])
                    if not math.isfinite(v) or v < 0:
                        continue
                    # statsmodels labels the intercept random effect
                    # as "Group" by default; remap to bare group name.
                    if str(rn).lower() in ("group", "(intercept)", "intercept"):
                        key = group_var_name
                    else:
                        key = f"{group_var_name}.{rn}"
                    re_var[key] = v
            except Exception:  # noqa: BLE001
                pass
        scale = _safe_float(_safe_attr(model, "scale"))
        if scale is not None and scale >= 0:
            re_var["residual"] = scale
        if re_var:
            fields["random_effects_variance"] = re_var
        # n_groups_per_level: single-grouping statsmodels exposes
        # ``model.model.n_groups`` (the inner-model's int).
        if inner is not None:
            n_g = _safe_int(_safe_attr(inner, "n_groups"))
            if n_g is not None:
                fields["n_groups_per_level"] = {group_var_name: n_g}
        # Fit method.
        reml = _safe_attr(model, "reml")
        if isinstance(reml, bool):
            fields["fit_method"] = "REML" if reml else "ML"
        # ICC for the one-grouping intercept-only case.
        if "random_effects_variance" in fields:
            rev = fields["random_effects_variance"]
            if len(rev) == 2 and "residual" in rev:
                grp_keys = [k for k in rev if k != "residual"]
                if len(grp_keys) == 1:
                    s_u2 = rev[grp_keys[0]]
                    s_e2 = rev["residual"]
                    if s_u2 + s_e2 > 0:
                        fields["icc"] = s_u2 / (s_u2 + s_e2)
        for src, dst in (
            ("llf", "log_likelihood"), ("aic", "aic"), ("bic", "bic"),
        ):
            v = _safe_float(_safe_attr(model, src))
            if v is not None:
                fields[dst] = v

    if is_cox:
        # ``PHRegResults`` carries ``llf`` but not ``aic`` / ``bic`` on
        # the wrapper. Subject + failure counts come from the inner
        # ``model`` — ``model.endog`` is the observed time vector,
        # ``model.status`` the event indicator.
        llf = _safe_float(_safe_attr(model, "llf"))
        if llf is not None:
            fields["log_likelihood"] = llf
        if inner is not None:
            endog = getattr(inner, "endog", None)
            status = getattr(inner, "status", None)
            try:
                if endog is not None:
                    fields["n_subjects"] = int(
                        getattr(endog, "shape", (len(endog),))[0]
                    )
            except (TypeError, AttributeError):
                pass
            try:
                if status is not None:
                    fields["n_failures"] = int(sum(int(v != 0) for v in status))
            except (TypeError, AttributeError):
                pass

    # Cluster-robust SE metadata. statsmodels signals clustering via
    # ``cov_type == "cluster"`` and stashes the cluster assignment
    # vector in ``cov_kwds["groups"]``. We emit:
    #   * ``cluster_variables`` — the column name(s); for multi-way
    #     clustering ``groups`` is a 2-D array, treat each axis as
    #     a separate dimension.
    #   * ``n_clusters`` — cardinality per dimension (the same shape
    #     and disclosure profile as ``fixed_effects``).
    # Listing the cluster level identities is forbidden — only
    # cardinality and column names cross the boundary. Both are
    # already in the dataset schema the model saw.
    cov_type = _safe_attr(model, "cov_type")
    if isinstance(cov_type, str) and cov_type.lower() == "cluster":
        cov_kwds = _safe_attr(model, "cov_kwds") or {}
        groups = cov_kwds.get("groups") if isinstance(cov_kwds, dict) else None
        cluster_names, n_clusters = _extract_cluster_metadata(groups)
        if cluster_names:
            fields["cluster_variables"] = cluster_names
        if n_clusters:
            fields["n_clusters"] = n_clusters
        fields["robust_se_type"] = "cluster"
    else:
        # Non-cluster variance estimator. Map statsmodels' ``cov_type``
        # values onto the sanitizer's canonical robust_se_type enum
        # so the model can interpret the variance flavour at a glance
        # ("hc1" / "hac_newey_west" / "bootstrap") without parsing the
        # raw label. Helpers don't need to flag classical SEs explicitly
        # — absence of ``robust_se_type`` already implies model-based
        # SEs — but emitting it makes the choice legible on
        # ``expand_result(view="full")``.
        rse = _normalise_robust_se_type(cov_type)
        if rse is not None:
            fields["robust_se_type"] = rse

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
    # ``coefficient_table_with_fit_stats`` is the canonical bucket
    # name covering OLS / GLM / Cox / IV / fixest. ``linear_regression``
    # stays as a back-compat alias in the sanitizer for existing
    # stored payloads; new emissions use the descriptive name.
    result(type="coefficient_table_with_fit_stats", **fields)


def _normalise_robust_se_type(cov_type: Any) -> str | None:
    """Map a statsmodels ``cov_type`` label onto the sanitizer's
    canonical robust_se_type enum.

    Returns ``None`` when the label isn't recognised (so the helper
    omits the field rather than smuggling a free-text value through),
    or when the label is the model-based default (``"nonrobust"``)
    where absence of the field already communicates "classical SEs".
    Cluster handling lives in the calling site — it needs the
    cov_kwds["groups"] payload alongside the label, so it stays
    inline rather than routing through here.
    """
    if not isinstance(cov_type, str):
        return None
    key = cov_type.strip().lower()
    if not key or key == "nonrobust":
        return None
    # Heteroskedasticity-consistent. statsmodels accepts both bare
    # ``"HC0"`` / ``"HC1"`` / ... and the prefixed ``"hc0"`` / etc.
    # depending on the call path.
    if key in ("hc0", "hc1", "hc2", "hc3"):
        return key
    # Newey-West HAC. statsmodels: ``"HAC"`` for kernel HAC,
    # ``"hac-panel"`` / ``"hac-groupsum"`` for panel-data flavours.
    # All collapse to ``hac_newey_west`` for the model — the gain
    # from distinguishing them at the wire-format level is small
    # compared to the cost of a wider enum.
    if key.startswith("hac"):
        return "hac_newey_west"
    # Bootstrap covariance — surfaces under names like
    # ``"bootstrap"`` or ``"clusterbootstrap"`` depending on package
    # version. ``cluster`` is handled separately above.
    if "bootstrap" in key:
        return "bootstrap"
    # Robust default in some packages — the typical mapping is HC1.
    # statsmodels' ``"robust"`` doesn't exist canonically; this
    # branch absorbs out-of-tree adapters that emit it.
    if key in ("robust", "sandwich"):
        return "hc1"
    return None


def _extract_cluster_metadata(
    groups: Any,
) -> tuple[list[str], dict[str, int]]:
    """Pull ``(cluster_variables, n_clusters)`` from a statsmodels
    ``cov_kwds["groups"]``.

    The shape is one of:
      * pandas Series — single-cluster, ``.name`` carries the column
        name (or empty if the caller passed a bare ndarray).
      * 1-D ndarray / list — single-cluster, no name available
        (caller used a raw array). Emit a positional label.
      * 2-D ndarray (rows × ndim) — multi-way clustering with
        ``ndim`` dimensions. Each column is one clustering axis.
      * pandas DataFrame — multi-way clustering with column names.

    Returns ``([], {})`` when the groups object isn't recognisable —
    the helper omits the fields rather than emitting incoherent
    metadata.
    """
    if groups is None:
        return [], {}
    try:
        import numpy as np
    except Exception:  # noqa: BLE001
        return [], {}
    # pandas DataFrame: multi-column → multi-way.
    if hasattr(groups, "columns"):
        names: list[str] = [str(c) for c in groups.columns]
        counts: dict[str, int] = {}
        for c in groups.columns:
            try:
                counts[str(c)] = int(groups[c].nunique())
            except Exception:  # noqa: BLE001
                pass
        return names, counts
    # pandas Series: single-cluster with a name.
    name_attr = getattr(groups, "name", None)
    if name_attr is not None and not isinstance(groups, (list, tuple)):
        try:
            arr = np.asarray(groups)
        except Exception:  # noqa: BLE001
            return [], {}
        if arr.ndim == 1:
            return [str(name_attr)], {str(name_attr): int(np.unique(arr).size)}
    # ndarray / list. 2-D → multi-way (no names); 1-D → single.
    try:
        arr = np.asarray(groups)
    except Exception:  # noqa: BLE001
        return [], {}
    if arr.ndim == 1:
        return ["cluster"], {"cluster": int(np.unique(arr).size)}
    if arr.ndim == 2:
        names = [f"cluster_{i+1}" for i in range(arr.shape[1])]
        counts = {
            names[i]: int(np.unique(arr[:, i]).size)
            for i in range(arr.shape[1])
        }
        return names, counts
    return [], {}


def from_iv(
    model: Any,
    *,
    instrument_variables: list[str] | None = None,
    endogenous_variables: list[str] | None = None,
    first_stage_f: float | None = None,
    weak_instrument_p: float | None = None,
    hansen_j: float | None = None,
    hansen_j_p: float | None = None,
    endogeneity_p: float | None = None,
    **extra: Any,
) -> None:
    """Emit a regression-bucket payload from a 2SLS / IV fit, plus
    the IV-specific diagnostic scalars.

    Decision pinned in ``docs/direction.md`` "IV as regression-bucket
    extension": 2SLS is structurally a regression-shape payload (the
    structural-equation coefficient table) with a handful of extra
    diagnostic scalars (first-stage F, Sargan / Hansen J,
    Wu-Hausman). It does NOT need a composite shape — that territory
    is reserved for genuine multi-stage estimators (3SLS, mediation,
    control-function corrections) where the model needs two
    independent coefficient tables.

    statsmodels' ``sandbox.regression.gmm.IV2SLS`` does not compute
    the first-stage F or Sargan / Wu-Hausman automatically — its
    sandbox status reflects that incomplete diagnostics surface.
    Compute them script-side and pass them through:

        from statsmodels.sandbox.regression.gmm import IV2SLS
        m = IV2SLS(y, exog, instruments).fit()
        first_stage = sm.OLS(endo, instruments).fit()
        nora.from_iv(
            m,
            instrument_variables=["z1", "z2"],
            endogenous_variables=["x_endo"],
            first_stage_f=float(first_stage.fvalue),
        )

    If you're using ``linearmodels.iv.IV2SLS`` (which DOES compute
    these), pass ``model.first_stage.diagnostics["f.stat"]``,
    ``model.sargan.stat`` / ``model.sargan.pval``, and
    ``model.wu_hausman().stat`` / ``model.wu_hausman().pval``.
    """
    iv_extra: dict[str, Any] = {}
    if instrument_variables is not None:
        iv_extra["instrument_variables"] = list(instrument_variables)
        iv_extra["n_instruments"] = len(instrument_variables)
    if endogenous_variables is not None:
        iv_extra["endogenous_variables"] = list(endogenous_variables)
        iv_extra["n_endogenous"] = len(endogenous_variables)
    for k, v in (
        ("first_stage_f", first_stage_f),
        ("weak_instrument_p", weak_instrument_p),
        ("hansen_j", hansen_j),
        ("hansen_j_p", hansen_j_p),
        ("endogeneity_p", endogeneity_p),
    ):
        vf = _safe_float(v)
        if vf is not None:
            iv_extra[k] = vf
    iv_extra.update(extra)
    from_lm(model, **iv_extra)


def from_marginal_effects(
    margeff: Any,
    *,
    variables: list[str] | None = None,
    method: str | None = None,
    outcome_variable: str | None = None,
    model_family: str | None = None,
    at_values: dict[str, float] | None = None,
    label: str | None = None,
    **extra: Any,
) -> None:
    """Emit a ``marginal_effects`` payload from a statsmodels
    ``DiscreteMargins`` / ``GenericMargins`` result.

    Wraps the output of ``fit.get_margeff(at=..., method=...)`` on a
    fitted statsmodels Logit / Probit / Poisson / GLM result. The
    ``DiscreteMargins`` object exposes:

      * ``margeff``           — per-variable marginal effects (ndarray)
      * ``margeff_se``        — delta-method standard errors
      * ``tvalues`` / ``pvalues`` — Wald-style test outputs
      * ``conf_int()``        — 95% CIs as a 2-D array
      * ``results``           — back-reference to the underlying fit;
                                ``.model.exog_names`` provides the
                                column labels.
      * ``margeff_options``   — dict carrying the ``at`` / ``method``
                                Stata-vocabulary choices the caller passed.

    Method mapping from statsmodels' ``at`` keyword onto the
    sanitizer's enum:

      * ``"overall"``  → ``"ame"``  (average over the sample)
      * ``"mean"``     → ``"mem"``  (evaluated at sample means)
      * ``"median"``   → ``"at_representative"`` (median is one
        specific representative covariate vector; the medians are
        passed through ``at_values``)
      * any explicit ``at`` dict → ``"at_representative"`` with the
        dict in ``at_values``

    Example:
        from statsmodels.formula.api import logit
        m = logit("y ~ age + female + income", data=df).fit()
        me = m.get_margeff(at="overall", method="dydx")
        nora.from_marginal_effects(
            me, outcome_variable="y", model_family="logit",
            label="AME from logit",
        )

    The helper is intentionally narrow — it reads from the
    ``DiscreteMargins`` shape statsmodels produces and routes onto
    the sanitizer's enum. R's ``marginaleffects::avg_slopes()``
    output is structurally different; that path uses
    ``nora$from_marginal_effects`` in the R runtime.

    **Disclosure note on ``at_values``** (relevant only for
    ``method="at_representative"``): the conditioning vector you
    pass is precision-clamped by the sample N before it reaches the
    model — at n=1000 you get ~4 sigfigs, at n=100 you get ~3. Pass
    interpretable summary points (means, medians, percentiles,
    round reference values from the literature). An exact-precision
    value pulled from a single row is gated by the precision floor;
    it won't cross as raw bytes, but the right interpretation is
    still "this is a representative point at this precision".
    """
    try:
        print(margeff.summary() if callable(getattr(margeff, "summary", None))
              else margeff)
    except Exception:  # noqa: BLE001
        pass

    # Duck-typed access — don't import statsmodels at module load.
    eff_arr = _safe_attr(margeff, "margeff")
    if eff_arr is None:
        raise TypeError(
            "nora.from_marginal_effects: ``margeff`` must expose "
            "``.margeff`` (statsmodels DiscreteMargins / GenericMargins "
            "shape). Try ``fit.get_margeff(at=..., method='dydx')``."
        )

    try:
        import numpy as np
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "nora.from_marginal_effects requires numpy"
        ) from e

    eff = np.asarray(eff_arr, dtype=float).ravel()
    se_attr = _safe_attr(margeff, "margeff_se")
    se = np.asarray(se_attr, dtype=float).ravel() if se_attr is not None else None
    t_attr = _safe_attr(margeff, "tvalues")
    tv = np.asarray(t_attr, dtype=float).ravel() if t_attr is not None else None
    p_attr = _safe_attr(margeff, "pvalues")
    pv = np.asarray(p_attr, dtype=float).ravel() if p_attr is not None else None
    ci_fn = _safe_attr(margeff, "conf_int")
    ci_arr = None
    if callable(ci_fn):
        try:
            ci_arr = np.asarray(ci_fn(), dtype=float)
        except Exception:  # noqa: BLE001
            ci_arr = None

    # Variable names. statsmodels' ``get_margeff`` drops the constant
    # automatically; the remaining ``margeff_options["exog_names"]``
    # carries the surviving column labels in order. If not present,
    # fall back to ``model.exog_names`` minus standard intercept
    # aliases.
    if variables is None:
        opts = _safe_attr(margeff, "margeff_options") or {}
        if isinstance(opts, dict) and isinstance(opts.get("exog_names"), list):
            variables = [str(v) for v in opts["exog_names"]]
        else:
            inner = _safe_attr(margeff, "results")
            inner_model = (
                _safe_attr(inner, "model") if inner is not None else None
            )
            exog_names = (
                list(getattr(inner_model, "exog_names", []) or [])
                if inner_model is not None else []
            )
            variables = [
                n for n in exog_names
                if n not in ("const", "Intercept", "(Intercept)", "intercept")
            ]
    variables = [str(v) for v in variables]
    if len(variables) != eff.size:
        raise ValueError(
            f"nora.from_marginal_effects: ``variables`` has "
            f"{len(variables)} entries but margeff has {eff.size}"
        )

    effects: dict[str, float] = {}
    ses: dict[str, float] = {}
    pvs: dict[str, float] = {}
    zs: dict[str, float] = {}
    los: dict[str, float] = {}
    his: dict[str, float] = {}
    for i, v in enumerate(variables):
        if i < eff.size and math.isfinite(float(eff[i])):
            effects[v] = float(eff[i])
        if se is not None and i < se.size and math.isfinite(float(se[i])):
            ses[v] = float(se[i])
        if tv is not None and i < tv.size and math.isfinite(float(tv[i])):
            zs[v] = float(tv[i])
        if pv is not None and i < pv.size and math.isfinite(float(pv[i])):
            pvs[v] = float(pv[i])
        if ci_arr is not None and i < ci_arr.shape[0] and ci_arr.shape[1] >= 2:
            lo, hi = float(ci_arr[i, 0]), float(ci_arr[i, 1])
            if math.isfinite(lo):
                los[v] = lo
            if math.isfinite(hi):
                his[v] = hi

    # Method resolution. Caller-supplied wins; otherwise infer from
    # the ``margeff_options["at"]`` value statsmodels stashes on the
    # result.
    if method is None:
        opts = _safe_attr(margeff, "margeff_options") or {}
        at = opts.get("at") if isinstance(opts, dict) else None
        if at == "overall" or at is None:
            method = "ame"
        elif at == "mean":
            method = "mem"
        else:
            method = "at_representative"
    method = str(method)

    # n: rows of the design statsmodels fit on. Pull from the inner
    # model — ``margeff`` itself doesn't carry a count directly.
    n_val: int | None = None
    inner = _safe_attr(margeff, "results")
    if inner is not None:
        n_val = _safe_int(_safe_attr(inner, "nobs"))
        if n_val is None:
            inner_model = _safe_attr(inner, "model")
            endog = (
                getattr(inner_model, "endog", None)
                if inner_model is not None else None
            )
            if endog is not None:
                try:
                    n_val = int(getattr(endog, "shape", (len(endog),))[0])
                except (TypeError, AttributeError):
                    n_val = None

    fields: dict[str, Any] = {
        "type": "marginal_effects",
        "method": method,
        "variables": variables,
        "effects": effects,
    }
    if ses:
        fields["standard_errors"] = ses
    if zs:
        fields["z_statistics"] = zs
    if pvs:
        fields["p_values"] = pvs
    if los:
        fields["ci_lower"] = los
    if his:
        fields["ci_upper"] = his
    if n_val is not None:
        fields["n"] = n_val
    if outcome_variable is not None:
        fields["outcome_variable"] = str(outcome_variable)
    if model_family is not None:
        fields["model_family"] = str(model_family)
    elif inner is not None:
        # Auto-detect: ``inner.model.__class__.__name__`` reveals
        # whether we're in Logit / Probit / Poisson / etc.
        cls = _safe_attr(_safe_attr(inner, "model"), "__class__")
        if cls is not None:
            name = getattr(cls, "__name__", "")
            if isinstance(name, str) and name:
                fields["model_family"] = name.lower()
    if at_values is not None and at_values:
        clean_at: dict[str, float] = {}
        for k, val in at_values.items():
            vf = _safe_float(val)
            if vf is not None:
                clean_at[str(k)] = vf
        if clean_at:
            fields["at_values"] = clean_at
    if label is not None:
        fields["label"] = str(label)
    fields.update(extra)
    result(**fields)


def from_cluster(
    fit: Any,
    X: Any = None,
    *,
    variables: list[str] | None = None,
    label: str | None = None,
) -> None:
    """Emit a ``cluster_analysis`` payload from a sklearn clustering
    fit. Dispatches on class:

      * ``KMeans`` — cluster centers and labels read directly from
        the fit; ``X`` not needed.
      * ``AgglomerativeClustering`` — sklearn agglomerative fits
        don't store cluster centers (they're not centroid-based),
        so ``X`` (the matrix the fit was built on) is required.
        Centroids and within-cluster SS computed post-hoc from
        ``X[fit.labels_ == k].mean(axis=0)``.

    DBSCAN and HDBSCAN are intentionally not supported. Their
    inference-adequacy story (density parameters, noise points, no
    centroids by construction) needs a separate design pass. The
    helper raises with a clear pointer to the generic
    ``nora.result(type="cluster_analysis", method="dbscan", ...)``
    path.

    Per-observation cluster assignments (``fit.labels_``) are NOT
    emitted on any path — per-row data, no allowlist slot.

    Examples:
        from sklearn.cluster import KMeans, AgglomerativeClustering
        X = df[["age", "income", "tenure"]].values
        nora.from_cluster(KMeans(n_clusters=4, random_state=42,
                                 n_init=10).fit(X),
                         variables=["age", "income", "tenure"])
        nora.from_cluster(AgglomerativeClustering(n_clusters=4,
                                                  linkage="ward").fit(X),
                         X=X, variables=["age", "income", "tenure"])
    """
    cls_name = type(fit).__name__
    if cls_name == "KMeans":
        _from_kmeans_impl(fit, variables=variables, label=label)
        return
    if cls_name == "AgglomerativeClustering":
        if X is None:
            raise ValueError(
                "nora.from_cluster: AgglomerativeClustering fits don't store "
                "centers — pass ``X`` (the matrix the fit was built on) so "
                "the helper can compute centroids from "
                "``X[fit.labels_ == k].mean(axis=0)``."
            )
        _from_agglomerative_impl(fit, X, variables=variables, label=label)
        return
    if cls_name in ("DBSCAN", "HDBSCAN"):
        raise TypeError(
            f"nora.from_cluster: dedicated {cls_name} helper not yet "
            f"shipped. Construct the payload via "
            f'`nora.result(type="cluster_analysis", method="dbscan", '
            f"cluster_sizes=..., n_noise_points=..., variables=..., ...)` "
            f"from the script — the cluster_analysis shape accepts dbscan "
            f"with centroids absent."
        )
    raise TypeError(
        f"nora.from_cluster: unknown clustering class {cls_name!r}. "
        f"Supported: KMeans, AgglomerativeClustering. DBSCAN-family: "
        f"use generic ``nora.result(type='cluster_analysis', ...)`` "
        f"until a dedicated helper ships."
    )


def _from_kmeans_impl(
    fit: Any,
    *,
    variables: list[str] | None = None,
    label: str | None = None,
) -> None:
    try:
        print(fit)
    except Exception:  # noqa: BLE001
        pass
    import numpy as np
    centers_attr = _safe_attr(fit, "cluster_centers_")
    labels_attr = _safe_attr(fit, "labels_")
    if centers_attr is None or labels_attr is None:
        raise RuntimeError(
            "nora.from_cluster: fit.cluster_centers_ or fit.labels_ missing"
        )
    centers = np.asarray(centers_attr)
    labels = np.asarray(labels_attr)
    n_clusters_actual, n_features = centers.shape

    if variables is None:
        variables = [f"feature_{i+1}" for i in range(n_features)]
    variables = [str(v) for v in variables]
    if len(variables) != n_features:
        raise ValueError(
            f"nora.from_cluster: variables has {len(variables)} entries "
            f"but KMeans was fit on {n_features} features"
        )

    cluster_labels = [f"cluster_{i+1}" for i in range(n_clusters_actual)]
    cluster_sizes: dict[str, int] = {}
    for i in range(n_clusters_actual):
        cluster_sizes[cluster_labels[i]] = int((labels == i).sum())

    centroids: dict[str, dict[str, float]] = {}
    for ci in range(n_clusters_actual):
        row: dict[str, float] = {}
        for fi, var in enumerate(variables):
            v = float(centers[ci, fi])
            if math.isfinite(v):
                row[var] = v
        if row:
            centroids[cluster_labels[ci]] = row

    inertia = _safe_float(_safe_attr(fit, "inertia_"))
    n_iter = _safe_int(_safe_attr(fit, "n_iter_"))
    n_obs = int(labels.size)
    fields: dict[str, Any] = {
        "type": "cluster_analysis",
        "method": "kmeans",
        "distance_metric": "euclidean",
        "n_observations": n_obs,
        "n_clusters": n_clusters_actual,
        "n_features": n_features,
        "variables": variables,
        "cluster_labels": cluster_labels,
        "cluster_sizes": cluster_sizes,
        "centroids": centroids,
    }
    if inertia is not None:
        fields["total_within_ss"] = inertia
        fields["inertia"] = inertia
    if n_iter is not None:
        fields["n_iterations"] = n_iter
    if label is not None:
        fields["label"] = str(label)
    result(**fields)


def _from_agglomerative_impl(
    fit: Any,
    X: Any,
    *,
    variables: list[str] | None = None,
    label: str | None = None,
) -> None:
    """sklearn ``AgglomerativeClustering`` doesn't store centroids
    or within-cluster SS. Both are computed post-hoc from ``X``:
        centroid_k = X[labels == k].mean(axis=0)
        within_ss_k = ||X[labels == k] - centroid_k||²

    The dendrogram (``fit.children_``, ``fit.distances_``) is
    structurally absent from the payload — per-merge records over
    the data, researcher-only by construction.
    """
    try:
        print(fit)
    except Exception:  # noqa: BLE001
        pass
    import numpy as np
    X_arr = np.asarray(X, dtype=float)
    if X_arr.ndim != 2:
        raise ValueError(
            "nora.from_cluster: ``X`` must be 2-D (n_observations × n_features)"
        )
    labels = np.asarray(_safe_attr(fit, "labels_"))
    n_clusters_actual = int(_safe_attr(fit, "n_clusters_") or labels.max() + 1)
    n_features = X_arr.shape[1]

    if variables is None:
        variables = [f"feature_{i+1}" for i in range(n_features)]
    variables = [str(v) for v in variables]
    if len(variables) != n_features:
        raise ValueError(
            f"nora.from_cluster: variables has {len(variables)} entries "
            f"but X has {n_features} columns"
        )

    cluster_labels = [f"cluster_{i+1}" for i in range(n_clusters_actual)]
    cluster_sizes: dict[str, int] = {}
    centroids: dict[str, dict[str, float]] = {}
    within_cluster_ss: dict[str, float] = {}
    total_within_ss = 0.0
    grand_mean = X_arr.mean(axis=0)
    for i in range(n_clusters_actual):
        cl = cluster_labels[i]
        mask = labels == i
        size = int(mask.sum())
        cluster_sizes[cl] = size
        if size == 0:
            continue
        sub = X_arr[mask]
        centroid = sub.mean(axis=0)
        centroids[cl] = {variables[fi]: float(centroid[fi]) for fi in range(n_features)}
        wss = float(((sub - centroid) ** 2).sum())
        within_cluster_ss[cl] = wss
        total_within_ss += wss
    total_ss = float(((X_arr - grand_mean) ** 2).sum())
    between_ss = total_ss - total_within_ss

    linkage_attr = _safe_attr(fit, "linkage")
    linkage = str(linkage_attr) if isinstance(linkage_attr, str) else None
    # sklearn distance defaults to euclidean for ward; the attribute
    # is ``metric`` (newer) or ``affinity`` (older).
    metric_attr = _safe_attr(fit, "metric") or _safe_attr(fit, "affinity")

    fields: dict[str, Any] = {
        "type": "cluster_analysis",
        "method": "hierarchical",
        "n_observations": int(X_arr.shape[0]),
        "n_clusters": n_clusters_actual,
        "n_features": n_features,
        "variables": variables,
        "cluster_labels": cluster_labels,
        "cluster_sizes": cluster_sizes,
        "centroids": centroids,
        "within_cluster_ss": within_cluster_ss,
        "total_within_ss": total_within_ss,
        "inertia": total_within_ss,
        "between_cluster_ss": between_ss,
        "total_ss": total_ss,
    }
    if total_ss > 0:
        fields["ss_ratio"] = between_ss / total_ss
    if linkage:
        fields["linkage"] = linkage
    if isinstance(metric_attr, str):
        fields["distance_metric"] = metric_attr
    if label is not None:
        fields["label"] = str(label)
    result(**fields)


# Back-compat alias. ``from_kmeans`` was the public name in earlier
# releases; ``from_cluster`` with class dispatch is the new
# canonical entry point.
def from_kmeans(
    fit: Any,
    *,
    variables: list[str] | None = None,
    label: str | None = None,
) -> None:
    """Back-compat alias for ``from_cluster`` on KMeans fits."""
    from_cluster(fit, variables=variables, label=label)


def from_pca(
    fit: Any,
    *,
    variables: list[str] | None = None,
    n_components: int | None = None,
    label: str | None = None,
) -> None:
    """Emit a ``factor_decomposition`` payload from a fitted
    ``sklearn.decomposition.PCA``.

    sklearn's PCA stores loadings transposed relative to R's prcomp:
    ``components_`` is ``(n_components, n_features)`` (each row is a
    component's loadings on the features). We pivot to the
    ``{variable: {component: value}}`` shape the sanitizer's
    ``loadings`` field expects.

    Caller passes ``variables`` (the column-name list matching the
    order of features the PCA was fit on). sklearn doesn't store
    column names — its design takes a 2-D array — so the names must
    come from the caller. If omitted, generic ``feature_1`` … fall-
    backs are used; less useful to the model.

    Privacy carve-out: ``fit.transform(X)`` (the per-observation
    factor scores) is researcher-only by structural absence —
    nothing in this helper emits it, and no field on the
    ``factor_decomposition`` allowlist would accept it.

    Example:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=3).fit(df[["v1","v2","v3","v4","v5"]])
        nora.from_pca(pca, variables=["v1","v2","v3","v4","v5"],
                     label="five-variable PCA")
    """
    cls_name = type(fit).__name__
    if cls_name != "PCA":
        raise TypeError(
            "nora.from_pca: ``fit`` must be a sklearn.decomposition.PCA "
            f"instance; got {cls_name!r}"
        )
    try:
        print(fit)
    except Exception:  # noqa: BLE001
        pass
    try:
        import numpy as np
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("nora.from_pca requires numpy") from e

    components_arr = _safe_attr(fit, "components_")
    if components_arr is None:
        raise RuntimeError(
            "nora.from_pca: fit.components_ missing — was the PCA fitted?"
        )
    components_arr = np.asarray(components_arr)  # (n_components, n_features)
    n_comp_full, n_feat = components_arr.shape

    n_obs = _safe_int(_safe_attr(fit, "n_samples_"))
    if n_obs is None:
        raise RuntimeError(
            "nora.from_pca: fit.n_samples_ missing — n_observations is required"
        )

    if variables is None:
        variables = [f"feature_{i+1}" for i in range(n_feat)]
    variables = [str(v) for v in variables]
    if len(variables) != n_feat:
        raise ValueError(
            f"nora.from_pca: variables has {len(variables)} entries but "
            f"PCA was fit on {n_feat} features"
        )

    n_comp = (
        min(n_components, n_comp_full) if isinstance(n_components, int)
        else n_comp_full
    )
    comp_labels = [f"PC{i+1}" for i in range(n_comp)]

    # Loadings: pivot from (n_comp, n_feat) to {variable: {component: value}}.
    loadings: dict[str, dict[str, float]] = {}
    for fi, var in enumerate(variables):
        row: dict[str, float] = {}
        for ci, c_lab in enumerate(comp_labels):
            v = float(components_arr[ci, fi])
            if math.isfinite(v):
                row[c_lab] = v
        if row:
            loadings[var] = row

    # Explained variance / ratio / cumulative — sklearn exposes
    # ``explained_variance_`` (eigenvalues) and
    # ``explained_variance_ratio_`` (normalized to sum=1 over
    # *retained* components). Compute cumulative from the ratio.
    # ``or []`` doesn't compose with ndarrays — ``bool(array)`` raises
    # for arrays with more than one element. Write the None check
    # explicitly.
    _ev = _safe_attr(fit, "explained_variance_")
    ev_attr = np.asarray(_ev if _ev is not None else [])
    _evr = _safe_attr(fit, "explained_variance_ratio_")
    evr_attr = np.asarray(_evr if _evr is not None else [])
    eigenvalues: dict[str, float] = {}
    explained_variance: dict[str, float] = {}
    explained_variance_ratio: dict[str, float] = {}
    cumulative_variance: dict[str, float] = {}
    cum = 0.0
    for ci, c_lab in enumerate(comp_labels):
        if ci < len(ev_attr) and math.isfinite(float(ev_attr[ci])):
            eigenvalues[c_lab] = float(ev_attr[ci])
            explained_variance[c_lab] = float(ev_attr[ci])
        if ci < len(evr_attr) and math.isfinite(float(evr_attr[ci])):
            r = float(evr_attr[ci])
            explained_variance_ratio[c_lab] = r
            cum += r
            cumulative_variance[c_lab] = cum

    # Communalities (PCA): sum of squared loadings across retained
    # components, per variable.
    communalities: dict[str, float] = {}
    for fi, var in enumerate(variables):
        h2 = float(np.sum(components_arr[:n_comp, fi] ** 2))
        if math.isfinite(h2):
            communalities[var] = h2

    fields: dict[str, Any] = {
        "type": "factor_decomposition",
        "method": "pca",
        "rotation": "none",
        "n_observations": n_obs,
        "n_variables": n_feat,
        "n_components": n_comp,
        "variables": variables,
        "components": comp_labels,
        "loadings": loadings,
        "explained_variance": explained_variance,
        "explained_variance_ratio": explained_variance_ratio,
        "cumulative_variance": cumulative_variance,
        "eigenvalues": eigenvalues,
        "communalities": communalities,
    }
    if label is not None:
        fields["label"] = str(label)
    result(**fields)


def from_factor_analyzer(
    fit: Any,
    *,
    variables: list[str] | None = None,
    method: str | None = None,
    rotation: str | None = None,
    n_observations: int | None = None,
    label: str | None = None,
    **extra: Any,
) -> None:
    """Emit a ``factor_decomposition`` payload from a fitted
    ``factor_analyzer.FactorAnalyzer``.

    factor_analyzer is the Python-side standard for exploratory
    factor analysis (the closest analogue to R ``psych::fa``).
    The fit object exposes:

      * ``loadings_``        — (n_features, n_factors) ndarray
      * ``get_uniquenesses()``
      * ``get_communalities()``
      * ``get_eigenvalues()`` — returns (original, common-factor)
      * ``get_factor_variance()`` — (variance, proportional,
                                     cumulative) per factor

    factor_analyzer doesn't store column names (it's fit on a
    bare ndarray or DataFrame), so ``variables`` must be passed
    explicitly when the fit was built from a numpy array. When
    fit from a DataFrame, factor_analyzer stashes the columns in
    ``.feature_names_`` (newer versions) — probed below as a
    fallback.

    Privacy carve-out: the per-row factor scores (the result of
    ``fit.transform(X)``) are structurally absent from the
    sanitizer's allowlist — no field accepts a 2-D array of
    per-observation values.

    Example:
        from factor_analyzer import FactorAnalyzer
        fa = FactorAnalyzer(n_factors=3, rotation="varimax", method="ml")
        fa.fit(df[["v1","v2","v3","v4","v5"]])
        nora.from_factor_analyzer(
            fa, variables=["v1","v2","v3","v4","v5"],
            n_observations=len(df), label="ML FA with varimax",
        )
    """
    cls_name = type(fit).__name__
    if cls_name != "FactorAnalyzer":
        raise TypeError(
            "nora.from_factor_analyzer: ``fit`` must be a "
            "factor_analyzer.FactorAnalyzer instance; got "
            f"{cls_name!r}"
        )
    try:
        print(fit)
    except Exception:  # noqa: BLE001
        pass
    try:
        import numpy as np
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "nora.from_factor_analyzer requires numpy"
        ) from e

    loadings_arr = _safe_attr(fit, "loadings_")
    if loadings_arr is None:
        raise RuntimeError(
            "nora.from_factor_analyzer: fit.loadings_ missing — "
            "was the fit run?"
        )
    loadings_arr = np.asarray(loadings_arr)
    n_feat, n_factors = loadings_arr.shape

    # Variable names. Prefer the caller's list; fall back to
    # factor_analyzer's stash; default to feature_N for unnamed.
    if variables is None:
        fnames = _safe_attr(fit, "feature_names_")
        if fnames is not None:
            variables = [str(x) for x in fnames]
        else:
            variables = [f"feature_{i+1}" for i in range(n_feat)]
    variables = [str(v) for v in variables]
    if len(variables) != n_feat:
        raise ValueError(
            f"nora.from_factor_analyzer: variables has "
            f"{len(variables)} entries but FA was fit on {n_feat} features"
        )

    fac_labels = [f"factor{i+1}" for i in range(n_factors)]

    # Method: map factor_analyzer's ``method`` slot to the sanitizer
    # enum. The valid values on the fit are "minres" / "ml" /
    # "principal".
    if method is None:
        m_attr = _safe_attr(fit, "method")
        if isinstance(m_attr, str):
            m = m_attr.lower()
            if m == "ml":
                method = "maximum_likelihood"
            elif m == "minres":
                method = "minimum_residual"
            elif m == "principal":
                method = "principal_factor"
            else:
                method = "factor_analysis"
        else:
            method = "factor_analysis"
    method = str(method)

    if rotation is None:
        r_attr = _safe_attr(fit, "rotation")
        rotation = str(r_attr) if isinstance(r_attr, str) else "none"
    # Normalize None → "none" so the sanitizer's enum check passes.
    if rotation in (None, "None"):
        rotation = "none"

    loadings: dict[str, dict[str, float]] = {}
    for fi, var in enumerate(variables):
        row: dict[str, float] = {}
        for j, lab in enumerate(fac_labels):
            v = float(loadings_arr[fi, j])
            if math.isfinite(v):
                row[lab] = v
        if row:
            loadings[var] = row

    # Uniqueness / communalities via factor_analyzer's getters.
    uniqueness: dict[str, float] = {}
    communalities: dict[str, float] = {}
    try:
        u_arr = np.asarray(fit.get_uniquenesses())
        for fi, var in enumerate(variables):
            v = float(u_arr[fi])
            if math.isfinite(v):
                uniqueness[var] = v
    except Exception:  # noqa: BLE001
        pass
    try:
        c_arr = np.asarray(fit.get_communalities())
        for fi, var in enumerate(variables):
            v = float(c_arr[fi])
            if math.isfinite(v):
                communalities[var] = v
    except Exception:  # noqa: BLE001
        pass

    eigenvalues: dict[str, float] = {}
    explained_variance: dict[str, float] = {}
    explained_variance_ratio: dict[str, float] = {}
    cumulative_variance: dict[str, float] = {}
    # ``get_factor_variance()`` → tuple (variance, proportional,
    # cumulative), each an ndarray of length n_factors.
    try:
        var, prop, cum = fit.get_factor_variance()
        var = np.asarray(var)
        prop = np.asarray(prop)
        cum = np.asarray(cum)
        for j, lab in enumerate(fac_labels):
            v = float(var[j])
            if math.isfinite(v):
                eigenvalues[lab] = v
                explained_variance[lab] = v
            p = float(prop[j])
            if math.isfinite(p):
                explained_variance_ratio[lab] = p
            c = float(cum[j])
            if math.isfinite(c):
                cumulative_variance[lab] = c
    except Exception:  # noqa: BLE001
        pass

    fields: dict[str, Any] = {
        "type": "factor_decomposition",
        "method": method,
        "rotation": rotation,
        "n_variables": n_feat,
        "n_components": n_factors,
        "variables": variables,
        "components": fac_labels,
        "loadings": loadings,
    }
    if n_observations is not None:
        fields["n_observations"] = int(n_observations)
    if communalities:            fields["communalities"] = communalities
    if uniqueness:               fields["uniqueness"] = uniqueness
    if eigenvalues:              fields["eigenvalues"] = eigenvalues
    if explained_variance:       fields["explained_variance"] = explained_variance
    if explained_variance_ratio: fields["explained_variance_ratio"] = explained_variance_ratio
    if cumulative_variance:      fields["cumulative_variance"] = cumulative_variance

    # Goodness-of-fit scalars when available. factor_analyzer 0.4+
    # exposes a ``sufficiency`` test that ships chi² + p; not
    # universally present so probe quietly.
    suf_fn = _safe_attr(fit, "sufficiency")
    if callable(suf_fn):
        try:
            chi2, dof, pval = suf_fn(n_observations or 0)
            if math.isfinite(float(chi2)):
                fields["chi_squared"] = float(chi2)
            if math.isfinite(float(pval)):
                fields["chi_squared_p_value"] = float(pval)
            if int(dof) > 0:
                fields["degrees_of_freedom"] = int(dof)
        except Exception:  # noqa: BLE001
            pass

    if label is not None:
        fields["label"] = str(label)
    fields.update(extra)
    result(**fields)


def from_callaway_santanna(
    attgt: Any,
    fit_result: Any | None = None,
    *,
    outcome_variable: str | None = None,
    treatment_variable: str | None = None,
    aggregation_method: str = "event",
    label: str | None = None,
    **extra: Any,
) -> None:
    """Emit a ``did_event_study`` payload from a Callaway-Sant'Anna
    fit produced by the ``differences`` package.

    Two-argument form keeps the ATTgt config (cohort column, data,
    anticipation, base_period) reachable alongside the per-(g, t)
    estimates the fitted result carries:

        from differences import ATTgt
        attgt = ATTgt(data=df.set_index(["id","period"]),
                      cohort_column="G", base_period="varying",
                      anticipation=0)
        result = attgt.fit(formula="y", control_group="never_treated")
        nora.from_callaway_santanna(attgt, result,
                                    outcome_variable="y",
                                    treatment_variable="G",
                                    label="headline DiD")

    If ``result`` is omitted, ``attgt`` is assumed to BE the fitted
    result (``differences`` happens to allow this fluent shape too).
    The helper pivots ATT(cohort, base_period, time) → ATT(cohort,
    event_time), pulls per-cohort treated counts from ``attgt.data``,
    and reads the aggregate ATT from
    ``result.aggregate(type_of_aggregation="simple")``.

    ``estimator`` is hard-coded to ``"callaway_santanna"`` for this
    helper; Sun-Abraham / de Chaisemartin land under their own
    helpers when those ship.
    """
    try:
        import numpy as np
        import pandas as pd
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "nora.from_callaway_santanna requires numpy + pandas"
        ) from e

    # If only one argument was passed, treat it as the fitted result
    # and try to recover the ATTgt config from it. Otherwise the
    # caller must pass both.
    if fit_result is None:
        fit_result = attgt
        attgt = None

    # Per-cell ATT(g, t) table. differences exposes a multi-indexed
    # DataFrame via ``result.to_pandas()`` with index levels
    # (cohort, base_period, time). The cell value lives under the
    # column tuple ('ATTgtElements', '', 'ATT'); SE under
    # ('ATTgtElements', 'analytic', 'std_error'). The pointwise
    # band columns are ('ATTgtElements', 'pointwise conf. band',
    # 'lower' | 'upper').
    if not hasattr(fit_result, "to_pandas"):
        raise TypeError(
            "nora.from_callaway_santanna: ``fit_result`` must be an "
            "ATTgtResult (returned by differences.ATTgt.fit(...))"
        )
    table = fit_result.to_pandas()
    # Robust column lookup — names rolled through the package version
    # could shift; match by trailing element.
    def _col(suffix: str) -> Any:
        for c in table.columns:
            if isinstance(c, tuple) and c[-1] == suffix:
                return c
        return None
    col_att = _col("ATT")
    col_se  = _col("std_error")
    col_lo  = _col("lower")
    col_hi  = _col("upper")
    if col_att is None:
        raise RuntimeError(
            "nora.from_callaway_santanna: ``ATT`` column not found in "
            "result.to_pandas() — package version mismatch?"
        )

    # Pivot to {cohort: {event_time: value}}. Index is
    # (cohort, base_period, time); event_time = time - cohort.
    att_dict: dict[str, dict[str, float]] = {}
    se_dict: dict[str, dict[str, float]] = {}
    ci_lo_dict: dict[str, dict[str, float]] = {}
    ci_hi_dict: dict[str, dict[str, float]] = {}
    cohorts_seen: set[int | float] = set()
    event_times_seen: set[int] = set()
    for idx, row in table.iterrows():
        if not isinstance(idx, tuple) or len(idx) < 3:
            continue
        cohort = idx[0]
        time   = idx[-1]
        try:
            e = int(time) - int(cohort)
        except (TypeError, ValueError):
            continue
        # The same (cohort, event_time) can appear under multiple
        # base periods when ``base_period="varying"``. Keep the
        # last one (the immediately-pre-treatment base period —
        # the conventional CS reporting).
        c_lab = str(int(cohort) if float(cohort).is_integer() else cohort)
        e_lab = str(e)
        cohorts_seen.add(c_lab)
        event_times_seen.add(e)
        att_dict.setdefault(c_lab, {})[e_lab] = float(row[col_att])
        if col_se is not None:
            se_dict.setdefault(c_lab, {})[e_lab] = float(row[col_se])
        if col_lo is not None:
            ci_lo_dict.setdefault(c_lab, {})[e_lab] = float(row[col_lo])
        if col_hi is not None:
            ci_hi_dict.setdefault(c_lab, {})[e_lab] = float(row[col_hi])

    # Per-cohort treated counts — number of distinct entity ids per
    # cohort in the input panel. ``attgt.data`` is the indexed
    # dataframe the fit was built from; entity id is the first level.
    n_treated_per_group: dict[str, int] = {}
    if attgt is not None and hasattr(attgt, "data") and hasattr(attgt, "cohort_column"):
        try:
            ent_name = attgt.data.index.names[0]
            sizes = (
                attgt.data.reset_index()
                .drop_duplicates(subset=[ent_name])
                .groupby(attgt.cohort_column)[ent_name].nunique()
            )
            for k, v in sizes.items():
                if pd.isna(k):
                    continue
                kk = str(int(k) if float(k).is_integer() else k)
                if kk in cohorts_seen:
                    n_treated_per_group[kk] = int(v)
        except Exception:  # noqa: BLE001
            pass

    fields: dict[str, Any] = {
        "type": "did_event_study",
        "estimator": "callaway_santanna",
        "groups": sorted(cohorts_seen, key=lambda x: float(x)),
        "event_times": sorted(event_times_seen),
        "att": att_dict,
        "standard_errors": se_dict,
        "ci_lower": ci_lo_dict,
        "ci_upper": ci_hi_dict,
        "n_treated_per_group": n_treated_per_group,
        "aggregation_method": str(aggregation_method),
    }
    if outcome_variable is not None:
        fields["outcome_variable"] = str(outcome_variable)
    if treatment_variable is not None:
        fields["treatment_variable"] = str(treatment_variable)
    if label is not None:
        fields["label"] = str(label)

    # Pass through CS configuration so the model knows the
    # identification assumptions the estimator ran under.
    if attgt is not None:
        bp = _safe_attr(attgt, "base_period_type")
        if isinstance(bp, str):
            fields["base_period"] = bp
        ant = _safe_attr(attgt, "anticipation")
        if isinstance(ant, int):
            fields["anticipation_periods"] = ant
    # control_group lives on ``attgt.estimation_details()`` (a method
    # returning a dict, populated after fit). Older versions exposed
    # it as an attribute; probe both shapes.
    if attgt is not None:
        det = _safe_attr(attgt, "estimation_details")
        if callable(det):
            try:
                det = det()
            except Exception:  # noqa: BLE001
                det = None
        if isinstance(det, dict):
            cg = det.get("control_group")
            if isinstance(cg, str):
                fields["comparison_group"] = cg

    # Aggregate scalars via ``aggregate(type_of_aggregation="simple")``.
    try:
        simple = fit_result.aggregate(type_of_aggregation="simple")
        # The aggregate is a DataFrame with one row.
        simple_pd = (
            simple.to_pandas() if hasattr(simple, "to_pandas") else simple
        )
        for c in simple_pd.columns:
            tail = c[-1] if isinstance(c, tuple) else c
            if tail == "ATT":
                fields["aggregate_att"] = float(simple_pd[c].iloc[0])
            elif tail == "std_error":
                v = float(simple_pd[c].iloc[0])
                fields["aggregate_se"] = v
                if "aggregate_att" in fields and v > 0:
                    z = abs(fields["aggregate_att"] / v)
                    fields["aggregate_p_value"] = float(
                        math.erfc(z / math.sqrt(2.0))
                    )
            elif tail == "lower":
                fields["aggregate_ci_lower"] = float(simple_pd[c].iloc[0])
            elif tail == "upper":
                fields["aggregate_ci_upper"] = float(simple_pd[c].iloc[0])
    except Exception:  # noqa: BLE001
        pass

    fields.update(extra)
    result(**fields)


def from_sun_abraham(
    fit: Any,
    n_treated: int,
    *,
    outcome_variable: str | None = None,
    treatment_variable: str | None = None,
    label: str | None = None,
    **extra: Any,
) -> None:
    """Emit a ``did_event_study`` payload from a Sun-Abraham
    interaction-weighted (IW) event study fit produced by
    ``pyfixest.event_study(..., estimator="saturated")``.

    The Sun-Abraham (2021) IW estimator solves the bias TWFE event
    studies pick up under treatment-effect heterogeneity. pyfixest's
    ``event_study`` with ``estimator="saturated"`` produces a
    cohort-saturated fit and binds an ``aggregate(agg, weighting)``
    method onto the returned Feols object that collapses the
    cohort × event-time grid to per-period IW estimates (the
    Sun-Abraham aggregate).

    Like the R helper, this emits a single synthetic cohort
    ``"all"`` because the aggregation happens inside the estimator;
    the model sees one ATT per event-time. ``n_treated`` is the
    total count of treated units across all cohorts — required,
    because the cohort-N gate has no input without it.

    Stata's ``eventstudyinteract`` is the SSC port of Sun-Abraham;
    a dedicated Stata helper is deferred (same SSC-auth + maintenance-
    lag posture as ``csdid``). For Stata-side Sun-Abraham today,
    emit via ``nora.result(type="did_event_study",
    estimator="sun_abraham", ...)``.

    Example:
        import pyfixest as pf
        # cohort variable ``g``; never-treated coded as a far-future
        # value (e.g. 10000) or as 0 per the package convention.
        fit = pf.event_study(
            df, yname="y", idname="id", tname="period", gname="g",
            estimator="saturated",
        )
        n_t = df.loc[df["g"] > 0, "id"].nunique()
        nora.from_sun_abraham(fit, n_treated=n_t,
                              outcome_variable="y",
                              treatment_variable="g",
                              label="Sun-Abraham IW")
    """
    if not isinstance(n_treated, int) or n_treated < 0:
        raise ValueError(
            "nora.from_sun_abraham: ``n_treated`` (total treated units) "
            "is required and must be a non-negative int"
        )
    aggregate_fn = getattr(fit, "aggregate", None)
    if not callable(aggregate_fn):
        raise TypeError(
            "nora.from_sun_abraham: ``fit`` must expose an "
            "``aggregate`` method (pyfixest saturated event study "
            "shape). Run ``pyfixest.event_study(..., "
            "estimator='saturated')`` first."
        )
    method_attr = getattr(fit, "_method", None)
    if isinstance(method_attr, str) and method_attr not in (
        "saturated", "sun_abraham"
    ):
        raise TypeError(
            "nora.from_sun_abraham: ``fit._method`` is "
            f"{method_attr!r} — expected 'saturated' (Sun-Abraham). "
            "Did you mean ``from_twfe_event_study`` for the TWFE fit?"
        )
    try:
        print(fit)
    except Exception:  # noqa: BLE001
        pass

    try:
        agg_df = aggregate_fn(agg="period", weighting="shares")
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "nora.from_sun_abraham: fit.aggregate(agg='period', "
            f"weighting='shares') failed: {e}"
        ) from e
    if agg_df is None or not hasattr(agg_df, "iterrows"):
        raise RuntimeError(
            "nora.from_sun_abraham: aggregate() did not return a "
            "DataFrame — pyfixest version mismatch?"
        )

    cn = list(agg_df.columns)

    def _pick(candidates: tuple[str, ...]) -> str | None:
        for c in candidates:
            if c in cn:
                return c
        return None

    est_col = _pick(("Estimate", "estimate"))
    se_col  = _pick(("Std. Error", "std_error", "std.error", "se"))
    p_col   = _pick(("Pr(>|t|)", "Pr(>|z|)", "p_value", "p.value"))
    lo_col  = _pick(("2.5%", "conf_low", "conf.low"))
    hi_col  = _pick(("97.5%", "conf_high", "conf.high"))
    if est_col is None:
        raise RuntimeError(
            "nora.from_sun_abraham: aggregate() output missing an "
            "Estimate column — pyfixest version mismatch?"
        )

    att_all: dict[str, float] = {}
    se_all:  dict[str, float] = {}
    p_all:   dict[str, float] = {}
    ci_lo:   dict[str, float] = {}
    ci_hi:   dict[str, float] = {}
    event_times: list[int] = []
    for period_label, row in agg_df.iterrows():
        try:
            et = int(period_label)
        except (TypeError, ValueError):
            continue
        est_raw = row[est_col]
        try:
            est = float(est_raw)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(est):
            continue
        lab = str(et)
        att_all[lab] = est
        event_times.append(et)
        if se_col is not None:
            try:
                se = float(row[se_col])
            except (TypeError, ValueError):
                se = float("nan")
            if math.isfinite(se):
                se_all[lab] = se
        if p_col is not None:
            try:
                pv = float(row[p_col])
            except (TypeError, ValueError):
                pv = float("nan")
            if math.isfinite(pv):
                p_all[lab] = pv
        if lo_col is not None:
            try:
                lo = float(row[lo_col])
            except (TypeError, ValueError):
                lo = float("nan")
            if math.isfinite(lo):
                ci_lo[lab] = lo
        if hi_col is not None:
            try:
                hi = float(row[hi_col])
            except (TypeError, ValueError):
                hi = float("nan")
            if math.isfinite(hi):
                ci_hi[lab] = hi
        # Synthesize ±1.96 SE CI when pyfixest didn't ship explicit
        # CI columns and SE is present.
        if lab not in ci_lo and lab in se_all:
            ci_lo[lab] = est - 1.96 * se_all[lab]
        if lab not in ci_hi and lab in se_all:
            ci_hi[lab] = est + 1.96 * se_all[lab]

    if not event_times:
        raise RuntimeError(
            "nora.from_sun_abraham: aggregate() returned no rows with "
            "integer period labels"
        )

    fields: dict[str, Any] = {
        "type": "did_event_study",
        "estimator": "sun_abraham",
        "aggregation_method": "dynamic",
        "groups": ["all"],
        "event_times": sorted(event_times),
        "att": {"all": att_all},
        "standard_errors": {"all": se_all},
        "p_values": {"all": p_all},
        "ci_lower": {"all": ci_lo},
        "ci_upper": {"all": ci_hi},
        "n_treated_per_group": {"all": int(n_treated)},
    }
    if outcome_variable is not None:
        fields["outcome_variable"] = str(outcome_variable)
    if treatment_variable is not None:
        fields["treatment_variable"] = str(treatment_variable)
    if label is not None:
        fields["label"] = str(label)
    fields.update(extra)
    result(**fields)


def from_rdd(
    fit: Any,
    *,
    running_variable: str | None = None,
    outcome_variable: str | None = None,
    fuzzy_treatment_variable: str | None = None,
    first_stage_f: float | None = None,
    label: str | None = None,
    **extra: Any,
) -> None:
    """Emit an ``rdd`` payload from an ``rdrobust`` fit.

    Wraps the rdrobust Python package (Calonico-Cattaneo-Titiunik
    2014) — the standard cross-language implementation. The payload
    carries the three-flavor τ table, bandwidth(s), kernel,
    polynomial order, effective N per side, and the bandwidth
    selector. For fuzzy RDD pass ``fuzzy_treatment_variable``; the
    estimator is tagged ``fuzzy_2sls`` and the caller may also
    supply ``first_stage_f``.

    Privacy carve-out is structural. The helper signature does not
    accept density / binscatter / mccrary keyword arguments — passing
    one raises. McCrary density and binscatter near the cutoff are
    visual diagnostics for the researcher; they have no field on
    the ``rdd`` shape's allowlist so even hand-crafted payloads
    through ``nora.result(type="rdd", ...)`` cannot smuggle them.

    Example:
        from rdrobust import rdrobust
        m = rdrobust(y=df["voted"], x=df["income"], c=50000)
        nora.from_rdd(m, running_variable="income",
                     outcome_variable="voted", label="headline RDD")

        # Fuzzy:
        m = rdrobust(y=df["voted"], x=df["income"], c=50000,
                     fuzzy=df["takeup"])
        nora.from_rdd(m, running_variable="income",
                     outcome_variable="voted",
                     fuzzy_treatment_variable="takeup",
                     first_stage_f=24.3)
    """
    # Privacy carve-out: reject density/binscatter kwargs that a
    # script might try to slip through ``**extra``.
    banned = (
        "mccrary_density_curve", "mccrary_density",
        "binscatter_bins", "binscatter", "density_curve",
    )
    for b in banned:
        if b in extra:
            raise ValueError(
                f"nora.from_rdd: ``{b}`` is a visual diagnostic for the "
                f"researcher and is not allowed on the rdd payload. The "
                f"model sees the analytical fields (tau / bandwidth / "
                f"effective N); ask the researcher qualitatively about "
                f"manipulation evidence if it bears on the design."
            )
    # rdrobust's result class is rdrobust_output. Duck-type rather
    # than importing rdrobust here (keeps the runtime import-light
    # for non-RDD scripts).
    cls = type(fit).__name__
    if cls != "rdrobust_output":
        raise TypeError(
            "nora.from_rdd: ``fit`` must be an rdrobust_output (returned "
            "by rdrobust.rdrobust(y, x, c=cutoff))."
        )
    try:
        print(fit)
    except Exception:  # noqa: BLE001
        pass

    fields: dict[str, Any] = {
        "type": "rdd",
        "estimator": (
            "fuzzy_2sls" if fuzzy_treatment_variable is not None
            else "local_polynomial"
        ),
    }
    if running_variable is not None:
        fields["running_variable"] = str(running_variable)
    if outcome_variable is not None:
        fields["outcome_variable"] = str(outcome_variable)
    if label is not None:
        fields["label"] = str(label)

    # Pull the three-flavor row indexes from the DataFrame-shaped
    # outputs. Python rdrobust uses the same row labels as R
    # (Conventional / Bias-Corrected / Robust).
    def _row(df: Any, label: str) -> float | None:
        try:
            v = df.loc[label].iloc[0]
            f = float(v)
            return f if math.isfinite(f) else None
        except Exception:  # noqa: BLE001
            return None

    coef = _safe_attr(fit, "coef")
    se   = _safe_attr(fit, "se")
    pv   = _safe_attr(fit, "pv")
    ci   = _safe_attr(fit, "ci")
    if coef is not None:
        for flavor, slot in (
            ("Conventional", "tau_conventional"),
            ("Bias-Corrected", "tau_bias_corrected"),
            ("Robust", "tau_robust"),
        ):
            v = _row(coef, flavor)
            if v is not None:
                fields[slot] = v
    if se is not None:
        for flavor, slot in (
            ("Conventional", "se_conventional"),
            ("Bias-Corrected", "se_bias_corrected"),
            ("Robust", "se_robust"),
        ):
            v = _row(se, flavor)
            if v is not None:
                fields[slot] = v
    if pv is not None:
        for flavor, slot in (
            ("Conventional", "p_conventional"),
            ("Bias-Corrected", "p_bias_corrected"),
            ("Robust", "p_robust"),
        ):
            v = _row(pv, flavor)
            if v is not None:
                fields[slot] = v
    if ci is not None:
        for flavor, lo_slot, hi_slot in (
            ("Conventional", "ci_lower_conventional", "ci_upper_conventional"),
            ("Bias-Corrected", "ci_lower_bias_corrected", "ci_upper_bias_corrected"),
            ("Robust", "ci_lower_robust", "ci_upper_robust"),
        ):
            try:
                lo = float(ci.loc[flavor].iloc[0])
                hi = float(ci.loc[flavor].iloc[1])
                if math.isfinite(lo): fields[lo_slot] = lo
                if math.isfinite(hi): fields[hi_slot] = hi
            except Exception:  # noqa: BLE001
                pass

    # Bandwidths: rdrobust stores ``h`` (main) and ``b`` (bias-
    # correction) as a DataFrame indexed by ["h", "b"] with columns
    # ["left", "right"].
    bws = _safe_attr(fit, "bws")
    if bws is not None:
        try:
            fields["bandwidth_left"]  = float(bws.loc["h", "left"])
            fields["bandwidth_right"] = float(bws.loc["h", "right"])
        except Exception:  # noqa: BLE001
            pass
        try:
            fields["bandwidth_bias_correction_left"]  = float(bws.loc["b", "left"])
            fields["bandwidth_bias_correction_right"] = float(bws.loc["b", "right"])
        except Exception:  # noqa: BLE001
            pass

    # Effective N inside the main bandwidth: ``N_h`` is a list /
    # array of [left, right].
    n_h = _safe_attr(fit, "N_h")
    if n_h is not None:
        try:
            fields["effective_n_left"]  = int(n_h[0])
            fields["effective_n_right"] = int(n_h[1])
        except Exception:  # noqa: BLE001
            pass

    p_order = _safe_attr(fit, "p")
    if p_order is not None:
        try:
            fields["polynomial_order"] = int(p_order)
        except (TypeError, ValueError):
            pass
    cutoff = _safe_attr(fit, "c")
    if cutoff is not None:
        cf = _safe_float(cutoff)
        if cf is not None:
            fields["cutoff"] = cf

    bwsel = _safe_attr(fit, "bwselect")
    if isinstance(bwsel, str):
        fields["bandwidth_selector"] = bwsel
    kernel = _safe_attr(fit, "kernel")
    if isinstance(kernel, str):
        # rdrobust reports kernel capitalized ("Triangular"); the
        # sanitizer accepts lowercase only.
        fields["kernel"] = kernel.lower()

    fs_f = _safe_float(first_stage_f)
    if fs_f is not None:
        fields["first_stage_f"] = fs_f

    fields.update(extra)
    result(**fields)


def from_kaplan_meier(
    fit: Any,
    horizons: dict[str, float] | None = None,
    *,
    time_variable: str | None = None,
    event_variable: str | None = None,
    group_variable: str | None = None,
    logrank_chi_squared: float | None = None,
    logrank_p_value: float | None = None,
    n_groups: int | None = None,
    label: str | None = None,
    **extra: Any,
) -> None:
    """Emit a ``kaplan_meier`` payload from a fitted survival curve.

    Duck-typed on the attributes statsmodels' ``SurvfuncRight``
    exposes; lifelines' ``KaplanMeierFitter`` works via the same
    shape if the caller wraps it (rare in practice — most lifelines
    users will already have ``KaplanMeierFitter.survival_function_``
    and can pass a thin adapter).

    Required attributes on ``fit``:
        time         — 1-D array of observed times (the input
                       duration vector)
        status       — 1-D event indicator (1 = event, 0 = censored)
        surv_times   — event-time array (post-fit, sorted)
        surv_prob    — survival probability array, same length
                       as ``surv_times``
    Optional:
        surv_prob_se — Greenwood SE per event time (enables CIs)
        quantile     — callable for median (``fit.quantile(0.5)``)
        quantile_ci  — callable for CI (``fit.quantile_ci(0.5)``)

    ``horizons`` maps canonical labels (``"1y"`` / ``"3y"`` /
    ``"5y"`` / ``"10y"`` — the only labels the sanitizer's
    kaplan_meier shape accepts) to numeric time values in whatever
    unit the fit was built in. The helper interpolates S(h) using
    a step-look-up (KM is a step function) and computes n_at_risk(h)
    from the original duration vector ``fit.time``.

    Log-rank inference across groups isn't computed by statsmodels'
    ``SurvfuncRight``. The caller computes it (manually or via
    lifelines / R) and passes the chi² + p-value as kwargs.
    """
    try:
        print(fit.summary() if callable(getattr(fit, "summary", None)) else fit)
    except Exception:  # noqa: BLE001
        pass

    try:
        import numpy as np
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "nora.from_kaplan_meier requires numpy in the runtime"
        ) from e

    time_arr = _safe_attr(fit, "time")
    status_arr = _safe_attr(fit, "status")
    surv_times = _safe_attr(fit, "surv_times")
    surv_prob = _safe_attr(fit, "surv_prob")
    if surv_times is None or surv_prob is None:
        raise TypeError(
            "nora.from_kaplan_meier: ``fit`` must expose ``surv_times`` "
            "and ``surv_prob`` (statsmodels.SurvfuncRight shape)"
        )

    surv_times = np.asarray(surv_times, dtype=float)
    surv_prob  = np.asarray(surv_prob,  dtype=float)
    surv_prob_se = _safe_attr(fit, "surv_prob_se")
    if surv_prob_se is not None:
        surv_prob_se = np.asarray(surv_prob_se, dtype=float)

    fields: dict[str, Any] = {}
    if time_variable is not None:   fields["time_variable"]  = str(time_variable)
    if event_variable is not None:  fields["event_variable"] = str(event_variable)
    if group_variable is not None:  fields["group_variable"] = str(group_variable)
    if label is not None:           fields["label"]          = str(label)

    if time_arr is not None and status_arr is not None:
        time_arr   = np.asarray(time_arr,   dtype=float)
        status_arr = np.asarray(status_arr, dtype=int)
        fields["n_subjects"] = int(time_arr.size)
        fields["n_failures"] = int((status_arr != 0).sum())
    else:
        # Fall back to derived counts when raw inputs aren't exposed
        # (e.g., a thin adapter that only ships the curve). Total
        # events = sum of n_events; total at-risk = n_risk at t=0.
        n_events = _safe_attr(fit, "n_events")
        n_risk   = _safe_attr(fit, "n_risk")
        if n_events is not None and n_risk is not None:
            n_events = np.asarray(n_events, dtype=int)
            n_risk   = np.asarray(n_risk,   dtype=int)
            if n_risk.size > 0:
                fields["n_subjects"] = int(n_risk[0])
                fields["n_failures"] = int(n_events.sum())

    # Median + CI. ``SurvfuncRight.quantile(0.5)`` raises on heavily-
    # censored curves where the median is undefined; absorb that.
    qfn = _safe_attr(fit, "quantile")
    if callable(qfn):
        try:
            med = float(qfn(0.5))
            if math.isfinite(med):
                fields["median_survival_time"] = med
        except Exception:  # noqa: BLE001
            pass
    qcifn = _safe_attr(fit, "quantile_ci")
    if callable(qcifn):
        try:
            lo, hi = qcifn(0.5)
            if lo is not None and math.isfinite(float(lo)):
                fields["median_survival_ci_lower"] = float(lo)
            if hi is not None and math.isfinite(float(hi)):
                fields["median_survival_ci_upper"] = float(hi)
        except Exception:  # noqa: BLE001
            pass

    # Per-horizon scalars. KM is a step function — S(h) is the
    # survival probability at the latest event time ≤ h. n_at_risk(h)
    # is the count of subjects whose observed time ≥ h, computed from
    # the original duration array (the SurvfuncRight's ``n_risk``
    # attribute is at-event-time, not at-arbitrary-horizon).
    if horizons:
        z_975 = 1.959963984540054  # 97.5th percentile of N(0,1) for 95% CI
        for label_str, h_time in horizons.items():
            if not isinstance(label_str, str):
                continue
            try:
                h = float(h_time)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(h):
                continue
            # Step look-up: largest event time ≤ h.
            idx = int(np.searchsorted(surv_times, h, side="right")) - 1
            if idx < 0:
                # h precedes the first event time — S(h) = 1.
                s_h = 1.0
                se_h: float | None = 0.0
            elif idx >= surv_prob.size:
                idx = surv_prob.size - 1
                s_h = float(surv_prob[idx])
                se_h = (
                    float(surv_prob_se[idx])
                    if surv_prob_se is not None and idx < surv_prob_se.size
                    else None
                )
            else:
                s_h = float(surv_prob[idx])
                se_h = (
                    float(surv_prob_se[idx])
                    if surv_prob_se is not None and idx < surv_prob_se.size
                    else None
                )
            if math.isfinite(s_h):
                fields[f"survival_at_{label_str}"] = s_h
            # n_at_risk from raw durations when accessible.
            if time_arr is not None:
                n_risk_h = int((time_arr >= h).sum())
                fields[f"n_at_risk_{label_str}"] = n_risk_h
            # Linear Greenwood CI (clamped to [0, 1]).
            if se_h is not None and math.isfinite(se_h) and math.isfinite(s_h):
                lo = max(0.0, s_h - z_975 * se_h)
                hi = min(1.0, s_h + z_975 * se_h)
                fields[f"survival_at_{label_str}_ci_lower"] = lo
                fields[f"survival_at_{label_str}_ci_upper"] = hi

    for k, v in (
        ("logrank_chi_squared", logrank_chi_squared),
        ("logrank_p_value", logrank_p_value),
    ):
        vf = _safe_float(v)
        if vf is not None:
            fields[k] = vf
    if isinstance(n_groups, int) and n_groups > 0:
        fields["n_groups"] = n_groups

    fields.update(extra)
    result(type="kaplan_meier", **fields)


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

    # ``_safe_int`` instead of bare ``int(...)``: ``int(float('nan'))``
    # raises ``ValueError`` and crashes the helper before any payload
    # reaches disk. NaN sample sizes are unusual (you don't usually
    # have NaN counts of observations) but a careless caller passing
    # ``len(df_with_nans_in_index)`` or similar shouldn't lose the
    # entire result. Coerce to None and let the sanitizer reject the
    # field with a clear reason instead.
    fields: dict[str, Any] = {
        "test_type": test_type,
        "t_statistic": statistic,
        "p_value": pvalue,
        "degrees_of_freedom": df,
        "n1": _safe_int(n1),
        "mean1": mean1,
    }
    if n2 is not None:
        fields["n2"] = _safe_int(n2)
    if mean2 is not None:
        fields["mean2"] = mean2
    fields.update(extra)
    result(type="t_test", **fields)


def from_summarize(variable: str, *, n: int, mean: float, sd: float,
                   missing_count: int = 0,
                   **extra: Any) -> None:
    """Emit a ``descriptive`` payload for a single numeric variable.
    Mirrors ``nora$from_summarize`` in the R library.

    ``min_value`` / ``max_value`` are no longer accepted: the
    sanitizer drops them in every payload because nothing in the
    payload binds the reported values to the named variable's actual
    column. Researchers who need a variable's range should use a
    Nora-owned path (request_data with a future bounds extension)
    rather than a script-emitted descriptive.
    """
    # ``_safe_int`` instead of bare ``int(...)``: avoid crashing the
    # whole helper on a NaN count that a careless caller forwarded
    # from a partial aggregation. The sanitizer will reject ``None``
    # integer fields with a clear reason; that's strictly better than
    # losing the entire result to a ``ValueError``.
    fields: dict[str, Any] = {
        "variable": variable,
        "n": _safe_int(n),
        "mean": _safe_float(mean),
        "sd": _safe_float(sd),
        "missing_count": _safe_int(missing_count),
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
    # Filter out non-finite values up front. ``int(float('nan'))``
    # raises ``ValueError`` and would crash the helper before the
    # payload reaches disk — pandas value_counts() doesn't usually
    # produce NaN cells, but a hand-built counts dict or a
    # ``pd.crosstab`` with all-missing combinations can. Drop the
    # NaN levels rather than coercing to 0 (which would lie about
    # observed-zero vs. unobserved).
    clean_counts: dict[str, int] = {}
    for k, v in counts_dict.items():
        iv = _safe_int(v)
        if iv is not None:
            clean_counts[str(k)] = iv
    # Default: auto-compute n from the clean counts. Branch on caller-
    # supplied n so a NaN / non-finite caller value flows through
    # ``_safe_int`` and lands as ``None`` (which the sanitizer
    # rejects), rather than getting silently coerced to 0 by
    # ``or 0``. ``n=0`` and a malformed/non-finite n both used to
    # serialize as ``"n": 0`` — a valid-looking sanitizer payload
    # that hid the upstream undefined-count problem.
    if n is None:
        safe_n: int | None = sum(clean_counts.values())
    else:
        safe_n = _safe_int(n)
    fields = {
        "variable": variable,
        "counts": clean_counts,
        "n": safe_n,
        "missing_count": _safe_int(missing_count),
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

    # NaN-tolerant cell coercion. ``pd.crosstab`` can produce NaN
    # cells when ``dropna=False`` leaves un-observed row/col
    # combinations, and bare ``int(float('nan'))`` raises
    # ``ValueError`` and crashes the helper before any payload
    # reaches disk. Drop NaN cells silently — "the cell was never
    # observed" is structurally different from "the cell has 0
    # observations" and conflating them via ``or 0`` would lie.
    def _cell(v: Any) -> int | None:
        return _safe_int(v)

    if pd is not None and isinstance(table, pd.DataFrame):
        counts: dict[str, dict[str, int]] = {}
        for row_label, row in table.iterrows():
            row_dict: dict[str, int] = {}
            for col in table.columns:
                iv = _cell(row[col])
                if iv is not None:
                    row_dict[str(col)] = iv
            counts[str(row_label)] = row_dict
        if row_variable is None:
            row_variable = str(table.index.name or "row")
        if col_variable is None:
            col_variable = str(table.columns.name or "column")
    else:
        # Caller passed a pre-built nested dict.
        counts = {}
        for rk, rv in dict(table).items():
            row_dict = {}
            for ck, cv in (rv or {}).items():
                iv = _cell(cv)
                if iv is not None:
                    row_dict[str(ck)] = iv
            counts[str(rk)] = row_dict
        row_variable = row_variable or "row"
        col_variable = col_variable or "column"

    # ``missing_count`` rides as ``None`` when the caller passed
    # a NaN / non-finite value so the sanitizer rejects it. The
    # previous ``or 0`` silently coerced bad inputs to a
    # valid-looking ``0``, hiding the upstream undefined-count
    # problem from disclosure-control review.
    fields = {
        "row_variable": row_variable,
        "col_variable": col_variable,
        "counts": counts,
        "missing_count": _safe_int(missing_count),
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
    # Reject ``**extra`` keys that would override fields the helper
    # computes from raw data. Without this guard, a caller could pass
    # ``cells={...}`` (or ``row_variable=...`` etc.) and the
    # ``fields.update(extra)`` below would clobber the helper's
    # computation. The ``_via_helper`` marker stamped at write time
    # would then authenticate attacker-supplied values, and the
    # sanitizer (which trusts the marker to skip recomputing
    # ``max_share``) would let a forged ``max_share=0`` bypass the
    # dominance gate. The marker is meant to prove the disclosure-
    # metric fields came from the helper, not just that the helper
    # was called.
    _reserved = {
        "type", "row_variable", "value_variable", "aggregation",
        "cells", "_via_helper",
    }
    forbidden = sorted(set(extra) & _reserved)
    if forbidden:
        raise ValueError(
            "from_magnitude_table: cannot override helper-computed "
            f"fields via keyword arguments: {forbidden}. These are "
            "computed from the DataFrame and bound to the "
            "_via_helper provenance marker."
        )
    fields = {
        "row_variable": group_var,
        "value_variable": value_var,
        "aggregation": aggregation,
        "cells": cells,
    }
    fields.update(extra)
    # Helper-provenance marker. The sanitizer requires this for
    # ``magnitude_table`` because cell-level ``max_share`` is
    # consulted-only and stripped; without proof that max_share
    # came from raw-data computation a malicious script could
    # publish a dominance-violating value with a forged
    # ``max_share=0`` and skip the dominance gate. Write directly
    # via _write_result, bypassing ``result()`` (which strips this
    # field from caller fields), so the marker can't be forged
    # through the generic API.
    payload = {
        "type": "magnitude_table",
        **fields,
        "_via_helper": "from_magnitude_table",
    }
    _write_result(payload)


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


def _to_dict(thing: Any, names: list[str] | None = None) -> dict[str, Any]:
    """Normalise a pandas Series / dict-like / ndarray to a plain
    ``{name: value}`` dict the encoder can serialise without further
    coercion.

    ``names`` is used as a fallback labelling when ``thing`` doesn't
    carry its own index — statsmodels ``PHRegResults`` ships params /
    bse / tvalues / pvalues as bare ndarrays without column names, so
    we pair them with the inner model's ``exog_names``. Without this,
    ``dict(ndarray)`` raises ``TypeError: cannot convert dictionary
    update sequence element #0 to a sequence`` and the helper aborts
    before the payload is written — the same silent-failure mode the
    R/Stata audits caught for Cox.
    """
    if hasattr(thing, "to_dict"):
        return {str(k): v for k, v in thing.to_dict().items()}
    # Index-less iterable (ndarray, list, tuple): require a names list.
    try:
        items = list(thing)
    except TypeError:
        return {str(k): v for k, v in dict(thing).items()}
    if names is not None and len(names) == len(items):
        return {str(names[i]): items[i] for i in range(len(items))}
    # Last-resort positional naming so an unnamed numeric vector at
    # least round-trips with deterministic keys rather than raising.
    return {f"x{i}": items[i] for i in range(len(items))}


def _safe_attr(obj: Any, name: str) -> Any:
    """Attribute fetch that survives properties raising
    ``NotImplementedError`` / ``ValueError`` etc. — statsmodels
    ``IV2SLS`` results define ``llf`` / ``aic`` / ``bic`` as
    properties that raise rather than missing, so a plain
    ``getattr(obj, name, None)`` propagates and aborts the helper."""
    try:
        return getattr(obj, name, None)
    except Exception:  # noqa: BLE001
        return None


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


def _unique_plot_name(d: Any, base: str) -> str:
    """Return ``base`` if no file by that name exists in ``d``;
    otherwise append ``_2``, ``_3``, ... before the extension.

    Without this, calling ``plot_coefficients(fit1)`` then
    ``plot_coefficients(fit2)`` in one script would:
      1. write ``coefficients.png`` (fit1) + manifest entry
      2. *overwrite* ``coefficients.png`` (now fit2) + a SECOND
         manifest entry pointing at the same path
    The bridge would read two manifest rows that point to the same
    image and the model would see two "different" plots that are
    in fact fit2 twice. ``plot_interaction`` already side-steps
    this by suffixing the variable name into the filename; the
    other helpers need a counter.
    """
    from pathlib import Path
    p = Path(base)
    stem, ext = p.stem, p.suffix
    candidate = base
    i = 2
    while (d / candidate).exists():
        candidate = f"{stem}_{i}{ext}"
        i += 1
    return candidate


def _append_plot_manifest(file: str, kind: str, label: str | None) -> None:
    d = _plots_dir()
    if d is None:
        return
    # Stamp every entry with the per-run token. The executor validates
    # this field after the script finishes and drops any entry whose
    # token is missing or wrong; that strips manifest rows a script
    # could otherwise have appended directly (saving a raw-data plot
    # under ``_nora_plots/`` and labeling it ``coefficients`` to slip
    # past the disclosure-control allowlist for vision attachment).
    # Same posture as the result-payload ``_token`` field — a
    # determined script can still reach into ``nora._RUN_TOKEN`` to
    # forge the value, but doing so requires obvious code in the
    # script the researcher reviews.
    entry: dict[str, Any] = {"file": file, "kind": kind, "_token": _RUN_TOKEN}
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


def _helper_failed(
    helper: str,
    message: str,
    exc: BaseException | None = None,
) -> None:
    """Record a graceful plot-helper failure: human-readable stderr
    line for the researcher's raw log AND a structured jsonl entry
    so the model's tool result surfaces the failure cause.

    The plot helpers wrap their bodies in ``try/except Exception``,
    but they ALSO have several early-return paths for shape problems
    (no ``.params``, malformed ``models`` dict, etc.). Those paths
    used to write only stderr — the model saw "no plots produced"
    with no hint why, then guessed and looped. Calling this helper
    at each early-return keeps both audiences informed without
    forcing the caller to raise (which would also unwind the
    surrounding analysis script's own bookkeeping).
    """
    sys.stderr.write(f"nora.{helper}: {message}\n")
    _append_plot_helper_error(
        helper, exc if exc is not None else RuntimeError(message)
    )


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
            _helper_failed(
                "plot_residuals",
                "fitted object has no .resid / .fittedvalues; skipping",
            )
            return

        try:
            import numpy as _np
            resid_arr = _np.asarray(resid, dtype=float)
            fitted_arr = _np.asarray(fitted_vals, dtype=float)
        except ImportError as e:
            _helper_failed("plot_residuals", "numpy missing", exc=e)
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

        fname = _unique_plot_name(d, "residuals.png")
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
            _helper_failed(
                "plot_coefficients",
                "fitted object has no .params; need a statsmodels-style fit",
            )
            return
        try:
            ci = fitted.conf_int(alpha=0.05)
        except Exception as e:  # noqa: BLE001
            _helper_failed(
                "plot_coefficients", f"conf_int failed: {e}", exc=e,
            )
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
            _helper_failed(
                "plot_coefficients",
                "nothing to plot after dropping intercept term",
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

        fname = _unique_plot_name(d, "coefficients.png")
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
            _helper_failed(
                "plot_estimate_comparison",
                "`models` must be a dict of at least 2 fits keyed by label",
            )
            return
        if not isinstance(coef, str) or not coef:
            _helper_failed(
                "plot_estimate_comparison",
                "`coef` must be a coefficient name string",
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
                _helper_failed(
                    "plot_estimate_comparison",
                    f"model {nm!r} has no .params; need a statsmodels-style fit",
                )
                return
            try:
                idx = list(params.index)
            except AttributeError:
                idx = [str(i) for i in range(len(params))]
            if coef not in idx:
                _helper_failed(
                    "plot_estimate_comparison",
                    f"coef {coef!r} not in model {nm!r}",
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

        fname = _unique_plot_name(d, "estimate_comparison.png")
        fig.savefig(d / fname, dpi=110)
        plt.close(fig)
        _append_plot_manifest(
            fname, "estimate_comparison",
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
            _helper_failed(
                "plot_interaction",
                f"pass data=... that contains column {var!r}; "
                f"couldn't derive it from the fit",
            )
            return

        # Build the prediction grid and the held-at-means template.
        # Disclosure-control note: the rendered PNG is allowlisted for
        # model vision (kind="interaction"), so anything legible on
        # the x-axis crosses the SDC boundary. The previous version
        # used col.min()/col.max() (numeric) and col.unique()
        # (categorical), which surfaced raw extrema and rare-level
        # identities the JSON sanitizer would have refused. We now
        # build a disclosure-safe grid:
        #   - numeric: mean ± 2*sd, which discloses no more than
        #     the descriptive sanitizer's already-allowed mean+sd
        #     pair. Tick labels are stripped so the model only sees
        #     curve SHAPE, not absolute x values.
        #   - categorical: drop levels whose observed count is
        #     below the SDC cell_suppression_threshold (matches
        #     frequency_table policy: rare-level identities are
        #     themselves disclosive). If that leaves nothing, refuse
        #     the plot rather than silently dropping back to all
        #     levels.
        col = data[var]
        is_numeric = _pd.api.types.is_numeric_dtype(col)
        # Mirror SDCConfig.cell_suppression_threshold default; the
        # runtime library has no direct access to the runner-side
        # config object.
        _CELL_SUPPRESSION_THRESHOLD = 10
        # Cap how many categorical bars can land on the plot — a
        # 200-level bar chart isn't readable AND multiplies the
        # data-channel surface through tick labels.
        _CATEGORICAL_LEVEL_CAP = 20
        suppression_note: str | None = None
        if is_numeric:
            cleaned = col.dropna()
            if len(cleaned) < _CELL_SUPPRESSION_THRESHOLD:
                _helper_failed(
                    "plot_interaction",
                    f"variable {var!r} has fewer than "
                    f"{_CELL_SUPPRESSION_THRESHOLD} non-missing "
                    f"observations; below the disclosure threshold",
                )
                return
            mu = float(cleaned.mean())
            sd = float(cleaned.std(ddof=1)) if len(cleaned) > 1 else 0.0
            if not _np.isfinite(sd) or sd <= 0:
                # Constant variable or single observation — nothing
                # meaningful to plot, and reading min would expose
                # the constant value.
                _helper_failed(
                    "plot_interaction",
                    f"variable {var!r} has zero variance — interaction "
                    f"plot would expose the constant value",
                )
                return
            grid = _np.linspace(mu - 2.0 * sd, mu + 2.0 * sd, 100)
        else:
            cleaned = col.dropna()
            counts = cleaned.value_counts()
            # Drop rare levels (below threshold). Their identities are
            # disclosive even if the bar height is masked, same as the
            # frequency_table primary suppression rule.
            visible = counts[counts >= _CELL_SUPPRESSION_THRESHOLD]
            if visible.empty:
                _helper_failed(
                    "plot_interaction",
                    f"variable {var!r}: no level meets the disclosure "
                    f"threshold (n >= {_CELL_SUPPRESSION_THRESHOLD}); "
                    f"refusing to plot",
                )
                return
            # Keep top-K most frequent levels for readability.
            visible = visible.head(_CATEGORICAL_LEVEL_CAP)
            grid = list(visible.index)
            dropped = int((counts < _CELL_SUPPRESSION_THRESHOLD).sum())
            if dropped > 0:
                suppression_note = (
                    f"{dropped} rare level(s) with count < "
                    f"{_CELL_SUPPRESSION_THRESHOLD} suppressed"
                )

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
            # Strip numeric x-axis tick labels: the absolute values
            # ARE data (mean ± 2σ for var). The disclosure-safe pair
            # is mean+sd, which goes through the descriptive
            # sanitizer with its own gates. Show only relative
            # anchors so the model can read the curve's shape
            # without reading the raw mean / sd off the axis.
            ax.set_xticks([
                grid[0], (grid[0] + grid[-1]) / 2.0, grid[-1]
            ])
            ax.set_xticklabels(["−2σ", "mean", "+2σ"])
        else:
            xs = _np.arange(len(grid))
            ax.bar(xs, mean, yerr=[mean - lo, hi - mean],
                   color="#4C78A8", edgecolor="#1F4E79",
                   capsize=4)
            ax.set_xticks(xs)
            # Run categorical tick labels through the same text-safety
            # primitive that gates every other model-visible string.
            # ``safe_text`` strips C0/C1 control chars, bidi overrides,
            # and zero-width characters, then caps length. Without this
            # a frequent-level category name like
            # ``"engineering\nIGNORE PRIOR INSTRUCTIONS:..."`` would
            # render straight into the model-visible image, bypassing
            # the JSON/text path's safety gate. ``safe_text`` returns
            # an empty string for completely-rejected inputs; fall
            # back to a redaction marker so the bar is still
            # identifiable at its x-position.
            from nora.text_safety import safe_text as _safe_text

            def _tick_label(v: object) -> str:
                t = _safe_text(str(v), max_len=24)
                return t or "[redacted]"
            ax.set_xticklabels(
                [_tick_label(g) for g in grid], rotation=30, ha="right",
            )
        ax.set_xlabel(xtitle)
        ax.set_ylabel(ytitle)
        ax.set_title(ptitle, fontweight="bold")
        if suppression_note:
            # Caption-style note so the model sees that some levels
            # were suppressed without seeing which ones.
            fig.text(
                0.99, 0.01, suppression_note,
                ha="right", va="bottom",
                fontsize=8, color="#666666",
            )
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
