"""Real-fit audit of ``nora.from_lm`` (Python helper).

Mirrors ``scripts/audit_regression_bucket.R`` for statsmodels. Run
the audit script as a subprocess so ``nora.runtime.nora``'s import-
time NORA_RUN_TOKEN check sees the env var (it pops the var on
import; running inside the audit-driver process would fail the
second invocation).

The companion pytest module ``tests/test_from_lm_python_real_fits.py``
spawns this script and parses the JSONL it writes.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path


def _emit_label(result_path: str, label: str) -> None:
    """Sentinel line so the audit driver can map payloads to estimators."""
    with open(result_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"_audit_label": label}) + "\n")


def main() -> int:
    result_path = os.environ.get("NORA_RESULT_PATH")
    if not result_path:
        print("NORA_RESULT_PATH not set", file=sys.stderr)
        return 1
    Path(result_path).unlink(missing_ok=True)

    # Importing nora.runtime.nora pops NORA_RUN_TOKEN, so we have to
    # import AFTER ensuring it's set in the env. The audit driver sets
    # both env vars before launching us.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "nora" / "runtime"))
    import nora as nora_runtime  # type: ignore[import-not-found]

    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    rng = np.random.default_rng(42)
    n = 400
    df = pd.DataFrame({
        "x1": rng.normal(size=n),
        "x2": rng.normal(size=n),
    })
    df["y_cont"]  = 1 + 0.5 * df["x1"] - 0.3 * df["x2"] + rng.normal(size=n)
    df["y_bin"]   = (1 / (1 + np.exp(-(0.3 * df["x1"] - 0.5 * df["x2"]))) > rng.uniform(size=n)).astype(int)
    df["y_count"] = rng.poisson(np.exp(0.2 + 0.1 * df["x1"]))
    df["t_event"] = rng.exponential(scale=1 / np.exp(-0.3 + 0.4 * df["x1"]))
    df["cens"]    = (df["t_event"] < 2).astype(int)
    df["t_obs"]   = np.minimum(df["t_event"], 2)
    # Instrument for IV: z correlates with x1, exclusion holds for y_cont
    df["z"]       = 0.7 * df["x1"] + rng.normal(size=n)

    def run(label: str, factory):
        try:
            fit = factory()
            nora_runtime.from_lm(fit)
        except Exception as e:  # noqa: BLE001
            print(f"HELPER ERROR {label}: {type(e).__name__}: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
        finally:
            _emit_label(result_path, label)

    # 1. OLS via formula API
    run("ols", lambda: smf.ols("y_cont ~ x1 + x2", data=df).fit())

    # 2. Logit
    run("logit", lambda: smf.logit("y_bin ~ x1 + x2", data=df).fit(disp=False))

    # 3. Probit
    run("probit", lambda: smf.probit("y_bin ~ x1 + x2", data=df).fit(disp=False))

    # 4. Poisson
    run("poisson", lambda: smf.poisson("y_count ~ x1 + x2", data=df).fit(disp=False))

    # 5. Negative binomial
    run("negbin", lambda: smf.negativebinomial("y_count ~ x1 + x2", data=df).fit(disp=False))

    # 6. GLM with Binomial family (alt path to logit)
    run("glm_binomial", lambda: smf.glm(
        "y_bin ~ x1 + x2", data=df, family=sm.families.Binomial()
    ).fit())

    # 7. Cox PH via PHReg
    run("phreg", lambda: sm.PHReg(
        df["t_obs"].values, df[["x1", "x2"]].values,
        status=df["cens"].values,
    ).fit())

    # 8. 2SLS / IV via the generic ``from_lm`` path (structural table
    # only — diagnostic fields still missing).
    try:
        from statsmodels.sandbox.regression.gmm import IV2SLS
        run("iv2sls", lambda: IV2SLS(
            df["y_cont"], sm.add_constant(df[["x1", "x2"]]),
            sm.add_constant(df[["z",  "x2"]]),
        ).fit())
    except ImportError:
        _emit_label(result_path, "iv2sls")

    # 9. 2SLS via the new ``from_iv`` helper with diagnostic scalars
    # passed in. The model's script computes them; the helper packages.
    try:
        from statsmodels.sandbox.regression.gmm import IV2SLS
        endo = df["x1"].values
        instruments_mat = sm.add_constant(df[["z", "x2"]])
        first_stage = sm.OLS(endo, instruments_mat).fit()
        iv_fit = IV2SLS(
            df["y_cont"],
            sm.add_constant(df[["x1", "x2"]]),
            instruments_mat,
        ).fit()
        try:
            nora_runtime.from_iv(
                iv_fit,
                instrument_variables=["z"],
                endogenous_variables=["x1"],
                first_stage_f=float(first_stage.fvalue),
                weak_instrument_p=float(first_stage.f_pvalue),
            )
        except Exception as e:  # noqa: BLE001
            print(f"HELPER ERROR iv2sls_full: {e}", file=sys.stderr)
        finally:
            _emit_label(result_path, "iv2sls_full")
    except ImportError:
        _emit_label(result_path, "iv2sls_full")

    # 10. OLS with cluster-robust SE — pin the auto-extraction of
    # ``cluster_variables`` / ``n_clusters`` from
    # ``cov_kwds["groups"]``.
    def _ols_clustered():
        g = pd.Series(np.tile(np.arange(40), 10), name="firm_id")
        # Match the running n=400 dataset; sample 400 entries from g.
        g400 = pd.Series(np.tile(np.arange(40), 10), name="firm_id")
        return smf.ols(
            "y_cont ~ x1 + x2", data=df
        ).fit(cov_type="cluster", cov_kwds={"groups": g400})
    run("ols_clustered", _ols_clustered)

    print(f"audit complete: {result_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
