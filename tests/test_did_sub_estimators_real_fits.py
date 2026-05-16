"""Real-fit coverage for the DiD sub-estimator helpers:
Sun-Abraham (interaction-weighted) and TWFE event study.

Both use the existing ``did_event_study`` shape — the schema
already accepts ``estimator: "sun_abraham"`` / ``"twfe_event_study"``.
This module exercises the helpers (which wrap ``fixest::feols`` in
R) and pins that the emitted payloads sanitize cleanly.

The shape is single-synthetic-cohort: Sun-Abraham aggregates across
treated cohorts via IW weights, so its natural output is one ATT
per event-time, not per (cohort, event-time). We package this as
``groups: ["all"]`` with ``n_treated_per_group: {"all": <total>}``.
The ``estimator`` field tells the model the aggregation happened
inside the estimator. Same construction for TWFE-ES.

de Chaisemartin-D'Haultfœuille (``DIDmultiplegt``) and the Stata
``csdid`` port are deferred — both need explicit auth to install
and have maintenance-lag risk.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nora.sanitizer import sanitize  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
_NORA_R = _REPO_ROOT / "src" / "nora" / "runtime" / "nora.R"
_RSCRIPT = shutil.which("Rscript")


def _r_pkg_available(pkg: str) -> bool:
    if _RSCRIPT is None:
        return False
    res = subprocess.run(
        [_RSCRIPT, "-e", f'suppressMessages(library({pkg}))'],
        capture_output=True, text=True, timeout=20,
    )
    return res.returncode == 0


requires_r_fixest = pytest.mark.skipif(
    _RSCRIPT is None or not _NORA_R.is_file() or not _r_pkg_available("fixest"),
    reason="Rscript / nora.R / R fixest package not available",
)


_R_SUNAB_SCRIPT = r"""
Sys.setenv(NORA_RUN_TOKEN = "test-token-not-secret")
result_path <- "{result_path}"
if (file.exists(result_path)) file.remove(result_path)
Sys.setenv(NORA_RESULT_PATH = result_path)
source("{nora_r}")
suppressMessages(library(fixest))

set.seed(20260516)
n_units <- 240; n_periods <- 8
g_options <- c(10000, 4, 6)  # 10000 = never-treated sentinel
unit_g <- sample(g_options, n_units, replace = TRUE, prob = c(0.4, 0.3, 0.3))
df <- do.call(rbind, lapply(seq_len(n_units), function(i) {{
  g <- unit_g[i]
  data.frame(id = i, period = 1:n_periods, G = g,
             y = 1 + 0.05*(1:n_periods) +
                 as.integer(g <= n_periods & (1:n_periods) >= g) * 0.5 *
                 ((1:n_periods) - g + 1) + rnorm(n_periods, sd=0.3))
}}))
n_treated_total <- length(unique(df$id[df$G <= n_periods]))

m <- feols(y ~ sunab(G, period) | id + period, data = df, cluster = ~id)
nora$from_sun_abraham(m, n_treated = n_treated_total,
                      outcome_variable = "y", treatment_variable = "G",
                      label = "Sun-Abraham real-fit pin")
"""


_R_TWFE_SCRIPT = r"""
Sys.setenv(NORA_RUN_TOKEN = "test-token-not-secret")
result_path <- "{result_path}"
if (file.exists(result_path)) file.remove(result_path)
Sys.setenv(NORA_RESULT_PATH = result_path)
source("{nora_r}")
suppressMessages(library(fixest))

set.seed(20260516)
n_units <- 240; n_periods <- 8
g_options <- c(10000, 4, 6)
unit_g <- sample(g_options, n_units, replace = TRUE, prob = c(0.4, 0.3, 0.3))
df <- do.call(rbind, lapply(seq_len(n_units), function(i) {{
  g <- unit_g[i]
  data.frame(id = i, period = 1:n_periods, G = g,
             y = 1 + 0.05*(1:n_periods) +
                 as.integer(g <= n_periods & (1:n_periods) >= g) * 0.5 *
                 ((1:n_periods) - g + 1) + rnorm(n_periods, sd=0.3))
}}))
df$rel_time <- ifelse(df$G > n_periods, -999, df$period - df$G)
df$treated <- as.integer(df$G <= n_periods)
n_treated_total <- length(unique(df$id[df$G <= n_periods]))

m <- feols(y ~ i(rel_time, treated, ref=-1) | id + period,
           data = df[df$rel_time != -999, ], cluster = ~id)
nora$from_twfe_event_study(m, n_treated = n_treated_total,
                           outcome_variable = "y",
                           event_time_pattern = "rel_time::([^:]+)",
                           label = "TWFE-ES real-fit pin")
"""


def _read_one(path: Path) -> dict:
    line = path.read_text().strip().splitlines()[0]
    d = json.loads(line)
    d.pop("_token", None)
    return d


@requires_r_fixest
def test_r_from_sun_abraham_real_fit(tmp_path: Path) -> None:
    result_path = tmp_path / "sa.jsonl"
    script_path = tmp_path / "audit.R"
    script_path.write_text(_R_SUNAB_SCRIPT.format(
        result_path=str(result_path).replace("\\", "/"),
        nora_r=str(_NORA_R).replace("\\", "/"),
    ))
    proc = subprocess.run(
        [_RSCRIPT, str(script_path)],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    res = sanitize(_read_one(result_path))
    assert res.ok, res.rejection_reason
    s = res.sanitized
    assert s["type"] == "did_event_study"
    assert s["estimator"] == "sun_abraham"
    # Single synthetic cohort "all" since Sun-Abraham aggregates
    # across cohorts internally.
    assert s["groups"] == ["all"]
    assert "all" in s["n_treated_per_group"]
    assert s["n_treated_per_group"]["all"] >= 10  # cohort gate passed
    # ATT series indexed by event time; treatment effect (~0.5 per
    # period since exposure) should be detectable at event_time=0.
    att_all = s["att"]["all"]
    assert "0" in att_all, "missing event_time=0"
    # Treatment-effect signal: ATT(0) should be roughly 0.5.
    assert 0.2 < att_all["0"] < 1.0
    # Pre-trend coefficients should be small (correctly-specified DGP).
    if "-3" in att_all:
        assert abs(att_all["-3"]) < 0.2


@requires_r_fixest
def test_r_from_twfe_event_study_real_fit(tmp_path: Path) -> None:
    result_path = tmp_path / "twfe.jsonl"
    script_path = tmp_path / "audit.R"
    script_path.write_text(_R_TWFE_SCRIPT.format(
        result_path=str(result_path).replace("\\", "/"),
        nora_r=str(_NORA_R).replace("\\", "/"),
    ))
    proc = subprocess.run(
        [_RSCRIPT, str(script_path)],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    res = sanitize(_read_one(result_path))
    assert res.ok, res.rejection_reason
    s = res.sanitized
    assert s["type"] == "did_event_study"
    assert s["estimator"] == "twfe_event_study"
    assert s["groups"] == ["all"]
    assert "all" in s["n_treated_per_group"]


@requires_r_fixest
def test_sun_abraham_helper_requires_n_treated(tmp_path: Path) -> None:
    """The cohort-N gate has no input without ``n_treated``. Helper
    must raise rather than emit a payload that bypasses the gate.
    Run with NORA_RUN_TOKEN set so the runtime loads — the missing
    arg error is what we want to surface, not the missing-token
    one."""
    script = r"""
Sys.setenv(NORA_RUN_TOKEN = "test-token-not-secret")
Sys.setenv(NORA_RESULT_PATH = "{result_path}")
source("{nora_r}")
suppressMessages(library(fixest))
set.seed(42)
df <- do.call(rbind, lapply(1:60, function(i) {{
  data.frame(id = i, period = 1:5, G = sample(c(2, 4, 10000), 1), y = rnorm(5))
}}))
m <- feols(y ~ sunab(G, period) | id + period, data = df)
tryCatch({{
  nora$from_sun_abraham(m, outcome_variable = "y")
  cat("FAIL_NO_ERROR\n")
}}, error = function(e) cat("ERR:", conditionMessage(e), "\n"))
""".format(
        nora_r=str(_NORA_R).replace("\\", "/"),
        result_path=str(tmp_path / "out.jsonl").replace("\\", "/"),
    )
    script_path = tmp_path / "refuse.R"
    script_path.write_text(script)
    proc = subprocess.run(
        [_RSCRIPT, str(script_path)],
        capture_output=True, text=True, timeout=60,
    )
    out = proc.stdout + proc.stderr
    assert "ERR:" in out and "n_treated" in out, f"unexpected output:\n{out}"
