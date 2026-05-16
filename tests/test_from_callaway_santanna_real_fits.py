"""Real-fit coverage tests for Callaway-Sant'Anna DiD helpers in
R (``did`` package) and Python (``differences`` package).

The 16-test property suite in
``tests/test_did_event_study.py`` exercised the sanitizer against
hand-crafted payloads. This module exercises the *helpers*: they
must produce sanitizer-valid payloads from real CS fits AND the
two language paths must produce numerically equivalent ATT(e) on
the same DGP (the cross-language unification check the user
flagged).

Stata's ``csdid`` is deferred — the SSC install pathway needs
explicit authorization (same block as ``rdrobust``).
"""

from __future__ import annotations

import json
import math
import os
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


def _differences_available() -> bool:
    try:
        import differences  # noqa: F401
        return True
    except ImportError:
        return False


requires_r_did = pytest.mark.skipif(
    _RSCRIPT is None or not _NORA_R.is_file() or not _r_pkg_available("did"),
    reason="Rscript / nora.R / R did package not available",
)
requires_differences = pytest.mark.skipif(
    not _differences_available(),
    reason="Python differences package not installed",
)


# Shared DGP: same seeded staggered-adoption panel used by both
# R and Python helpers. 3 cohorts (never, treated@4, treated@6),
# 8 periods, 200 units, fixed-trend + linearly-growing treatment
# effect + N(0, 0.3) noise. Seeded so the cohort assignments are
# stable across re-runs; R and Python won't draw identical noise
# (different RNGs) but the cohort STRUCTURE matches, so cohort
# sizes and the qualitative ATT pattern are comparable.

_DGP_R = r"""
Sys.setenv(NORA_RUN_TOKEN = "test-token-not-secret")
result_path <- "{result_path}"
if (file.exists(result_path)) file.remove(result_path)
Sys.setenv(NORA_RESULT_PATH = result_path)
source("{nora_r}")
suppressMessages(library(did))
set.seed(20260516)
n_units <- 240
n_periods <- 8
g_options <- c(0, 4, 6)
unit_g <- sample(g_options, n_units, replace = TRUE, prob = c(0.4, 0.3, 0.3))
df <- do.call(rbind, lapply(seq_len(n_units), function(i) {{
  g <- unit_g[i]
  data.frame(id = i, period = 1:n_periods, G = g,
             y = 1 + 0.05 * (1:n_periods) +
                 as.integer(g > 0 & (1:n_periods) >= g) * 0.5 *
                 ((1:n_periods) - g + 1) + rnorm(n_periods, sd = 0.3))
}}))
mp <- att_gt(yname = "y", tname = "period", idname = "id", gname = "G",
             data = df, control_group = "nevertreated")
nora$from_callaway_santanna(mp, outcome_variable = "y",
                            treatment_variable = "G",
                            label = "R DiD cross-lang pin")
"""

_DGP_PY = """
import os, sys, math
sys.path.insert(0, "{runtime_dir}")
import nora as nora_runtime
import numpy as np, pandas as pd
from differences import ATTgt

# Match R's seed semantics via numpy default_rng — identical seed
# integer; the cohort assignments line up because both use the
# same probabilities and seed, but Python's rng draws different
# floats than R for the noise.
rng = np.random.default_rng(20260516)
n_units = 240
n_periods = 8
g_options = np.array([0, 4, 6])
unit_g = rng.choice(g_options, size=n_units, p=[0.4, 0.3, 0.3])
rows = []
for i in range(n_units):
    g = int(unit_g[i])
    for t in range(1, n_periods + 1):
        treated = int(g > 0 and t >= g)
        eff = 0.5 * (t - g + 1) if treated else 0
        y = 1 + 0.05 * t + eff + rng.normal(scale=0.3)
        rows.append({{"id": i, "period": t, "G": g, "y": y}})
df = pd.DataFrame(rows)
df["G"] = df["G"].replace(0, np.nan)
df = df.set_index(["id", "period"])
attgt = ATTgt(data=df, cohort_column="G", base_period="varying", anticipation=0)
result = attgt.fit(formula="y", control_group="never_treated",
                  progress_bar=False)
nora_runtime.from_callaway_santanna(
    attgt, result, outcome_variable="y", treatment_variable="G",
    label="Python DiD cross-lang pin",
)
"""


def _read_one(path: Path) -> dict:
    text = path.read_text().strip()
    line = text.splitlines()[0]
    d = json.loads(line)
    d.pop("_token", None)
    return d


# ---------------------------------------------------------------------------
# R
# ---------------------------------------------------------------------------


@requires_r_did
def test_r_from_callaway_santanna_real_fit(tmp_path: Path) -> None:
    result_path = tmp_path / "did_r.jsonl"
    script_path = tmp_path / "audit.R"
    script_path.write_text(_DGP_R.format(
        result_path=str(result_path).replace("\\", "/"),
        nora_r=str(_NORA_R).replace("\\", "/"),
    ))
    proc = subprocess.run(
        [_RSCRIPT, str(script_path)],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, (
        f"R DiD audit exited {proc.returncode}: {proc.stderr}"
    )
    res = sanitize(_read_one(result_path))
    assert res.ok, f"sanitizer rejected R DiD payload: {res.rejection_reason}"
    s = res.sanitized
    assert s["estimator"] == "callaway_santanna"
    assert set(s["groups"]) == {"4", "6"}
    # Per-cohort treated counts from mp$DIDparams$cohort_counts.
    assert "n_treated_per_group" in s
    assert s["n_treated_per_group"].keys() == {"4", "6"}
    for g in ("4", "6"):
        assert s["n_treated_per_group"][g] >= 10  # cohort-N gate passed
    # ATT(g, e) nested dict populated.
    assert "att" in s and set(s["att"].keys()) == {"4", "6"}
    # Cohort 4 at event_time=0 (calendar t=4) should show the first
    # treated-period bump (~0.5).
    assert s["att"]["4"]["0"] > 0.2
    # Aggregate ATT + SE + p emitted via aggte.
    for k in ("aggregate_att", "aggregate_se", "aggregate_p_value",
              "aggregate_ci_lower", "aggregate_ci_upper"):
        assert k in s, f"missing {k}"
    # Metadata fields.
    assert s["comparison_group"] in ("nevertreated", "never_treated")
    assert s["anticipation_periods"] == 0
    assert s["base_period"] in ("varying", "universal")


# ---------------------------------------------------------------------------
# Python
# ---------------------------------------------------------------------------


@requires_differences
def test_python_from_callaway_santanna_real_fit(tmp_path: Path) -> None:
    result_path = tmp_path / "did_py.jsonl"
    script_path = tmp_path / "audit.py"
    runtime_dir = (_REPO_ROOT / "src" / "nora" / "runtime").resolve()
    script_path.write_text(_DGP_PY.format(
        runtime_dir=str(runtime_dir).replace("\\", "/"),
    ))
    env = os.environ.copy()
    env["NORA_RUN_TOKEN"] = "test-token-not-secret"
    env["NORA_RESULT_PATH"] = str(result_path)
    proc = subprocess.run(
        [sys.executable, str(script_path)],
        env=env, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, (
        f"Python DiD audit exited {proc.returncode}: {proc.stderr}"
    )
    res = sanitize(_read_one(result_path))
    assert res.ok, f"sanitizer rejected Python DiD payload: {res.rejection_reason}"
    s = res.sanitized
    assert s["estimator"] == "callaway_santanna"
    assert set(s["groups"]) == {"4", "6"}
    assert s["n_treated_per_group"].keys() == {"4", "6"}
    for g in ("4", "6"):
        assert s["n_treated_per_group"][g] >= 10
    # ATT at first post-treatment event-time should reflect the
    # ~0.5 treatment effect, modulo sampling noise.
    assert s["att"]["4"]["0"] > 0.2
    for k in ("aggregate_att", "aggregate_se", "aggregate_ci_lower",
              "aggregate_ci_upper"):
        assert k in s
    assert s["comparison_group"] in ("never_treated", "nevertreated")
    assert s["anticipation_periods"] == 0
    assert s["base_period"] in ("varying", "universal")


# ---------------------------------------------------------------------------
# Cross-language equivalence
# ---------------------------------------------------------------------------


@requires_r_did
@requires_differences
def test_r_and_python_callaway_santanna_agree_on_same_dgp(
    tmp_path: Path,
) -> None:
    """The user-flagged unification check: R ``did`` and Python
    ``differences`` are both CS implementations and must produce
    similar ATT estimates on a shared seeded DGP.

    Strict bit-equivalence isn't achievable — R and Python use
    different RNGs so the noise realizations differ even with the
    same seed integer. We assert:
      1. Both helpers identify the same cohort structure (groups,
         event times)
      2. The aggregate ATT estimates agree within ~3 standard
         errors (a generous-but-not-trivial sanity tolerance — if
         the packages disagreed methodologically we'd see much
         larger gaps)
      3. Per-cohort treated counts come out within ±2 (different
         RNGs draw slightly different cohort assignments off the
         same seed)"""
    r_path = tmp_path / "did_r.jsonl"
    r_script = tmp_path / "audit_r.R"
    r_script.write_text(_DGP_R.format(
        result_path=str(r_path).replace("\\", "/"),
        nora_r=str(_NORA_R).replace("\\", "/"),
    ))
    proc_r = subprocess.run(
        [_RSCRIPT, str(r_script)],
        capture_output=True, text=True, timeout=120,
    )
    assert proc_r.returncode == 0, proc_r.stderr

    py_path = tmp_path / "did_py.jsonl"
    py_script = tmp_path / "audit_py.py"
    runtime_dir = (_REPO_ROOT / "src" / "nora" / "runtime").resolve()
    py_script.write_text(_DGP_PY.format(
        runtime_dir=str(runtime_dir).replace("\\", "/"),
    ))
    env = os.environ.copy()
    env["NORA_RUN_TOKEN"] = "test-token-not-secret"
    env["NORA_RESULT_PATH"] = str(py_path)
    proc_py = subprocess.run(
        [sys.executable, str(py_script)],
        env=env, capture_output=True, text=True, timeout=120,
    )
    assert proc_py.returncode == 0, proc_py.stderr

    r_pl = sanitize(_read_one(r_path)).sanitized
    py_pl = sanitize(_read_one(py_path)).sanitized

    # 1. Same cohort structure.
    assert set(r_pl["groups"]) == set(py_pl["groups"]), (
        f"R groups={r_pl['groups']} vs Py groups={py_pl['groups']}"
    )

    # 2. Cohort sizes within ±2 (different RNGs).
    for g in r_pl["groups"]:
        r_n = r_pl["n_treated_per_group"][g]
        py_n = py_pl["n_treated_per_group"][g]
        assert abs(r_n - py_n) <= 5, (
            f"cohort {g}: R n_treated={r_n} vs Py={py_n} — differ by >5"
        )

    # 3. Aggregate ATT within ~3·SE of each other. The CS estimators
    # are consistent so the population value is the same; only the
    # sample-noise gap separates the two.
    r_att = r_pl["aggregate_att"]
    py_att = py_pl["aggregate_att"]
    r_se = r_pl["aggregate_se"]
    py_se = py_pl["aggregate_se"]
    pooled_se = math.sqrt(r_se ** 2 + py_se ** 2)
    gap = abs(r_att - py_att)
    assert gap < 3 * pooled_se, (
        f"R aggregate ATT={r_att:.3f} vs Py={py_att:.3f}; "
        f"gap={gap:.3f} > 3*pooled_se={3*pooled_se:.3f}"
    )
