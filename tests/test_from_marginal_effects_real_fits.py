"""Real-fit coverage for the ``marginal_effects`` helpers.

Python: wraps ``statsmodels`` ``Logit.fit().get_margeff()``.
R: wraps ``marginaleffects::avg_slopes`` (the actively-maintained
successor to ``margins``).

The 21-test property suite in ``tests/test_marginal_effects.py``
exercises the sanitizer on hand-crafted payloads. This module
verifies the helpers extract enough fit-metric / per-variable detail
from a real fit that the emitted payload sanitizes AND carries the
fields the model needs to interpret the result (the inference-
adequacy bar from ``docs/extending_analysis_shapes.md``).
"""

from __future__ import annotations

import json
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


def _statsmodels_available() -> bool:
    try:
        import statsmodels  # noqa: F401
        return True
    except ImportError:
        return False


requires_statsmodels = pytest.mark.skipif(
    not _statsmodels_available(), reason="statsmodels not installed",
)

requires_r_marginaleffects = pytest.mark.skipif(
    _RSCRIPT is None or not _NORA_R.is_file()
    or not _r_pkg_available("marginaleffects"),
    reason="Rscript / nora.R / R marginaleffects package not available",
)


_PY_AME_SCRIPT = """
import os, sys
sys.path.insert(0, "{runtime_dir}")
import nora as nora_runtime
import numpy as np
import pandas as pd
import statsmodels.api as sm

rng = np.random.default_rng(20260516)
n = 800
age = rng.normal(45, 12, n)
female = rng.binomial(1, 0.5, n)
income = rng.normal(40000, 12000, n)
# Logit DGP: probability of voting.
lin = -3.0 + 0.05 * age + 0.4 * female + 0.00001 * income
prob = 1 / (1 + np.exp(-lin))
y = rng.binomial(1, prob)
df = pd.DataFrame(dict(y=y, age=age, female=female, income=income))
X = sm.add_constant(df[["age", "female", "income"]])
m = sm.Logit(df["y"], X).fit(disp=0)
me = m.get_margeff(at="overall", method="dydx")
nora_runtime.from_marginal_effects(
    me, outcome_variable="y", model_family="logit",
    label="AME from logit",
)
"""


_R_AME_SCRIPT = r"""
Sys.setenv(NORA_RUN_TOKEN = "test-token-not-secret")
result_path <- "{result_path}"
if (file.exists(result_path)) file.remove(result_path)
Sys.setenv(NORA_RESULT_PATH = result_path)
source("{nora_r}")
suppressMessages(library(marginaleffects))
set.seed(20260516)
n <- 800
age    <- rnorm(n, 45, 12)
female <- rbinom(n, 1, 0.5)
income <- rnorm(n, 40000, 12000)
lin    <- -3.0 + 0.05 * age + 0.4 * female + 0.00001 * income
prob   <- 1 / (1 + exp(-lin))
y      <- rbinom(n, 1, prob)
df <- data.frame(y = y, age = age, female = female, income = income)
m  <- glm(y ~ age + female + income, data = df, family = binomial())
ame <- avg_slopes(m)
nora$from_marginal_effects(
  ame, method = "ame",
  outcome_variable = "y", model_family = "logit",
  n = nrow(df), label = "AME from logit"
)
"""


def _read_one(path: Path) -> dict:
    line = path.read_text().strip().splitlines()[0]
    d = json.loads(line)
    d.pop("_token", None)
    return d


@requires_statsmodels
def test_python_from_marginal_effects_ame_real_fit(tmp_path: Path) -> None:
    result_path = tmp_path / "me_py.jsonl"
    script_path = tmp_path / "audit.py"
    runtime_dir = (_REPO_ROOT / "src" / "nora" / "runtime").resolve()
    script_path.write_text(_PY_AME_SCRIPT.format(
        runtime_dir=str(runtime_dir).replace("\\", "/"),
    ))
    env = os.environ.copy()
    env["NORA_RUN_TOKEN"] = "test-token-not-secret"
    env["NORA_RESULT_PATH"] = str(result_path)
    proc = subprocess.run(
        [sys.executable, str(script_path)],
        env=env, capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    res = sanitize(_read_one(result_path))
    assert res.ok, res.rejection_reason
    s = res.sanitized
    assert s["type"] == "marginal_effects"
    assert s["method"] == "ame"
    # statsmodels' Logit get_margeff drops the intercept; the
    # surviving names should be age / female / income.
    assert set(s["variables"]) == {"age", "female", "income"}
    # All three effects present.
    assert set(s["effects"].keys()) == {"age", "female", "income"}
    # SE + p + CI should be present (statsmodels emits all of them).
    assert "standard_errors" in s
    assert "p_values" in s
    assert "ci_lower" in s and "ci_upper" in s
    # Sanity: age effect is positive (DGP coefficient 0.05 on age).
    assert s["effects"]["age"] > 0
    # n round-tripped.
    assert s["n"] == 800


@requires_r_marginaleffects
def test_r_from_marginal_effects_ame_real_fit(tmp_path: Path) -> None:
    result_path = tmp_path / "me_r.jsonl"
    script_path = tmp_path / "audit.R"
    script_path.write_text(_R_AME_SCRIPT.format(
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
    assert s["type"] == "marginal_effects"
    assert s["method"] == "ame"
    assert s["model_family"] == "logit"
    assert set(s["variables"]) == {"age", "female", "income"}
    assert set(s["effects"].keys()) == {"age", "female", "income"}
    assert "standard_errors" in s
    assert "p_values" in s
    assert "ci_lower" in s and "ci_upper" in s
    # Sanity: age effect positive on the same DGP.
    assert s["effects"]["age"] > 0


@requires_statsmodels
@requires_r_marginaleffects
def test_r_and_python_ame_agree_on_sign_and_magnitude(tmp_path: Path) -> None:
    """The two helpers wrap structurally different libraries
    (statsmodels DiscreteMargins vs marginaleffects::avg_slopes) but
    on the same DGP at the same seed the AMEs should agree on sign
    and rough magnitude. Bit-identical isn't achievable (the RNGs
    differ) — 4 · pooled-SE is generous-but-not-trivial."""
    py_path = tmp_path / "me_py.jsonl"
    py_script = tmp_path / "audit.py"
    runtime_dir = (_REPO_ROOT / "src" / "nora" / "runtime").resolve()
    py_script.write_text(_PY_AME_SCRIPT.format(
        runtime_dir=str(runtime_dir).replace("\\", "/"),
    ))
    env = os.environ.copy()
    env["NORA_RUN_TOKEN"] = "test-token-not-secret"
    env["NORA_RESULT_PATH"] = str(py_path)
    proc_py = subprocess.run(
        [sys.executable, str(py_script)],
        env=env, capture_output=True, text=True, timeout=60,
    )
    assert proc_py.returncode == 0, proc_py.stderr
    s_py = sanitize(_read_one(py_path)).sanitized

    r_path = tmp_path / "me_r.jsonl"
    r_script = tmp_path / "audit.R"
    r_script.write_text(_R_AME_SCRIPT.format(
        result_path=str(r_path).replace("\\", "/"),
        nora_r=str(_NORA_R).replace("\\", "/"),
    ))
    proc_r = subprocess.run(
        [_RSCRIPT, str(r_script)],
        capture_output=True, text=True, timeout=120,
    )
    assert proc_r.returncode == 0, proc_r.stderr
    s_r = sanitize(_read_one(r_path)).sanitized

    # Sign agreement on every variable.
    for v in ("age", "female", "income"):
        py_eff = s_py["effects"][v]
        r_eff = s_r["effects"][v]
        # On the structured DGP, sign should agree even across
        # different RNG draws.
        assert (py_eff > 0) == (r_eff > 0), (
            f"sign disagreement on {v!r}: py={py_eff} r={r_eff}"
        )
        # Loose magnitude check: difference within 4 · pooled SE.
        py_se = s_py["standard_errors"][v]
        r_se = s_r["standard_errors"][v]
        pooled_se = (py_se ** 2 + r_se ** 2) ** 0.5
        assert abs(py_eff - r_eff) < 4 * pooled_se, (
            f"AME divergence on {v!r}: py={py_eff} r={r_eff} "
            f"(pooled SE={pooled_se})"
        )
