"""Real-fit coverage tests for mixed-effects models routed through
``from_lm`` in R and Python.

Per the extension-guide pre-decision, mixed-effects fits the
``coefficient_table_with_fit_stats`` bucket structurally — fixed-
effects coefficient table + new variance-components block — so it
lands as **allowlist extensions** (``random_effects_variance``,
``n_groups_per_level``, ``fit_method``, ``icc``) plus per-class
helper branches in ``from_lm``, NOT a new shape.

Both helpers must:
    * detect the mixed-effects class (R ``merMod``; Python
      ``MixedLMResultsWrapper``) and route to the new branch
    * emit ``random_effects_variance`` keyed by RE-factor name
      (with a ``.term`` suffix for random slopes), values are
      variance components from the RE covariance matrix
    * emit ``n_groups_per_level`` keyed by RE-factor name, values
      are group cardinalities (same disclosure profile as
      ``fixed_effects`` and ``n_clusters``)
    * emit ``fit_method`` ("REML" / "ML") and (where the math is
      well-defined) ``icc`` for the single-grouping intercept-only
      case
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


requires_r_lme4 = pytest.mark.skipif(
    _RSCRIPT is None or not _NORA_R.is_file() or not _r_pkg_available("lme4"),
    reason="Rscript / nora.R / R lme4 package not available",
)
requires_statsmodels = pytest.mark.skipif(
    not _statsmodels_available(),
    reason="statsmodels not installed",
)


# ---------------------------------------------------------------------------
# R via lme4
# ---------------------------------------------------------------------------

_R_AUDIT_SCRIPT = r"""
Sys.setenv(NORA_RUN_TOKEN = "test-token-not-secret")
result_path <- "{result_path}"
if (file.exists(result_path)) file.remove(result_path)
Sys.setenv(NORA_RESULT_PATH = result_path)
source("{nora_r}")
suppressMessages(library(lme4))

set.seed(20260516)
n_groups <- 40
n_per <- 25
school <- rep(1:n_groups, each = n_per)
x <- rnorm(n_groups * n_per)
school_eff <- rnorm(n_groups, sd = 0.6)
y <- school_eff[school] + 0.4 * x + rnorm(n_groups * n_per, sd = 0.4)
df <- data.frame(school = factor(school), x = x, y = y)

# Single-level RE — emit as the first payload.
m_lmer <- lmer(y ~ x + (1 | school), data = df)
nora$from_lm(m_lmer, label = "R lmer single-level")

# glmer logistic — emit as the second payload.
df$bin <- as.integer(y > 0)
m_glmer <- glmer(bin ~ x + (1 | school), data = df, family = binomial)
nora$from_lm(m_glmer, label = "R glmer logistic")
"""


def _read_lines(path: Path) -> list[dict]:
    payloads: list[dict] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        d.pop("_token", None)
        payloads.append(d)
    return payloads


@requires_r_lme4
def test_r_lmer_real_fit_emits_variance_components(tmp_path: Path) -> None:
    result_path = tmp_path / "mixed_r.jsonl"
    script_path = tmp_path / "audit.R"
    script_path.write_text(_R_AUDIT_SCRIPT.format(
        result_path=str(result_path).replace("\\", "/"),
        nora_r=str(_NORA_R).replace("\\", "/"),
    ))
    proc = subprocess.run(
        [_RSCRIPT, str(script_path)],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, (
        f"R mixed audit exited {proc.returncode}: {proc.stderr}"
    )
    payloads = _read_lines(result_path)
    assert len(payloads) == 2

    # First payload: lmer (single-level RE).
    res = sanitize(payloads[0])
    assert res.ok, res.rejection_reason
    s = res.sanitized
    # Type is the regression bucket — mixed isn't a new shape.
    assert s["type"] == "coefficient_table_with_fit_stats"
    # Variance components: school + residual, both finite.
    rev = s.get("random_effects_variance")
    assert isinstance(rev, dict) and set(rev.keys()) == {"school", "residual"}
    assert all(v > 0 for v in rev.values())
    # Group counts.
    assert s.get("n_groups_per_level") == {"school": 40}
    # Fit method (lmer default is REML).
    assert s.get("fit_method") == "REML"
    # ICC well-defined for one-group intercept-only fits.
    icc = s.get("icc")
    assert isinstance(icc, float) and 0.0 < icc < 1.0
    # AIC / BIC / log-likelihood all present.
    for k in ("aic", "bic", "log_likelihood"):
        assert k in s
    # lmer omits p-values by design (no Pr(...) column in summary).
    # The helper now skips p_values cleanly rather than mis-stamping
    # t-stats as p-values.
    assert "p_values" not in s or not s["p_values"]


@requires_r_lme4
def test_r_glmer_real_fit_emits_variance_components(tmp_path: Path) -> None:
    result_path = tmp_path / "mixed_r.jsonl"
    script_path = tmp_path / "audit.R"
    script_path.write_text(_R_AUDIT_SCRIPT.format(
        result_path=str(result_path).replace("\\", "/"),
        nora_r=str(_NORA_R).replace("\\", "/"),
    ))
    proc = subprocess.run(
        [_RSCRIPT, str(script_path)],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0
    payloads = _read_lines(result_path)
    # Second payload: glmer logistic.
    res = sanitize(payloads[1])
    assert res.ok, res.rejection_reason
    s = res.sanitized
    rev = s.get("random_effects_variance")
    assert isinstance(rev, dict) and "school" in rev
    assert s.get("n_groups_per_level") == {"school": 40}
    # glmer is ML (no REML for non-linear).
    assert s.get("fit_method") == "ML"
    # glmer DOES emit p-values (Pr(>|z|) column).
    pvals = s.get("p_values")
    assert isinstance(pvals, dict) and pvals


# ---------------------------------------------------------------------------
# Python via statsmodels.MixedLM
# ---------------------------------------------------------------------------

_PY_AUDIT_SCRIPT = """
import os, sys
sys.path.insert(0, "{runtime_dir}")
import nora as nora_runtime
import numpy as np, pandas as pd
import statsmodels.formula.api as smf

rng = np.random.default_rng(20260516)
n_groups, n_per = 40, 25
df = pd.DataFrame({{
    "school": np.repeat(range(n_groups), n_per),
    "x": rng.normal(size=n_groups*n_per),
}})
df["y"] = (rng.normal(size=n_groups, scale=0.6).repeat(n_per)
           + 0.4*df["x"] + rng.normal(size=len(df), scale=0.4))
m = smf.mixedlm("y ~ x", df, groups=df["school"]).fit()
nora_runtime.from_lm(m, group_variable="school",
                     label="Python MixedLM real-fit")
"""


@requires_statsmodels
def test_python_mixedlm_real_fit(tmp_path: Path) -> None:
    result_path = tmp_path / "mixed_py.jsonl"
    script_path = tmp_path / "audit.py"
    runtime_dir = (_REPO_ROOT / "src" / "nora" / "runtime").resolve()
    script_path.write_text(_PY_AUDIT_SCRIPT.format(
        runtime_dir=str(runtime_dir).replace("\\", "/"),
    ))
    env = os.environ.copy()
    env["NORA_RUN_TOKEN"] = "test-token-not-secret"
    env["NORA_RESULT_PATH"] = str(result_path)
    proc = subprocess.run(
        [sys.executable, str(script_path)],
        env=env, capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, (
        f"Python MixedLM audit exited {proc.returncode}: {proc.stderr}"
    )
    payloads = _read_lines(result_path)
    assert len(payloads) == 1
    res = sanitize(payloads[0])
    assert res.ok, res.rejection_reason
    s = res.sanitized
    assert s["type"] == "coefficient_table_with_fit_stats"
    rev = s.get("random_effects_variance")
    assert isinstance(rev, dict) and set(rev.keys()) == {"school", "residual"}
    assert all(v > 0 for v in rev.values())
    assert s.get("n_groups_per_level") == {"school": 40}
    assert s.get("fit_method") == "REML"
    icc = s.get("icc")
    assert isinstance(icc, float) and 0.0 < icc < 1.0


# ---------------------------------------------------------------------------
# Cross-language equivalence (variance components within tolerance)
# ---------------------------------------------------------------------------


@requires_r_lme4
@requires_statsmodels
def test_r_and_python_mixed_agree_on_same_dgp(tmp_path: Path) -> None:
    """R lme4 ``lmer`` and Python statsmodels ``mixedlm`` are both
    REML implementations of the same one-grouping random-intercept
    model. On a shared seeded DGP they should produce variance
    components within the same order of magnitude — not bit-equivalent
    (different RNGs even with the same seed integer), but a 2× factor
    is a generous-but-not-trivial sanity gap that would surface a
    methodological divergence."""
    r_path = tmp_path / "mixed_r.jsonl"
    r_script = tmp_path / "audit.R"
    r_script.write_text(_R_AUDIT_SCRIPT.format(
        result_path=str(r_path).replace("\\", "/"),
        nora_r=str(_NORA_R).replace("\\", "/"),
    ))
    proc_r = subprocess.run(
        [_RSCRIPT, str(r_script)],
        capture_output=True, text=True, timeout=120,
    )
    assert proc_r.returncode == 0, proc_r.stderr
    r_lmer = _read_lines(r_path)[0]  # first payload is lmer

    py_path = tmp_path / "mixed_py.jsonl"
    py_script = tmp_path / "audit.py"
    runtime_dir = (_REPO_ROOT / "src" / "nora" / "runtime").resolve()
    py_script.write_text(_PY_AUDIT_SCRIPT.format(
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
    py_mixed = _read_lines(py_path)[0]

    r_pl = sanitize(r_lmer).sanitized
    py_pl = sanitize(py_mixed).sanitized

    # 1. Same n_groups for the school factor.
    assert r_pl["n_groups_per_level"]["school"] == py_pl["n_groups_per_level"]["school"]

    # 2. Variance components within 2× of each other on each side.
    for key in ("school", "residual"):
        rv = r_pl["random_effects_variance"][key]
        pv = py_pl["random_effects_variance"][key]
        ratio = max(rv, pv) / min(rv, pv)
        assert ratio < 2.0, (
            f"{key}: R variance={rv:.4f} vs Py={pv:.4f}; "
            f"ratio={ratio:.2f}x > 2.0"
        )

    # 3. Both pin ICC inside (0, 1) and agree to ±0.2 (a relatively
    # generous tolerance, since variance ratios compound noise).
    assert abs(r_pl["icc"] - py_pl["icc"]) < 0.2
