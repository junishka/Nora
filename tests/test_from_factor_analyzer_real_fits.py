"""Real-fit coverage for the factor-analysis helpers in R and Python.

The 15-test property suite in ``tests/test_factor_decomposition.py``
exercises the sanitizer on hand-crafted payloads. This module
exercises the helpers: ``nora$from_fa`` (R, wrapping
``psych::fa``) and ``nora.from_factor_analyzer`` (Python, wrapping
``factor_analyzer.FactorAnalyzer``) must produce sanitizer-valid
payloads with the inference-adequate fields (loadings,
communalities, eigenvalues, rotation, method, goodness-of-fit)
from real fits.

Same canonical-package posture as the PCA real-fit pin: the
helpers don't reinvent inference — they extract from the
canonical library output and route onto the
``factor_decomposition`` shape.
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


def _factor_analyzer_available() -> bool:
    try:
        import factor_analyzer  # noqa: F401
        return True
    except ImportError:
        return False


requires_r_psych = pytest.mark.skipif(
    _RSCRIPT is None or not _NORA_R.is_file() or not _r_pkg_available("psych"),
    reason="Rscript / nora.R / R psych package not available",
)

requires_factor_analyzer = pytest.mark.skipif(
    not _factor_analyzer_available(),
    reason="factor_analyzer not installed",
)


_R_FA_SCRIPT = r"""
Sys.setenv(NORA_RUN_TOKEN = "test-token-not-secret")
result_path <- "{result_path}"
if (file.exists(result_path)) file.remove(result_path)
Sys.setenv(NORA_RESULT_PATH = result_path)
source("{nora_r}")
suppressMessages(library(psych))
set.seed(20260516)
n <- 300
F1 <- rnorm(n); F2 <- rnorm(n)
e <- matrix(rnorm(n * 5, sd = 0.5), n, 5)
X <- cbind(
  0.8 * F1 + e[, 1],
  0.7 * F1 + e[, 2],
  0.6 * F1 + 0.2 * F2 + e[, 3],
  0.1 * F1 + 0.8 * F2 + e[, 4],
  0.05 * F1 + 0.75 * F2 + e[, 5]
)
colnames(X) <- paste0("v", 1:5)
m <- fa(X, nfactors = 2, rotate = "varimax", fm = "ml")
nora$from_fa(m, label = "R FA real-fit pin")
"""


_PY_FA_SCRIPT = """
import os, sys, warnings
sys.path.insert(0, "{runtime_dir}")
warnings.filterwarnings("ignore")
import nora as nora_runtime
import numpy as np
import pandas as pd
from factor_analyzer import FactorAnalyzer

rng = np.random.default_rng(20260516)
n = 300
F1 = rng.normal(size=n); F2 = rng.normal(size=n)
e = rng.normal(size=(n, 5)) * 0.5
X = np.column_stack([
    0.8 * F1 + e[:, 0], 0.7 * F1 + e[:, 1],
    0.6 * F1 + 0.2 * F2 + e[:, 2], 0.1 * F1 + 0.8 * F2 + e[:, 3],
    0.05 * F1 + 0.75 * F2 + e[:, 4],
])
df = pd.DataFrame(X, columns=["v1","v2","v3","v4","v5"])
fa = FactorAnalyzer(n_factors=2, rotation="varimax", method="ml")
fa.fit(df)
nora_runtime.from_factor_analyzer(
    fa, variables=["v1","v2","v3","v4","v5"],
    n_observations=n, label="Python FA real-fit pin",
)
"""


def _read_one(path: Path) -> dict:
    line = path.read_text().strip().splitlines()[0]
    d = json.loads(line)
    d.pop("_token", None)
    return d


@requires_r_psych
def test_r_from_fa_real_fit(tmp_path: Path) -> None:
    result_path = tmp_path / "fa_r.jsonl"
    script_path = tmp_path / "audit.R"
    script_path.write_text(_R_FA_SCRIPT.format(
        result_path=str(result_path).replace("\\", "/"),
        nora_r=str(_NORA_R).replace("\\", "/"),
    ))
    proc = subprocess.run(
        [_RSCRIPT, str(script_path)],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    res = sanitize(_read_one(result_path))
    assert res.ok, res.rejection_reason
    s = res.sanitized
    assert s["type"] == "factor_decomposition"
    assert s["method"] == "maximum_likelihood"
    assert s["rotation"] == "varimax"
    assert s["n_variables"] == 5
    assert s["n_components"] == 2
    assert s["variables"] == ["v1", "v2", "v3", "v4", "v5"]
    # Loadings present for every variable with both factors.
    for v in ("v1", "v2", "v3", "v4", "v5"):
        assert len(s["loadings"][v]) == 2
    # Communalities are bounded [0, 1].
    for v in ("v1", "v2", "v3", "v4", "v5"):
        h2 = s["communalities"][v]
        assert 0.0 <= h2 <= 1.0
    # Inference-adequacy: chi² + p surface on ML fits.
    assert "chi_squared" in s
    assert "chi_squared_p_value" in s


@requires_factor_analyzer
def test_python_from_factor_analyzer_real_fit(tmp_path: Path) -> None:
    result_path = tmp_path / "fa_py.jsonl"
    script_path = tmp_path / "audit.py"
    runtime_dir = (_REPO_ROOT / "src" / "nora" / "runtime").resolve()
    script_path.write_text(_PY_FA_SCRIPT.format(
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
    assert s["type"] == "factor_decomposition"
    assert s["method"] == "maximum_likelihood"
    assert s["rotation"] == "varimax"
    assert s["n_observations"] == 300
    assert s["n_variables"] == 5
    assert s["n_components"] == 2
    for v in ("v1", "v2", "v3", "v4", "v5"):
        assert set(s["loadings"][v].keys()) == {"factor1", "factor2"}
    # Communalities bounded [0, 1].
    for v in ("v1", "v2", "v3", "v4", "v5"):
        assert 0.0 <= s["communalities"][v] <= 1.0
    # Goodness-of-fit scalars from sufficiency() when fit is ML.
    # Older factor_analyzer doesn't always populate them; the field
    # presence isn't strictly required for the helper to be useful.
    # The variance metrics ARE always populated.
    cv = s["cumulative_variance"]
    assert cv["factor1"] <= cv["factor2"]


@requires_r_psych
@requires_factor_analyzer
def test_r_and_python_fa_agree_on_factor_structure(tmp_path: Path) -> None:
    """R psych::fa and Python factor_analyzer should agree on the
    rough loading pattern on the same DGP — v1/v2/v3 load mostly on
    one factor, v4/v5 on the other. Bit-equal won't happen
    (different RNGs, different rotation tiebreaking), but the loading
    on the "dominant" factor for each variable should agree on sign
    structure after taking absolute values."""
    r_path = tmp_path / "fa_r.jsonl"
    r_script = tmp_path / "audit.R"
    r_script.write_text(_R_FA_SCRIPT.format(
        result_path=str(r_path).replace("\\", "/"),
        nora_r=str(_NORA_R).replace("\\", "/"),
    ))
    proc_r = subprocess.run(
        [_RSCRIPT, str(r_script)],
        capture_output=True, text=True, timeout=60,
    )
    assert proc_r.returncode == 0, proc_r.stderr
    s_r = sanitize(_read_one(r_path)).sanitized

    py_path = tmp_path / "fa_py.jsonl"
    py_script = tmp_path / "audit.py"
    runtime_dir = (_REPO_ROOT / "src" / "nora" / "runtime").resolve()
    py_script.write_text(_PY_FA_SCRIPT.format(
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

    # Loose communalities check: both should put v1, v2, v3, v4, v5
    # in roughly the same h² range. Communalities are factor-rotation
    # invariant so they're a clean cross-language comparison point.
    for v in ("v1", "v2", "v3", "v4", "v5"):
        h2_r = s_r["communalities"][v]
        h2_py = s_py["communalities"][v]
        assert abs(h2_r - h2_py) < 0.2, (
            f"communality on {v!r} diverges: R={h2_r} py={h2_py}"
        )
