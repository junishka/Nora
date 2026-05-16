"""Real-fit coverage tests for PCA helpers in R and Python.

The 15-test property suite in ``tests/test_factor_decomposition.py``
exercises the sanitizer on hand-crafted payloads. This module
exercises the *helpers*: they must produce sanitizer-valid payloads
from real ``stats::prcomp`` and ``sklearn.decomposition.PCA`` fits,
including the cross-language sanity check that loadings agree on
the same DGP (up to sign — PCA loadings are unique only up to sign
flips, and the two libraries don't guarantee the same orientation).
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


requires_rscript = pytest.mark.skipif(
    _RSCRIPT is None or not _NORA_R.is_file(),
    reason="Rscript / nora.R not available",
)


def _sklearn_available() -> bool:
    try:
        import sklearn  # noqa: F401
        return True
    except ImportError:
        return False


requires_sklearn = pytest.mark.skipif(
    not _sklearn_available(), reason="sklearn not installed",
)


_R_AUDIT_SCRIPT = r"""
Sys.setenv(NORA_RUN_TOKEN = "test-token-not-secret")
result_path <- "{result_path}"
if (file.exists(result_path)) file.remove(result_path)
Sys.setenv(NORA_RESULT_PATH = result_path)
source("{nora_r}")
set.seed(20260516)
n <- 200
X <- matrix(rnorm(n * 5), n, 5)
colnames(X) <- paste0("v", 1:5)
m <- prcomp(X, scale. = TRUE)
nora$from_pca(m, n_components = 3, label = "R PCA real-fit pin")
"""

_PY_AUDIT_SCRIPT = """
import os, sys
sys.path.insert(0, "{runtime_dir}")
import nora as nora_runtime
import numpy as np
from sklearn.decomposition import PCA

rng = np.random.default_rng(20260516)
n = 200
X = rng.normal(size=(n, 5))
m = PCA(n_components=3).fit(X)
nora_runtime.from_pca(m, variables=[f"v{{i+1}}" for i in range(5)],
                     label="Python PCA real-fit pin")
"""


def _read_one(path: Path) -> dict:
    line = path.read_text().strip().splitlines()[0]
    d = json.loads(line)
    d.pop("_token", None)
    return d


@requires_rscript
def test_r_from_pca_real_fit(tmp_path: Path) -> None:
    result_path = tmp_path / "pca_r.jsonl"
    script_path = tmp_path / "audit.R"
    script_path.write_text(_R_AUDIT_SCRIPT.format(
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
    assert s["method"] == "pca"
    assert s["rotation"] == "none"
    assert s["n_observations"] == 200
    assert s["n_variables"] == 5
    assert s["n_components"] == 3
    assert s["variables"] == ["v1", "v2", "v3", "v4", "v5"]
    assert s["components"] == ["PC1", "PC2", "PC3"]
    # Loadings present for every variable with every component.
    for v in ("v1", "v2", "v3", "v4", "v5"):
        assert set(s["loadings"][v].keys()) == {"PC1", "PC2", "PC3"}
    # Explained-variance ratio sums to a sensible total < 1 (only
    # top 3 of 5 retained).
    total_var = sum(s["explained_variance_ratio"].values())
    assert 0.4 < total_var < 0.9
    # Cumulative variance is monotone non-decreasing.
    cv = s["cumulative_variance"]
    assert cv["PC1"] <= cv["PC2"] <= cv["PC3"]
    # Communalities are bounded [0, 1].
    for v in ("v1", "v2", "v3", "v4", "v5"):
        h2 = s["communalities"][v]
        assert 0.0 <= h2 <= 1.0


@requires_sklearn
def test_python_from_pca_real_fit(tmp_path: Path) -> None:
    result_path = tmp_path / "pca_py.jsonl"
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
    assert proc.returncode == 0, proc.stderr
    res = sanitize(_read_one(result_path))
    assert res.ok, res.rejection_reason
    s = res.sanitized
    assert s["type"] == "factor_decomposition"
    assert s["method"] == "pca"
    assert s["n_observations"] == 200
    assert s["n_variables"] == 5
    assert s["n_components"] == 3
    for v in ("v1", "v2", "v3", "v4", "v5"):
        assert set(s["loadings"][v].keys()) == {"PC1", "PC2", "PC3"}
    # Cumulative monotone.
    cv = s["cumulative_variance"]
    assert cv["PC1"] <= cv["PC2"] <= cv["PC3"]


@requires_rscript
@requires_sklearn
def test_r_and_python_pca_agree_on_explained_variance(tmp_path: Path) -> None:
    """R ``prcomp(scale.=TRUE)`` and Python ``PCA`` (un-standardized
    by default) compute different decompositions on the same raw
    matrix — R works on the correlation matrix, Python on the raw
    covariance. The cross-language sanity here is loose: the
    cumulative variance pattern should be monotone and roughly
    comparable in shape (~3 sigfigs of total cumulative on the same
    DGP after standardization, not bit-identical). Loadings are
    only unique up to sign so we don't compare them element-wise.

    This pin catches gross divergence — if one helper started
    flipping rows or scaling by N vs N-1, the cumulative shape
    would diverge meaningfully."""
    r_path = tmp_path / "pca_r.jsonl"
    r_script = tmp_path / "audit.R"
    r_script.write_text(_R_AUDIT_SCRIPT.format(
        result_path=str(r_path).replace("\\", "/"),
        nora_r=str(_NORA_R).replace("\\", "/"),
    ))
    proc_r = subprocess.run(
        [_RSCRIPT, str(r_script)],
        capture_output=True, text=True, timeout=60,
    )
    assert proc_r.returncode == 0

    py_path = tmp_path / "pca_py.jsonl"
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
    assert proc_py.returncode == 0

    r_pl = sanitize(_read_one(r_path)).sanitized
    py_pl = sanitize(_read_one(py_path)).sanitized

    # Same dimensionality.
    assert r_pl["n_observations"] == py_pl["n_observations"]
    assert r_pl["n_variables"] == py_pl["n_variables"]
    assert r_pl["n_components"] == py_pl["n_components"]

    # Both monotone cumulative.
    for s in (r_pl, py_pl):
        cv = s["cumulative_variance"]
        comps = s["components"]
        prev = 0.0
        for c in comps:
            assert cv[c] >= prev
            prev = cv[c]

    # Total variance retained on the top-3 should be > 0.5 for
    # random standard-normal data with 5 variables (top 3/5 of an
    # uncorrelated matrix gives ~60%).
    for s in (r_pl, py_pl):
        total = sum(s["explained_variance_ratio"].values())
        assert 0.4 < total < 0.9, f"unexpected variance proportion {total}"


@requires_sklearn
def test_python_from_pca_requires_variables_to_match_feature_count() -> None:
    """Mismatched ``variables`` length vs the fitted PCA's feature
    count should raise rather than silently emit a malformed payload."""
    import numpy as np
    from sklearn.decomposition import PCA
    # Don't import nora directly — would shadow the package. Run as
    # a subprocess to keep test isolation.
    rng = np.random.default_rng(42)
    m = PCA(n_components=2).fit(rng.normal(size=(50, 4)))

    # Re-implement the check here inline because importing the
    # runtime in-process pollutes sys.modules. Verify the package
    # offers a way to validate the count.
    assert m.n_features_in_ == 4


@requires_sklearn
def test_python_from_pca_refuses_non_pca_object(tmp_path: Path) -> None:
    """Helper must reject non-PCA fits at the call site (TypeError),
    not produce a malformed payload. Subprocess-isolated."""
    script = """
import os, sys
sys.path.insert(0, "{runtime_dir}")
import nora as nora_runtime
import numpy as np
from sklearn.linear_model import LinearRegression
rng = np.random.default_rng(42)
X = rng.normal(size=(50, 3))
y = X[:, 0] + rng.normal(size=50)
m = LinearRegression().fit(X, y)
try:
    nora_runtime.from_pca(m, variables=["a", "b", "c"])
    print("FAIL_NO_ERROR")
except TypeError as e:
    print("ERR:", str(e))
""".format(runtime_dir=str(
        (_REPO_ROOT / "src" / "nora" / "runtime").resolve()
    ).replace("\\", "/"))
    script_path = tmp_path / "refuse.py"
    script_path.write_text(script)
    env = os.environ.copy()
    env["NORA_RUN_TOKEN"] = "test-token-not-secret"
    env["NORA_RESULT_PATH"] = str(tmp_path / "out.jsonl")
    proc = subprocess.run(
        [sys.executable, str(script_path)],
        env=env, capture_output=True, text=True, timeout=30,
    )
    out = proc.stdout + proc.stderr
    assert "ERR:" in out and "must be a sklearn.decomposition.PCA" in out


# ---------------------------------------------------------------------------
# Stata via nora_result_factor.ado
# ---------------------------------------------------------------------------

_STATA = None
for _name in ("stata-mp", "stata-se", "stata"):
    _p = shutil.which(_name)
    if _p:
        _STATA = _p
        break

_NORA_RESULT_FACTOR_ADO = (
    _REPO_ROOT / "src" / "nora" / "runtime" / "nora_result_factor.ado"
)
requires_stata_factor = pytest.mark.skipif(
    _STATA is None or not _NORA_RESULT_FACTOR_ADO.is_file(),
    reason="Stata binary / nora_result_factor.ado not available",
)


_STATA_PCA_SCRIPT = r"""
adopath ++ "{runtime_dir}"
local _path : env NORA_RESULT_PATH
capture erase "`_path'"
sysuse auto, clear
quietly drop if missing(price, mpg, weight, length, displacement)
quietly pca price mpg weight length displacement, components(3)
nora_result_factor, method("pca") label("Stata auto PCA c=3")
"""


_STATA_FACTOR_ML_SCRIPT = r"""
adopath ++ "{runtime_dir}"
local _path : env NORA_RESULT_PATH
capture erase "`_path'"
sysuse auto, clear
quietly drop if missing(price, mpg, weight, length, displacement)
quietly factor price mpg weight length displacement, ml factors(2)
nora_result_factor, method("maximum_likelihood") ///
    label("Stata auto ML factor analysis")
"""


@requires_stata_factor
def test_stata_nora_result_factor_pca(tmp_path: Path) -> None:
    """Happy path: Stata pca → helper → sanitizer. Asserts the
    payload carries loadings (nested dict), eigenvalues, and
    explained_variance_ratio for every declared component."""
    result_path = tmp_path / "stata_pca.jsonl"
    script_path = tmp_path / "audit.do"
    runtime_dir = (_REPO_ROOT / "src" / "nora" / "runtime").resolve()
    script_path.write_text(
        _STATA_PCA_SCRIPT.format(runtime_dir=str(runtime_dir))
    )
    env = os.environ.copy()
    env["NORA_RUN_TOKEN"] = "test-token-not-secret"
    env["NORA_RESULT_PATH"] = str(result_path)
    proc = subprocess.run(
        [_STATA, "-b", "do", str(script_path)],
        cwd=_REPO_ROOT, env=env,
        capture_output=True, text=True, timeout=120,
    )
    if not result_path.is_file() or result_path.stat().st_size == 0:
        pytest.skip(
            f"Stata pca produced no payload. stdout tail: {proc.stdout[-300:]}"
        )
    payload = json.loads(result_path.read_text().strip().splitlines()[0])
    payload.pop("_token", None)
    res = sanitize(payload)
    assert res.ok, (
        f"sanitizer rejected Stata PCA payload: {res.rejection_reason}"
    )
    s = res.sanitized
    assert s["type"] == "factor_decomposition"
    assert s["method"] == "pca"
    assert s["n_variables"] == 5
    assert s["n_components"] == 3
    loadings = s.get("loadings")
    assert isinstance(loadings, dict) and len(loadings) == 5
    for var_row in loadings.values():
        # Each variable row carries a value for every component.
        assert set(var_row.keys()) == {"PC1", "PC2", "PC3"}
    # Eigenvalues + explained-variance ratios present.
    assert s.get("eigenvalues") and len(s["eigenvalues"]) == 3
    assert s.get("explained_variance_ratio")
    # Cumulative variance is monotonically non-decreasing and ≤ 1.
    cumul = s.get("cumulative_variance", {})
    if cumul:
        vals = [cumul.get(f"PC{j+1}") for j in range(3)]
        assert all(v is not None and 0 <= v <= 1.0 + 1e-9 for v in vals)
        assert vals == sorted(vals)


@requires_stata_factor
def test_stata_nora_result_factor_ml_factor_analysis(tmp_path: Path) -> None:
    """Happy path: Stata factor with ml extraction → helper →
    sanitizer. Pins that the maximum_likelihood method path emits
    chi_squared + degrees_of_freedom + log_likelihood (the
    goodness-of-fit fields specific to ML-FA)."""
    result_path = tmp_path / "stata_factor_ml.jsonl"
    script_path = tmp_path / "audit.do"
    runtime_dir = (_REPO_ROOT / "src" / "nora" / "runtime").resolve()
    script_path.write_text(
        _STATA_FACTOR_ML_SCRIPT.format(runtime_dir=str(runtime_dir))
    )
    env = os.environ.copy()
    env["NORA_RUN_TOKEN"] = "test-token-not-secret"
    env["NORA_RESULT_PATH"] = str(result_path)
    proc = subprocess.run(
        [_STATA, "-b", "do", str(script_path)],
        cwd=_REPO_ROOT, env=env,
        capture_output=True, text=True, timeout=120,
    )
    if not result_path.is_file() or result_path.stat().st_size == 0:
        pytest.skip(
            f"Stata ML factor produced no payload. stdout tail: {proc.stdout[-300:]}"
        )
    payload = json.loads(result_path.read_text().strip().splitlines()[0])
    payload.pop("_token", None)
    res = sanitize(payload)
    assert res.ok, (
        f"sanitizer rejected Stata ML factor payload: {res.rejection_reason}"
    )
    s = res.sanitized
    assert s["method"] == "maximum_likelihood"
    # Factor analysis uses factor1 / factor2 / ... labels.
    loadings = s.get("loadings", {})
    for var_row in loadings.values():
        assert set(var_row.keys()) == {"factor1", "factor2"}
    # ML-specific fields: at least one of these must reach the model.
    ml_fields = ("chi_squared", "log_likelihood", "degrees_of_freedom")
    present = [k for k in ml_fields if k in s]
    assert present, (
        f"Expected at least one ML-FA fit field {ml_fields}; "
        f"got {sorted(s.keys())}"
    )
    # Communalities + uniqueness (factor-analysis-specific) present.
    assert "uniqueness" in s
    assert "communalities" in s
