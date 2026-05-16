"""Real-fit coverage tests for ``from_rdd`` helpers in R and Python.

Both helpers wrap the rdrobust package (Calonico-Cattaneo-Titiunik
2014). Stata's RDD path is deferred — the user's earlier note
flagged the Stata rdrobust port's maintenance lag and the SSC
install pathway requires explicit authorization. Once that ships,
mirror these pins for Stata.

The privacy carve-out for McCrary density and binscatter near the
cutoff is structural: the helper signatures don't accept density /
binscatter kwargs (raising on attempt), and the sanitizer's ``rdd``
allowlist has no corresponding fields. These tests pin BOTH halves
of that defense:
    * Helper refuses density/binscatter kwargs at call time
    * Sanitizer drops them silently if a hand-crafted payload
      tries to smuggle them through ``nora.result(type="rdd", ...)``
      (already covered by ``tests/test_rdd_and_kaplan_meier.py``)
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


def _python_rdrobust_available() -> bool:
    try:
        import rdrobust  # noqa: F401
        return True
    except ImportError:
        return False


requires_r_rdrobust = pytest.mark.skipif(
    _RSCRIPT is None or not _NORA_R.is_file()
    or not _r_pkg_available("rdrobust"),
    reason="Rscript / nora.R / R rdrobust package not available",
)
requires_python_rdrobust = pytest.mark.skipif(
    not _python_rdrobust_available(),
    reason="Python rdrobust package not installed",
)


# ---------------------------------------------------------------------------
# R via Rscript + rdrobust
# ---------------------------------------------------------------------------

_R_AUDIT_SCRIPT = r"""
Sys.setenv(NORA_RUN_TOKEN = "test-token-not-secret")
result_path <- "{result_path}"
if (file.exists(result_path)) file.remove(result_path)
Sys.setenv(NORA_RESULT_PATH = result_path)
source("{nora_r}")
suppressMessages(library(rdrobust))

set.seed(42)
n <- 800
x <- runif(n, -1, 1)
treated <- as.integer(x >= 0)
y <- 1 + 0.5*x + 0.3*treated + 0.2*x*treated + rnorm(n, sd = 0.3)

m <- rdrobust(y, x, c = 0)
nora$from_rdd(m, running_variable = "x", outcome_variable = "y",
              label = "R RDD real-fit pin")
"""


@requires_r_rdrobust
def test_r_from_rdd_real_fit(tmp_path: Path) -> None:
    result_path = tmp_path / "rdd_r.jsonl"
    script_path = tmp_path / "audit.R"
    script_path.write_text(_R_AUDIT_SCRIPT.format(
        result_path=str(result_path).replace("\\", "/"),
        nora_r=str(_NORA_R).replace("\\", "/"),
    ))
    proc = subprocess.run(
        [_RSCRIPT, str(script_path)],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, (
        f"R RDD audit exited {proc.returncode}\nstderr: {proc.stderr}"
    )
    payload = json.loads(result_path.read_text().strip().splitlines()[0])
    payload.pop("_token", None)
    res = sanitize(payload)
    assert res.ok, f"sanitizer rejected R RDD payload: {res.rejection_reason}"
    s = res.sanitized
    # Three flavors of point estimate.
    for k in ("tau_conventional", "tau_bias_corrected", "tau_robust"):
        assert k in s, f"missing {k}"
    # SEs / p-values / CIs all three flavors.
    for k in ("se_conventional", "se_bias_corrected", "se_robust",
              "p_conventional", "p_bias_corrected", "p_robust",
              "ci_lower_robust", "ci_upper_robust"):
        assert k in s
    # Per-side bandwidth + effective N.
    assert s.get("bandwidth_left") is not None
    assert s.get("bandwidth_right") is not None
    assert isinstance(s.get("effective_n_left"), int)
    assert isinstance(s.get("effective_n_right"), int)
    # Selector + kernel pass the enum check.
    assert s.get("bandwidth_selector") == "mserd"
    assert s.get("kernel") == "triangular"
    assert s.get("polynomial_order") == 1
    assert s.get("estimator") == "local_polynomial"
    assert s.get("cutoff") == 0


@requires_r_rdrobust
def test_r_from_rdd_refuses_mccrary_kwarg(tmp_path: Path) -> None:
    """R helper raises if a script passes a density/binscatter kwarg —
    structural privacy carve-out, not just sanitizer-side dropping."""
    refuse_script = r"""
Sys.setenv(NORA_RUN_TOKEN = "test-token-not-secret")
Sys.setenv(NORA_RESULT_PATH = "{result_path}")
source("{nora_r}")
suppressMessages(library(rdrobust))
set.seed(42)
n <- 400
x <- runif(n, -1, 1)
y <- 1 + 0.5*x + rnorm(n)
m <- rdrobust(y, x, c = 0)
tryCatch({{
  nora$from_rdd(m, running_variable = "x",
                mccrary_density_curve = list(c(-0.1, 0.001), c(0, 0.0012)))
  cat("FAIL_NO_ERROR\n")
}}, error = function(e) cat("ERR:", conditionMessage(e), "\n"))
"""
    result_path = tmp_path / "rdd_r_refuse.jsonl"
    script_path = tmp_path / "refuse.R"
    script_path.write_text(refuse_script.format(
        result_path=str(result_path).replace("\\", "/"),
        nora_r=str(_NORA_R).replace("\\", "/"),
    ))
    proc = subprocess.run(
        [_RSCRIPT, str(script_path)],
        capture_output=True, text=True, timeout=60,
    )
    out = proc.stdout + proc.stderr
    assert "ERR:" in out, f"helper did not raise: {out}"
    assert "mccrary_density_curve" in out
    # Critically: no payload written, even though we set the result
    # path. The helper aborted before writing.
    assert not result_path.is_file() or result_path.stat().st_size == 0


# ---------------------------------------------------------------------------
# Python via rdrobust
# ---------------------------------------------------------------------------

_PY_AUDIT_SCRIPT = """
import os, sys
sys.path.insert(0, "{runtime_dir}")
import nora as nora_runtime

import numpy as np
from rdrobust import rdrobust

rng = np.random.default_rng(42)
n = 800
x = rng.uniform(-1, 1, size=n)
treated = (x >= 0).astype(int)
y = 1 + 0.5*x + 0.3*treated + 0.2*x*treated + rng.normal(scale=0.3, size=n)
m = rdrobust(y=y, x=x, c=0)
nora_runtime.from_rdd(m, running_variable="x", outcome_variable="y",
                     label="Python RDD real-fit pin")
"""


@requires_python_rdrobust
def test_python_from_rdd_real_fit(tmp_path: Path) -> None:
    result_path = tmp_path / "rdd_py.jsonl"
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
        f"Python RDD audit exited {proc.returncode}: {proc.stderr}"
    )
    payload = json.loads(result_path.read_text().strip().splitlines()[0])
    payload.pop("_token", None)
    res = sanitize(payload)
    assert res.ok, (
        f"sanitizer rejected Python RDD payload: {res.rejection_reason}"
    )
    s = res.sanitized
    for k in ("tau_conventional", "tau_bias_corrected", "tau_robust",
              "se_conventional", "se_bias_corrected", "se_robust",
              "p_conventional", "p_bias_corrected", "p_robust",
              "bandwidth_left", "bandwidth_right",
              "effective_n_left", "effective_n_right"):
        assert k in s, f"missing {k}"
    assert s.get("bandwidth_selector") == "mserd"
    assert s.get("kernel") == "triangular"
    assert s.get("estimator") == "local_polynomial"


_PY_REFUSE_SCRIPT = """
import os, sys
sys.path.insert(0, "{runtime_dir}")
import nora as nora_runtime

import numpy as np
from rdrobust import rdrobust

rng = np.random.default_rng(42)
n = 400
x = rng.uniform(-1, 1, size=n)
y = 1 + 0.5*x + rng.normal(size=n)
m = rdrobust(y=y, x=x, c=0)
try:
    nora_runtime.from_rdd(m, running_variable="x", **{{ {banned_arg}: "anything" }})
    print("FAIL_NO_ERROR")
except ValueError as e:
    print("ERR:", str(e))
"""


@requires_python_rdrobust
@pytest.mark.parametrize("banned_kwarg", [
    "mccrary_density_curve",
    "binscatter_bins",
    "density_curve",
    "binscatter",
])
def test_python_from_rdd_refuses_density_and_binscatter_kwargs(
    tmp_path: Path, banned_kwarg: str
) -> None:
    """Structural privacy refusal — density/binscatter kwargs raise
    at the helper call site, not at sanitizer dispatch. Run as a
    subprocess to keep ``import nora`` from polluting the test
    process's ``sys.modules`` (the package ``nora`` and the runtime
    ``nora`` share a name, so direct in-process import shadows the
    package import for subsequent tests)."""
    result_path = tmp_path / "rdd_py_refuse.jsonl"
    script_path = tmp_path / "refuse.py"
    runtime_dir = (_REPO_ROOT / "src" / "nora" / "runtime").resolve()
    script_path.write_text(_PY_REFUSE_SCRIPT.format(
        runtime_dir=str(runtime_dir).replace("\\", "/"),
        banned_arg=repr(banned_kwarg),
    ))
    env = os.environ.copy()
    env["NORA_RUN_TOKEN"] = "test-token-not-secret"
    env["NORA_RESULT_PATH"] = str(result_path)
    proc = subprocess.run(
        [sys.executable, str(script_path)],
        env=env, capture_output=True, text=True, timeout=60,
    )
    out = proc.stdout + proc.stderr
    assert "ERR:" in out, f"helper did not raise: {out}"
    assert banned_kwarg in out, f"raised but wrong kwarg name: {out}"
    # Critically: no payload was written.
    assert not result_path.is_file() or result_path.stat().st_size == 0
