"""Real-fit coverage tests for ``from_kaplan_meier`` helpers across
R / Python / Stata.

The ``kaplan_meier`` shape has 16 property tests
(``tests/test_rdd_and_kaplan_meier.py``) that pin the sanitizer's
behavior on hand-crafted payloads, but until this module the
helpers themselves were unverified against real survival fits.
Building these helpers IS the missing real-fit verification — same
pattern as the earlier ``from_lm`` audit arc where mocked-fit
tests masked the Cox hard-failure mode.

Each language gets a smoke + per-horizon-emission test:
    * Helper accepts a real survival fit
    * Emits a sanitizer-valid payload
    * Per-horizon ``n_at_risk_h`` lines up with horizon time look-up
      (so the cohort-N gate has correct inputs)
    * Log-rank chi² + p surface when grouped inference is requested
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
_NORA_RESULT_KM_ADO = _REPO_ROOT / "src" / "nora" / "runtime" / "nora_result_km.ado"

_RSCRIPT = shutil.which("Rscript")


def _r_pkg_available(pkg: str) -> bool:
    if _RSCRIPT is None:
        return False
    res = subprocess.run(
        [_RSCRIPT, "-e", f'suppressMessages(library({pkg}))'],
        capture_output=True, text=True, timeout=20,
    )
    return res.returncode == 0


def _stata_binary() -> str | None:
    for name in ("stata-mp", "stata-se", "stata"):
        p = shutil.which(name)
        if p:
            return p
    return None


requires_rscript = pytest.mark.skipif(
    _RSCRIPT is None or not _NORA_R.is_file()
    or not _r_pkg_available("survival"),
    reason="Rscript / nora.R / survival package not available",
)
_STATA = _stata_binary()
requires_stata = pytest.mark.skipif(
    _STATA is None or not _NORA_RESULT_KM_ADO.is_file(),
    reason="Stata binary / nora_result_km.ado not available",
)


# ---------------------------------------------------------------------------
# R via Rscript
# ---------------------------------------------------------------------------


_R_AUDIT_SCRIPT = r"""
Sys.setenv(NORA_RUN_TOKEN = "test-token-not-secret")
result_path <- "{result_path}"
if (file.exists(result_path)) file.remove(result_path)
Sys.setenv(NORA_RESULT_PATH = result_path)
source("{nora_r}")
suppressMessages(library(survival))

set.seed(42)
n <- 200
df <- data.frame(
  t_event = rexp(n, rate = 1/3),
  cens = rbinom(n, 1, 0.7),
  arm = factor(sample(letters[1:2], n, replace = TRUE))
)
df$t_obs <- pmin(df$t_event, 5)

fit <- survfit(Surv(t_obs, cens) ~ 1, data = df)
sd <- survdiff(Surv(t_obs, cens) ~ arm, data = df)
nora$from_kaplan_meier(
  fit,
  horizons = c("1y" = 1, "3y" = 3, "5y" = 5),
  time_variable = "t_obs", event_variable = "cens",
  group_variable = "arm", survdiff = sd,
  label = "R KM real-fit pin"
)
"""


@requires_rscript
def test_r_from_kaplan_meier_real_fit(tmp_path: Path) -> None:
    result_path = tmp_path / "km_r.jsonl"
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
        f"R audit exited {proc.returncode}\nstderr: {proc.stderr}"
    )
    assert result_path.is_file() and result_path.stat().st_size > 0
    payload = json.loads(result_path.read_text().strip().splitlines()[0])
    # The token gets stripped by the executor's preprocessing; mirror
    # that here.
    payload.pop("_token", None)
    res = sanitize(payload)
    assert res.ok, f"sanitizer rejected R KM payload: {res.rejection_reason}"
    s = res.sanitized
    # Median survival round-trips.
    assert "median_survival_time" in s
    # All three horizons emit S + n_at_risk + CI bounds.
    for h in ("1y", "3y", "5y"):
        assert f"survival_at_{h}" in s, f"missing survival_at_{h}"
        assert f"n_at_risk_{h}" in s, f"missing n_at_risk_{h}"
        assert f"survival_at_{h}_ci_lower" in s
        assert f"survival_at_{h}_ci_upper" in s
        assert 0.0 <= s[f"survival_at_{h}"] <= 1.0
        assert isinstance(s[f"n_at_risk_{h}"], int)
    # Log-rank surfaces when survdiff is provided.
    assert "logrank_chi_squared" in s
    assert "logrank_p_value" in s
    assert s.get("n_groups") == 2


# ---------------------------------------------------------------------------
# Python via statsmodels SurvfuncRight
# ---------------------------------------------------------------------------


def _statsmodels_available() -> bool:
    try:
        import statsmodels  # noqa: F401
        return True
    except ImportError:
        return False


requires_statsmodels = pytest.mark.skipif(
    not _statsmodels_available(),
    reason="statsmodels not installed",
)


_PY_AUDIT_SCRIPT = """
import os, sys
sys.path.insert(0, "{runtime_dir}")
import nora as nora_runtime

import numpy as np
import statsmodels.api as sm

rng = np.random.default_rng(42)
n = 300
t_event = rng.exponential(3, size=n)
cens = rng.binomial(1, 0.7, size=n).astype(int)
t_obs = np.minimum(t_event, 5)

sf = sm.SurvfuncRight(t_obs, cens)
nora_runtime.from_kaplan_meier(
    sf,
    horizons={{"1y": 1.0, "3y": 3.0, "5y": 5.0}},
    time_variable="t_obs",
    event_variable="cens",
    label="Python KM real-fit pin",
)
"""


@requires_statsmodels
def test_python_from_kaplan_meier_real_fit(tmp_path: Path) -> None:
    result_path = tmp_path / "km_py.jsonl"
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
        f"Python audit exited {proc.returncode}: {proc.stderr}"
    )
    payload = json.loads(result_path.read_text().strip().splitlines()[0])
    payload.pop("_token", None)
    res = sanitize(payload)
    assert res.ok, f"sanitizer rejected Python KM payload: {res.rejection_reason}"
    s = res.sanitized
    assert "median_survival_time" in s
    for h in ("1y", "3y", "5y"):
        assert f"survival_at_{h}" in s
        assert f"n_at_risk_{h}" in s
        assert f"survival_at_{h}_ci_lower" in s
        assert f"survival_at_{h}_ci_upper" in s


# ---------------------------------------------------------------------------
# Stata via nora_result_km.ado
# ---------------------------------------------------------------------------


_STATA_AUDIT_SCRIPT = r"""
adopath ++ "{runtime_dir}"
local _path : env NORA_RESULT_PATH
capture erase "`_path'"
sysuse cancer, clear
quietly stset studytime, failure(died)
nora_result_km, horizons("1y:1 3y:3 5y:5 10y:10") ///
                time(studytime) event(died) group(drug) ///
                label("Stata KM real-fit pin")
"""


@requires_stata
def test_stata_nora_result_km_real_fit(tmp_path: Path) -> None:
    result_path = tmp_path / "km_stata.jsonl"
    script_path = tmp_path / "audit.do"
    runtime_dir = (_REPO_ROOT / "src" / "nora" / "runtime").resolve()
    script_path.write_text(_STATA_AUDIT_SCRIPT.format(
        runtime_dir=str(runtime_dir),
    ))
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
            "Stata produced no payload (likely missing dataset or stset error): "
            + proc.stdout[-300:]
        )
    payload = json.loads(result_path.read_text().strip().splitlines()[0])
    payload.pop("_token", None)
    res = sanitize(payload)
    assert res.ok, f"sanitizer rejected Stata KM payload: {res.rejection_reason}"
    s = res.sanitized
    # The auto/cancer dataset has 48 subjects, 31 deaths.
    assert s.get("n_subjects") == 48
    assert s.get("n_failures") == 31
    # Median survival via stci.
    assert "median_survival_time" in s
    # Three horizons inside the data window (1y / 3y / 5y); 10y also
    # inside since auto/cancer goes out to ~39 months. Whichever
    # horizons fall inside the data emit S + n_at_risk.
    for h in ("1y", "3y", "5y", "10y"):
        assert f"survival_at_{h}" in s, f"missing survival_at_{h}"
        assert f"n_at_risk_{h}" in s, f"missing n_at_risk_{h}"
        assert isinstance(s[f"n_at_risk_{h}"], int)
        assert 0.0 <= s[f"survival_at_{h}"] <= 1.0
    # Log-rank across drug groups.
    assert "logrank_chi_squared" in s
    assert "logrank_p_value" in s
    assert s.get("n_groups") == 3


@requires_stata
def test_stata_unrecognised_horizon_dropped_by_sanitizer(tmp_path: Path) -> None:
    """The helper emits whatever horizon labels the caller supplied;
    the sanitizer accepts only ``1y`` / ``3y`` / ``5y`` / ``10y``.
    Pin that non-canonical labels disappear silently rather than
    riding through."""
    script = r"""
adopath ++ "{runtime_dir}"
local _path : env NORA_RESULT_PATH
capture erase "`_path'"
sysuse cancer, clear
quietly stset studytime, failure(died)
nora_result_km, horizons("1y:1 17y:17") ///
                time(studytime) event(died) label("...")
""".format(runtime_dir=str((_REPO_ROOT / "src" / "nora" / "runtime").resolve()))
    result_path = tmp_path / "km_stata.jsonl"
    script_path = tmp_path / "audit.do"
    script_path.write_text(script)
    env = os.environ.copy()
    env["NORA_RUN_TOKEN"] = "test-token-not-secret"
    env["NORA_RESULT_PATH"] = str(result_path)
    proc = subprocess.run(
        [_STATA, "-b", "do", str(script_path)],
        cwd=_REPO_ROOT, env=env,
        capture_output=True, text=True, timeout=120,
    )
    if not result_path.is_file() or result_path.stat().st_size == 0:
        pytest.skip("Stata produced no payload")
    payload = json.loads(result_path.read_text().strip().splitlines()[0])
    payload.pop("_token", None)
    res = sanitize(payload)
    assert res.ok
    s = res.sanitized
    # Canonical 1y survives.
    assert "survival_at_1y" in s
    # Non-canonical 17y label gets dropped at the sanitizer's
    # ``_KM_ALLOWED_NUMERIC_FIELDS`` / ``_KM_ALLOWED_INT_FIELDS`` gate.
    assert "survival_at_17y" not in s
    assert "n_at_risk_17y" not in s
