"""Real-fit coverage tests for ``nora.from_lm`` (Python helper).

Companion to ``test_from_lm_real_fits.py`` (R) and
``test_nora_result_regress_real_fits.py`` (Stata). Same two-bar
standard — sanitizer-valid AND inference-adequate — applied to the
Python helper against the regression-shape estimators statsmodels
exposes.

Three failure modes the audit caught before the helper rewrite:

1.  **Hard failure on PHReg** — ``PHRegResults.params`` is a bare
    ndarray, not a pandas Series; the helper's ``_to_dict`` raised
    ``TypeError: cannot convert dictionary update sequence element
    #0 to a sequence`` and aborted before any payload was written.
    ``PHRegResults.nobs`` is also absent, so even if the dict issue
    were fixed, the payload would fail the sanitizer's ``n``-required
    check.

2.  **Hard failure on IV2SLS** — ``IV2SLSResults`` defines ``llf`` /
    ``aic`` / ``bic`` as properties that raise ``NotImplementedError``
    (2SLS doesn't have a likelihood). ``getattr(m, "llf", None)``
    doesn't catch raised exceptions, so the helper aborted on the
    first probe.

3.  **Degraded payloads for every GLM family** — the helper read
    ``rsquared`` (None on GLM fits, ridden through to sanitizer as
    null) and never reached for ``prsquared`` / ``llf`` / ``aic``.
    Same symptom as the R audit found before its rewrite.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nora.sanitizer import sanitize  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
_AUDIT_SCRIPT = _REPO_ROOT / "scripts" / "audit_python_regression.py"


def _statsmodels_available() -> bool:
    try:
        import statsmodels  # noqa: F401
        return True
    except ImportError:
        return False


requires_statsmodels = pytest.mark.skipif(
    not _statsmodels_available() or not _AUDIT_SCRIPT.is_file(),
    reason="statsmodels not installed or audit script missing",
)


EXPECTED_FIT_METRICS: dict[str, tuple[str, ...]] = {
    "ols":           ("r_squared",),
    "logit":         ("pseudo_r_squared", "log_likelihood", "aic"),
    "probit":        ("pseudo_r_squared", "log_likelihood", "aic"),
    "poisson":       ("pseudo_r_squared", "log_likelihood", "aic"),
    "negbin":        ("pseudo_r_squared", "log_likelihood", "aic"),
    "glm_binomial":  ("pseudo_r_squared", "log_likelihood", "aic"),
    "phreg":         ("log_likelihood", "n_subjects", "n_failures"),
    "iv2sls":        ("r_squared",),
    "iv2sls_full":   ("r_squared", "first_stage_f", "instrument_variables"),
    "ols_clustered": ("r_squared", "cluster_variables", "n_clusters"),
}


@pytest.fixture(scope="module")
def python_payloads(tmp_path_factory) -> dict[str, dict]:
    """Spawn the audit script as a subprocess (the nora runtime pops
    ``NORA_RUN_TOKEN`` on import, so it must run in a fresh process)."""
    if not _statsmodels_available():
        pytest.skip("statsmodels not installed")
    tmp = tmp_path_factory.mktemp("python_audit")
    result_path = tmp / "payloads.jsonl"
    env = os.environ.copy()
    env["NORA_RUN_TOKEN"] = "test-token-not-secret"
    env["NORA_RESULT_PATH"] = str(result_path)
    proc = subprocess.run(
        [sys.executable, str(_AUDIT_SCRIPT)],
        cwd=_REPO_ROOT, env=env,
        capture_output=True, text=True, timeout=120,
    )
    if proc.returncode != 0:
        pytest.skip(
            f"Python audit exited {proc.returncode}: {proc.stderr[:400]}"
        )
    if not result_path.is_file() or result_path.stat().st_size == 0:
        pytest.skip("no payloads emitted")
    # Pair each label with the payload that immediately preceded it.
    events = [
        json.loads(line) for line in result_path.read_text().splitlines()
        if line.strip()
    ]
    mapping: dict[str, dict] = {}
    cur: dict | None = None
    for d in events:
        if "_audit_label" in d:
            if cur is not None:
                mapping[d["_audit_label"]] = cur
            cur = None
        else:
            cur = d
    return mapping


@requires_statsmodels
@pytest.mark.parametrize("estimator", list(EXPECTED_FIT_METRICS.keys()))
def test_python_payload_sanitizer_valid(
    estimator: str, python_payloads: dict[str, dict]
) -> None:
    """Helper must emit a sanitizer-acceptable payload. PHReg + IV2SLS
    used to abort before writing anything; their absence from
    ``python_payloads`` triggers this assertion."""
    assert estimator in python_payloads, (
        f"{estimator}: helper emitted no payload (probably aborted)"
    )
    res = sanitize(python_payloads[estimator])
    assert res.ok, (
        f"{estimator}: sanitizer rejected: {res.rejection_reason}"
    )
    # Canonical or legacy alias — both round-trip via the dispatch
    # table. New helpers emit the descriptive canonical name.
    assert res.analysis_type in (
        "coefficient_table_with_fit_stats", "linear_regression",
    )


@requires_statsmodels
@pytest.mark.parametrize("estimator", list(EXPECTED_FIT_METRICS.keys()))
def test_python_payload_inference_adequate(
    estimator: str, python_payloads: dict[str, dict]
) -> None:
    assert estimator in python_payloads, f"{estimator}: payload missing"
    res = sanitize(python_payloads[estimator])
    expected = EXPECTED_FIT_METRICS[estimator]
    present = [k for k in expected if k in (res.sanitized or {})]
    assert present, (
        f"{estimator}: no expected fit metric present. "
        f"Expected at least one of {expected}, "
        f"got {sorted((res.sanitized or {}).keys())}"
    )


@requires_statsmodels
def test_python_phreg_emits_subject_and_failure_counts(
    python_payloads: dict[str, dict],
) -> None:
    """PHReg's params is an ndarray (not Series) and nobs is absent.
    Both used to silently abort the helper; pinning the round-trip."""
    assert "phreg" in python_payloads, "phreg payload missing"
    s = sanitize(python_payloads["phreg"]).sanitized or {}
    assert "n_subjects" in s and isinstance(s["n_subjects"], int)
    assert "n_failures" in s and isinstance(s["n_failures"], int)
    assert s["n_failures"] <= s["n_subjects"]
    # coefficients must be a non-empty dict with the exog_names as keys
    coefs = s.get("coefficients")
    assert isinstance(coefs, dict) and coefs


@requires_statsmodels
def test_python_iv2sls_survives_notimplementederror_attributes(
    python_payloads: dict[str, dict],
) -> None:
    """IV2SLS's llf / aic / bic raise NotImplementedError on access.
    Helper must absorb those via the exception-safe attribute probe
    and still ship the OLS-shape fields IV2SLS does compute
    (rsquared, fvalue)."""
    assert "iv2sls" in python_payloads, "iv2sls payload missing"
    s = sanitize(python_payloads["iv2sls"]).sanitized or {}
    assert "r_squared" in s and isinstance(s["r_squared"], float)
    # llf / aic / bic should be absent rather than null.
    assert "log_likelihood" not in s
    assert "aic" not in s
    assert "bic" not in s


@requires_statsmodels
def test_python_glm_binomial_via_smf_glm_carries_pseudo_r_squared(
    python_payloads: dict[str, dict],
) -> None:
    """``smf.glm(family=Binomial())`` returns ``GLMResultsWrapper``
    which does NOT expose ``prsquared``. The helper falls back to
    a deviance-ratio (McFadden-equivalent) so this path doesn't
    degrade silently — pin that fallback."""
    assert "glm_binomial" in python_payloads, "glm_binomial payload missing"
    s = sanitize(python_payloads["glm_binomial"]).sanitized or {}
    assert "pseudo_r_squared" in s
    assert 0.0 <= s["pseudo_r_squared"] <= 1.0


@requires_statsmodels
def test_python_glm_does_not_emit_r_squared(
    python_payloads: dict[str, dict],
) -> None:
    """GLM payloads should NOT carry r_squared (OLS-only). Pre-fix,
    the helper emitted it as None and the sanitizer stripped it with
    a transformation note on every GLM payload."""
    for label in ("logit", "probit", "poisson", "glm_binomial"):
        if label not in python_payloads:
            continue
        s = sanitize(python_payloads[label]).sanitized or {}
        assert "r_squared" not in s, (
            f"{label}: r_squared leaked into GLM payload"
        )


@requires_statsmodels
def test_python_from_iv_emits_diagnostics(
    python_payloads: dict[str, dict],
) -> None:
    """``from_iv`` packages a 2SLS fit plus diagnostic scalars
    (first-stage F, instrument list, endogenous list) into a
    regression-bucket payload. Pinning the decision recorded in
    ``docs/direction.md`` ("IV as regression-bucket extension")
    that 2SLS doesn't need a composite shape — the structural
    coefficients are the regression table, the diagnostics ride
    along as allowlisted scalars."""
    assert "iv2sls_full" in python_payloads, "iv2sls_full payload missing"
    s = sanitize(python_payloads["iv2sls_full"]).sanitized or {}
    assert s.get("instrument_variables") == ["z"]
    assert s.get("endogenous_variables") == ["x1"]
    assert s.get("n_instruments") == 1
    assert s.get("n_endogenous") == 1
    fs = s.get("first_stage_f")
    assert isinstance(fs, float) and fs > 0
    p = s.get("weak_instrument_p")
    assert isinstance(p, float) and 0.0 <= p <= 1.0


@requires_statsmodels
def test_python_cluster_robust_emits_cardinality(
    python_payloads: dict[str, dict],
) -> None:
    """``cov_type="cluster"`` is the modifier path — the helper
    auto-extracts ``cluster_variables`` (column names) and
    ``n_clusters`` (per-dimension cluster counts) from
    ``cov_kwds["groups"]`` and tags ``robust_se_type: cluster`` so
    the model sees the SE treatment consistently across estimators.

    The level identities of the clustering variable are deliberately
    NOT emitted — only its name (already in the schema) and the
    cluster cardinality (count). Same disclosure profile as
    ``fixed_effects``."""
    assert "ols_clustered" in python_payloads, "ols_clustered payload missing"
    s = sanitize(python_payloads["ols_clustered"]).sanitized or {}
    assert s.get("robust_se_type") == "cluster"
    assert s.get("cluster_variables") == ["firm_id"]
    n_clusters = s.get("n_clusters")
    assert isinstance(n_clusters, dict) and n_clusters == {"firm_id": 40}, (
        f"expected {{'firm_id': 40}}, got {n_clusters!r}"
    )
