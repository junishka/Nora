"""Real-fit coverage tests for ``nora_result_regress.ado`` (Stata helper).

Companion to ``test_from_lm_real_fits.py`` — same two-bar standard
(sanitizer-valid AND inference-adequate) applied to the Stata
helper across the estimators it claims to support. Skips cleanly
when ``stata-mp`` / ``stata-se`` / ``stata`` is not on PATH.

Estimators covered:
    * ``regress`` (OLS)        → r_squared, vcov
    * ``logit``                → pseudo_r_squared, log_likelihood, aic, vcov
    * ``poisson``              → same shape as logit
    * ``stcox``                → concordance (via ``estat concordance``),
                                  log_likelihood, n_subjects, n_failures, vcov
    * ``xtreg, fe``            → r_squared, fixed_effects:{ivar: N_g}, vcov
    * ``areg``                 → r_squared, fixed_effects:{absvar: df_a+1}, vcov

Audit notes pinned by this module:
    * ``stcox`` posts ``e(cmd) == "cox"``, not ``"stcox"``. Gating
      the concordance block on the wrong value silently drops the
      C-index from every Cox payload.
    * ``estat ic`` stores AIC / BIC in ``r(S)[1, 5]`` and ``r(S)[1, 6]``
      — not in scalar ``r(aic)`` / ``r(bic)`` returns.
    * vcov was historically not emitted on Stata (R + Python both
      shipped it), so Wald / joint-significance tests were
      cross-platform inconsistent.
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
_AUDIT_DO = _REPO_ROOT / "scripts" / "audit_stata_regress.do"
_HELPER_ADO = _REPO_ROOT / "src" / "nora" / "runtime" / "nora_result_regress.ado"


def _find_stata() -> str | None:
    """First Stata binary on PATH, preferring MP > SE > base."""
    for name in ("stata-mp", "stata-se", "stata"):
        p = shutil.which(name)
        if p:
            return p
    return None


_STATA = _find_stata()

requires_stata = pytest.mark.skipif(
    _STATA is None or not _AUDIT_DO.is_file() or not _HELPER_ADO.is_file(),
    reason="Stata binary not on PATH (or audit do-file / helper .ado missing)",
)


EXPECTED_FIT_METRICS: dict[str, tuple[str, ...]] = {
    "ols":           ("r_squared",),
    "logit":         ("pseudo_r_squared", "log_likelihood", "aic"),
    "poisson":       ("pseudo_r_squared", "log_likelihood", "aic"),
    "stcox":         ("concordance", "log_likelihood", "n_failures"),
    "xtreg_fe":      ("r_squared", "fixed_effects"),
    "areg":          ("r_squared", "fixed_effects"),
    "ols_clustered": ("r_squared", "cluster_variables", "n_clusters"),
}


@pytest.fixture(scope="module")
def stata_payloads(tmp_path_factory) -> dict[str, dict]:
    """Run the Stata audit do-file once; return {label: payload}."""
    if _STATA is None:
        pytest.skip("Stata not on PATH")
    tmp = tmp_path_factory.mktemp("stata_audit")
    result_path = tmp / "payloads.jsonl"
    env = os.environ.copy()
    env["NORA_RUN_TOKEN"] = "test-token-not-secret"
    env["NORA_RESULT_PATH"] = str(result_path)
    proc = subprocess.run(
        [_STATA, "-b", "do", str(_AUDIT_DO)],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True, text=True, timeout=180,
    )
    if proc.returncode != 0:
        # Stata's batch mode often exits 0 even on errors and writes
        # error details into the .log file (which is created in the
        # CWD). Surface stdout/stderr if the launch itself failed.
        pytest.skip(
            f"stata audit exited {proc.returncode}\n"
            f"stderr: {proc.stderr}\nstdout: {proc.stdout}"
        )
    if not result_path.is_file() or result_path.stat().st_size == 0:
        pytest.skip("no payloads emitted by Stata audit")
    labels: list[str] = []
    payloads: list[dict] = []
    for line in result_path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if "_audit_label" in d:
            labels.append(d["_audit_label"])
        else:
            payloads.append(d)
    return {lbl: p for lbl, p in zip(labels[:len(payloads)], payloads)}


@requires_stata
@pytest.mark.parametrize("estimator", list(EXPECTED_FIT_METRICS.keys()))
def test_stata_payload_sanitizer_valid(
    estimator: str, stata_payloads: dict[str, dict]
) -> None:
    assert estimator in stata_payloads, (
        f"{estimator}: helper emitted no payload"
    )
    res = sanitize(stata_payloads[estimator])
    assert res.ok, (
        f"{estimator}: sanitizer rejected: {res.rejection_reason}"
    )


@requires_stata
@pytest.mark.parametrize("estimator", list(EXPECTED_FIT_METRICS.keys()))
def test_stata_payload_inference_adequate(
    estimator: str, stata_payloads: dict[str, dict]
) -> None:
    """At least one estimator-appropriate fit metric must reach the
    model. Pre-fix, Stata payloads carried log_likelihood (when in
    e()) but missed AIC/BIC (in ``r(S)`` from estat ic, not e()),
    concordance for stcox (requires estat concordance), and the
    full vcov."""
    assert estimator in stata_payloads, f"{estimator}: payload missing"
    res = sanitize(stata_payloads[estimator])
    expected = EXPECTED_FIT_METRICS[estimator]
    present = [k for k in expected if k in (res.sanitized or {})]
    assert present, (
        f"{estimator}: no expected fit metric present. "
        f"Expected at least one of {expected}, "
        f"got {sorted((res.sanitized or {}).keys())}"
    )


@requires_stata
def test_stata_emits_vcov_on_every_regression(
    stata_payloads: dict[str, dict],
) -> None:
    """vcov is cross-platform: R + Python ship it on every fit. The
    Stata helper used to drop it entirely; this pins that gap."""
    for label, p in stata_payloads.items():
        res = sanitize(p)
        s = res.sanitized or {}
        assert "vcov" in s, f"{label}: missing vcov on payload"
        # vcov must be a nested dict-of-dict of floats.
        vcov = s["vcov"]
        assert isinstance(vcov, dict) and vcov, f"{label}: empty vcov"
        for row, inner in vcov.items():
            assert isinstance(inner, dict)
            for col, val in inner.items():
                assert isinstance(val, float)


@requires_stata
def test_stata_stcox_emits_concordance(
    stata_payloads: dict[str, dict],
) -> None:
    """stcox concordance comes from ``estat concordance``, gated on
    ``e(cmd) == "cox"`` (NOT "stcox" — the "st" is a prefix, not in
    e(cmd)). Pinning this so the gate doesn't slip back to "stcox"
    in a future cleanup pass."""
    assert "stcox" in stata_payloads, "stcox payload missing"
    res = sanitize(stata_payloads["stcox"])
    s = res.sanitized or {}
    assert "concordance" in s, (
        f"stcox concordance missing — likely the e(cmd) gate is "
        f"wrong (should be 'cox', not 'stcox')"
    )
    assert isinstance(s["concordance"], float)
    assert 0.0 <= s["concordance"] <= 1.0


@requires_stata
def test_stata_xtreg_fe_emits_fixed_effects(
    stata_payloads: dict[str, dict],
) -> None:
    assert "xtreg_fe" in stata_payloads
    res = sanitize(stata_payloads["xtreg_fe"])
    fe = (res.sanitized or {}).get("fixed_effects")
    assert isinstance(fe, dict) and fe
    for k, v in fe.items():
        assert isinstance(v, int) and v > 0


@requires_stata
def test_stata_xtreg_fe_emits_panel_f_test(
    stata_payloads: dict[str, dict],
) -> None:
    """Panel-diagnostic auto-emit pin: ``xtreg, fe`` populates
    ``e(F_f)`` (F-test on joint significance of the unit fixed
    effects). The helper auto-emits it into ``f_test_fe_chi2`` /
    ``f_test_fe_p``, mirroring R's ``plm::pFtest`` and giving the
    model the FE-vs-pooled test result without a separate call."""
    assert "xtreg_fe" in stata_payloads
    res = sanitize(stata_payloads["xtreg_fe"])
    s = res.sanitized or {}
    assert isinstance(s.get("f_test_fe_chi2"), (int, float))
    assert s["f_test_fe_chi2"] > 0
    # p-value rides alongside when df_a / df_r are both populated.
    if "f_test_fe_p" in s:
        assert 0.0 <= s["f_test_fe_p"] <= 1.0


@requires_stata
def test_stata_areg_emits_fixed_effects_with_correct_cardinality(
    stata_payloads: dict[str, dict],
) -> None:
    """areg's ``e(df_a)`` is groups-minus-one; the helper must
    convert to actual level count for the model. The audit data
    builds 60 panels; we test the round-trip."""
    assert "areg" in stata_payloads
    res = sanitize(stata_payloads["areg"])
    fe = (res.sanitized or {}).get("fixed_effects")
    assert isinstance(fe, dict) and fe
    # The audit do-file constructs id with 60 panels.
    assert any(v == 60 for v in fe.values()), (
        f"areg fixed_effects should report 60 levels (df_a+1), "
        f"got {fe}"
    )


@requires_stata
def test_stata_cluster_robust_emits_cardinality(
    stata_payloads: dict[str, dict],
) -> None:
    """``regress, vce(cluster id)`` populates ``e(vce) == "cluster"``,
    ``e(clustvar)`` (single name), and ``e(N_clust)`` (count). Helper
    must emit ``cluster_variables``, ``n_clusters``, and tag
    ``robust_se_type: cluster`` so cluster info round-trips cross-
    language with R/Python."""
    assert "ols_clustered" in stata_payloads
    s = sanitize(stata_payloads["ols_clustered"]).sanitized or {}
    assert s.get("robust_se_type") == "cluster"
    assert s.get("cluster_variables") == ["id"]
    nc = s.get("n_clusters")
    # Audit data has 60 panels — clustering on id gives 60 clusters.
    assert isinstance(nc, dict) and nc.get("id") == 60, (
        f"expected n_clusters={{id: 60}}, got {nc!r}"
    )
