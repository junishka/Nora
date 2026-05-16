"""Real-fit coverage tests for ``nora$from_lm`` (R helper).

The existing helper tests (``test_python_runtime_sanitizer.py``) use
**mocked** fit objects with OLS-shaped attributes, so they verify
"if a fit happens to have these attributes, the helper works" rather
than "a real ``glm`` / ``coxph`` / ``feols`` fit has these
attributes."  This module closes that gap by fitting real models in
R and asserting that the emitted payload **both** sanitizes cleanly
**and** carries an estimator-appropriate fit metric.

Two failure modes were observed in audit before the helper rewrite:
1.  Hard failure — Cox (``coxph``) and fixest with absorbed FE
    aborted ``from_lm`` with "undefined columns selected" because the
    helper assumed lm/glm-shaped column names ("Estimate", "Std.
    Error", "t value" / "z value").  No payload reached the model.
2.  Degraded payload — GLM fits (logit / probit / Poisson / negative
    binomial) sanitized fine but carried zero estimator-appropriate
    fit metrics (no ``pseudo_r_squared``, ``log_likelihood``, ``aic``,
    or ``bic``), forcing the model to either ask for them (wasting a
    turn) or proceed without them (worse analysis).

Each test below pins one estimator against both failure modes:
    * ``ok=True`` from the sanitizer
    * at least one ``EXPECTED_FIT_METRICS[estimator]`` field present
      in the sanitized output.

Estimators that need a specific R package skip cleanly when that
package isn't installed on the test machine.
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

_RSCRIPT = shutil.which("Rscript")
_REPO_ROOT = Path(__file__).resolve().parents[1]
_NORA_R = _REPO_ROOT / "src" / "nora" / "runtime" / "nora.R"


requires_rscript = pytest.mark.skipif(
    _RSCRIPT is None or not _NORA_R.is_file(),
    reason="Rscript not on PATH or nora.R not present",
)


def _r_pkg_available(pkg: str) -> bool:
    """True if ``library(pkg)`` succeeds in a one-shot Rscript."""
    if _RSCRIPT is None:
        return False
    res = subprocess.run(
        [_RSCRIPT, "-e", f'suppressMessages(library({pkg}))'],
        capture_output=True, text=True, timeout=20,
    )
    return res.returncode == 0


# Minimum set of fields each estimator's payload should carry so the
# model can evaluate model adequacy on the first turn (rather than
# round-tripping for it via ``request_data``).
EXPECTED_FIT_METRICS: dict[str, tuple[str, ...]] = {
    "ols":      ("r_squared",),
    "logit":    ("pseudo_r_squared", "log_likelihood", "aic"),
    "probit":   ("pseudo_r_squared", "log_likelihood", "aic"),
    "poisson":  ("pseudo_r_squared", "log_likelihood", "aic"),
    "negbin":   ("pseudo_r_squared", "log_likelihood", "aic"),
    "cox":      ("concordance", "log_likelihood", "n_failures"),
    "feols_fe": ("r_squared", "fixed_effects", "log_likelihood"),
}


_AUDIT_SCRIPT = r"""
Sys.setenv(NORA_RUN_TOKEN = "test-token-not-secret")
result_path <- "{result_path}"
if (file.exists(result_path)) file.remove(result_path)
Sys.setenv(NORA_RESULT_PATH = result_path)

source("{nora_r}")

set.seed(42)
n <- 400
df <- data.frame(
  x1 = rnorm(n),
  x2 = rnorm(n),
  group = factor(sample(letters[1:3], n, replace = TRUE))
)
df$y_cont <- 1 + 0.5 * df$x1 - 0.3 * df$x2 + rnorm(n)
df$y_bin  <- as.integer(plogis(0.3 * df$x1 - 0.5 * df$x2) > runif(n))
df$y_count <- rpois(n, lambda = exp(0.2 + 0.1 * df$x1))
df$t_event <- rexp(n, rate = exp(-0.3 + 0.4 * df$x1))
df$cens <- as.integer(df$t_event < 2)
df$t_obs <- pmin(df$t_event, 2)

# Mark each payload with a label sentinel so the Python side can map
# payload -> estimator without relying on emission order.
emit <- function(label, fn) {{
  fit <- tryCatch(fn(), error = function(e) {{
    cat(sprintf("FIT-ERROR %s: %s\n", label, conditionMessage(e)), file = stderr())
    return(NULL)
  }})
  if (is.null(fit)) return(invisible(NULL))
  ok <- tryCatch({{ nora$from_lm(fit); TRUE }}, error = function(e) {{
    cat(sprintf("HELPER-ERROR %s: %s\n", label, conditionMessage(e)), file = stderr())
    FALSE
  }})
  if (ok) {{
    con <- file(result_path, open = "a", encoding = "UTF-8")
    writeLines(sprintf('{{"_audit_label": "%s"}}', label), con)
    close(con)
  }}
}}

{tasks}
"""


_TASK_BY_LABEL: dict[str, tuple[str | None, str]] = {
    # (required_pkg or None for base R, R-snippet that returns a fit)
    "ols":      (None,       "function() lm(y_cont ~ x1 + x2, data = df)"),
    "logit":    (None,       "function() glm(y_bin ~ x1 + x2, family = binomial, data = df)"),
    "probit":   (None,       "function() glm(y_bin ~ x1 + x2, family = binomial(link = \"probit\"), data = df)"),
    "poisson":  (None,       "function() glm(y_count ~ x1 + x2, family = poisson, data = df)"),
    "negbin":   ("MASS",     "function() {suppressMessages(library(MASS)); MASS::glm.nb(y_count ~ x1 + x2, data = df)}"),
    "cox":      ("survival", "function() {suppressMessages(library(survival)); coxph(Surv(t_obs, cens) ~ x1 + x2, data = df)}"),
    "feols_fe": ("fixest",   "function() {suppressMessages(library(fixest)); feols(y_cont ~ x1 + x2 | group, data = df, cluster = ~group)}"),
}


@pytest.fixture(scope="module")
def real_fit_payloads(tmp_path_factory) -> dict[str, dict]:
    """Run the audit R script once, return a {label: sanitized_payload}
    dict for the labels whose packages are installed."""
    if _RSCRIPT is None:
        pytest.skip("Rscript not on PATH")
    runnable_labels = [
        lbl for lbl, (pkg, _src) in _TASK_BY_LABEL.items()
        if pkg is None or _r_pkg_available(pkg)
    ]
    if not runnable_labels:
        pytest.skip("no runnable R estimators on this machine")

    tmp = tmp_path_factory.mktemp("from_lm_audit")
    result_path = tmp / "payloads.jsonl"
    tasks = "\n".join(
        f'emit("{lbl}", {_TASK_BY_LABEL[lbl][1]})' for lbl in runnable_labels
    )
    script = _AUDIT_SCRIPT.format(
        result_path=str(result_path).replace("\\", "/"),
        nora_r=str(_NORA_R).replace("\\", "/"),
        tasks=tasks,
    )
    script_path = tmp / "audit.R"
    script_path.write_text(script)
    proc = subprocess.run(
        [_RSCRIPT, str(script_path)],
        capture_output=True, text=True, timeout=120,
    )
    if proc.returncode != 0:
        pytest.fail(
            f"audit Rscript exited {proc.returncode}\n"
            f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}"
        )
    if not result_path.is_file() or result_path.stat().st_size == 0:
        pytest.skip("no payloads emitted — likely missing R packages mid-run")

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


@requires_rscript
@pytest.mark.parametrize("estimator", list(_TASK_BY_LABEL.keys()))
def test_from_lm_emits_sanitizer_valid_payload(
    estimator: str, real_fit_payloads: dict[str, dict]
) -> None:
    """The helper must emit a payload that the sanitizer accepts.

    Before the per-class-dispatch rewrite, Cox and fixest-with-FE
    aborted before writing any payload — the corresponding entries
    were absent from ``real_fit_payloads`` and this assertion failed
    on missing payloads, which is the hard-failure regression we
    explicitly pin against.
    """
    pkg, _src = _TASK_BY_LABEL[estimator]
    if pkg is not None and not _r_pkg_available(pkg):
        pytest.skip(f"R package {pkg!r} not installed")
    assert estimator in real_fit_payloads, (
        f"{estimator}: no payload emitted (helper aborted?)"
    )
    payload = real_fit_payloads[estimator]
    res = sanitize(payload)
    assert res.ok, (
        f"{estimator}: sanitizer rejected payload: {res.rejection_reason}"
    )
    # Helpers now emit the canonical descriptive name; the legacy
    # alias is still accepted for back-compat with stored payloads.
    assert res.analysis_type in (
        "coefficient_table_with_fit_stats", "linear_regression",
    )


@requires_rscript
@pytest.mark.parametrize("estimator", list(_TASK_BY_LABEL.keys()))
def test_from_lm_payload_is_inference_adequate(
    estimator: str, real_fit_payloads: dict[str, dict]
) -> None:
    """Beyond sanitizer-passing, the payload must carry at least one
    estimator-appropriate fit metric so the model can evaluate model
    adequacy without an additional turn.

    This is the second failure mode the audit uncovered: GLM fits
    sanitized cleanly but shipped zero ``pseudo_r_squared`` /
    ``log_likelihood`` / ``aic`` because the R helper computed
    ``s$r.squared`` from a glm summary (always NULL) and never
    reached for the GLM-specific fields the sanitizer's allowlist
    accepts.
    """
    pkg, _src = _TASK_BY_LABEL[estimator]
    if pkg is not None and not _r_pkg_available(pkg):
        pytest.skip(f"R package {pkg!r} not installed")
    assert estimator in real_fit_payloads, f"{estimator}: missing payload"
    res = sanitize(real_fit_payloads[estimator])
    expected = EXPECTED_FIT_METRICS[estimator]
    present = [k for k in expected if k in (res.sanitized or {})]
    assert present, (
        f"{estimator}: no expected fit metric present. "
        f"Expected at least one of {expected}, "
        f"got fields {sorted(res.sanitized.keys()) if res.sanitized else '[]'}"
    )


@requires_rscript
def test_feols_emits_fixed_effects_cardinality(
    real_fit_payloads: dict[str, dict],
) -> None:
    """fixest FE absorption: cardinality reported, levels NOT reported.

    ``fixed_effects: {group: 3}`` is the disclosure-safe metadata
    surface — the model learns the FE dimension was absorbed and
    knows its level count without ever seeing the level labels.
    """
    if not _r_pkg_available("fixest"):
        pytest.skip("R package 'fixest' not installed")
    assert "feols_fe" in real_fit_payloads, "feols_fe payload missing"
    res = sanitize(real_fit_payloads["feols_fe"])
    fe = (res.sanitized or {}).get("fixed_effects")
    assert isinstance(fe, dict) and len(fe) >= 1, (
        f"expected non-empty fixed_effects dict, got {fe!r}"
    )
    # Values must be positive integer cardinalities.
    for k, v in fe.items():
        assert isinstance(v, int) and v > 0, (
            f"fixed_effects[{k!r}] should be a positive int, got {v!r}"
        )


@requires_rscript
def test_cox_emits_subject_and_failure_counts(
    real_fit_payloads: dict[str, dict],
) -> None:
    """Cox PH-specific metadata: ``n_subjects`` and ``n_failures``
    must surface so the model can report the standard "N subjects,
    M events" header researchers expect on survival tables."""
    if not _r_pkg_available("survival"):
        pytest.skip("R package 'survival' not installed")
    assert "cox" in real_fit_payloads, "cox payload missing"
    res = sanitize(real_fit_payloads["cox"])
    s = res.sanitized or {}
    assert "n_subjects" in s and isinstance(s["n_subjects"], int)
    assert "n_failures" in s and isinstance(s["n_failures"], int)
    assert s["n_failures"] <= s["n_subjects"]


@requires_rscript
def test_feols_emits_cluster_metadata_when_cluster_robust(
    real_fit_payloads: dict[str, dict],
) -> None:
    """The audit fits ``feols(y ~ x | g, cluster=~g)`` — the helper
    must emit ``cluster_variables: ["group"]`` and
    ``n_clusters: {"group": 3}`` so the model knows the SE treatment
    and the cluster cardinality without ever seeing cluster labels."""
    if not _r_pkg_available("fixest"):
        pytest.skip("R package 'fixest' not installed")
    assert "feols_fe" in real_fit_payloads
    s = sanitize(real_fit_payloads["feols_fe"]).sanitized or {}
    assert s.get("robust_se_type") == "cluster"
    assert s.get("cluster_variables") == ["group"]
    nc = s.get("n_clusters")
    assert isinstance(nc, dict) and nc.get("group") == 3, (
        f"expected n_clusters={{group: 3}}, got {nc!r}"
    )


@requires_rscript
def test_glm_does_not_emit_r_squared(real_fit_payloads: dict[str, dict]) -> None:
    """GLM fits should NOT carry ``r_squared`` — that field is
    OLS-specific. The old helper emitted it as ``NULL`` (which the
    sanitizer stripped with a noisy transformation note on every
    GLM payload); the rewrite suppresses the field at the helper
    level for non-OLS classes."""
    for label in ("logit", "probit", "poisson"):
        if label not in real_fit_payloads:
            continue
        res = sanitize(real_fit_payloads[label])
        s = res.sanitized or {}
        assert "r_squared" not in s, (
            f"{label}: r_squared leaked into GLM payload: {s.get('r_squared')!r}"
        )
