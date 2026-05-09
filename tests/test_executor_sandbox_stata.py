"""Integration tests for the executor's sandbox — Stata variant.

Mirror of ``test_executor_sandbox.py`` but exercising Stata as the
interpreter. Gated on ``find_stata() is not None`` AND the
sandbox-apply preflight succeeding, so these will skip cleanly on CI
(Stata is commercial; license files generally aren't on CI runners)
and on nested-sandbox developer environments.

Keeping these in a separate file (vs extending the R tests) so gate
conditions compose cleanly — R-only machines skip nothing; Stata-
only (unusual) machines exercise just the Stata suite; machines with
both run both.

See also ``docs/verification.md`` for the manual recipe a developer
should run before a release when CI can't cover Stata.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from nora.env_detect import find_sandbox_exec, find_stata
from nora.executor import run_script


def _sandbox_apply_works() -> bool:
    exe = find_sandbox_exec()
    if exe is None:
        return False
    try:
        r = subprocess.run(
            [exe, "-p", "(version 1)(allow default)", "/usr/bin/true"],
            capture_output=True, timeout=5, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return r.returncode == 0


requires_stata = pytest.mark.skipif(
    find_stata() is None,
    reason=(
        "Stata not installed on this machine (commercial license). Run "
        "manually on a Stata-equipped dev machine via the recipe in "
        "docs/verification.md."
    ),
)


@pytest.fixture
def tmp_path_home(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Test cwd under ``~/.nora-sessions/`` instead of pytest's default
    ``/var/folders/...`` tempdir.

    Production sessions live in ``~/.nora-sessions/<id>/`` where the
    sandbox profile's other allows don't overlap. The default
    ``tmp_path`` fixture places the cwd under ``/private/var/folders``
    which IS covered by a broader allow (R/Stata need $TMPDIR scratch
    access), masking the ``.nora`` deny carve-out. Tests that exercise
    the carve-out must use a session-realistic cwd.

    Created and torn down per-test under
    ``~/.nora-sessions/_pytest_<random>/`` so concurrent tests don't
    collide."""
    import shutil
    import uuid
    from nora.ui import SESSIONS_ROOT
    SESSIONS_ROOT.mkdir(parents=True, exist_ok=True)
    name = f"_pytest_{uuid.uuid4().hex[:8]}"
    path = SESSIONS_ROOT / name
    path.mkdir(parents=True, exist_ok=True)
    yield path
    shutil.rmtree(path, ignore_errors=True)

requires_sandbox_apply = pytest.mark.skipif(
    sys.platform != "darwin" or not _sandbox_apply_works(),
    reason=(
        "sandbox-exec cannot apply a profile in this environment "
        "(non-macOS or nested-sandbox harness)."
    ),
)


# ---------------------------------------------------------------------------
# Happy path — sandbox lets Stata do its job
# ---------------------------------------------------------------------------

@requires_sandbox_apply
@requires_stata
def test_stata_sandbox_allows_runtime_write(tmp_path: Path):
    """The profile must let Stata run a regression via the runtime
    library and emit a result payload. Uses `sysuse auto`, which
    loads Stata's bundled dataset — no cwd-file setup needed."""
    code = '''
sysuse auto, clear
regress price mpg
nora_result_regress, label("stata-happy-path")
'''
    r = run_script("Stata", code, tmp_path)
    assert r.ok, f"Stata script failed: error={r.error}\nstdout tail={r.raw_stdout[-500:]}"
    assert r.result_payloads
    payload = r.result_payloads[0]
    assert payload["type"] == "linear_regression"
    assert payload["n"] == 74
    pvals = payload.get("p_values")
    assert isinstance(pvals, dict) and set(pvals) == {"mpg", "_cons"}
    for term, p in pvals.items():
        assert isinstance(p, float) and 0.0 <= p <= 1.0, f"{term}={p!r}"

    # No category-error fields: ``regress`` does not run a
    # chi-squared test, so neither ``chi_squared`` nor
    # ``chi_squared_p_value`` may appear in an OLS payload. The
    # helper gates these on ``e(chi2)`` being populated, which
    # ``regress`` leaves empty. (Stata's ``regress`` also does not
    # populate ``e(p)`` for OLS — the F-test p-value lives in the
    # display output as ``Prob > F`` but is not stored in ``e()``,
    # which is why this test does not assert ``f_p_value``.)
    assert "f_statistic" in payload, (
        "regress emits e(F); the helper must surface it as f_statistic"
    )
    assert "chi_squared" not in payload, (
        "regress does not run a chi-squared test; e(chi2) is empty, "
        "so no chi_squared field should appear"
    )
    assert "chi_squared_p_value" not in payload, (
        "regress does not run a chi-squared test; chi_squared_p_value "
        "must NOT leak into an OLS payload — even if a future Stata "
        "version starts populating e(p) for regress, the e(chi2) gate "
        "keeps the fields disjoint"
    )


@requires_sandbox_apply
@requires_stata
def test_stata_sandbox_allows_cwd_read(tmp_path: Path):
    """Stata must be able to read a user dataset from cwd.

    Uses a non-degenerate y ~ x relationship (slope 10, intercept 3,
    small noise) so `regress` keeps `_cons` in the model. A perfect
    y=10x causes Stata to drop the intercept as `(omitted)`, which
    makes `e(b)` shape-inconsistent and the runtime library's JSON
    emission produces a payload with `_cons` written as a missing
    value — unrelated to the sandbox behavior being tested.
    """
    csv = tmp_path / "data.csv"
    csv.write_text(
        "x,y\n"
        "1,13\n2,24\n3,32\n4,42\n5,54\n6,63\n"
        "7,72\n8,83\n9,94\n10,102\n11,114\n12,123\n"
    )
    code = '''
import delimited "data.csv", clear
regress y x
nora_result_regress, label("stata-cwd-read")
'''
    r = run_script("Stata", code, tmp_path)
    assert r.ok, f"Stata script failed: error={r.error}\nstdout tail={r.raw_stdout[-500:]}"
    assert r.result_payloads[0]["n"] == 12


# ---------------------------------------------------------------------------
# Estimator-family coverage — non-OLS p-values via Wald z-fallback
# ---------------------------------------------------------------------------

@requires_sandbox_apply
@requires_stata
def test_stata_logit_emits_wald_p_values(tmp_path: Path):
    """Logit doesn't populate ``e(df_r)`` (z-tests, not t-tests).

    The helper used to gate ``p_values`` emission on ``e(df_r)``
    being set, which silently dropped the entire dict for every
    non-OLS estimator. The renderer then dropped the p-value
    column, making logit/probit/Poisson/Cox cards look like the
    estimator doesn't define p-values when the Wald z-test is in
    fact well-defined and is what Stata's own display uses.

    Asserts the helper now falls back to the asymptotic Wald
    z-test (``2 * normal(-|b/se|)``) when df_r is empty, so the
    payload carries one finite p-value per coefficient.
    """
    code = '''
sysuse auto, clear
logit foreign mpg weight
nora_result_regress, label("stata-logit-wald")
'''
    r = run_script("Stata", code, tmp_path)
    assert r.ok, f"Stata logit failed: error={r.error}\nstdout tail={r.raw_stdout[-500:]}"
    payload = r.result_payloads[0]
    pvals = payload.get("p_values")
    assert isinstance(pvals, dict), (
        "non-OLS estimator must still emit p_values via the Wald "
        f"z-fallback; got {type(pvals).__name__}"
    )
    assert set(pvals) == {"mpg", "weight", "_cons"}
    for term, p in pvals.items():
        assert isinstance(p, float) and 0.0 <= p <= 1.0, f"{term}={p!r}"
    # Sanity: logit always populates e(chi2) — keeps the omnibus
    # caption populated alongside the per-coefficient p-values.
    assert "chi_squared" in payload
    assert "chi_squared_p_value" in payload


# ---------------------------------------------------------------------------
# Sandbox — `.nora` carve-out (Nora's session state)
# ---------------------------------------------------------------------------

@requires_sandbox_apply
@requires_stata
def test_stata_sandbox_blocks_read_of_nora_session_state(tmp_path_home: Path):
    """The session cwd is readable so scripts can ``use mydata.dta``,
    but ``<cwd>/.nora`` holds Nora's own session state — chat
    history, results.db, prior run logs. A script-readable .nora
    means a model-authored script can read raw stdout/stderr from
    earlier runs (which carry pre-sanitizer rows from ``list`` /
    ``summarize, detail``) and smuggle excerpts back through any
    sanitizer-allowed channel.

    Test cwd lives under ``~/.nora-sessions/`` (NOT under pytest's
    ``tmp_path``) because the sandbox profile's broader ``/private/
    var/folders`` allow — needed for R/Stata's $TMPDIR scratch —
    would mask the carve-out under a tempfile-style path. In
    production sessions never live under that tree."""
    # Plant a victim file the sandbox should refuse. Path is under
    # cwd/.nora — exactly the carve-out the new rule denies.
    victim = tmp_path_home / ".nora" / "results.db"
    victim.parent.mkdir(parents=True, exist_ok=True)
    victim.write_text("SECRET_STATE_DO_NOT_LEAK\n", encoding="utf-8")
    code = f'''
capture file open probe using "{victim}", read binary
local probe_status "DENIED"
if _rc == 0 {{
    local probe_status "READ_SUCCESS"
    file close probe
}}

sysuse auto, clear
regress price mpg
nora_result_regress, label("nora-state-probe=`probe_status'")
'''
    r = run_script("Stata", code, tmp_path_home)
    assert r.ok, f"Stata script failed: error={r.error}\nstdout tail={r.raw_stdout[-500:]}"
    label = r.result_payloads[0]["label"]
    assert "DENIED" in label, (
        f"Expected sandbox to block read of {victim}; got label={label!r}. "
        f"The .nora deny rule is not effective."
    )
    assert "READ_SUCCESS" not in label, (
        f"Sandbox allowed read of Nora's session state at {victim}; "
        f"this is the SDC bypass the carve-out is meant to close."
    )


@requires_sandbox_apply
@requires_stata
def test_stata_sandbox_allows_run_dir_under_nora(tmp_path_home: Path):
    """Although ``.nora`` is carved out, the current ``run_dir``
    lives at ``.nora/runs/<id>/`` and MUST stay readable + writable
    — that's where the runtime library stages, where result.json is
    written, and where Stata drops its batch .log. A test that runs
    a regression at all already exercises this; this test asserts it
    explicitly so a future profile change can't regress it."""
    code = '''
sysuse auto, clear
regress price mpg
nora_result_regress, label("run-dir-still-works")
'''
    r = run_script("Stata", code, tmp_path_home)
    assert r.ok, f"run_dir broke: error={r.error}\nstdout tail={r.raw_stdout[-500:]}"
    assert r.result_payloads[0]["label"] == "run-dir-still-works"


# ---------------------------------------------------------------------------
# Diagnostics — condition_number on factor-variable models
# ---------------------------------------------------------------------------

@requires_sandbox_apply
@requires_stata
def test_stata_regress_with_factor_var_emits_condition_number(tmp_path: Path):
    """``regress y i.foreign mpg`` populates e(V) with a row/column
    for the structurally omitted base level whose variance is zero.
    Earlier versions ran symeigen on the full e(V), found a zero
    eigenvalue, hit the ``_emin > 0`` guard, and silently dropped
    ``condition_number`` — even though the estimable design has a
    finite condition number.

    The helper now restricts to the estimable submatrix (columns
    with strictly positive diagonal variance) before symeigen, so
    factor-variable regressions surface a finite condition number
    just like all-numeric ones do."""
    code = '''
sysuse auto, clear
regress price i.foreign mpg
nora_result_regress, label("stata-factor-cond")
'''
    r = run_script("Stata", code, tmp_path)
    assert r.ok, f"Stata script failed: error={r.error}\nstdout tail={r.raw_stdout[-500:]}"
    payload = r.result_payloads[0]
    cond = payload.get("condition_number")
    assert isinstance(cond, (int, float)) and cond > 0, (
        f"condition_number must be a finite positive scalar on a "
        f"factor-variable regression; got {cond!r}. Was the helper "
        f"running symeigen across the full e(V) (which includes the "
        f"zero-variance base level) instead of the estimable "
        f"submatrix?"
    )


# ---------------------------------------------------------------------------
# Security invariants — sandbox blocks what matters
# ---------------------------------------------------------------------------

# Existing system files outside the narrowed sandbox allowlist.
_OUTSIDE_ALLOWLIST_PROBES = [
    "/Library/Keychains/System.keychain",
    "/private/var/log/system.log",
]


def _probe_read_do(target: str) -> str:
    r'''Return a Stata snippet that tries to read `target`, captures
    the status into local `probe_status`, and then emits a regression
    payload with `probe_status` in the label so the test can inspect
    what happened. Mirrors the R tryCatch pattern used elsewhere.'''
    return f'''
capture file open probe using "{target}", read binary
local probe_status "DENIED"
if _rc == 0 {{
    local probe_status "READ_SUCCESS"
    file close probe
}}

sysuse auto, clear
regress price mpg
nora_result_regress, label("stata-probe=`probe_status'")
'''


@requires_sandbox_apply
@requires_stata
def test_stata_sandbox_blocks_read_outside_allowlist(tmp_path: Path):
    """Stata probing a file outside the sandbox allowlist must fail.
    Same invariant as the R test; Stata uses `file open` with
    `capture` to detect the denial via `_rc`."""
    target = next(
        (p for p in _OUTSIDE_ALLOWLIST_PROBES if Path(p).exists()), None
    )
    if target is None:
        pytest.skip("no out-of-allowlist probe file present")

    r = run_script("Stata", _probe_read_do(target), tmp_path)
    assert r.ok, f"executor failure: {r.error}"
    label = r.result_payloads[0]["label"]
    assert "DENIED" in label, (
        f"sandbox failed — Stata read out-of-allowlist path: {label!r}"
    )


@requires_sandbox_apply
@requires_stata
def test_stata_sandbox_blocks_home_dotfile_reads(tmp_path: Path):
    """Reads from ~/.zshrc / etc. must fail from Stata as from R."""
    home = Path.home()
    candidates = [home / ".zshrc", home / ".bashrc", home / ".profile"]
    target = next((p for p in candidates if p.exists()), None)
    if target is None:
        pytest.skip("no standard dotfile in HOME to probe")

    r = run_script("Stata", _probe_read_do(str(target)), tmp_path)
    assert r.ok
    label = r.result_payloads[0]["label"]
    assert "DENIED" in label, (
        f"home dotfile read was NOT denied in Stata: {label!r}"
    )


@requires_sandbox_apply
@requires_stata
def test_stata_sandbox_blocks_write_outside_run_dir(tmp_path: Path):
    """Writes from Stata to a path outside the allowed write trees
    must fail. Targets /Library/Caches — present on every Mac,
    user-writable outside the sandbox, but NOT in the write
    allowlist."""
    import uuid

    caches = Path("/Library/Caches")
    if not caches.is_dir():
        pytest.skip("/Library/Caches not present")
    probe = caches / f".nora_test_permcheck_{uuid.uuid4().hex[:8]}"
    try:
        probe.write_text("x")
        probe.unlink()
    except OSError:
        pytest.skip("/Library/Caches not user-writable here")

    victim = caches / f".nora_test_stata_victim_{uuid.uuid4().hex[:8]}.txt"
    if victim.exists():
        victim.unlink()
    try:
        code = f'''
capture file open victim using "{victim}", write replace
if _rc == 0 {{
    file write victim "pwned" _n
    file close victim
}}

sysuse auto, clear
regress price mpg
nora_result_regress, label("stata-write-probe")
'''
        r = run_script("Stata", code, tmp_path)
        assert not victim.exists(), (
            f"sandbox failed — Stata wrote {victim} outside its "
            f"scratch dir (r.error={r.error})"
        )
    finally:
        try:
            victim.unlink()
        except FileNotFoundError:
            pass
