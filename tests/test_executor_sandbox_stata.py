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
