"""Integration tests for the executor's sandbox profile.

These exercise the real subprocess path — sandbox-exec spawning Rscript
— to confirm the ``(deny default)`` profile enforces what the design
requires: scripts can read the researcher's cwd and their R package
library, and they cannot read the rest of the home directory or write
outside the run scratch dir.

Portability notes (learned from the 2026-04-20 review):

- Some environments run pytest inside a harness that blocks nested
  ``sandbox-exec`` with ``sandbox_apply: Operation not permitted``.
  ``requires_sandbox_apply`` preflights this by trying a trivial
  ``sandbox-exec`` invocation before each test and skipping if the
  outer sandbox prevents it.

- Some environments mount HOME read-only, so we cannot ``write_text``
  into ``~/`` during test setup. The probes below therefore target
  files that already exist on every Mac and live OUTSIDE the narrowed
  sandbox allowlist (e.g. ``/Library/Keychains/System.keychain``), and
  write-block tests target paths the test process does not need to
  create first.

The pure-SBPL invariants (profile shape, no /private broadly, HOME
narrowly, etc.) live in ``test_executor_profile.py`` and run
unconditionally. This file is the belt-and-suspenders run-it-for-real
layer.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from builder.env_detect import find_sandbox_exec
from builder.executor import run_script


_RSCRIPT = shutil.which("Rscript")


def _sandbox_apply_works() -> bool:
    """Return True iff ``sandbox-exec`` can actually apply a profile in
    the current environment. Nested sandbox harnesses return False.
    """
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


requires_rscript = pytest.mark.skipif(
    _RSCRIPT is None,
    reason="Rscript not on PATH; sandbox integration tests need R.",
)

requires_sandbox_apply = pytest.mark.skipif(
    sys.platform != "darwin" or not _sandbox_apply_works(),
    reason=(
        "sandbox-exec cannot apply a profile in this environment "
        "(non-macOS or nested-sandbox harness)."
    ),
)


@pytest.fixture
def tiny_csv(tmp_path: Path) -> Path:
    """A minimal dataset so scripts have something to load from cwd."""
    path = tmp_path / "data.csv"
    path.write_text(
        "x,y\n1,10\n2,20\n3,30\n4,40\n5,50\n6,60\n"
        "7,70\n8,80\n9,90\n10,100\n11,110\n12,120\n"
    )
    return path


# ---------------------------------------------------------------------------
# Happy path — sandbox lets R do its job
# ---------------------------------------------------------------------------

@requires_sandbox_apply
@requires_rscript
def test_sandbox_allows_cwd_read_and_runtime_write(tmp_path: Path, tiny_csv: Path):
    """The profile must let R read data from cwd and emit a result payload."""
    code = r'''
df <- read.csv("data.csv")
builder$from_lm(lm(y ~ x, data = df), label = "ok")
'''
    r = run_script("R", code, tmp_path)
    assert r.ok, f"script failed: error={r.error}\nstderr={r.raw_stderr}"
    assert r.result_payload is not None
    assert r.result_payload["type"] == "linear_regression"
    assert r.result_payload["n"] == 12


# ---------------------------------------------------------------------------
# Security invariants — sandbox blocks what matters
# ---------------------------------------------------------------------------

# System files guaranteed to exist on macOS and OUTSIDE the narrowed
# sandbox allowlist (keychains were part of a broad ``/Library``, now
# excluded by the narrowed profile; system.log lives under
# ``/private/var/log`` which is no longer allowed).
_OUTSIDE_ALLOWLIST_PROBES = [
    "/Library/Keychains/System.keychain",
    "/private/var/log/system.log",
]


@requires_sandbox_apply
@requires_rscript
def test_sandbox_blocks_read_outside_allowlist(tmp_path: Path, tiny_csv: Path):
    """A read to a system path OUTSIDE the narrowed allowlist must fail.

    We probe a pre-existing file (no test-setup writes needed — the
    earlier HOME-setup version broke in environments where HOME is
    mounted read-only) and verify the script can't recover its
    contents.
    """
    target = next(
        (p for p in _OUTSIDE_ALLOWLIST_PROBES if Path(p).exists()),
        None,
    )
    if target is None:
        pytest.skip("no out-of-allowlist probe file present on this machine")

    code = (
        f'probe <- tryCatch(readLines("{target}", n = 1, warn = FALSE),\n'
        '  error = function(e) paste("DENIED:", conditionMessage(e)),\n'
        '  warning = function(w) paste("DENIED:", conditionMessage(w)))\n'
        'df <- data.frame(x = 1:12, y = (1:12) * 2)\n'
        'builder$from_lm(lm(y ~ x, data = df), '
        'label = paste0("probe=", substr(paste(probe, collapse="|"), 1, 80)))\n'
    )
    r = run_script("R", code, tmp_path)
    assert r.ok, f"executor failure: {r.error}"
    label = r.result_payload["label"]
    assert "DENIED" in label, (
        f"sandbox failed — out-of-allowlist read was permitted: {label!r}"
    )


@requires_sandbox_apply
@requires_rscript
def test_sandbox_blocks_home_dotfile_reads(tmp_path: Path, tiny_csv: Path):
    """Reads from ``~/.zshrc`` etc must fail — the canary for 'malicious
    script exfils user config files via HOME'.

    No setup: we probe existing dotfiles, no test writes to HOME.
    """
    home = Path.home()
    candidates = [home / ".zshrc", home / ".bashrc", home / ".profile"]
    target = next((p for p in candidates if p.exists()), None)
    if target is None:
        pytest.skip("no standard dotfile in HOME to probe")

    code = (
        f'probe <- tryCatch(readLines("{target}", n = 1, warn = FALSE),\n'
        '  error = function(e) paste("DENIED:", conditionMessage(e)),\n'
        '  warning = function(w) paste("DENIED:", conditionMessage(w)))\n'
        'df <- data.frame(x = 1:12, y = (1:12) * 3)\n'
        'builder$from_lm(lm(y ~ x, data = df), '
        'label = paste0("home-probe=", substr(paste(probe, collapse="|"), 1, 80)))\n'
    )
    r = run_script("R", code, tmp_path)
    assert r.ok
    label = r.result_payload["label"]
    assert "DENIED" in label, f"home dotfile read was NOT denied: label={label!r}"


@requires_sandbox_apply
@requires_rscript
def test_sandbox_blocks_write_outside_run_dir(tmp_path: Path, tiny_csv: Path):
    """Writes to a path outside the run scratch dir and allowed temp
    trees must fail.

    We target ``/Library/Caches``: present on every Mac, world-writable
    by the user OUTSIDE the sandbox (so a non-sandbox baseline would
    succeed), but NOT in the sandbox write allowlist. If the file
    appears, the sandbox didn't enforce the write boundary.

    If ``/Library/Caches`` isn't user-writable in this environment
    (unlikely but possible), we skip — there's no portable victim
    that's both out-of-allowlist and guaranteed-writable-outside-
    sandbox across every macOS configuration.
    """
    import uuid

    caches = Path("/Library/Caches")
    if not caches.is_dir():
        pytest.skip("/Library/Caches not present")
    # Probe whether the test process itself can write there; if not,
    # the test can't distinguish sandbox-denied from permission-denied.
    probe = caches / f".builder_test_permcheck_{uuid.uuid4().hex[:8]}"
    try:
        probe.write_text("x")
        probe.unlink()
    except OSError:
        pytest.skip("/Library/Caches not user-writable here")

    victim = caches / f".builder_test_victim_{uuid.uuid4().hex[:8]}.txt"
    if victim.exists():
        victim.unlink()
    try:
        code = (
            f'tryCatch(writeLines("pwned", "{victim}"), '
            'error = function(e) e, warning = function(w) w)\n'
            'df <- data.frame(x = 1:12, y = (1:12) * 4)\n'
            'builder$from_lm(lm(y ~ x, data = df), label = "write-probe")\n'
        )
        r = run_script("R", code, tmp_path)
        assert not victim.exists(), (
            f"sandbox failed — script wrote {victim} outside its "
            f"scratch dir (r.error={r.error})"
        )
    finally:
        try:
            victim.unlink()
        except FileNotFoundError:
            pass


# ---------------------------------------------------------------------------
# Missing-sandbox preflight — pure Python, no sandbox-exec needed
# ---------------------------------------------------------------------------

def test_run_script_refuses_without_sandbox(tmp_path: Path):
    """If sandbox-exec is unavailable (e.g. Linux/Windows), run_script
    must refuse rather than fall through to an unsandboxed subprocess.
    """
    from builder import env_detect, executor

    fake_env = env_detect.Environment(
        r=env_detect.Tool(name="R", binary="/bin/true"),
        stata=None,
        sandbox_exec=None,
    )
    r = executor.run_script("R", "cat('hi')", tmp_path, env=fake_env)
    assert not r.ok
    assert "sandbox" in (r.error or "").lower()
