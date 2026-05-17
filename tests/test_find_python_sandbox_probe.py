"""Tests for the sandbox-health probe gating ``find_python``.

The probe exists to catch interpreters that respond to ``-c`` outside
the sandbox but die under it — canonically Apple's ``/usr/bin/python3``
xcselect stub, which dlopens libxcrun from a path Nora's sandbox profile
doesn't allow. Without the probe, every script silently fails at
interpreter startup and the diagnostic loop sends the model chasing
phantom missing-package issues.

The invariants here:

  * The probe uses the SAME profile builder as the executor
    (``executor._sandbox_profile_string``) so probe success implies
    the executor would also succeed at interpreter startup.
  * Results are cached in-memory per binary path so callers don't
    pay the ~200ms probe cost multiple times within a Nora session.
  * ``find_python`` skips candidates whose probe failed, and the
    cache exposes the failure detail via
    ``python_sandbox_probe_results`` so downstream surfaces (the
    executor's "no python found" error, the doctor command) can
    explain *why* the candidate was rejected.
  * On non-macOS (no ``sandbox-exec``) the probe is a no-op — the
    executor doesn't sandbox there anyway, so probing for sandbox
    compatibility is moot.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from nora import env_detect
from nora import env_detect as _env_detect_mod
from nora.env_detect import (
    _SANDBOX_PROBE_CACHE,
    _probe_sandbox_health,
    find_python,
    find_sandbox_exec,
    python_sandbox_probe_results,
    sandbox_baseline_result,
)


def _sandbox_apply_works() -> bool:
    """Match the gating helper in ``test_executor_sandbox.py`` — nested
    sandbox harnesses (some CI / dev containers) block ``sandbox-exec``
    and would make probe tests permanently fail."""
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


requires_sandbox_apply = pytest.mark.skipif(
    sys.platform != "darwin" or not _sandbox_apply_works(),
    reason=(
        "sandbox-exec cannot apply a profile here "
        "(non-macOS or nested-sandbox harness)."
    ),
)


@pytest.fixture(autouse=True)
def _clear_probe_caches():
    """Reset both caches before each test. The baseline cache is a
    module-level optional that survives across tests by default; we
    null it explicitly so ordering doesn't masquerade as a cached
    result."""
    _SANDBOX_PROBE_CACHE.clear()
    _env_detect_mod._SANDBOX_BASELINE_CACHE = None
    yield
    _SANDBOX_PROBE_CACHE.clear()
    _env_detect_mod._SANDBOX_BASELINE_CACHE = None


# ---------------------------------------------------------------------------
# Probe shape — degenerate inputs return a clean failure, not a crash
# ---------------------------------------------------------------------------

def test_probe_on_nonexistent_binary_returns_failure(tmp_path: Path):
    """A binary path that doesn't exist must come back as
    ``(False, "...")`` so ``find_python`` can skip it cleanly. No
    exceptions reach the caller; the failing-launch path matters for
    misconfigured shims (broken pyenv version, deleted homebrew
    bottle, etc.)."""
    if find_sandbox_exec() is None:
        pytest.skip("non-macOS: probe is a no-op, nothing to test here")
    fake = str(tmp_path / "definitely-not-a-python")
    ok, stderr = _probe_sandbox_health(fake)
    assert ok is False
    assert stderr  # something was captured


def test_probe_on_non_macos_is_noop(monkeypatch):
    """On non-macOS hosts there is no ``sandbox-exec``; the executor
    doesn't sandbox there either. The probe MUST short-circuit to
    ``(True, "")`` so ``find_python`` doesn't reject every candidate
    on Linux / Windows."""
    monkeypatch.setattr(env_detect, "find_sandbox_exec", lambda: None)
    ok, stderr = _probe_sandbox_health("/anything/here")
    assert ok is True
    assert stderr == ""


# ---------------------------------------------------------------------------
# Sandbox baseline — distinguishes broken sandbox from broken interpreter
# ---------------------------------------------------------------------------

def test_baseline_failure_attributed_to_sandbox_not_interpreter(monkeypatch):
    """Phase A check: if ``sandbox-exec`` can't apply a minimal
    profile, every per-candidate probe would otherwise fail identically
    and the doctor would mis-attribute the failure to the interpreter
    ("install Homebrew Python" advice that won't help). The probe
    must short-circuit with a message that names the sandbox layer
    as the broken component."""
    # Force the baseline cache into a failed state to simulate
    # ``sandbox-exec rejected a minimal profile``.
    _env_detect_mod._SANDBOX_BASELINE_CACHE = (
        False,
        "sandbox-exec rejected a minimal allow-default profile (exit 1).",
    )
    ok, stderr = _probe_sandbox_health("/opt/homebrew/bin/python3")
    assert ok is False
    # The rejection message must name the SANDBOX layer, not the
    # interpreter. A doctor that read this would point the
    # researcher at the right root cause.
    assert "sandbox-exec itself is unusable" in stderr
    assert "not specific to the interpreter" in stderr


def test_baseline_cached_across_calls():
    """The baseline result is process-lifetime cached. Once it's
    computed (or pre-seeded by a test), subsequent calls return the
    same tuple without re-running ``sandbox-exec``."""
    _env_detect_mod._SANDBOX_BASELINE_CACHE = (True, "")
    first = sandbox_baseline_result()
    second = sandbox_baseline_result()
    assert first == second == (True, "")


@requires_sandbox_apply
def test_baseline_passes_on_healthy_macos_host():
    """On a host where ``sandbox-exec`` actually works, the baseline
    check returns ok with no error. The fixture
    ``requires_sandbox_apply`` already verifies the trivial profile
    apply works; this confirms the cached accessor agrees."""
    _env_detect_mod._SANDBOX_BASELINE_CACHE = None  # force fresh probe
    ok, stderr = sandbox_baseline_result()
    assert ok is True
    assert stderr == ""


# ---------------------------------------------------------------------------
# Probe success — sharing the executor's profile builder
# ---------------------------------------------------------------------------

@requires_sandbox_apply
def test_probe_passes_on_a_working_interpreter():
    """If ``find_python`` returns a Tool, the cached probe entry for
    that binary must be ``ok=True`` — otherwise the gating logic
    let a rejected interpreter through, which is the bug this whole
    feature exists to prevent."""
    tool = find_python()
    if tool is None:
        pytest.skip("no usable python3 on PATH for this test env")
    assert _SANDBOX_PROBE_CACHE.get(tool.binary, (False, ""))[0] is True


@requires_sandbox_apply
def test_probe_uses_shared_profile_builder(monkeypatch):
    """The probe calls ``executor._sandbox_profile_string`` — the same
    function the executor uses for real script runs. Drift between the
    two would defeat the probe's purpose (probe passes, real run
    fails, or vice versa). Verify by spying on the import."""
    from nora import executor as _executor
    calls: list[tuple] = []
    real = _executor._sandbox_profile_string

    def _spy(*args, **kwargs):
        calls.append((args, kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(_executor, "_sandbox_profile_string", _spy)
    # Pick any candidate to drive the probe — using shutil.which mirrors
    # find_python's lookup so we exercise the same path.
    binary = shutil.which("python3") or shutil.which("python")
    if binary is None:
        pytest.skip("no python on PATH")
    _probe_sandbox_health(binary)
    assert calls, (
        "probe did not call executor._sandbox_profile_string — drift "
        "between probe profile and executor profile would mask the "
        "exact bug this probe exists to catch"
    )


# ---------------------------------------------------------------------------
# Caching — lazy, in-memory, by binary path
# ---------------------------------------------------------------------------

def test_find_python_caches_probe_result_per_path(monkeypatch):
    """First ``find_python`` call probes; subsequent calls hit the
    cache and do NOT re-probe. The whole point of the cache is that
    callers (system_prompt, ui banner, executor preamble lookup) hit
    one path within a session without paying ~200ms × N."""
    probe_calls: list[str] = []

    def _fake_probe(binary: str):
        probe_calls.append(binary)
        return True, ""

    monkeypatch.setattr(env_detect, "_probe_sandbox_health", _fake_probe)

    # Force a real path so find_python proceeds past shutil.which.
    binary = shutil.which("python3") or shutil.which("python")
    if binary is None:
        pytest.skip("no python on PATH")

    t1 = find_python()
    t2 = find_python()
    assert t1 is not None and t2 is not None
    assert t1.binary == t2.binary
    # Probe fires once per binary across both calls.
    assert probe_calls.count(t1.binary) == 1


def test_find_python_skips_probe_rejected_candidate(monkeypatch):
    """A candidate whose probe returned ok=False MUST NOT be accepted.
    Models the Apple ``/usr/bin/python3`` case: returning the
    rejected interpreter would let the executor proceed and every
    subsequent script would silently die at interpreter startup —
    exactly the failure mode the probe exists to prevent."""

    real_which = shutil.which

    def _fake_which(name):
        # Pretend ``python3`` lives at a stable fake path. The probe
        # is then stubbed to reject it.
        if name == "python3":
            return "/opt/fake/python3"
        if name == "python":
            return None
        return real_which(name)

    monkeypatch.setattr(env_detect.shutil, "which", _fake_which)
    monkeypatch.setattr(
        env_detect, "_python_version", lambda p: "Python 3.11.0",
    )
    monkeypatch.setattr(
        env_detect, "_probe_sandbox_health",
        lambda p: (False, "xcrun: unable to load libxcrun (...)"),
    )

    tool = find_python()
    assert tool is None
    # The rejection detail is queryable via the public accessor.
    results = python_sandbox_probe_results()
    assert "/opt/fake/python3" in results
    ok, stderr = results["/opt/fake/python3"]
    assert ok is False
    assert "libxcrun" in stderr


def test_probe_results_accessor_returns_snapshot():
    """``python_sandbox_probe_results`` must return a *copy* of the
    cache so callers can't mutate the canonical state. Without this,
    a doctor command that filtered its returned dict could
    accidentally wipe the cache for the rest of the process."""
    _SANDBOX_PROBE_CACHE["/some/path"] = (True, "")
    snapshot = python_sandbox_probe_results()
    snapshot.clear()
    assert "/some/path" in _SANDBOX_PROBE_CACHE


# ---------------------------------------------------------------------------
# Executor error message — probe-rejected candidates surface to the user
# ---------------------------------------------------------------------------

def test_executor_error_lists_probe_rejected_candidates(monkeypatch, tmp_path):
    """When every python3 candidate fails the probe, the executor must
    NOT say "python3 not found on PATH" — that's misleading when the
    binary is literally there. Instead it must list the rejected
    binaries and a snippet of why, so the researcher has a
    concrete fix path."""
    from nora.env_detect import Environment
    from nora.executor import run_script, clear_environment_cache

    # Prime the probe cache with a rejected candidate (the Apple-stub
    # failure shape). The executor's "no python" branch reads the
    # cache directly via ``python_sandbox_probe_results``.
    _SANDBOX_PROBE_CACHE["/usr/bin/python3"] = (
        False,
        "xcrun: error: unable to load libxcrun (dlopen(...): "
        "file system sandbox blocked open())",
    )

    # Bypass cached env probe and inject one with python=None so the
    # executor takes the "no python" branch deterministically.
    clear_environment_cache()
    fake_env = Environment(
        r=None, stata=None, python=None,
        sandbox_exec=find_sandbox_exec(),
    )

    result = run_script("Python", "print('hi')", tmp_path, env=fake_env)
    assert result.ok is False
    assert result.error is not None
    assert "/usr/bin/python3" in result.error
    assert "libxcrun" in result.error
    # The misleading legacy message must NOT appear when we have
    # probe failures to report.
    assert "python3 not found on PATH" not in result.error
