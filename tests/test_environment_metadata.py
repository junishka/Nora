"""Tests for the structured environment metadata Nora surfaces back to
the model on every ``submit_script`` response.

Why this exists: the diagnostic spiral on the Apple xcrun-stub bug
happened because the model couldn't see which interpreter Nora picked
or what it provided. Even with the sandbox probe rejecting the stub,
without structured env context the model is reasoning from absence —
it knows the script failed but not where the environment landed.

The invariants here:

  * ``ExecutionResult.environment`` is populated on every response,
    success and failure, including the early-return preflight
    branches (sandbox missing, interpreter missing, hard packages
    missing). The shape is consistent across all of them.
  * The snapshot is phase-safe: every field is an interpreter path,
    a version string, a package name, or sandbox-probe launcher
    stderr. None of them touch researcher data — so they can be
    forwarded unredacted regardless of where in script execution a
    failure landed.
  * The language under test is at the top level (``interpreter``,
    ``language``); the other two runtimes are summarised under
    ``other_runtimes`` for cross-language context without bloat.
  * Sandbox-probe rejections (the Apple-stub failure shape) appear
    under ``python_sandbox_probe_failures`` whenever the probe cache
    has rejected candidates — across all run languages, not just
    Python, so a researcher who switched to R after a Python failure
    can still see why Python didn't work.
  * The model-facing response envelope carries the same snapshot
    under ``_environment``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from nora.env_detect import (
    Environment,
    Tool,
    _SANDBOX_PROBE_CACHE,
)
from nora.executor import (
    ExecutionResult,
    _build_environment_metadata,
    run_script,
)


@pytest.fixture(autouse=True)
def _clear_probe_cache():
    """Probe cache leaks across tests would make the
    ``python_sandbox_probe_failures`` assertion flaky depending on
    which test order pytest picks. Reset between each test."""
    _SANDBOX_PROBE_CACHE.clear()
    yield
    _SANDBOX_PROBE_CACHE.clear()


def _env_with(python: Tool | None = None,
              r: Tool | None = None,
              stata: Tool | None = None,
              sandbox: str | None = "/usr/bin/sandbox-exec") -> Environment:
    return Environment(r=r, stata=stata, python=python, sandbox_exec=sandbox)


# ---------------------------------------------------------------------------
# Shape — primary interpreter for the run, others summarised
# ---------------------------------------------------------------------------

def test_metadata_primary_is_run_language_python():
    py = Tool(
        name="Python", binary="/opt/homebrew/bin/python3",
        version="Python 3.12.0",
        missing_packages=(),
        optional_missing_packages=("matplotlib",),
        extra_read_paths=("/opt/homebrew/opt/python@3.12",),
    )
    env = _env_with(python=py)
    meta = _build_environment_metadata(env, "Python")
    assert meta["language"] == "Python"
    assert meta["interpreter"]["present"] is True
    assert meta["interpreter"]["binary"] == "/opt/homebrew/bin/python3"
    assert meta["interpreter"]["version"] == "Python 3.12.0"
    # Python-only field present on the primary.
    assert meta["interpreter"]["sys_prefix"] == "/opt/homebrew/opt/python@3.12"
    # Missing/installed/optional sets all show up.
    assert "missing_required" in meta["interpreter"]
    assert "missing_optional" in meta["interpreter"]
    assert "matplotlib" in meta["interpreter"]["missing_optional"]
    # Other runtimes summarised under their own block.
    assert "R" in meta["other_runtimes"]
    assert "Stata" in meta["other_runtimes"]


def test_metadata_primary_is_run_language_r():
    r = Tool(name="R", binary="/opt/homebrew/bin/Rscript", version="R 4.3.0")
    py = Tool(name="Python", binary="/usr/bin/python3", version="Python 3.9.0")
    env = _env_with(r=r, python=py)
    meta = _build_environment_metadata(env, "R")
    assert meta["language"] == "R"
    assert meta["interpreter"]["binary"] == "/opt/homebrew/bin/Rscript"
    # Python still surfaced under others — useful when the model
    # wants to suggest a language switch.
    assert meta["other_runtimes"]["Python"]["binary"] == "/usr/bin/python3"


# ---------------------------------------------------------------------------
# Absent interpreter — ``present: False`` shape stays consistent
# ---------------------------------------------------------------------------

def test_metadata_missing_interpreter_marks_present_false():
    env = _env_with(python=None)
    meta = _build_environment_metadata(env, "Python")
    assert meta["interpreter"] == {"present": False}
    # No language-specific keys when the interpreter is absent.
    assert "binary" not in meta["interpreter"]
    assert "sys_prefix" not in meta["interpreter"]


# ---------------------------------------------------------------------------
# Sandbox probe rejections — surfaced for ALL run languages
# ---------------------------------------------------------------------------

def test_metadata_surfaces_python_sandbox_probe_failures_for_python_run():
    _SANDBOX_PROBE_CACHE["/usr/bin/python3"] = (
        False,
        "xcrun: error: unable to load libxcrun ... blocked open()",
    )
    py = Tool(name="Python", binary="/opt/homebrew/bin/python3",
              version="Python 3.12.0")
    env = _env_with(python=py)
    meta = _build_environment_metadata(env, "Python")
    failures = meta.get("python_sandbox_probe_failures")
    assert failures is not None
    assert any(f["binary"] == "/usr/bin/python3" for f in failures)
    # The proximate-cause tail of stderr makes it through.
    tail = next(f["stderr_excerpt"] for f in failures
                if f["binary"] == "/usr/bin/python3")
    assert "libxcrun" in tail


def test_metadata_surfaces_python_probe_failures_even_for_r_runs():
    """A researcher who switched to R after a Python failure should
    still see the Python rejection — otherwise they lose context for
    why Python is unavailable. The probe-failure block is global to
    the response, not gated on language."""
    _SANDBOX_PROBE_CACHE["/usr/bin/python3"] = (False, "boom")
    env = _env_with(r=Tool(name="R", binary="/opt/homebrew/bin/Rscript"))
    meta = _build_environment_metadata(env, "R")
    failures = meta.get("python_sandbox_probe_failures")
    assert failures is not None
    assert failures[0]["binary"] == "/usr/bin/python3"


def test_metadata_omits_probe_failures_block_when_cache_is_clean():
    """No rejected candidates → no clutter. The optional key keeps
    the response minimal in the common case."""
    env = _env_with(python=Tool(name="Python", binary="/p", version="Python 3"))
    meta = _build_environment_metadata(env, "Python")
    assert "python_sandbox_probe_failures" not in meta


# ---------------------------------------------------------------------------
# End-to-end: metadata reaches ExecutionResult on every preflight branch
# ---------------------------------------------------------------------------

def test_executionresult_carries_env_on_sandbox_missing(tmp_path: Path):
    env = _env_with(sandbox=None)
    res = run_script("Python", "x=1", tmp_path, env=env)
    assert res.environment is not None
    assert res.environment["sandbox_exec_present"] is False


def test_executionresult_carries_env_on_python_missing(tmp_path: Path):
    env = _env_with(python=None)
    res = run_script("Python", "x=1", tmp_path, env=env)
    assert res.environment is not None
    assert res.environment["interpreter"]["present"] is False


def test_executionresult_carries_env_on_python_hard_packages_missing(
    tmp_path: Path,
):
    py = Tool(
        name="Python", binary="/p", version="Python 3.12",
        missing_packages=("pandas", "numpy"),
    )
    env = _env_with(python=py)
    res = run_script("Python", "x=1", tmp_path, env=env)
    assert res.environment is not None
    primary = res.environment["interpreter"]
    assert "pandas" in primary["missing_required"]
    assert "numpy" in primary["missing_required"]


# ---------------------------------------------------------------------------
# Response envelope wiring — model-facing surface
# ---------------------------------------------------------------------------

def test_response_envelope_includes_environment(monkeypatch, tmp_path: Path):
    """``_build_response_envelope`` must copy the executor's env
    snapshot to ``_environment``. Without this the metadata is
    collected but never reaches the model — the whole point is the
    model-visible surface."""
    from nora.tools import _build_response_envelope

    fake_exec = ExecutionResult(
        ok=True, language="Python", raw_stdout="", raw_stderr="",
        exit_code=0, result_payloads=[{"type": "x"}],
        error=None, run_dir=tmp_path, script_path=None,
        duration_seconds=0.1, warnings=[],
        environment={
            "language": "Python",
            "interpreter": {"present": True, "binary": "/p", "version": "Python 3.12"},
            "sandbox_exec_present": True,
            "other_runtimes": {},
        },
    )
    envelope = _build_response_envelope(
        overall_status="ok",
        script_run_id="run-test",
        results=[],
        exec_result=fake_exec,
        language="Python",
        sanitize_seconds=0.0,
        store_seconds=0.0,
        row_count_audit_seconds=0.0,
    )
    assert "_environment" in envelope
    assert envelope["_environment"]["interpreter"]["binary"] == "/p"
