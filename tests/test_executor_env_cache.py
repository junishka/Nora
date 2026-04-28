"""Regression tests for executor environment caching.

``detect_environment()`` probes multiple runtimes via subprocesses. If
``run_script()`` re-runs that probe on every call, quick successive
Stata submissions pick up a large fixed tax before the regression even
starts.
"""

from __future__ import annotations

from pathlib import Path

from nora import env_detect, executor


def _fake_env() -> env_detect.Environment:
    return env_detect.Environment(
        r=None,
        stata=None,
        python=None,
        sandbox_exec=None,
    )


def test_run_script_reuses_cached_environment(
    tmp_path: Path, monkeypatch,
) -> None:
    """Two default-environment runs should probe the machine once.

    We force the preflight to stop at ``sandbox_exec is None`` so the
    test never launches a real interpreter; the point is only that the
    environment probe is memoized across calls.
    """
    calls = {"n": 0}

    def _detect() -> env_detect.Environment:
        calls["n"] += 1
        return _fake_env()

    monkeypatch.setattr(executor, "detect_environment", _detect)
    executor.clear_environment_cache()
    try:
        r1 = executor.run_script("Stata", "display 1", tmp_path)
        r2 = executor.run_script("Stata", "display 2", tmp_path)
    finally:
        executor.clear_environment_cache()

    assert calls["n"] == 1
    assert r1.ok is False
    assert r2.ok is False
    assert "sandbox-exec not available" in (r1.error or "")
    assert "sandbox-exec not available" in (r2.error or "")


def test_run_script_explicit_env_bypasses_cached_probe(
    tmp_path: Path, monkeypatch,
) -> None:
    """Passing ``env=`` should skip the cached/default probe entirely."""
    calls = {"n": 0}

    def _detect() -> env_detect.Environment:
        calls["n"] += 1
        return _fake_env()

    monkeypatch.setattr(executor, "detect_environment", _detect)
    executor.clear_environment_cache()
    try:
        res = executor.run_script(
            "Stata", "display 1", tmp_path, env=_fake_env(),
        )
    finally:
        executor.clear_environment_cache()

    assert calls["n"] == 0
    assert res.ok is False
    assert "sandbox-exec not available" in (res.error or "")
