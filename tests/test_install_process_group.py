"""Tests that ``install_packages`` kills the whole subprocess tree
on timeout and cancellation, not just the direct child.

``subprocess.run(..., timeout=...)`` only kills the immediate child
on timeout. Package installers (pip especially) fork build / compile
grandchildren that survive that signal and keep mutating the
researcher's environment after Stop. The fix is ``Popen`` with
``start_new_session=True`` plus ``os.killpg`` on timeout / cancel.

These tests substitute a fake installer (``bash -c '...'``) that
spawns a grandchild writing its PID to a temp file, so the test can
verify post-kill liveness with ``os.kill(pid, 0)``.
"""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import time
from pathlib import Path

import pytest

from nora.env_detect import Environment, Tool


def _stub_environment(python_binary: str) -> Environment:
    return Environment(
        r=None,
        stata=None,
        python=Tool(
            name="Python", binary=python_binary, version="Python 3.11",
            missing_packages=(), optional_missing_packages=(),
            extra_read_paths=(),
        ),
        sandbox_exec=None,
    )


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def test_install_kills_grandchild_on_timeout(
    tmp_path: Path, monkeypatch
) -> None:
    """When the installer subprocess times out, its grandchildren
    (pip's compile workers, R's configure, etc.) must die too.

    The fake installer forks a grandchild that sleeps for 60s and
    records its PID. We set a 1-second install timeout, then verify
    the grandchild is dead within a short window after the timeout
    fires.
    """
    pid_file = tmp_path / "grandchild.pid"
    # The script: spawn a long-sleeping grandchild detached from the
    # direct child (via ``&``), record its PID, then sleep so the
    # parent times out. ``setsid``-style detach isn't needed —
    # killpg(getpgid(parent)) reaches every descendant in the same
    # session because ``start_new_session=True`` put them all in one.
    fake_installer = tmp_path / "fake_installer.sh"
    fake_installer.write_text(
        f"#!/bin/bash\n"
        f"# Spawn a grandchild that outlives a plain kill on the parent\n"
        f"(sleep 60) &\n"
        f"echo $! > {pid_file}\n"
        f"sleep 60\n",
        encoding="utf-8",
    )
    fake_installer.chmod(0o755)

    import nora.package_installer as pi
    import nora.env_detect as env_detect

    # Replace the cmd builders so install_packages invokes the fake.
    monkeypatch.setattr(
        pi, "_python_command",
        lambda binary, packages, action: [str(fake_installer)],
    )
    monkeypatch.setattr(env_detect, "detect_environment", lambda: _stub_environment("/usr/bin/python3"))
    # Speed up the test by tightening the install timeout.
    monkeypatch.setattr(pi, "_INSTALL_TIMEOUT_SECONDS", 1)

    started = time.monotonic()
    result = asyncio.run(pi.install_packages("Python", ["pandas"], "install"))
    duration = time.monotonic() - started

    # Should time out, not run to 60s.
    assert duration < 10, f"install took {duration}s — timeout didn't fire"
    assert "timed out" in (result.error or "").lower(), result.error

    # The grandchild PID file must exist; the grandchild itself
    # must be dead within a short grace window after killpg.
    assert pid_file.is_file(), "fake installer never wrote its grandchild PID"
    pid = int(pid_file.read_text(encoding="utf-8").strip())
    # Give the OS a moment to deliver SIGKILL.
    deadline = time.monotonic() + 3
    while _pid_alive(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not _pid_alive(pid), (
        f"grandchild PID {pid} survived the install timeout — "
        f"killpg did not reach the whole process group"
    )


def test_install_kills_grandchild_on_cancel(
    tmp_path: Path, monkeypatch
) -> None:
    """When the asyncio task running install_packages is cancelled
    (researcher hit Stop), the subprocess tree must die. The plain
    ``asyncio.to_thread(subprocess.run, ...)`` pattern did not
    propagate cancellation — the subprocess kept running until its
    own timeout. With Popen + killpg, cancel kills the tree.
    """
    pid_file = tmp_path / "grandchild.pid"
    fake_installer = tmp_path / "fake_installer.sh"
    fake_installer.write_text(
        f"#!/bin/bash\n"
        f"(sleep 60) &\n"
        f"echo $! > {pid_file}\n"
        f"sleep 60\n",
        encoding="utf-8",
    )
    fake_installer.chmod(0o755)

    import nora.package_installer as pi
    import nora.env_detect as env_detect

    monkeypatch.setattr(
        pi, "_python_command",
        lambda binary, packages, action: [str(fake_installer)],
    )
    monkeypatch.setattr(env_detect, "detect_environment", lambda: _stub_environment("/usr/bin/python3"))
    # Long timeout — we're testing cancel, not timeout.
    monkeypatch.setattr(pi, "_INSTALL_TIMEOUT_SECONDS", 60)

    async def _run_then_cancel() -> None:
        task = asyncio.create_task(
            pi.install_packages("Python", ["pandas"], "install")
        )
        # Wait until the grandchild PID has been recorded so we
        # know the fake installer has actually launched.
        deadline = time.monotonic() + 5
        while not pid_file.is_file() and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        assert pid_file.is_file(), "fake installer never started"
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(_run_then_cancel())

    pid = int(pid_file.read_text(encoding="utf-8").strip())
    deadline = time.monotonic() + 3
    while _pid_alive(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not _pid_alive(pid), (
        f"grandchild PID {pid} survived asyncio cancel"
    )


def test_install_passes_proc_register_callback(
    tmp_path: Path, monkeypatch
) -> None:
    """The ``proc_register`` callback fires with the spawned
    Popen so the runner's per-turn registry can target it on Stop.
    Without this hook, the tool's asyncio task is cancellable but
    the subprocess is unreachable from the runner.
    """
    fake_installer = tmp_path / "true.sh"
    fake_installer.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    fake_installer.chmod(0o755)

    import nora.package_installer as pi
    import nora.env_detect as env_detect

    monkeypatch.setattr(
        pi, "_python_command",
        lambda binary, packages, action: [str(fake_installer)],
    )
    monkeypatch.setattr(env_detect, "detect_environment", lambda: _stub_environment("/usr/bin/python3"))

    captured: list[subprocess.Popen] = []

    def _register(proc: subprocess.Popen) -> None:
        captured.append(proc)

    asyncio.run(pi.install_packages(
        "Python", ["pandas"], "install", proc_register=_register,
    ))
    assert len(captured) == 1, "proc_register was not invoked"
    assert isinstance(captured[0], subprocess.Popen)
