"""Regression: pressing Stop mid-``submit_script`` actually halts the
subprocess.

Field report (2026-04-28): the researcher pressed Stop while a long
Stata regression was running, then sent the same prompt again. The
first script kept running to completion because the executor used
synchronous ``subprocess.run`` and the asyncio task's
``CancelledError`` only fired AFTER the subprocess returned. From
the researcher's seat that looked identical to "Stop did nothing".

The fix: ``executor.run_script`` now spawns the subprocess via
``Popen`` and exposes a ``proc_register`` callback. ``submit_script``
runs the executor in a worker thread (so the asyncio path stays
responsive), records the ``Popen`` handle, and on ``CancelledError``
calls ``proc.kill()`` so the script actually halts.

This test pins that behaviour by submitting a deliberately-long
sleep script and cancelling the asyncio task mid-run; without the
fix, the test takes the full sleep duration. With the fix, it
finishes promptly after the kill.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from nora.config import set_cwd
from nora.env_detect import detect_environment
from nora.tools import submit_script


pytestmark = pytest.mark.skipif(
    detect_environment().r is None,
    reason="R not installed; the cancellable executor path needs an interpreter",
)


def test_cancel_kills_running_subprocess(tmp_path: Path) -> None:
    """Submit a script that sleeps for 30 seconds, cancel the task
    after 1 second, and assert the whole flow finishes in under
    ~5 seconds. Without the kill-on-cancel fix, this test would
    take the full 30 seconds (the synchronous ``subprocess.run``
    blocked the asyncio task until the script returned).
    """
    set_cwd(tmp_path)

    code = """
# Long sleep so we have a wide cancellation window. The runtime
# library is not invoked; we never reach a result emission.
Sys.sleep(30)
nora$result(list(type = "descriptive", n = 1L, missing_count = 0L))
"""

    async def _drive() -> float:
        start = time.monotonic()
        # Spawn the submit_script handler as a task so we can cancel it.
        task = asyncio.create_task(
            submit_script.handler({
                "language": "R",
                "code": code,
                "label": "long sleep",
            })
        )
        # Wait briefly to make sure the subprocess is actually
        # running (Popen returns fast; sleep gives the R process
        # time to start).
        await asyncio.sleep(1.0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return time.monotonic() - start

    elapsed = asyncio.run(_drive())

    # The script slept for 30s. Without the kill, the task waits
    # the full 30s. With the kill, the cancel takes effect within
    # the proc.wait(timeout=2) window, so total elapsed should be
    # well under 10s. Pin a generous ceiling so the test isn't
    # flaky on a busy CI box.
    assert elapsed < 10.0, (
        f"submit_script cancel took {elapsed:.1f}s — Stop should kill "
        f"the subprocess promptly, not wait for it to finish naturally"
    )


def test_no_orphaned_subprocesses_after_cancel(tmp_path: Path) -> None:
    """After a cancelled run, the killed subprocess must NOT be left
    as a zombie or a leaked process. ``Popen.wait`` after ``kill``
    reaps it; we check that the process really is gone."""
    set_cwd(tmp_path)

    code = "Sys.sleep(20)\nnora$result(list(type='descriptive', n=1L, missing_count=0L))\n"

    captured_pid: list[int] = []

    # Monkey-patch the register so we can grab the pid.
    from nora import executor

    real_run = executor.run_script

    def _wrapped(*args, proc_register=None, **kwargs):
        def _capture(p: subprocess.Popen[str]) -> None:
            captured_pid.append(p.pid)
            if proc_register is not None:
                proc_register(p)
        return real_run(*args, proc_register=_capture, **kwargs)

    executor.run_script = _wrapped  # type: ignore[assignment]
    try:
        async def _drive() -> None:
            task = asyncio.create_task(
                submit_script.handler({
                    "language": "R",
                    "code": code,
                    "label": "leak check",
                })
            )
            await asyncio.sleep(0.8)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        asyncio.run(_drive())
    finally:
        executor.run_script = real_run  # type: ignore[assignment]

    assert captured_pid, "executor.run_script wrapper did not see a Popen"
    pid = captured_pid[0]
    # Give the OS a beat to reap. If the proc is still alive,
    # ``os.kill(pid, 0)`` succeeds; once it's reaped, it raises.
    time.sleep(0.5)
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_executor_starts_subprocess_in_new_session(tmp_path: Path) -> None:
    """The executor must pass ``start_new_session=True`` to ``Popen``,
    AND ``_kill_proc_quietly`` must use ``killpg`` on the resulting
    process group. Without both, a user script that spawns
    ``parallel::makeCluster`` / ``multiprocessing.Pool`` workers
    leaves them running as orphans of init when Stop fires —
    ``proc.kill()`` only SIGKILLs the direct child.

    Verify by spawning an in-process subprocess (no sandbox-exec,
    no submit_script plumbing) that itself spawns a grandchild, then
    calling ``_kill_proc_quietly`` and asserting the grandchild dies.
    """
    import signal as _signal
    from nora.runner import _kill_proc_quietly

    pid_file = tmp_path / "grandchild.pid"

    proc = subprocess.Popen(
        [
            sys.executable, "-c",
            (
                "import os, subprocess, time\n"
                f"p = subprocess.Popen(['/bin/sleep', '60'])\n"
                f"open({str(pid_file)!r}, 'w').write(str(p.pid))\n"
                "time.sleep(30)\n"
            ),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )

    # Wait for the grandchild's PID to land on disk.
    for _ in range(80):
        if pid_file.exists() and pid_file.read_text().strip():
            break
        time.sleep(0.05)
    assert pid_file.exists() and pid_file.read_text().strip(), (
        "script never spawned the grandchild — test setup is wrong"
    )
    grandchild_pid = int(pid_file.read_text().strip())

    # Sanity-check: both processes are in the same process group
    # (the one rooted at proc.pid because of ``start_new_session=True``).
    pgid = os.getpgid(proc.pid)
    assert pgid == proc.pid, (
        f"parent should be its own process-group leader; got pgid={pgid}, pid={proc.pid}"
    )
    assert os.getpgid(grandchild_pid) == pgid, (
        f"grandchild ({grandchild_pid}) is in a different process group "
        f"than its parent — start_new_session inheritance is broken"
    )

    # The fix under test: ``_kill_proc_quietly`` should tear down the
    # whole process group, not just the parent.
    _kill_proc_quietly(proc)

    # Both should be gone after the killpg. SIGKILL is synchronous so
    # 200ms is plenty of slack for launchd to reap on macOS / init on
    # Linux.
    time.sleep(0.2)
    with pytest.raises(ProcessLookupError):
        os.kill(proc.pid, 0)
    # If killpg didn't fire (bug), the grandchild — running as an
    # /bin/sleep with no parent — would still be alive here.
    try:
        os.kill(grandchild_pid, 0)
        # If we got here, kill it manually so we don't leak.
        try:
            os.kill(grandchild_pid, _signal.SIGKILL)
        except ProcessLookupError:
            pass
        pytest.fail(
            f"grandchild {grandchild_pid} survived _kill_proc_quietly — "
            f"process-group kill did NOT reach it"
        )
    except ProcessLookupError:
        pass  # expected — killpg reached the grandchild
