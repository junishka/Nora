"""Turn identity propagation for tool handlers.

Every chat turn carries a unique id assigned at the bridge before it
hits the runner. The id is the routing key the cancellation story
hangs on:

- The runner stamps every event with the active turn id, so the
  bridge can drop late events from a turn the researcher already
  cancelled (instead of relying only on the JS-side filter, which
  used to clear its suppression flag the moment a new message
  started — letting late events from the old turn slip through).
- ``submit_script`` registers its subprocess into the runner's
  per-turn registry under the active turn id. If interrupt fires
  before the registration window closes, the runner has already
  marked the turn cancelled, so the registration site kills the
  subprocess synchronously instead of letting it run to completion.

Mirrors the ``use_cwd`` / ``get_cwd`` pattern from ``config.py`` —
contextvars scoped to the asyncio task running the turn, so
concurrent runners don't trample each other. Tool handlers read
``current_turn_id()`` and ``register_turn_process(...)``; the runner
binds the variables via ``use_turn_context(...)``.
"""

from __future__ import annotations

import subprocess
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator, Protocol


class _TurnRegistry(Protocol):
    """Subset of ``SessionRunner`` that the registration helpers below
    need. Defined as a Protocol so this module doesn't import the
    runner (and create a cycle).
    """

    def register_turn_process(
        self, turn_id: str, proc: subprocess.Popen[Any]
    ) -> None:
        ...

    def is_turn_cancelled(self, turn_id: str) -> bool:
        ...


# Per-asyncio-task overrides. Sister tasks (concurrent runners) see
# their own bindings, or ``None`` if no turn is in flight on the
# context.
_turn_id_var: ContextVar[str | None] = ContextVar(
    "nora_turn_id", default=None,
)
_runner_var: ContextVar[_TurnRegistry | None] = ContextVar(
    "nora_turn_runner", default=None,
)


def current_turn_id() -> str | None:
    """Return the turn id bound to the current asyncio task, or
    ``None`` outside a runner-managed turn (startup, tests).
    """
    return _turn_id_var.get()


def is_current_turn_cancelled() -> bool:
    """True iff the runner has marked the current turn cancelled.

    Tool handlers can poll this to short-circuit work after a long
    subprocess returns: if Stop fired during execution, dropping the
    sanitization / persistence chain matches the user's expectation
    that cancelled-turn output never enters the chat or the result
    store.
    """
    runner = _runner_var.get()
    turn_id = _turn_id_var.get()
    if runner is None or turn_id is None:
        return False
    return runner.is_turn_cancelled(turn_id)


def register_turn_process(proc: subprocess.Popen[Any]) -> None:
    """Register a subprocess against the current turn.

    If interrupt fires before this register call returns, the runner
    has already flipped the turn's cancelled flag — the runner's
    own ``register_turn_process`` checks that flag under its lock
    and kills ``proc`` immediately. That closes the
    Popen-returned-but-not-yet-registered race that the previous
    ``proc_box`` pattern had.

    No-op outside a runner-managed turn (terminal path, tests). The
    subprocess still runs to completion in those contexts; the
    runner's interrupt-driven kill path simply isn't relevant
    without a runner.
    """
    runner = _runner_var.get()
    turn_id = _turn_id_var.get()
    if runner is None or turn_id is None:
        return
    runner.register_turn_process(turn_id, proc)


@contextmanager
def use_turn_context(
    turn_id: str, runner: _TurnRegistry,
) -> Iterator[str]:
    """Bind ``turn_id`` and ``runner`` for the current asyncio task.

    Usage from the runner:

        with use_cwd(self.cwd), use_turn_context(turn_id, self):
            async for evt in session.send(...):
                ...

    Tool handlers running under that context see the turn id via
    ``current_turn_id()`` and register subprocesses via
    ``register_turn_process(...)``. Sister tasks running other
    runners' turns concurrently see their own bindings.
    """
    id_token = _turn_id_var.set(turn_id)
    runner_token = _runner_var.set(runner)
    try:
        yield turn_id
    finally:
        _turn_id_var.reset(id_token)
        _runner_var.reset(runner_token)
