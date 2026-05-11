"""UI confirmation gate for the ``install_packages`` tool.

Background
----------
The ``install_packages`` tool reaches outside the sandbox to mutate the
researcher's machine — pip / R install / SSC writes to the user
library. The system prompt instructs the model to ask the researcher
in chat first, but a prompt-only consent gate is brittle: a jailbroken
or buggy model can call the tool directly. Both providers (Anthropic
via ``ALLOWED_TOOL_NAMES`` and OpenAI via the generic handler
dispatch) allow the tool through without an additional runtime gate.

This module is the hard gate. The tool handler calls
``request_confirmation`` and only proceeds with the install if a
human-side approval comes back. The approval is surfaced through the
bridge, which emits an event to the page; the page shows a modal
with the language / action / package names; the researcher clicks
Approve or Deny; that decision returns through the bridge and
resolves the awaiting Future on the tool's event loop.

Threading model
---------------
Confirmations are stored keyed by a one-time token. ``request_confirmation``
captures the tool handler's running event loop at registration time;
``respond`` is called from the bridge's pywebview thread when JS
posts back. The response uses ``loop.call_soon_threadsafe`` to set
the result on the tool's loop, so a tool handler awaiting the Future
wakes up correctly regardless of which thread the bridge runs on.

Test / headless mode
--------------------
If no emitter is registered (e.g., the MCP server is being exercised
outside the UI in tests, or the bridge couldn't attach to a window),
``request_confirmation`` defaults to denying the install. This is the
safe choice for a tool that mutates the user's environment — failing
closed prevents an accidentally-headless deployment from auto-
installing packages.

Tests that want to exercise an approved install can register a
synchronous emitter via ``set_request_emitter`` that calls
``respond(token, True)`` immediately, simulating an auto-approval.
"""

from __future__ import annotations

import asyncio
import secrets
from dataclasses import dataclass
from typing import Callable


@dataclass
class _PendingConfirmation:
    """One in-flight confirmation request.

    ``future`` is the awaiting Future the tool handler is blocked on.
    ``loop`` is the event loop that future lives on — required because
    ``respond`` may be called from a different thread (the pywebview
    bridge thread) than the one running the tool.
    """
    future: asyncio.Future
    loop: asyncio.AbstractEventLoop
    language: str
    packages: tuple[str, ...]
    action: str


# Module-level registry of pending confirmations, keyed by token.
# Lifetime: entries are removed when the future resolves or times
# out. A leak would only happen if neither the response nor the
# timeout fires, which is impossible given the bounded ``wait_for``
# in ``request_confirmation``.
_pending: dict[str, _PendingConfirmation] = {}

# Emitter callable registered by the bridge. Signature:
#   (token: str, language: str, packages: list[str], action: str) -> None
# Implementations push the request to the UI (e.g. evaluate_js a
# ``nora_event``). ``None`` means no UI is attached — the gate fails
# closed.
EmitterFn = Callable[[str, str, list[str], str], None]
_request_emitter: EmitterFn | None = None

# Default timeout for a researcher to respond. Five minutes is
# generous enough that the researcher can context-switch, glance at
# the modal, and approve without the request silently expiring.
# Below 60 seconds is too aggressive — even a quick "is this right?"
# check on a multi-package install can take that long. Above 10
# minutes risks a stale modal accumulating across an unattended
# laptop sleep.
DEFAULT_TIMEOUT_SECONDS = 300.0


def set_request_emitter(emitter: EmitterFn) -> None:
    """Bridge calls this on attach. Replaces any previously-registered
    emitter — only one bridge is active at a time, and re-attaching
    after a window restart should overwrite the stale handle."""
    global _request_emitter
    _request_emitter = emitter


def clear_request_emitter() -> None:
    """Drop the emitter — used on bridge teardown and from tests."""
    global _request_emitter
    _request_emitter = None


def _has_emitter() -> bool:
    """Test/introspection helper. Avoids exporting the private global."""
    return _request_emitter is not None


async def request_confirmation(
    language: str,
    packages: list[str] | tuple[str, ...],
    action: str,
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> bool:
    """Block until the researcher approves or denies the install.

    Returns:
        True iff the researcher explicitly approved. Anything else —
        no emitter registered, timeout, JS posted back deny, exception
        from the emitter — returns False. The tool layer treats False
        as "do not install."
    """
    emitter = _request_emitter
    if emitter is None:
        # Failing closed. The system prompt's request-in-chat workflow
        # is the prompt-only side; this module is the hard gate. With
        # no UI to surface the request, the researcher cannot
        # affirmatively consent, so the safe default is deny.
        return False
    loop = asyncio.get_running_loop()
    fut: asyncio.Future = loop.create_future()
    token = secrets.token_hex(16)
    _pending[token] = _PendingConfirmation(
        future=fut,
        loop=loop,
        language=language,
        packages=tuple(packages),
        action=action,
    )
    try:
        try:
            emitter(token, language, list(packages), action)
        except Exception:  # noqa: BLE001 — emitter is bridge-supplied
            # Emitter raised (e.g. window disappeared mid-call). Treat
            # as a failed emit and deny rather than blocking forever
            # waiting for a response that will never come.
            return False
        try:
            result = await asyncio.wait_for(fut, timeout=timeout)
        except asyncio.TimeoutError:
            return False
        return bool(result)
    finally:
        # Always clear the registry. Even on a successful response,
        # the entry has served its purpose; leaving it would
        # accumulate stale tokens across many installs.
        _pending.pop(token, None)


def respond(token: str, approved: bool) -> bool:
    """Resolve a pending confirmation.

    Returns:
        True if a matching pending request was found and the result
        was scheduled. False if the token was unknown (expired or
        never issued).

    Safe to call from any thread — uses ``call_soon_threadsafe`` to
    schedule the result on the tool's event loop.
    """
    pending = _pending.get(token)
    if pending is None:
        return False
    decision = bool(approved)

    def _set_result() -> None:
        if not pending.future.done():
            pending.future.set_result(decision)

    try:
        pending.loop.call_soon_threadsafe(_set_result)
    except RuntimeError:
        # Loop is closed — tool has already given up (timeout) and
        # bowed out. Nothing to do; the registry entry will have
        # been cleared by the timeout path.
        return False
    return True


def cancel_all() -> None:
    """Resolve every pending confirmation as deny. Bridge calls this
    on teardown so an awaiting tool handler doesn't block on a
    confirmation that can never arrive (the page is gone)."""
    for token in list(_pending.keys()):
        respond(token, False)
