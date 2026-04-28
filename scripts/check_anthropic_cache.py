"""Headless probe: does Anthropic prompt-caching fire on Nora's setup?

Drives the real ``AnthropicSession`` through two short turns against a
throwaway temp dir, captures the ``TurnDone`` event from each turn, and
prints the per-turn token usage including ``cache_read_input_tokens``
and ``cache_creation_input_tokens``.

Interpretation:

- **Turn 1**: ``cache_creation_input_tokens`` should be roughly the size
  of the system prompt + tool schemas (~12k for Nora's setup).
  ``cache_read_input_tokens`` is 0 (nothing in the cache yet).
- **Turn 2**: ``cache_read_input_tokens`` should be ~the same number,
  served from cache. ``cache_creation`` drops to 0 (or just covers any
  new content added since turn 1).

If both ``cache_read`` and ``cache_creation`` are 0 across both turns,
caching is NOT firing — the CLI is not setting ``cache_control`` on
the prefix.

Run::

    uv run python scripts/check_anthropic_cache.py

Requires a working Anthropic auth source (``ANTHROPIC_API_KEY`` env,
``claude`` CLI subscription, or keyring-stored key — same resolution
order as the app).
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from nora.provider.anthropic import AnthropicSession, detect_auth
from nora.provider.base import TurnDone, TurnError, AuthFailure
from nora.system_prompt import build_system_prompt
from nora.tools import SERVER_NAME


# Pick a small, cheap model for the probe.
PROBE_MODEL = "claude-sonnet-4-6"

PROBE_TURNS = (
    "Reply with just the word 'one'.",
    "Reply with just the word 'two'.",
)


async def probe() -> int:
    auth = detect_auth()
    if auth == "unknown":
        print(
            "[check_anthropic_cache] no Anthropic auth detected. "
            "Set ANTHROPIC_API_KEY, sign into the claude CLI, or "
            "store a key in the keyring before running."
        )
        return 1

    print(f"[check_anthropic_cache] auth source: {auth}")
    print(f"[check_anthropic_cache] model: {PROBE_MODEL}")

    with tempfile.TemporaryDirectory() as td:
        cwd = Path(td)
        sys_prompt = build_system_prompt(cwd, SERVER_NAME)
        print(f"[check_anthropic_cache] system_prompt chars: {len(sys_prompt)} "
              f"(~{len(sys_prompt) // 4} tokens)")

        session = AnthropicSession(
            cwd=cwd,
            model=PROBE_MODEL,
            system_prompt=sys_prompt,
        )
        await session.open()
        try:
            for i, prompt in enumerate(PROBE_TURNS, start=1):
                done: TurnDone | None = None
                err: str | None = None
                async for evt in session.send(prompt):
                    if isinstance(evt, TurnDone):
                        done = evt
                    elif isinstance(evt, TurnError):
                        err = evt.message
                    elif isinstance(evt, AuthFailure):
                        err = f"auth: {evt.reason}"
                if err:
                    print(f"[turn {i}] FAILED: {err}")
                    return 2
                if done is None:
                    print(f"[turn {i}] no TurnDone event")
                    return 2
                print(
                    f"[turn {i}] "
                    f"input={done.input_tokens} "
                    f"output={done.output_tokens} "
                    f"cache_read={done.cache_read_input_tokens} "
                    f"cache_creation={done.cache_creation_input_tokens} "
                    f"cost_usd={done.cost_usd}"
                )
        finally:
            await session.close()

    print("[check_anthropic_cache] done.")
    print("[check_anthropic_cache] interpretation:")
    print("  - turn 1 cache_creation should be ~system prompt + tools (~12k).")
    print("  - turn 2 cache_read should be roughly the same value.")
    print("  - if both stay at 0, prompt caching is NOT firing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(probe()))
