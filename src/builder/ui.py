"""Builder — web UI entry point.

Opens a native WKWebView window (via pywebview) hosting a local HTML
chat interface, and bridges it to the same backend plumbing the
terminal UI uses: ``ClaudeSDKClient`` + the 5 MCP tools + sanitizer +
policy + sandboxed executor.

Launched via:

    uv run python -m builder.ui [cwd]

or the ``builder-ui`` console script. Terminal UI (``builder``) is
unchanged and still works. The two share everything except
rendering.

MVP scope for this session:
- Render assistant text, thinking, tool calls, tool results.
- Send a user message from the web form.
- Show schema policy summary in the composer footer.
- Surface auth / error events visibly.

Deferred to later sessions:
- Markdown + syntax-highlighted code blocks in assistant text.
- Inline raw R/Stata output panel (currently just shows the path).
- Policy editing in the UI (`/policy` wizard).
- Dataset picker sidebar.
- Packaging the web assets into the .app bundle.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import threading
from concurrent.futures import Future
from pathlib import Path
from typing import Any

from claude_agent_sdk import ClaudeSDKClient, ClaudeSDKError

from builder import chat_service
from builder.config import set_cwd
from builder.env_detect import detect_environment
from builder.policy import (
    get_max_depth,
    has_explicit_policy,
    load_policy,
)


# ---------------------------------------------------------------------------
# The bridge between the web UI and the Python backend
# ---------------------------------------------------------------------------

class BuilderBridge:
    """JS-visible API. pywebview exposes methods on this object to the
    loaded page as ``window.pywebview.api.<method_name>``.

    Each method runs on a background thread pywebview manages. The
    Claude SDK is asyncio-based, so we drive it from a dedicated
    event loop in a worker thread (see ``_loop_thread``).
    """

    def __init__(self, cwd: Path):
        self.cwd = cwd
        self._window: Any = None  # set after the window is created
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        # Persistent SDK client for the session. Opened lazily on first
        # send_message; closed on shutdown.
        self._client: ClaudeSDKClient | None = None
        self._client_ready = asyncio.Event()  # set once entered
        # A lock to serialize send_message calls. The SDK client
        # assumes one turn at a time.
        self._send_lock: asyncio.Lock | None = None

    # -------- lifecycle --------

    def attach(self, window: Any) -> None:
        self._window = window

    def start_loop(self) -> None:
        """Start the asyncio worker thread. Called once, before the
        webview starts serving the page."""
        self._loop = asyncio.new_event_loop()
        self._send_lock = asyncio.Lock()

        def _run() -> None:
            asyncio.set_event_loop(self._loop)
            self._loop.run_forever()

        self._loop_thread = threading.Thread(
            target=_run, name="builder-ui-loop", daemon=True
        )
        self._loop_thread.start()

    def stop_loop(self) -> None:
        if self._loop is None:
            return
        # Close the SDK client if we opened one.
        async def _close() -> None:
            if self._client is not None:
                try:
                    await self._client.__aexit__(None, None, None)
                except Exception:
                    pass

        fut = asyncio.run_coroutine_threadsafe(_close(), self._loop)
        try:
            fut.result(timeout=3)
        except Exception:
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)

    # -------- JS-visible methods --------

    def ui_ready(self) -> None:
        """Called by the page after its JS has loaded. Emit initial
        state so the UI can populate itself."""
        self._push_event({
            "type": "ready",
            "cwd": str(self.cwd),
            "greeting": (
                "Connected. Tell Claude about your research question, "
                "or ask what datasets are available."
            ),
            "policy": self._policy_summary(),
        })

    def send_message(self, text: str) -> None:
        """Called from the web form. Runs a single chat turn on the
        worker loop. Returns immediately; events stream back via
        `_push_event`."""
        if self._loop is None:
            self._push_event({
                "type": "turn_error",
                "message": "worker loop not running",
            })
            return
        asyncio.run_coroutine_threadsafe(
            self._run_turn(text), self._loop
        )

    # -------- internals --------

    async def _ensure_client(self) -> ClaudeSDKClient:
        if self._client is None:
            # Build the same ClaudeAgentOptions the terminal app uses.
            # Importing lazily to avoid a circular import at module
            # load — app.py imports from ui.py's siblings.
            from builder.app import _build_options
            opts = _build_options(self.cwd)
            self._client = await ClaudeSDKClient(options=opts).__aenter__()
        return self._client

    async def _run_turn(self, text: str) -> None:
        assert self._send_lock is not None
        async with self._send_lock:
            try:
                client = await self._ensure_client()
            except ClaudeSDKError as e:
                self._push_event({
                    "type": "turn_error",
                    "message": f"SDK error: {e}",
                })
                return
            except Exception as e:  # noqa: BLE001
                self._push_event({
                    "type": "turn_error",
                    "message": f"client setup failed: {e}",
                })
                return

            try:
                async for evt in chat_service.run_turn(client, text):
                    self._push_event(_event_to_dict(evt))
            except ClaudeSDKError as e:
                self._push_event({
                    "type": "turn_error",
                    "message": f"SDK error during turn: {e}",
                })
            except Exception as e:  # noqa: BLE001
                self._push_event({
                    "type": "turn_error",
                    "message": f"turn failed: {e}",
                })

    def _push_event(self, payload: dict[str, Any]) -> None:
        """Send a JSON event to the web UI. pywebview's evaluate_js
        takes a string of JS to run in the page's context."""
        if self._window is None:
            return
        # json.dumps so Python None / True / False / dicts render as
        # JSON literals in JS land.
        js = f"window.builder_event({json.dumps(payload)});"
        try:
            self._window.evaluate_js(js)
        except Exception:  # noqa: BLE001 — webview may be closing
            pass

    def _policy_summary(self) -> dict[str, Any]:
        """Build a tiny JSON-serializable summary of the current
        policy + dataset list for the topbar footer. Re-compute this
        any time the policy changes (TODO: wire into /policy)."""
        from builder.app import _scan_datasets  # reuse terminal helper
        policy = load_policy(self.cwd)
        return {
            "default_max_depth": policy.default_max_depth,
            "datasets": [
                {
                    "name": p.name,
                    "ceiling": get_max_depth(policy, p.name),
                    "explicit": has_explicit_policy(policy, p.name),
                }
                for p in _scan_datasets(self.cwd)
            ],
        }


def _read_raw_logs(run_dir: str | None) -> tuple[str, str]:
    """Read ``stdout.log`` and ``stderr.log`` from a run dir, if they
    exist. Returns empty strings when the dir is missing or the files
    haven't been written. Mirrors ``app.py:_read_log``.

    Content is capped at 32 KB per stream — enough to show a full
    regression table, short of letting a runaway log blow up the
    browser. The full log is still on disk at ``run_dir`` for audit.
    """
    if not run_dir:
        return "", ""
    stdout_text = ""
    stderr_text = ""
    for name, bucket in (("stdout.log", "stdout"), ("stderr.log", "stderr")):
        try:
            content = Path(run_dir, name).read_text(
                encoding="utf-8", errors="replace"
            )
        except OSError:
            continue
        cap = 32 * 1024
        if len(content) > cap:
            content = content[-cap:] + f"\n[… truncated; full log at {run_dir}/{name}]"
        if bucket == "stdout":
            stdout_text = content
        else:
            stderr_text = content
    return stdout_text, stderr_text


def _event_to_dict(evt: Any) -> dict[str, Any]:
    """Flatten a chat_service.Event dataclass into a JSON-serializable
    dict with a type tag the JS side switches on."""
    if isinstance(evt, chat_service.AssistantText):
        return {"type": "assistant_text", "text": evt.text}
    if isinstance(evt, chat_service.AssistantThinking):
        return {"type": "assistant_thinking", "text": evt.text}
    if isinstance(evt, chat_service.ToolCall):
        return {
            "type": "tool_call",
            "name": evt.name,
            "input": evt.input,
            "call_id": evt.call_id,
        }
    if isinstance(evt, chat_service.ToolCallResult):
        # Read the raw stdout/stderr logs when the tool call was a
        # submit_script or expand_result — those are the two paths
        # that plant `_run_dir` in their response for this purpose.
        # Raw logs are researcher-only: they reach the UI, never
        # Claude (the executor's stderr-isolation regression locks
        # that in; see test_stderr_isolation.py).
        raw_stdout, raw_stderr = _read_raw_logs(evt.run_dir)
        return {
            "type": "tool_result",
            "call_id": evt.call_id,
            "text": evt.text,
            "is_error": evt.is_error,
            "run_dir": evt.run_dir,
            "raw_stdout": raw_stdout,
            "raw_stderr": raw_stderr,
        }
    if isinstance(evt, chat_service.TurnDone):
        return {
            "type": "turn_done",
            "input_tokens": evt.input_tokens,
            "output_tokens": evt.output_tokens,
            "cost_usd": evt.cost_usd,
        }
    if isinstance(evt, chat_service.AuthFailure):
        return {"type": "auth_failure", "reason": evt.reason}
    if isinstance(evt, chat_service.TurnError):
        return {"type": "turn_error", "message": evt.message}
    return {"type": "unknown"}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _resolve_cwd(raw: str | None) -> Path:
    """Validate the data directory argument; fall back to ~/Documents.

    Includes a friendlier error message for the most common misformat
    — typing ``~/Users/<name>/...`` (which expands to
    ``/Users/<name>/Users/<name>/...``) when the researcher means
    ``~/...`` or ``/Users/<name>/...``.
    """
    if raw:
        path = Path(raw).expanduser()
    else:
        path = Path.home() / "Documents"
    try:
        path = path.resolve()
    except OSError as e:
        print(f"builder-ui: {e}", file=sys.stderr)
        sys.exit(2)
    if not path.is_dir():
        msg = [f"builder-ui: not a directory: {path}"]
        # Common path-expansion gotcha: `~/Users/<name>/...` expands
        # to `/Users/<name>/Users/<name>/...` because `~` already
        # means `/Users/<name>`. Detect and suggest the fix.
        if raw and raw.startswith("~/Users/"):
            suggested = raw.replace("~/Users/", "/Users/", 1)
            msg.append(
                f"  Hint: `~` already expands to /Users/<you>. You "
                f"may have meant: {suggested}"
            )
        elif raw and raw.startswith("~/"):
            home = Path.home()
            parent = path.parent
            if parent.is_dir():
                siblings = sorted(
                    p.name for p in parent.iterdir() if p.is_dir()
                )[:10]
                if siblings:
                    msg.append(
                        f"  Hint: in {parent}, I see: "
                        f"{', '.join(siblings)}"
                    )
            _ = home
        else:
            msg.append(
                "  Hint: pass an absolute path (like "
                "/Users/bb/Downloads) or a tilde path (~/Downloads)."
            )
        print("\n".join(msg), file=sys.stderr)
        sys.exit(2)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="builder-ui",
        description=(
            "Builder — web UI. Same backend as `builder` (terminal), "
            "different frontend."
        ),
    )
    parser.add_argument(
        "cwd", nargs="?", default=None,
        help=(
            "Working directory — the sandbox Builder operates in. "
            "Defaults to ~/Documents."
        ),
    )
    args = parser.parse_args()

    cwd = _resolve_cwd(args.cwd)
    set_cwd(cwd)

    # Preflight: sandbox-exec on macOS. The webview can still run for
    # schema + request_data even without it, but the user should see
    # the warning.
    env = detect_environment()
    if env.sandbox_exec is None:
        print(
            "builder-ui: warning — sandbox-exec unavailable; "
            "submit_script will refuse to run.",
            file=sys.stderr,
        )

    import webview  # lazy import so `builder` (terminal) doesn't
                    # require pywebview at import time

    web_dir = Path(__file__).parent / "web"
    index_path = web_dir / "index.html"
    if not index_path.is_file():
        print(
            f"builder-ui: missing web assets at {index_path}",
            file=sys.stderr,
        )
        sys.exit(2)

    bridge = BuilderBridge(cwd)
    bridge.start_loop()

    window = webview.create_window(
        title="Builder",
        url=str(index_path),
        js_api=bridge,
        width=960,
        height=720,
        resizable=True,
    )
    bridge.attach(window)

    try:
        # debug=False: no dev-tools context menu in production. Flip
        # to True while iterating on the web UI.
        webview.start(debug=False)
    finally:
        bridge.stop_loop()


if __name__ == "__main__":
    main()
