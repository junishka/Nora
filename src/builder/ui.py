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

Session model (new in this commit):

- With a ``cwd`` argument, behave like before: open straight into
  the chat view against that directory.
- Without one, show a landing screen: drop files in, or click
  "Choose files" (native file picker) or "Choose folder". Files
  are staged into ``~/.builder-sessions/<timestamp>_<id>/`` —
  spaces-free, persistent across restarts, Stata-safe. A "Choose
  folder" uses the folder as cwd directly (no copy, no staging).

The session dir lives outside the researcher's project so the
sandbox scope is exactly the files they uploaded — not whatever
else happened to be in the Dropbox folder.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import threading
import uuid
from datetime import datetime, timezone
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


# Where uploaded-file sessions live. Chosen for three properties:
# 1. No spaces — Stata's batch-mode parser trips on them.
# 2. Per-user and persistent — researchers can come back to a
#    past session and look at its `.builder/results.db`.
# 3. Outside any Dropbox / iCloud path — the sandbox scope is
#    exactly the files that were uploaded, not whatever else the
#    researcher happened to have in the source directory.
SESSIONS_ROOT = Path.home() / ".builder-sessions"


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

    def __init__(self, cwd: Path | None = None):
        # cwd may be None at construction time — the UI's landing
        # screen lets the researcher pick files or a folder on first
        # run. `set_cwd` finalizes it.
        self.cwd: Path | None = cwd
        self._window: Any = None  # set after the window is created
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        # Persistent SDK client for the session. Opened lazily on first
        # send_message; closed on shutdown.
        self._client: ClaudeSDKClient | None = None
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

    def ui_ready(self) -> dict[str, Any]:
        """Called by the page after its JS has loaded. Returns a
        payload describing the startup state — either "ready to
        chat" (cwd already set from argv) or "needs session" (land
        on the drop / choose-files screen)."""
        if self.cwd is None:
            return {"state": "needs_session"}
        return self._ready_payload()

    def choose_files(self) -> dict[str, Any]:
        """Open a native file-picker dialog (multi-select) restricted
        to the data formats Builder understands, then stage the
        selected files into a new session dir.

        Returns: ``{ok: True, ...ready_payload}`` on success,
        ``{ok: False, reason: str}`` on cancel / failure.
        """
        if self._window is None:
            return {"ok": False, "reason": "window not ready"}
        try:
            import webview
            # WKWebView wants a single "glob" expression per type. On
            # macOS the picker still shows "All data files" as the
            # most useful option; researchers can switch to "All files"
            # if they want to ignore extension filtering.
            file_types = (
                "Data files (*.csv;*.dta;*.rds)",
                "CSV (*.csv)",
                "Stata (*.dta)",
                "R (*.rds)",
                "All files (*.*)",
            )
            result = self._window.create_file_dialog(
                webview.OPEN_DIALOG,
                allow_multiple=True,
                file_types=file_types,
            )
        except Exception as e:  # noqa: BLE001 — webview can error in various ways
            return {"ok": False, "reason": f"dialog error: {e}"}
        if not result:
            return {"ok": False, "reason": "cancelled"}
        return self._stage_session(list(result))

    def choose_folder(self) -> dict[str, Any]:
        """Open a native folder-picker dialog. The chosen folder is
        used as cwd directly — no copy, no staging. Convenient when
        the researcher already has a tidy project directory.

        Returns ``{ok: True, ...ready_payload}`` or
        ``{ok: False, reason}``.
        """
        if self._window is None:
            return {"ok": False, "reason": "window not ready"}
        try:
            import webview
            result = self._window.create_file_dialog(
                webview.FOLDER_DIALOG, allow_multiple=False
            )
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "reason": f"dialog error: {e}"}
        if not result:
            return {"ok": False, "reason": "cancelled"}
        folder = Path(result[0]).expanduser().resolve()
        if not folder.is_dir():
            return {"ok": False, "reason": f"not a directory: {folder}"}
        return self._set_cwd(folder)

    def upload_files(
        self, files: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Receive files from JS drag-and-drop. Each ``files[i]`` is
        ``{name: str, content: str (base64)}``. We decode and stage
        into a fresh session dir.

        Size capped per-file at 500 MB (plenty for typical research
        data; the real issue is memory while transferring through
        the bridge). Callers enforcing larger files should use the
        file-picker path which reads directly from disk.
        """
        if not files:
            return {"ok": False, "reason": "no files"}
        import base64
        max_bytes = 500 * 1024 * 1024
        decoded: list[tuple[str, bytes]] = []
        for item in files:
            name = item.get("name", "")
            content_b64 = item.get("content", "")
            if not name or not isinstance(content_b64, str):
                continue
            # Strip any data URL prefix JS may have added.
            if "," in content_b64:
                content_b64 = content_b64.split(",", 1)[1]
            try:
                blob = base64.b64decode(content_b64, validate=False)
            except Exception:  # noqa: BLE001
                return {"ok": False, "reason": f"could not decode {name!r}"}
            if len(blob) > max_bytes:
                return {
                    "ok": False,
                    "reason": (
                        f"{name!r} is {len(blob) // (1024*1024)} MB — "
                        f"above the 500 MB drag-drop cap. Use Choose "
                        f"Files instead; it reads directly from disk."
                    ),
                }
            decoded.append((Path(name).name, blob))
        if not decoded:
            return {"ok": False, "reason": "no valid files in drop"}
        return self._stage_session_from_blobs(decoded)

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
        if self.cwd is None:
            self._push_event({
                "type": "turn_error",
                "message": (
                    "no working directory set — choose files or a "
                    "folder first"
                ),
            })
            return
        asyncio.run_coroutine_threadsafe(
            self._run_turn(text), self._loop
        )

    # -------- internals --------

    def _set_cwd(self, path: Path) -> dict[str, Any]:
        """Finalize the working directory and return the ready
        payload. Builds up the config.get_cwd side-effect at the
        module level so other code (schema, policy, executor) picks
        it up via the singleton."""
        self.cwd = path
        set_cwd(path)
        return {"ok": True, "state": "ready", **self._ready_payload()}

    def _stage_session(self, source_paths: list[str]) -> dict[str, Any]:
        """Copy user-selected files into a fresh session dir. Each
        source file must be a real readable file; otherwise the
        whole staging aborts so the session never starts with
        partial data."""
        paths: list[Path] = []
        for s in source_paths:
            p = Path(s).expanduser().resolve()
            if not p.is_file():
                return {"ok": False, "reason": f"not a file: {p}"}
            paths.append(p)
        session = _new_session_dir()
        try:
            for src in paths:
                shutil.copy2(src, session / src.name)
        except OSError as e:
            return {"ok": False, "reason": f"copy failed: {e}"}
        return self._set_cwd(session)

    def _stage_session_from_blobs(
        self, blobs: list[tuple[str, bytes]]
    ) -> dict[str, Any]:
        """Drag-drop path: write decoded bytes into a fresh session
        dir. Filenames are basename-only (sanitized by the caller)
        to prevent path traversal."""
        session = _new_session_dir()
        try:
            for name, content in blobs:
                target = session / name
                # Defense in depth: never write outside the session.
                if target.resolve().parent != session.resolve():
                    return {"ok": False, "reason": f"suspicious filename: {name!r}"}
                target.write_bytes(content)
        except OSError as e:
            return {"ok": False, "reason": f"write failed: {e}"}
        return self._set_cwd(session)

    def _ready_payload(self) -> dict[str, Any]:
        """The event body emitted after cwd is finalized — tells the
        UI to switch from landing to chat view."""
        assert self.cwd is not None
        return {
            "type": "ready",
            "cwd": str(self.cwd),
            "greeting": (
                "Connected. Tell Claude about your research question, "
                "or ask what datasets are available."
            ),
            "policy": self._policy_summary(),
        }

    async def _ensure_client(self) -> ClaudeSDKClient:
        assert self.cwd is not None
        if self._client is None:
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
        js = f"window.builder_event({json.dumps(payload)});"
        try:
            self._window.evaluate_js(js)
        except Exception:  # noqa: BLE001 — webview may be closing
            pass

    def _policy_summary(self) -> dict[str, Any]:
        """Compact JSON-serializable summary of the current policy +
        dataset list for the topbar footer."""
        if self.cwd is None:
            return {"default_max_depth": "names_types", "datasets": []}
        from builder.app import _scan_datasets
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


def _new_session_dir() -> Path:
    """Create a new per-session staging dir under SESSIONS_ROOT.
    Returns the absolute path. Safe to call repeatedly — each call
    gets a unique timestamp+uuid suffix."""
    SESSIONS_ROOT.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = uuid.uuid4().hex[:8]
    session = SESSIONS_ROOT / f"{ts}_{suffix}"
    session.mkdir(parents=True, exist_ok=False)
    return session


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
        # Raw stdout/stderr comes from run_dir/stdout.log and
        # stderr.log — researcher-visible, never reaches Claude.
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

def _resolve_cwd(raw: str | None) -> Path | None:
    """Validate the optional ``cwd`` argument.

    Returns ``None`` if no argument was given — the UI then shows a
    landing screen that lets the researcher choose / upload. Returns
    an absolute Path if the argument points at an existing directory.
    Exits with a helpful error if the argument was given but bad
    (typically the classic ``~/Users/<name>/…`` path-doubling mistake).
    """
    if not raw:
        return None
    path = Path(raw).expanduser()
    try:
        path = path.resolve()
    except OSError as e:
        print(f"builder-ui: {e}", file=sys.stderr)
        sys.exit(2)
    if not path.is_dir():
        msg = [f"builder-ui: not a directory: {path}"]
        if raw.startswith("~/Users/"):
            suggested = raw.replace("~/Users/", "/Users/", 1)
            msg.append(
                f"  Hint: `~` already expands to /Users/<you>. You "
                f"may have meant: {suggested}"
            )
        elif raw.startswith("~/"):
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
        else:
            msg.append(
                "  Hint: pass an absolute path (like "
                "/Users/bb/Downloads) or a tilde path (~/Downloads), "
                "or launch without a path and choose files from the "
                "landing screen."
            )
        print("\n".join(msg), file=sys.stderr)
        sys.exit(2)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="builder-ui",
        description=(
            "Builder — web UI. Same backend as `builder` (terminal), "
            "different frontend. Launch without a path to drop/choose "
            "files from the UI."
        ),
    )
    parser.add_argument(
        "cwd", nargs="?", default=None,
        help=(
            "Optional. Working directory Builder operates in. If "
            "omitted, the UI prompts for files or a folder on startup."
        ),
    )
    args = parser.parse_args()

    cwd = _resolve_cwd(args.cwd)
    if cwd is not None:
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

    bridge = BuilderBridge(cwd=cwd)
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
