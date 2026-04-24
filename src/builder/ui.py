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
    VALID_DEPTHS,
    BuilderPolicy,
    DatasetPolicy,
    get_max_depth,
    has_explicit_policy,
    load_policy,
    save_policy,
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
        # Handle to the currently-running turn's asyncio Task, so
        # `interrupt_turn` can cancel it when the researcher clicks
        # the Stop button. None when no turn is in flight. Captured
        # on the worker loop, cleared when the turn returns.
        self._current_turn_task: asyncio.Task[None] | None = None
        # Which Claude model the researcher has selected. Defaults to
        # Sonnet 4.6 (1M context). Changed via `set_model` from the
        # composer chip; takes effect on the next turn because
        # `_ensure_client` reads it when opening a fresh SDK client.
        from builder.app import DEFAULT_MODEL
        self._model: str = DEFAULT_MODEL

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
        into a fresh session dir. Multiple files are supported —
        they all land in the same session.

        Size capped per-file at 2 GB. The real constraint is peak
        memory while transferring through the bridge: a file of N
        bytes needs roughly 3N during upload (JS ArrayBuffer + JS
        base64 string + Python-side decode), so 2 GB is already
        ~6 GB of peak heap. Larger datasets should use the file
        picker (`choose_files`) which copies directly from disk
        with no memory overhead.
        """
        if not files:
            return {"ok": False, "reason": "no files"}
        import base64
        max_bytes = 2 * 1024 * 1024 * 1024
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
                mb = len(blob) // (1024 * 1024)
                return {
                    "ok": False,
                    "reason": (
                        f"{name!r} is {mb} MB — above the 2 GB "
                        f"drag-drop cap. Use Choose Files… instead; "
                        f"it copies straight from disk with no "
                        f"memory overhead, so there's no size limit."
                    ),
                }
            decoded.append((Path(name).name, blob))
        if not decoded:
            return {"ok": False, "reason": "no valid files in drop"}
        return self._stage_session_from_blobs(decoded)

    def open_path(
        self, path: str, mode: str | None = None
    ) -> dict[str, Any]:
        """Open a file or folder in its macOS default handler.

        ``mode`` tunes the invocation when the researcher wants more
        than a plain "open":

          - ``None`` (default) — hand to LaunchServices. R files
            open in RStudio / R.app, .do files in Stata, logs in
            TextEdit, folders in Finder.
          - ``"run_stata"`` — launch Stata with the file as a
            do-file argument. Whether Stata auto-runs on open
            depends on the user's Stata preferences, but at
            minimum Stata opens with the file loaded; the
            researcher presses Cmd-D to execute.
          - ``"run_r"`` — launch RStudio with the file. Same caveat
            — user presses Cmd-Enter to source.

        **Restricted to Builder-managed locations** — only paths
        inside the current session's working directory or inside
        the ``~/.builder-sessions/`` tree are allowed.
        """
        if not path:
            return {"ok": False, "reason": "empty path"}
        try:
            p = Path(path).expanduser().resolve()
        except OSError as e:
            return {"ok": False, "reason": f"bad path: {e}"}
        if not p.exists():
            return {"ok": False, "reason": f"not found: {p}"}
        allowed_roots: list[Path] = [SESSIONS_ROOT.resolve()]
        if self.cwd is not None:
            allowed_roots.append(self.cwd.resolve())
        if not any(_is_within(p, root) for root in allowed_roots):
            return {
                "ok": False,
                "reason": (
                    "path is outside Builder's managed directories — "
                    "refused as a precaution"
                ),
            }
        import subprocess
        # Default: let LaunchServices pick the handler.
        cmd: list[str] = ["/usr/bin/open", str(p)]
        if mode == "run_stata":
            # Try common Stata .app names in order of likelihood. We
            # bias toward StataMP (most common on modern licenses)
            # but fall back through SE and plain Stata. `open -a
            # <AppName>` lets LaunchServices find the app regardless
            # of where it's installed.
            for app_name in ("StataMP", "StataSE", "StataNow", "Stata"):
                cmd = ["/usr/bin/open", "-a", app_name, str(p)]
                try:
                    r = subprocess.run(cmd, capture_output=True, timeout=5)
                    if r.returncode == 0:
                        return {"ok": True, "app": app_name}
                except (OSError, subprocess.TimeoutExpired):
                    continue
            # None worked — fall through to the default handler.
            cmd = ["/usr/bin/open", str(p)]
        elif mode == "run_r":
            # Prefer RStudio; fall back to R.app; fall back to
            # default handler.
            for app_name in ("RStudio", "R"):
                cmd2 = ["/usr/bin/open", "-a", app_name, str(p)]
                try:
                    r = subprocess.run(cmd2, capture_output=True, timeout=5)
                    if r.returncode == 0:
                        return {"ok": True, "app": app_name}
                except (OSError, subprocess.TimeoutExpired):
                    continue
            cmd = ["/usr/bin/open", str(p)]
        try:
            subprocess.run(cmd, check=False, timeout=5)
        except (OSError, subprocess.TimeoutExpired) as e:
            return {"ok": False, "reason": f"open failed: {e}"}
        return {"ok": True}

    def set_dataset_policy(
        self, name: str, depth: str
    ) -> dict[str, Any]:
        """Update the schema-depth ceiling for one dataset and persist
        to ``.builder/policy.json``. Returns the refreshed policy
        summary so the UI can re-render.

        No-ops cleanly if ``cwd`` isn't set or the depth isn't one of
        the valid tiers — a malformed JS caller shouldn't be able to
        corrupt the policy file.
        """
        if self.cwd is None:
            return {"ok": False, "reason": "session not started"}
        if depth not in VALID_DEPTHS:
            return {"ok": False, "reason": f"invalid depth: {depth!r}"}
        if not isinstance(name, str) or not name:
            return {"ok": False, "reason": "empty dataset name"}

        current = load_policy(self.cwd)
        # When the researcher selects the same depth as the app-wide
        # default, drop any explicit entry for this dataset rather
        # than saving "explicit at the default value." Result: the
        # dataset's `explicit` flag goes back to False, matching the
        # researcher's mental model that "I chose the default" is the
        # same state as "I never changed it." Without this, a round
        # trip (change away, change back) left the entry stuck at
        # `explicit=True`.
        updated_datasets = dict(current.datasets)
        if depth == current.default_max_depth:
            updated_datasets.pop(name, None)
        else:
            updated_datasets[name] = DatasetPolicy(
                max_depth=depth,
                set_at=datetime.now(timezone.utc).isoformat(),
            )
        updated = BuilderPolicy(
            version=current.version,
            default_max_depth=current.default_max_depth,
            datasets=updated_datasets,
        )
        try:
            save_policy(self.cwd, updated)
        except OSError as e:
            return {"ok": False, "reason": f"save failed: {e}"}
        return {"ok": True, "policy": self._policy_summary()}

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
        # Record the researcher's message so replay reconstructs the
        # whole exchange. The JS side already rendered the bubble on
        # its own (appendUser) before calling us, so we only persist
        # here rather than re-pushing to the UI.
        self._persist_event({"type": "user_message", "text": text})
        asyncio.run_coroutine_threadsafe(
            self._run_turn(text), self._loop
        )

    def send_message_with_images(
        self, text: str, images: list[dict[str, Any]]
    ) -> None:
        """Send a user message with one or more attached images.
        ``images[i] = {"data": <base64>, "mime": <image/png|jpeg|webp|gif>}``.
        The SDK's query() takes a string shortcut OR an async
        iterable of message dicts; for vision we need the latter so
        we can attach image content blocks alongside the text.
        """
        if self._loop is None:
            self._push_event({"type": "turn_error", "message": "worker loop not running"})
            return
        if self.cwd is None:
            self._push_event({
                "type": "turn_error",
                "message": "no working directory set — choose files or a folder first",
            })
            return
        # Persist as a user_message text-only for the chat log (we
        # don't store image bytes in the JSONL — the storage cost
        # would dwarf the analytical log and replay doesn't need
        # them). Flag that images were attached so future session
        # browsers can show a marker.
        self._persist_event({
            "type": "user_message",
            "text": text,
            "attachments": len(images),
        })
        asyncio.run_coroutine_threadsafe(
            self._run_turn(text, images=images), self._loop
        )

    def add_files(self) -> dict[str, Any]:
        """Open a native file picker that accepts both data files
        (.csv/.dta/.rds) and images (.png/.jpg/.webp/.gif), and route
        each selected file according to its extension:

          Data files → copied into the session's working directory
                       so Claude can reference them through
                       ``get_schema`` / ``submit_script``.
          Images    → read, base64-encoded, and returned so the
                       frontend can stage them as attachments on
                       the next outgoing message (vision).

        Returns ``{ok, added: [data filenames], images: [{data,
        mime, name}], policy, session_title}``.
        """
        if self._window is None:
            return {"ok": False, "reason": "window not ready"}
        if self.cwd is None:
            return {"ok": False, "reason": "no active session — start one first"}
        try:
            import webview
            # pywebview validates filter strings with a regex that only
            # allows [\w\s] before the parens — the old "Data + images"
            # label tripped that check because of the `+`. Keep the
            # description to plain words.
            file_types = (
                "Everything Builder handles (*.csv;*.dta;*.rds;*.do;*.r;*.gph;*.log;*.smcl;*.rmd;*.png;*.jpg;*.jpeg;*.webp;*.gif)",
                "Data files (*.csv;*.dta;*.rds)",
                "Scripts and logs (*.do;*.r;*.gph;*.log;*.smcl;*.rmd)",
                "Images (*.png;*.jpg;*.jpeg;*.webp;*.gif)",
                "All files (*.*)",
            )
            result = self._window.create_file_dialog(
                webview.OPEN_DIALOG,
                allow_multiple=True,
                file_types=file_types,
            )
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "reason": f"dialog error: {e}"}
        if not result:
            return {"ok": False, "reason": "cancelled"}

        import base64
        # Anything in this set is copied into the session cwd so
        # Claude can reference it through get_schema / submit_script,
        # or so the researcher can open it alongside the chat. Data
        # files, R and Stata scripts, Stata graphs, log output, and
        # R Markdown all qualify.
        _COPY_EXTS = {
            ".csv", ".dta", ".rds",              # data
            ".do",                                # Stata script
            ".r",                                 # R script
            ".gph",                               # Stata graph
            ".log", ".smcl",                      # Stata / R logs
            ".rmd",                               # R Markdown
        }
        _IMAGE_EXTS_MIMES = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".gif": "image/gif",
        }
        _IMAGE_MAX_BYTES = 5 * 1024 * 1024

        added: list[str] = []
        images: list[dict[str, str]] = []
        skipped: list[str] = []

        for s in result:
            try:
                src = Path(s).expanduser().resolve()
            except OSError as e:
                return {"ok": False, "reason": f"bad path: {e}"}
            if not src.is_file():
                return {"ok": False, "reason": f"not a file: {src}"}
            ext = src.suffix.lower()
            if ext in _COPY_EXTS:
                try:
                    shutil.copy2(src, self.cwd / src.name)
                    added.append(src.name)
                except OSError as e:
                    return {"ok": False, "reason": f"copy failed: {e}"}
            elif ext in _IMAGE_EXTS_MIMES:
                try:
                    raw = src.read_bytes()
                except OSError as e:
                    return {"ok": False, "reason": f"image read failed: {e}"}
                if len(raw) > _IMAGE_MAX_BYTES:
                    skipped.append(f"{src.name} (>5 MB)")
                    continue
                images.append({
                    "data": base64.b64encode(raw).decode("ascii"),
                    "mime": _IMAGE_EXTS_MIMES[ext],
                    "name": src.name,
                })
            else:
                skipped.append(src.name)

        return {
            "ok": True,
            "added": added,
            "images": images,
            "skipped": skipped,
            "policy": self._policy_summary(),
            "session_title": _session_title(self.cwd),
        }

    def add_files_from_blobs(
        self, files: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Twin of :meth:`add_files`, but for files dropped or pasted
        directly onto the composer from JS — no native dialog involved.

        Each ``files[i]`` is ``{name, content (base64), mime?}``. Data
        and script files are copied into ``self.cwd``; images are
        decoded and returned so the frontend can stage them as vision
        attachments. Returns the same shape as :meth:`add_files`.
        """
        if self.cwd is None:
            return {
                "ok": False,
                "reason": "no active session — start one first",
            }
        if not files:
            return {"ok": False, "reason": "no files"}

        import base64
        _COPY_EXTS = {
            ".csv", ".dta", ".rds",
            ".do", ".r",
            ".gph",
            ".log", ".smcl",
            ".rmd",
        }
        _IMAGE_EXTS_MIMES = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".gif": "image/gif",
        }
        _IMAGE_MAX_BYTES = 5 * 1024 * 1024

        added: list[str] = []
        images: list[dict[str, str]] = []
        skipped: list[str] = []

        for item in files:
            name = item.get("name", "")
            content_b64 = item.get("content", "")
            if not name or not isinstance(content_b64, str):
                continue
            if "," in content_b64:
                content_b64 = content_b64.split(",", 1)[1]
            try:
                blob = base64.b64decode(content_b64, validate=False)
            except Exception:  # noqa: BLE001
                return {"ok": False, "reason": f"could not decode {name!r}"}
            safe_name = Path(name).name
            ext = Path(safe_name).suffix.lower()
            if ext in _COPY_EXTS:
                try:
                    (self.cwd / safe_name).write_bytes(blob)
                    added.append(safe_name)
                except OSError as e:
                    return {"ok": False, "reason": f"copy failed: {e}"}
            elif ext in _IMAGE_EXTS_MIMES:
                if len(blob) > _IMAGE_MAX_BYTES:
                    skipped.append(f"{safe_name} (>5 MB)")
                    continue
                images.append({
                    "data": base64.b64encode(blob).decode("ascii"),
                    "mime": _IMAGE_EXTS_MIMES[ext],
                    "name": safe_name,
                })
            else:
                skipped.append(safe_name)

        return {
            "ok": True,
            "added": added,
            "images": images,
            "skipped": skipped,
            "policy": self._policy_summary(),
            "session_title": _session_title(self.cwd),
        }

    def list_models(self) -> dict[str, Any]:
        """Return the list of selectable Claude models and which one
        is currently active. The JS side renders a popup from this
        so the frontend doesn't need to hard-code the model catalog
        separately from the backend."""
        from builder.app import SUPPORTED_MODELS
        return {
            "ok": True,
            "current": self._model,
            "models": [
                {
                    "id": mid,
                    "label": info["label"],
                    "context_window": info["context_window"],
                }
                for mid, info in SUPPORTED_MODELS.items()
            ],
        }

    def set_model(self, model_id: str) -> dict[str, Any]:
        """Switch the active Claude model. Uses the SDK client's
        in-place ``set_model`` so the running conversation is
        preserved; Claude picks the new model up on the next turn
        with full memory of what was said before. If no client is
        open yet (model swapped before the first message), we just
        stash the choice and the client opens fresh with the new
        model on first send_message.

        Mid-turn switches are refused so we don't race the streaming
        response.
        """
        from builder.app import SUPPORTED_MODELS
        if model_id not in SUPPORTED_MODELS:
            return {"ok": False, "reason": f"unknown model: {model_id}"}
        if self._current_turn_task is not None and not self._current_turn_task.done():
            return {
                "ok": False,
                "reason": "a turn is in flight; wait for it to finish",
            }
        if model_id == self._model:
            return {"ok": True, "model": model_id, "unchanged": True}

        self._model = model_id
        # Swap the model on the existing client in place. Previous
        # version closed+reopened, which lost the conversation; the
        # SDK exposes set_model for exactly this case.
        if self._loop is not None and self._client is not None:
            async def _swap() -> None:
                await self._client.set_model(model_id)
            fut = asyncio.run_coroutine_threadsafe(_swap(), self._loop)
            try:
                fut.result(timeout=5)
            except Exception as e:  # noqa: BLE001
                # If the SDK rejects the model id (e.g., unrecognized
                # `[1m]` suffix) fall back to the teardown path so
                # at worst the researcher gets a fresh conversation
                # rather than a broken client.
                try:
                    await_close = asyncio.run_coroutine_threadsafe(
                        self._client.__aexit__(None, None, None), self._loop
                    )
                    await_close.result(timeout=3)
                except Exception:  # noqa: BLE001
                    pass
                self._client = None
                return {
                    "ok": False,
                    "reason": f"model switch failed: {e}. Conversation reset.",
                }

        info = SUPPORTED_MODELS[model_id]
        return {
            "ok": True,
            "model": model_id,
            "label": info["label"],
            "context_window": info["context_window"],
        }

    def list_sessions(self) -> dict[str, Any]:
        """Return a newest-first list of past Builder sessions living
        under ``~/.builder-sessions/``. Each entry carries the
        absolute path, the directory name, a human-friendly timestamp,
        the names of the data files inside, and the on-disk size in
        bytes so the sidebar can show what's heavy. Also flags the
        session that's currently loaded.
        """
        current = str(self.cwd.resolve()) if self.cwd else None
        entries: list[dict[str, Any]] = []
        if not SESSIONS_ROOT.exists():
            return {"ok": True, "sessions": entries, "current": current}

        _DATA_EXTS = (".csv", ".dta", ".rds")
        for child in SESSIONS_ROOT.iterdir():
            if not child.is_dir():
                continue
            try:
                stat = child.stat()
            except OSError:
                continue
            # Timestamp parsing — dir names look like
            # `20260422T160059Z_f13630f4`. Fall back to mtime if the
            # prefix doesn't match (user manually renamed, etc.).
            ts = _parse_session_timestamp(child.name) or stat.st_mtime
            datasets: list[str] = []
            try:
                for f in child.iterdir():
                    if f.is_file() and f.suffix.lower() in _DATA_EXTS:
                        datasets.append(f.name)
            except OSError:
                pass
            datasets.sort()
            entries.append({
                "path": str(child.resolve()),
                "name": child.name,
                "timestamp": ts,  # epoch seconds, JS formats
                "datasets": datasets,
                "size": _dir_size(child),
            })
        entries.sort(key=lambda e: e["timestamp"], reverse=True)
        return {"ok": True, "sessions": entries, "current": current}

    def delete_session(self, path: str) -> dict[str, Any]:
        """Delete a session directory and everything under it: data
        copies, run dirs, results.db, chat_history.jsonl. Refuses to
        delete the currently-active session (which would leave the
        backend pointing at a vanished cwd). Only paths inside
        ``~/.builder-sessions/`` are allowed.
        """
        if not path:
            return {"ok": False, "reason": "empty path"}
        try:
            target = Path(path).expanduser().resolve()
        except OSError as e:
            return {"ok": False, "reason": f"bad path: {e}"}
        if not _is_within(target, SESSIONS_ROOT.resolve()):
            return {
                "ok": False,
                "reason": "path is outside ~/.builder-sessions/",
            }
        if not target.exists():
            return {"ok": False, "reason": "already gone"}
        if self.cwd and target == self.cwd.resolve():
            return {
                "ok": False,
                "reason": "cannot delete the active session — switch first",
            }
        try:
            shutil.rmtree(target)
        except OSError as e:
            return {"ok": False, "reason": f"delete failed: {e}"}
        return {"ok": True, "path": str(target)}

    def switch_session(self, path: str) -> dict[str, Any]:
        """Switch the active working directory to an existing Builder
        session. Closes the current SDK client so the next turn
        starts a fresh conversation against the new cwd. Chat history
        from the previous session isn't replayed (persistence is a
        future feature); the researcher sees a clean slate pointed at
        the chosen session's data.
        """
        if not path:
            return {"ok": False, "reason": "empty path"}
        try:
            target = Path(path).expanduser().resolve()
        except OSError as e:
            return {"ok": False, "reason": f"bad path: {e}"}
        if not target.is_dir():
            return {"ok": False, "reason": f"not a directory: {target}"}
        # Only allow switching into paths we manage — prevents a
        # page-side exploit from pointing cwd at an arbitrary folder.
        if not _is_within(target, SESSIONS_ROOT.resolve()):
            return {
                "ok": False,
                "reason": "path is outside ~/.builder-sessions/",
            }

        # Close the existing client so the new cwd gets a fresh SDK
        # session instead of leaking state across directories.
        if self._loop is not None and self._client is not None:
            async def _close() -> None:
                try:
                    await self._client.__aexit__(None, None, None)
                except Exception:  # noqa: BLE001
                    pass
                finally:
                    self._client = None
            fut = asyncio.run_coroutine_threadsafe(_close(), self._loop)
            try:
                fut.result(timeout=3)
            except Exception:  # noqa: BLE001
                pass

        return self._set_cwd(target)

    def interrupt_turn(self) -> dict[str, Any]:
        """Cancel the currently-running turn. Called when the
        researcher clicks the Stop button. Cancellation propagates
        through the asyncio Task running in ``_run_turn``: the
        ``async for`` loop over ``chat_service.run_turn`` raises
        CancelledError, we surface that as a ``turn_error`` event
        with a clear message, and the Send button re-enables on the
        JS side via its normal terminal-event handling.

        The SDK's socket/subprocess may still be mid-write to the
        upstream service when we cancel; we close the persistent
        client so the next ``send_message`` opens a fresh one and
        server-side state doesn't leak across turns.
        """
        if self._loop is None:
            return {"ok": False, "reason": "worker loop not running"}
        task = self._current_turn_task
        if task is None or task.done():
            return {"ok": False, "reason": "no turn in flight"}
        # Cancel on the worker loop; safe from this thread.
        self._loop.call_soon_threadsafe(task.cancel)
        # Also tear down the client so any half-finished request
        # doesn't leak into the next turn. Reopens lazily.
        async def _close_client() -> None:
            if self._client is not None:
                try:
                    await self._client.__aexit__(None, None, None)
                except Exception:  # noqa: BLE001
                    pass
                finally:
                    self._client = None

        asyncio.run_coroutine_threadsafe(_close_client(), self._loop)
        return {"ok": True}

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
        UI to switch from landing to chat view. ``session_title`` is
        a human-friendly label for the topbar: the primary dataset
        name if there's one, otherwise a short timestamp from the
        dir name. Always something a researcher would recognize."""
        assert self.cwd is not None
        return {
            "type": "ready",
            "cwd": str(self.cwd),
            "session_title": _session_title(self.cwd),
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
            # Ask the SDK to resume the cwd's prior conversation when
            # we already have persisted history for this session.
            # First-ever open of a session has no history, so Claude
            # starts fresh; switching into an old session or landing
            # in one that already hosted a chat keeps the memory.
            history_path = self.cwd / ".builder" / "chat_history.jsonl"
            resume = history_path.exists() and history_path.stat().st_size > 0
            opts = _build_options(
                self.cwd,
                model=self._model,
                continue_conversation=resume,
            )
            self._client = await ClaudeSDKClient(options=opts).__aenter__()
        return self._client

    async def _run_turn(
        self, text: str, images: list[dict[str, Any]] | None = None
    ) -> None:
        assert self._send_lock is not None
        # Register this task so interrupt_turn() can cancel it. Using
        # current_task() (set by the event loop) rather than passing
        # the Task in from the scheduler — scheduler yields a
        # concurrent.futures.Future wrapper, which isn't cancellable
        # in the asyncio sense.
        self._current_turn_task = asyncio.current_task()
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
                async for evt in chat_service.run_turn(
                    client, text, images=images
                ):
                    self._push_event(_event_to_dict(evt))
            except asyncio.CancelledError:
                # Researcher hit Stop. Surface a terminal event so
                # the UI re-enables the composer via its standard
                # event handler. Don't re-raise — cancellation is
                # expected here, not an error condition.
                self._push_event({
                    "type": "turn_error",
                    "message": "cancelled",
                })
                return
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
            finally:
                self._current_turn_task = None

    def _push_event(self, payload: dict[str, Any]) -> None:
        """Send a JSON event to the web UI. pywebview's evaluate_js
        takes a string of JS to run in the page's context."""
        # Persist the event to the session's chat history before
        # firing it at the UI. Only transcript-forming events get
        # recorded; transient status (turn_done / auth_failure /
        # ready / policy_updated) would clutter the log without
        # helping a future re-open.
        self._persist_event(payload)
        if self._window is None:
            return
        js = f"window.builder_event({json.dumps(payload)});"
        try:
            self._window.evaluate_js(js)
        except Exception:  # noqa: BLE001 — webview may be closing
            pass

    # Event types we keep in the chat log. Everything else is either
    # transient (turn_done, auth_failure) or reconstructible from
    # session state (ready, policy_updated).
    _PERSIST_TYPES = frozenset({
        "assistant_text",
        "assistant_thinking",
        "tool_call",
        "tool_result",
        "user_message",
    })

    def _persist_event(self, payload: dict[str, Any]) -> None:
        if self.cwd is None:
            return
        etype = payload.get("type")
        if etype not in self._PERSIST_TYPES:
            return
        try:
            history_dir = self.cwd / ".builder"
            history_dir.mkdir(parents=True, exist_ok=True)
            path = history_dir / "chat_history.jsonl"
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except OSError:
            # Persistence failing shouldn't block live chat. The
            # transcript on screen stays intact; replay just won't
            # include this event.
            pass

    def get_chat_history(self) -> dict[str, Any]:
        """Return the persisted chat log for the active session so
        the UI can replay past messages after a session switch.
        Empty list if the session has no history yet."""
        if self.cwd is None:
            return {"ok": True, "events": []}
        path = self.cwd / ".builder" / "chat_history.jsonl"
        events: list[dict[str, Any]] = []
        if not path.exists():
            return {"ok": True, "events": events}
        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except OSError as e:
            return {"ok": False, "reason": str(e)}
        return {"ok": True, "events": events}

    def _policy_summary(self) -> dict[str, Any]:
        """Compact JSON-serializable summary of the current policy +
        dataset list for the topbar footer."""
        if self.cwd is None:
            from builder.policy import DEFAULT_MAX_DEPTH
            return {"default_max_depth": DEFAULT_MAX_DEPTH, "datasets": []}
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


def _is_within(child: Path, parent: Path) -> bool:
    """Path-safe "is this inside that" check. Uses resolved paths to
    follow symlinks and normalize `..`, so a symlink escape can't
    sneak an outside path past the allowlist."""
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except (ValueError, OSError):
        return False


def _dir_size(path: Path) -> int:
    """Walk a directory and sum file sizes. Returns 0 on any I/O
    error rather than raising — the sidebar display is cosmetic, so
    a permission hiccup on one run dir shouldn't break the list.
    Uses os.scandir for speed (cached stat) and skips symlinks to
    avoid walking into arbitrary locations."""
    total = 0
    stack = [path]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as it:
                for entry in it:
                    try:
                        if entry.is_symlink():
                            continue
                        if entry.is_file(follow_symlinks=False):
                            total += entry.stat(follow_symlinks=False).st_size
                        elif entry.is_dir(follow_symlinks=False):
                            stack.append(Path(entry.path))
                    except OSError:
                        continue
        except OSError:
            continue
    return total


def _session_title(cwd: Path) -> str:
    """Human-friendly title for a session. Preference order:
      1. A single dataset's filename (most common case: one upload).
      2. "<first> +N more" when multiple datasets live in the dir.
      3. A "Session MMM DD, HH:MM" stamp derived from the dir name.
      4. The dir's basename as a last-resort fallback.
    The goal is that the topbar always shows something a researcher
    recognizes, never a raw absolute path.
    """
    _DATA_EXTS = (".csv", ".dta", ".rds")
    try:
        datasets = sorted(
            p.name for p in cwd.iterdir()
            if p.is_file() and p.suffix.lower() in _DATA_EXTS
        )
    except OSError:
        datasets = []
    if len(datasets) == 1:
        return datasets[0]
    if len(datasets) > 1:
        return f"{datasets[0]} +{len(datasets) - 1} more"

    ts = _parse_session_timestamp(cwd.name)
    if ts is not None:
        from datetime import datetime
        return "Session " + datetime.fromtimestamp(ts).strftime("%b %d, %H:%M")
    return cwd.name


def _parse_session_timestamp(name: str) -> float | None:
    """Parse the ``YYYYMMDDThhmmssZ_<id>`` prefix of a session dir
    name and return it as epoch seconds. Returns None if the name
    doesn't match (e.g. manually renamed dirs); callers fall back to
    mtime."""
    from datetime import datetime, timezone
    import re
    m = re.match(r"^(\d{8}T\d{6}Z)", name)
    if not m:
        return None
    try:
        dt = datetime.strptime(m.group(1), "%Y%m%dT%H%M%SZ")
        return dt.replace(tzinfo=timezone.utc).timestamp()
    except ValueError:
        return None


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
            "language": evt.language,
            "raw_stdout": raw_stdout,
            "raw_stderr": raw_stderr,
        }
    if isinstance(evt, chat_service.TurnDone):
        return {
            "type": "turn_done",
            "input_tokens": evt.input_tokens,
            "output_tokens": evt.output_tokens,
            "cache_read_input_tokens": evt.cache_read_input_tokens,
            "cache_creation_input_tokens": evt.cache_creation_input_tokens,
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
        # to True while iterating on the web UI — WKWebView remembers
        # the inspector-open state across launches, so leaving it on
        # means the Web Inspector pops up every restart.
        webview.start(debug=False)
    finally:
        bridge.stop_loop()


if __name__ == "__main__":
    main()
