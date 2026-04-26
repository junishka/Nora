"""Nora — web UI entry point.

Opens a native WKWebView window (via pywebview) hosting a local HTML
chat interface, and bridges it to the same backend plumbing the
terminal UI uses: a ``ProviderSession`` (Anthropic today, OpenAI also
supported) + the 6 MCP tools + sanitizer + policy + sandboxed
executor.

Launched via:

    uv run python -m nora.ui [cwd]

or the ``nora-ui`` console script. Terminal UI (``nora``) is
unchanged and still works. The two share everything except
rendering.

Session model (new in this commit):

- With a ``cwd`` argument, behave like before: open straight into
  the chat view against that directory.
- Without one, show a landing screen: drop files in, or click
  "Choose files" (native file picker) or "Choose folder". Files
  are staged into ``~/.nora-sessions/<timestamp>_<id>/`` —
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

from nora import chat_service
from nora.chat_history import Turn, read_turns
from nora.config import set_cwd
from nora.env_detect import detect_environment
from nora.policy import (
    VALID_DEPTHS,
    NoraPolicy,
    DatasetPolicy,
    get_max_depth,
    has_explicit_policy,
    load_policy,
    save_policy,
)
from nora.provider import (
    AuthFailure,
    ProviderSession,
    TurnDone,
    TurnError,
    open_session,
    provider_for_model,
)
from nora.provider.catalog import (
    ALL_MODELS,
    PROVIDER_DEFAULTS,
    PROVIDER_PRICING_URLS,
)
from nora.system_prompt import build_system_prompt
from nora.tools import SERVER_NAME


# Where uploaded-file sessions live. Chosen for three properties:
# 1. No spaces — Stata's batch-mode parser trips on them.
# 2. Per-user and persistent — researchers can come back to a
#    past session and look at its `.nora/results.db`.
# 3. Outside any Dropbox / iCloud path — the sandbox scope is
#    exactly the files that were uploaded, not whatever else the
#    researcher happened to have in the source directory.
SESSIONS_ROOT = Path.home() / ".nora-sessions"


# ---------------------------------------------------------------------------
# The bridge between the web UI and the Python backend
# ---------------------------------------------------------------------------

class NoraBridge:
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
        # Persistent provider session. Opened lazily on first
        # send_message; closed on shutdown / cwd switch / model swap
        # rejection. ProviderSession is a Protocol; the concrete
        # instance is whatever ``open_session(self._provider, ...)``
        # returns (AnthropicSession today, OpenAISession when the
        # researcher's auth chooses OpenAI).
        self._session: ProviderSession | None = None
        # A lock to serialize send_message calls. The session client
        # assumes one turn at a time.
        self._send_lock: asyncio.Lock | None = None
        # Handle to the currently-running turn's asyncio Task, so
        # `interrupt_turn` can cancel it when the researcher clicks
        # the Stop button. None when no turn is in flight. Captured
        # on the worker loop, cleared when the turn returns.
        self._current_turn_task: asyncio.Task[None] | None = None
        # Which provider + model the researcher has selected. Default
        # to Anthropic Sonnet 4.6 — the auth screen can override
        # before the first session opens. Changed via ``set_model``
        # from the composer chip; takes effect on the next turn
        # because ``_ensure_session`` reads it when opening a fresh
        # session.
        self._provider: str = "anthropic"
        self._model: str = PROVIDER_DEFAULTS[self._provider]
        # Memory: whenever we open a fresh SDK client (first turn,
        # model switch, session switch, app restart) we prepend the
        # last N turns from chat_history.jsonl to the first user
        # message so Claude picks up where we left off. The flag is
        # set when the client is (re)opened and cleared after the
        # prefix has been emitted exactly once.
        self._needs_context_prefix: bool = False
        # Mid-chat script attachments. When the researcher drops a
        # ``.py`` / ``.do`` / ``.r`` / ``.rmd`` file into the
        # composer, the bridge copies it into the session cwd AND
        # stages its content here so the next ``send_message`` can
        # prepend the source as a context block — same shape as
        # image attachments, just textual. Cleared on successful
        # send; restored on cancel/error so the user doesn't lose
        # the attachment to a transient failure.
        self._pending_script_attachments: list[dict[str, Any]] = []
        # Datasets the model has been told about so far. Snapshotted
        # at session open from the system-prompt's ``datasets_list``
        # block; mid-chat uploads compare against this set and any
        # new names get prepended to the next turn as a "researcher
        # just added X" notice. Without this the system prompt's
        # listing is frozen at session open and a parquet dropped
        # ten messages in is invisible to the model until the
        # session is reopened.
        self._known_datasets: frozenset[str] = frozenset()
        # When the launcher hands us a cwd up-front (``nora-ui /path``)
        # we never go through ``_set_cwd``, so the per-session model
        # restore there gets skipped. Run it here too: a session with
        # ``active_model = "claude-opus-4-7[1m]"`` recorded in its
        # state file should reopen on Opus, not on whatever the
        # Anthropic default happens to be.
        if cwd is not None:
            self._restore_session_model_preference()

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
            target=_run, name="nora-ui-loop", daemon=True
        )
        self._loop_thread.start()

    def stop_loop(self) -> None:
        if self._loop is None:
            return
        # Close the provider session if we opened one.
        async def _close() -> None:
            if self._session is not None:
                try:
                    await self._session.close()
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
        payload describing which screen to show:

        - ``"needs_auth"`` — no provider has a usable credential yet.
          Show the auth screen first (researcher must enter at least
          one API key or sign in via the Claude CLI subscription path).
        - ``"needs_session"`` — auth is good, but no working directory
          is selected. Show the drop / choose-files landing.
        - ``"ready"`` — auth + cwd both done, jump straight to chat.

        The auth check runs on every ``ui_ready`` so a researcher who
        deletes a credential in Keychain Access between sessions
        bounces back to the auth screen on next launch.
        """
        # Make sure the active provider matches what's actually
        # authed. The bridge defaults to Anthropic at construction;
        # without this guard, a researcher who configures only
        # OpenAI would hit the chat with ``self._provider ==
        # "anthropic"`` and ``_ensure_session`` would fail at first
        # turn with "no Anthropic credential."
        self._reconcile_active_provider_with_auth()
        status = self._auth_status_payload()
        if not status["any_authed"]:
            return {"state": "needs_auth", "auth": status}
        if self.cwd is None:
            return {"state": "needs_session", "auth": status}
        # Frontend's startup branch checks ``state === 'ready'`` to
        # decide whether to land on chat or the file picker. The
        # ready_payload carries ``type: 'ready'`` (an event-shape
        # holdover) but not ``state`` — without this explicit add,
        # ``nora-ui <cwd>`` and the auth-Continue path both fall
        # through to ``showLanding`` even though the bridge knows
        # the session is ready to chat.
        return {"state": "ready", **self._ready_payload(), "auth": status}

    def _reconcile_active_provider_with_auth(self) -> None:
        """Ensure ``self._provider`` is one the researcher can
        actually use right now.

        Default at construction is Anthropic. If the researcher
        configures only OpenAI, the bridge needs to flip to
        OpenAI before the first turn opens a session — otherwise
        ``_ensure_session("anthropic", ...)`` runs with no
        credential and the model picker chip lies about what's
        active.

        Picks deterministically from ``PROVIDER_DEFAULTS`` so the
        ordering is stable across calls. Closes any open session
        when the active provider changes (the model is a different
        family, conversation continuity isn't preserved across
        provider swaps).
        """
        authed = self._authed_providers()
        if not authed or self._provider in authed:
            return
        # First authed provider in catalog order — Anthropic when
        # both are present, OpenAI only when it's the only choice.
        for candidate in PROVIDER_DEFAULTS:
            if candidate in authed:
                self._close_session_blocking()
                self._provider = candidate
                self._model = PROVIDER_DEFAULTS[candidate]
                # Persist the swap so a session reload doesn't
                # bounce the researcher back to the wrong default.
                self._persist_active_model()
                return

    def choose_files(self) -> dict[str, Any]:
        """Open a native file-picker dialog (multi-select) restricted
        to the data formats Nora understands, then stage the
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
                "Data files (*.csv;*.tsv;*.dta;*.rds;*.parquet;*.jsonl;*.ndjson)",
                "CSV (*.csv)",
                "TSV (*.tsv)",
                "Stata (*.dta)",
                "R (*.rds)",
                "Parquet (*.parquet)",
                "JSON Lines (*.jsonl;*.ndjson)",
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
          - ``"run_python"`` — launch the researcher's preferred
            Python editor (VS Code → PyCharm → Cursor → Sublime),
            falling back to the system default for ``.py``. Nora
            does NOT auto-execute the script in any case — the
            researcher reviews and runs it themselves.

        **Restricted to Nora-managed locations** — only paths
        inside the current session's working directory or inside
        the ``~/.nora-sessions/`` tree are allowed.
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
                    "path is outside Nora's managed directories — "
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
        elif mode == "run_python":
            # Researcher-friendly Python editors first; bias toward
            # VS Code since it's the most common modern install.
            # Falls back to the OS default ``.py`` handler — usually
            # IDLE or whatever the user wired up.
            for app_name in ("Visual Studio Code", "PyCharm", "PyCharm CE",
                             "Cursor", "Sublime Text", "Positron"):
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
        to ``.nora/policy.json``. Returns the refreshed policy
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
        updated = NoraPolicy(
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
        self._record_user_message(text)
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
        self._record_user_message(text, image_count=len(images))
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
                "Everything Nora handles (*.csv;*.tsv;*.dta;*.rds;*.parquet;*.jsonl;*.ndjson;*.do;*.r;*.py;*.ipynb;*.gph;*.log;*.smcl;*.rmd;*.png;*.jpg;*.jpeg;*.webp;*.gif)",
                "Data files (*.csv;*.tsv;*.dta;*.rds;*.parquet;*.jsonl;*.ndjson)",
                "Scripts and logs (*.do;*.r;*.py;*.ipynb;*.gph;*.log;*.smcl;*.rmd)",
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
        # files, R / Stata / Python scripts, Stata graphs, log output,
        # and R Markdown all qualify.
        from nora.schema import DATA_EXTENSIONS
        _COPY_EXTS = {
            *DATA_EXTENSIONS,                    # data — single source of truth
            ".do",                                # Stata script
            ".r",                                 # R script
            ".py",                                # Python script
            ".ipynb",                             # Jupyter notebook (referenced, not run)
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

        # Names that already exist in the session — surfaced separately
        # so the UI can prompt the researcher rather than silently
        # overwriting prior data (which would invalidate any stored
        # results computed against the previous file).
        skipped_existing: list[str] = []

        for s in result:
            try:
                src = Path(s).expanduser().resolve()
            except OSError as e:
                return {"ok": False, "reason": f"bad path: {e}"}
            if not src.is_file():
                return {"ok": False, "reason": f"not a file: {src}"}
            ext = src.suffix.lower()
            if ext in _COPY_EXTS:
                dst = self.cwd / src.name
                if dst.exists():
                    skipped_existing.append(src.name)
                    continue
                try:
                    shutil.copy2(src, dst)
                    added.append(src.name)
                except OSError as e:
                    return {"ok": False, "reason": f"copy failed: {e}"}
                # Script files (.py / .do / .r / .rmd) get their
                # contents staged for the next turn so the model
                # can SEE what was uploaded — without this the file
                # silently lands in cwd and the researcher's "what
                # does this do?" hits the model with no context.
                if ext in _INLINE_SCRIPT_EXTS:
                    try:
                        _stage_script_for_next_turn(
                            self._pending_script_attachments,
                            src.name, ext, dst.read_bytes(),
                        )
                    except OSError:
                        pass  # script is on disk; just no inline copy
            elif ext in _IMAGE_EXTS_MIMES:
                try:
                    raw = src.read_bytes()
                except OSError as e:
                    return {"ok": False, "reason": f"image read failed: {e}"}
                if len(raw) > _IMAGE_MAX_BYTES:
                    skipped.append(f"{src.name} (>5 MB)")
                    continue
                # Save the image to cwd alongside the vision staging.
                # Researchers expect "I uploaded this" to mean "the
                # file is in my session" — and with the image
                # on disk, they can re-open or reference it later
                # (e.g., embed in the next paper draft) without
                # going back to the original. Auto-rename on
                # collision so two ``chart.png`` drops don't
                # silently overwrite.
                img_dst = _disambiguate_target(self.cwd, src.name)
                try:
                    img_dst.write_bytes(raw)
                except OSError as e:
                    return {"ok": False, "reason": f"image save failed: {e}"}
                images.append({
                    "data": base64.b64encode(raw).decode("ascii"),
                    "mime": _IMAGE_EXTS_MIMES[ext],
                    "name": img_dst.name,
                })
            else:
                skipped.append(src.name)

        return {
            "ok": True,
            "added": added,
            "images": images,
            "skipped": skipped,
            "skipped_existing": skipped_existing,
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
        from nora.schema import DATA_EXTENSIONS
        _COPY_EXTS = {
            *DATA_EXTENSIONS,
            ".do", ".r", ".py", ".ipynb",
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
        # Drop names that already exist in the session — the UI prompts
        # the researcher rather than silently overwriting prior data
        # (see add_files() for the same guard on the native-dialog
        # path).
        skipped_existing: list[str] = []

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
                dst = self.cwd / safe_name
                if dst.exists():
                    skipped_existing.append(safe_name)
                    continue
                try:
                    dst.write_bytes(blob)
                    added.append(safe_name)
                except OSError as e:
                    return {"ok": False, "reason": f"copy failed: {e}"}
                # Script files: stage their contents alongside the
                # next user message so the model has the same
                # awareness as if the researcher had pasted them
                # inline. See ``add_files`` for the docstring on
                # which extensions qualify and why.
                if ext in _INLINE_SCRIPT_EXTS:
                    _stage_script_for_next_turn(
                        self._pending_script_attachments,
                        safe_name, ext, blob,
                    )
            elif ext in _IMAGE_EXTS_MIMES:
                if len(blob) > _IMAGE_MAX_BYTES:
                    skipped.append(f"{safe_name} (>5 MB)")
                    continue
                # Save to cwd alongside vision staging — see the
                # mirror code path in ``add_files`` for the
                # rationale (researchers expect "I uploaded this"
                # to mean the file is in their session, not just
                # that the model can see it once).
                img_dst = _disambiguate_target(self.cwd, safe_name)
                try:
                    img_dst.write_bytes(blob)
                except OSError as e:
                    return {"ok": False, "reason": f"image save failed: {e}"}
                images.append({
                    "data": base64.b64encode(blob).decode("ascii"),
                    "mime": _IMAGE_EXTS_MIMES[ext],
                    "name": img_dst.name,
                })
            else:
                skipped.append(safe_name)

        return {
            "ok": True,
            "added": added,
            "images": images,
            "skipped": skipped,
            "skipped_existing": skipped_existing,
            "policy": self._policy_summary(),
            "session_title": _session_title(self.cwd),
        }

    def list_models(self) -> dict[str, Any]:
        """Return the list of selectable models across every provider
        the researcher has authenticated, plus which one is currently
        active. The JS side renders a popup grouped by provider; the
        ``provider`` field on each row is what drives the grouping.

        Models for un-authed providers are still listed but flagged
        ``available=False`` so the picker can render them disabled
        with a "configure auth" hint rather than hiding them entirely
        — the researcher needs to know what could be there."""
        authed = self._authed_providers()
        return {
            "ok": True,
            "current": self._model,
            "current_provider": self._provider,
            "models": [
                {
                    "id": m.id,
                    "label": m.label,
                    "context_window": m.context_window,
                    "provider": m.provider,
                    "available": m.provider in authed,
                    # Pricing URL surfaced as a "view pricing" link on
                    # the row. JS routes the click through
                    # ``open_external`` which validates against an
                    # allowlist before handing to the system browser.
                    "pricing_url": PROVIDER_PRICING_URLS.get(m.provider),
                }
                for m in ALL_MODELS
            ],
        }

    def list_session_files(self) -> dict[str, Any]:
        """Return every researcher-uploaded file in the active session
        cwd, grouped by kind, for the topbar Files panel.

        ``policy.datasets`` carries data files only (the SDC layer
        cares about schemas; scripts and graphs aren't part of that
        story). The Files panel's job is broader: a researcher who
        dropped ``regression.py``, ``ols.do``, and ``data.csv`` wants
        to see all three at a glance — without this endpoint, only
        the CSV showed up and the count stayed wrong.

        Kinds:
          - ``data``   — .csv / .tsv / .dta / .rds / .parquet / .jsonl
          - ``script`` — .py / .do / .r / .rmd / .ipynb
          - ``graph``  — .gph
          - ``log``    — .log / .smcl

        Top-level scan only (matches every other Nora scanner). Files
        are sorted by name within their kind.
        """
        if self.cwd is None:
            return {"ok": True, "files": []}
        from nora.schema import DATA_EXTENSIONS

        # Mapping ext → (kind, sort priority). Priority orders the
        # kinds in the rendered popup: data first (the analysis
        # subjects), then scripts, then graphs, then logs.
        kind_for_ext: dict[str, tuple[str, int]] = {}
        for ext in DATA_EXTENSIONS:
            kind_for_ext[ext] = ("data", 0)
        for ext in (".py", ".do", ".r", ".rmd", ".ipynb"):
            kind_for_ext[ext] = ("script", 1)
        kind_for_ext[".gph"] = ("graph", 2)
        for ext in (".log", ".smcl"):
            kind_for_ext[ext] = ("log", 3)

        rows: list[dict[str, Any]] = []
        try:
            for child in self.cwd.iterdir():
                if not child.is_file():
                    continue
                ext = child.suffix.lower()
                kind_pri = kind_for_ext.get(ext)
                if kind_pri is None:
                    continue
                kind, priority = kind_pri
                try:
                    size = child.stat().st_size
                except OSError:
                    size = 0
                rows.append({
                    "name": child.name,
                    "kind": kind,
                    "priority": priority,
                    "size": size,
                    "ext": ext,
                })
        except OSError:
            return {"ok": True, "files": []}
        rows.sort(key=lambda r: (r["priority"], r["name"].lower()))
        return {"ok": True, "files": rows}

    def unstage_attachment(self, name: str) -> dict[str, Any]:
        """Remove a previously-staged script from
        ``_pending_script_attachments`` so it does NOT get prepended
        to the next message.

        Called when the researcher clicks the × on a composer chip.
        Without this, the JS chip would disappear but the bridge
        would still inline the script — the model sees content the
        researcher thought they unstaged. Real privacy mismatch
        (the file is on disk anyway, but "I removed this" should
        mean it.)

        Idempotent — unstaging a name that isn't staged is a no-op
        success, so a JS double-click can't 500. The on-disk copy
        is left alone (use ``delete_session_file`` if/when we add
        that — out of scope for this fix).
        """
        if not name:
            return {"ok": False, "reason": "no name"}
        target = Path(name).name  # basename only
        before = len(self._pending_script_attachments)
        self._pending_script_attachments = [
            a for a in self._pending_script_attachments
            if a.get("name") != target
        ]
        return {
            "ok": True,
            "name": target,
            "removed": before - len(self._pending_script_attachments),
        }

    def attach_session_file(self, name: str) -> dict[str, Any]:
        """Stage a file already in the session cwd as inline context
        for the next user message. Used by the Files panel to let
        the researcher click an uploaded script and "bring it to the
        chat" without having to re-drop it from Finder.

        Only script extensions (``.py`` / ``.do`` / ``.r`` / ``.rmd``)
        are accepted — data files reach the model via ``get_schema``
        and inlining a 5M-row CSV would just blow up the prompt.
        ``.gph`` and ``.log`` are also refused (not source code).

        Path safety: ``name`` is treated as a basename; any
        directory component is stripped before resolving against
        cwd. The resolved path must live inside cwd, otherwise the
        request is refused.
        """
        if self.cwd is None:
            return {"ok": False, "reason": "no active session"}
        if not name:
            return {"ok": False, "reason": "no file name"}
        # Basename-only — refuse traversal attempts even though the
        # JS side only sends filenames from list_session_files.
        safe_name = Path(name).name
        target = (self.cwd / safe_name).resolve()
        if not _is_within(target, self.cwd.resolve()):
            return {"ok": False, "reason": "file is outside the session"}
        if not target.is_file():
            return {"ok": False, "reason": f"not found: {safe_name}"}
        ext = target.suffix.lower()
        if ext not in _INLINE_SCRIPT_EXTS:
            return {
                "ok": False,
                "reason": (
                    f"only script files (.py/.do/.r/.rmd) can be "
                    f"attached this way; {safe_name} is a "
                    f"{ext or 'unknown'} file. Data files are "
                    f"already visible to the model via get_schema."
                ),
            }
        try:
            content = target.read_bytes()
        except OSError as e:
            return {"ok": False, "reason": f"read failed: {e}"}
        # Idempotent: if the same script is already staged for the
        # next turn, don't add a duplicate. Researchers who click
        # the same row twice expect "already attached" rather than
        # the model seeing two copies of the file.
        for staged in self._pending_script_attachments:
            if staged.get("name") == safe_name:
                return {
                    "ok": True,
                    "name": safe_name,
                    "already_attached": True,
                }
        _stage_script_for_next_turn(
            self._pending_script_attachments, safe_name, ext, content,
        )
        return {"ok": True, "name": safe_name}

    def open_external(self, url: str) -> dict[str, Any]:
        """Open ``url`` in the OS default browser, NOT inside the
        WKWebView. Used by the model picker's "view pricing" links so
        researchers don't lose their chat session navigating to
        anthropic.com or openai.com.

        Allowlist-gated — only URLs in ``PROVIDER_PRICING_URLS`` are
        accepted. The bridge is reachable from page-rendered JS, so a
        compromised page (e.g., a malicious tool result that escaped
        sanitisation and somehow injected JS) MUST NOT be able to
        coerce Nora into opening attacker-controlled URLs.
        """
        allowed = set(PROVIDER_PRICING_URLS.values())
        if url not in allowed:
            return {"ok": False, "reason": f"url not on allowlist: {url!r}"}
        try:
            import webbrowser
            webbrowser.open(url)
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "reason": f"open failed: {e}"}
        return {"ok": True}

    # -------- multi-provider auth surface --------

    def auth_status(self) -> dict[str, Any]:
        """Return the per-provider auth state used by the auth screen
        and any "you need to sign in" banners. The auth screen calls
        this on mount and after every save/delete to refresh badges."""
        return self._auth_status_payload()

    def save_credential(self, provider: str, api_key: str) -> dict[str, Any]:
        """Write an API key to the OS keyring under the named
        provider. Returns the post-write auth status so the UI can
        update badges without an extra round-trip."""
        from nora import auth as _auth
        res = _auth.set_credential(provider, api_key)
        if not res.get("ok"):
            return {**res, "auth": self._auth_status_payload()}
        # When the active provider can't actually be used (no
        # credential, no subscription), saving a key here is the
        # researcher's effective "use this provider" gesture —
        # promote it to active so the first turn doesn't fail. When
        # the active provider IS already authed, we leave it alone:
        # a researcher who configured Claude first and OpenAI second
        # shouldn't get silently switched off Claude.
        self._reconcile_active_provider_with_auth()
        return {**res, "auth": self._auth_status_payload()}

    def delete_credential(self, provider: str) -> dict[str, Any]:
        """Remove a stored credential. If the deleted provider was
        active, the next ``ui_ready`` may bounce back to the auth
        screen (depending on whether anything else is configured)."""
        from nora import auth as _auth
        res = _auth.delete_credential(provider)
        # Tear down any active session for the now-unauthed provider.
        if provider == self._provider:
            self._close_session_blocking()
        # Anthropic specifically: ``_ensure_anthropic_env`` copies the
        # keyring credential into ``ANTHROPIC_API_KEY`` so the SDK
        # picks it up. Without clearing that injected env var here,
        # ``detect_auth()`` keeps returning ``api_key`` and the auth
        # screen claims the provider is still configured until the
        # app restarts. ``clear_injected_env`` no-ops when the user's
        # own shell exported the variable.
        if provider == "anthropic":
            from nora.provider.anthropic import clear_injected_env
            clear_injected_env()
        return {**res, "auth": self._auth_status_payload()}

    def set_active_provider(self, provider: str) -> dict[str, Any]:
        """Switch the active provider explicitly (e.g., from the auth
        screen's "Use Anthropic" / "Use OpenAI" buttons). The model
        defaults to the provider's catalog default. The current
        session, if any, is closed so the next turn opens fresh
        against the new provider."""
        if provider not in PROVIDER_DEFAULTS:
            return {"ok": False, "reason": f"unknown provider: {provider!r}"}
        if self._current_turn_task is not None and not self._current_turn_task.done():
            return {
                "ok": False,
                "reason": "a turn is in flight; wait for it to finish",
            }
        from nora import auth as _auth
        from nora.provider import detect_auth as _detect
        # Block switching to a provider with no usable credential.
        if not (_auth.has_credential(provider) or _detect(provider) != "unknown"):
            return {
                "ok": False,
                "reason": (
                    f"no credential configured for {provider!r}; "
                    f"add an API key first"
                ),
            }
        if provider != self._provider:
            self._close_session_blocking()
            self._provider = provider
            self._model = PROVIDER_DEFAULTS[provider]
        return {
            "ok": True,
            "provider": self._provider,
            "model": self._model,
            "auth": self._auth_status_payload(),
        }

    # -------- internal helpers --------

    def _authed_providers(self) -> set[str]:
        """Set of provider ids the researcher can actually use right
        now (keyring-stored API key, ``ANTHROPIC_API_KEY`` env var, or
        Claude CLI subscription token)."""
        from nora.provider import detect_auth as _detect
        out: set[str] = set()
        for p in PROVIDER_DEFAULTS:
            if _detect(p) != "unknown":
                out.add(p)
        return out

    def _auth_status_payload(self) -> dict[str, Any]:
        """Build the dict the JS side reads to render the auth screen
        and gate the model picker. ``method`` distinguishes API-key
        configs from the Anthropic CLI subscription path so the UI
        can show the right "stored where?" copy."""
        from nora import auth as _auth
        from nora.provider import detect_auth as _detect

        providers: dict[str, dict[str, Any]] = {}
        for p in PROVIDER_DEFAULTS:
            mode = _detect(p)  # "subscription" | "api_key" | "unknown"
            providers[p] = {
                "configured": mode != "unknown",
                "method": mode,
                # ``has_keyring_entry`` lets the auth screen show
                # "Forget" only when there's actually something to
                # forget (subscription auth has nothing to forget here).
                "has_keyring_entry": _auth.has_credential(p),
            }
        return {
            "providers": providers,
            "any_authed": any(v["configured"] for v in providers.values()),
            "active_provider": self._provider,
        }

    def set_model(self, model_id: str) -> dict[str, Any]:
        """Switch the active model.

        Behaviour depends on whether the new model is from the same
        provider as the current session:

        - Same provider (Anthropic→Anthropic): delegate to
          ``session.set_model``, which uses the SDK's in-place swap
          when supported so the conversation is preserved.
        - Different provider (Anthropic→OpenAI or vice-versa): close
          the current session; the next ``send_message`` will open a
          fresh session for the new provider. Conversation context
          carries forward via the bridge's existing context-prefix
          injection on first turn.
        - No session yet: just stash the choice; first
          ``_ensure_session`` builds with the new selection.

        Mid-turn switches are refused so we don't race the streaming
        response.
        """
        try:
            new_provider = provider_for_model(model_id)
        except KeyError:
            return {"ok": False, "reason": f"unknown model: {model_id}"}
        if self._current_turn_task is not None and not self._current_turn_task.done():
            return {
                "ok": False,
                "reason": "a turn is in flight; wait for it to finish",
            }
        info = next((m for m in ALL_MODELS if m.id == model_id), None)
        if info is None:
            return {"ok": False, "reason": f"unknown model: {model_id}"}

        if model_id == self._model and new_provider == self._provider:
            return {"ok": True, "model": model_id, "unchanged": True}

        # Snapshot pre-switch state. Every failure path below restores
        # both fields so a rejected swap doesn't wedge the bridge with
        # an invalid id — without rollback, the next ``_ensure_session``
        # would reopen with the rejected id and fail again, while the
        # JS chip still showed the old name. Researcher sees a chat
        # that "stopped working" with no way back besides restart.
        old_model = self._model
        old_provider = self._provider

        # Provider change: drop the current session entirely. Both the
        # current session and the new (untested) model id are taken on
        # faith; if the next turn's session-open fails, the state has
        # already moved. That's still better than the no-rollback
        # baseline — the field assignments are atomic from the JS
        # caller's perspective and the failure surfaces on the next
        # turn rather than silently.
        if new_provider != self._provider:
            self._close_session_blocking()
            self._provider = new_provider
            self._model = model_id
            self._persist_active_model()
            return {
                "ok": True,
                "model": model_id,
                "label": info.label,
                "context_window": info.context_window,
                "provider": new_provider,
            }

        # Same provider, model swap. Delegate to the session if open.
        self._model = model_id
        if self._loop is not None and self._session is not None:
            session = self._session
            async def _swap() -> dict[str, Any]:
                return await session.set_model(model_id)
            fut = asyncio.run_coroutine_threadsafe(_swap(), self._loop)
            try:
                res = fut.result(timeout=5)
            except Exception as e:  # noqa: BLE001
                # Restore so the next turn doesn't reopen with a
                # rejected id. ``_close_session_blocking`` already
                # tore the session down, so the next ``_ensure_session``
                # builds fresh against the restored old model.
                self._model = old_model
                self._provider = old_provider
                self._close_session_blocking()
                return {
                    "ok": False,
                    "reason": f"model switch failed: {e}. Conversation reset.",
                }
            if not res.get("ok"):
                # Same restoration: session refused the swap (unknown
                # id at the SDK level, model not in the researcher's
                # plan, …). The session may already have been torn
                # down inside ``session.set_model``; clearing the
                # reference forces a fresh open against ``old_model``.
                self._model = old_model
                self._provider = old_provider
                if self._session is not None and self._loop is not None:
                    self._close_session_blocking()
                return res

        self._persist_active_model()
        return {
            "ok": True,
            "model": model_id,
            "label": info.label,
            "context_window": info.context_window,
            "provider": new_provider,
        }

    def _persist_active_model(self) -> None:
        """Refresh ``.nora/session_state.json`` so a successful
        ``set_model`` survives an app restart even before the
        researcher sends the first message in this session.

        Without this, the per-turn session-state writer would be the
        only path that records the new choice — meaning a researcher
        who swaps to Opus and then closes the app immediately would
        come back to Sonnet (the prior recorded value), not Opus.
        """
        if self.cwd is None:
            return
        try:
            from nora.session_state import write_session_state
            write_session_state(self.cwd, model=self._model)
        except Exception:  # noqa: BLE001 — never let state write break a swap
            pass

    def list_sessions(self) -> dict[str, Any]:
        """Return a newest-first list of past Nora sessions living
        under ``~/.nora-sessions/``. Each entry carries the
        absolute path, the directory name, a human-friendly timestamp,
        the names of the data files inside, and the on-disk size in
        bytes so the sidebar can show what's heavy. Also flags the
        session that's currently loaded.
        """
        current = str(self.cwd.resolve()) if self.cwd else None
        entries: list[dict[str, Any]] = []
        if not SESSIONS_ROOT.exists():
            return {"ok": True, "sessions": entries, "current": current}

        from nora.schema import DATA_EXTENSIONS as _DATA_EXTS
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
        ``~/.nora-sessions/`` are allowed.
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
                "reason": "path is outside ~/.nora-sessions/",
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
        """Switch the active working directory to an existing Nora
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
                "reason": "path is outside ~/.nora-sessions/",
            }

        # Close the existing session so the new cwd gets a fresh
        # provider session instead of leaking state across directories.
        self._close_session_blocking()

        return self._set_cwd(target)

    def interrupt_turn(self) -> dict[str, Any]:
        """Cancel the currently-running turn. Called when the
        researcher clicks the Stop button. Cancellation propagates
        through the asyncio Task running in ``_run_turn``: the
        ``async for`` loop over ``session.send()`` raises
        CancelledError, we surface that as a ``turn_error`` event
        with a clear message, and the Send button re-enables on the
        JS side via its normal terminal-event handling.

        The provider's socket/subprocess may still be mid-write to
        the upstream service when we cancel; we close the persistent
        session so the next ``send_message`` opens a fresh one and
        server-side state doesn't leak across turns.
        """
        if self._loop is None:
            return {"ok": False, "reason": "worker loop not running"}
        task = self._current_turn_task
        if task is None or task.done():
            return {"ok": False, "reason": "no turn in flight"}
        # Cancel on the worker loop; safe from this thread.
        self._loop.call_soon_threadsafe(task.cancel)
        # Also tear down the session so any half-finished request
        # doesn't leak into the next turn. Reopens lazily.
        session = self._session
        self._session = None

        async def _close_session() -> None:
            if session is not None:
                try:
                    await session.close()
                except Exception:  # noqa: BLE001
                    pass

        asyncio.run_coroutine_threadsafe(_close_session(), self._loop)
        return {"ok": True}

    # -------- internals --------

    def _set_cwd(self, path: Path) -> dict[str, Any]:
        """Finalize the working directory and return the ready
        payload. Builds up the config.get_cwd side-effect at the
        module level so other code (schema, policy, executor) picks
        it up via the singleton."""
        # Session switch safety: if cwd changes, tear down the active
        # provider session so the next turn starts a fresh conversation
        # AND drop the cached ResultStore for the old cwd so future
        # tool calls resolve the new session's DB. Earlier versions
        # kept a process-wide singleton store that stuck to whichever
        # cwd asked first — Project A could then see Project B's
        # stored sanitized results. See nora/store.py :: get_store.
        old_cwd = self.cwd.resolve() if self.cwd is not None else None
        new_cwd = path.resolve()
        if old_cwd is not None and old_cwd != new_cwd:
            self._close_session_blocking()
            try:
                from nora.store import close_store
                close_store(old_cwd)
            except Exception:  # noqa: BLE001 — store close isn't safety-critical
                pass
        self.cwd = path
        set_cwd(path)
        # Per-session model memory: every successful turn writes
        # ``active_model`` into ``.nora/session_state.json`` (and so
        # does a successful ``set_model``). On session open, restore
        # that choice so a researcher who switched to Opus for a
        # particular project comes back to Opus next time. Falls back
        # to whatever ``self._model`` already held (set at __init__
        # time from the catalog default) when the session has no
        # recorded preference yet.
        self._restore_session_model_preference()
        return {"ok": True, "state": "ready", **self._ready_payload()}

    def _restore_session_model_preference(self) -> None:
        """Read ``active_model`` from this session's state file and
        apply it if it's a model we still know about whose provider
        is currently authed. Silent no-op when there's nothing to
        restore — leaves ``self._model`` and ``self._provider``
        untouched in that case."""
        if self.cwd is None:
            return
        try:
            from nora.session_state import read_session_state
            state = read_session_state(self.cwd)
        except Exception:  # noqa: BLE001 — never let state read break a session open
            return
        if state is None or not state.active_model:
            return
        try:
            from nora.provider.catalog import get_model
            info = get_model(state.active_model)
        except KeyError:
            # Stored model was removed from the catalog (e.g., an
            # OpenAI model id renamed). Drop the preference silently
            # rather than wedge the session.
            return
        if info.provider not in self._authed_providers():
            # Researcher rotated their key out since the last session
            # touched this dir; fall back to current selection rather
            # than try to open a session against a provider with no
            # credential.
            return
        self._provider = info.provider
        self._model = info.id

    def _stage_session(self, source_paths: list[str]) -> dict[str, Any]:
        """Copy user-selected files into a fresh session dir. Each
        source file must be a real readable file; otherwise the
        whole staging aborts so the session never starts with
        partial data.

        Two source files with the same basename (e.g., the researcher
        picks ``data.csv`` from two different folders) auto-rename to
        ``data (1).csv`` so the second doesn't clobber the first.
        Auto-rename is safe here because the session dir is
        brand-new — no prior state to preserve.
        """
        paths: list[Path] = []
        for s in source_paths:
            p = Path(s).expanduser().resolve()
            if not p.is_file():
                return {"ok": False, "reason": f"not a file: {p}"}
            paths.append(p)
        session = _new_session_dir()
        try:
            for src in paths:
                shutil.copy2(src, _disambiguate_target(session, src.name))
        except OSError as e:
            return {"ok": False, "reason": f"copy failed: {e}"}
        return self._set_cwd(session)

    def _stage_session_from_blobs(
        self, blobs: list[tuple[str, bytes]]
    ) -> dict[str, Any]:
        """Drag-drop path: write decoded bytes into a fresh session
        dir. Filenames are basename-only (sanitized by the caller)
        to prevent path traversal.

        Two dropped files with the same basename auto-rename to
        ``name (1).ext`` (same logic as ``_stage_session``).
        """
        session = _new_session_dir()
        try:
            for name, content in blobs:
                target = session / name
                # Defense in depth: never write outside the session.
                if target.resolve().parent != session.resolve():
                    return {"ok": False, "reason": f"suspicious filename: {name!r}"}
                target = _disambiguate_target(session, name)
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
                "Connected. Tell Nora about your research question, "
                "or ask what datasets are available."
            ),
            "policy": self._policy_summary(),
        }

    async def _ensure_session(self) -> ProviderSession:
        assert self.cwd is not None
        if self._session is None:
            # Build the system prompt fresh each open so it always
            # reflects the current cwd's dataset listing. Both
            # providers consume the same rendered prompt.
            system_prompt = build_system_prompt(self.cwd, SERVER_NAME)
            self._session = open_session(
                self._provider,
                cwd=self.cwd,
                model=self._model,
                system_prompt=system_prompt,
                # We don't rely on any provider-side conversation
                # resume — those features (Claude CLI session store,
                # OpenAI ``previous_response_id``) are opaque and can
                # silently miss. Instead we open a fresh conversation
                # every time and prepend our own condensed history on
                # the first turn (see the ``_needs_context_prefix``
                # branch in ``_run_turn``).
                continue_conversation=False,
            )
            await self._session.open()
            # Fresh session → next turn must carry the prior-turns
            # prefix so the model picks up memory. Cleared after one
            # successful emission so mid-session turns stay lean.
            self._needs_context_prefix = True
            # Snapshot the datasets that just went into the system
            # prompt's ``datasets_list``. ``_run_turn`` diffs against
            # this on every turn and surfaces any new names to the
            # model so mid-chat uploads don't go unnoticed.
            from nora.system_prompt import scan_datasets
            self._known_datasets = frozenset(
                p.name for p in scan_datasets(self.cwd)
            )
        return self._session

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
                session = await self._ensure_session()
            except Exception as e:  # noqa: BLE001
                self._push_event({
                    "type": "turn_error",
                    "message": f"session setup failed: {e}",
                })
                return

            # Memory: on the first turn after a fresh session open,
            # prepend the condensed prior-turn transcript so the model
            # picks up where the conversation left off. The prefix is
            # wrapped in a clearly-marked block so the model treats
            # it as background rather than as content to respond to.
            # We clear the flag preemptively but restore it on any
            # error/cancellation so the researcher never loses
            # memory injection because the first turn after a reopen
            # happened to fail.
            prompt = text
            carried_prefix = False
            if self._needs_context_prefix:
                prefix = _build_context_prefix(self.cwd)
                if prefix:
                    prompt = prefix + "\n\n" + text
                    carried_prefix = True
                self._needs_context_prefix = False

            # Mid-chat dataset uploads: the system prompt's
            # ``datasets_list`` block is frozen at session open, so
            # a parquet dropped ten turns in is invisible to the
            # model. Diff what's currently on disk against the
            # snapshot taken when the session opened (or last
            # refreshed); any new names get prepended as a one-line
            # notice so the model can reach them via get_schema
            # without the researcher having to spell out the path.
            try:
                from nora.system_prompt import scan_datasets as _scan
                current_datasets = frozenset(
                    p.name for p in _scan(self.cwd)
                )
            except Exception:  # noqa: BLE001 — never let scan break a turn
                current_datasets = self._known_datasets
            new_datasets = current_datasets - self._known_datasets
            carried_dataset_diff: frozenset[str] = frozenset()
            if new_datasets:
                added_lines = "\n".join(
                    f"  - {n}" for n in sorted(new_datasets)
                )
                dataset_notice = (
                    "[The researcher added new datasets to the "
                    "working directory mid-session. These weren't in "
                    "the original prompt's listing but are reachable "
                    "via get_schema / submit_script:\n"
                    f"{added_lines}\n]\n\n"
                )
                prompt = dataset_notice + prompt
                carried_dataset_diff = new_datasets
            self._known_datasets = current_datasets

            # Mid-chat script attachments: render the staged ``.py`` /
            # ``.do`` / ``.r`` / ``.rmd`` files (drag-dropped into the
            # composer) as a prefix block so the model sees the source
            # alongside the researcher's question. Same restore-on-
            # error pattern as the context prefix above — if the turn
            # fails the attachments come back so a re-send still
            # carries them.
            carried_attachments: list[dict[str, Any]] = []
            if self._pending_script_attachments:
                attach_block = _build_script_attachment_prefix(
                    self._pending_script_attachments, self.cwd,
                )
                if attach_block:
                    prompt = attach_block + prompt
                    carried_attachments = list(self._pending_script_attachments)
                self._pending_script_attachments = []

            # Track whether the provider stream emitted a terminal
            # event (turn_done / turn_error / auth_failure). If it
            # closes WITHOUT one — rare but observed in practice on
            # SDK glitches and dropped sockets — the JS state machine
            # would otherwise stay stuck on "sending" forever, with
            # no terminal event to flip the Send button back. We
            # synthesise one in that case so the composer always
            # recovers.
            saw_terminal = False
            try:
                async for evt in session.send(prompt, images=images):
                    if isinstance(evt, (TurnDone, TurnError, AuthFailure)):
                        saw_terminal = True
                    self._push_event(_event_to_dict(evt))
                if not saw_terminal:
                    self._push_event({
                        "type": "turn_error",
                        "message": (
                            "the provider stream ended without a "
                            "result — try again, or use Stop and "
                            "resend if the chat feels stuck"
                        ),
                    })
                # Refresh the durable session snapshot after a clean
                # turn. Best-effort — write_session_state swallows
                # OSError internally so a disk-full or permission
                # hiccup can't break chat. We skip this on cancel /
                # error paths so a partial turn doesn't get recorded
                # as "last activity".
                try:
                    from nora.session_state import write_session_state
                    write_session_state(self.cwd, model=self._model)
                except Exception:  # noqa: BLE001 — never let state write break a turn
                    pass
            except asyncio.CancelledError:
                # Researcher hit Stop. Surface a terminal event so
                # the UI re-enables the composer via its standard
                # event handler. Don't re-raise — cancellation is
                # expected here, not an error condition.
                if carried_prefix:
                    self._needs_context_prefix = True
                if carried_attachments:
                    # Front-prepend so any new attachments staged
                    # during the cancelled turn still come first.
                    self._pending_script_attachments = (
                        carried_attachments + self._pending_script_attachments
                    )
                if carried_dataset_diff:
                    # Roll the dataset snapshot back so the next turn
                    # re-emits the "newly added" notice; otherwise a
                    # cancelled-during-first-turn parquet would never
                    # be announced.
                    self._known_datasets = (
                        self._known_datasets - carried_dataset_diff
                    )
                self._push_event({
                    "type": "turn_error",
                    "message": "cancelled",
                })
                return
            except Exception as e:  # noqa: BLE001
                if carried_prefix:
                    self._needs_context_prefix = True
                if carried_attachments:
                    self._pending_script_attachments = (
                        carried_attachments + self._pending_script_attachments
                    )
                if carried_dataset_diff:
                    self._known_datasets = (
                        self._known_datasets - carried_dataset_diff
                    )
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
        js = f"window.nora_event({json.dumps(payload)});"
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

    def _record_user_message(self, text: str, *, image_count: int = 0) -> None:
        """Persist the user-side record for a newly queued turn.

        Before appending the new record, drop any trailing orphaned
        ``user_message`` from a previously failed / unsent turn. That
        keeps chat replay and warm-start context from accumulating
        "I sent this but nothing ever came back" bubbles forever; the
        next real send replaces the failed attempt.
        """
        self._drop_trailing_orphan_user_message()
        attached_names = [
            a["name"] for a in self._pending_script_attachments
        ]
        record: dict[str, Any] = {"type": "user_message", "text": text}
        if attached_names:
            record["attachments"] = attached_names
        if image_count > 0:
            record["image_count"] = image_count
        self._persist_event(record)

    def _drop_trailing_orphan_user_message(self) -> None:
        """Remove the last persisted record iff it is a bare
        ``user_message`` with no assistant/tool events after it.

        Nora persists user bubbles immediately when the researcher
        presses Send so the live transcript and replay stay aligned.
        If that turn fails before any response artifact is persisted,
        the log ends with a lone ``user_message``. On the next send we
        treat that stale attempt as disposable and replace it with the
        new one, matching the UI behavior of dropping no-reply turns
        once the researcher retries.
        """
        if self.cwd is None:
            return
        path = self.cwd / ".nora" / "chat_history.jsonl"
        if not path.exists():
            return
        try:
            lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        except OSError:
            return

        last_idx: int | None = None
        last_record: dict[str, Any] | None = None
        for i in range(len(lines) - 1, -1, -1):
            raw = lines[i].strip()
            if not raw:
                continue
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                last_idx = i
                last_record = parsed
                break
        if last_idx is None or last_record is None:
            return
        if last_record.get("type") != "user_message":
            return
        try:
            with path.open("w", encoding="utf-8") as f:
                for i, line in enumerate(lines):
                    if i == last_idx:
                        continue
                    f.write(line)
        except OSError:
            pass

    def _persist_event(self, payload: dict[str, Any]) -> None:
        if self.cwd is None:
            return
        etype = payload.get("type")
        if etype not in self._PERSIST_TYPES:
            return
        # Stamp the event with a UTC ISO timestamp if the caller
        # didn't provide one. Readers (chat_history.read_turns,
        # session_state writer) tolerate missing timestamps for
        # backwards compat with older logs, but adding one per
        # event going forward lets us surface "last active" times,
        # order events from mixed sources, and feed the rolling
        # session summary. We don't mutate the caller's dict —
        # a shallow copy is cheap and avoids surprising _push_event
        # consumers that keep the original reference.
        record = dict(payload)
        record.setdefault(
            "timestamp",
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        try:
            history_dir = self.cwd / ".nora"
            history_dir.mkdir(parents=True, exist_ok=True)
            path = history_dir / "chat_history.jsonl"
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            # Persistence failing shouldn't block live chat. The
            # transcript on screen stays intact; replay just won't
            # include this event.
            pass

    def _close_session_blocking(self) -> None:
        """Best-effort close of the persistent provider session.

        Safe to call repeatedly. Used by session switches, provider
        switches, and teardown paths to guarantee the next turn starts
        with a fresh provider session.
        """
        if self._session is None:
            return
        if self._loop is None:
            self._session = None
            return

        session = self._session

        async def _close() -> None:
            try:
                await session.close()
            except Exception:  # noqa: BLE001
                pass

        fut = asyncio.run_coroutine_threadsafe(_close(), self._loop)
        try:
            fut.result(timeout=3)
        except Exception:  # noqa: BLE001
            pass
        self._session = None

    def get_chat_history(self) -> dict[str, Any]:
        """Return the persisted chat log for the active session so
        the UI can replay past messages after a session switch.
        Empty list if the session has no history yet."""
        if self.cwd is None:
            return {"ok": True, "events": []}
        path = self.cwd / ".nora" / "chat_history.jsonl"
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
            from nora.policy import DEFAULT_MAX_DEPTH
            return {"default_max_depth": DEFAULT_MAX_DEPTH, "datasets": []}
        from nora.app import _scan_datasets
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


def _build_context_prefix(cwd: Path | None) -> str:
    """Thin shim around :func:`nora.chat_history.build_context_prefix`.

    Kept as a module-local alias because ``_run_turn`` references it;
    the actual pure-function implementation (no SDK deps, trivially
    testable) lives in chat_history.py alongside the Turn reader.
    Production callers pull recent results from the sanitized-payload
    store here — tests import the pure function directly and inject
    their own results list.
    """
    from nora.chat_history import build_context_prefix

    if cwd is None:
        return ""
    # Pull recent results from the store. Opening it may fail on a
    # brand-new cwd whose .nora/results.db hasn't been created
    # yet — swallow and treat as "no prior results".
    try:
        from nora.store import get_store
        rows: list[Any] = list(get_store(cwd).list_all())
    except Exception:  # noqa: BLE001 — store unavailable shouldn't block resume
        rows = []
    return build_context_prefix(cwd, results=rows)


# Extensions whose contents we inline into the next user message
# when the researcher drops one into the composer mid-chat. R / Stata
# / Python source files plus R Markdown — bounded text formats whose
# whole point is to be read end-to-end. ``.ipynb`` is intentionally
# excluded (JSON envelope, very wordy), ``.log`` / ``.smcl`` excluded
# (often huge, and rarely useful as model context — researchers
# normally want the raw output, not the file).
_INLINE_SCRIPT_EXTS: frozenset[str] = frozenset({
    ".py", ".do", ".r", ".rmd",
})

# Hint that travels with the script — "this file is Python / Stata /
# R / R Markdown" — so the model knows what fence to use if it
# decides to reuse the content.
_SCRIPT_LANGUAGE_HINTS: dict[str, tuple[str, str]] = {
    ".py": ("Python", "python"),
    ".do": ("Stata", "stata"),
    ".r": ("R", "r"),
    ".rmd": ("R Markdown", "rmarkdown"),
}

# Per-file size cap for inlined script content. Above this, the
# attachment is summarised (filename + size + first chunk + truncation
# marker) so a multi-MB log accidentally renamed to ``.py`` can't
# blow up the next prompt.
_INLINE_SCRIPT_MAX_BYTES = 64 * 1024
# Aggregate cap across all attachments staged for a single turn.
# Generous enough to attach 5 typical analysis scripts; small enough
# that a malicious / mistaken drop of 50 files can't OOM the model.
_INLINE_SCRIPT_TOTAL_CAP = 256 * 1024


def _build_script_attachment_prefix(
    attachments: list[dict[str, Any]], cwd: Path | None,
) -> str:
    """Render the staged script attachments as a single text block
    that prefixes the next user message.

    Format mirrors the existing ``_build_context_prefix`` idiom:
    a clearly-marked opener, one fenced code block per file (with a
    language hint so the model knows what dialect it is), and a
    matching closer that names where the prompt resumes. The model
    is told these are background — the researcher's question follows.

    Aggregate size is capped at ``_INLINE_SCRIPT_TOTAL_CAP``; once
    full, remaining files are listed by name only with a "(too large
    to include)" note so the model knows they exist on disk.
    """
    if not attachments:
        return ""
    parts: list[str] = [
        "[Files the researcher attached to this message — reference "
        "them as needed; the originals are saved alongside the data "
        "in this session]\n"
    ]
    used = len(parts[0])
    for att in attachments:
        name = att.get("name", "(unnamed)")
        content = att.get("content", "")
        lang_label, fence_lang = _SCRIPT_LANGUAGE_HINTS.get(
            att.get("ext", ""), ("plaintext", "")
        )
        rel_path = name
        if cwd is not None:
            try:
                rel_path = str((cwd / name).relative_to(cwd))
            except (ValueError, OSError):
                rel_path = name
        header = f"\n### {rel_path} ({lang_label})\n"
        block = (
            f"{header}```{fence_lang}\n"
            f"{content}\n"
            f"```\n"
        )
        if used + len(block) > _INLINE_SCRIPT_TOTAL_CAP:
            parts.append(
                f"\n### {rel_path} ({lang_label}) — "
                f"omitted, total attachment budget exceeded "
                f"(file is on disk at {rel_path})\n"
            )
            continue
        parts.append(block)
        used += len(block)
    parts.append(
        "\n[End of attached files. The researcher's message follows.]\n\n"
    )
    return "".join(parts)


def _stage_script_for_next_turn(
    pending: list[dict[str, Any]],
    name: str,
    ext: str,
    content_bytes: bytes,
) -> None:
    """Decode ``content_bytes`` as text (best-effort UTF-8) and append
    a stage entry. Decoding failures are recorded as a placeholder
    so the model still knows the file exists, even if we couldn't
    surface its contents.

    Per-file size cap is enforced here: scripts longer than
    ``_INLINE_SCRIPT_MAX_BYTES`` get their first chunk + a clear
    truncation marker. The on-disk copy is always full.
    """
    if ext not in _INLINE_SCRIPT_EXTS:
        return
    raw = content_bytes
    truncated_note = ""
    if len(raw) > _INLINE_SCRIPT_MAX_BYTES:
        raw = raw[:_INLINE_SCRIPT_MAX_BYTES]
        truncated_note = (
            f"\n[… {len(content_bytes) - _INLINE_SCRIPT_MAX_BYTES} bytes "
            f"truncated; full file on disk]"
        )
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("utf-8", errors="replace")
    pending.append({
        "name": name,
        "ext": ext,
        "content": text + truncated_note,
        "bytes": len(content_bytes),
    })


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
    from nora.schema import DATA_EXTENSIONS as _DATA_EXTS
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


def _disambiguate_target(directory: Path, name: str) -> Path:
    """Pick a non-colliding write path inside ``directory``.

    If ``directory / name`` doesn't exist, returns it unchanged.
    Otherwise returns ``stem (1).ext``, ``stem (2).ext``, … — first
    free slot, capped at 999 to refuse pathological inputs cleanly
    rather than spin forever.

    Used by the fresh-session staging paths where two source files
    can share a basename (researcher selects ``data.csv`` from two
    folders) — autorenaming inside a freshly-created dir is safe and
    UX-friendly because the dir has no prior state. The "add to
    existing session" paths take the opposite tack and refuse on
    collision so a re-uploaded ``data.csv`` doesn't silently
    invalidate prior analyses against the original.
    """
    target = directory / name
    if not target.exists():
        return target
    stem = target.stem
    suffix = target.suffix
    for i in range(1, 1000):
        candidate = directory / f"{stem} ({i}){suffix}"
        if not candidate.exists():
            return candidate
    raise OSError(
        f"too many basename collisions for {name!r} in {directory}"
    )


def _read_raw_logs(run_dir: str | None) -> tuple[str, str]:
    """Read ``stdout.log`` and ``stderr.log`` from a run dir, if they
    exist. Returns empty strings when the dir is missing or the files
    haven't been written.

    Content is capped at 32 KB per stream — enough to show a full
    regression table, short of letting a runaway log blow up the
    browser. The full log is still on disk at ``run_dir`` for audit.

    Truncation keeps **both ends** rather than just the tail. For
    exploratory scripts like ``df.head(10)`` on a wide dataset, the
    start carries what the researcher actually needs (shape, dtypes,
    column-named first rows of the table); the runtime helpers
    (``nora.from_lm``, etc.) print their summary at the end. A
    tail-only cap dropped the column names every time and showed
    only trailing rows whose context was lost.
    """
    if not run_dir:
        return "", ""
    stdout_text = ""
    stderr_text = ""
    cap = 32 * 1024
    # 75/25 head/tail split. Empirically the start is more useful for
    # exploratory scripts and the very end is where helper-printed
    # summaries live; the middle is usually repetitive (pandas'
    # wrapped to_string continuation blocks, dtypes-per-column
    # listings) and the cheapest to drop.
    head_cap = (cap * 3) // 4
    tail_cap = cap - head_cap
    for name, bucket in (("stdout.log", "stdout"), ("stderr.log", "stderr")):
        try:
            content = Path(run_dir, name).read_text(
                encoding="utf-8", errors="replace"
            )
        except OSError:
            continue
        if len(content) > cap:
            head = content[:head_cap]
            tail = content[-tail_cap:]
            dropped = len(content) - len(head) - len(tail)
            content = (
                f"{head}\n"
                f"[… {dropped} bytes truncated from the middle; "
                f"full log at {run_dir}/{name} …]\n"
                f"{tail}"
            )
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
        print(f"nora-ui: {e}", file=sys.stderr)
        sys.exit(2)
    if not path.is_dir():
        msg = [f"nora-ui: not a directory: {path}"]
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
        prog="nora-ui",
        description=(
            "Nora — web UI. Same backend as `nora` (terminal), "
            "different frontend. Launch without a path to drop/choose "
            "files from the UI."
        ),
    )
    parser.add_argument(
        "cwd", nargs="?", default=None,
        help=(
            "Optional. Working directory Nora operates in. If "
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
            "nora-ui: warning — sandbox-exec unavailable; "
            "submit_script will refuse to run.",
            file=sys.stderr,
        )

    import webview  # lazy import so `nora` (terminal) doesn't
                    # require pywebview at import time

    web_dir = Path(__file__).parent / "web"
    index_path = web_dir / "index.html"
    if not index_path.is_file():
        print(
            f"nora-ui: missing web assets at {index_path}",
            file=sys.stderr,
        )
        sys.exit(2)

    bridge = NoraBridge(cwd=cwd)
    bridge.start_loop()

    window = webview.create_window(
        title="Nora",
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
