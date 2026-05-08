"""Nora — UI entry point.

Opens a native WKWebView window (via pywebview) hosting a local HTML
chat interface, and bridges it to the rest of the stack: a
``ProviderSession`` (Anthropic or OpenAI) + the MCP tool surface +
sanitizer + policy + sandboxed executor.

Launched via:

    uv run python -m nora [cwd]

or the ``nora`` console script.

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
    provider_for_model,
)
from nora.provider.catalog import (
    ALL_MODELS,
    PROVIDER_DEFAULTS,
    PROVIDER_PRICING_URLS,
)
from nora.runner import SessionRunner


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
        # run. ``self.cwd`` is the *focused* session — what the UI is
        # currently showing. The actual execution state per session
        # lives in ``self._runners``, keyed by str(cwd).
        self.cwd: Path | None = cwd
        self._window: Any = None  # set after the window is created
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        # Per-session execution state. Each :class:`SessionRunner`
        # owns its own provider session, send-lock, current turn
        # task, warm-start flag, model, and pending attachments. The
        # bridge holds them by str(cwd) and keeps them alive across
        # focus switches: a long regression in session A keeps
        # streaming events even while the UI is showing session B.
        self._runners: dict[str, SessionRunner] = {}
        # Defaults applied when a NEW runner is created (i.e., a
        # session is focused for the first time and has no recorded
        # active_model). Set by the auth-reconcile flow + the
        # auth-screen's "Use OpenAI / Use Anthropic" buttons; never
        # mutated by per-session model swaps (those affect only the
        # active runner).
        self._default_provider: str = "anthropic"
        self._default_model: str = PROVIDER_DEFAULTS[self._default_provider]
        if cwd is not None:
            # If the launcher handed us a cwd, eagerly create its
            # runner so per-session model preference is applied
            # before the page is even loaded. Otherwise the first
            # ``ui_ready`` would try to read ``active_model`` against
            # an absent runner and fall back to the default.
            self._ensure_runner_for_cwd(cwd)

    # -------- lifecycle --------

    def attach(self, window: Any) -> None:
        self._window = window

    def list_busy_sessions(self) -> dict[str, Any]:
        """Return the set of session cwds whose runner currently has a
        turn in flight. The web UI calls this on page load (initial
        boot AND after a hard reload / Cmd+Shift+R) to rebuild its
        ``busySessions`` Set — that state lives in JS module scope and
        gets wiped on every page navigation, so without this call the
        sidebar busy dot and the loading indicator both disappear
        even though the backend turn is still streaming.

        Returns ``{ok: True, cwds: [...]}``. Best-effort: a runner
        whose ``is_busy()`` raises is treated as not busy rather than
        crashing the whole call."""
        cwds: list[str] = []
        for cwd_str, runner in list(self._runners.items()):
            try:
                if runner.is_busy():
                    cwds.append(cwd_str)
            except Exception:  # noqa: BLE001 — defensive
                continue
        return {"ok": True, "cwds": cwds}

    def hard_reload(self) -> dict[str, Any]:
        """Recompute the cache-bust build-id from current asset mtimes,
        write a fresh ``.index.bust-<id>.html``, and navigate the
        window to its file:// URL. Bound to ``Cmd+Shift+R`` in
        ``app.js``.

        Why this exists: editing CSS / JS while nora is running and
        then doing an in-app reload (Cmd+R) re-fetches the SAME
        ``style.css?v=<old-build-id>`` URL — WKWebView's persistent
        disk cache hits, and the researcher sees old rendering. Only
        a full nora restart re-runs ``_materialize_cache_busted_index``
        and produces a new URL. ``hard_reload`` does that work
        in-place so iteration doesn't require quitting the app.

        Returns the new build-id so the caller can verify the reload
        actually rolled the cache key (useful in dev console)."""
        if self._window is None:
            return {"ok": False, "reason": "window not attached"}
        try:
            web_dir = Path(__file__).parent / "web"
            index_path = web_dir / "index.html"
            served = _materialize_cache_busted_index(web_dir, index_path)
            self._window.load_url(str(served))
            return {
                "ok": True,
                "build_id": served.stem.split(".")[-1],
                "url": str(served),
            }
        except Exception as e:  # noqa: BLE001 — surface failure to JS
            return {"ok": False, "reason": str(e)}

    def start_loop(self) -> None:
        """Start the asyncio worker thread. Called once, before the
        webview starts serving the page. Per-runner locks are created
        inside each :class:`SessionRunner`; the bridge no longer holds
        a global send-lock."""
        self._loop = asyncio.new_event_loop()

        def _run() -> None:
            asyncio.set_event_loop(self._loop)
            self._loop.run_forever()

        self._loop_thread = threading.Thread(
            target=_run, name="nora-event-loop", daemon=True
        )
        self._loop_thread.start()

    def stop_loop(self) -> None:
        """Tear down all runners and stop the worker loop. Called once
        on app shutdown — this is the ONLY place runners get closed
        in normal operation. Session focus changes do NOT close
        runners (that was the bug — closing under an in-flight stream
        killed the turn)."""
        if self._loop is None:
            return
        runners = list(self._runners.values())

        async def _close_all() -> None:
            for r in runners:
                try:
                    await r.close()
                except Exception:  # noqa: BLE001
                    pass

        fut = asyncio.run_coroutine_threadsafe(_close_all(), self._loop)
        try:
            fut.result(timeout=5)
        except Exception:  # noqa: BLE001
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
        # Make sure the bridge default + active runner agree with
        # what's actually authed. Without this, a researcher who
        # configured only OpenAI would hit chat with the Anthropic
        # default still selected and the first turn would fail with
        # "no Anthropic credential."
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
        # ``nora <cwd>`` and the auth-Continue path both fall
        # through to ``showLanding`` even though the bridge knows
        # the session is ready to chat.
        return {"state": "ready", **self._ready_payload(), "auth": status}

    def _reconcile_active_provider_with_auth(self) -> None:
        """Ensure the bridge's *defaults* (used for new runners) name a
        provider the researcher can actually use right now. Also
        promotes the active runner (if any) when its provider is
        unauthed.

        Default at construction is Anthropic. If the researcher
        configures only OpenAI, the bridge flips its defaults to
        OpenAI before any new session opens.

        Existing runners that aren't currently focused are left
        alone — they may still hold a session against a now-unauthed
        provider, but that's a per-session problem and surfaces as
        an auth_failure on the next send for that runner. We don't
        force-close idle runners here because that's a side-effect
        the researcher didn't ask for.
        """
        authed = self._authed_providers()
        if not authed:
            return
        # Update bridge defaults if unauthed.
        if self._default_provider not in authed:
            for candidate in PROVIDER_DEFAULTS:
                if candidate in authed:
                    self._default_provider = candidate
                    self._default_model = PROVIDER_DEFAULTS[candidate]
                    break
        # If the active runner is using an unauthed provider, swap
        # it to the default — and persist so a reload survives.
        # Skip the swap if the runner is mid-turn: ``swap_model``
        # closes and reopens the underlying provider session, which
        # would tear down a live stream. The unauthed turn will
        # still surface an ``auth_failure`` naturally on the next
        # event from the SDK; the next ``ui_ready`` (after the
        # researcher dismisses the failure or reloads) will catch
        # the swap when the runner is idle. ``delete_credential``
        # already follows the same "leave busy runners alone" rule
        # for its idle-runner close pass — without this guard,
        # ``ui_ready`` would race ahead of that policy and replace
        # the very session ``delete_credential`` deliberately
        # spared.
        active = self._active_runner()
        if (
            active is not None
            and active.provider not in authed
            and not active.is_busy()
        ):
            new_provider = self._default_provider
            new_model = self._default_model
            self._run_on_loop(active.swap_model(new_model, new_provider))
            self._persist_active_model()

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

        Size capped per-file at ``_DRAG_DROP_MAX_BYTES`` (512 MB).
        The constraint is peak memory while transferring through
        the bridge: a file of N bytes needs roughly 3–4N during
        upload (JS ArrayBuffer + JS base64 string + Python-side
        decode), so 512 MB peaks around 2 GB — comfortable on any
        modern Mac. Larger datasets should use the file picker
        (:meth:`choose_files`), which copies directly from disk
        with no memory overhead and no size limit.

        The cap is enforced on the base64 string length BEFORE
        ``b64decode`` runs so a forged or malicious oversize blob
        doesn't allocate multiple GB of decoded bytes just to be
        rejected. The JS side gates on ``file.size`` first; this is
        defense-in-depth for clients that bypass the JS check.
        """
        if not files:
            return {"ok": False, "reason": "no files"}
        import base64
        decoded: list[tuple[str, bytes]] = []
        for item in files:
            name = item.get("name", "")
            content_b64 = item.get("content", "")
            if not name or not isinstance(content_b64, str):
                continue
            # Strip any data URL prefix JS may have added.
            if "," in content_b64:
                content_b64 = content_b64.split(",", 1)[1]
            # Pre-decode size gate. base64 expands 4:3, so a 512 MB
            # binary file is ~683 MB encoded; ``_b64_oversize`` does
            # the comparison without materializing the decoded blob.
            if _b64_oversize(content_b64, _DRAG_DROP_MAX_BYTES):
                approx_mb = (len(content_b64) * 3 // 4) // (1024 * 1024)
                return {
                    "ok": False,
                    "reason": _drag_drop_oversize_message(
                        name, approx_mb, "Choose Files…",
                    ),
                }
            try:
                blob = base64.b64decode(content_b64, validate=False)
            except Exception:  # noqa: BLE001
                return {"ok": False, "reason": f"could not decode {name!r}"}
            if len(blob) > _DRAG_DROP_MAX_BYTES:
                mb = len(blob) // (1024 * 1024)
                return {
                    "ok": False,
                    "reason": _drag_drop_oversize_message(
                        name, mb, "Choose Files…",
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

    def send_message(self, text: str) -> str | None:
        """Schedule a turn on the active session's runner.

        Returns the new turn's id (a 16-char hex string), or ``None``
        if the send couldn't be scheduled (no cwd, worker loop down).
        Events stream back via ``_dispatch_event``, each stamped with
        the same id so the JS event filter can drop late events from
        a turn the researcher cancels later.
        """
        return self._send_to_active(text, images=None, target_cwd=None)

    def send_message_with_images(
        self, text: str, images: list[dict[str, Any]]
    ) -> str | None:
        """Schedule a turn with attached images on the active runner.

        Same return contract as ``send_message``: the new turn id, or
        ``None`` on early failure. ``images[i] = {"data": <base64>,
        "mime": ...}``.
        """
        return self._send_to_active(text, images=images, target_cwd=None)

    def send_message_to_session(
        self, session_cwd: str, text: str,
    ) -> str | None:
        """Schedule a turn on the runner whose cwd matches ``session_cwd``.

        Used by the JS-side queue-flush path: when a background
        turn finishes on session A, its queued follow-up has to fire
        AGAINST A even if the user has since switched the focus to
        session B. Without this explicit-target variant, the queued
        send routed through ``send_message`` would land on whatever
        ``self.cwd`` happened to be at flush time — a cross-session
        execution mix-up where A's pending message ran in B's
        working directory.
        """
        return self._send_to_active(
            text, images=None, target_cwd=session_cwd,
        )

    def send_message_with_images_to_session(
        self, session_cwd: str, text: str, images: list[dict[str, Any]],
    ) -> str | None:
        """Image-bearing twin of :meth:`send_message_to_session`.

        Same routing rule: the turn fires on the runner registered
        under ``session_cwd``, regardless of which session is
        currently focused.
        """
        return self._send_to_active(
            text, images=images, target_cwd=session_cwd,
        )

    def _send_to_active(
        self,
        text: str,
        images: list[dict[str, Any]] | None,
        target_cwd: str | None,
    ) -> str | None:
        """Schedule a turn on a runner.

        ``target_cwd`` selects the runner explicitly (used by the
        queue-flush path so a background-finished turn fires its
        follow-up against the right session, not whichever happens
        to be focused). When ``None``, the active runner is used —
        the normal interactive-send path.

        Each runner has its own send-lock, so kicking off a turn on
        runner A while runner B is still streaming does NOT block —
        they execute concurrently. The runner stamps every event
        with ``session_cwd`` AND ``turn_id`` so the JS side can
        filter for the active focus AND drop late events from a
        cancelled turn; persistence always lands in the runner's
        own ``chat_history.jsonl``.

        Generates the turn id here (synchronously, before scheduling
        the coroutine) so it can be returned to the JS-side
        ``send_message`` immediately. JS captures it on the awaited
        Promise; if Stop fires before the first event arrives, the
        bridge already knows which id is in flight on this runner
        and the cancel path can mark it cancelled atomically.
        """
        if self._loop is None:
            self._dispatch_event({
                "type": "turn_error",
                "message": "worker loop not running",
                "session_cwd": (
                    target_cwd if target_cwd is not None
                    else (str(self.cwd) if self.cwd else None)
                ),
            })
            return None
        # Resolve which runner this send goes to. Explicit target
        # wins; without one, fall back to the focused cwd.
        if target_cwd is not None:
            try:
                resolved = Path(target_cwd).resolve()
            except (OSError, RuntimeError):
                self._dispatch_event({
                    "type": "turn_error",
                    "message": f"invalid target cwd: {target_cwd!r}",
                    "session_cwd": target_cwd,
                })
                return None
            # Match against the runners dict — its keys are resolved
            # paths. A targeted send must hit a runner that already
            # exists; lazily creating one for an arbitrary caller-
            # supplied path would let a stale queue resurrect a
            # session the researcher has since deleted.
            runner_key = str(resolved)
            runner = self._runners.get(runner_key)
            if runner is None:
                self._dispatch_event({
                    "type": "turn_error",
                    "message": (
                        "queued message dropped — its session is no "
                        "longer open"
                    ),
                    "session_cwd": runner_key,
                })
                return None
        else:
            if self.cwd is None:
                self._dispatch_event({
                    "type": "turn_error",
                    "message": (
                        "no working directory set — choose files or a "
                        "folder first"
                    ),
                    "session_cwd": None,
                })
                return None
            runner = self._ensure_runner_for_cwd(self.cwd)
        self._record_user_message(runner, text, image_count=len(images or []))
        # 16 hex chars = 64 bits of entropy. Vastly more than enough
        # for a per-session non-collision guarantee, short enough to
        # log + paste comfortably when debugging cancellation issues.
        turn_id = uuid.uuid4().hex[:16]
        coro = runner.run_turn(
            text,
            images=images,
            on_event=self._dispatch_event,
            build_context_prefix=_build_context_prefix,
            build_script_prefix=_build_script_attachment_prefix,
            turn_id=turn_id,
        )
        # Register the id BEFORE scheduling so a Stop fired in the
        # tiny window between this method returning and ``run_turn``
        # actually starting on the worker loop still has something to
        # cancel. Without this, ``interrupt_turn`` would see no
        # current turn AND no pending turn and report "no turn in
        # flight" while the runner went on to execute the supposedly-
        # cancelled turn.
        runner.register_pending_turn(turn_id)
        asyncio.run_coroutine_threadsafe(coro, self._loop)
        return turn_id

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
                    runner = self._active_runner()
                    if runner is not None:
                        try:
                            _stage_script_for_next_turn(
                                runner.pending_script_attachments,
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
            safe_name = Path(name).name
            ext = Path(safe_name).suffix.lower()
            # Pre-decode size gate. Both data/script files (capped at
            # _DRAG_DROP_MAX_BYTES, 512 MB) and images (capped at
            # _IMAGE_MAX_BYTES, 5 MB) are checked BEFORE the
            # ``base64.b64decode`` allocation. Without the image-side
            # check, an oversize image flowed through to the decode
            # call which materialised the full encoded + decoded
            # bytes before rejection — a 100 MB image briefly held
            # ~233 MB of transient heap. The savings is in skipping
            # the allocation, not the arithmetic. The "+" button next
            # to the composer uses the native picker and has no size
            # limit.
            if ext in _COPY_EXTS:
                cap: int | None = _DRAG_DROP_MAX_BYTES
                hint = "the + button next to the composer"
            elif ext in _IMAGE_EXTS_MIMES:
                cap = _IMAGE_MAX_BYTES
                hint = ""  # images skipped silently below, not error-returned
            else:
                cap = None
                hint = ""
            if cap is not None and _b64_oversize(content_b64, cap):
                approx_mb = (len(content_b64) * 3 // 4) // (1024 * 1024)
                # Images skip via the per-file ``skipped`` list (the
                # researcher dragged a folder of mixed sizes; the
                # 50 MB screenshot shouldn't fail the whole drop).
                # Data and script files error-return because they're
                # the primary target of the drop and silently dropping
                # one would hide the failure.
                if ext in _IMAGE_EXTS_MIMES:
                    skipped.append(
                        f"{safe_name} (>{cap // (1024 * 1024)} MB)"
                    )
                    continue
                return {
                    "ok": False,
                    "reason": _drag_drop_oversize_message(
                        safe_name, approx_mb, hint,
                    ),
                }
            try:
                blob = base64.b64decode(content_b64, validate=False)
            except Exception:  # noqa: BLE001
                return {"ok": False, "reason": f"could not decode {name!r}"}
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
                    runner = self._active_runner()
                    if runner is not None:
                        _stage_script_for_next_turn(
                            runner.pending_script_attachments,
                            safe_name, ext, blob,
                        )
            elif ext in _IMAGE_EXTS_MIMES:
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
                # Reuse the original ``content_b64`` for the model-facing
                # data field instead of re-encoding ``blob``. The image
                # arrived as base64; round-tripping through decode +
                # encode wastes ~10 MB of transient memory per 5 MB
                # image for no semantic gain. ``content_b64`` is
                # already stripped of any data-URL prefix above.
                images.append({
                    "data": content_b64,
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
        # Surface the focused runner's choice when there is one;
        # fall back to the bridge defaults for the landing screen.
        active = self._active_runner()
        current_model = active.model if active is not None else self._default_model
        current_provider = active.provider if active is not None else self._default_provider
        return {
            "ok": True,
            "current": current_model,
            "current_provider": current_provider,
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

        Filesystem walk + classification live in
        :func:`nora.session_files.enumerate_session_files`; this
        method orchestrates the call and adds Files-panel-only
        thumbnail enrichment (base64 inline thumbs for image rows,
        PDF/EPS rasterisation via ``plot_convert.png_for``).
        """
        if self.cwd is None:
            return {"ok": True, "files": []}
        from nora.session_files import enumerate_session_files

        rows = enumerate_session_files(
            self.cwd,
            include_data=True,
            include_run_scripts=True,
        )
        for row in rows:
            self._enrich_files_panel_row(row)
        return {"ok": True, "files": rows}

    @staticmethod
    def _enrich_files_panel_row(row: dict[str, Any]) -> None:
        """Add inline thumbnail bytes (``data`` + ``mime``) to image
        rows in the Files panel. PDF/EPS rows get a sips-rasterised
        PNG sidecar shipped instead. 3 MB cap matches the chat-
        thumbnail cap so 1600px Stata PDFs / PNGs render at full res
        in the panel + lightbox; larger files still appear in the
        panel (with a placeholder + click-to-open) but their bytes
        don't ride through ``evaluate_js``.
        """
        import base64 as _base64

        _IMAGE_THUMB_CAP = 3 * 1024 * 1024
        _IMAGE_THUMB_EXTS = {".png", ".jpg", ".jpeg"}
        _IMAGE_MIME = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
        }

        ext = row.get("ext", "")
        size = row.get("size", 0)
        path = Path(row["path"])
        if ext in _IMAGE_THUMB_EXTS and size <= _IMAGE_THUMB_CAP:
            try:
                row["data"] = _base64.b64encode(path.read_bytes()).decode("ascii")
                row["mime"] = _IMAGE_MIME.get(ext, "image/png")
            except OSError:
                pass
        elif ext in (".pdf", ".eps"):
            from nora.plot_convert import png_for
            sidecar = png_for(path)
            if sidecar is not None:
                try:
                    sidecar_size = sidecar.stat().st_size
                except OSError:
                    sidecar_size = size
                if sidecar_size <= _IMAGE_THUMB_CAP:
                    try:
                        row["data"] = _base64.b64encode(
                            sidecar.read_bytes()
                        ).decode("ascii")
                        row["mime"] = "image/png"
                    except OSError:
                        pass

    def delete_session_file(self, path: str) -> dict[str, Any]:
        """Delete a file inside the active session.

        Used by the Files-panel trash icon. Accepts a full path (the
        listing already has it via ``list_session_files``) so we can
        delete files in run dirs (helper-produced plots) as well as
        in the session-cwd top level. Containment in the session
        cwd is verified before any unlink — outside paths are
        refused.

        Side effect: if the deleted file was a script staged for
        attachment, drop it from the runner's pending list so the
        composer chip vanishes too. Without that, the chip would
        still show even though the file is gone, and the next send
        would silently no-op the inline content.
        """
        if self.cwd is None:
            return {"ok": False, "reason": "no active session"}
        if not path:
            return {"ok": False, "reason": "no path"}
        try:
            target = Path(path).expanduser().resolve()
        except OSError as e:
            return {"ok": False, "reason": f"bad path: {e}"}
        cwd_resolved = self.cwd.resolve()
        if not _is_within(target, cwd_resolved):
            return {
                "ok": False,
                "reason": "path is outside the session — refused as a precaution",
            }
        if not target.is_file():
            return {"ok": False, "reason": f"not found: {target.name}"}
        try:
            target.unlink()
        except OSError as e:
            return {"ok": False, "reason": f"delete failed: {e}"}
        # Drop a matching staged attachment so the composer chip
        # follows the file's life cycle. Cover every pending list
        # the file could have landed on (script content inline,
        # @-mention notice, @-mention vision).
        runner = self._active_runner()
        if runner is not None:
            runner.pending_script_attachments = [
                a for a in runner.pending_script_attachments
                if a.get("name") != target.name
            ]
            runner.pending_mentioned_files = [
                n for n in runner.pending_mentioned_files
                if n != target.name
            ]
            runner.pending_mentioned_images = [
                a for a in runner.pending_mentioned_images
                if a.get("name") != target.name
            ]
        # Best-effort: also remove the cached PDF/EPS → PNG sidecar
        # if there was one. Otherwise the orphan PNG would keep
        # showing in the Files panel until the bridge restarted.
        sidecar = target.with_name(target.stem + ".nora.png")
        if sidecar.is_file():
            try:
                sidecar.unlink()
            except OSError:
                pass
        return {"ok": True, "name": target.name}

    def read_session_file_text(self, path: str) -> dict[str, Any]:
        """Read a session-resident text file's UTF-8 contents so the
        Files-panel "copy" button can hand them to the JS clipboard.

        Replaces the old "send to next message" affordance for
        scripts: researchers wanted to grab a do-file Nora wrote and
        paste it into another chat (or an external editor) without
        opening Finder, and "send" was a different verb that confused
        the action. Logs are also copyable now — same flow.

        Allowed kinds: scripts (``.py`` / ``.do`` / ``.r`` / ``.rmd``
        / ``.ipynb``) and logs (``.log`` / ``.smcl``). Binary kinds
        (data, graphs) are refused — the JS side has its own
        copy-image path for raster graphs and there's no useful
        text to put on the clipboard for a ``.dta`` or ``.gph``.

        Takes a full ``path`` rather than a basename (mirroring
        :meth:`delete_session_file`) because :meth:`list_session_files`
        surfaces ``submit_script``-written scripts from
        ``.nora/runs/<id>/`` with rewritten display names like
        ``script_a1b2c3d4.do``. A basename lookup against cwd would
        miss every one of those — the file on disk is plain
        ``script.do`` in a run dir. Containment in cwd is verified
        before any read.

        Size cap: 4 MB. The clipboard can hold more, but multi-MB
        log dumps don't paste cleanly into most editors and the
        researcher's intent ("grab this script") is better served
        by pointing them at the session folder via the topbar pill.
        """
        if self.cwd is None:
            return {"ok": False, "reason": "no active session"}
        if not path:
            return {"ok": False, "reason": "no path"}
        try:
            target = Path(path).expanduser().resolve()
        except OSError as e:
            return {"ok": False, "reason": f"bad path: {e}"}
        cwd_resolved = self.cwd.resolve()
        if not _is_within(target, cwd_resolved) or not target.is_file():
            return {"ok": False, "reason": f"not found: {Path(path).name}"}
        ext = target.suffix.lower()
        text_exts = _INLINE_SCRIPT_EXTS | {".ipynb", ".log", ".smcl"}
        if ext not in text_exts:
            return {
                "ok": False,
                "reason": (
                    f"{target.name} isn't a text file Nora can copy "
                    f"(scripts and logs only)."
                ),
            }
        try:
            size = target.stat().st_size
        except OSError as e:
            return {"ok": False, "reason": f"stat failed: {e}"}
        copy_text_max = 4 * 1024 * 1024
        if size > copy_text_max:
            return {
                "ok": False,
                "reason": (
                    f"{target.name} is {size // (1024 * 1024)} MB, "
                    f"over the 4 MB copy-text cap. Open the session "
                    f"folder via the topbar pill to grab the file."
                ),
            }
        try:
            text = target.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            return {"ok": False, "reason": f"read failed: {e}"}
        # Strip the executor's bootstrap (adopath / sys.path insert /
        # cd) before handing the script to the clipboard. The
        # preamble depends on ``NORA_LIB_DIR`` / ``NORA_CWD`` and a
        # run-dir-specific ``sys.path`` entry — outside Nora those
        # references don't resolve, and the stated purpose of this
        # button ("grab the script and use it in Stata/RStudio") is
        # only well-served if what lands on the clipboard is the
        # researcher's code, not Nora's plumbing. Detection is by
        # the unique separator line the executor writes
        # (``executor._write_script``); files that don't have it
        # (R scripts, researcher-uploaded scripts) pass through
        # unchanged.
        text = _strip_executor_preamble(text)
        return {
            "ok": True,
            "name": target.name,
            "kind": _classify_kind(ext),
            "text": text,
        }

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
        runner = self._active_runner()
        if runner is None:
            return {"ok": True, "name": Path(name).name, "removed": 0}
        target = Path(name).name  # basename only
        before = (
            len(runner.pending_script_attachments)
            + len(runner.pending_mentioned_files)
            + len(runner.pending_mentioned_images)
        )
        runner.pending_script_attachments = [
            a for a in runner.pending_script_attachments
            if a.get("name") != target
        ]
        runner.pending_mentioned_files = [
            n for n in runner.pending_mentioned_files if n != target
        ]
        runner.pending_mentioned_images = [
            a for a in runner.pending_mentioned_images
            if a.get("name") != target
        ]
        after = (
            len(runner.pending_script_attachments)
            + len(runner.pending_mentioned_files)
            + len(runner.pending_mentioned_images)
        )
        return {"ok": True, "name": target, "removed": before - after}

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
        # JS side only sends filenames from list_session_files /
        # list_mentionable_files.
        safe_name = Path(name).name
        cwd_resolved = self.cwd.resolve()
        candidate = (self.cwd / safe_name).resolve()
        target: Path | None = None
        if _is_within(candidate, cwd_resolved) and candidate.is_file():
            target = candidate
        else:
            # Fall through to the helper-plot dirs so an @-mention of
            # a plot like ``residuals_lm1.png`` (which lives in
            # ``.nora/runs/<id>/_nora_plots/``) resolves correctly.
            runs_root = self.cwd / ".nora" / "runs"
            if runs_root.is_dir():
                try:
                    for run_dir in runs_root.iterdir():
                        plots_dir = run_dir / "_nora_plots"
                        if not plots_dir.is_dir():
                            continue
                        nested = (plots_dir / safe_name).resolve()
                        if (
                            _is_within(nested, cwd_resolved)
                            and nested.is_file()
                        ):
                            target = nested
                            break
                except OSError:
                    pass
        if target is None:
            # Run-dir scripts: ``list_session_files`` surfaces the
            # ``script.{do,R,py}`` file at the root of each run dir
            # under a label-derived display name (e.g.
            # "Linear Regression Run.do"). The mention dropdown
            # offers those rows but their display names don't match
            # any on-disk basename in cwd or _nora_plots — that's
            # what made selecting a run-dir script fail with
            # "not found" even though it appears in the list. Resolve
            # the display name back through the same enumeration
            # the panel uses so the advertised recovery path
            # actually works.
            from nora.run_files import find_run_dir_script_by_name
            run_script = find_run_dir_script_by_name(self.cwd, safe_name)
            if (
                run_script is not None
                and _is_within(run_script, cwd_resolved)
                and run_script.is_file()
            ):
                target = run_script
        if target is None:
            return {"ok": False, "reason": f"not found: {safe_name}"}
        runner = self._active_runner()
        if runner is None:
            return {"ok": False, "reason": "no active session"}

        ext = target.suffix.lower()
        if ext in _INLINE_SCRIPT_EXTS:
            try:
                content = target.read_bytes()
            except OSError as e:
                return {"ok": False, "reason": f"read failed: {e}"}
            for staged in runner.pending_script_attachments:
                if staged.get("name") == safe_name:
                    return {
                        "ok": True,
                        "name": safe_name,
                        "kind": "script",
                        "already_attached": True,
                    }
            _stage_script_for_next_turn(
                runner.pending_script_attachments, safe_name, ext, content,
            )
            return {"ok": True, "name": safe_name, "kind": "script"}

        if ext in _MENTION_VISION_EXTS:
            blob_path = target
            mime = _MENTION_VISION_MIMES.get(ext)
            if ext in (".pdf", ".eps"):
                from nora.plot_convert import png_for
                sidecar = png_for(target)
                if sidecar is not None and sidecar.is_file():
                    blob_path = sidecar
                    mime = "image/png"
                else:
                    return _attach_as_announcement(
                        runner, safe_name, kind="graph",
                    )
            try:
                blob_size = blob_path.stat().st_size
            except OSError as e:
                return {"ok": False, "reason": f"stat failed: {e}"}
            if blob_size > _MENTION_VISION_MAX_BYTES:
                return {
                    "ok": False,
                    "reason": (
                        f"{safe_name} is {blob_size // (1024 * 1024)} MB, "
                        f"over the 5 MB vision limit. Reference it by "
                        f"name in your message and the model can read "
                        f"it from disk if needed."
                    ),
                }
            try:
                blob = blob_path.read_bytes()
            except OSError as e:
                return {"ok": False, "reason": f"read failed: {e}"}
            for staged in runner.pending_mentioned_images:
                if staged.get("name") == safe_name:
                    return {
                        "ok": True,
                        "name": safe_name,
                        "kind": "image",
                        "already_attached": True,
                    }
            import base64 as _b64
            runner.pending_mentioned_images.append({
                "data": _b64.b64encode(blob).decode("ascii"),
                "mime": mime or "image/png",
                "name": safe_name,
            })
            if safe_name not in runner.pending_mentioned_files:
                runner.pending_mentioned_files.append(safe_name)
            return {"ok": True, "name": safe_name, "kind": "image"}

        # Anything else: data files (.csv, .dta, .parquet, …),
        # logs (.log, .smcl), Stata graphs (.gph). Announce by name
        # only. The model already has dataset awareness via the
        # system prompt's listing (or the mid-session diff notice
        # for late additions); the mention notice just brings the
        # file to the foreground for THIS message.
        return _attach_as_announcement(
            runner, safe_name, kind=_classify_kind(ext),
        )

    def list_mentionable_files(self) -> dict[str, Any]:
        """Return every session-resident file the @-mention dropdown
        can offer, as a flat list with no thumbnails. Sister of
        :meth:`list_session_files` but lighter (no base64 image
        bytes; the dropdown only needs name + kind for filtering and
        rendering).

        The shape matches what the dropdown's filter/render code
        wants: ``[{name, kind, ext, mtime, size, path}]`` sorted by
        kind priority (data first, then scripts, graphs, logs) and
        mtime within each kind. Files in ``.nora/runs/<id>/_nora_plots/``
        are included so a researcher can mention helper-produced
        plots by name (``residuals_lm1.png``) the same way they'd
        mention a top-level upload.
        """
        if self.cwd is None:
            return {"ok": True, "files": []}
        listing = self.list_session_files()
        files: list[dict[str, Any]] = []
        for entry in listing.get("files", []):
            files.append({
                k: v for k, v in entry.items()
                if k not in ("data", "mime")
            })
        return {"ok": True, "files": files}

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
        # Close any IDLE runner that's bound to the now-unauthed
        # provider. We deliberately leave busy runners alone — their
        # turn will surface an auth_failure on the next request, but
        # interrupting an in-flight stream is worse than letting it
        # error out naturally. Closed runners reopen lazily on the
        # next send (and will see the unauthed provider then too).
        for runner in list(self._runners.values()):
            if runner.provider == provider and not runner.is_busy():
                self._run_on_loop(runner.close())
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
        """Set the bridge's default provider for newly-created runners,
        and swap the focused runner (if any) to that provider's
        default model.

        The auth-screen's "Use OpenAI / Use Anthropic" buttons call
        this. Only the focused runner is swapped; OTHER runners keep
        their existing provider/model so the user's prior choices
        for those sessions stick.
        """
        if provider not in PROVIDER_DEFAULTS:
            return {"ok": False, "reason": f"unknown provider: {provider!r}"}
        active = self._active_runner()
        if active is not None and active.is_busy():
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
        self._default_provider = provider
        self._default_model = PROVIDER_DEFAULTS[provider]
        if active is not None and active.provider != provider:
            self._run_on_loop(
                active.swap_model(self._default_model, provider)
            )
            self._persist_active_model()
        return {
            "ok": True,
            "provider": provider,
            "model": active.model if active is not None else self._default_model,
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
        # The "active provider" surfaced to the auth screen is the
        # focused runner's provider, falling back to the bridge
        # default for sessions not yet created.
        active = self._active_runner()
        active_provider = active.provider if active is not None else self._default_provider
        return {
            "providers": providers,
            "any_authed": any(v["configured"] for v in providers.values()),
            "active_provider": active_provider,
        }

    def set_model(self, model_id: str) -> dict[str, Any]:
        """Switch the focused session's active model.

        Operates on THIS session's runner only. Other sessions keep
        whatever model they're using — researcher with two open
        chats can run Sonnet in one and Opus in the other.

        Same-provider swaps go through the SDK in-place set_model
        (preserves conversation). Cross-provider swaps close and
        reopen the runner's session; conversation continuity flows
        through the bridge's first-turn context-prefix injection.

        Refused while THIS runner has a turn in flight (another
        session being busy is fine — we only check the focused
        runner). Persists ``active_model`` to the runner's
        ``.nora/session_state.json`` so a reload restores the
        choice.
        """
        try:
            new_provider = provider_for_model(model_id)
        except KeyError:
            return {"ok": False, "reason": f"unknown model: {model_id}"}
        info = next((m for m in ALL_MODELS if m.id == model_id), None)
        if info is None:
            return {"ok": False, "reason": f"unknown model: {model_id}"}

        active = self._active_runner()
        if active is None:
            # No focused session yet — update bridge defaults so the
            # next runner picks up the choice.
            self._default_provider = new_provider
            self._default_model = model_id
            return {
                "ok": True,
                "model": model_id,
                "label": info.label,
                "context_window": info.context_window,
                "provider": new_provider,
            }
        if active.is_busy():
            return {
                "ok": False,
                "reason": "a turn is in flight; wait for it to finish",
            }
        if model_id == active.model and new_provider == active.provider:
            return {"ok": True, "model": model_id, "unchanged": True}

        res = self._run_on_loop(active.swap_model(model_id, new_provider))
        if res is None or not res.get("ok"):
            return res or {"ok": False, "reason": "model switch failed"}
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
        researcher sends the first message in this session. Writes
        the focused runner's current model — per-session memory.
        """
        active = self._active_runner()
        if active is None:
            return
        try:
            from nora.session_state import write_session_state
            write_session_state(active.cwd, model=active.model)
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
            # Pull the researcher-set name (if any) so the sidebar can
            # show it as the primary label. ``title`` is the
            # already-resolved label that respects custom_name; the
            # raw ``custom_name`` lets the UI tell "user named this"
            # apart from "auto-derived from datasets" without having
            # to re-derive on the page side.
            try:
                from nora.session_state import read_session_state
                state = read_session_state(child)
            except Exception:  # noqa: BLE001
                state = None
            custom = state.custom_name if state is not None else None
            entries.append({
                "path": str(child.resolve()),
                "name": child.name,
                "timestamp": ts,  # epoch seconds, JS formats
                "datasets": datasets,
                "size": _dir_size(child),
                "title": _session_title(child),
                "custom_name": custom,
            })
        entries.sort(key=lambda e: e["timestamp"], reverse=True)
        return {"ok": True, "sessions": entries, "current": current}

    def delete_session(self, path: str) -> dict[str, Any]:
        """Delete a session directory and everything under it: data
        copies, run dirs, results.db, chat_history.jsonl. Only paths
        inside ``~/.nora-sessions/`` are allowed.

        Deleting the currently-focused session is supported: the
        bridge closes the active runner, drops ``self.cwd`` to
        ``None``, and returns ``was_active=True`` so the page can
        navigate back to the landing screen. (Without resetting
        ``self.cwd`` the bridge would keep handing out a path that
        no longer exists on disk, and the next ``ui_ready`` /
        ``policy_summary`` call would crash.)

        Refuses to delete a session whose runner has an in-flight
        turn — wait or interrupt first.
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
        was_active = bool(self.cwd) and target == self.cwd.resolve()
        # Refuse if the target's runner has a turn in flight. A
        # rmtree under a live SDK session and subprocess would yank
        # the cwd / run dirs / results.db out from under whatever's
        # still running — exactly the cross-session interference
        # the multi-runner refactor exists to prevent.
        runner_key = str(target)
        runner = self._runners.get(runner_key)
        if runner is not None and runner.is_busy():
            return {
                "ok": False,
                "reason": (
                    "this session has a turn in flight; wait for "
                    "it to finish (or interrupt it from the focused "
                    "session) before deleting"
                ),
            }
        # Idle runner: close its SDK session and drop the entry
        # before rmtree so we're not holding any handles into the
        # directory we're about to remove.
        if runner is not None:
            self._run_on_loop(runner.close())
            self._runners.pop(runner_key, None)
            try:
                from nora.store import close_store
                close_store(target)
            except Exception:  # noqa: BLE001 — store close isn't safety-critical
                pass
        try:
            shutil.rmtree(target)
        except OSError as e:
            return {"ok": False, "reason": f"delete failed: {e}"}
        # Drop the cached state-file lock so a long-running daemon
        # doesn't accumulate one ``threading.Lock`` entry per session
        # ever opened. Eviction is safe here: the runner above is
        # closed, no thread is mid-write on this cwd's state file.
        try:
            from nora.session_state import evict_state_lock
            evict_state_lock(target)
        except Exception:  # noqa: BLE001 — eviction isn't safety-critical
            pass
        # If we just deleted the focused session, drop the bridge's
        # reference to it. The page is responsible for navigating to
        # the landing screen on ``was_active=True``; until it does,
        # any policy / dataset query would otherwise read a
        # phantom path.
        if was_active:
            self.cwd = None
        return {"ok": True, "path": str(target), "was_active": was_active}

    def switch_session(self, path: str) -> dict[str, Any]:
        """Move UI focus to an existing Nora session.

        This is a pure focus change — does NOT close the previous
        session's runner, does NOT cancel any in-flight turn there.
        A regression streaming in session A keeps streaming after
        the researcher clicks B in the sidebar; events continue to
        be persisted to A's ``chat_history.jsonl`` and will be
        visible when they click back. (See SessionRunner for the
        execution model.)

        The new session's runner is lazy-created if it doesn't
        already exist, applying any per-session model preference
        recorded in ``.nora/session_state.json``.
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

        return self._set_cwd(target)

    def set_session_name(self, path: str, name: str) -> dict[str, Any]:
        """Persist a researcher-set label for a session.

        ``name`` is trimmed and capped; an empty (or whitespace-only)
        string clears the custom name and falls back to the
        auto-derived title (dataset filename / timestamp). The path
        must live under ``~/.nora-sessions/``.

        Returns the new resolved title so the page can update the
        topbar pill and sidebar row in one round trip.
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
        if not target.is_dir():
            return {"ok": False, "reason": f"not a directory: {target}"}
        try:
            from nora.session_state import set_custom_name
            set_custom_name(target, name)
        except Exception as e:  # noqa: BLE001 — surface, don't crash
            return {"ok": False, "reason": f"could not save name: {e}"}
        title = _session_title(target)
        return {
            "ok": True,
            "path": str(target),
            "title": title,
            "custom_name": (name.strip() or None) if isinstance(name, str) else None,
        }

    def interrupt_turn(self) -> dict[str, Any]:
        """Cancel the active session's in-flight turn, if any.

        Per-session: only the focused runner is interrupted. Other
        runners' turns keep running. We do NOT tear down the
        runner's session on cancel — the session stays open so the
        next turn reuses it. (Closing on every cancel was the old
        bug: it killed the whole conversation rather than just the
        current turn, and SDK retry semantics are robust enough to
        not need a fresh socket per attempt.)

        Returns ``{ok, turn_id}`` so the JS side can add ``turn_id``
        to its ``cancelledTurnIds`` set. That set is the authoritative
        drop list for late events: the runner stamps every event with
        its turn id, the dispatcher drops events whose id is in the
        runner's cancelled set, and the JS filter drops anything that
        slipped through. Returning the id here guarantees JS knows
        about the cancellation even if no ``activeLiveTurn`` was
        recorded yet (e.g., Stop fires immediately after Send, before
        any event lands).
        """
        if self._loop is None:
            return {"ok": False, "reason": "worker loop not running"}
        runner = self._active_runner()
        if runner is None or not runner.is_busy():
            return {"ok": False, "reason": "no turn in flight"}
        # ``cancel_turn`` synchronously marks the turn cancelled and
        # kills any registered subprocesses under the runner's lock.
        # The asyncio cancellation it then triggers needs to land on
        # the worker loop; ``cancel_turn`` itself does the
        # ``task.cancel`` call, but ``Task.cancel`` is loop-thread
        # safe in modern Python so we don't need ``call_soon_threadsafe``
        # here.
        turn_id = runner.cancel_turn()
        return {"ok": True, "turn_id": turn_id}

    # -------- internals --------

    def _set_cwd(self, path: Path) -> dict[str, Any]:
        """Set the focused session and ensure its runner exists.

        This is the pure UI-focus operation: pick the runner whose
        cwd matches ``path``, lazy-create it if absent, and return
        the ready payload so the page can render that session's
        topbar / policy / chat. We do NOT close any other runner —
        their turns keep running.
        """
        new_cwd = path.resolve()
        self.cwd = new_cwd
        # Lazy-create the runner for this cwd, applying any
        # recorded ``active_model`` preference. Existing runners
        # (including the one we may be switching AWAY from) are
        # untouched.
        self._ensure_runner_for_cwd(new_cwd)
        return {"ok": True, "state": "ready", **self._ready_payload()}

    def _ensure_runner_for_cwd(self, cwd: Path) -> SessionRunner:
        """Return the runner for ``cwd``, creating one if absent.

        On first creation, applies the session's recorded
        ``active_model`` (from ``.nora/session_state.json``) when
        the model is in the catalog and its provider is authed.
        Otherwise the bridge defaults are used.
        """
        key = str(cwd.resolve())
        runner = self._runners.get(key)
        if runner is not None:
            return runner
        provider, model = self._initial_model_for_session(cwd)
        runner = SessionRunner(cwd=cwd, provider=provider, model=model)
        self._runners[key] = runner
        return runner

    def _initial_model_for_session(
        self, cwd: Path,
    ) -> tuple[str, str]:
        """Pick (provider, model) for a freshly-created runner.

        Honours per-session memory recorded in
        ``.nora/session_state.json`` when the recorded model is
        still in the catalog and its provider is authed. Falls back
        to the bridge defaults otherwise.
        """
        try:
            from nora.session_state import read_session_state
            state = read_session_state(cwd)
        except Exception:  # noqa: BLE001
            state = None
        if state is None or not state.active_model:
            return self._default_provider, self._default_model
        try:
            from nora.provider.catalog import get_model
            info = get_model(state.active_model)
        except KeyError:
            return self._default_provider, self._default_model
        if info.provider not in self._authed_providers():
            return self._default_provider, self._default_model
        return info.provider, info.id

    def _active_runner(self) -> SessionRunner | None:
        """Convenience accessor: the runner for the focused session."""
        if self.cwd is None:
            return None
        return self._runners.get(str(self.cwd.resolve()))

    # Back-compat read-only views for tests and legacy callers that
    # treat the bridge as the single-session shape it used to be.
    # They reflect the FOCUSED runner — falling back to the bridge
    # default when no runner is created yet (landing screen).

    @property
    def _model(self) -> str:
        active = self._active_runner()
        return active.model if active is not None else self._default_model

    @_model.setter
    def _model(self, value: str) -> None:
        active = self._active_runner()
        if active is not None:
            active.model = value
        else:
            self._default_model = value

    @property
    def _provider(self) -> str:
        active = self._active_runner()
        return active.provider if active is not None else self._default_provider

    @_provider.setter
    def _provider(self, value: str) -> None:
        active = self._active_runner()
        if active is not None:
            active.provider = value
        else:
            self._default_provider = value

    @property
    def _pending_script_attachments(self) -> list[dict[str, Any]]:
        """Active runner's pending-attachments list. Returned by
        reference so legacy callers that ``.append`` to it still
        affect the runner's state. When no session is focused, a
        throwaway list is returned (mutations are silently
        discarded — there's no runner to attach to)."""
        active = self._active_runner()
        if active is None:
            return []
        return active.pending_script_attachments

    @property
    def _pending_mentioned_files(self) -> list[str]:
        """Active runner's @-mention announcement list. Same semantics
        as :attr:`_pending_script_attachments`: by-reference proxy
        for tests and any external introspection."""
        active = self._active_runner()
        if active is None:
            return []
        return active.pending_mentioned_files

    @property
    def _pending_mentioned_images(self) -> list[dict[str, Any]]:
        """Active runner's @-mention vision attachments."""
        active = self._active_runner()
        if active is None:
            return []
        return active.pending_mentioned_images

    @property
    def _session(self) -> Any:
        """Active runner's underlying provider session, or None.

        R/W for back-compat with tests that inject a fake session
        directly. Setting goes to the active runner; when no session
        is focused, the assignment is silently discarded."""
        active = self._active_runner()
        return active._session if active is not None else None

    @_session.setter
    def _session(self, value: Any) -> None:
        active = self._active_runner()
        if active is not None:
            active._session = value

    @property
    def _known_datasets(self) -> frozenset[str]:
        active = self._active_runner()
        return active.known_datasets if active is not None else frozenset()

    @_known_datasets.setter
    def _known_datasets(self, value: frozenset[str]) -> None:
        active = self._active_runner()
        if active is not None:
            active.known_datasets = value

    @property
    def _needs_context_prefix(self) -> bool:
        active = self._active_runner()
        # When no runner exists, the bridge has nothing to prefix
        # against — return False so legacy callers don't trip.
        return active.needs_context_prefix if active is not None else False

    @_needs_context_prefix.setter
    def _needs_context_prefix(self, value: bool) -> None:
        active = self._active_runner()
        if active is not None:
            active.needs_context_prefix = value

    def _run_on_loop(self, coro: Any, timeout: float = 5.0) -> Any:
        """Drive an awaitable on the worker loop and wait for it.

        Falls back to ``asyncio.run`` when the worker loop isn't
        started (tests, headless construction). The fallback is only
        appropriate for short, self-contained coroutines — not for
        the streaming turn loop, which always needs the worker
        thread.
        """
        if self._loop is None:
            try:
                return asyncio.run(coro)
            except Exception:  # noqa: BLE001
                return None
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return fut.result(timeout=timeout)
        except Exception:  # noqa: BLE001
            return None

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

    def _log_dispatch_diag(
        self, payload: dict[str, Any], plots: list[dict[str, Any]],
    ) -> None:
        """Emit a one-line diagnostic to stderr for every tool_result
        dispatch that included plot collection. Visible in the
        terminal where the researcher ran ``uv run nora`` so we
        can debug "I don't see thumbnails" claims without screen-
        sharing — the line tells us run_dir, session_cwd, and which
        plot files (if any) the collector found."""
        run_dir = payload.get("run_dir")
        if not run_dir:
            return
        names = [p.get("name", "?") for p in plots]
        sizes = [p.get("size", 0) for p in plots]
        has_data = sum(1 for p in plots if p.get("data"))
        print(
            f"[nora] tool_result dispatch  "
            f"run_dir={run_dir}  "
            f"session_cwd={payload.get('session_cwd')}  "
            f"plots_found={len(plots)}  "
            f"with_thumbnail_data={has_data}  "
            f"names={names}  "
            f"sizes={sizes}",
            file=sys.stderr, flush=True,
        )

    def _dispatch_event(self, payload: dict[str, Any]) -> None:
        """Route an event from a runner.

        Persistence ALWAYS lands in the runner's own
        ``chat_history.jsonl`` (keyed off ``payload['session_cwd']``)
        — this is what makes background sessions safe: a turn streaming
        in session A persists to A's log even while the UI is showing
        session B. The frontend is then responsible for filtering
        which events it renders based on the focused session.

        Cancellation drop. If the event carries a ``turn_id`` AND that
        id is in the originating runner's cancelled set, drop the
        event ENTIRELY: no persist, no JS dispatch. The runner stamps
        every event with its turn id; ``cancel_turn`` adds the id to
        the set the moment Stop fires; so any late event the SDK or
        a subprocess emits after the cancel reaches us here and
        terminates without polluting chat history or the rendered
        transcript. This is the backend boundary the user-pressed-Stop
        contract leans on; the JS-side filter is best-effort defense
        in depth, not the authoritative drop.

        For tool-result events we enrich the payload with the raw
        stdout/stderr from the run dir so the JS can render the
        native R/Stata/Python output panel. The enrichment runs here
        rather than in the runner because it's a UI concern (the
        researcher sees raw logs; the model sees only the sanitized
        payload).
        """
        turn_id = payload.get("turn_id")
        cwd_str = payload.get("session_cwd")
        if turn_id and cwd_str:
            runner = self._runners.get(str(Path(cwd_str).resolve()))
            if runner is not None and runner.is_turn_cancelled(turn_id):
                return
        if payload.get("type") == "tool_result":
            run_dir = payload.get("run_dir")
            if run_dir:
                raw_stdout, raw_stderr = _read_raw_logs(run_dir)
                # Researcher-side plot thumbnails: scan run_dir AND
                # the originating runner's cwd for .png files the
                # script produced. Stata's executor preamble ``cd``s
                # to the session cwd before user code runs, so a
                # bare ``graph export "fig.png"`` lands in the
                # session cwd — without scanning there too,
                # Stata-generated plots never appeared as thumbnails.
                plots = _collect_run_dir_plots(
                    run_dir, session_cwd=payload.get("session_cwd"),
                )
                self._log_dispatch_diag(payload, plots)
                # Diagnostic: when no plots came back AND the helper
                # left telltale traces (mkdir of _nora_plots/, stderr
                # line starting with ``nora.plot_*``), surface a
                # one-liner so the researcher doesn't stare at a blank
                # thumbnail row wondering whether the helper ran at all.
                diagnostic = _detect_plot_helper_diagnostics(run_dir, len(plots))
                payload = {
                    **payload,
                    "raw_stdout": raw_stdout,
                    "raw_stderr": raw_stderr,
                    "plots": plots,
                }
                if diagnostic:
                    payload["plot_diagnostic"] = diagnostic
        self._persist_event(payload)
        if self._window is None:
            return
        js = f"window.nora_event({json.dumps(payload)});"
        try:
            self._window.evaluate_js(js)
        except Exception:  # noqa: BLE001 — webview may be closing
            pass

    # Event types we keep in the chat log. ``turn_done`` is included
    # so post-hoc diagnostics (cache hit rate, per-turn input/output
    # tokens, cost) can be inspected from the persisted log; the
    # turn-grouped reader in ``chat_history.read_turns`` ignores
    # unknown types so this is additive. Everything else is either
    # transient (auth_failure) or reconstructible from session state
    # (ready, policy_updated).
    _PERSIST_TYPES = frozenset({
        "assistant_text",
        "assistant_thinking",
        "tool_call",
        "tool_result",
        "turn_done",
        "user_message",
    })

    def _record_user_message(
        self,
        runner: SessionRunner,
        text: str,
        *,
        image_count: int = 0,
    ) -> None:
        """Persist the user-side record for a newly queued turn.

        Bound to the *runner's* cwd, not the bridge's focus, so a
        send issued while another session is being viewed lands in
        the right session's history. (Today the bridge schedules
        sends only on the focused runner, but stamping by runner
        future-proofs against UI changes that allow background
        sends.)

        Before appending, drops any trailing orphaned ``user_message``
        from a previous failed / unsent turn so retries replace
        rather than accumulate stale "no-reply" bubbles.
        """
        self._drop_trailing_orphan_user_message(runner.cwd)
        attached_names = [
            a["name"] for a in runner.pending_script_attachments
        ]
        record: dict[str, Any] = {
            "type": "user_message",
            "text": text,
            "session_cwd": str(runner.cwd),
        }
        if attached_names:
            record["attachments"] = attached_names
        if image_count > 0:
            record["image_count"] = image_count
        self._persist_event(record)

    def _drop_trailing_orphan_user_message(self, cwd: Path) -> None:
        """Remove the last persisted record iff it is a bare
        ``user_message`` with no assistant/tool events after it.
        Operates on a specific session's history (passed in) rather
        than the bridge's focus, so concurrent retries on different
        runners don't clobber each other's logs.
        """
        path = cwd / ".nora" / "chat_history.jsonl"
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
        """Persist a transcript-forming event to the session's
        ``chat_history.jsonl``.

        The session is identified by ``payload['session_cwd']``, NOT
        by the bridge's focus. This is the routing rule that makes
        concurrent sessions safe: a tool_result from runner A's
        in-flight turn lands in A's history even while the UI is
        showing B. Falls back to the bridge focus only if the event
        carries no session_cwd (legacy paths, defensive).
        """
        etype = payload.get("type")
        if etype not in self._PERSIST_TYPES:
            return
        target_cwd_str = payload.get("session_cwd")
        if target_cwd_str:
            target_cwd: Path | None = Path(target_cwd_str)
        else:
            target_cwd = self.cwd
        if target_cwd is None:
            return
        # Strip session_cwd before writing — it's a routing
        # annotation, not part of the persisted record. (Legacy
        # readers don't expect the field; keeping it would noisily
        # appear in transcripts.) We don't mutate the caller's
        # dict — a shallow copy is cheap and avoids surprising
        # _dispatch_event consumers that keep the original reference.
        record = {k: v for k, v in payload.items() if k != "session_cwd"}
        record.setdefault(
            "timestamp",
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        try:
            history_dir = target_cwd / ".nora"
            history_dir.mkdir(parents=True, exist_ok=True)
            path = history_dir / "chat_history.jsonl"
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            # Persistence failing shouldn't block live chat. The
            # transcript on screen stays intact; replay just won't
            # include this event.
            pass

    def rewind_to(self, turn_index: int) -> dict[str, Any]:
        """Truncate this session's chat history at the ``turn_index``-th
        user message, hide every result row no longer referenced in
        the kept prefix, clear pending attachments, and drop the
        provider session — but do NOT send a new message.

        After this returns ok, the JS side calls ``get_chat_history()``
        to re-render the trimmed transcript and then ``send_message``
        as usual to fire the revised turn. Splitting the operation in
        two avoids a double-render race: if the bridge fired the new
        send itself, the live ``user_message`` event would land
        alongside the same record JS just pulled out of the truncated
        history.

        Returns ``{ok, truncated_from_index, hidden_count}`` on
        success. Refused (``ok=false``) when:
          - no active session,
          - the active runner is busy (researcher must Stop first),
          - ``turn_index`` is out of range,
          - the chat-history file can't be read or written.

        Stored result rows from the dropped branch stay in SQLite
        for audit but are stamped with ``hidden_at`` so the warm-
        start prefix, ``list_results`` / ``list_results_global``
        tools, and ``expand_result`` no longer surface them to the
        model.

        Crash safety: SQL hide commits before file truncate. If the
        process dies between, history still references hidden rows
        but ``expand_result`` returns ``not_found`` — the model
        handles that gracefully on retry. The reverse order would
        leave model-visible rows that the audit position no longer
        points to.
        """
        runner = self._active_runner()
        if runner is None or self.cwd is None:
            return {"ok": False, "reason": "no active session"}
        if runner.is_busy():
            return {
                "ok": False,
                "reason": (
                    "a turn is still running on this session — stop "
                    "it first, then try the edit again"
                ),
            }
        if not isinstance(turn_index, int) or turn_index < 0:
            return {
                "ok": False,
                "reason": f"turn_index must be a non-negative int, got {turn_index!r}",
            }

        history_path = self.cwd / ".nora" / "chat_history.jsonl"
        if not history_path.exists():
            return {
                "ok": False,
                "reason": "no chat history to rewind",
            }

        # 1. Locate the byte offset of the N-th user_message. Binary
        # mode so the offset matches what we'll truncate at — text-
        # mode reads on a UTF-8 file with non-ASCII user messages
        # would mis-account for multi-byte characters and produce a
        # corrupt file on truncate.
        offset = _find_user_message_offset(history_path, turn_index)
        if offset is None:
            return {
                "ok": False,
                "reason": f"no user message at index {turn_index}",
            }

        # 2. Collect every ``result_id`` referenced in the kept
        # prefix. Everything not in this set will be hidden.
        kept_ids = _result_ids_in_history_prefix(history_path, offset)

        # 3. SQL hide first (crash safety — see docstring).
        # ``get_store`` is imported locally to match the pattern used
        # elsewhere in this module (``_build_context_prefix``,
        # ``close_store``); the store package isn't pulled into module
        # scope because most ui.py methods don't touch it, and a top-
        # level import would resurrect a circular-import risk that
        # the local-import pattern was set up to avoid.
        try:
            from nora.store import get_store
            store = get_store(self.cwd)
            hidden_ids = store.hide_results_not_in(kept_ids, reason="rewind")
        except Exception as e:  # noqa: BLE001 — surface store errors to JS
            return {
                "ok": False,
                "reason": f"could not update result store: {e}",
            }

        # 4. Truncate the chat-history file at the offset.
        # On failure, roll back the hide. Without this, the bridge
        # returns ok=false to JS while the store has already mutated
        # — the model would see different result visibility than the
        # chat history reflects, and the researcher's UI would say
        # the rewind failed even as the model's next turn surfaced
        # the half-applied state. Crash safety from the docstring is
        # preserved: a process death between hide and truncate still
        # leaves the documented "hidden rows + intact history" state;
        # only the explicit-failure path rolls back.
        try:
            _truncate_at_offset(history_path, offset)
        except OSError as e:
            try:
                store.unhide_results(hidden_ids)
            except Exception:  # noqa: BLE001 — best-effort rollback
                pass
            return {
                "ok": False,
                "reason": f"could not truncate chat history: {e}",
            }
        hidden_count = len(hidden_ids)

        # 5. Clear runner pending state. Anything queued for the
        # original next turn is no longer relevant — the rewind
        # creates a new branch.
        runner.clear_pending_attachments()

        # 6. Drop the provider session so the next ``send_message``
        # opens a fresh one against the truncated history; flag
        # ``needs_context_prefix`` so the warm-start replay rebuilds
        # the model's view from what's left in chat_history.jsonl.
        if self._loop is not None:
            try:
                fut = asyncio.run_coroutine_threadsafe(
                    runner.close(), self._loop,
                )
                fut.result(timeout=5.0)
            except Exception:  # noqa: BLE001 — close errors aren't fatal here
                pass
        runner.needs_context_prefix = True

        return {
            "ok": True,
            "truncated_from_index": turn_index,
            "hidden_count": hidden_count,
        }

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

    def count_next_context(
        self,
        draft_text: str = "",
        n_images: int = 0,
        n_pending_attachments: int = 0,
        request_id: int = 0,
    ) -> dict[str, Any]:
        """Pre-flight count of the next request's size, in tokens.

        Single source of truth for the context chip. JS calls this
        on a small set of triggers (session open/switch, rewind
        success, turn complete, attachment add/remove) and renders
        the returned ``tokens / ceiling`` directly. The chip never
        derives its own number from ``turn_done`` usage,
        ``post_turn_tokens``, cache fields, or chars/4 of pending
        messages — those produced visible fluctuation that didn't
        correspond to any single useful question.

        ``request_id`` rides through the response so JS can reject
        responses landing after a newer request — recounts triggered
        in quick succession (e.g., model swap during a rewind) won't
        let an old draft's count overwrite a newer one.

        Returns ``{ok, tokens, exact, ceiling, request_id}``. Today
        ``exact=False`` for both providers (chars/3.5 approximation).
        The chip text itself shows the bare number — no ``~`` prefix
        — and the chip's hover tooltip carries the
        approximate-vs-exact disclosure so the researcher never
        mistakes an estimate for a measurement (see
        ``web/app.js`` ``updateContextChip`` for the rendering
        contract).
        """
        if self.cwd is None:
            return {
                "ok": False,
                "reason": "no active session",
                "request_id": request_id,
            }

        from nora.context_count import count_next_context, to_payload
        from nora.system_prompt import build_system_prompt

        # Provider-specific lengths. Building the system prompt is
        # the one expensive piece (a few hundred KB of template +
        # dataset listing + runtime probe), but it's cached at
        # session level and the call is cheap on warm runs.
        provider = self._provider_id() if hasattr(self, "_provider_id") else "anthropic"
        try:
            sys_prompt = build_system_prompt(
                self.cwd, "nora", provider=provider,
            )
            system_prompt_chars = len(sys_prompt)
        except Exception:  # noqa: BLE001
            system_prompt_chars = 0

        # Tool schemas: rough estimate. The full JSON-rendered tool
        # array sits at ~14k tokens (~50KB) on Anthropic with the
        # current set; on OpenAI the description bodies are leaner.
        # Using a single value matters less than the chat history
        # bytes (which dominate after a few turns) and the
        # caller-provided draft / image counts; refine later.
        tool_schema_chars = 50_000

        ceiling = self._context_ceiling_for_active_model()

        # Override the JS-supplied attachment count with the runner's
        # actual staging list, and compute the inlined content bytes
        # that ``_build_script_attachment_prefix`` will emit on the
        # next send. Without this the chip stayed flat when a
        # researcher attached a 90 KB ``.do`` / ``.py`` file, even
        # though those bytes will ride into the next request.
        #
        # Same posture for pending vision images: ``run_turn``
        # auto-merges the runner's ``pending_plot_images`` (up to
        # eight result plots from the previous script) and
        # ``pending_mentioned_images`` (researcher @-mentions) into
        # the next provider request. JS only sees its OWN composer-
        # staged image count, so without adding the runner-side
        # totals here the chip would recount as if no images were
        # pending right after a script emitted plots — even though
        # the next send would silently attach them.
        active = self._active_runner()
        if active is not None:
            attachments = active.pending_script_attachments
            n_pending_attachments = len(attachments)
            pending_attachment_chars = _sum_inline_attachment_chars(attachments)
            n_images = (
                n_images
                + len(active.pending_plot_images)
                + len(active.pending_mentioned_images)
            )
        else:
            pending_attachment_chars = 0

        count = count_next_context(
            cwd=self.cwd,
            draft_text=draft_text,
            n_images=n_images,
            n_pending_attachments=n_pending_attachments,
            pending_attachment_chars=pending_attachment_chars,
            system_prompt_chars=system_prompt_chars,
            tool_schema_chars=tool_schema_chars,
            ceiling=ceiling,
            request_id=request_id,
        )
        return {"ok": True, **to_payload(count)}

    def _context_ceiling_for_active_model(self) -> int:
        """Resolve the active model's context window in tokens.

        Falls back to 1M when the runner / model registry can't be
        consulted — every frontier model on both providers is in
        the 1M-ish band, so a missing-ceiling default of 1M is
        honest rather than a placeholder.
        """
        try:
            from nora.provider.catalog import ALL_MODELS
            runner = self._active_runner()
            if runner is None:
                return 1_000_000
            model = getattr(runner, "model", None)
            if model:
                for info in ALL_MODELS:
                    if info.id == model:
                        return info.context_window
        except Exception:  # noqa: BLE001
            pass
        return 1_000_000

    def _provider_id(self) -> str:
        """Active provider name ('anthropic' / 'openai'), defaulting
        to 'anthropic' when the runner can't be consulted."""
        try:
            runner = self._active_runner()
            if runner is None:
                return "anthropic"
            return getattr(runner, "provider", "anthropic") or "anthropic"
        except Exception:  # noqa: BLE001
            return "anthropic"

    def _policy_summary(self) -> dict[str, Any]:
        """Compact JSON-serializable summary of the current policy +
        dataset list for the topbar footer."""
        if self.cwd is None:
            from nora.policy import DEFAULT_MAX_DEPTH
            return {"default_max_depth": DEFAULT_MAX_DEPTH, "datasets": []}
        from nora.system_prompt import scan_datasets as _scan_datasets
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


def _find_user_message_offset(
    history_path: Path, turn_index: int,
) -> int | None:
    """Return the byte offset of the ``turn_index``-th ``user_message``
    line in ``history_path``, or ``None`` if there are fewer
    user_messages than that.

    Walks the file in binary mode so the returned offset matches what
    ``os.truncate`` would cut at — even with multi-byte UTF-8 user
    messages. ``turn_index`` is 0-based; passing ``0`` returns the
    offset of the very first user message.

    Best-effort on parse errors: a malformed line is treated as "not
    a user_message" rather than aborting the scan, so a single
    corrupted record can't make a rewind unreachable.
    """
    if turn_index < 0:
        return None
    seen = 0
    offset = 0
    try:
        with history_path.open("rb") as f:
            for raw in f:
                line_start = offset
                offset += len(raw)
                # Strip newline + whitespace before parse. The trailing
                # ``\n`` byte is included in ``len(raw)`` so ``offset``
                # correctly points at the START of the next line.
                stripped = raw.strip()
                if not stripped:
                    continue
                try:
                    rec = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                if not isinstance(rec, dict):
                    continue
                if rec.get("type") != "user_message":
                    continue
                if seen == turn_index:
                    return line_start
                seen += 1
    except OSError:
        return None
    return None


def _result_ids_in_history_prefix(
    history_path: Path, offset: int,
) -> set[str]:
    """Collect every ``result_id`` referenced in the chat-history
    bytes BEFORE ``offset``.

    The kept prefix is what survives a rewind; this set is the model-
    visible result_ids the rewind must preserve. Everything not in
    this set gets hidden from the store on the rewind path.

    Reads tool_result events (the canonical site for result_ids,
    via :func:`nora.chat_history._extract_result_ids`). Tool_call
    audit events that pre-quote an id in their input are not a
    primary source — the matching tool_result will carry the same
    id when present — but we tolerate them defensively.
    """
    from nora.chat_history import _extract_result_ids

    kept: set[str] = set()
    if offset <= 0:
        return kept
    try:
        with history_path.open("rb") as f:
            chunk = f.read(offset)
    except OSError:
        return kept
    for raw in chunk.splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        try:
            rec = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if not isinstance(rec, dict):
            continue
        if rec.get("type") == "tool_result":
            for rid in _extract_result_ids(rec.get("text", "") or ""):
                kept.add(rid)
    return kept


def _truncate_at_offset(history_path: Path, offset: int) -> None:
    """Truncate ``history_path`` to exactly ``offset`` bytes.

    Wraps ``os.truncate`` so the caller doesn't have to manage a file
    handle. The offset must come from
    :func:`_find_user_message_offset` so it lands at a record
    boundary; truncating mid-line would leave a malformed JSONL line
    that the next ``read_turns`` would skip with a parse error
    (recoverable, but we'd rather not).
    """
    import os
    os.truncate(str(history_path), offset)


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

# Vision-eligible mention attachments. When the researcher
# @-mentions one of these, the bytes ride the next turn so the
# model can actually see the image (and not just be told a file
# named "residuals.png" exists). PDF / EPS get raster-converted
# via plot_convert.png_for first.
_MENTION_VISION_MIMES: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".pdf": "image/png",   # converted via sips → PNG sidecar
    ".eps": "image/png",   # ditto
}
_MENTION_VISION_EXTS: frozenset[str] = frozenset(_MENTION_VISION_MIMES)

# 5 MB hard cap on a single mentioned image. Matches the composer
# drop limit. Above this the model would either reject the request
# (Anthropic) or take a long time to ingest (OpenAI), and the
# researcher's intent ("look at this plot") is better served by
# pointing them at a smaller export.
_MENTION_VISION_MAX_BYTES = 5 * 1024 * 1024


# Per-file cap on JS drag-drop / paste uploads (data and script files,
# not images — those have their own 5 MB cap upstream). The chain is
# FileReader → base64 string → pywebview bridge → ``b64decode``, with
# peak memory roughly 3–4× the file size while the encoded string and
# decoded bytes both live on the heap. 512 MB peaks around 2 GB total,
# which is comfortable on any modern Mac without swap pressure.
#
# Files larger than this should use the native picker
# (:meth:`Bridge.choose_files` / :meth:`Bridge.add_files`), which uses
# ``shutil.copy2`` and has no size limit. The frontend gates on
# ``file.size`` before calling FileReader; this constant is the
# backend's matching defense-in-depth check, in case a client bypasses
# the JS gate or sends a forged base64 string from a non-browser path.
_DRAG_DROP_MAX_BYTES = 512 * 1024 * 1024


def _b64_oversize(content_b64: str, max_decoded_bytes: int) -> bool:
    """Return True iff a base64 string would decode to more than
    ``max_decoded_bytes``. Cheap arithmetic — no decoding.

    base64 encodes every 3 input bytes as 4 output chars with up to
    2 trailing ``=`` pad chars; each pad char represents zero bytes.
    Decoded length is ``len(b64) * 3 // 4`` minus the padding count.
    Subtracting pad makes the boundary exact at the byte — a 1 MB
    file encoded into a ~1.33 MB string compares as exactly 1 MB,
    not 1 MB + 2 bytes. Saves materializing a multi-GB decoded blob
    just to measure it.
    """
    n = len(content_b64)
    if n == 0:
        return 0 > max_decoded_bytes
    pad = 0
    if content_b64[-1] == "=":
        pad += 1
        if n >= 2 and content_b64[-2] == "=":
            pad += 1
    return (n * 3 // 4 - pad) > max_decoded_bytes


def _drag_drop_oversize_message(
    name: str, mb: int, picker_label: str,
) -> str:
    """Format the rejection message for drag-drop oversize. The JS
    side has its own copy of the same wording — keep them aligned so
    a researcher who hits one path sees a recognizable error if they
    later hit the other."""
    return (
        f"{name!r} is {mb} MB — drag-drop is capped at "
        f"{_DRAG_DROP_MAX_BYTES // (1024 * 1024)} MB because the "
        f"file is read fully into memory (peak ~3–4× the file size). "
        f"Use {picker_label} instead — it copies directly from disk "
        f"with no memory overhead and no size limit."
    )


def _classify_kind(ext: str) -> str:
    """Map a file extension to the same ``kind`` string
    :meth:`Bridge.list_session_files` uses, so the JS chip-renderer
    only has one vocabulary to track. Unknown extensions fall through
    to ``"data"`` so the chat-bubble renderer always has a kind."""
    from nora.session_files import classify_ext
    return classify_ext(ext, include_data=True, default="data")


# Exact marker lines the executor writes between its bootstrap and
# the researcher's code (see ``executor._write_script``). Stata uses
# ``*!`` and Python uses ``#`` as the comment prefix; both are
# anchored to start-of-line and verbose enough that no researcher
# would write either form by accident.
_EXECUTOR_PREAMBLE_MARKERS = (
    "*! ----- Nora preamble above; researcher code below -----",
    "# ----- Nora preamble above; researcher code below -----",
)


def _strip_executor_preamble(text: str) -> str:
    """Drop the executor's bootstrap from a Stata / Python ``script.do``
    or ``script.py`` so the body the researcher gets on their
    clipboard is portable.

    The on-disk script that the runner executes opens with
    ``adopath +`` / ``cd`` (Stata) or a ``sys.path.insert`` (Python)
    that resolves against the run dir's ``lib/`` and the per-session
    ``NORA_CWD`` / ``NORA_LIB_DIR`` env vars. Outside Nora those names
    don't exist, so a copied raw file fails on the first line. The
    stated purpose of the Files-panel "copy" button is "grab the
    script and use it in Stata/RStudio," which only works if the
    bootstrap is gone.

    Detection is by the exact marker line the executor writes between
    bootstrap and user code, anchored to the start of a line. If
    none of the markers match (R scripts have no preamble;
    researcher-uploaded ``.py`` / ``.do`` files won't either), the
    text is returned unchanged.
    """
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if line in _EXECUTOR_PREAMBLE_MARKERS:
            # Drop everything up through this marker line, plus a
            # single trailing blank that the executor pads in for
            # readability. ``"\n".join`` reconstructs the rest with
            # original line endings preserved.
            rest = lines[i + 1:]
            if rest and rest[0] == "":
                rest = rest[1:]
            return "\n".join(rest)
    return text


def _attach_as_announcement(
    runner: Any, name: str, kind: str,
) -> dict[str, Any]:
    """Append ``name`` to the runner's mention-notice list (idempotent)
    and return the bridge-shape success payload. The file's bytes
    don't ride along; the model picks it up from disk on the next
    turn via ``get_schema`` / ``expand_result`` / direct read."""
    if name in runner.pending_mentioned_files:
        return {
            "ok": True,
            "name": name,
            "kind": kind,
            "already_attached": True,
        }
    runner.pending_mentioned_files.append(name)
    return {"ok": True, "name": name, "kind": kind}


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


def _sum_inline_attachment_chars(
    attachments: list[dict[str, Any]],
) -> int:
    """Estimate of inlined char count for the next-turn prefix.

    Mirrors the truncation logic in :func:`_build_script_attachment_prefix`
    so the context chip's count tracks what the prefix actually emits
    rather than just the number of attachments. Without this the chip
    stayed flat when a researcher attached a 90 KB script.

    Approximation deliberately chosen over re-rendering the full
    prefix string: the chip is recounted on every keystroke that
    triggers a recount, and we don't want to allocate tens of KB of
    formatted text just to measure its length.
    """
    if not attachments:
        return 0
    used = 0
    for att in attachments:
        # ~120 bytes covers the ``\n### name (Lang)\n``` ... ``` \n``
        # framing _build_script_attachment_prefix wraps each block
        # in. The exact number doesn't matter — content dominates.
        block_size = len(att.get("content", "") or "") + 120
        if used + block_size > _INLINE_SCRIPT_TOTAL_CAP:
            # Past this, _build_script_attachment_prefix emits an
            # "omitted" line per remaining attachment instead of the
            # content, so the chars stop climbing in proportion to
            # file size. Stop counting once we hit the cap.
            break
        used += block_size
    return used


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
    from nora.text_safety import safe_text
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
        # Sanitize the display name before interpolating it into a
        # markdown heading. Filenames on macOS / Linux can contain
        # newlines, bidi/control characters, and markdown syntax;
        # without this, a hostile filename can break out of the
        # intended ``### name (Lang)`` line and inject prompt
        # instructions that ride above the researcher's message.
        # ``safe_text`` flattens whitespace (so newlines become
        # spaces), strips bidi/zero-width chars, and caps length —
        # matching the boundary every other data-origin string
        # crossing to Claude already respects. The on-disk basename
        # in ``att["name"]`` is preserved unchanged for any actual
        # file lookups; this sanitization is purely for the prompt
        # rendering surface.
        display_name = safe_text(
            name if isinstance(name, str) else str(name),
            max_len=120,
        ) or "(attached file)"
        rel_path = display_name
        if cwd is not None and isinstance(name, str):
            try:
                # Resolve the on-disk relative path off the REAL
                # basename, then sanitize the result. Falls back to
                # the sanitized display name on error.
                resolved = str((cwd / name).relative_to(cwd))
                rel_path = safe_text(resolved, max_len=120) or display_name
            except (ValueError, OSError):
                rel_path = display_name
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
      0. A researcher-set ``custom_name`` from session_state.json.
      1. A single dataset's filename (most common case: one upload).
      2. "<first> +N more" when multiple datasets live in the dir.
      3. A "Session MMM DD, HH:MM" stamp derived from the dir name.
      4. The dir's basename as a last-resort fallback.
    The goal is that the topbar always shows something a researcher
    recognizes, never a raw absolute path.
    """
    try:
        from nora.session_state import read_session_state
        state = read_session_state(cwd)
        if state is not None and state.custom_name:
            return state.custom_name
    except Exception:  # noqa: BLE001 — never let title resolution crash the UI
        pass
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


def _materialize_cache_busted_index(web_dir: Path, index_path: Path) -> Path:
    """Write a sibling ``index.bust-<id>.html`` next to the source
    index whose script/link refs carry a per-launch ``?v=<build-id>``
    query string. WKWebView caches file:// resources by full URL
    (including the query string), so a unique build-id per launch
    forces a fresh fetch of every JS/CSS asset and prevents
    "I restarted but the new code isn't running" — a real failure
    mode where users iterate on the frontend, restart the bridge,
    and still see stale rendering because WKWebView served the
    cached app.js.

    The build-id hashes the mtimes of every .js / .css / .html file
    under ``web_dir`` so logically-identical reloads reuse the same
    cache key, while real code changes invalidate it.

    Falls back to the original index_path on any error — cache busting
    is a polish feature, not a correctness one.
    """
    import hashlib
    try:
        # Wipe stale ``.index.bust-*.html`` siblings before generating
        # a fresh one. Without this, every launch leaves a new sibling
        # in ``web/`` and the directory accumulates indefinitely
        # (each iteration on app.js / style.css produced one). Also
        # avoids confusion when reading mtimes during debugging — only
        # the live bust-file should be present after startup.
        for stale in web_dir.glob(".index.bust-*.html"):
            try:
                stale.unlink()
            except OSError:
                # Best-effort cleanup — don't crash the launch if a
                # sibling is locked / read-only / already gone.
                continue
        stamps: list[str] = []
        for child in sorted(web_dir.iterdir()):
            if child.suffix.lower() in {".js", ".css", ".html"}:
                try:
                    stamps.append(f"{child.name}:{child.stat().st_mtime_ns}")
                except OSError:
                    continue
        if not stamps:
            return index_path
        build_id = hashlib.sha256("\n".join(stamps).encode()).hexdigest()[:12]
        html = index_path.read_text(encoding="utf-8")
        # Append ``?v=<build-id>`` to script/link refs. We rewrite
        # only the local refs (no protocol) so the Google Fonts
        # preconnects and any future remote CDN loads stay alone.
        import re
        def _add_bust(m: re.Match[str]) -> str:
            attr = m.group(1)
            url = m.group(2)
            if "://" in url or url.startswith("//"):
                return m.group(0)
            sep = "&" if "?" in url else "?"
            return f'{attr}="{url}{sep}v={build_id}"'
        html = re.sub(
            r'(src|href)="([^"]+\.(?:js|css))"',
            _add_bust, html,
        )
        # Stash the rewritten file beside the original. .gitignore'd
        # via the leading dot so accidental git status noise stays
        # out of the working tree.
        out = web_dir / f".index.bust-{build_id}.html"
        out.write_text(html, encoding="utf-8")
        return out
    except Exception:  # noqa: BLE001 — fall back to source index on any failure
        return index_path


# Researcher-side plot rendering — caps + extensions
_RESEARCHER_PLOT_MAX_BYTES = 3 * 1024 * 1024  # 3 MB / image inline so 1600px PNGs render sharply on retina; larger get a path-only entry
_RESEARCHER_PLOT_MAX_PER_RESULT = 6
# Includes ``.pdf`` so Stata's PDF fallback (when Graph2png is missing)
# still produces a chat thumbnail. PDFs are converted to PNG sidecars
# at collect time via macOS ``sips`` — see ``nora.plot_convert``.
_RESEARCHER_PLOT_EXTS: tuple[str, ...] = (
    ".png", ".jpg", ".jpeg", ".pdf", ".eps",
)


def _detect_plot_helper_diagnostics(
    run_dir: str | None, n_plots_found: int,
) -> str | None:
    """Look for evidence that a plot helper was CALLED but produced
    no usable output. Three signals:

    - ``_nora_plots/`` exists in the run dir (a helper ran the
      mkdir at the top of its body)
    - The plot file count is 0 (or the manifest is empty)
    - stderr.log contains a ``nora.plot_*`` / ``nora$plot_*`` line

    When all three line up, we surface a one-line note in the tool
    result card so the researcher doesn't stare at an empty
    thumbnail row wondering "did the helper even run?".

    Returns a short human-readable string or None.
    """
    if not run_dir:
        return None
    base = Path(run_dir)
    plots_dir = base / "_nora_plots"
    if not plots_dir.is_dir():
        return None
    if n_plots_found > 0:
        return None
    # Empty plots dir + helper trace in stderr → matplotlib missing
    # is the overwhelmingly common cause. Surface that hypothesis
    # explicitly so the researcher knows what to install.
    stderr_path = base / "stderr.log"
    helper_lines: list[str] = []
    if stderr_path.is_file():
        try:
            stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            stderr_text = ""
        for line in stderr_text.splitlines():
            stripped = line.strip()
            if (
                stripped.startswith("nora.plot_")
                or stripped.startswith("nora$plot_")
                or "nora_plot_" in stripped
            ):
                helper_lines.append(stripped)
    if not helper_lines:
        return None
    # Take the first helper-failure line — usually the most
    # informative; later lines tend to be Python tracebacks the
    # researcher can find via Show Folder if needed.
    first = helper_lines[0]
    if "matplotlib" in first.lower() or "no module named 'matplotlib'" in first.lower():
        return (
            "Plot helper called but matplotlib isn't installed in this "
            "Python environment. Install with `pip install matplotlib` "
            "and re-run the script."
        )
    return f"Plot helper called but produced no output: {first[:160]}"


def _collect_run_dir_plots(
    run_dir: str | None,
    session_cwd: str | None = None,
) -> list[dict[str, Any]]:
    """Find image files produced by a script run and return them as
    a JSON-friendly list ready to embed in a tool_result event.

    Three locations are scanned, in priority order:

    1. ``<run_dir>/_nora_plots/`` — the manifest-allowlisted
       location used by ``nora.plot_residuals`` / etc. (model-
       visible plots; surfaced to the researcher too).
    2. ``<run_dir>/`` — Python / R scripts whose subprocess cwd is
       run_dir write direct ``plt.savefig`` / ``ggsave`` outputs
       here (when bare filenames are used).
    3. ``<session_cwd>/`` — Stata scripts ``cd`` into the session
       cwd via the executor preamble (so the batch ``.log`` lands
       outside the project), so a bare ``graph export "fig.png"``
       writes the plot file into the session cwd, not the run dir.
       Without scanning here, Stata-generated plots never appeared
       as thumbnails — that was the "stata still not working" bug.
       To avoid surfacing every old PNG in the project, files in
       this location are kept ONLY if they were modified at or
       after the run started (script.do / script.py / script.R
       mtime — written at run start before subprocess exec).

    Files at or below ``_RESEARCHER_PLOT_MAX_BYTES`` carry a
    ``data`` field (base64); larger files carry only metadata so
    the JS can render an "Open externally" placeholder without
    bloating the event payload.

    Most-recent first; capped at ``_RESEARCHER_PLOT_MAX_PER_RESULT``.

    Privacy: this is the RESEARCHER's view. Bytes never reach the
    model — they're injected only into the tool_result event that
    the bridge sends to the local pywebview window. The
    model-vision path is the manifest-gated
    :meth:`SessionRunner._capture_plots`.
    """
    if not run_dir:
        return []
    base = Path(run_dir)
    if not base.is_dir():
        return []

    # Determine when this run started so we can filter ``session_cwd``
    # PNGs to only those written by this run (otherwise every prior
    # plot in the session dir would show up on every tool result).
    # The script file is written at the very start of the run, before
    # subprocess execution — its mtime is the canonical start signal.
    run_start: float | None = None
    for script_name in ("script.do", "script.py", "script.R"):
        p = base / script_name
        if p.is_file():
            try:
                run_start = p.stat().st_mtime
                break
            except OSError:
                pass

    candidates: list[Path] = []
    seen: set[Path] = set()

    def _is_sidecar(p: Path) -> bool:
        """``png_for`` writes ``<basename>.nora.png`` next to the
        original PDF. Those sidecars are an internal artifact of
        the PDF→PNG conversion path; they shouldn't appear as
        their own thumbnail row alongside the source PDF."""
        return p.name.endswith(".nora.png")

    # 1 + 2: anywhere inside run_dir.
    for parent in (base / "_nora_plots", base):
        try:
            for p in parent.iterdir():
                if (
                    p.is_file()
                    and not p.is_symlink()
                    and p.suffix.lower() in _RESEARCHER_PLOT_EXTS
                    and not _is_sidecar(p)
                ):
                    rp = p.resolve()
                    if rp in seen:
                        continue
                    seen.add(rp)
                    candidates.append(p)
        except OSError:
            continue

    # 3: Stata writes plots into session_cwd (the executor's preamble
    # ``cd``s there before user code runs). Filter by run_start so
    # we don't surface stale plots from prior runs.
    if session_cwd and run_start is not None:
        try:
            sc = Path(session_cwd)
        except (OSError, ValueError):
            sc = None
        if sc is not None and sc.is_dir():
            try:
                for p in sc.iterdir():
                    if (
                        p.is_file()
                        and not p.is_symlink()
                        and p.suffix.lower() in _RESEARCHER_PLOT_EXTS
                        and not _is_sidecar(p)
                    ):
                        try:
                            if p.stat().st_mtime + 0.5 < run_start:
                                # Strictly older than run start (with
                                # half-second slack for filesystem mtime
                                # rounding) — predates this run.
                                continue
                            rp = p.resolve()
                            if rp in seen:
                                continue
                            seen.add(rp)
                            candidates.append(p)
                        except OSError:
                            continue
            except OSError:
                pass

    if not candidates:
        return []
    # Most-recent first — the researcher cares about the plots from
    # the latest run more than any leftover from prior iterations.
    try:
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    except OSError:
        pass
    candidates = candidates[:_RESEARCHER_PLOT_MAX_PER_RESULT]

    import base64 as _base64
    from nora.plot_convert import png_for
    out: list[dict[str, Any]] = []
    for p in candidates:
        try:
            size = p.stat().st_size
        except OSError:
            continue
        # PDFs are surfaced as thumbnails too — convert via sips
        # to a sibling PNG (cached). The researcher sees a real
        # raster preview; clicking the tile opens the original
        # PDF in Preview via the path field.
        ext = p.suffix.lower()
        if ext == ".pdf":
            png_sidecar = png_for(p)
            if png_sidecar is None:
                # Conversion failed — keep the row with path-only
                # so the JS can still offer "Open externally" via
                # the OS Preview handler.
                out.append({
                    "name": p.name,
                    "path": str(p),
                    "size": size,
                    "mime": "application/pdf",
                })
                continue
            display = png_sidecar
            try:
                display_size = display.stat().st_size
            except OSError:
                display_size = size
            mime = "image/png"
        else:
            display = p
            display_size = size
            mime = {
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
            }.get(ext, "image/png")
        entry: dict[str, Any] = {
            "name": p.name,
            "path": str(p),
            "size": display_size,
            "mime": mime,
        }
        if display_size <= _RESEARCHER_PLOT_MAX_BYTES:
            try:
                entry["data"] = _base64.b64encode(display.read_bytes()).decode("ascii")
            except OSError:
                pass
        out.append(entry)
    return out


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
        print(f"nora: {e}", file=sys.stderr)
        sys.exit(2)
    if not path.is_dir():
        msg = [f"nora: not a directory: {path}"]
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
        prog="nora",
        description=(
            "Nora — local research assistant. Launch without a path "
            "to drop/choose files from the landing screen, or pass a "
            "directory to open straight into chat."
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
            "nora: warning — sandbox-exec unavailable; "
            "submit_script will refuse to run.",
            file=sys.stderr,
        )

    import webview  # lazy so module import doesn't pay pywebview's
                    # startup cost when something else (e.g. a test)
                    # imports nora.ui without launching the window.

    web_dir = Path(__file__).parent / "web"
    index_path = web_dir / "index.html"
    if not index_path.is_file():
        print(
            f"nora: missing web assets at {index_path}",
            file=sys.stderr,
        )
        sys.exit(2)

    # WKWebView caches file:// resources persistently. Without a
    # cache-bust the user has to manually clear ~/Library/Caches to
    # see code changes. Compute a build-id from the mtimes of the
    # JS/CSS bundle and rewrite the script/link refs in a temporary
    # copy of index.html so each launch's URLs are unique.
    served_index = _materialize_cache_busted_index(web_dir, index_path)
    print(
        f"[nora] starting bridge — web build-id={served_index.stem.split('.')[-1]}",
        file=sys.stderr, flush=True,
    )

    bridge = NoraBridge(cwd=cwd)
    bridge.start_loop()

    window = webview.create_window(
        title="Nora",
        url=str(served_index),
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
