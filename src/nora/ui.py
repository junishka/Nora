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
import re
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
    DEFAULT_EFFORT,
    EFFORT_LEVELS,
    PROVIDER_API_KEY_URLS,
    PROVIDER_DEFAULTS,
    PROVIDER_EFFORTS,
    PROVIDER_PRICING_URLS,
    clamp_effort,
    effort_levels_for_provider,
    efforts_for_provider,
    get_effort,
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


# Hard cap on the ``send_feedback`` message body. The textarea in
# ``index.html`` carries the same ``maxlength`` and ``app.js`` mirrors
# the value in its character counter so all three stay in lockstep —
# bump them together when changing this. The cap is the privacy
# guardrail on the feedback carve-out: the modal's "only this note
# leaves your machine" claim only holds if "this note" stays
# bounded. 8000 characters is comfortably more than any legitimate
# bug report and much less than an accidentally-pasted log or
# dataset chunk.
_FEEDBACK_MAX_MESSAGE_CHARS = 8000


# Cwd choices that are too broad to act as a sandbox root. The
# sandbox profile grants ``file-read*`` and ``file-write*`` over the
# entire ``cwd`` subtree (minus the narrow ``cwd/.nora`` carve-out),
# so a cwd of ``~`` would let scripts read ``~/.ssh/id_rsa``,
# ``~/.aws/credentials``, ``~/Documents/...``, etc. — every
# personal file the user has — and write anywhere under the home.
# The deny carve-out only protects Nora's own state.
#
# This list captures the unambiguous cases. ``~/Documents`` /
# ``~/Desktop`` / ``~/Downloads`` are intentionally NOT here:
# they're plausible project parents (a researcher might keep a
# study under ``~/Documents/research/my-study/``), and rejecting
# them would over-block real workflows. The check fires only on
# roots so broad that no realistic project lives directly there.
_DANGEROUS_CWD_LITERALS: frozenset[Path] = frozenset({
    Path("/"),
    Path("/Users"),
    Path("/home"),
    Path("/tmp"),
    Path("/private"),
    Path("/private/tmp"),
    Path("/var"),
    Path("/private/var"),
    Path("/etc"),
    Path("/private/etc"),
    Path("/usr"),
    Path("/System"),
    Path("/Library"),
    Path("/Applications"),
    Path("/opt"),
    Path("/bin"),
    Path("/sbin"),
    Path("/dev"),
})


def _reject_dangerous_cwd(path: Path) -> str | None:
    """Return a researcher-readable reason if ``path`` is too broad to
    be a sandbox cwd, or ``None`` if it's fine.

    Rejects the user's home dir itself and a handful of system roots.
    Anything under those (e.g. ``~/Documents/project/``) passes — the
    user is expected to point Nora at a specific project subdirectory,
    not at a root that contains every other file they own.
    """
    resolved = path.resolve()
    if resolved in _DANGEROUS_CWD_LITERALS:
        return (
            f"{resolved} is too broad to use as a project folder — "
            "the sandbox would grant scripts read+write access to "
            "this entire subtree. Pick a specific project directory "
            "instead."
        )
    if resolved == Path.home().resolve():
        return (
            "your home directory is too broad to use as a project "
            "folder — the sandbox would grant scripts read+write "
            "access to ~/.ssh, ~/.aws, every Document, and any "
            "other personal file. Pick a specific project "
            "subdirectory (e.g. ~/Documents/<project>/) instead."
        )
    return None


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
        # Default reasoning effort for a NEW runner with no recorded
        # ``active_effort``. Provider-neutral; ``set_effort`` with no
        # focused session updates it, a per-session pick doesn't.
        self._default_effort: str = DEFAULT_EFFORT
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
        # Register the install-confirmation emitter so the
        # ``install_packages`` tool can surface a modal before the
        # underlying installer runs. The tool's hard consent gate
        # fails closed without an emitter, so attaching here is what
        # makes the install path usable inside the running UI while
        # still refusing the install for any process that loads the
        # tools module without a window attached (headless servers,
        # tests that haven't opted in).
        try:
            from nora.install_confirmation import set_request_emitter
            set_request_emitter(self._emit_install_confirmation_request)
        except Exception:  # noqa: BLE001 — never let registration block boot
            pass

    def _emit_install_confirmation_request(
        self,
        token: str,
        language: str,
        packages: list[str],
        action: str,
    ) -> None:
        """Push an install-confirmation request to the page.

        Sent through the same ``nora_event`` channel the chat
        transcript uses so the page's existing event dispatcher
        picks it up. The token round-trips back via
        ``respond_install_confirmation`` to release the awaiting
        future on the tool handler's loop.

        ``evaluate_js`` exceptions are deliberately NOT caught here:
        :func:`nora.install_confirmation.request_confirmation` wraps
        every emitter call in its own try/except and treats a raise as
        deny-immediately. Swallowing the exception at this layer
        prevented that handling from ever firing — the awaiting Future
        then sat on its 5-minute timeout, hanging the tool. Letting
        the exception propagate restores the documented contract
        (failed emit → instant deny) and matches the comment in
        ``install_confirmation.py`` ("Treat as a failed emit and deny
        rather than blocking forever").
        """
        if self._window is None:
            return
        payload = json.dumps({
            "type": "install_confirmation_request",
            "token": token,
            "language": language,
            "packages": list(packages),
            "action": action,
        })
        # Let the exception propagate. ``request_confirmation`` in
        # ``nora.install_confirmation`` wraps this call in a try/except
        # that catches emitter exceptions and resolves the awaiting
        # future as denied (see ``install_confirmation.py:147``). If we
        # swallowed the failure here, that handler would never see it
        # and the install request would block until the per-request
        # timeout — minutes of UI hang on every install attempt while
        # the webview is closing or reloading. Re-raise so the deny is
        # immediate.
        self._window.evaluate_js(f"window.nora_event({payload});")

    def respond_install_confirmation(
        self, token: str, approved: bool,
    ) -> dict[str, Any]:
        """JS calls this when the researcher clicks Approve or Deny.

        ``token`` was issued by the awaiting tool handler; the bridge
        forwards the decision to the install-confirmation registry,
        which resolves the matching Future on the tool's loop.
        Returns ``{ok}`` so the page can detect a stale token (e.g.
        the modal was clicked AFTER the request timed out).
        """
        from nora.install_confirmation import respond
        ok = respond(token, bool(approved))
        return {"ok": ok}

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
        # Release any awaiting install-confirmation futures BEFORE
        # the loop stops. Without this, a tool handler that's blocked
        # on ``request_confirmation`` would wait its full timeout
        # against a dead loop and lock the SDK client. ``cancel_all``
        # resolves every pending request as deny.
        try:
            from nora.install_confirmation import (
                cancel_all as _cancel_install_confirmations,
                clear_request_emitter,
            )
            _cancel_install_confirmations()
            clear_request_emitter()
        except Exception:  # noqa: BLE001 — defensive
            pass
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

    def doctor_report(self) -> dict[str, Any]:
        """Return the environment health report as a JSON-safe dict.

        Mirrors the data the ``nora --doctor`` CLI prints, so the UI
        can render the same checks as a banner. Surfaced as a bridge
        method (rather than baked into ``ui_ready``) because:

          * ``ui_ready`` runs once per page load; the doctor report
            can change mid-session (the researcher installs Homebrew
            Python in a terminal, ``install_packages`` adds a
            scientific-stack package). A separate method lets the
            UI refresh on demand without re-running auth logic.
          * Tests can assert against this method in isolation
            without spinning up the auth flow.

        The returned shape is stable JSON: every field is a
        primitive (str / bool / list / dict). The frontend can
        ``await window.pywebview.api.doctor_report()`` and render
        the result directly.
        """
        from nora.doctor import run_doctor
        report = run_doctor()
        return {
            "blocked": report.blocked,
            "runtimes": [
                {
                    "runtime": r.runtime,
                    "status": r.status,
                    "detail": r.detail,
                    "advice": list(r.advice),
                }
                for r in report.runtimes
            ],
            "rejected_python_candidates": [
                {"binary": path, "stderr_excerpt": stderr}
                for path, stderr in report.rejected_python_candidates
            ],
        }

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
            self._persist_active_model(active)

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
                webview.FileDialog.OPEN,
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

        The chosen folder is recorded in the external-sessions
        registry so it surfaces in the sidebar and is reachable via
        ``switch_session`` after the researcher moves away. Without
        that, picking a folder created a session that couldn't be
        navigated back to without re-picking it through the dialog.
        """
        if self._window is None:
            return {"ok": False, "reason": "window not ready"}
        try:
            import webview
            result = self._window.create_file_dialog(
                webview.FileDialog.FOLDER, allow_multiple=False
            )
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "reason": f"dialog error: {e}"}
        if not result:
            return {"ok": False, "reason": "cancelled"}
        folder = Path(result[0]).expanduser().resolve()
        if not folder.is_dir():
            return {"ok": False, "reason": f"not a directory: {folder}"}
        # Refuse cwd choices broad enough to make the sandbox
        # functionally toothless — see ``_reject_dangerous_cwd``.
        # The user-driven file picker is the only entry point where a
        # researcher can hand Nora an arbitrary directory; staged
        # sessions land under SESSIONS_ROOT and ``switch_session``
        # already enforces parent == SESSIONS_ROOT. Run this check
        # BEFORE registering the folder so a rejected path never
        # ends up in the recent-folders sidebar.
        reason = _reject_dangerous_cwd(folder)
        if reason is not None:
            return {"ok": False, "reason": reason}
        # Register before ``_set_cwd`` so a downstream failure (e.g.
        # provenance init) doesn't leave the folder unreachable from
        # the sidebar. The registry survives an app restart, so even
        # if the page never re-renders the researcher can re-open
        # the project from "Recent folders" next time.
        try:
            from nora.external_sessions import register
            register(SESSIONS_ROOT, folder)
        except Exception:  # noqa: BLE001 — registry must not block opens
            pass
        return self._set_cwd(folder)

    def upload_files(
        self, files: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Receive files from JS drag-and-drop. Each ``files[i]`` is
        ``{name: str, content: str (base64)}``. We decode and stage
        into a fresh session dir. Multiple files are supported —
        they all land in the same session.

        Size capped per-file at ``_DRAG_DROP_MAX_BYTES`` (1 GB).
        The constraint is peak memory while transferring through
        the bridge: a file of N bytes needs roughly 3–4N during
        upload (JS ArrayBuffer + JS base64 string + Python-side
        decode), so 1 GB peaks around 3–4 GB. Larger datasets
        should use the file picker (:meth:`choose_files`), which
        copies directly from disk with no memory overhead and no
        size limit.

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
        # Aggregate cap on decoded bytes: each file passes the
        # per-file cap, but a multi-file drop accumulates them all
        # in this list before staging. Two 800 MB files would each
        # pass ``_DRAG_DROP_MAX_BYTES`` per-file yet hold ~1.6 GB of
        # decoded bytes in this scope. Bound the total too — same
        # threshold as per-file so the rule reads consistently from
        # the user's side ("drag-drop moves up to 1 GB total").
        # The JS side gates first; this is defense-in-depth.
        aggregate_bytes = 0
        rejected_exts: list[str] = []
        for item in files:
            name = item.get("name", "")
            content_b64 = item.get("content", "")
            if not name or not isinstance(content_b64, str):
                continue
            # Server-side extension gate. JS already filters via
            # ``acceptedByDropZone`` / ``COMPOSER_DATA_EXTS``, but a
            # forged ``add_dropped_files`` call from a different
            # frontend path would otherwise let arbitrary extensions
            # land in the session dir. Compare against the basename's
            # final suffix (lowercased), normalising ``Path`` so a
            # name like ``data/csv.zip`` cannot smuggle the ``.zip``
            # past the ``.csv`` substring.
            ext = Path(name).suffix.lower()
            if ext not in _DRAG_DROP_ALLOWED_EXTS:
                rejected_exts.append(Path(name).name)
                continue
            # Strip any data URL prefix JS may have added.
            if "," in content_b64:
                content_b64 = content_b64.split(",", 1)[1]
            # Pre-decode size gate. base64 expands 4:3, so a 1 GB
            # binary file is ~1.33 GB encoded; ``_b64_oversize`` does
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
            aggregate_bytes += len(blob)
            if aggregate_bytes > _DRAG_DROP_MAX_BYTES:
                agg_mb = aggregate_bytes // (1024 * 1024)
                cap_mb = _DRAG_DROP_MAX_BYTES // (1024 * 1024)
                return {
                    "ok": False,
                    "reason": (
                        f"Total drop size is {agg_mb} MB — drag-drop is "
                        f"capped at {cap_mb} MB total. Use Choose Files… "
                        f"to import larger batches with no aggregate "
                        f"limit."
                    ),
                }
            decoded.append((Path(name).name, blob))
        if not decoded:
            if rejected_exts:
                # Every file was rejected on extension. Tell the user
                # which ones and why so they can re-export.
                return {
                    "ok": False,
                    "reason": (
                        f"Drop rejected — {', '.join(rejected_exts)}: "
                        f"only data, script, log, and image files are "
                        f"accepted on the landing zone."
                    ),
                }
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
        existing = current.datasets.get(name)
        # When the researcher selects the same depth as the app-wide
        # default, drop the explicit ``max_depth`` entry rather than
        # saving "explicit at the default value." But preserve any
        # ``non_disclosive_variables`` opt-ins the researcher made
        # earlier — those are an independent dimension of the policy
        # (per-variable min/max disclosure consent) and have no
        # relationship with the schema-depth ceiling. Without this
        # carry-over, a researcher who had opted ``year_of_birth`` and
        # ``education_years`` into safe min/max disclosure would lose
        # both opt-ins simply by clicking the schema-depth chip back
        # to the default tier — surprising, silent data-policy regression.
        updated_datasets = dict(current.datasets)
        preserved_ndv = existing.non_disclosive_variables if existing else ()
        if depth == current.default_max_depth:
            if preserved_ndv:
                # Keep the opt-in list alive even though the depth is
                # back at the default. The entry stays "explicit" in
                # the sense that the researcher has expressed an
                # opinion — just on a different axis.
                updated_datasets[name] = DatasetPolicy(
                    max_depth=depth,
                    set_at=datetime.now(timezone.utc).isoformat(),
                    non_disclosive_variables=preserved_ndv,
                )
            else:
                updated_datasets.pop(name, None)
        else:
            updated_datasets[name] = DatasetPolicy(
                max_depth=depth,
                set_at=datetime.now(timezone.utc).isoformat(),
                non_disclosive_variables=preserved_ndv,
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

    # -------- queued-send attachment freezing --------
    #
    # Two endpoints that the JS-side queue uses to freeze the runner's
    # current pending_* lists into a per-message snapshot, then thaw
    # the snapshot back when the queued message fires. See
    # ``Runner.freeze_pending_for_queue`` / ``restore_frozen_pending``
    # for the race these endpoints close.

    def freeze_pending_attachments(self, session_cwd: str) -> str | None:
        """Snapshot the runner's pending_* lists under a fresh token,
        clear the runner's pending state, and return the token.

        Called by the JS submit handler when the user hits Send while
        a turn is already running — the queued message owns the
        snapshot, and any attachments staged AFTER the queue (for a
        later message) land in a fresh runner slot. Returns ``None``
        if the named session has no live runner; the JS path then
        falls back to the older fire-without-token shape so the
        message at least sends.
        """
        if not session_cwd:
            return None
        try:
            resolved = Path(session_cwd).resolve()
        except (OSError, RuntimeError):
            return None
        runner = self._runners.get(str(resolved))
        if runner is None:
            return None
        token = uuid.uuid4().hex[:16]
        runner.freeze_pending_for_queue(token)
        return token

    def clear_pending_for_session(self, session_cwd: str) -> dict[str, Any]:
        """Drop researcher-staged @-mentions, staged scripts, and
        mentioned images on the named session's runner. Queued-
        message frozen snapshots and model-captured plot images are
        intentionally left alone — see below.

        Called by the JS session-switch handler. The frontend wipes
        its staged composer state (image thumbs, data notices, mention
        chips) when the researcher leaves a session — without this
        bridge call, the BACKEND runner's user-staged lists for that
        session survive hidden, and the next plain message sent on
        return would silently inline @-mentions / scripts the
        researcher staged before the switch (and which the UI no
        longer shows). The desync makes vanished attachments ride
        invisibly. Clearing here keeps both sides aligned: nothing
        staged in the UI, nothing staged on the runner.

        Queued-message frozen snapshots stay because the researcher
        already committed to send those messages (they sit in the
        JS queue). Plot images stay because they're model output
        from the previous turn's submit_script, queued to ride this
        session's next user turn regardless of focus — a researcher
        who returns and asks "interpret the plot" must still find
        the image attached.

        ``session_cwd`` is the path of the runner to clear (not the
        bridge's ``self.cwd`` — the JS already knows which session
        it's leaving). Idempotent: returns ``{"ok": True}`` even when
        the runner doesn't exist yet (a session the researcher
        clicked into but never typed in).
        """
        if not session_cwd:
            return {"ok": False, "reason": "empty session_cwd"}
        try:
            resolved = Path(session_cwd).resolve()
        except (OSError, RuntimeError) as e:
            return {"ok": False, "reason": f"bad path: {e}"}
        runner = self._runners.get(str(resolved))
        if runner is None:
            # No live runner = nothing to clear. Not an error: the JS
            # may call this on a sidebar entry the researcher never
            # actually opened, or after a runner has already been
            # closed.
            return {"ok": True, "cleared": False}
        # Use the user-staged-only clear: frozen queued-message
        # snapshots survive (they belong to messages the researcher
        # already committed to send, sitting in the JS queue, and
        # must still fire when the in-flight turn finishes), AND
        # ``pending_plot_images`` survives. Plot images are captured
        # by ``_capture_plots`` from the prior turn's submit_script
        # output, not by anything the researcher staged through the
        # composer, so a focus toggle should not erase them — a
        # return to this session with "interpret the plot" must
        # still attach the image the model just produced.
        runner.clear_unsent_user_staged()
        return {"ok": True, "cleared": True}

    def discard_pending_attachments_token(
        self, session_cwd: str, token: str,
    ) -> bool:
        """Drop a frozen snapshot without firing it. Used when a
        queued message gets cancelled (Stop fires, rewind drops the
        queue). Returns ``True`` if a token was dropped, ``False`` if
        the runner / token couldn't be found."""
        if not session_cwd or not token:
            return False
        try:
            resolved = Path(session_cwd).resolve()
        except (OSError, RuntimeError):
            return False
        runner = self._runners.get(str(resolved))
        if runner is None:
            return False
        had = token in runner.frozen_pending_attachments
        runner.discard_frozen_pending(token)
        return had

    def send_message_with_token_to_session(
        self, session_cwd: str, text: str, token: str,
    ) -> str | None:
        """Like :meth:`send_message_to_session`, but restore the
        named frozen snapshot into ``pending_*`` first so the queued
        message rides with EXACTLY the attachments it owned at queue
        time. Missing token = no-op restore (the send still fires;
        relevant when an older JS path didn't capture a token)."""
        return self._send_to_active(
            text, images=None, target_cwd=session_cwd, frozen_token=token,
        )

    def send_message_with_images_and_token_to_session(
        self, session_cwd: str, text: str,
        images: list[dict[str, Any]], token: str,
    ) -> str | None:
        """Image+token combination of the queued send."""
        return self._send_to_active(
            text, images=images, target_cwd=session_cwd, frozen_token=token,
        )

    def _send_to_active(
        self,
        text: str,
        images: list[dict[str, Any]] | None,
        target_cwd: str | None,
        frozen_token: str | None = None,
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
        # Restore the queued message's frozen attachments BEFORE
        # the user-message bookkeeping (so any logged context counts
        # see them) and BEFORE scheduling the turn coroutine (which
        # consumes pending_*). Missing tokens are silent no-ops:
        # the older fire-without-token path stays valid.
        if frozen_token:
            runner.restore_frozen_pending(frozen_token)
        self._record_user_message(runner, text, images=images)
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
        try:
            asyncio.run_coroutine_threadsafe(coro, self._loop)
        except RuntimeError as e:
            # The pre-check above only catches ``self._loop is None``;
            # a non-None loop can still be closed or stopping (e.g.,
            # mid-shutdown). Without this cleanup the registered
            # pending id would never be drained (the coroutine never
            # starts → ``_consume_pending_turn`` never runs), and
            # ``is_busy`` would keep returning True forever, wedging
            # the UI on this session. Drop the unstarted coroutine,
            # discard the pending id, and surface a turn_error so
            # the JS state machine moves the composer back out of
            # the "sending" state.
            coro.close()
            runner.discard_pending_turn(turn_id)
            self._dispatch_event({
                "type": "turn_error",
                "message": f"could not schedule turn: {e}",
                "session_cwd": str(runner.cwd),
                "turn_id": turn_id,
            })
            return None
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
                webview.FileDialog.OPEN,
                allow_multiple=True,
                file_types=file_types,
            )
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "reason": f"dialog error: {e}"}
        if not result:
            return {"ok": False, "reason": "cancelled"}

        # Same capture the drop path makes: copying a multi-GB .dta
        # takes long enough for the researcher to switch sessions,
        # and re-reading ``self.cwd`` per file would split one
        # selection across two sessions.
        cwd = self.cwd

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
                dst = cwd / src.name
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
                    # Attach to the session that received the file,
                    # not whatever is focused now — see the drop
                    # path in ``add_files_from_blobs``.
                    runner = self._runner_for_cwd(cwd)
                    if runner is not None:
                        try:
                            _stage_script_for_next_turn(
                                runner.pending_script_attachments,
                                src.name, ext, dst.read_bytes(),
                                path=str(dst.resolve()),
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
                img_dst = _disambiguate_target(cwd, src.name)
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

        # Mark every file the researcher staged this call as known to
        # the provenance manifest. Both data files (``added``) and
        # images (``images``) are explicit researcher actions; the
        # manifest's append-only contract means a re-stage of the
        # same name is a no-op. ``read_attached_file`` /
        # ``submit_script_file`` consult ``is_known`` to refuse cwd
        # top-level source files that didn't come through this path
        # (the SDC-bypass channel where a sandboxed script writes a
        # ``.R`` file containing raw rows and the model later asks to
        # read it back).
        try:
            from nora.file_provenance import mark_known
            mark_known(
                cwd,
                [*added, *(img["name"] for img in images)],
            )
        except Exception:  # noqa: BLE001 — provenance is best-effort
            pass

        return {
            "ok": True,
            "added": added,
            "images": images,
            "skipped": skipped,
            "skipped_existing": skipped_existing,
            # Which session the files landed in — the focus may have
            # moved during the copy. Mirrors ``add_files_from_blobs``.
            "cwd": str(cwd),
            "policy": self._policy_summary(cwd),
            "session_title": _session_title(cwd),
        }

    def add_files_from_blobs(
        self, files: list[dict[str, Any]], target_cwd: str | None = None
    ) -> dict[str, Any]:
        """Twin of :meth:`add_files`, but for files dropped or pasted
        directly onto the composer from JS — no native dialog involved.

        Each ``files[i]`` is ``{name, content (base64), mime?}``. Data
        and script files are copied into the target session; images
        are decoded and returned so the frontend can stage them as
        vision attachments. Returns the same shape as
        :meth:`add_files`.

        ``target_cwd`` is the session the drop landed on, captured by
        the JS before it reads the file. Reading a large file with
        FileReader takes long enough for the researcher to switch
        sessions, so without it the bytes follow focus into whatever
        session happens to be up when the request arrives. Omitted
        (older JS) falls back to the focused session.
        """
        if self.cwd is None:
            return {
                "ok": False,
                "reason": "no active session — start one first",
            }
        if not files:
            return {"ok": False, "reason": "no files"}
        # Bind to the session the drop landed on, once, up front.
        # Decoding is slow enough for the researcher to switch
        # sessions, and re-reading ``self.cwd`` per write would drop
        # the file in the wrong one.
        cwd = self.cwd
        if target_cwd:
            try:
                requested = Path(target_cwd).expanduser().resolve()
            except OSError as e:
                return {"ok": False, "reason": f"bad path: {e}"}
            # Only a session the bridge already knows. The caller is
            # naming where its own drop started, so anything else is
            # either a deleted session or a caller we shouldn't be
            # writing for.
            if (
                self._runners.get(str(requested)) is None
                and requested != cwd.resolve()
            ):
                return {
                    "ok": False,
                    "reason": "that session is no longer open",
                }
            cwd = requested

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
        # Aggregate-bytes accumulator for the data/script copy path.
        # The per-file cap protects each individual decode, but a
        # multi-file drop accumulates blobs in this method's scope
        # before writing to disk. Two 800 MB .dta files would each
        # pass per-file yet hold ~1.6 GB of decoded bytes concurrently
        # in transient heap. Bound the total at the same threshold so
        # the drag-drop rule reads the same as the landing page.
        # Images are NOT counted (5 MB cap each, capped count) — a
        # researcher who drops a screenshot alongside data files
        # shouldn't see the data files rejected because of the
        # image's bytes.
        aggregate_data_bytes = 0

        for item in files:
            name = item.get("name", "")
            content_b64 = item.get("content", "")
            if not name or not isinstance(content_b64, str):
                continue
            if "," in content_b64:
                content_b64 = content_b64.split(",", 1)[1]
            safe_name = Path(name).name
            ext = Path(safe_name).suffix.lower()
            # Server-side extension gate. JS already filters via
            # ``acceptedByComposer``; this is defense-in-depth for any
            # client that bypasses it, AND a perf win — without it,
            # an unknown-extension drop would still flow through
            # ``base64.b64decode`` (allocating up to 1 GB of bytes)
            # before being silently dropped by the dispatch below.
            # Surface the name in ``skipped`` so the researcher gets a
            # clear "we ignored these" signal rather than the file
            # vanishing without a trace.
            if ext not in _COPY_EXTS and ext not in _IMAGE_EXTS_MIMES:
                skipped.append(f"{safe_name} (unsupported file type)")
                continue
            # Pre-decode size gate. Both data/script files (capped at
            # _DRAG_DROP_MAX_BYTES, 1 GB) and images (capped at
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
                cap: int = _DRAG_DROP_MAX_BYTES
                hint = "the + button next to the composer"
            else:  # ext in _IMAGE_EXTS_MIMES
                cap = _IMAGE_MAX_BYTES
                hint = ""  # images skipped silently below, not error-returned
            if _b64_oversize(content_b64, cap):
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
                aggregate_data_bytes += len(blob)
                if aggregate_data_bytes > _DRAG_DROP_MAX_BYTES:
                    agg_mb = aggregate_data_bytes // (1024 * 1024)
                    cap_mb = _DRAG_DROP_MAX_BYTES // (1024 * 1024)
                    return {
                        "ok": False,
                        "reason": (
                            f"Total drop size is {agg_mb} MB — drag-drop "
                            f"is capped at {cap_mb} MB total. Use the + "
                            f"button next to the composer to import "
                            f"larger batches with no aggregate limit."
                        ),
                    }
                dst = cwd / safe_name
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
                    # The runner for ``cwd``, not the focused one:
                    # the bytes went to ``cwd``, so its contents have
                    # to be attached to the same conversation.
                    runner = self._runner_for_cwd(cwd)
                    if runner is not None:
                        _stage_script_for_next_turn(
                            runner.pending_script_attachments,
                            safe_name, ext, blob,
                            path=str(dst.resolve()),
                        )
            elif ext in _IMAGE_EXTS_MIMES:
                # Save to cwd alongside vision staging — see the
                # mirror code path in ``add_files`` for the
                # rationale (researchers expect "I uploaded this"
                # to mean the file is in their session, not just
                # that the model can see it once).
                img_dst = _disambiguate_target(cwd, safe_name)
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

        # Mark every file the researcher just dropped/pasted as known
        # to the provenance manifest — same rationale as ``add_files``
        # above. Without this hook, files staged through the
        # composer drop path would be presumed sandbox-output the
        # next time the model tried to read them.
        try:
            from nora.file_provenance import mark_known
            mark_known(
                cwd,
                [*added, *(img["name"] for img in images)],
            )
        except Exception:  # noqa: BLE001 — provenance is best-effort
            pass

        return {
            "ok": True,
            "added": added,
            "images": images,
            "skipped": skipped,
            "skipped_existing": skipped_existing,
            # Which session the files landed in. May not be the focused
            # one, so JS checks before painting ``policy`` / title.
            "cwd": str(cwd),
            "policy": self._policy_summary(cwd),
            "session_title": _session_title(cwd),
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
        current_effort = active.effort if active is not None else self._default_effort
        return {
            "ok": True,
            "current": current_model,
            "current_provider": current_provider,
            # Effort rides the same payload: the picker renders an
            # Effort bar under the model list, and the chip shows
            # "<model> · <effort>". The ladders DIFFER per provider
            # (Anthropic has ``max``, OpenAI stops at ``xhigh``), so
            # ship all of them keyed by provider — the JS re-renders
            # the bar from the selected model's provider without a
            # round-trip on every model switch. ``efforts`` is the
            # ladder for the CURRENT provider, so a caller that only
            # wants today's bar doesn't have to index the map.
            "current_effort": current_effort,
            "default_effort": DEFAULT_EFFORT,
            "efforts": [
                {"id": e.id, "label": e.label}
                for e in efforts_for_provider(current_provider)
            ],
            "efforts_by_provider": {
                provider: [{"id": e.id, "label": e.label} for e in ladder]
                for provider, ladder in PROVIDER_EFFORTS.items()
            },
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

    # Global byte budget for inline thumbnails returned by
    # ``list_session_files``. The per-row cap (3 MB) bounds a single
    # image, but a plot-heavy session can accumulate hundreds of plots
    # under ``_nora_plots/`` and base64 expansion adds ~33% on top —
    # without a global cap the bridge payload would balloon past the
    # WebView's tolerance for ``evaluate_js`` and stall the panel
    # render. 32 MB comfortably accommodates a normal working session
    # (a dozen 1600px PNGs, a few PDF sidecars) while keeping
    # pathological cases bounded.
    _FILES_PANEL_TOTAL_THUMB_BUDGET = 32 * 1024 * 1024

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
        from nora.session_files import enumerate_files_panel_rows

        # Files panel uses the researcher-facing view: hide things
        # that already render on a result card (run-dir scripts,
        # ``_nora_plots/`` helper outputs) and hide files in cwd
        # that a ``submit_script`` run CREATED (per its
        # ``cwd_writes.json`` manifest). The model-facing
        # ``list_session_files`` tool keeps the full view.
        # ``enumerate_files_panel_rows`` is the shared definition so
        # the read/delete defence gates stay in lock-step with this
        # listing.
        rows = enumerate_files_panel_rows(self.cwd)
        # Running budget shared across all rows. Once exhausted, the
        # remaining image / PDF rows still appear in the panel but
        # ship without ``data`` — the UI falls back to a placeholder
        # plus click-to-open, the same path used for over-cap singles.
        bytes_used = [0]
        for row in rows:
            self._enrich_files_panel_row(
                row, budget_used=bytes_used,
                budget_total=self._FILES_PANEL_TOTAL_THUMB_BUDGET,
            )
        return {"ok": True, "files": rows}

    @staticmethod
    def _enrich_files_panel_row(
        row: dict[str, Any],
        *,
        budget_used: list[int] | None = None,
        budget_total: int | None = None,
    ) -> None:
        """Add inline thumbnail bytes (``data`` + ``mime``) to image
        rows in the Files panel. PDF/EPS rows get a sips-rasterised
        PNG sidecar shipped instead. 3 MB cap matches the chat-
        thumbnail cap so 1600px Stata PDFs / PNGs render at full res
        in the panel + lightbox; larger files still appear in the
        panel (with a placeholder + click-to-open) but their bytes
        don't ride through ``evaluate_js``.

        ``budget_used`` is a single-element list acting as a shared
        running total across rows; ``budget_total`` is the cap. Both
        optional so single-row tests can call this without bookkeeping.
        When the budget is exhausted, this row's ``data`` is skipped
        and the row falls through to the placeholder/click-to-open
        path on the UI side — same as a >3 MB single.
        """
        import base64 as _base64

        _IMAGE_THUMB_CAP = 3 * 1024 * 1024
        # Match the composer-accepted set so a dropped WebP / GIF
        # screenshot renders in the Files panel rather than appearing
        # as a "?" placeholder despite being in the listing. All four
        # raster formats render natively in WebView.
        _IMAGE_THUMB_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
        _IMAGE_MIME = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".gif": "image/gif",
        }

        def _within_budget(raw_size: int) -> bool:
            # base64 expansion is 4/3; budget against the encoded size
            # since that's what actually rides through ``evaluate_js``.
            if budget_used is None or budget_total is None:
                return True
            projected = budget_used[0] + ((raw_size * 4 + 2) // 3)
            return projected <= budget_total

        def _charge(raw_size: int) -> None:
            if budget_used is not None:
                budget_used[0] += ((raw_size * 4 + 2) // 3)

        ext = row.get("ext", "")
        size = row.get("size", 0)
        path = Path(row["path"])
        if ext in _IMAGE_THUMB_EXTS and size <= _IMAGE_THUMB_CAP:
            try:
                raw = path.read_bytes()
            except OSError:
                return
            if not _within_budget(len(raw)):
                return
            row["data"] = _base64.b64encode(raw).decode("ascii")
            row["mime"] = _IMAGE_MIME.get(ext, "image/png")
            _charge(len(raw))
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
                        raw = sidecar.read_bytes()
                    except OSError:
                        return
                    if not _within_budget(len(raw)):
                        return
                    row["data"] = _base64.b64encode(raw).decode("ascii")
                    row["mime"] = "image/png"
                    _charge(len(raw))

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
        # Defence-in-depth: only permit deletion of files the Files
        # panel actually surfaces. Containment in cwd alone is not
        # enough — the bridge is callable from page-rendered JS, and
        # a compromised result that escaped sanitization could call
        # ``delete_session_file('.nora/chat_history.jsonl')`` (or
        # ``results.db`` / ``policy.json`` / ``session_state.json``)
        # and corrupt audit history, the policy file, or session
        # state. The Files panel itself never lists those, so a
        # request to delete something outside the listing has no
        # legitimate UI origin. We also accept the PDF/EPS sidecar
        # raster (``*.nora.png`` next to a listed PDF) because the
        # delete path below cleans those up implicitly.
        #
        # Enumeration MUST match the panel's enumeration in
        # ``list_session_files`` (above) — same kwargs. Without that,
        # the gate widens to files that aren't in the panel (run-dir
        # scripts, helper plots, cwd files a script created or
        # modified) and a page-JS call could touch what the panel
        # intentionally hid. The "panel listing" framing only holds
        # if both sides agree on what the panel shows.
        from nora.session_files import enumerate_session_files
        listing_paths: set[Path] = set()
        for row in enumerate_session_files(
            cwd_resolved,
            include_data=True,
            include_run_scripts=False,
            include_run_plots=False,
            exclude_script_writes=True,
        ):
            try:
                listing_paths.add(Path(row["path"]).resolve())
            except OSError:
                continue
        if target not in listing_paths:
            return {
                "ok": False,
                "reason": (
                    "this file isn't in the Files panel listing — "
                    "refused as a precaution"
                ),
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
        #
        # Match by absolute path where the staging recorded one
        # (script attachments and mentioned-image vision blobs),
        # falling back to name for the older mention-notice list
        # where only the display name is tracked. Path matching is
        # necessary because run-dir scripts live on disk as
        # ``<run_dir>/script.<ext>`` but were staged under their
        # label-derived display name (``linear_regression.py``);
        # a name-only filter against ``target.name`` would leave
        # the deleted script content staged in memory and the next
        # send would inline a file the researcher just deleted.
        # Helper plots in different run dirs can also share a
        # basename, so name-only matching there could clear the
        # wrong entry.
        target_str = str(target)
        unstaged_names: list[str] = []
        runner = self._active_runner()
        if runner is not None:
            kept_scripts: list[dict[str, Any]] = []
            for a in runner.pending_script_attachments:
                staged_path = a.get("path")
                matched = (
                    staged_path == target_str
                    if staged_path
                    else a.get("name") == target.name
                )
                if matched:
                    unstaged_names.append(a.get("name", ""))
                else:
                    kept_scripts.append(a)
            runner.pending_script_attachments = kept_scripts

            kept_imgs: list[dict[str, Any]] = []
            for a in runner.pending_mentioned_images:
                staged_path = a.get("path")
                matched = (
                    staged_path == target_str
                    if staged_path
                    else a.get("name") == target.name
                )
                if matched:
                    unstaged_names.append(a.get("name", ""))
                else:
                    kept_imgs.append(a)
            runner.pending_mentioned_images = kept_imgs

            # pending_mentioned_files only tracks display names, but
            # those display names match top-level basenames for the
            # non-script, non-image kinds that land here (run-dir
            # scripts go into pending_script_attachments instead).
            kept_mentions: list[str] = []
            for n in runner.pending_mentioned_files:
                if n == target.name:
                    unstaged_names.append(n)
                else:
                    kept_mentions.append(n)
            runner.pending_mentioned_files = kept_mentions
        # Best-effort: also remove the cached PDF/EPS → PNG sidecar
        # if there was one. Otherwise the orphan PNG would keep
        # showing in the Files panel until the bridge restarted.
        sidecar = target.with_name(target.stem + ".nora.png")
        if sidecar.is_file():
            try:
                sidecar.unlink()
            except OSError:
                pass
        # Return the dropped staged names so JS can splice them
        # out of ``stagedDataNotices`` — the chip-list array
        # rendered by ``renderAttachments``. Without this, a
        # delete that targeted a run-dir script (display name
        # ``linear_regression.py``, disk name ``script.py``)
        # would still see its chip on screen because JS only
        # had ``res.name`` (= ``script.py``) to splice with.
        return {
            "ok": True,
            "name": target.name,
            "unstaged": [n for n in unstaged_names if n],
        }

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
        :meth:`delete_session_file`) so the caller can identify the
        exact on-disk file. The read gate then requires the path to
        appear in the Files-panel listing
        (:func:`nora.session_files.enumerate_files_panel_rows`); any
        path the panel doesn't show — run-dir scripts, helper plots,
        raw subprocess logs, ``.nora/`` internals — is refused.
        Containment in cwd is verified before any read.

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
        # Defence-in-depth, mirroring ``delete_session_file``: only
        # permit reads of files the Files panel actually surfaces.
        # Containment in cwd is too loose — the bridge is callable
        # from page-rendered JS, and a compromised result that escaped
        # sanitization could call read_session_file_text() with
        # ``<cwd>/.nora/runs/<id>/stdout.log`` to pull the raw
        # subprocess transcript onto the page. The executor's
        # contract is that raw .log files never cross to the model
        # (see executor.py's "raw .log files NEVER cross" comment),
        # so the panel never lists those files — and any request
        # naming one has no legitimate UI origin.
        #
        # Use the shared ``enumerate_files_panel_rows`` helper so this
        # gate stays in lock-step with the panel listing. Calling
        # ``enumerate_session_files`` with broader flags inline drifted
        # from the actual panel mode (PR #55 narrowed the panel but
        # not this gate), which made the "only what the panel surfaces"
        # claim untrue.
        from nora.session_files import enumerate_files_panel_rows
        listing_paths: set[Path] = set()
        for row in enumerate_files_panel_rows(cwd_resolved):
            try:
                listing_paths.add(Path(row["path"]).resolve())
            except OSError:
                continue
        if target not in listing_paths:
            return {
                "ok": False,
                "reason": (
                    "this file isn't in the Files panel listing — "
                    "refused as a precaution"
                ),
            }
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
        # The Files-panel "Copy" button surfaces researcher-uploaded
        # scripts (run-dir scripts are hidden from the panel by
        # design). Researcher uploads never carry Nora's preamble,
        # and the run-dir ``script.do`` / ``script.py`` files the
        # executor writes are themselves clean (preamble lives in a
        # sibling ``_nora_wrapper.*`` file). So the bytes already on
        # disk are exactly what the researcher wants on the
        # clipboard — no further processing required.
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

    def attach_session_file(
        self, name: str, path: str | None = None,
    ) -> dict[str, Any]:
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

        ``path`` is the optional absolute on-disk path the JS
        mention dropdown was looking at when the row was clicked.
        Helper plots in different run dirs can share a basename
        (``coefficients.png`` in both run A and run B), so a
        name-only resolution walks the run dirs and stages
        whichever ``iterdir`` returns first — not the row the user
        actually selected. When ``path`` is provided we use it
        directly after the same containment check the name-only
        path would have applied; without it we fall through to
        the name-based search for back-compat with older JS.
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
        # Rewind-aware: ``list_session_files`` and
        # ``read_attached_file`` already filter run-dir artifacts to
        # the rewind-visible set so a discarded chat branch's scripts
        # / plots are not reachable. This bridge attach path is the
        # researcher-side counterpart and used to bypass the same
        # contract — staging a known display name from a hidden
        # branch still worked. Compute the visible set once and use
        # it on both the helper-plot iteration and the run-script
        # lookup below.
        from nora.session_files import visible_run_dir_names
        visible_runs = visible_run_dir_names(self.cwd)
        # Explicit path wins. The JS mention dropdown carries the
        # exact ``path`` of the clicked row, so passing it through
        # avoids the basename-collision bug: helper plots in
        # different run dirs can share a name (``coefficients.png``
        # in run A and run B) and the name-only fallback below
        # iterates run dirs in filesystem order, staging whichever
        # appears first rather than the row the user clicked. The
        # supplied path MUST still pass the same gates a name-only
        # call would — contained in cwd, real file, not a
        # symlink-escape, and rewind-visible if it lives in a run
        # dir.
        if path:
            try:
                supplied = Path(path).resolve()
            except OSError:
                supplied = None
            if (
                supplied is not None
                and _is_within(supplied, cwd_resolved)
                and supplied.is_file()
                and not supplied.is_symlink()
            ):
                runs_root_resolved = (
                    (self.cwd / ".nora" / "runs").resolve()
                )
                in_run_dir = _is_within(supplied, runs_root_resolved)
                run_dir_visible = True
                if in_run_dir and visible_runs is not None:
                    try:
                        rel = supplied.relative_to(runs_root_resolved)
                    except ValueError:
                        run_dir_visible = False
                    else:
                        run_dir_name = rel.parts[0] if rel.parts else ""
                        run_dir_visible = run_dir_name in visible_runs
                if run_dir_visible:
                    target = supplied
        if target is not None:
            # Path-based resolution already landed; skip the
            # name-based fallbacks so we don't overwrite the
            # caller's selected row with a same-name file in cwd.
            pass
        elif _is_within(candidate, cwd_resolved) and candidate.is_file():
            target = candidate
        else:
            # Fall through to the helper-plot dirs so an @-mention of
            # a plot like ``residuals_lm1.png`` (which lives in
            # ``.nora/runs/<id>/_nora_plots/``) resolves correctly.
            runs_root = self.cwd / ".nora" / "runs"
            if runs_root.is_dir():
                try:
                    for run_dir in runs_root.iterdir():
                        if (visible_runs is not None
                                and run_dir.name not in visible_runs):
                            continue
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
            run_script = find_run_dir_script_by_name(
                self.cwd, safe_name, visible_run_dirs=visible_runs,
            )
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

        # @-mention is an explicit researcher action — they clicked a
        # row in the Files panel / mention dropdown vouching for this
        # file. If the target is a cwd top-level file, fold it into
        # the provenance manifest so a subsequent ``read_attached_file``
        # /``submit_script_file`` doesn't reject it as sandbox-output.
        # This closes the folder-backed-session gap: a researcher who
        # adds ``analysis_v2.py`` to their project dir outside Nora
        # between sessions can now @-mention it and have it be
        # recallable — without this hook, the manifest's "first-open
        # snapshot only" rule made externally-added files invisible
        # to recall even though they appear in the mention dropdown.
        # Run-dir files (helper plots, run-dir scripts) are not in
        # the cwd top-level scope and never have been gated by the
        # manifest, so skip them.
        target_resolved = target.resolve()
        if target_resolved.parent == cwd_resolved:
            try:
                from nora.file_provenance import mark_known
                mark_known(self.cwd, [target_resolved.name])
            except Exception:  # noqa: BLE001 — provenance is best-effort
                pass

        ext = target.suffix.lower()
        target_str = str(target_resolved)
        if ext in _INLINE_SCRIPT_EXTS:
            try:
                content = target.read_bytes()
            except OSError as e:
                return {"ok": False, "reason": f"read failed: {e}"}
            # Dedup by absolute path so two run-dir scripts that
            # happen to share a display name (e.g. both lack a
            # label.txt and fall back to ``script_<short_id>.py``,
            # which disambiguation already widens, but the same
            # pattern protects against any future collision) are
            # both stageable. Fall back to name when the staged
            # entry pre-dates the path field.
            for staged in runner.pending_script_attachments:
                staged_path = staged.get("path")
                matched = (
                    staged_path == target_str
                    if staged_path
                    else staged.get("name") == safe_name
                )
                if matched:
                    return {
                        "ok": True,
                        "name": safe_name,
                        "kind": "script",
                        "already_attached": True,
                    }
            _stage_script_for_next_turn(
                runner.pending_script_attachments, safe_name, ext, content,
                path=target_str,
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
            # Same path-aware dedup as the script branch — helper
            # plots in different run dirs commonly share a basename
            # (``coefficients.png``), so dedup-by-name would silently
            # noop the second mention even though the researcher
            # selected a different row.
            for staged in runner.pending_mentioned_images:
                staged_path = staged.get("path")
                matched = (
                    staged_path == target_str
                    if staged_path
                    else staged.get("name") == safe_name
                )
                if matched:
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
                # ``path`` lets ``delete_session_file`` drop this
                # entry by absolute path. Helper plots from
                # different run dirs can share a basename
                # (``coefficients.png`` in run A and run B), so
                # filtering by ``name`` alone would clear both
                # when only one was deleted.
                "path": target_str,
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
        can offer, as a flat list with no thumbnails.

        The @-mention dropdown is a re-attachment workflow: the
        researcher types ``@`` to find a prior script, plot, or
        upload and feed it back to Nora. It needs the FULL view
        — including ``<run_dir>/script.{do,R,py}`` files and
        ``<run_dir>/_nora_plots/`` outputs — so old scripts and
        helper plots stay reachable for re-attachment. The Files
        chip popup uses the tighter :meth:`list_session_files`
        view (panel-mode) for browsability, but mention is
        action-oriented and wants the full surface.

        The shape matches what the dropdown's filter/render code
        wants: ``[{name, kind, ext, mtime, size, path}]`` sorted by
        kind priority (data first, then scripts, graphs, logs) and
        mtime within each kind.
        """
        if self.cwd is None:
            return {"ok": True, "files": []}
        from nora.session_files import enumerate_session_files

        rows = enumerate_session_files(
            self.cwd,
            include_data=True,
            include_run_scripts=True,
            include_run_plots=True,
        )
        # The dropdown doesn't need base64 thumbnails — those only
        # ride for the Files panel render. Strip if any leaked in
        # (the enumeration helper doesn't add them, but keep the
        # filter defensively in case future enrichment shows up).
        files: list[dict[str, Any]] = [
            {k: v for k, v in entry.items() if k not in ("data", "mime")}
            for entry in rows
        ]
        return {"ok": True, "files": files}

    def send_feedback(
        self,
        message: str,
        reply_to: str | None = None,
        subject: str | None = None,
    ) -> dict[str, Any]:
        """POST a feedback note to the maintainer's Web3Forms
        endpoint.

        Web3Forms forwards the submission to the inbox the access
        key is bound to. The key only allows submissions to that
        inbox, so embedding it in the distributed app is safe:
        extraction grants no capability beyond spamming the
        maintainer's own feedback channel, which the service
        already rate-limits.

        Privacy trade-off: feedback bodies transit a third-party
        server (web3forms.com) on the way to the maintainer. This
        is a deliberate carve-out for the feedback surface only.
        Chat history, datasets, and analysis state never leave
        the machine. The modal UI states this so the researcher
        consents at submit time.

        Text-only by design: Web3Forms' free tier doesn't
        forward file attachments, so the modal exposes no upload
        affordance. If attachment support is needed later,
        upgrading the plan or swapping endpoints is a localised
        change.

        Length cap: the message is truncated at 12k chars before
        leaving the machine. The modal's textarea enforces the
        same cap client-side (``maxlength="12000"``) and warns the
        researcher as they approach it, so this server-side cap is
        defense in depth against a paste that slips past the UI.
        Truncate rather than reject so a researcher who hits the
        cap mid-paste doesn't lose their note; the trailing marker
        tells the maintainer what happened.
        """
        text = (message or "").strip()
        if not text:
            return {"ok": False, "reason": "empty feedback"}
        # Backend cap mirrors the textarea's ``maxlength`` in
        # ``index.html`` AND the JS counter's ``FEEDBACK_MESSAGE_CAP``.
        # The UI side stops normal typing at the same limit; this
        # is the defence-in-depth path for any caller that reaches
        # the bridge endpoint directly (page JS bypassing the
        # textarea constraint, a future automation, etc.).
        #
        # The cap is generous for legitimate use — 8000 characters
        # is several thick paragraphs of bug report — but tight
        # enough that an accidental "paste my whole log / dataset
        # / .env" mishap is rejected at the door rather than
        # forwarded to web3forms.com. The privacy carve-out for
        # feedback is bounded by the rule "only this note leaves
        # your machine"; an uncapped textarea quietly widens that
        # bound to "whatever the researcher happened to paste".
        if len(text) > _FEEDBACK_MAX_MESSAGE_CHARS:
            return {
                "ok": False,
                "reason": (
                    f"feedback is too long ({len(text)} characters); "
                    f"the limit is {_FEEDBACK_MAX_MESSAGE_CHARS}. "
                    f"Please don't paste raw data, full logs, or "
                    f"large outputs into feedback — describe what "
                    f"you saw and the maintainer can ask for "
                    f"specifics if needed."
                ),
            }

        # Embedded Web3Forms access key. Bound to the maintainer's
        # inbox; extraction grants no capability beyond filling
        # that same inbox.
        access_key = "732eaf71-0eac-4f34-9f14-2aef3000ae4d"

        # Build a multipart/form-data body by hand. Stdlib has no
        # multipart builder, so the boundary is assembled
        # explicitly. The hex suffix is random per request so the
        # body bytes can never accidentally contain the delimiter.
        import secrets
        boundary = "----nora-feedback-" + secrets.token_hex(12)
        crlf = b"\r\n"
        parts: list[bytes] = []

        def _field(name: str, value: str) -> None:
            parts.extend([
                f"--{boundary}".encode("ascii"), crlf,
                f'Content-Disposition: form-data; name="{name}"'.encode("ascii"),
                crlf, crlf,
                value.encode("utf-8"), crlf,
            ])

        # Researcher-typed subject when present so the inbox can
        # be sorted at a glance ("Bug: ...", "Feature: ..."). Fall
        # back to a generic label only when blank, and cap length
        # so an accidentally pasted novel can't bloat the header.
        cleaned_subject = ""
        if isinstance(subject, str):
            cleaned_subject = subject.strip()[:120]
        _field("access_key", access_key)
        _field("subject", cleaned_subject or "Nora feedback")
        _field("message", text)
        _field("from_name", "Nora app")
        # ``_replyto`` magic field rewrites the Reply-To header of
        # the notification email so the maintainer's Reply button
        # goes to the researcher. Light validation only — must
        # contain ``@`` — so a typo doesn't poison the header.
        if isinstance(reply_to, str):
            cleaned_reply = reply_to.strip()
            if cleaned_reply and "@" in cleaned_reply:
                _field("_replyto", cleaned_reply)

        parts.extend([f"--{boundary}--".encode("ascii"), crlf])
        body = b"".join(parts)

        # Browser-shaped headers. Web3Forms sits behind Cloudflare,
        # and a generic urllib User-Agent gets served a 200-OK bot-
        # challenge HTML page instead of the JSON API response. A
        # recognizable Chrome UA + Accept-Language + Origin makes
        # the request pass Cloudflare's default heuristics without
        # needing JS execution. The UA identifies the kind of
        # client, not a specific user.
        import urllib.error
        import urllib.request
        req = urllib.request.Request(
            "https://api.web3forms.com/submit",
            data=body,
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/127.0.0.0 Safari/537.36"
                ),
                "Accept": "application/json, text/plain, */*",
                "Accept-Language": "en-US,en;q=0.9",
                "Origin": "https://api.web3forms.com",
                "Referer": "https://api.web3forms.com/",
            },
            method="POST",
        )
        status = 0
        raw = b""
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                status = resp.status
                raw = resp.read()
        except urllib.error.HTTPError as e:
            # Non-2xx — Web3Forms still puts a JSON error body on
            # most failures, so read it so we can surface the
            # actual rejection reason instead of a bare status.
            status = e.code
            try:
                raw = e.read() or b""
            except Exception:  # noqa: BLE001
                raw = b""
        except urllib.error.URLError as e:  # network / DNS / TLS failures
            return {
                "ok": False,
                "reason": f"could not reach feedback service: {e.reason}",
            }
        except Exception as e:  # noqa: BLE001 — surface any other failure
            return {"ok": False, "reason": f"send failed: {e}"}

        # Response can be one of three shapes:
        #   * JSON (``/ajax/<email>`` endpoint) — parse and look for
        #     ``success``.
        #   * HTML success page (``/el/<alias>`` endpoint after
        #     urllib follows the success redirect) — status is 2xx
        #     and the body is FormSubmit's thank-you page.
        #   * Error page or non-2xx — treat as failure.
        text_body = raw.decode("utf-8", errors="replace").strip()

        # Non-2xx: always an error. Surface a snippet so the cause
        # (404 from wrong URL, 422 from missing field, 5xx from the
        # service, etc.) is visible to the maintainer reading bug
        # reports.
        if not (200 <= status < 300):
            snippet = text_body[:400] if text_body else "(empty body)"
            return {
                "ok": False,
                "reason": f"rejected (HTTP {status}): {snippet}",
            }

        # 2xx response. Web3Forms always returns JSON on success
        # (``{"success": true, ...}``). A 2xx HTML body would be a
        # Cloudflare bot-challenge page slipping through despite
        # our browser-shaped headers — treat that as a failure
        # rather than reporting a phantom "sent" while no email
        # actually leaves.
        try:
            payload = json.loads(text_body) if text_body else None
        except Exception:  # noqa: BLE001
            payload = None
        if not isinstance(payload, dict):
            snippet = text_body[:300] if text_body else "(empty body)"
            return {
                "ok": False,
                "reason": f"unexpected response (HTTP {status}): {snippet}",
            }
        success_raw = payload.get("success")
        ok = (
            success_raw is True
            or (isinstance(success_raw, str)
                and success_raw.strip().lower() == "true")
        )
        if not ok:
            msg = payload.get("message") or "rejected"
            return {"ok": False, "reason": f"{msg} (HTTP {status})"}
        return {"ok": True}

    def open_external(self, url: str) -> dict[str, Any]:
        """Open ``url`` in the OS default browser, NOT inside the
        WKWebView. Used by the model picker's "view pricing" links and
        the auth screen's "get an API key" links so researchers don't
        lose their chat session navigating to anthropic.com or
        openai.com.

        Allowlist-gated — only URLs in ``PROVIDER_PRICING_URLS`` and
        ``PROVIDER_API_KEY_URLS`` are accepted. The bridge is reachable
        from page-rendered JS, so a compromised page (e.g., a malicious
        tool result that escaped sanitisation and somehow injected JS)
        MUST NOT be able to coerce Nora into opening attacker-
        controlled URLs.
        """
        allowed = set(PROVIDER_PRICING_URLS.values()) | set(
            PROVIDER_API_KEY_URLS.values()
        )
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
        # Gate every side effect on the keyring delete actually
        # succeeding. ``auth.delete_credential`` returns ``ok=False``
        # on Keychain backend failure (locked, denied prompt, securityd
        # error) — in that case the secret is still in the OS store
        # and the credential is still usable, so closing idle runners
        # or clearing the injected env would force a spurious auth
        # screen on the next turn while reporting "delete failed" to
        # the user. Match the success of the underlying call.
        if not res.get("ok"):
            return {**res, "auth": self._auth_status_payload()}
        # Close any IDLE runner that's bound to the now-unauthed
        # provider. BUSY runners get marked for close-after-turn
        # instead — interrupting the in-flight stream is worse than
        # letting it complete, but we MUST evict the cached provider
        # client once the turn finishes. Both the OpenAI and
        # Anthropic SDKs capture ``api_key`` at client construction
        # and reuse it until close, so without the deferred-close
        # path a busy runner would keep authenticating with the
        # deleted credential on every subsequent send in the same
        # process — effectively making "Delete API key" a no-op for
        # any session that happened to be mid-turn.
        for runner in list(self._runners.values()):
            if runner.provider != provider:
                continue
            if runner.is_busy():
                runner.mark_close_after_turn()
            else:
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
                "reason": "wait for the turn in flight to finish",
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
            self._persist_active_model(active)
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
        can show the right "stored where?" copy.

        Every render is a fresh keyring read (``force_refresh=True``
        on the FIRST per-provider lookup), so a credential the
        researcher deleted directly in Keychain Access — outside
        Nora — is noticed on the next page load instead of waiting
        until process restart. The fresh read repopulates the
        process-wide credential cache, so the SECOND and later
        ``has_credential`` calls in the same render (``detect_auth``
        and the Forget-button gate below) hit the cache and do NOT
        re-prompt — the burst-suppression contract is preserved.
        """
        from nora import auth as _auth
        from nora.provider import detect_auth as _detect

        providers: dict[str, dict[str, Any]] = {}
        for p in PROVIDER_DEFAULTS:
            # Single fresh read per provider per render. Repopulates
            # ``_CRED_CACHE`` so the subsequent ``_detect`` /
            # ``has_credential`` calls in this loop don't re-hit the
            # backend.
            state = _auth.credential_state(p, force_refresh=True)
            mode = _detect(p)  # "subscription" | "api_key" | "unknown"
            providers[p] = {
                "configured": mode != "unknown",
                "method": mode,
                # ``has_keyring_entry`` lets the auth screen show
                # "Forget" only when there's actually something to
                # forget (subscription auth has nothing to forget here).
                "has_keyring_entry": _auth.has_credential(p),
                # Tri-state status field for the auth screen. Lets the
                # frontend distinguish "definitely no credential" from
                # "keyring locked / denied / unavailable so we
                # genuinely don't know." Without this, a denied
                # Keychain prompt rendered as "Not configured" and the
                # researcher might re-paste a key that's already
                # present (or worse, dismiss the prompt thinking the
                # credential was forgotten when it's actually still
                # there). The boolean ``configured`` field above stays
                # available for callers that only need yes/no routing.
                "status": (
                    _auth.AUTH_STATE_CONFIGURED
                    if mode != "unknown"
                    else state
                ),
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
                "reason": "wait for the turn in flight to finish",
            }
        if model_id == active.model and new_provider == active.provider:
            return {"ok": True, "model": model_id, "unchanged": True}

        res = self._run_on_loop(active.swap_model(model_id, new_provider))
        if res is None or not res.get("ok"):
            return res or {"ok": False, "reason": "model switch failed"}
        # Save the runner we swapped, not whatever is focused now —
        # the swap is slow enough for the focus to change.
        self._persist_active_model(active)
        return {
            "ok": True,
            "model": model_id,
            "label": info.label,
            "context_window": info.context_window,
            "provider": new_provider,
            # Which session this applied to. Focus may have moved
            # during the swap, so JS checks before repainting.
            "cwd": str(active.cwd),
            # A cross-provider swap can clamp the effort (Anthropic
            # ``max`` has no OpenAI rung), and the bar itself is a
            # different ladder either way — hand both back so the JS
            # repaints without a second call.
            "effort": active.effort,
            "efforts": [
                {"id": e.id, "label": e.label}
                for e in efforts_for_provider(new_provider)
            ],
        }

    def set_effort(self, effort: str) -> dict[str, Any]:
        """Switch the focused session's reasoning-effort level.

        Validated against the ladder the session's provider actually
        offers — they differ (Anthropic has ``max``, OpenAI stops at
        ``xhigh``). Operates on THIS session's runner only, like
        ``set_model``. With no focused session it updates the bridge
        default so the next runner picks it up. Refused while THIS
        runner has a turn in flight.
        Persists ``active_effort`` to the runner's
        ``.nora/session_state.json`` so a reload restores the choice.

        On Anthropic a live session is closed and re-warmed on the
        next message (the Agent SDK takes effort only at launch);
        the payload carries ``conversation_rewarmed`` so the UI can
        say so. On OpenAI it just applies to the next request.
        """
        target_provider = (
            active.provider if (active := self._active_runner()) is not None
            else self._default_provider
        )
        if effort not in effort_levels_for_provider(target_provider):
            return {
                "ok": False,
                "reason": (
                    f"{target_provider} does not support effort "
                    f"level {effort!r}"
                ),
            }
        info = get_effort(effort)
        if active is None:
            self._default_effort = effort
            return {"ok": True, "effort": effort, "label": info.label}
        if active.is_busy():
            return {
                "ok": False,
                "reason": "wait for the turn in flight to finish",
            }
        if effort == active.effort:
            return {
                "ok": True, "effort": effort, "label": info.label,
                "unchanged": True,
            }
        res = self._run_on_loop(active.swap_effort(effort))
        if res is None or not res.get("ok"):
            return res or {"ok": False, "reason": "effort switch failed"}
        # Save the runner we swapped — see set_model.
        self._persist_active_model(active)
        return {
            "ok": True,
            "effort": effort,
            "label": info.label,
            # Which session this applied to — see set_model.
            "cwd": str(active.cwd),
            "conversation_rewarmed": bool(res.get("conversation_rewarmed")),
        }

    def _persist_active_model(self, runner: SessionRunner | None = None) -> None:
        """Save the runner's model + effort to
        ``.nora/session_state.json`` so the choice survives a restart.

        Pass ``runner`` if anything slow ran since you resolved it — a
        provider swap gives the researcher time to switch sessions, and
        the focused runner would then be the wrong one to save.
        Defaults to the focused runner.
        """
        active = runner if runner is not None else self._active_runner()
        if active is None:
            return
        try:
            from nora.session_state import write_session_state
            write_session_state(
                active.cwd, model=active.model, effort=active.effort,
            )
        except Exception:  # noqa: BLE001 — never let state write break a swap
            pass

    def list_sessions(self) -> dict[str, Any]:
        """Return a list of past Nora sessions living under
        ``~/.nora-sessions/``, sorted most-recently-worked first. Each
        entry carries the absolute path, the directory name, the
        creation timestamp, the last-activity timestamp (drives sort),
        the names of the data files inside, and the on-disk size in
        bytes. Also flags the session that's currently loaded.

        Folder-backed sessions (opened via ``choose_folder``) are
        surfaced alongside the staged sessions with ``kind="folder"``
        and the same ``last_activity``-driven sort. Without this the
        sidebar can't navigate back to a project directory the
        researcher opened earlier in the session, even though Nora
        has been actively writing chat history and result rows inside
        ``<folder>/.nora/``.
        """
        current = str(self.cwd.resolve()) if self.cwd else None
        entries: list[dict[str, Any]] = []
        from nora.schema import DATA_EXTENSIONS as _DATA_EXTS
        seen_paths: set[str] = set()

        def _build_entry(
            child: Path, *, kind: str,
        ) -> dict[str, Any] | None:
            try:
                stat = child.stat()
            except OSError:
                return None
            # Timestamp parsing — staged dir names look like
            # `20260422T160059Z_f13630f4`. Fall back to mtime if the
            # prefix doesn't match (user-named folder-backed sessions
            # almost never match the timestamp shape).
            ts = _parse_session_timestamp(child.name) or stat.st_mtime
            # Last activity = mtime of chat_history.jsonl (appended on
            # every turn), fallback to dir mtime, fallback to creation.
            # Used for sort order so the most-recently-worked session
            # rises to the top regardless of when it was created.
            try:
                hist_stat = (child / ".nora" / "chat_history.jsonl").stat()
                last_activity = hist_stat.st_mtime
            except OSError:
                last_activity = stat.st_mtime or ts
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
            pinned = state.pinned if state is not None else False
            pinned_at = state.pinned_at if state is not None else ""
            # ``size`` drives the delete-confirm dialog so the
            # researcher knows how much data is about to be wiped.
            # Folder-backed sessions don't get a delete button (the
            # backend rejects rmtree on anything outside SESSIONS_ROOT
            # and the sidebar hides the affordance), so the recursive
            # walk would be pure cost — on a real project dir with
            # node_modules / .venv / build artefacts, _dir_size can
            # stat tens of thousands of files just to fill a field
            # nothing reads.
            size = 0 if kind == "folder" else _dir_size(child)
            return {
                "path": str(child.resolve()),
                "name": child.name,
                "timestamp": ts,
                "last_activity": last_activity,
                "datasets": datasets,
                "size": size,
                "title": _session_title(child),
                "custom_name": custom,
                "pinned": pinned,
                "pinned_at": pinned_at,
                # ``kind`` distinguishes staged (under SESSIONS_ROOT)
                # from folder-backed (registered via choose_folder).
                # The page uses this to render a different icon /
                # disable the delete-session affordance on folder-
                # backed entries (deleting a project dir is not
                # something the sidebar should offer).
                "kind": kind,
            }

        if SESSIONS_ROOT.exists():
            for child in SESSIONS_ROOT.iterdir():
                if not child.is_dir():
                    continue
                entry = _build_entry(child, kind="staged")
                if entry is None:
                    continue
                entries.append(entry)
                seen_paths.add(entry["path"])

        # Folder-backed sessions (opened via choose_folder). The
        # registry stores absolute paths; ``list_entries`` filters out
        # paths that no longer exist so a deleted project dir vanishes
        # from the sidebar naturally.
        try:
            from nora.external_sessions import list_entries as _list_external
            for ext_entry in _list_external(SESSIONS_ROOT):
                folder = Path(ext_entry["path"])
                entry = _build_entry(folder, kind="folder")
                if entry is None:
                    continue
                if entry["path"] in seen_paths:
                    continue
                entries.append(entry)
                seen_paths.add(entry["path"])
        except Exception:  # noqa: BLE001 — registry is supplementary
            pass

        # Two-tier sort, all descending so reverse=True drops out:
        #   1. Pinned sessions sit ahead of unpinned. With ``reverse=
        #      True`` and a bool first key, True (pinned) > False
        #      (unpinned), so pinned rows come first.
        #   2. Within pinned: sort by ``pinned_at`` desc — most
        #      recently pinned floats to the very top. Unpinned rows
        #      carry an empty pinned_at in the key so a stale stamp
        #      (left behind after an unpin) doesn't perturb the
        #      unpinned group's last_activity ordering.
        #   3. Within unpinned: ``last_activity`` desc — the existing
        #      most-recently-worked-first behaviour.
        def _sort_key(e: dict[str, Any]) -> tuple[bool, str, float]:
            is_pinned = bool(e.get("pinned", False))
            stamp = e.get("pinned_at", "") if is_pinned else ""
            return (is_pinned, stamp, float(e.get("last_activity", 0.0)))
        entries.sort(key=_sort_key, reverse=True)
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
        sessions_root = SESSIONS_ROOT.resolve()
        # Must be a *direct* child of SESSIONS_ROOT — not the root
        # itself, and not a nested directory inside a session.
        # ``_is_within`` returns True for ``target == sessions_root``
        # (Path.relative_to returns ``'.'`` for equal paths) and for
        # any nested path beneath a session, both of which would let
        # rmtree wipe far more than the caller named. The narrower
        # check below requires the parent to be exactly SESSIONS_ROOT.
        if target == sessions_root or target.parent != sessions_root:
            return {
                "ok": False,
                "reason": (
                    "must be a direct session directory under "
                    "~/.nora-sessions/"
                ),
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
        # Two acceptable shapes for ``target``:
        #
        # 1. A direct child of SESSIONS_ROOT (staged sessions). Same
        #    narrow gate ``delete_session`` uses; ``_is_within`` is
        #    too loose (it accepts ``target == sessions_root``
        #    itself, which would set cwd to the directory containing
        #    every session and break cross-session isolation, and any
        #    nested subdir beneath a session, which would spawn a
        #    runner whose cwd doesn't see the session's ``.nora/``
        #    state).
        #
        # 2. A folder registered via ``choose_folder``. Without this
        #    second gate, the sidebar could surface a folder-backed
        #    session (via ``list_sessions``) but clicking it returned
        #    "must be a direct session directory" — a dead chip.
        #    Registry membership is the durable record that the
        #    researcher previously opened this folder as a session.
        sessions_root = SESSIONS_ROOT.resolve()
        is_staged_session = (
            target != sessions_root and target.parent == sessions_root
        )
        is_registered_folder = False
        try:
            from nora.external_sessions import is_registered as _is_registered
            is_registered_folder = _is_registered(SESSIONS_ROOT, target)
        except Exception:  # noqa: BLE001 — registry is supplementary
            is_registered_folder = False
        if not (is_staged_session or is_registered_folder):
            return {
                "ok": False,
                "reason": (
                    "must be a session directory under ~/.nora-sessions/ "
                    "or a folder previously opened via the picker"
                ),
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
        # Same direct-child gate as ``delete_session`` /
        # ``switch_session``: ``_is_within`` alone would accept
        # SESSIONS_ROOT itself (relative_to of equal paths is
        # ``Path('.')``) and writing a custom_name at that level
        # would corrupt the sessions root with a phantom session_state.
        sessions_root = SESSIONS_ROOT.resolve()
        if target == sessions_root or target.parent != sessions_root:
            return {
                "ok": False,
                "reason": (
                    "must be a direct session directory under "
                    "~/.nora-sessions/"
                ),
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

    def set_session_pinned(self, path: str, pinned: bool) -> dict[str, Any]:
        """Toggle a session's pin-to-top flag so the sidebar surfaces
        frequently-revisited sessions ahead of the time-sorted list.

        Unlike :meth:`set_session_name`, this accepts BOTH staged
        sessions (under ``~/.nora-sessions/``) AND folder-backed
        sessions registered via :meth:`choose_folder`. Pinning is a
        non-destructive UI preference — there's no reason to deny it
        for a project directory the researcher already opened.

        Returns ``{ok, path, pinned}`` on success so the page can
        update the row without a full re-list (a ``list_sessions``
        refresh still follows for the sort).
        """
        if not path:
            return {"ok": False, "reason": "empty path"}
        try:
            target = Path(path).expanduser().resolve()
        except OSError as e:
            return {"ok": False, "reason": f"bad path: {e}"}
        if not target.is_dir():
            return {"ok": False, "reason": f"not a directory: {target}"}
        # Accept the same two shapes ``switch_session`` accepts: a
        # direct staged session under SESSIONS_ROOT, or a folder
        # previously registered via ``choose_folder``. Anything else
        # would let a JS caller seed a phantom ``.nora/`` skeleton
        # under an arbitrary path.
        sessions_root = SESSIONS_ROOT.resolve()
        is_staged_session = (
            target != sessions_root and target.parent == sessions_root
        )
        is_registered_folder = False
        try:
            from nora.external_sessions import is_registered as _is_registered
            is_registered_folder = _is_registered(SESSIONS_ROOT, target)
        except Exception:  # noqa: BLE001 — registry is supplementary
            is_registered_folder = False
        if not (is_staged_session or is_registered_folder):
            return {
                "ok": False,
                "reason": (
                    "must be a session directory under ~/.nora-sessions/ "
                    "or a folder previously opened via the picker"
                ),
            }
        try:
            from nora.session_state import set_pinned
            state = set_pinned(target, bool(pinned))
        except Exception as e:  # noqa: BLE001 — surface, don't crash
            return {"ok": False, "reason": f"could not save pin: {e}"}
        return {
            "ok": True,
            "path": str(target),
            "pinned": bool(state.pinned) if state is not None else bool(pinned),
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
        # Snapshot the cwd top-level into the staged-files manifest
        # so subsequent ``read_attached_file`` / ``submit_script_file``
        # calls can distinguish researcher-known files from sandbox-
        # output. Idempotent: re-opening an existing session merges
        # the snapshot with the existing manifest, so this also
        # serves as the upgrade path for sessions that pre-date the
        # provenance feature. Best-effort — a manifest write failure
        # mustn't block the session opening.
        try:
            from nora.file_provenance import initialize as _init_staged
            _init_staged(new_cwd)
        except Exception:  # noqa: BLE001 — provenance must not block opens
            pass
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
        effort = self._initial_effort_for_session(cwd, provider)
        runner = SessionRunner(
            cwd=cwd, provider=provider, model=model, effort=effort,
        )
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

    def _initial_effort_for_session(self, cwd: Path, provider: str) -> str:
        """Pick the reasoning-effort level for a freshly-created
        runner: the session's recorded ``active_effort``, clamped to
        what ``provider`` actually offers, else the bridge default.

        Runs independently of the model restore — a session whose
        recorded model fell out of the catalog still gets its effort
        back. The clamp matters when the two disagree: a file
        recording Anthropic ``max`` against a session that now opens
        on OpenAI steps down to ``xhigh`` rather than sending a level
        the OpenAI client can't express.
        """
        try:
            from nora.session_state import read_session_state
            state = read_session_state(cwd)
        except Exception:  # noqa: BLE001
            state = None
        if state is None or not state.active_effort:
            # Fall back to the BRIDGE default, not the catalog one — a
            # researcher who set "low" on the landing screen shouldn't
            # get "xhigh" here — but still hold it to the ladder.
            return clamp_effort(self._default_effort, provider)
        if state.active_effort not in EFFORT_LEVELS:
            return clamp_effort(self._default_effort, provider)
        return clamp_effort(state.active_effort, provider)

    def _runner_for_cwd(self, cwd: Path) -> SessionRunner | None:
        """The runner for a named session, focused or not.

        Upload paths capture their target session before the slow
        part (a FileReader read in the browser, a multi-file copy
        loop here) and must stage against THAT runner.
        ``_active_runner`` would follow a focus change that landed
        mid-upload and attach the file to the wrong conversation.
        """
        return self._runners.get(str(cwd.resolve()))

    def _active_runner(self) -> SessionRunner | None:
        """Convenience accessor: the runner for the focused session."""
        if self.cwd is None:
            return None
        return self._runner_for_cwd(self.cwd)

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
    def _effort(self) -> str:
        active = self._active_runner()
        return active.effort if active is not None else self._default_effort

    @_effort.setter
    def _effort(self, value: str) -> None:
        active = self._active_runner()
        if active is not None:
            active.effort = value
        else:
            self._default_effort = value

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

        All-or-nothing: if any copy raises, the freshly-created
        session dir (and any files already copied into it) is
        removed before returning the error. Without that cleanup,
        a partial session under ``~/.nora-sessions/`` would later
        appear in the sidebar / global recall surfaces as if it
        were a real session, contradicting the docstring promise
        and confusing the researcher.
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
            shutil.rmtree(session, ignore_errors=True)
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

        All-or-nothing: see ``_stage_session`` for why every error
        path here also tears the session dir down.
        """
        session = _new_session_dir()
        try:
            for name, content in blobs:
                target = session / name
                # Defense in depth: never write outside the session.
                if target.resolve().parent != session.resolve():
                    shutil.rmtree(session, ignore_errors=True)
                    return {"ok": False, "reason": f"suspicious filename: {name!r}"}
                target = _disambiguate_target(session, name)
                target.write_bytes(content)
        except OSError as e:
            shutil.rmtree(session, ignore_errors=True)
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
        images: list[dict[str, Any]] | None = None,
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

        Image attachments are persisted as ``{data, mime}`` blobs
        inline (per-image cap below), the same shape ``tool_result``
        uses for plot thumbnails. Without this, a reload or session
        switch replays an image-only prompt as bare text and the
        researcher / model-visible transcript loses the evidence
        that was actually sent. The blobs are flagged as UI-only in
        :mod:`nora.context_count` so they don't inflate the chip's
        denominator. ``image_count`` is still emitted alongside for
        legacy histories that pre-date this persistence.
        """
        self._drop_trailing_orphan_user_message(runner.cwd)
        # The transcript needs to reflect every file/image the
        # model actually saw with this message. Three classes ride
        # the live turn:
        #   1. ``pending_script_attachments`` — uploaded scripts
        #      inlined as a context block (already persisted).
        #   2. ``pending_mentioned_files`` — @-mention notices for
        #      non-image session files. Shown as composer chips in
        #      the live UI; the runner prepends a "the researcher
        #      referenced these" notice to the prompt.
        #   3. ``pending_mentioned_images`` — @-mention vision
        #      blobs for PDF/PNG/JPG/etc. The runner merges them
        #      into the turn's image list; without persisting the
        #      bytes, a reload shows a bare text bubble even
        #      though the model saw vision input.
        # ``attachments`` is the unified chip list (existing JS
        # already replays it as filename chips); ``images``
        # carries both direct composer images and @-mentioned
        # vision blobs so the bubble shows real thumbnails on
        # reload, not a phantom "you sent an image".
        attached_names: list[str] = []
        for a in runner.pending_script_attachments:
            n = a.get("name")
            if n and n not in attached_names:
                attached_names.append(n)
        for n in runner.pending_mentioned_files:
            if n and n not in attached_names:
                attached_names.append(n)
        for a in runner.pending_mentioned_images:
            n = a.get("name")
            if n and n not in attached_names:
                attached_names.append(n)
        record: dict[str, Any] = {
            "type": "user_message",
            "text": text,
            "session_cwd": str(runner.cwd),
        }
        if attached_names:
            record["attachments"] = attached_names
        # Combine direct composer images with @-mentioned vision
        # blobs so the persisted ``images`` list mirrors what the
        # model received. Both shapes carry ``data`` + ``mime``;
        # the ``name`` on a mentioned-image entry is already
        # reflected via ``attachments`` above.
        combined_images: list[dict[str, Any]] = []
        if images:
            combined_images.extend(images)
        if runner.pending_mentioned_images:
            combined_images.extend(runner.pending_mentioned_images)
        if combined_images:
            record["image_count"] = len(combined_images)
            persisted_images: list[dict[str, str]] = []
            for img in combined_images:
                data = img.get("data") if isinstance(img, dict) else None
                mime = img.get("mime") if isinstance(img, dict) else None
                if not isinstance(data, str) or not isinstance(mime, str):
                    continue
                # Same per-image ceiling as the Files-panel thumbnail
                # cap (see ``_enrich_files_panel_row``): 3 MB decoded,
                # which covers ~4 MB of base64. Anything above this
                # rides the live turn but isn't persisted — the
                # transcript shows the count without the bytes, which
                # is the same fallback the placeholder branch in the
                # plot-render path uses.
                if _b64_oversize(data, 3 * 1024 * 1024):
                    continue
                persisted_images.append({"data": data, "mime": mime})
            if persisted_images:
                record["images"] = persisted_images
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
            from nora.config import ensure_private_nora_dir
            history_dir = ensure_private_nora_dir(target_cwd)
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
        # Bind to this session. ``self.cwd`` follows the focused
        # session and can change mid-method — pywebview runs each JS
        # call on its own thread — so re-reading it below would hide
        # another session's results using this one's ids.
        cwd = runner.cwd
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

        history_path = cwd / ".nora" / "chat_history.jsonl"
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
            store = get_store(cwd)
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

        # Past the commit point. Rollback via ``unhide_results`` is no
        # longer in play, so it's safe to drop the verbatim
        # ``script_code`` column on the hidden rows. Rationale: a
        # researcher who pasted a credential or PII into a script
        # (``api_key = "sk-…"``, an SSN string-literal in an
        # exploratory block) would otherwise leave that text on disk
        # in ``results.db`` indefinitely — the row is invisible to
        # the model but the bytes remain, and anyone who later
        # exfiltrates the file (or queries it via ``sqlite3``) sees
        # the secret. Best-effort: a failure here doesn't undo the
        # rewind, since the rewind has already committed.
        try:
            store.purge_script_code(hidden_ids)
        except Exception:  # noqa: BLE001 — best-effort
            pass

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

        # 7. Regenerate the durable snapshot. The chat history was
        # truncated and several results were hidden, but
        # ``session_state.json`` still describes the discarded
        # branch — its ``last_user_message`` /
        # ``last_assistant_summary`` were pulled from a turn that no
        # longer exists, and ``recent_results`` lists rows we just
        # marked hidden. The next turn's natural rewrite will
        # eventually fix this, but the sidebar / session picker
        # reads the snapshot on every reload and session switch, so
        # without an immediate refresh a researcher who reloads
        # right after rewinding still sees the old branch in the
        # sidebar — breaks the "rewind takes effect now" promise.
        # Failures here are advisory; the rewind itself already
        # committed.
        try:
            from nora.session_state import write_session_state
            write_session_state(
                cwd, model=runner.model, effort=runner.effort,
            )
        except Exception:  # noqa: BLE001 — snapshot is advisory
            pass

        return {
            "ok": True,
            "truncated_from_index": turn_index,
            "hidden_count": hidden_count,
            # Which session this applied to. JS checks it before
            # sending the edit, so the edit can't land elsewhere.
            "cwd": str(cwd),
        }

    def get_chat_history(self) -> dict[str, Any]:
        """Return the persisted chat log for the active session so
        the UI can replay past messages after a session switch.
        Empty list if the session has no history yet.

        Heavy-payload fields (raw stdout/stderr from
        ``submit_script``, full tool-call inputs, long thinking
        blocks) are TRUNCATED before crossing to the WebView. A
        long script-heavy session can otherwise serialise tens of
        MB of base64'd plot thumbnails + raw R/Stata output into a
        single ``evaluate_js`` payload, which freezes the renderer
        on session switch / reload. The on-disk
        ``chat_history.jsonl`` keeps the un-truncated bytes — that
        file is the model's source of truth on warm start, NOT the
        replay surface. The replay UI just needs enough to
        reconstruct the visible cards.
        """
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
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    events.append(_trim_event_for_replay(rec))
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

    def _policy_summary(self, cwd: Path | None = None) -> dict[str, Any]:
        """Compact summary of a session's policy + datasets for the
        topbar footer.

        ``cwd`` defaults to the focused session. Pass it if you
        captured a session earlier, so a focus change can't swap in
        another session's datasets.
        """
        target = cwd if cwd is not None else self.cwd
        if target is None:
            from nora.policy import DEFAULT_MAX_DEPTH
            return {"default_max_depth": DEFAULT_MAX_DEPTH, "datasets": []}
        from nora.system_prompt import scan_datasets as _scan_datasets
        policy = load_policy(target)
        return {
            "default_max_depth": policy.default_max_depth,
            "datasets": [
                {
                    "name": p.name,
                    "ceiling": get_max_depth(policy, p.name),
                    "explicit": has_explicit_policy(policy, p.name),
                }
                for p in _scan_datasets(target)
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
# decoded bytes both live on the heap. 1 GB peaks around 3–4 GB total,
# which is comfortable on a 16 GB Mac and tight (but workable) on an
# 8 GB Mac during the transfer window. Researchers on entry-tier
# hardware with multi-GB datasets should prefer the Choose Files path
# regardless — it avoids the in-memory round-trip entirely.
#
# Files larger than this should use the native picker
# (:meth:`Bridge.choose_files` / :meth:`Bridge.add_files`), which uses
# ``shutil.copy2`` and has no size limit. The frontend gates on
# ``file.size`` before calling FileReader; this constant is the
# backend's matching defense-in-depth check, in case a client bypasses
# the JS gate or sends a forged base64 string from a non-browser path.
_DRAG_DROP_MAX_BYTES = 1024 * 1024 * 1024


# Server-side mirror of the JS-side ``COMPOSER_DATA_EXTS`` plus the
# ``ALLOWED_IMAGE_MIMES`` extension set in ``web/app.js``. The JS
# filter is the primary UX gate (skipped files surface as a friendly
# error in the chat); this set is defense-in-depth for any client that
# bypasses the JS check — most realistically a custom JS handler the
# researcher writes for testing, or a pywebview API call from a
# different origin if a future change loads remote content. Without it
# a forged ``add_dropped_files`` call could write arbitrary
# extensions (``.sh``, ``.app``, ``.dylib``) into the session dir,
# from which a researcher's own tooling might later auto-execute.
# Updating: keep in sync with ``COMPOSER_DATA_EXTS`` in
# ``src/nora/web/app.js``. Both lists carry the same per-set rationale.
_DRAG_DROP_ALLOWED_EXTS: frozenset[str] = frozenset({
    # Data — must mirror schema.DATA_EXTENSIONS plus the JS aliases.
    ".csv", ".tsv", ".dta", ".rds", ".parquet", ".jsonl", ".ndjson",
    # Scripts the researcher might want alongside their data.
    ".do", ".r", ".py", ".ipynb",
    # Stata graph + log formats.
    ".gph", ".log", ".smcl",
    # R Markdown.
    ".rmd",
    # Image formats accepted by the chat composer (ALLOWED_IMAGE_MIMES
    # in app.js — png/jpeg/webp/gif).
    ".png", ".jpg", ".jpeg", ".webp", ".gif",
})


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


def _adaptive_backtick_fence(content: str) -> str:
    """Return a fence longer than any backtick run inside ``content``.

    Markdown's CommonMark fenced-code rule closes a fenced block on
    the first line that starts with at least N backticks where N is
    the opening fence's length. Hard-coding ``"```"`` lets a script
    that contains ``"```"`` anywhere (in a comment, docstring,
    embedded example, or prompt-injection payload) close the fence
    prematurely and inject the rest of the script as ordinary prompt
    text. We scan for the longest run of consecutive backticks and
    pick a fence one longer.
    """
    longest_run = 0
    current_run = 0
    for ch in content:
        if ch == "`":
            current_run += 1
            if current_run > longest_run:
                longest_run = current_run
        else:
            current_run = 0
    fence_len = max(3, longest_run + 1)
    return "`" * fence_len


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
        # Markdown fences close on the FIRST run of backticks of the
        # same length or longer. A fixed three-backtick fence is
        # therefore breakable: a script that contains ``` anywhere
        # (a comment, a docstring, an embedded example) closes the
        # fence prematurely, and everything after it is read as
        # ordinary prompt text — including any "[system] override"
        # the model is expected to follow. Pick a fence one backtick
        # longer than the longest backtick run in ``content`` so the
        # closer is unambiguous regardless of what's inside.
        fence = _adaptive_backtick_fence(content)
        block = (
            f"{header}{fence}{fence_lang}\n"
            f"{content}\n"
            f"{fence}\n"
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
    *,
    path: str | None = None,
) -> None:
    """Decode ``content_bytes`` as text (best-effort UTF-8) and append
    a stage entry. Decoding failures are recorded as a placeholder
    so the model still knows the file exists, even if we couldn't
    surface its contents.

    Per-file size cap is enforced here: scripts longer than
    ``_INLINE_SCRIPT_MAX_BYTES`` get their first chunk + a clear
    truncation marker. The on-disk copy is always full.

    ``path`` records the absolute on-disk path the staged content
    came from. ``delete_session_file`` uses it to drop staged
    entries by path rather than by display name — for run-dir
    scripts the staged ``name`` is the label-derived display name
    (e.g. ``linear_regression.py``) but the file on disk is
    ``script.py``, so a name-only match would leave deleted script
    content staged in memory and the next send would inline a file
    the researcher just deleted.
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
        "path": path,
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


# Caps on per-event payload fields the WebView replay path doesn't
# need at full length. The on-disk chat_history.jsonl keeps the
# un-truncated bytes — that file is the model's warm-start source.
# These caps only bound what crosses ``evaluate_js`` for the visible
# replay cards.
#
# ``_REPLAY_TEXT_ENVELOPE_CAP`` MUST be larger than the maximum
# sanitized envelope a ``submit_script`` tool result can carry, or
# else a moderately complex regression card's JSON gets truncated on
# warm-start replay → ``renderCanonicalResultTables`` can't parse it
# → the result table renders blank. The tool side caps the inline
# payload at ``_INLINE_PAYLOAD_BUDGET + _INLINE_MARKDOWN_BUDGET``
# ≈ 57 KB (see ``tools.py``); 96 KB gives ~70% headroom for JSON
# overhead and small fields (label, summary, transformations,
# plot info across many results), so cards never disappear on reload.
# Bumped from 64 KB alongside the markdown-budget bump from 30 KB
# to 45 KB so the linked-constraint margin stays comfortable.
_REPLAY_TEXT_ENVELOPE_CAP = 96 * 1024    # tool_result ``text`` JSON envelope
_REPLAY_RAW_OUTPUT_CAP = 1 * 1024 * 1024  # raw_stdout / raw_stderr per tool_result
_REPLAY_THINKING_CAP = 8 * 1024          # assistant_thinking text
_REPLAY_TOOL_INPUT_CAP = 8 * 1024        # tool_call input fields (script code)


def _trim_str_for_replay(s: Any, cap: int) -> Any:
    """Truncate a string to ``cap`` chars with a marker, leaving
    non-strings untouched."""
    if not isinstance(s, str) or len(s) <= cap:
        return s
    overflow = len(s) - cap
    return s[:cap] + f"\n\n[…{overflow} chars trimmed for replay; full output preserved on disk]"


def _trim_event_for_replay(evt: dict[str, Any]) -> dict[str, Any]:
    """Cap heavy-payload fields on a chat-history event before it
    rides ``evaluate_js`` to the WebView. Returns a shallow copy so
    the original record (used elsewhere in the bridge process) is
    unchanged.

    Cards still render, scripts/output still show in the collapsed
    panel; researchers who need the un-truncated text open the run
    directory or expand the entry from disk on demand.
    """
    if not isinstance(evt, dict):
        return evt
    t = evt.get("type")
    if t == "tool_result":
        out = dict(evt)
        # ``raw_stdout`` / ``raw_stderr`` carry the researcher's
        # full audit surface — they ride at ~1 MB / stream, far
        # above what normal regressions print, so the cap is a
        # safety belt against a runaway log, not a routine
        # trimmer. The model never sees these fields (stripped
        # from model-facing projections in ``context_count``).
        for field in ("raw_stdout", "raw_stderr"):
            if field in out:
                out[field] = _trim_str_for_replay(
                    out[field], _REPLAY_RAW_OUTPUT_CAP
                )
        # The ``text`` field is the JSON tool-result envelope.
        # Capping it too tightly breaks ``renderCanonicalResultTables``
        # on warm-start replay (truncated JSON → parse fail → blank
        # card). The cap here MUST exceed the worst-case sanitized
        # envelope produced by ``submit_script``; see the comment
        # on ``_REPLAY_TEXT_ENVELOPE_CAP``.
        if "text" in out:
            out["text"] = _trim_str_for_replay(
                out["text"], _REPLAY_TEXT_ENVELOPE_CAP
            )
        return out
    if t == "tool_call":
        out = dict(evt)
        inp = out.get("input")
        if isinstance(inp, dict):
            new_inp = dict(inp)
            # ``code`` (submit_script) and ``script`` are the
            # large fields. Cap each.
            for field in ("code", "script"):
                if field in new_inp:
                    new_inp[field] = _trim_str_for_replay(
                        new_inp[field], _REPLAY_TOOL_INPUT_CAP
                    )
            out["input"] = new_inp
        return out
    if t == "assistant_thinking":
        out = dict(evt)
        if "text" in out:
            out["text"] = _trim_str_for_replay(
                out["text"], _REPLAY_THINKING_CAP
            )
        return out
    return evt


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
      2. "<first> +N" when multiple datasets live in the dir.
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
        return f"{datasets[0]} +{len(datasets) - 1}"

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


# R-stderr boilerplate patterns. R uses ``stderr`` for warnings AND
# for package-load chatter — keeping it all dumps "Attaching package"
# banners, masked-objects lists, and tidyverse decorations into every
# regression card. The filter drops those while letting ``Warning``
# and ``Error`` lines through; the on-disk ``stderr.log`` is
# untouched for audit.
_R_PACKAGE_LOAD_RE = re.compile(
    r"^(Loading required package:|Attaching package:|Loading namespace:)"
)
_R_MASKED_HEADER_RE = re.compile(r"^The following objects? (is|are) masked")
# Tidyverse banners use either em-dash or Unicode horizontal-line
# decorations around "Attaching" / "Conflicts" labels.
_TIDYVERSE_BANNER_RE = re.compile(
    r"^[─━—]+\s*(Attaching core tidyverse|Conflicts)"
)
_TIDYVERSE_TICK_RE = re.compile(r"^[✔✖]\s")


def _filter_stderr_boilerplate(text: str) -> str:
    """Drop R package-load chatter while keeping statistically
    meaningful diagnostics. Returns input unchanged for empty stderr
    or streams without recognised boilerplate.

    Removed:
    - ``Loading required package:`` / ``Attaching package:`` /
      ``Loading namespace:`` lines.
    - ``The following object(s) is/are masked …`` header AND the
      indented symbol list that follows it (until the next
      non-indented non-blank line).
    - Tidyverse decorated banners (``── Attaching core tidyverse``,
      ``── Conflicts``, the ``✔`` / ``✖`` tick lines).

    Kept:
    - ``Warning message:`` / ``Warning:`` and continuation context
      (convergence, rank deficiency, singular fit, NA coercion).
    - ``Error in`` / ``Error:`` and surrounding context.
    - Anything that doesn't match a boilerplate pattern.
    """
    if not text:
        return text
    out: list[str] = []
    in_masked_block = False
    for line in text.splitlines(keepends=True):
        stripped = line.rstrip("\r\n")
        if in_masked_block:
            # Block continues across blanks and indented symbol-list
            # lines. The first non-indented non-blank line ends it —
            # but we don't consume that line; fall through so it can
            # be classified normally (it may start another masked
            # block or be real content).
            if not stripped.strip() or stripped[:1] in (" ", "\t"):
                continue
            in_masked_block = False
        if _R_MASKED_HEADER_RE.match(stripped):
            in_masked_block = True
            continue
        if _R_PACKAGE_LOAD_RE.match(stripped):
            continue
        if _TIDYVERSE_BANNER_RE.match(stripped):
            continue
        if _TIDYVERSE_TICK_RE.match(stripped):
            continue
        out.append(line)
    # Collapse leading blank lines left behind by filtering so the
    # card doesn't open with whitespace.
    while out and not out[0].strip():
        out.pop(0)
    return "".join(out)


# Per-stream ceiling on raw logs shipped to the WebView. Normal R /
# Stata / Python regression output sits comfortably in the low
# hundreds of KB; this cap is a safety belt against pathological
# runaway logs (a stuck loop printing megabytes a second) rather
# than a routine trimmer. Head+tail truncation with a marker
# preserves the call/data-shape context up top AND any final
# warnings or errors at the bottom; the on-disk ``stderr.log`` /
# ``stdout.log`` always carries the full bytes for audit.
_RAW_LOG_STREAM_CAP = 1 * 1024 * 1024  # 1 MB per stream


def _read_raw_logs(run_dir: str | None) -> tuple[str, str]:
    """Read ``stdout.log`` and ``stderr.log`` from a run dir, if they
    exist. Returns empty strings when the dir is missing or the
    files haven't been written.

    Each stream is capped at ``_RAW_LOG_STREAM_CAP`` (1 MB). For
    normal regression traces the cap is far above what scripts
    produce, so content passes through untouched. Pathological logs
    keep their head (75%) and tail (25%) with a truncation marker
    pointing at the full on-disk log.

    The raw stream is local-only — it never crosses to the model
    (see ``_trim_event_for_replay``, which leaves these fields out
    of the model-facing projection in ``context_count``). Cost is
    WebView memory, not inference tokens.

    ``stderr`` is filtered through ``_filter_stderr_boilerplate`` to
    drop R package-load chatter (loading banners, masked-objects
    blocks, tidyverse decorations) so the card stays readable.
    Warnings and errors pass through. ``stdout`` is returned
    verbatim.
    """
    if not run_dir:
        return "", ""
    cap = _RAW_LOG_STREAM_CAP
    # 75/25 head/tail split. The start carries call lines and data
    # shape; the end is where helper-printed summaries and trailing
    # warnings live. The middle is the cheapest stretch to drop.
    head_cap = (cap * 3) // 4
    tail_cap = cap - head_cap

    def _read_one(name: str) -> str:
        try:
            content = Path(run_dir, name).read_text(
                encoding="utf-8", errors="replace"
            )
        except OSError:
            return ""
        if len(content) <= cap:
            return content
        head = content[:head_cap]
        tail = content[-tail_cap:]
        dropped = len(content) - len(head) - len(tail)
        return (
            f"{head}\n"
            f"[… {dropped} bytes truncated from the middle; "
            f"full log at {run_dir}/{name} …]\n"
            f"{tail}"
        )

    stdout_text = _read_one("stdout.log")
    stderr_text = _read_one("stderr.log")
    if stderr_text:
        stderr_text = _filter_stderr_boilerplate(stderr_text)
    return stdout_text, stderr_text


def _materialize_cache_busted_index(web_dir: Path, index_path: Path) -> Path:
    """Write an ``index.bust-<id>.html`` whose script/link refs carry
    a per-launch ``?v=<build-id>`` query string. WKWebView caches
    file:// resources by full URL (including the query string), so a
    unique build-id per launch forces a fresh fetch of every JS/CSS
    asset and prevents "I restarted but the new code isn't running" —
    a real failure mode where users iterate on the frontend, restart
    the bridge, and still see stale rendering because WKWebView
    served the cached app.js.

    The build-id hashes the mtimes of every .js / .css / .html file
    under ``web_dir`` so logically-identical reloads reuse the same
    cache key, while real code changes invalidate it.

    Bundle-safety: in a packaged macOS .app, ``web_dir`` lives under
    ``Contents/Resources/nora/web/`` — sealed by codesign. Writing
    the bust file there at first launch modifies the bundle and
    breaks ``spctl --assess`` ("sealed resource is missing or
    invalid"), blocking Gatekeeper on clean installs. When ``web_dir``
    isn't writable we route the bust file to a process-private temp
    directory and inject a ``<base href>`` so the bust file (now
    outside ``web_dir``) still resolves the HTML's relative
    ``app.js`` / ``style.css`` / image refs back to the bundle's
    web_dir.

    Falls back to the original index_path on any error — cache busting
    is a polish feature, not a correctness one.
    """
    import hashlib
    import os
    import re
    import tempfile
    try:
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

        # Pick the output directory based on whether ``web_dir`` is
        # writable. Dev mode keeps the bust file co-located with
        # assets so relative refs resolve as before; packaged-app /
        # read-only-web_dir mode routes to a temp directory and adds
        # a ``<base>`` tag so relative refs still resolve.
        web_dir_writable = os.access(web_dir, os.W_OK)
        if web_dir_writable:
            out_dir = web_dir
            base_tag = ""
        else:
            out_dir = Path(tempfile.gettempdir()) / "nora-cache-bust"
            try:
                out_dir.mkdir(parents=True, exist_ok=True)
            except OSError:
                return index_path
            base_tag = (
                f'<base href="{web_dir.resolve().as_uri()}/" />\n'
            )

        # Wipe stale ``.index.bust-*.html`` siblings before generating
        # a fresh one. Without this, every launch leaves a new sibling
        # in ``out_dir`` and the directory accumulates indefinitely
        # (each iteration on app.js / style.css produced one). Also
        # avoids confusion when reading mtimes during debugging — only
        # the live bust-file should be present after startup.
        for stale in out_dir.glob(".index.bust-*.html"):
            try:
                stale.unlink()
            except OSError:
                # Best-effort cleanup — don't crash the launch if a
                # sibling is locked / read-only / already gone.
                continue

        html = index_path.read_text(encoding="utf-8")

        # Append ``?v=<build-id>`` to local script/link refs. Remote
        # URLs (Google Fonts preconnect, future CDN loads) are left
        # alone.
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
        if base_tag:
            # Inject just after <head> so the base applies to every
            # subsequent relative ref (scripts, stylesheets, images).
            html = html.replace("<head>", f"<head>\n  {base_tag}", 1)
        out = out_dir / f".index.bust-{build_id}.html"
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
                "/Users/you/Downloads) or a tilde path (~/Downloads), "
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
    parser.add_argument(
        "--doctor", action="store_true",
        help=(
            "Run the environment health check and exit. Useful for "
            "diagnosing why a script failed before launching the UI. "
            "Exit code is non-zero when the environment is unusable."
        ),
    )
    args = parser.parse_args()

    if args.doctor:
        # Short-circuit before any UI / bridge setup. The doctor only
        # needs ``env_detect`` and prints to stdout — keeps the
        # command usable as a shell-init wrapper that gates the .app
        # launch.
        from nora.doctor import main_cli as _doctor_main
        sys.exit(_doctor_main())

    cwd = _resolve_cwd(args.cwd)
    if cwd is not None:
        # Same privacy gate the folder picker applies (see
        # _reject_dangerous_cwd). The CLI is the other entry point
        # that can hand Nora an arbitrary directory, so it has to
        # honour the same boundary — without this, `nora ~` or
        # `nora /` silently grants the sandbox read+write over the
        # whole home tree (or worse), which is exactly what the
        # picker refuses.
        reason = _reject_dangerous_cwd(cwd)
        if reason is not None:
            print(f"nora: {reason}", file=sys.stderr)
            sys.exit(2)
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
