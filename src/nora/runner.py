"""Per-session execution runner.

A :class:`SessionRunner` owns everything that used to live as bridge
singleton state:

- the :class:`~nora.provider.ProviderSession` (Anthropic SDK client or
  OpenAI session)
- the asyncio :class:`~asyncio.Lock` that serialises turns within a
  single session
- the currently-running turn's :class:`~asyncio.Task` (so Stop only
  affects this session)
- ``needs_context_prefix`` (warm-start memory injection)
- ``active_model`` / ``provider``
- ``cwd`` (bound at construction; immutable for the runner's lifetime)
- mid-chat staging (script attachments, dataset diff snapshot)

The bridge holds a ``dict[str, SessionRunner]`` keyed by cwd. Switching
the visible session in the sidebar is a pure UI focus change — it does
NOT close any runner. A long-running turn in session A keeps making
progress while the UI shows session B; events fire through ``on_event``
which both persists to the runner's own cwd and streams to the page
(filtered by the active focus on the JS side).

Concurrency safety rests on two things:

1. **Per-runner asyncio task.** Each runner schedules its own
   ``run_turn`` coroutine via ``asyncio.run_coroutine_threadsafe``;
   tasks are sister tasks on the bridge's worker loop and complete
   independently.
2. **Per-task cwd via ContextVar.** ``run_turn`` enters
   :func:`nora.config.use_cwd` so every tool handler invoked under
   this turn (including SDK-spawned subtasks) reads the runner's cwd,
   not whichever session the UI is currently showing. Without this,
   two simultaneous turns in different cwds would race over
   ``config.get_cwd`` and trample each other's tool-execution paths.
"""

from __future__ import annotations

import asyncio
import base64
import json
import subprocess
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Awaitable, Callable

from nora.config import use_cwd
from nora.provider import (
    AuthFailure,
    ProviderSession,
    TurnDone,
    TurnError,
    open_session,
)
from nora.runtime.turn_context import use_turn_context
from nora.system_prompt import build_system_prompt, scan_datasets
from nora.text_safety import safe_text
from nora.tools import SERVER_NAME


# Cap on the per-runner cancelled-turn-id set. Bounded so a long-lived
# session that's cancelled often doesn't grow the set without limit.
# The set's only job is to drop late events from cancelled turns; once
# the model is far enough past a turn that no SDK / subprocess can
# still emit for it, dropping the id from the set is fine — any
# straggler that arrives after eviction passes through the dispatcher
# and renders, at worst as a brief flash before the staleness sweep
# clears it. 256 covers ~weeks of normal use.
_CANCELLED_TURN_ID_HISTORY = 256


# Plot vision: caps and allowed kinds. Plots that exceed the byte cap
# are dropped (the researcher still sees them on disk). The manifest
# kind is enforced against this allowlist — a future helper has to
# land here AND in the runtime libraries to be visible to the model.
#
# Privacy line: ONLY plots that are pure functions of MODEL OUTPUT
# (point estimates, CIs, predicted-response curves on a synthetic
# grid, coefficient comparisons across specifications) cross to
# Claude. Per-observation plots — residuals vs. fitted, Q-Q,
# scale-location, leverage — are deliberately NOT on this list:
# they expose individual residuals + fitted values, which together
# are essentially the row-level data the JSON sanitizer refuses to
# surface. An image side channel around SDC is still SDC bypass.
# ``residuals.png`` still gets written to disk on every
# ``nora.plot_residuals`` call so the researcher can inspect the
# diagnostic; the model just doesn't see the image.
_PLOT_KIND_ALLOWLIST: frozenset[str] = frozenset({
    "interaction",
    "coefficients",
    "estimate_comparison",
    "marginal_effects",
})
# Kinds we explicitly know about but deliberately keep researcher-
# only (see allowlist comment). Listed so ``_summarize_plot_helpers``
# can surface the fact that the plot was MADE without attaching the
# image — otherwise the model calls ``nora.plot_residuals(fitted)``
# and sees no acknowledgement at all, which has been observed to
# trigger retry loops.
_PLOT_KIND_RESEARCHER_ONLY: frozenset[str] = frozenset({
    "residuals",
})
_PLOT_MAX_BYTES = 2 * 1024 * 1024   # 2 MB / image
_PLOT_MAX_PER_TURN = 8
# Caps on metadata that comes back into the model's context. Labels
# and filenames written by user-authored scripts are data-origin
# strings — they go through ``safe_text`` before being interpolated
# into the next-turn prompt notice, same posture as every other
# string the sanitizer surfaces.
_PLOT_LABEL_MAX_LEN = 120
_PLOT_NAME_MAX_LEN = 80


# Type alias: bridge passes in an event dispatcher that takes a payload
# dict. The runner stamps ``session_cwd`` so the dispatcher can route
# (persist + decide whether to fire to the visible UI). Returning an
# awaitable so the dispatcher can do disk I/O without blocking the
# turn's await pipeline if it wants to.
EventDispatcher = Callable[[dict[str, Any]], None]


class SessionRunner:
    """Owns one session's execution state.

    Construction is cheap — it just records cwd/provider/model and
    creates the lock. The :class:`ProviderSession` is opened lazily on
    the first ``run_turn`` so a runner that's been clicked into but
    never sent a message pays no SDK cost.

    Lifetime: created lazily by the bridge on first session focus or
    first send; held in the bridge's runners dict; closed only when
    the bridge shuts down (or a future "delete session" path
    explicitly evicts it).
    """

    def __init__(
        self,
        cwd: Path,
        provider: str,
        model: str,
    ) -> None:
        self.cwd: Path = cwd.resolve()
        # Mutable: ``set_model`` may swap provider+model in place.
        self.provider: str = provider
        self.model: str = model
        # Lazy: opened on first send. ``ensure_session`` is idempotent.
        self._session: ProviderSession | None = None
        # Created without a running loop on the bridge thread; binds
        # to the worker loop when first acquired. Python ≥ 3.10 makes
        # this safe.
        self._send_lock: asyncio.Lock = asyncio.Lock()
        # Set by ``run_turn`` to its own ``current_task()`` so
        # ``interrupt`` (called from another thread via the bridge)
        # can cancel it.
        self._current_turn_task: asyncio.Task[None] | None = None
        # Turn identity. Each call to ``run_turn`` is assigned a
        # unique id by the bridge; the runner stamps every event with
        # that id so the dispatcher (and the JS event filter) can
        # drop late events from a turn the researcher cancelled —
        # even after a fresh send starts on the same session.
        # ``_current_turn_id`` is the in-flight one (or None when no
        # turn is running). ``_cancelled_turn_ids`` records ids the
        # researcher hit Stop on; events stamped with one of those
        # ids are dropped at the dispatcher and never reach the JS
        # / chat history. Bounded LRU so a long-lived session can't
        # grow the set without limit.
        self._current_turn_id: str | None = None
        self._cancelled_turn_ids: "OrderedDict[str, None]" = OrderedDict()
        # Turn ids whose ``run_turn`` coroutine has been scheduled on
        # the worker loop but hasn't claimed ``_send_lock`` yet — i.e.
        # ``send_message`` returned an id to the JS side, but the
        # coroutine is still queued behind the previous turn (or
        # behind the worker loop's normal scheduling latency). Stop
        # fired in this gap used to find ``_current_turn_id is None``
        # and report "no turn in flight" while the runner went on to
        # execute the turn anyway; tracking pending ids here lets
        # ``cancel_turn`` mark the right id cancelled even before the
        # coroutine starts, and ``run_turn`` notices the flag on entry
        # and bails before opening a session or hitting the API.
        # Protected by ``_turn_lock`` (same lock that guards the
        # cancellation set, so the pending → cancelled transition is
        # atomic with respect to ``register_turn_process`` etc.).
        self._pending_turn_ids: list[str] = []
        # Per-turn subprocess registry. Keyed on turn id; each entry
        # holds the Popen handles ``submit_script`` spawned during
        # the turn. ``cancel_turn`` walks this under the lock and
        # kills any survivors so the script actually halts when
        # Stop fires (closes the prior race where the asyncio task
        # was cancelled but the subprocess kept running because the
        # cancellation propagated through the asyncio queue while
        # the thread was still inside Popen.communicate).
        self._turn_processes: dict[str, list[subprocess.Popen[Any]]] = {}
        # Lock guarding the cancellation set + process registry as a
        # single atomic unit. interrupt() acquires it, marks the
        # turn cancelled, and pops the proc list out for killing —
        # so any concurrent ``register_turn_process`` call either
        # sees the cancellation flag (and kills the proc on the
        # spot) or appends to a list that ``interrupt`` already took.
        # Either way the proc gets killed; neither thread can hide
        # a live subprocess from the cancellation path.
        self._turn_lock: threading.Lock = threading.Lock()
        # Deferred-close flag: set by the bridge's ``delete_credential``
        # path when this runner is busy with a turn whose underlying
        # provider client was constructed with the now-deleted key.
        # ``run_turn``'s finally block honours the flag and closes the
        # session after the in-flight turn finishes, evicting the
        # cached client (the OpenAI SDK captures ``api_key`` at
        # ``AsyncOpenAI(...)`` construction time and reuses it until
        # the client is closed). Without this, subsequent sends on
        # the same runner would keep authenticating with the deleted
        # credential. The next ``ensure_session`` then opens a fresh
        # session, which fails cleanly with ``no API key configured``.
        self._pending_close: bool = False
        # Detached teardown tasks for SDK sessions abandoned by Stop.
        # The cancel branch of ``run_turn`` nils ``self._session``
        # synchronously and schedules ``stale_session.close()`` here so
        # a follow-up cancel can't kill the teardown mid-flight; the
        # list also keeps strong refs so the asyncio task isn't GC'd
        # before the underlying provider subprocess actually exits.
        # Pruned of completed entries on each enqueue so a long-lived
        # session that Stops often doesn't grow this without bound.
        self._stale_close_tasks: list[asyncio.Task[Any]] = []
        # Warm-start: the next turn after a fresh session open
        # prepends prior-turn memory. Set whenever we open a session
        # (initial or after a model swap that closes/reopens).
        self.needs_context_prefix: bool = True
        # Datasets snapshotted at session open; mid-chat additions
        # diff against this so the next turn announces them.
        self.known_datasets: frozenset[str] = frozenset()
        # Mid-chat script staging (.py / .do / .r / .rmd dropped into
        # the composer for THIS session).
        self.pending_script_attachments: list[dict[str, Any]] = []
        # Plot images captured from ``submit_script`` runs. Each
        # entry: {data: <base64>, mime: "image/png", name: <str>,
        # kind: <str>, label: <str>}. The runner reads
        # ``<run_dir>/_nora_plots/manifest.jsonl`` after each tool
        # result and appends manifest-listed images here. Consumed
        # on the next user turn — merged into the ``images`` list
        # passed to ``session.send`` so the model sees the plots
        # alongside the next prompt. Manifest-only — files in the
        # run dir that AREN'T in the manifest never reach the
        # model. That's the privacy line: only model-output plots
        # produced via ``nora.plot_residuals`` / ``plot_interaction``
        # cross; raw ``ggsave`` / ``plt.savefig`` stays local.
        self.pending_plot_images: list[dict[str, Any]] = []
        # @-mention staging: files the researcher pulled in by name
        # via the composer dropdown (instead of re-uploading). The
        # bytes are already on disk in this session. These lists
        # only carry what the next turn needs to know about them.
        # ``pending_mentioned_files`` becomes a one-line "the
        # researcher referenced these" notice. ``pending_mentioned_images``
        # rides the next turn as vision so the model can actually see
        # any plots / images the researcher pointed at by name.
        self.pending_mentioned_files: list[str] = []
        self.pending_mentioned_images: list[dict[str, Any]] = []
        # Per-token frozen snapshots of the pending lists. Populated
        # by ``freeze_pending_for_queue`` when the JS queues a
        # message client-side; consumed by ``restore_frozen_pending``
        # when the queued message fires. See the freeze/restore
        # docstrings below for the race the snapshots close.
        self.frozen_pending_attachments: dict[str, dict[str, list[Any]]] = {}

    def clear_pending_attachments(self) -> None:
        """Drop everything staged for the next turn.

        Called by the rewind path: the researcher revised an earlier
        message, so any attachments / @-mentions / plot images they
        had queued up for the *original* next turn are no longer
        relevant. Without this, the truncated chat would still inline
        a script the researcher attached three turns ago, which would
        confuse both the model (why is this script here?) and the
        researcher (didn't I delete that?).

        All four pending lists are reset together because they all
        ride the same next-turn boundary; a partial reset would leave
        the runner in a state where some prior staging survives and
        some doesn't, with no visible signal to the researcher.
        """
        self.clear_unsent_pending()
        # Queued-message frozen attachments belong to messages that
        # would have flushed AFTER the rewound point. Drop them too —
        # the rewound conversation no longer expects those messages.
        self.frozen_pending_attachments.clear()

    def clear_unsent_pending(self) -> None:
        """Drop the four ``pending_*`` lists WITHOUT touching queued-
        message frozen snapshots.

        Called by the session-switch path. The frontend wipes its
        staged composer chips on the way out of a session; this is the
        matching backend wipe of the lists those chips would have
        ridden in on. Frozen snapshots stay — they belong to messages
        the researcher already committed to send (they sit in the JS
        queue) and must still fire with their original attachments
        when the in-flight turn finishes. Mixing the two clears was
        what made the rewind path's broader wipe wrong for switch.
        """
        self.pending_script_attachments.clear()
        self.pending_mentioned_files.clear()
        self.pending_mentioned_images.clear()
        self.pending_plot_images.clear()

    # -------- queued-send attachment freezing --------
    #
    # When a turn is in flight and the user hits Send on a follow-up,
    # the JS queues the message client-side. Earlier code left the
    # backend's ``pending_*`` lists alone — but those lists are
    # global per-session, so attachments staged for a LATER queued
    # message would accumulate alongside the FIRST queued message's
    # attachments, and the running turn's terminal-event handler
    # would consume them all into whichever queued message fired
    # next. Net effect: queued message #1 swallowed message #2's
    # script chip, and message #2 fired with nothing attached.
    #
    # The fix: at queue time, JS calls ``freeze_pending_attachments``
    # on the bridge. The bridge moves the runner's current pending
    # lists into a per-token dict and clears the runner's state so
    # subsequent stages go to a fresh slot. When the queued message
    # fires, JS passes the token back; the bridge restores the
    # frozen state into ``pending_*`` just before send, then deletes
    # the entry. Two queued messages each get exactly the
    # attachments staged at THEIR queue moment.
    #
    # ``frozen_pending_attachments`` lives on the runner (not the
    # bridge) so the same per-session lock that protects the regular
    # pending lists also protects this one — and so a runner kill
    # / session swap drops the frozen state alongside the rest of
    # the runner.

    def freeze_pending_for_queue(self, token: str) -> None:
        """Snapshot the current pending_* lists under ``token`` and
        clear the runner's pending state.

        Called by the bridge when JS queues a message client-side.
        Subsequent attachment stages land in a fresh runner state.
        """
        self.frozen_pending_attachments[token] = {
            "scripts": list(self.pending_script_attachments),
            "mentioned_files": list(self.pending_mentioned_files),
            "mentioned_images": list(self.pending_mentioned_images),
            # ``plot_images`` are NOT included here — those are
            # computed from the previous turn's run dir manifest, not
            # researcher-staged. They belong to the NEXT turn that
            # fires regardless of which queued message it is.
        }
        self.pending_script_attachments = []
        self.pending_mentioned_files = []
        self.pending_mentioned_images = []

    def restore_frozen_pending(self, token: str) -> bool:
        """Restore the named frozen state into ``pending_*`` and
        delete the entry.

        Returns ``True`` if the token was found and restored,
        ``False`` if it was absent (a no-op token is fine — falls
        through to whatever ``pending_*`` looks like, which may be
        the case when JS misses the token round-trip).
        """
        frozen = self.frozen_pending_attachments.pop(token, None)
        if frozen is None:
            return False
        # Prepend rather than replace so any plot_images captured
        # mid-queue (from a finished run) still ride the queued
        # turn alongside the frozen researcher-staged attachments.
        self.pending_script_attachments = (
            frozen["scripts"] + self.pending_script_attachments
        )
        self.pending_mentioned_files = (
            frozen["mentioned_files"] + self.pending_mentioned_files
        )
        self.pending_mentioned_images = (
            frozen["mentioned_images"] + self.pending_mentioned_images
        )
        return True

    def discard_frozen_pending(self, token: str) -> None:
        """Drop a frozen entry without restoring it. Used when a
        queued message is cancelled (Stop fires, or a rewind drops
        the queue)."""
        self.frozen_pending_attachments.pop(token, None)

    # -------- session lifecycle --------

    def is_busy(self) -> bool:
        """True iff a turn is currently in flight on this runner.

        Includes pending turns — ones whose ``run_turn`` coroutine
        was scheduled on the worker loop but hasn't claimed
        ``_send_lock`` yet. The bridge's ``interrupt_turn`` keys off
        this; without the pending check, a Stop fired in the tiny
        window between ``send_message`` returning a turn id and
        ``run_turn`` actually starting would report "no turn in
        flight" while the runner went ahead and executed the turn.
        """
        t = self._current_turn_task
        if t is not None and not t.done():
            return True
        with self._turn_lock:
            return bool(self._pending_turn_ids)

    async def ensure_session(self) -> ProviderSession:
        """Open the underlying provider session if it isn't already.

        Idempotent. Builds the system prompt fresh against
        ``self.cwd`` so dataset listings reflect what's actually on
        disk for THIS session (not whichever cwd was last focused).
        """
        if self._session is None:
            system_prompt = build_system_prompt(
                self.cwd, SERVER_NAME, provider=self.provider,
            )
            self._session = open_session(
                self.provider,
                cwd=self.cwd,
                model=self.model,
                system_prompt=system_prompt,
                continue_conversation=False,
            )
            await self._session.open()
            self.needs_context_prefix = True
            self.known_datasets = frozenset(
                p.name for p in scan_datasets(self.cwd)
            )
        return self._session

    async def close(self) -> None:
        """Close the underlying provider session. Idempotent."""
        session = self._session
        self._session = None
        self._pending_close = False
        if session is not None:
            try:
                await session.close()
            except Exception:  # noqa: BLE001 — close-time errors aren't useful
                pass

    def mark_close_after_turn(self) -> None:
        """Request that the runner close its provider session as soon
        as the in-flight turn finishes.

        Bridge use case: ``delete_credential`` was called while this
        runner is busy. Interrupting the in-flight stream is worse
        than letting it complete, but we MUST evict the cached
        provider client once it's done — both the OpenAI and
        Anthropic SDKs capture the API key at client construction
        and reuse it until close, so without this flag the runner
        would keep authenticating with the deleted credential on
        every subsequent send.

        Idempotent. No-op if the runner has no live session.
        """
        if self._session is not None:
            self._pending_close = True

    async def swap_model(self, model_id: str, provider: str) -> dict[str, Any]:
        """Swap the active model.

        Same-provider: delegate to the session if it exists; the SDK
        supports in-place swap. Different provider: close the session
        so the next turn opens fresh against the new provider. The
        runner's cwd does NOT change.
        """
        if provider != self.provider:
            await self.close()
            self.provider = provider
            self.model = model_id
            return {"ok": True, "model": model_id, "provider": provider}
        # Same provider: try in-place swap.
        old_model = self.model
        self.model = model_id
        if self._session is None:
            return {"ok": True, "model": model_id, "provider": provider}
        try:
            res = await self._session.set_model(model_id)
        except Exception as e:  # noqa: BLE001 — SDK shape varies
            self.model = old_model
            await self.close()
            return {
                "ok": False,
                "reason": f"model switch failed: {e}. Conversation reset.",
            }
        if not res.get("ok"):
            self.model = old_model
            if self._session is not None:
                await self.close()
        return res

    def _capture_plots(self, run_dir: Path) -> None:
        """Scan ``<run_dir>/_nora_plots/manifest.jsonl`` and stage
        manifest-listed plot files as ``pending_plot_images``.

        The manifest is the allowlist: a file landing in the dir
        without an entry is invisible to the model. The manifest's
        ``kind`` is checked against ``_PLOT_KIND_ALLOWLIST`` so a
        rogue runtime modification can't introduce a new "kind"
        (e.g., "raw_histogram") that would slip past this gate.

        Per-entry ``_token`` re-validation: the executor's
        ``_filter_plot_manifest`` strips/validates tokens at run
        completion and rewrites the manifest with cleaned content.
        That rewrite CAN fail (the manifest lives in script-writable
        territory; a script can chmod it read-only or otherwise
        block the host's write). When the rewrite fails, the executor
        currently leaves the original manifest in place — which
        would let a forged ``{"kind": "coefficients", ...}`` entry
        slip through and stage a row-level plot for vision. Solution:
        the runner re-validates each entry's ``_token`` against the
        per-run token registered by the executor. Entries with a
        missing / wrong token are dropped HERE regardless of what
        the on-disk manifest contains. If no token is registered
        for the run (replay / re-attach / unknown provenance), all
        entries are dropped — fail closed.

        Idempotency: re-reading the same manifest is fine — every
        consumed plot list is dropped after a successful turn, and
        only newly-produced runs land plots in their own run_dir.
        """
        from nora.executor import get_run_token, RESULT_TOKEN_FIELD
        import secrets as _secrets

        expected_token = get_run_token(run_dir)
        try:
            plots_dir = run_dir / "_nora_plots"
            manifest = plots_dir / "manifest.jsonl"
            if not manifest.is_file():
                return
            entries: list[dict[str, Any]] = []
            for raw in manifest.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(entry, dict):
                    continue
                # Authenticity gate. Every entry MUST carry a
                # ``_token`` that matches the executor's per-run
                # token. The runtime libraries (R/Stata/Python
                # helpers) stamp every entry they emit; a forged
                # entry hand-crafted by the script body (bypassing
                # the helpers) cannot mint a valid token without
                # introspecting the runtime library's loaded
                # state — the existing trust model. Missing /
                # mismatched / no-registry-entry all fail closed
                # so a rewrite-blocked manifest can't smuggle a
                # row-level plot through this side channel.
                if expected_token is None:
                    continue
                got = entry.get(RESULT_TOKEN_FIELD)
                if not isinstance(got, str) or not _secrets.compare_digest(
                    got, expected_token,
                ):
                    continue
                # Strip the token from the in-memory copy so it
                # never rides on a staged plot's metadata.
                entry = {
                    k: v for k, v in entry.items()
                    if k != RESULT_TOKEN_FIELD
                }
                entries.append(entry)
        except OSError:
            return

        # Cap how many plots can flow per turn. Take the most recent.
        if len(entries) > _PLOT_MAX_PER_TURN:
            entries = entries[-_PLOT_MAX_PER_TURN:]

        plots_dir = run_dir / "_nora_plots"
        plots_dir_resolved = plots_dir.resolve()
        for entry in entries:
            kind = entry.get("kind")
            file = entry.get("file")
            if kind not in _PLOT_KIND_ALLOWLIST:
                continue
            if not isinstance(file, str) or not file:
                continue
            # Path safety: file must be a basename or relative
            # path inside ``_nora_plots/``. We resolve it and check
            # containment before reading bytes.
            target = (plots_dir / file).resolve()
            try:
                target.relative_to(plots_dir_resolved)
            except ValueError:
                continue
            if not target.is_file():
                continue
            # The model's vision input only accepts raster image
            # formats. Stata helpers may write PDF (or .gph) when
            # the PNG translator is missing — convert PDF → PNG
            # via sips here so the model still sees the plot. .gph
            # falls through to "researcher-only" silently.
            ext = target.suffix.lower()
            if ext == ".png":
                png_path: Path | None = target
            elif ext in (".pdf", ".eps"):
                from nora.plot_convert import png_for
                png_path = png_for(target)
                if png_path is None:
                    # Conversion failed — log a helper error so the
                    # model knows the plot was produced but can't be
                    # surfaced for vision.
                    self._log_pdf_conversion_failure(plots_dir, target)
                    continue
            else:
                # .gph / .svg / unknown — researcher-only.
                continue
            try:
                blob = png_path.read_bytes()
            except OSError:
                continue
            if len(blob) > _PLOT_MAX_BYTES:
                continue
            # Labels and filenames are user-authored-script output;
            # they reach the model on the next turn through the prompt
            # notice. Run them through ``safe_text`` before storing
            # so a label crafted to inject prompt instructions, or a
            # filename with control characters, can't leak past the
            # text-safety boundary every other data-origin string
            # respects.
            raw_label = entry.get("label")
            sanitized_label = safe_text(
                raw_label if isinstance(raw_label, str) else "",
                max_len=_PLOT_LABEL_MAX_LEN,
            )
            # Use the (possibly-converted) PNG name for display
            # but the sanitized basename so the model never sees a
            # raw script-supplied filename through the prompt notice.
            sanitized_name = safe_text(png_path.name, max_len=_PLOT_NAME_MAX_LEN)
            self.pending_plot_images.append({
                "data": base64.b64encode(blob).decode("ascii"),
                "mime": "image/png",
                "name": sanitized_name or "plot.png",
                "kind": kind,
                "label": sanitized_label,
            })

    def _log_pdf_conversion_failure(
        self, plots_dir: Path, pdf_path: Path,
    ) -> None:
        """Append a helper_errors.jsonl entry when a manifest-listed
        PDF couldn't be converted to PNG. tools._summarize_plot_helpers
        reads this and surfaces the failure in the model-visible
        tool result, so the model says "thumbnail couldn't be
        produced" instead of "thumbnail should be visible above"."""
        entry = {
            "helper": "_capture_plots",
            "error": "PDFConversionError",
            "message": (
                f"PDF plot {pdf_path.name} could not be rasterized to "
                "PNG; the researcher can open it via Show folder but "
                "I (the model) won't see it on the next turn"
            ),
            "fix": "ensure /usr/bin/sips is available (macOS default)",
        }
        try:
            errors_path = plots_dir / "helper_errors.jsonl"
            with errors_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError:
            pass

    def current_turn_id(self) -> str | None:
        """Return the id of the in-flight turn, or ``None`` if idle.

        Used by the bridge's ``interrupt_turn`` to capture the id
        that's about to be cancelled, then surface it in the JS
        response so the frontend can add it to ``cancelledTurnIds``.
        """
        return self._current_turn_id

    def is_turn_cancelled(self, turn_id: str) -> bool:
        """True iff ``turn_id`` has been marked cancelled on this runner.

        Cheap to call — the lookup is on a small bounded ordered dict.
        Used by the dispatcher (drop late events) and by
        ``submit_script`` after a long subprocess returns (skip
        sanitize / persist for cancelled-turn output).
        """
        # ``in`` on an OrderedDict is O(1), and we don't need the
        # lock for a read-only contains check: even a stale read
        # just defers the drop to the next event from the same turn.
        return turn_id in self._cancelled_turn_ids

    def register_turn_process(
        self, turn_id: str, proc: subprocess.Popen[Any],
    ) -> None:
        """Add ``proc`` to the registry for ``turn_id``, OR kill it
        immediately if the turn was cancelled before this call landed.

        The lock-protected check + append closes the race where Stop
        fires after ``Popen()`` returns but before the registration
        completes. Two interleavings:

        - Stop wins: ``cancel_turn`` set the cancel flag and emptied
          ``_turn_processes[turn_id]`` first. We see the flag and
          kill ``proc`` ourselves.
        - Register wins: we append before the cancel flag is set.
          ``cancel_turn`` will pop the list under the lock and kill
          everything inside it.

        Either path leaves the subprocess dead. Without the lock the
        prior code had a window where Stop's pop returned an empty
        list while the executor was about to append a Popen handle
        a microsecond later, and the subprocess survived.
        """
        kill_now = False
        with self._turn_lock:
            if turn_id in self._cancelled_turn_ids:
                kill_now = True
            else:
                self._turn_processes.setdefault(turn_id, []).append(proc)
        if kill_now:
            _kill_proc_quietly(proc)

    def register_pending_turn(self, turn_id: str) -> None:
        """Record a turn id whose ``run_turn`` coroutine was just
        scheduled but hasn't started executing yet.

        Called by the bridge synchronously, immediately before
        ``asyncio.run_coroutine_threadsafe``. Closes the gap where
        ``send_message`` had already returned the id to JS but the
        coroutine had not yet claimed ``_send_lock`` and set
        ``_current_turn_id``. Without this, a Stop fired in that
        window would find no turn in flight and the runner would go
        on to execute the cancelled turn anyway.
        """
        with self._turn_lock:
            self._pending_turn_ids.append(turn_id)

    def _consume_pending_turn(self, turn_id: str) -> bool:
        """Atomically transition ``turn_id`` from pending → running,
        returning True if the turn was already cancelled while pending
        (caller should bail before doing any work).

        Run as the very first thing inside ``run_turn`` after
        acquiring ``_send_lock``: removes ``turn_id`` from the pending
        list so a later ``cancel_turn(None)`` won't double-target it,
        and reports whether ``cancel_turn`` already marked it
        cancelled before the coroutine got the lock.
        """
        with self._turn_lock:
            try:
                self._pending_turn_ids.remove(turn_id)
            except ValueError:
                # Already drained (e.g., bridge shutdown bypassed the
                # normal path). Treat as if cancellation may have
                # raced; the in_cancelled check below still decides.
                pass
            return turn_id in self._cancelled_turn_ids

    def cancel_turn(self, turn_id: str | None = None) -> str | None:
        """Mark ``turn_id`` cancelled, kill its registered subprocesses,
        and request asyncio cancellation of the in-flight task.

        Returns the id that was cancelled (or ``None`` if no turn was
        in flight). The bridge surfaces the returned id to the JS
        event filter so late events stamped with it get dropped.

        ``turn_id=None`` resolves to the running turn's id, falling
        back to the most-recently-scheduled pending turn — so a Stop
        fired in the gap between ``send_message`` returning and
        ``run_turn`` claiming the lock still cancels the right turn.
        Pass an explicit id only when cancelling cross-thread (e.g.,
        from a deferred handler that captured the id earlier).
        """
        target = turn_id
        if target is None:
            target = self._current_turn_id
        if target is None:
            # No running turn — fall back to the latest pending id so
            # a Stop fired before ``run_turn`` claims the lock still
            # marks the right id cancelled. The pending → cancelled
            # transition under ``_turn_lock`` is what ``run_turn``
            # checks on entry to bail early.
            with self._turn_lock:
                if self._pending_turn_ids:
                    target = self._pending_turn_ids[-1]
        if target is None:
            return None

        # Atomically mark cancelled + extract the proc list. Once the
        # flag is set, any concurrent ``register_turn_process`` call
        # for the same id will see the flag and self-kill instead of
        # appending; so popping here gives us every proc we need to
        # touch. Also evict from the pending list (no-op if not
        # present) so a later cancel_turn(None) doesn't re-resolve to
        # this same id.
        with self._turn_lock:
            self._cancelled_turn_ids[target] = None
            # Bounded LRU eviction: keep at most _CANCELLED_TURN_ID_HISTORY.
            while len(self._cancelled_turn_ids) > _CANCELLED_TURN_ID_HISTORY:
                self._cancelled_turn_ids.popitem(last=False)
            procs = self._turn_processes.pop(target, [])
            try:
                self._pending_turn_ids.remove(target)
            except ValueError:
                pass

        # Kill outside the lock so a stuck ``proc.kill`` / ``wait``
        # can't block another thread's ``register_turn_process``.
        for proc in procs:
            _kill_proc_quietly(proc)

        # Cancel the asyncio task last — but ONLY if it's the running
        # turn we're targeting. Cancelling a still-pending id must
        # not tear down the previous turn's task; the pending turn
        # has no task yet, and ``run_turn`` will see the cancelled
        # flag when it eventually wakes up and bail without doing
        # any LLM work.
        #
        # ``cancel_turn`` is called from the bridge thread (not the
        # worker loop thread), so ``Task.cancel`` has to be scheduled
        # via the task's own loop's ``call_soon_threadsafe``. Reading
        # the loop off the task itself avoids a parameter every call
        # site would otherwise have to thread through.
        t = self._current_turn_task
        if (
            t is not None
            and not t.done()
            and self._current_turn_id == target
        ):
            try:
                loop = t.get_loop()
                loop.call_soon_threadsafe(t.cancel)
            except RuntimeError:
                # Loop closed between the read and the schedule;
                # nothing left to cancel.
                pass
        return target

    def interrupt(self) -> bool:
        """Backwards-compat alias for ``cancel_turn(None)``.

        Returns True iff a turn was actually cancelled. Kept because
        external call sites and tests reference this name.
        """
        return self.cancel_turn() is not None

    # -------- turn execution --------

    async def run_turn(
        self,
        text: str,
        images: list[dict[str, Any]] | None,
        on_event: EventDispatcher,
        build_context_prefix: Callable[[Path], str],
        build_script_prefix: Callable[
            [list[dict[str, Any]], Path], str
        ],
        turn_id: str,
    ) -> None:
        """Drive one chat turn for THIS session.

        Holds ``_send_lock`` so a second send_message in the same
        session queues behind the first; sends to OTHER sessions
        proceed in parallel because they hold different locks.

        Every event is stamped with ``session_cwd`` (this runner's
        cwd) AND ``turn_id`` before being handed to ``on_event``. The
        dispatcher uses the turn id to drop late events from a
        cancelled turn before they reach the JS / chat history; it
        always persists to the runner's own ``chat_history.jsonl`` for
        events from non-cancelled turns.

        Wraps the entire await pipeline in :func:`use_cwd` AND
        :func:`use_turn_context` so tool handlers see THIS runner's
        cwd + turn id, not whichever session is currently focused
        in the UI. ``submit_script`` reads ``current_turn_id()`` /
        ``register_turn_process`` from the turn context to register
        its subprocess into the runner's per-turn registry — that's
        what closes the Popen-vs-register race the prior local
        ``proc_box`` had.

        ``turn_id`` is generated by the bridge before the call so it
        can be returned to the JS-side ``send_message`` synchronously,
        and so a Stop fired before the first event arrives still has
        a stable id to mark cancelled.
        """
        cwd = self.cwd

        # Wrap every emitted event so the dispatcher and JS filter
        # see the turn id alongside the cwd. Local lambda rather
        # than mutating ``_stamp`` to avoid touching every other
        # ``_stamp`` call site in a refactor — the closure here is
        # cheap and keeps the runner's events tagged consistently.
        def emit(payload: dict[str, Any]) -> None:
            payload["turn_id"] = turn_id
            on_event(_stamp(payload, cwd))

        with use_cwd(cwd), use_turn_context(turn_id, self):
            async with self._send_lock:
                # Pending → running transition. If Stop fired in the
                # gap between ``send_message`` returning the id and
                # this coroutine winning the lock, ``cancel_turn``
                # already added our id to ``_cancelled_turn_ids``;
                # bail before opening a session or hitting the API.
                # The dispatcher would drop our events anyway, but
                # we still don't want to spend an LLM call on output
                # the researcher won't see.
                if self._consume_pending_turn(turn_id):
                    return
                # Claim the in-flight pointer only after winning the
                # send lock. Earlier this happened at function entry,
                # outside the lock — which meant a second
                # ``run_turn`` for the same session, scheduled while
                # the first was still running, would overwrite
                # ``_current_turn_task`` / ``_current_turn_id`` with
                # its own values while waiting on the lock. A Stop
                # call would then read the queued id and cancel the
                # task that wasn't actually doing anything yet,
                # leaving the in-flight turn untouched. Setting the
                # pointers here means at most one turn ever owns
                # them at a time and ``cancel_turn(None)`` always
                # targets the running one. The web UI's JS-side
                # ``pendingMessages`` queue (``app.js`` ~1541)
                # serialises sends one level up so the bug rarely
                # surfaced through the chat path, but the runner's
                # contract advertises lock-based serialisation —
                # other call sites (programmatic embeds, future API
                # surfaces) deserve to rely on it.
                self._current_turn_task = asyncio.current_task()
                self._current_turn_id = turn_id
                try:
                    session = await self.ensure_session()
                except Exception as e:  # noqa: BLE001
                    emit({
                        "type": "turn_error",
                        "message": f"session setup failed: {e}",
                    })
                    self._current_turn_task = None
                    self._current_turn_id = None
                    return

                # Memory: warm-start prefix on first turn after open.
                # Restored on cancel/error so a flaky first turn
                # doesn't lose the injection.
                prompt = text
                carried_prefix = False
                if self.needs_context_prefix:
                    prefix = build_context_prefix(cwd)
                    if prefix:
                        prompt = prefix + "\n\n" + text
                        carried_prefix = True
                    self.needs_context_prefix = False

                # Mid-chat dataset uploads: announce any new files on
                # disk that weren't in the system prompt's listing.
                try:
                    current_datasets = frozenset(
                        p.name for p in scan_datasets(cwd)
                    )
                except Exception:  # noqa: BLE001 — never let scan break a turn
                    current_datasets = self.known_datasets
                new_datasets = current_datasets - self.known_datasets
                carried_dataset_diff: frozenset[str] = frozenset()
                if new_datasets:
                    # Sanitize each filename before interpolation. The
                    # startup ``dataset_listing`` already does this; the
                    # mid-turn diff path is the same threat surface (a
                    # researcher dropping a file named with embedded
                    # newlines / fake "###System:" markers / bidi
                    # overrides into cwd would otherwise reformat the
                    # next turn's prompt). Any name that sanitises to
                    # empty is dropped.
                    safe_added = [
                        s for s in (safe_text(n) for n in sorted(new_datasets))
                        if s
                    ]
                    if safe_added:
                        added_lines = "\n".join(
                            f"  - {n}" for n in safe_added
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
                self.known_datasets = current_datasets

                # @-mention pull-in, mid-chat script attachments, and
                # plot/image carries are populated from the BRIDGE
                # thread (researcher drops a file, @-mentions a name,
                # adds an image). The worker thread reads + clears
                # below. Without coordination, the previous pattern
                # had a race:
                #
                #   1. worker reads / iterates `self.pending_X`
                #   2. bridge appends a new entry to the SAME list
                #   3. worker assigns `self.pending_X = []`
                #
                # Anything appended between steps 1 and 3 was processed
                # AFTER the worker built its prefix block but BEFORE the
                # reset — so it landed in NEITHER the prompt nor the
                # next-turn queue. Atomic-swap closes the race: capture
                # the list reference and IMMEDIATELY rebind the
                # attribute to a fresh empty list. Any bridge append
                # racing the swap either lands in the OLD list (which
                # we still hold via the local) or in the NEW empty list
                # (where it gets picked up next turn). No item is lost.
                pending_mentions = self.pending_mentioned_files
                self.pending_mentioned_files = []
                pending_scripts = self.pending_script_attachments
                self.pending_script_attachments = []
                pending_mentioned_imgs = self.pending_mentioned_images
                self.pending_mentioned_images = []
                pending_plots = self.pending_plot_images
                self.pending_plot_images = []

                # @-mention pull-in: the researcher pointed at one or
                # more session-resident files by name. The files are
                # already on disk; surface a short notice so the model
                # treats them as the focus of THIS message rather than
                # generic ambient context.
                carried_mentioned_files: list[str] = []
                if pending_mentions:
                    # Same boundary check as the dataset diff above —
                    # @-mentioned filenames cross from the researcher's
                    # filesystem into the model's prompt and would
                    # otherwise let a hostile filename break out of the
                    # bracketed notice.
                    safe_mentions = [
                        s for s in (
                            safe_text(n) for n in pending_mentions
                        ) if s
                    ]
                    if safe_mentions:
                        mentioned_lines = "\n".join(
                            f"  - {n}" for n in safe_mentions
                        )
                        mention_notice = (
                            "[The researcher referenced these existing "
                            "session files in their message. Read or use "
                            "them as appropriate (no re-upload needed):\n"
                            f"{mentioned_lines}\n]\n\n"
                        )
                        prompt = mention_notice + prompt
                    carried_mentioned_files = list(pending_mentions)

                # Mid-chat script attachments. Consumed unconditionally
                # — failure of the turn does NOT carry these forward.
                # Earlier versions did carry forward on cancel/error to
                # spare the researcher a re-attach, but that left the
                # bridge holding state the JS chip already cleared, so
                # ``attach_session_file`` would respond "already
                # attached" for files no chip showed. Simpler model:
                # if the send fails, the researcher re-attaches and
                # tries again.
                if pending_scripts:
                    attach_block = build_script_prefix(
                        pending_scripts, cwd,
                    )
                    if attach_block:
                        prompt = attach_block + prompt

                # Plot vision: prepend any plots captured from a
                # previous turn's submit_script. Cleared after the
                # send returns so each plot is sent exactly once.
                merged_images: list[dict[str, Any]] = []
                attached_plots: list[dict[str, Any]] = []
                attached_mentioned_images: list[dict[str, Any]] = []
                if pending_mentioned_imgs:
                    merged_images.extend(pending_mentioned_imgs)
                    attached_mentioned_images = list(pending_mentioned_imgs)
                if pending_plots:
                    merged_images.extend(pending_plots)
                    attached_plots = list(pending_plots)
                    # Surface a short attachment notice so the model
                    # knows what it's looking at without having to
                    # guess from filenames alone.
                    notice_lines = [
                        f"  - {img.get('name', '?')} ({img.get('label', img.get('kind', 'plot'))})"
                        for img in pending_plots
                    ]
                    plot_notice = (
                        "[Result plots from your previous script are "
                        "attached:\n"
                        + "\n".join(notice_lines)
                        + "\nThese are model-output plots (coefficient "
                        "comparisons, predicted-response curves, "
                        "interactions, marginal effects). Per-"
                        "observation diagnostics like residuals are "
                        "produced for the researcher but deliberately "
                        "withheld from this attachment path.]\n\n"
                    )
                    prompt = plot_notice + prompt
                if images:
                    merged_images.extend(images)

                # Track terminal events. SDK glitches can drop the
                # stream without a terminal — synthesise one so the
                # JS state machine never wedges.
                saw_terminal = False
                # Distinguish a clean ``TurnDone`` from a yielded
                # ``TurnError`` / ``AuthFailure``. The Python
                # exception branches below already restore prefix /
                # diff / plot state so a thrown failure can be
                # retried with full context. The provider can also
                # emit those failures as normal events without
                # throwing (e.g. an HTTP-level auth failure caught
                # inside the SDK and surfaced via ``yield``); without
                # this flag the loop would treat that path as a
                # successful completion and consume the carried
                # context that the next attempt still needs.
                turn_failed_event = False
                # Set when a provider error indicates its server-side
                # conversation memory is gone (OpenAI's
                # ``previous_response_id`` expiry). The next turn must
                # re-prime via the warm-start context prefix, even if
                # this turn didn't itself carry one — otherwise we'd
                # restore prefix state only for the rare "first turn
                # after open" case and silently lose context for every
                # later turn that hits chain expiry. See the
                # ``context_reset`` field on ``TurnError`` in
                # ``provider/base.py``.
                turn_context_reset = False
                try:
                    async for evt in session.send(
                        prompt,
                        images=merged_images if merged_images else None,
                    ):
                        if isinstance(evt, (TurnDone, TurnError, AuthFailure)):
                            saw_terminal = True
                            if isinstance(evt, (TurnError, AuthFailure)):
                                turn_failed_event = True
                            if isinstance(evt, TurnError) and evt.context_reset:
                                turn_context_reset = True
                        # Capture any plots produced by submit_script
                        # so they're available on the NEXT user turn.
                        from nora.provider import ToolCallResult
                        if isinstance(evt, ToolCallResult) and evt.run_dir:
                            self._capture_plots(Path(evt.run_dir))
                        emit(_event_to_dict(evt))
                    if not saw_terminal:
                        # Treat a missing terminal as a failure too —
                        # the synthesised ``turn_error`` below tells
                        # the JS state machine the turn didn't
                        # complete, so the next attempt should rebuild
                        # the same context as a thrown-error retry.
                        turn_failed_event = True
                        emit({
                            "type": "turn_error",
                            "message": (
                                "the provider stream ended without a "
                                "result. Try again, or use Stop and "
                                "resend if the chat feels stuck."
                            ),
                        })
                    if turn_failed_event:
                        # Same restoration posture as the cancel /
                        # exception branches: a failed turn means the
                        # researcher's next attempt should see the
                        # warm-start prefix, dataset diff, and plot
                        # attachments that this turn consumed but
                        # never got to use. Mentioned files / images
                        # do NOT carry — the composer chip already
                        # cleared on send, so re-prepending would
                        # smuggle attachments the researcher no
                        # longer sees.
                        if carried_prefix or turn_context_reset:
                            # ``carried_prefix``: standard retry path
                            # — re-arm so the prefix that THIS turn
                            # consumed but failed to land rides on
                            # the retry. ``turn_context_reset``: the
                            # provider's server-side memory just
                            # expired, so the next turn must
                            # re-prime EVEN IF this turn didn't
                            # itself carry a prefix. Both lead to
                            # the same flag.
                            self.needs_context_prefix = True
                        if carried_dataset_diff:
                            self.known_datasets = (
                                self.known_datasets - carried_dataset_diff
                            )
                        if attached_plots:
                            self.pending_plot_images = (
                                attached_plots + self.pending_plot_images
                            )
                    # Persist the durable session snapshot.
                    try:
                        from nora.session_state import write_session_state
                        write_session_state(cwd, model=self.model)
                    except Exception:  # noqa: BLE001
                        pass
                except asyncio.CancelledError:
                    # Carry the prefix and dataset-diff state back so
                    # the next turn rebuilds context correctly. Plots
                    # carry too because they're produced by tools and
                    # the model hasn't yet reasoned about them.
                    #
                    # Mentioned files / images do NOT carry. The
                    # composer chip already cleared on send, so the
                    # researcher no longer sees those attachments.
                    # Re-prepending them silently sneaks them into
                    # the next message, which violates the
                    # what-you-see-is-what-you-send contract. The
                    # researcher can re-attach if they want.
                    if carried_prefix:
                        self.needs_context_prefix = True
                    if carried_dataset_diff:
                        self.known_datasets = (
                            self.known_datasets - carried_dataset_diff
                        )
                    if attached_plots:
                        self.pending_plot_images = (
                            attached_plots + self.pending_plot_images
                        )
                    # Drop the SDK session so unfinished work from the
                    # cancelled turn can't leak into the next one. The
                    # asyncio cancel above stops THIS coroutine, but the
                    # underlying provider client (Claude CLI subprocess
                    # for Anthropic, HTTP client state for OpenAI) is
                    # still mid-round if Stop fired between a tool_use
                    # and its tool_result — and the SDK pumps that
                    # leftover round through on the next ``client.query``,
                    # so the researcher's NEW message lands behind a
                    # cancelled turn's submit_script firing one more
                    # time. Closing here forces ``ensure_session`` to
                    # rebuild on the next run_turn; the rebuild flips
                    # ``needs_context_prefix`` back on, so the next
                    # send re-injects history from chat_history.jsonl
                    # via build_context_prefix and the conversation
                    # stays continuous from the researcher's POV. Cost:
                    # the 1h prompt cache is dropped, so the post-Stop
                    # turn pays a one-time cache-write surcharge —
                    # acceptable on a path the researcher reaches only
                    # by explicitly hitting Stop. Detach the actual
                    # close() with ``create_task`` so a follow-up
                    # cancel can't kill the teardown mid-flight; the
                    # critical bit (``self._session = None``) happens
                    # synchronously before this handler returns. The
                    # detached task is retained on the runner so the
                    # GC doesn't drop it before the SDK subprocess
                    # actually exits.
                    stale_session = self._session
                    self._session = None
                    if stale_session is not None:
                        try:
                            loop = asyncio.get_running_loop()
                            self._stale_close_tasks.append(
                                loop.create_task(stale_session.close())
                            )
                        except RuntimeError:
                            # Loop already closing — nothing left to do;
                            # the subprocess will be reaped on process
                            # exit.
                            pass
                        # Trim completed teardown tasks so the list
                        # doesn't accumulate across many Stops in one
                        # session lifetime.
                        self._stale_close_tasks = [
                            t for t in self._stale_close_tasks if not t.done()
                        ]
                    emit({
                        "type": "turn_error",
                        "message": "cancelled",
                    })
                    self._current_turn_task = None
                    self._current_turn_id = None
                    return
                except Exception as e:  # noqa: BLE001
                    # Same posture as the cancel branch above: prefix
                    # and dataset-diff carry, plots carry, mentioned
                    # files/images do not. The composer cleared the
                    # chip on send; re-prepending the attachments
                    # would smuggle them into the next message.
                    if carried_prefix:
                        self.needs_context_prefix = True
                    if carried_dataset_diff:
                        self.known_datasets = (
                            self.known_datasets - carried_dataset_diff
                        )
                    if attached_plots:
                        self.pending_plot_images = (
                            attached_plots + self.pending_plot_images
                        )
                    emit({
                        "type": "turn_error",
                        "message": f"turn failed: {e}",
                    })
                finally:
                    self._current_turn_task = None
                    self._current_turn_id = None
                    # Drop the per-turn process registry slot so the
                    # dict doesn't accumulate entries from completed
                    # turns. ``cancel_turn`` already pops the slot for
                    # cancelled turns; here we cover the success /
                    # natural-error path.
                    with self._turn_lock:
                        self._turn_processes.pop(turn_id, None)
                    # Honour any deferred-close request from a
                    # mid-turn ``delete_credential``. ``close()``
                    # clears the flag and evicts the cached provider
                    # client whose api_key was constructed BEFORE the
                    # delete; the next ``ensure_session`` will see no
                    # credential and fail cleanly. Best-effort: a
                    # close-time error here mustn't mask the turn's
                    # actual result, but we DO need ``_pending_close``
                    # reset either way so the next turn doesn't
                    # repeatedly hit a stale flag.
                    if self._pending_close:
                        try:
                            await self.close()
                        except Exception:  # noqa: BLE001
                            self._pending_close = False


# ---------------------------------------------------------------------------
# Event + subprocess helpers
# ---------------------------------------------------------------------------


def _kill_proc_quietly(proc: subprocess.Popen[Any]) -> None:
    """Kill ``proc`` if it's still running, swallowing all errors.

    Used by ``cancel_turn`` and ``register_turn_process`` — both call
    sites are in the cancellation path where any kill failure is
    advisory (the process either died, never started, or will die on
    its own); raising would propagate to the bridge thread and could
    wedge other sessions' turns. Bounded ``wait`` so a stuck process
    doesn't hold the cancellation thread indefinitely.

    Tears down the whole process group (executor starts subprocess
    with ``start_new_session=True``): a plain ``proc.kill()`` only
    SIGKILLs the direct child, leaving any
    ``parallel::makeCluster`` / ``multiprocessing.Pool`` workers as
    orphans of init, still able to append result lines to disk.
    ``os.killpg(pgid, SIGKILL)`` reaches every descendant in the
    session. Fall back to ``proc.kill()`` if the process group lookup
    fails (race between Stop and process exit).
    """
    import os
    import signal
    try:
        if proc.poll() is None:
            try:
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                # Process already gone, or no PG (the subprocess never
                # detached into its own session — older code paths or
                # non-POSIX). Fall back to the per-process kill.
                try:
                    proc.kill()
                except Exception:  # noqa: BLE001
                    pass
    except Exception:  # noqa: BLE001
        return
    try:
        proc.wait(timeout=2)
    except Exception:  # noqa: BLE001
        # The kill went through but the wait is failing or timing
        # out; leave the process to be reaped by the OS.
        pass


def _stamp(payload: dict[str, Any], cwd: Path) -> dict[str, Any]:
    """Attach the runner's cwd to every event so the bridge can route
    (persist + render-vs-suppress) without consulting any global
    state. The cwd is the routing key the JS side uses to decide
    "is this for the session I'm currently looking at?"."""
    payload["session_cwd"] = str(cwd)
    return payload


def _event_to_dict(evt: Any) -> dict[str, Any]:
    """Translate a provider Event dataclass into the JSON-friendly
    dict the JS side expects. Mirrors the existing translator that
    used to live in ui.py — kept here so the runner is self-contained.
    """
    from nora.provider import (
        AssistantText,
        AssistantThinking,
        ToolCall,
        ToolCallResult,
    )

    if isinstance(evt, AssistantText):
        return {"type": "assistant_text", "text": evt.text}
    if isinstance(evt, AssistantThinking):
        return {"type": "assistant_thinking", "text": evt.text}
    if isinstance(evt, ToolCall):
        return {
            "type": "tool_call",
            "name": evt.name,
            "input": evt.input,
            "call_id": evt.call_id,
        }
    if isinstance(evt, ToolCallResult):
        out: dict[str, Any] = {
            "type": "tool_result",
            "call_id": evt.call_id,
            "text": evt.text,
            "is_error": evt.is_error,
        }
        if evt.run_dir is not None:
            out["run_dir"] = evt.run_dir
        if evt.language is not None:
            out["language"] = evt.language
        return out
    if isinstance(evt, TurnDone):
        return {
            "type": "turn_done",
            "input_tokens": evt.input_tokens,
            "output_tokens": evt.output_tokens,
            "cache_read_input_tokens": evt.cache_read_input_tokens,
            "cache_creation_input_tokens": evt.cache_creation_input_tokens,
            "cost_usd": evt.cost_usd,
            "post_turn_tokens": evt.post_turn_tokens,
        }
    if isinstance(evt, AuthFailure):
        return {"type": "auth_failure", "reason": evt.reason}
    if isinstance(evt, TurnError):
        return {"type": "turn_error", "message": evt.message}
    # Unknown shape — fall through with a best-effort representation.
    return {"type": "unknown", "repr": repr(evt)}
