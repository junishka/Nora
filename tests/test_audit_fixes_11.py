"""Regression tests for the eleventh batch of reviewer-flagged fixes.

The behaviors pinned here:

1. ``install_packages`` is gated by a hard UI consent step.
   The MCP tool handler awaits ``install_confirmation.request_confirmation``
   before calling the underlying installer. With no emitter registered
   (headless / unauthed UI), the gate fails closed: the install is
   refused. With an emitter that approves, the install proceeds. With
   one that denies, the handler returns ``rejected`` and never touches
   the package manager.

2. ``fireQueuedMessage`` no longer corrupts focused-session state when
   a background queue flushes. JS-only logic, so the test is structural:
   inspect the source for the new ``isFocused`` gate and absence of the
   unconditional ``activeLiveTurn = {...}`` write inside the queued-
   message path.

3. Folder-backed sessions opened via ``choose_folder`` survive in the
   session sidebar and are reachable through ``switch_session``. The
   external-sessions registry is the durable record.

4. WebP and GIF files dropped into the composer surface in the Files
   panel and can be recalled via ``read_attached_file``. The asymmetry
   between the accept side (composer + ui.upload) and the recall side
   (session_files classification + tools.read_attached_file) is closed.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# 1. install_packages confirmation gate
# ---------------------------------------------------------------------------


def test_install_packages_denied_without_emitter() -> None:
    """No bridge / no emitter → the tool refuses to install. This is
    the headless default: the system prompt's "ask the researcher
    first" instruction cannot be enforced without a UI surface, so the
    handler fails closed."""
    from nora.install_confirmation import clear_request_emitter
    from nora.tools import HANDLERS

    # Defensive: ensure no stale emitter from another test file.
    clear_request_emitter()

    payload = asyncio.run(HANDLERS["install_packages"]({
        "language": "Python",
        "packages": ["pandas"],
        "action": "install",
    }))
    body = json.loads(payload["content"][0]["text"])
    assert body["status"] == "rejected"
    assert "declined" in body["reason"].lower() or "did not respond" in body["reason"].lower()


def test_install_packages_proceeds_on_approval(monkeypatch) -> None:
    """An emitter that approves immediately → the tool calls the
    underlying installer. We patch ``_do_install`` so the test doesn't
    actually touch pip, and verify the handler reached it with the
    expected args."""
    from nora.install_confirmation import (
        clear_request_emitter,
        respond,
        set_request_emitter,
    )
    from nora.tools import HANDLERS
    from nora.package_installer import InstallResult, PackageStatus

    captured: dict[str, object] = {}

    async def fake_install(language, packages, action, proc_register=None):
        captured["language"] = language
        captured["packages"] = list(packages)
        captured["action"] = action
        return InstallResult(
            language=language, action=action, duration_seconds=0.0,
            statuses=tuple(
                PackageStatus(name=p, status="ok", detail="fake")
                for p in packages
            ),
            error=None, raw_stdout="", raw_stderr="",
        )

    monkeypatch.setattr(
        "nora.package_installer.install_packages", fake_install,
    )

    def auto_approve(token, _lang, _pkgs, _act):
        # Synchronously respond on the same call — the tool awaits a
        # future, so this resolves it immediately. Safe in tests.
        respond(token, True)

    set_request_emitter(auto_approve)
    try:
        payload = asyncio.run(HANDLERS["install_packages"]({
            "language": "Python",
            "packages": ["pandas"],
            "action": "install",
        }))
    finally:
        clear_request_emitter()

    body = json.loads(payload["content"][0]["text"])
    assert body["status"] == "ok"
    assert captured["language"] == "Python"
    assert captured["packages"] == ["pandas"]
    assert captured["action"] == "install"


def test_install_packages_refused_on_explicit_deny() -> None:
    """An emitter that denies → the handler returns ``rejected`` and
    never reaches the installer. Mirrors the user clicking Deny in
    the modal."""
    from nora.install_confirmation import (
        clear_request_emitter,
        respond,
        set_request_emitter,
    )
    from nora.tools import HANDLERS

    install_was_called: dict[str, bool] = {"flag": False}

    async def fake_install(*_args, **_kwargs):
        install_was_called["flag"] = True
        raise AssertionError(
            "installer must not run when the researcher denies"
        )

    # Patch on the module path the tool handler imports from.
    import nora.package_installer as _pi
    original = _pi.install_packages
    _pi.install_packages = fake_install  # type: ignore[assignment]

    def auto_deny(token, _lang, _pkgs, _act):
        respond(token, False)

    set_request_emitter(auto_deny)
    try:
        payload = asyncio.run(HANDLERS["install_packages"]({
            "language": "Python",
            "packages": ["pandas"],
            "action": "install",
        }))
    finally:
        clear_request_emitter()
        _pi.install_packages = original  # type: ignore[assignment]

    body = json.loads(payload["content"][0]["text"])
    assert body["status"] == "rejected"
    assert install_was_called["flag"] is False


def test_install_packages_timeout_defaults_to_deny() -> None:
    """An emitter that never responds → the await times out and the
    handler returns ``rejected``. The default 5-minute timeout is too
    long for a unit test; we pass a short timeout via the module
    constant override pattern."""
    from nora.install_confirmation import (
        clear_request_emitter,
        request_confirmation,
        set_request_emitter,
    )

    def no_response(_token, _lang, _pkgs, _act):
        return None  # never resolves the future

    set_request_emitter(no_response)
    try:
        result = asyncio.run(
            request_confirmation("Python", ["pandas"], "install", timeout=0.05)
        )
    finally:
        clear_request_emitter()
    assert result is False


def test_install_packages_emitter_failure_denies_without_waiting_for_timeout() -> None:
    """An emitter that RAISES (webview reloading or closing) must
    deny immediately, NOT wait for the per-request timeout. The
    ``request_confirmation`` body catches emitter exceptions and
    returns False before the await — but only if the bridge's
    emitter actually propagates the failure rather than swallowing
    it.

    Regression: the bridge's ``_emit_install_confirmation_request``
    used to swallow ``evaluate_js`` failures with a misleading
    comment about ``request_confirmation`` catching them. Because
    no exception escaped, the await sat on the future until the
    full timeout elapsed (default 5 minutes) — multi-minute UI
    hangs on every install attempt during a webview close /
    reload. This test asserts the emitter's exception path
    short-circuits the wait.
    """
    import time

    from nora.install_confirmation import (
        clear_request_emitter,
        request_confirmation,
        set_request_emitter,
    )

    def emitter_raises(_token, _lang, _pkgs, _act):
        # Mirrors what the real bridge's ``_emit_install_confirmation_request``
        # now does when ``evaluate_js`` fails — let the exception
        # propagate to ``request_confirmation``.
        raise RuntimeError("webview disappeared mid-call")

    set_request_emitter(emitter_raises)
    # Generous timeout that we want to NOT hit. If the bridge
    # regressed to swallowing emitter failures, this test would
    # block ~2 seconds instead of resolving in microseconds.
    timeout = 2.0
    start = time.monotonic()
    try:
        result = asyncio.run(
            request_confirmation(
                "Python", ["pandas"], "install", timeout=timeout,
            )
        )
    finally:
        clear_request_emitter()
    elapsed = time.monotonic() - start
    assert result is False
    # Generous margin against CI scheduler jitter. The real signal:
    # we finished MUCH faster than the timeout, proving the deny
    # came from the emitter-exception path, not the timeout path.
    assert elapsed < timeout / 4, (
        f"emitter exception should deny immediately, not after "
        f"timeout — observed {elapsed:.3f}s of a {timeout}s timeout"
    )


def test_bridge_install_emitter_propagates_evaluate_js_failure() -> None:
    """The bridge's ``_emit_install_confirmation_request`` must let
    a failing ``evaluate_js`` raise. ``request_confirmation`` is the
    one that catches the exception and resolves the future as
    denied; if the bridge swallows the failure, the deny never
    fires until timeout.

    Regression: the bridge's ``try/except`` around ``evaluate_js``
    used to ``pass`` with a comment claiming
    ``request_confirmation`` would catch the exception — but with
    the exception swallowed at the bridge, nothing reached
    ``request_confirmation``.
    """
    from nora.ui import NoraBridge

    class _FakeWindow:
        def evaluate_js(self, _src):
            raise RuntimeError("webview reloading")

    bridge = NoraBridge.__new__(NoraBridge)
    bridge._window = _FakeWindow()  # type: ignore[attr-defined]

    with pytest.raises(RuntimeError, match="webview reloading"):
        bridge._emit_install_confirmation_request(
            token="tok", language="Python",
            packages=["pandas"], action="install",
        )


def test_install_confirmation_modal_enter_does_not_unconditionally_approve() -> None:
    """The install-confirmation modal focuses Deny by default. A
    document-level keydown handler used to call ``respond(true)`` on
    every Enter, contradicting the focus default — researchers
    pressing Enter on the focused Deny button got an Approve anyway,
    flipping a safety modal into auto-approve.

    Pin the structural fix: the Enter branch must gate on
    ``document.activeElement === approveBtn``. Source-grep test
    because the modal's keyboard behaviour can't be exercised
    headlessly without spinning up a browser.
    """
    app_js = (Path(__file__).resolve().parent.parent
              / "src" / "nora" / "web" / "app.js").read_text(encoding="utf-8")

    # Extract the showInstallConfirmationModal function body to keep
    # the assertion local to THIS modal, not any other dialog in the
    # file that might also handle Enter.
    m = re.search(
        r"function showInstallConfirmationModal\([^)]*\)\s*\{(.*?)\n\}\n",
        app_js,
        re.DOTALL,
    )
    assert m is not None, "showInstallConfirmationModal not found in app.js"
    body = m.group(1)

    assert "denyBtn.focus()" in body, (
        "modal must focus Deny by default — the keyboard-default-safe "
        "posture that the Enter gate complements"
    )
    # Unconditional Enter-to-approve must NOT be there.
    assert "e.key === 'Enter' && !e.shiftKey" not in body, (
        "Enter must not unconditionally approve; that contradicts the "
        "Deny-by-default focus and turns Enter on Deny into Approve"
    )
    # The Enter branch must gate on activeElement === approveBtn.
    assert "document.activeElement === approveBtn" in body, (
        "Enter approval must require Approve to actually have focus; "
        "without the gate the modal's safety default is bypassed"
    )


# ---------------------------------------------------------------------------
# 2. fireQueuedMessage no longer overwrites focused activeLiveTurn
# ---------------------------------------------------------------------------


def _app_js() -> str:
    app_js = Path(__file__).resolve().parent.parent / "src" / "nora" / "web" / "app.js"
    return app_js.read_text(encoding="utf-8")


def _strip_js_comments(src: str) -> str:
    """Drop // and /* */ comments so source assertions match real code
    rather than prose describing the bug that motivated it."""
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.DOTALL)
    return re.sub(r"^\s*//.*$", "", src, flags=re.MULTILINE)


def test_live_turns_are_keyed_by_session_not_a_global() -> None:
    """The in-flight turn handle must be per-cwd, never a single
    global.

    A global meaning "the focused session's live turn" is only correct
    while one session runs. It is written by whichever session sent
    last and nothing rebinds it on a focus change, so with two sessions
    running it points at the wrong turn. That is a P1: Stop then
    blacklists the wrong turn id (see the regression test below), and a
    background queue flush steals the focused session's
    ``hasVisibleReply`` / disposable-turn cleanup.

    Keying by cwd removes the failure mode structurally, so this test
    pins the shape rather than any one call site's guard."""
    code = _strip_js_comments(_app_js())

    assert "const liveTurnByCwd = new Map();" in code, (
        "live turns must be tracked per-cwd in a Map"
    )
    assert "activeLiveTurn" not in code, (
        "no code path may reintroduce a single global live-turn "
        "handle; every read/write must name its session via "
        "getLiveTurn / setLiveTurn / clearLiveTurn"
    )


def test_fire_queued_message_registers_turn_under_its_own_session() -> None:
    """A queued message can fire AFTER the researcher has switched
    sessions, so its turn must be registered under the queue's own cwd
    — not promoted onto whatever session happens to be focused."""
    m = re.search(
        r"async function fireQueuedMessage\(.*?\n\}\n",
        _app_js(),
        re.DOTALL,
    )
    assert m is not None, "fireQueuedMessage function not found"
    body = _strip_js_comments(m.group(0))

    assert "setLiveTurn(cwd, localTurn);" in body, (
        "fireQueuedMessage must register its turn under the queue's "
        "own cwd"
    )
    assert "setLiveTurn(currentCwd" not in body, (
        "fireQueuedMessage must not register its turn against the "
        "focused session — the flush may be for a background one"
    )


def test_stop_cancels_the_focused_sessions_turn() -> None:
    """Regression: switching between two running sessions must not
    corrupt cancellation state.

    Sequence that broke it: start A, switch to B and start B, switch
    back to A, press Stop. The bridge correctly cancelled focused A,
    but the JS blacklisted the stale global — B's turn id. B kept
    running while every subsequent B event, including its terminal one,
    was dropped at the top of ``nora_event`` before the busy-state
    cleanup: B's output vanished from the live UI and its sidebar dot
    stayed stuck busy.

    The Stop handler must therefore resolve the turn to cancel by the
    FOCUSED cwd."""
    m = re.search(
        r"stopBtn\.addEventListener\('click'.*?\n  \}\);\n",
        _app_js(),
        re.DOTALL,
    )
    assert m is not None, "Stop click handler not found"
    body = _strip_js_comments(m.group(0))

    assert "getLiveTurn(currentCwd)" in body, (
        "Stop must look the turn up by the focused cwd, so it can "
        "never blacklist another session's still-running turn"
    )
    assert "clearLiveTurn(currentCwd)" in body, (
        "Stop must retire only the focused session's live turn"
    )


# ---------------------------------------------------------------------------
# 3. Folder-backed session registry
# ---------------------------------------------------------------------------


def test_external_sessions_register_and_list(tmp_path: Path) -> None:
    """``register`` records a folder; ``list_entries`` returns it
    until the folder is deleted. Idempotent re-registration bumps
    the timestamp without duplicating the entry."""
    from nora.external_sessions import (
        is_registered,
        list_entries,
        register,
    )

    sessions_root = tmp_path / ".nora-sessions"
    project = tmp_path / "project"
    project.mkdir()

    assert list_entries(sessions_root) == []
    register(sessions_root, project)
    entries = list_entries(sessions_root)
    assert len(entries) == 1
    assert entries[0]["path"] == str(project.resolve())
    assert is_registered(sessions_root, project)

    # Idempotent: same path doesn't duplicate, just bumps timestamp.
    first_ts = entries[0]["registered_at"]
    register(sessions_root, project)
    entries2 = list_entries(sessions_root)
    assert len(entries2) == 1
    assert entries2[0]["registered_at"] >= first_ts


def test_external_sessions_filters_missing_paths(tmp_path: Path) -> None:
    """A registered folder that no longer exists must drop out of
    ``list_entries`` so the sidebar doesn't show dead chips."""
    from nora.external_sessions import list_entries, register

    sessions_root = tmp_path / ".nora-sessions"
    project = tmp_path / "project"
    project.mkdir()
    register(sessions_root, project)
    assert len(list_entries(sessions_root)) == 1

    # Simulate the researcher deleting the project directory.
    project.rmdir()
    assert list_entries(sessions_root) == []


def test_list_sessions_surfaces_folder_backed(tmp_path: Path, monkeypatch) -> None:
    """``NoraBridge.list_sessions`` returns folder-backed sessions
    alongside staged sessions, with ``kind="folder"`` so the UI can
    distinguish them."""
    from nora import ui as ui_mod
    from nora.ui import NoraBridge
    from nora.external_sessions import register

    sessions_root = tmp_path / ".nora-sessions"
    sessions_root.mkdir()
    monkeypatch.setattr(ui_mod, "SESSIONS_ROOT", sessions_root)

    project = tmp_path / "my-project"
    project.mkdir()
    register(sessions_root, project)

    bridge = NoraBridge(cwd=None)
    res = bridge.list_sessions()
    assert res["ok"]
    folder_entries = [s for s in res["sessions"] if s.get("kind") == "folder"]
    assert len(folder_entries) == 1
    assert folder_entries[0]["path"] == str(project.resolve())


def test_list_sessions_skips_dir_size_for_folder_backed(
    tmp_path: Path, monkeypatch,
) -> None:
    """``_dir_size`` recursively stats every file in a session dir to
    fill the ``size`` field that drives the delete-confirm prompt.
    Folder-backed sessions don't get a delete affordance (the
    backend rejects rmtree on anything outside SESSIONS_ROOT and the
    sidebar hides the button), so on a real project dir the walk is
    pure cost — ``node_modules`` alone can be tens of thousands of
    files. ``list_sessions`` must short-circuit ``size`` to 0 for
    folder-backed entries; staged sessions still get the real walk
    so their delete prompt remains informative.
    """
    from nora import ui as ui_mod
    from nora.ui import NoraBridge
    from nora.external_sessions import register

    sessions_root = tmp_path / ".nora-sessions"
    sessions_root.mkdir()
    monkeypatch.setattr(ui_mod, "SESSIONS_ROOT", sessions_root)

    # Staged session with one file — _dir_size should return >0.
    staged = sessions_root / "20260511T120000Z_aaaaaaaa"
    staged.mkdir()
    (staged / "data.csv").write_text("a,b\n1,2\n", encoding="utf-8")

    # Folder-backed project with a "node_modules-like" tree — we
    # want to prove _dir_size never runs over it. Drop a sentinel:
    # monkeypatch _dir_size to raise if it's called with the
    # folder-backed path, so any regression that re-enables the
    # walk fails the test loudly.
    project = tmp_path / "my-project"
    (project / "node_modules" / "pkg").mkdir(parents=True)
    (project / "node_modules" / "pkg" / "index.js").write_text(
        "module.exports = {}", encoding="utf-8",
    )
    register(sessions_root, project)

    real_dir_size = ui_mod._dir_size

    def guarded_dir_size(p: Path) -> int:
        if Path(p).resolve() == project.resolve():
            raise AssertionError(
                "list_sessions must not walk folder-backed project dirs"
            )
        return real_dir_size(p)

    monkeypatch.setattr(ui_mod, "_dir_size", guarded_dir_size)

    bridge = NoraBridge(cwd=None)
    res = bridge.list_sessions()
    assert res["ok"]

    folder_entries = [s for s in res["sessions"] if s.get("kind") == "folder"]
    staged_entries = [s for s in res["sessions"] if s.get("kind") == "staged"]
    assert len(folder_entries) == 1
    assert len(staged_entries) == 1

    assert folder_entries[0]["size"] == 0, (
        "folder-backed sessions must report size=0 — the field "
        "drives the delete prompt and folder sessions have no "
        "delete affordance"
    )
    assert staged_entries[0]["size"] > 0, (
        "staged sessions still need the real size for the "
        "delete-confirm dialog"
    )


def test_switch_session_accepts_folder_backed(tmp_path: Path, monkeypatch) -> None:
    """``switch_session`` must accept a folder-backed path even
    though the parent isn't ``SESSIONS_ROOT``. Without the registry
    check, every folder-backed entry in the sidebar would be a dead
    click."""
    from nora import ui as ui_mod
    from nora.ui import NoraBridge
    from nora.external_sessions import register

    sessions_root = tmp_path / ".nora-sessions"
    sessions_root.mkdir()
    monkeypatch.setattr(ui_mod, "SESSIONS_ROOT", sessions_root)

    project = tmp_path / "my-project"
    project.mkdir()
    register(sessions_root, project)

    bridge = NoraBridge(cwd=None)
    res = bridge.switch_session(str(project))
    assert res.get("ok"), (
        f"switch_session must accept registered folder-backed paths, "
        f"got: {res!r}"
    )


def test_switch_session_rejects_unregistered_folder(tmp_path: Path, monkeypatch) -> None:
    """A folder that's NOT in the registry must still be rejected.
    The registry is the durable record of "I previously opened this
    as a session" — accepting any random path would silently let
    cwd land in directories the researcher never intended."""
    from nora import ui as ui_mod
    from nora.ui import NoraBridge

    sessions_root = tmp_path / ".nora-sessions"
    sessions_root.mkdir()
    monkeypatch.setattr(ui_mod, "SESSIONS_ROOT", sessions_root)

    bystander = tmp_path / "bystander"
    bystander.mkdir()

    bridge = NoraBridge(cwd=None)
    res = bridge.switch_session(str(bystander))
    assert not res.get("ok")


# ---------------------------------------------------------------------------
# 4. WebP / GIF symmetry across upload, panel, and recall
# ---------------------------------------------------------------------------


def test_webp_gif_classified_as_graph() -> None:
    """``classify_ext`` must return ``"graph"`` for .webp and .gif so
    the Files panel surfaces them. Without this, a dropped WebP
    screenshot vanished from the panel even though it was on disk."""
    from nora.session_files import classify_ext

    assert classify_ext(".webp") == "graph"
    assert classify_ext(".gif") == "graph"
    # Sanity: the existing image extensions still classify the same.
    assert classify_ext(".png") == "graph"


def test_read_attached_file_recalls_webp(tmp_path: Path, monkeypatch) -> None:
    """``read_attached_file`` must recall WebP / GIF files dropped
    via the composer. Both formats are valid vision MIME types in
    Anthropic and OpenAI; the previous rejection broke the UI's
    "you can mention this later" promise."""
    from nora import tools as tools_mod
    from nora.tools import HANDLERS

    # Minimal valid-ish WebP: vision providers don't strictly parse
    # at the tool layer, and the tool only reads bytes off disk and
    # base64-encodes them. The content matters for actual model
    # consumption, not for the recall path under test.
    webp = tmp_path / "screenshot.webp"
    webp.write_bytes(b"RIFF\x00\x00\x00\x00WEBPVP8 ")

    # The recall path classifies via _is_disclosure_safe_image. Patch
    # it to allow the test file through — full SDC integration is
    # exercised in dedicated tests; here we're checking the extension
    # / mime mapping was updated.
    monkeypatch.setattr(
        tools_mod, "_is_disclosure_safe_image", lambda *_a, **_k: True,
    )
    from nora import config as _config_mod
    monkeypatch.setattr(_config_mod, "_cwd_default", tmp_path)
    # Provenance manifest gate — read_attached_file refuses cwd
    # files not snapshotted as researcher-staged. Tests for
    # researcher-staged paths initialize the manifest with the
    # current cwd's top-level files so the gate doesn't reject
    # a legitimately-uploaded asset.
    from nora.file_provenance import initialize as _init_staged
    _init_staged(tmp_path)

    res = asyncio.run(HANDLERS["read_attached_file"]({
        "name": "screenshot.webp",
    }))
    # On success the tool returns a content list with an image block.
    assert "content" in res
    image_blocks = [b for b in res["content"] if b.get("type") == "image"]
    assert image_blocks, (
        f"WebP must be recalled as an image content block; got {res!r}"
    )
    assert image_blocks[0]["mimeType"] == "image/webp"


def test_read_attached_file_recalls_gif(tmp_path: Path, monkeypatch) -> None:
    """Mirror of the WebP test for GIF — both extensions were added
    together to the recall path and both need coverage."""
    from nora import tools as tools_mod
    from nora.tools import HANDLERS

    gif = tmp_path / "anim.gif"
    gif.write_bytes(b"GIF89a\x00\x00\x00\x00")

    monkeypatch.setattr(
        tools_mod, "_is_disclosure_safe_image", lambda *_a, **_k: True,
    )
    from nora import config as _config_mod
    monkeypatch.setattr(_config_mod, "_cwd_default", tmp_path)
    # Provenance manifest gate — read_attached_file refuses cwd
    # files not snapshotted as researcher-staged. Tests for
    # researcher-staged paths initialize the manifest with the
    # current cwd's top-level files so the gate doesn't reject
    # a legitimately-uploaded asset.
    from nora.file_provenance import initialize as _init_staged
    _init_staged(tmp_path)

    res = asyncio.run(HANDLERS["read_attached_file"]({"name": "anim.gif"}))
    image_blocks = [b for b in res["content"] if b.get("type") == "image"]
    assert image_blocks
    assert image_blocks[0]["mimeType"] == "image/gif"


# ---------------------------------------------------------------------------
# Session pin to top
# ---------------------------------------------------------------------------

def test_list_sessions_surfaces_pinned_field(tmp_path: Path, monkeypatch) -> None:
    """``list_sessions`` must expose ``pinned`` and ``pinned_at`` per
    entry so the sidebar can render the pin icon and sort correctly."""
    from nora import ui as ui_mod
    from nora.ui import NoraBridge
    from nora.session_state import set_pinned

    sessions_root = tmp_path / ".nora-sessions"
    sessions_root.mkdir()
    monkeypatch.setattr(ui_mod, "SESSIONS_ROOT", sessions_root)

    s_unpinned = sessions_root / "20260511T120000Z_aaaaaaaa"
    s_unpinned.mkdir()
    s_pinned = sessions_root / "20260510T120000Z_bbbbbbbb"
    s_pinned.mkdir()
    set_pinned(s_pinned, True)

    bridge = NoraBridge(cwd=None)
    res = bridge.list_sessions()
    assert res["ok"]
    by_path = {s["path"]: s for s in res["sessions"]}
    assert by_path[str(s_pinned.resolve())]["pinned"] is True
    assert by_path[str(s_unpinned.resolve())]["pinned"] is False
    assert by_path[str(s_pinned.resolve())]["pinned_at"], (
        "pinned entries must carry the timestamp the UI uses to sort "
        "within the pinned group"
    )


def test_list_sessions_pins_sort_first(tmp_path: Path, monkeypatch) -> None:
    """The pinned session must come ahead of an unpinned session that
    was worked more recently. Without the two-tier sort the pin would
    have no visible effect — the whole point is to override
    last_activity ordering for items the researcher wants reachable."""
    import time as _time
    from nora import ui as ui_mod
    from nora.ui import NoraBridge
    from nora.session_state import set_pinned

    sessions_root = tmp_path / ".nora-sessions"
    sessions_root.mkdir()
    monkeypatch.setattr(ui_mod, "SESSIONS_ROOT", sessions_root)

    # The pinned session is OLDER on disk than the unpinned one.
    old_pinned = sessions_root / "20260101T120000Z_aaaaaaaa"
    old_pinned.mkdir()
    set_pinned(old_pinned, True)
    # Force its chat_history mtime older than the unpinned's. Without
    # this, both sessions share tmp_path's mtime ≈ now and the sort
    # would be ambiguous on last_activity.
    nora_dir = old_pinned / ".nora"
    nora_dir.mkdir(exist_ok=True)
    hist = nora_dir / "chat_history.jsonl"
    hist.write_text("", encoding="utf-8")
    old_mtime = _time.time() - 86400  # 1 day ago
    import os as _os
    _os.utime(hist, (old_mtime, old_mtime))

    new_unpinned = sessions_root / "20260512T120000Z_bbbbbbbb"
    new_unpinned.mkdir()

    bridge = NoraBridge(cwd=None)
    res = bridge.list_sessions()
    assert res["ok"]
    paths_in_order = [s["path"] for s in res["sessions"]]
    assert paths_in_order[0] == str(old_pinned.resolve()), (
        f"pinned session must lead the list; got {paths_in_order!r}"
    )


def test_list_sessions_pinned_group_sorted_by_pinned_at(
    tmp_path: Path, monkeypatch,
) -> None:
    """Within the pinned group, the most recently pinned session
    sits at the very top. Without this the order inside the pinned
    cluster is undefined and a fresh pin can land below older
    pins."""
    import time as _time
    from nora import ui as ui_mod
    from nora.ui import NoraBridge
    from nora.session_state import set_pinned

    sessions_root = tmp_path / ".nora-sessions"
    sessions_root.mkdir()
    monkeypatch.setattr(ui_mod, "SESSIONS_ROOT", sessions_root)

    first = sessions_root / "20260101T120000Z_aaaaaaaa"
    first.mkdir()
    set_pinned(first, True)

    _time.sleep(1.01)  # ISO timestamps are second-resolution
    second = sessions_root / "20260102T120000Z_bbbbbbbb"
    second.mkdir()
    set_pinned(second, True)

    bridge = NoraBridge(cwd=None)
    res = bridge.list_sessions()
    pinned = [s for s in res["sessions"] if s["pinned"]]
    assert len(pinned) == 2
    assert pinned[0]["path"] == str(second.resolve()), (
        "the most recently pinned session must sit ahead of older pins"
    )
    assert pinned[1]["path"] == str(first.resolve())


def test_set_session_pinned_round_trip(tmp_path: Path, monkeypatch) -> None:
    """The bridge method accepts a staged session path, persists the
    flag, and surfaces the new value through the next list_sessions
    call."""
    from nora import ui as ui_mod
    from nora.ui import NoraBridge

    sessions_root = tmp_path / ".nora-sessions"
    sessions_root.mkdir()
    monkeypatch.setattr(ui_mod, "SESSIONS_ROOT", sessions_root)

    session = sessions_root / "20260511T120000Z_aaaaaaaa"
    session.mkdir()
    bridge = NoraBridge(cwd=None)

    pin_res = bridge.set_session_pinned(str(session), True)
    assert pin_res["ok"] is True
    assert pin_res["pinned"] is True

    listing = bridge.list_sessions()
    entry = next(
        s for s in listing["sessions"] if s["path"] == str(session.resolve())
    )
    assert entry["pinned"] is True

    unpin_res = bridge.set_session_pinned(str(session), False)
    assert unpin_res["ok"] is True
    assert unpin_res["pinned"] is False

    listing2 = bridge.list_sessions()
    entry2 = next(
        s for s in listing2["sessions"] if s["path"] == str(session.resolve())
    )
    assert entry2["pinned"] is False


def test_set_session_pinned_accepts_folder_backed(
    tmp_path: Path, monkeypatch,
) -> None:
    """Pinning a folder-backed session must work too — the rename
    bridge method gates by SESSIONS_ROOT (a project dir can't be
    renamed via the sidebar pencil), but pinning is a non-destructive
    UI preference and there's no reason to deny it for a registered
    project directory."""
    from nora import ui as ui_mod
    from nora.ui import NoraBridge
    from nora.external_sessions import register

    sessions_root = tmp_path / ".nora-sessions"
    sessions_root.mkdir()
    monkeypatch.setattr(ui_mod, "SESSIONS_ROOT", sessions_root)

    project = tmp_path / "my-project"
    project.mkdir()
    register(sessions_root, project)

    bridge = NoraBridge(cwd=None)
    res = bridge.set_session_pinned(str(project), True)
    assert res["ok"] is True
    assert res["pinned"] is True


def test_set_session_pinned_refuses_arbitrary_path(
    tmp_path: Path, monkeypatch,
) -> None:
    """A path that isn't a staged session AND isn't a registered
    folder must be refused — without that gate any JS caller could
    seed a phantom ``.nora/`` skeleton under an arbitrary directory."""
    from nora import ui as ui_mod
    from nora.ui import NoraBridge

    sessions_root = tmp_path / ".nora-sessions"
    sessions_root.mkdir()
    monkeypatch.setattr(ui_mod, "SESSIONS_ROOT", sessions_root)

    stranger = tmp_path / "somewhere-else"
    stranger.mkdir()

    bridge = NoraBridge(cwd=None)
    res = bridge.set_session_pinned(str(stranger), True)
    assert res["ok"] is False
    assert "session directory" in res["reason"]


# ---------------------------------------------------------------------------
# clear_pending_for_session — backend desync on session switch
# ---------------------------------------------------------------------------

def test_clear_pending_for_session_drops_all_pending_lists(
    tmp_path: Path,
) -> None:
    """Stage @-mention, script, plot, and mentioned-image entries on a
    runner, call ``clear_pending_for_session``, and confirm every
    USER-STAGED pending list is empty afterwards. The frontend wipes
    its staged composer state when the researcher leaves a session;
    this bridge method is the matching backend wipe. Without it,
    attachments staged in A but never sent ride invisibly with the
    next plain message in A — UI shows no chip, runner inlines the
    file anyway.

    Plot images captured by the previous turn's ``submit_script`` are
    intentionally NOT cleared here — they are model output, not
    researcher-staged, and a session-focus toggle must not erase the
    image a returning "interpret the plot" message expects to find
    attached. ``clear_pending_for_session``'s docstring spells out
    the carve-out.
    """
    from nora.ui import NoraBridge

    bridge = NoraBridge(cwd=tmp_path)
    runner = bridge._active_runner()
    assert runner is not None

    runner.pending_script_attachments.append({
        "name": "model.py", "kind": "script", "content": "print(1)",
    })
    runner.pending_mentioned_files.append("residuals.png")
    runner.pending_mentioned_images.append({
        "data": "AA==", "mime": "image/png",
        "name": "residuals.png", "path": "/tmp/residuals.png",
    })
    plot_image = {
        "data": "AA==", "mime": "image/png",
        "name": "coefficients.png", "kind": "image",
    }
    runner.pending_plot_images.append(plot_image)

    res = bridge.clear_pending_for_session(str(tmp_path))
    assert res["ok"] is True
    assert res["cleared"] is True
    assert runner.pending_script_attachments == []
    assert runner.pending_mentioned_files == []
    assert runner.pending_mentioned_images == []
    # Model-captured plot images survive — see docstring above and the
    # `Plot images stay` comment in ``clear_pending_for_session``.
    assert runner.pending_plot_images == [plot_image]


def test_clear_pending_for_session_idempotent_on_unknown_runner(
    tmp_path: Path,
) -> None:
    """A session the researcher clicked into but never typed in has no
    live runner. ``clear_pending_for_session`` must succeed without
    materialising one (and without errors) — anything else makes the
    JS-side switch handler treat a routine no-op as a failure.
    """
    from nora.ui import NoraBridge

    bridge = NoraBridge(cwd=None)
    res = bridge.clear_pending_for_session(str(tmp_path / "never-opened"))
    assert res["ok"] is True
    assert res["cleared"] is False


def test_clear_pending_for_session_rejects_empty_path(
    tmp_path: Path,
) -> None:
    """An empty / null cwd should not silently no-op-OK — that would
    let a JS caller forget to pass the leaving session's cwd and have
    the request go through anyway. Surface the bad call so it's
    fixable at the call site."""
    from nora.ui import NoraBridge

    bridge = NoraBridge(cwd=tmp_path)
    res = bridge.clear_pending_for_session("")
    assert res["ok"] is False
    assert "session_cwd" in res["reason"]


def test_clear_pending_for_session_does_not_clear_queued_frozen_snapshots(
    tmp_path: Path,
) -> None:
    """Queued-message frozen snapshots are owned by messages the
    researcher has already committed to send (they're sitting in the
    JS queue). Those must NOT be dropped just because focus left the
    session — the queued message still needs its attachments when
    it eventually fires.

    ``runner.clear_pending_attachments`` does drop frozen snapshots
    on the REWIND path (which is the right call there). The session-
    switch path must NOT call that broader clear. Pin the boundary
    here so a future refactor doesn't widen the wipe.
    """
    from nora.ui import NoraBridge

    bridge = NoraBridge(cwd=tmp_path)
    runner = bridge._active_runner()
    assert runner is not None

    # Stage something AND freeze a queued message snapshot.
    runner.pending_script_attachments.append(
        {"name": "queued.py", "kind": "script", "content": "x"}
    )
    runner.freeze_pending_for_queue("token-queued")
    assert "token-queued" in runner.frozen_pending_attachments

    # Now stage fresh items for the NEXT plain message.
    runner.pending_mentioned_files.append("fresh.png")

    res = bridge.clear_pending_for_session(str(tmp_path))
    assert res["ok"] is True

    # The fresh pending list was cleared (it's what the leaving
    # composer was showing).
    assert runner.pending_mentioned_files == []
    # But the frozen queued snapshot survives — it belongs to a
    # message the researcher already committed to send.
    assert "token-queued" in runner.frozen_pending_attachments


def test_switch_session_clears_leaving_session_only_after_success() -> None:
    """JS-side structural check: ``switchSession`` must call
    ``clear_pending_for_session`` against the LEAVING session — but
    only AFTER the switch is known to have succeeded, and addressed
    by the ``leavingCwd`` captured before the await (so it still
    targets the OLD session, not the new focus).

    Both halves matter. Without the clear, the backend pending lists
    for the leaving session survive hidden and the next plain message
    on return inlines the (no-longer-visible) staged attachment. But
    clearing BEFORE the switch — the original ordering — erased the
    attachments behind the still-visible chips whenever the switch
    failed: the researcher stayed on the old session with its chips
    painted while the runner's lists were already empty, so the next
    message silently sent without them.
    """
    app_js = Path(__file__).resolve().parent.parent / "src" / "nora" / "web" / "app.js"
    src = app_js.read_text(encoding="utf-8")

    # Pull out the switchSession function body so the assertion fires
    # against THAT function, not some unrelated `clear_pending_for_session`
    # call elsewhere.
    m = re.search(
        r"async function switchSession\([^)]*\)\s*\{(.*?)\n\}\n",
        src,
        re.DOTALL,
    )
    assert m is not None, "switchSession function not found in app.js"
    body = m.group(1)

    assert "clear_pending_for_session" in body, (
        "switchSession must call api.clear_pending_for_session for the "
        "leaving session — without it, attachments staged in the prior "
        "session ride invisibly with that session's next plain message"
    )
    leaving_idx = body.find("const leavingCwd = currentCwd;")
    switch_idx = body.find("switch_session(")
    ok_idx = body.find("res.ok")
    clear_idx = body.find("clear_pending_for_session")
    assert 0 <= leaving_idx < switch_idx, (
        "the leaving session must be captured before the switch await, "
        "so the clear targets the OLD cwd even after focus moves"
    )
    assert 0 <= switch_idx < clear_idx, (
        "clear_pending_for_session must run only after switch_session "
        "resolves — clearing first destroys the attachments behind the "
        "still-visible chips when the switch fails"
    )
    assert 0 <= ok_idx < clear_idx, (
        "the clear must be gated on the switch actually succeeding "
        "(res.ok), not merely on the RPC resolving"
    )


def test_replay_history_has_stale_cwd_guard() -> None:
    """JS-side structural check: ``replayHistory`` must compare the
    expected cwd against the live ``currentCwd`` AFTER the
    ``get_chat_history`` await, and bail when they diverge. Without
    the guard, a late-arriving A history wipes ``messagesEl`` and
    paints A's transcript into the now-focused B session.
    """
    app_js = Path(__file__).resolve().parent.parent / "src" / "nora" / "web" / "app.js"
    src = app_js.read_text(encoding="utf-8")

    m = re.search(
        r"async function replayHistory\([^)]*\)\s*\{(.*?)\n\}\n",
        src,
        re.DOTALL,
    )
    assert m is not None, "replayHistory function not found in app.js"
    body = m.group(1)

    # The await on get_chat_history must come before the wipe that
    # follows — and the stale-cwd check must sit between them, not
    # after the wipe (else we've already corrupted the DOM).
    await_idx = body.find("await window.pywebview.api.get_chat_history")
    guard_idx = body.find("expectedCwd !== currentCwd")
    wipe_idx = body.find("messagesEl.innerHTML = ''")
    assert await_idx >= 0, "replayHistory must await get_chat_history"
    assert guard_idx > await_idx, (
        "stale-cwd guard must sit AFTER the await — checking before "
        "the await catches nothing because the user can switch during "
        "the in-flight RPC"
    )
    assert guard_idx < wipe_idx, (
        "stale-cwd guard must run BEFORE the messagesEl wipe — checking "
        "after the wipe means a stale response has already destroyed the "
        "newly-focused session's transcript"
    )
