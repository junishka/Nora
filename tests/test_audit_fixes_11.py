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

    async def fake_install(language, packages, action):
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


# ---------------------------------------------------------------------------
# 2. fireQueuedMessage no longer overwrites focused activeLiveTurn
# ---------------------------------------------------------------------------


def test_fire_queued_message_gates_active_live_turn_on_focused() -> None:
    """JS-side guard: the queued-message dispatcher must only promote
    its local turn handle to the focused-session global when the
    queue's cwd matches the currently-focused session. Without that
    gate, a background flush corrupted Stop / hasVisibleReply / the
    disposable-turn cleanup for the focused turn."""
    app_js = Path(__file__).resolve().parent.parent / "src" / "nora" / "web" / "app.js"
    src = app_js.read_text(encoding="utf-8")

    # Pull out the fireQueuedMessage function body so we don't pick up
    # other call sites (the focused-only send path elsewhere keeps an
    # unconditional ``activeLiveTurn = {...}`` assignment — that's
    # correct there because that path is only taken for the focused
    # session). The function is async; we extract from declaration to
    # the next top-level function.
    m = re.search(
        r"async function fireQueuedMessage\(.*?\n\}\n",
        src,
        re.DOTALL,
    )
    assert m is not None, "fireQueuedMessage function not found"
    body = m.group(0)

    assert "const isFocused = (cwd === currentCwd);" in body, (
        "fireQueuedMessage must capture the focused-session match "
        "before deciding whether to promote the local turn"
    )
    assert "if (isFocused) activeLiveTurn = localTurn;" in body, (
        "fireQueuedMessage must gate the activeLiveTurn assignment "
        "on isFocused — without the gate a background queue flush "
        "steals the focused session's live-turn tracking"
    )
    # The unconditional write must be gone.
    assert "activeLiveTurn = { id: null, nodes:" not in body, (
        "the old unconditional ``activeLiveTurn = {...}`` write must "
        "be removed from fireQueuedMessage"
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
