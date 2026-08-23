"""A bridge operation must stay bound to the session it started on.

``self.cwd`` follows the focused session and can change at any point
inside a bridge method — pywebview runs each JS call on its own
thread. A method that resolves a session at entry, then re-reads
``self.cwd`` later, races the researcher's next click. Three bugs came
from this, all of which moved data between unrelated sessions.

Each test forces the race by patching a seam that runs after the entry
check and before the session-dependent work. Don't switch these to
sleep-based timing: that version passes against the buggy code.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import pytest

import nora.ui as ui
from nora.session_state import read_session_state
from nora.store import get_store
from nora.ui import NoraBridge


def _mk_session(tmp_path: Path, name: str) -> Path:
    d = tmp_path / name
    (d / ".nora").mkdir(parents=True)
    return d


def _seed_history(cwd: Path, messages: list[str]) -> None:
    hp = cwd / ".nora" / "chat_history.jsonl"
    with hp.open("w", encoding="utf-8") as f:
        for i, text in enumerate(messages):
            f.write(json.dumps({"type": "user_message", "text": text}) + "\n")
            f.write(json.dumps({
                "type": "assistant_text",
                "text": f"reply {i}",
                "result_id": f"res{i}",
            }) + "\n")


def _seed_result(cwd: Path, label: str) -> None:
    get_store(cwd).insert(
        label=label,
        analysis_type="descriptive",
        sanitized_payload={"mean": 1.0},
        language="python",
        script_code="x = 1",
        transformations=[],
    )


def _visible_labels(cwd: Path) -> list[str]:
    return [r.label for r in get_store(cwd).list_all()]


# ---------------------------------------------------------------------------
# 1. rewind_to
# ---------------------------------------------------------------------------

def test_rewind_does_not_hide_another_sessions_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Switching sessions mid-rewind must not touch the new session.

    The old code took the history path from A but re-read ``self.cwd``
    for the store, so it hid every row in B using A's kept-ids. None
    matched, so B lost all its results.
    """
    a = _mk_session(tmp_path, "A")
    b = _mk_session(tmp_path, "B")
    _seed_history(a, ["a1", "a2", "a3"])
    _seed_history(b, ["b1"])
    _seed_result(a, "A-result")
    _seed_result(b, "B-only-result")

    bridge = NoraBridge(cwd=None)
    bridge._set_cwd(a)
    bridge._ensure_runner_for_cwd(b)
    bridge.start_loop()

    # Seam: runs after the entry check, before the store is opened.
    real_find = ui._find_user_message_offset

    def find_then_switch(*args: Any, **kwargs: Any):
        bridge._set_cwd(b)  # researcher clicks session B
        return real_find(*args, **kwargs)

    monkeypatch.setattr(ui, "_find_user_message_offset", find_then_switch)
    try:
        res = bridge.rewind_to(1)
    finally:
        bridge.stop_loop()

    assert res["ok"] is True
    # The rewind names the session it acted on, so the frontend can
    # refuse to send the edited message into a different one.
    assert Path(res["cwd"]).resolve() == a.resolve()
    # A was rewound...
    a_lines = (a / ".nora" / "chat_history.jsonl").read_text().splitlines()
    assert len(a_lines) == 2, "A's history should be truncated at turn 1"
    # ...and B was left completely alone.
    assert _visible_labels(b) == ["B-only-result"], (
        "the focus switch must not hide the other session's results"
    )
    b_lines = (b / ".nora" / "chat_history.jsonl").read_text().splitlines()
    assert len(b_lines) == 2, "B's history must be untouched"


# ---------------------------------------------------------------------------
# 2. add_files_from_blobs
# ---------------------------------------------------------------------------

def test_upload_lands_in_the_session_it_started_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A file dropped on A must land in A, even if the researcher
    switches to B while it decodes.

    The old code re-read ``self.cwd`` for the write target, so a switch
    mid-decode put the file in B. That mixes confidential data between
    unrelated analyses.
    """
    a = _mk_session(tmp_path, "A")
    b = _mk_session(tmp_path, "B")

    bridge = NoraBridge(cwd=None)
    bridge._set_cwd(a)

    payload = base64.b64encode(b"col1,col2\n1,2\n").decode()
    real_decode = base64.b64decode
    fired = {"done": False}

    def decode_then_switch(*args: Any, **kwargs: Any):
        if not fired["done"]:
            fired["done"] = True
            bridge._set_cwd(b)  # researcher clicks session B mid-decode
        return real_decode(*args, **kwargs)

    monkeypatch.setattr(base64, "b64decode", decode_then_switch)
    res = bridge.add_files_from_blobs(
        [{"name": "A-confidential.csv", "content": payload}]
    )

    assert res["ok"] is True
    assert (a / "A-confidential.csv").exists(), (
        "the file must land in the session it was dropped on"
    )
    assert not (b / "A-confidential.csv").exists(), (
        "the file must NOT leak into the session focused mid-upload"
    )
    # The response names where the bytes went, so the frontend can
    # avoid decorating the now-focused session with another's file.
    assert Path(res["cwd"]).resolve() == a.resolve()


# ---------------------------------------------------------------------------
# 3. set_model / set_effort persistence
# ---------------------------------------------------------------------------

def test_effort_persists_against_the_session_it_changed(
    tmp_path: Path, anthropic_authed: None,
) -> None:
    """A focus change during the swap must not redirect the save.

    The old code re-resolved the focused runner after the swap, so it
    saved B's settings and never saved A's change. A silently reverted
    on next open.
    """
    a = _mk_session(tmp_path, "A")
    b = _mk_session(tmp_path, "B")

    bridge = NoraBridge(cwd=None)
    bridge._set_cwd(a)
    bridge.start_loop()
    try:
        real_run_on_loop = bridge._run_on_loop

        def switch_then_run(coro):
            bridge._set_cwd(b)  # researcher clicks B during the swap
            return real_run_on_loop(coro)

        bridge._run_on_loop = switch_then_run  # type: ignore[method-assign]
        res = bridge.set_effort("low")
        bridge._run_on_loop = real_run_on_loop  # type: ignore[method-assign]

        assert res["ok"] is True
        assert Path(res["cwd"]).resolve() == a.resolve(), (
            "the response must name the session actually changed"
        )
        assert bridge._runners[str(a.resolve())].effort == "low"

        state_a = read_session_state(a)
        assert state_a is not None and state_a.active_effort == "low", (
            "the changed session's choice must be persisted"
        )
        state_b = read_session_state(b)
        assert state_b is None or state_b.active_effort != "xhigh", (
            "the session merely focused during the swap must not have "
            "its snapshot rewritten"
        )
    finally:
        bridge.stop_loop()


# ---------------------------------------------------------------------------
# 4. Frontend guards (source assertions — no JS runner in this repo)
# ---------------------------------------------------------------------------

def _app_js_without_comments() -> str:
    import re

    src = (
        Path(__file__).resolve().parent.parent
        / "src" / "nora" / "web" / "app.js"
    ).read_text(encoding="utf-8")
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.DOTALL)
    return re.sub(r"^\s*//.*$", "", src, flags=re.MULTILINE)


def test_edited_message_is_not_sent_into_another_session() -> None:
    """``runEditedMessage`` must check focus before sending.

    The rewind is bound correctly, but replay and send act on the
    focused session — so without this check, an edit made in A is sent
    into B.
    """
    import re

    code = _app_js_without_comments()
    m = re.search(
        r"async function runEditedMessage\(.*?\n\}\n", code, re.DOTALL
    )
    assert m is not None, "runEditedMessage not found"
    body = m.group(0)

    assert "const editCwd = currentCwd;" in body, (
        "the originating session must be captured before the rewind await"
    )
    assert "rewoundCwd !== currentCwd" in body, (
        "runEditedMessage must bail when focus moved during the rewind "
        "rather than sending the edit into the now-focused session"
    )


@pytest.mark.parametrize("fn", ["setModel", "setEffort"])
def test_picker_does_not_paint_another_sessions_setting(fn: str) -> None:
    """The picker shows the focused session, so painting another
    session's response would advertise the wrong model — maybe a
    cheaper one than is actually billing."""
    import re

    code = _app_js_without_comments()
    m = re.search(rf"async function {fn}\(.*?\n\}}\n", code, re.DOTALL)
    assert m is not None, f"{fn} not found"
    body = m.group(0)

    assert "const requestCwd = currentCwd;" in body, (
        f"{fn} must capture the focused session before its await"
    )
    assert "stillFocused" in body, (
        f"{fn} must gate the picker repaint on focus being unchanged"
    )


def test_upload_receipts_are_gated_on_focus() -> None:
    """Upload receipts paint the focused session, so skip them when
    the file landed somewhere else."""
    code = _app_js_without_comments()
    assert "const uploadCwd = currentCwd;" in code, (
        "the upload handler must capture the session it started in"
    )
