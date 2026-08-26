"""A bridge operation must stay bound to the session it started on.

``self.cwd`` follows the focused session and can change at any point
inside a bridge method — pywebview runs each JS call on its own
thread. A method that resolves a session at entry, then re-reads
``self.cwd`` later, races the researcher's next click. Several bugs
came from this, all of which moved data between unrelated sessions.

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


def test_dropped_script_is_described_to_the_session_that_got_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bytes and the inline copy must land together.

    The old code wrote the file into the captured session but then
    asked for the FOCUSED runner to stage its contents, so a script
    dropped on A was written to A and described to B — B's next
    prompt carried source the researcher never showed it.
    """
    a = _mk_session(tmp_path, "A")
    b = _mk_session(tmp_path, "B")

    bridge = NoraBridge(cwd=None)
    bridge._set_cwd(a)
    bridge._ensure_runner_for_cwd(b)

    code = b"import pandas as pd\nprint(secret_df.head())\n"
    payload = base64.b64encode(code).decode()
    real_decode = base64.b64decode
    fired = {"done": False}

    def decode_then_switch(*args: Any, **kwargs: Any):
        if not fired["done"]:
            fired["done"] = True
            bridge._set_cwd(b)  # researcher clicks session B mid-decode
        return real_decode(*args, **kwargs)

    monkeypatch.setattr(base64, "b64decode", decode_then_switch)
    res = bridge.add_files_from_blobs(
        [{"name": "A-analysis.py", "content": payload}]
    )

    assert res["ok"] is True
    assert (a / "A-analysis.py").exists()
    staged_a = bridge._runners[str(a.resolve())].pending_script_attachments
    assert [s["name"] for s in staged_a] == ["A-analysis.py"], (
        "the file's own session must be told about it"
    )
    assert bridge._runners[str(b.resolve())].pending_script_attachments == [], (
        "the session merely focused mid-upload must not be handed "
        "another session's source"
    )


def test_upload_honours_the_session_the_drop_named(tmp_path: Path) -> None:
    """The browser reads the file before it calls, so ``self.cwd`` can
    already be B by the time the request arrives — the switch happens
    entirely outside the bridge, where no seam can catch it. The drop
    names the session it started on, and that name wins over focus.
    """
    a = _mk_session(tmp_path, "A")
    b = _mk_session(tmp_path, "B")

    bridge = NoraBridge(cwd=None)
    bridge._set_cwd(a)
    bridge._set_cwd(b)  # switched during the FileReader read

    payload = base64.b64encode(b"col1,col2\n1,2\n").decode()
    res = bridge.add_files_from_blobs(
        [{"name": "A-confidential.csv", "content": payload}], str(a),
    )

    assert res["ok"] is True
    assert (a / "A-confidential.csv").exists()
    assert not (b / "A-confidential.csv").exists()


def test_upload_into_a_closed_session_is_refused(tmp_path: Path) -> None:
    """If the named session was deleted while the browser read the
    file, refuse it. Falling back to the focused session would be the
    original bug wearing a parameter."""
    a = _mk_session(tmp_path, "A")
    gone = _mk_session(tmp_path, "deleted")

    bridge = NoraBridge(cwd=None)
    bridge._set_cwd(a)

    payload = base64.b64encode(b"print(1)\n").decode()
    res = bridge.add_files_from_blobs(
        [{"name": "orphan.py", "content": payload}], str(gone),
    )

    assert res["ok"] is False
    assert "no longer open" in res["reason"]
    assert not (gone / "orphan.py").exists()
    assert not (a / "orphan.py").exists()
    assert bridge._pending_script_attachments == []


def test_upload_without_a_named_session_uses_the_focused_one(
    tmp_path: Path,
) -> None:
    """Back-compat: JS that sends one argument still gets the old
    focused-session behaviour."""
    a = _mk_session(tmp_path, "A")

    bridge = NoraBridge(cwd=None)
    bridge._set_cwd(a)

    payload = base64.b64encode(b"print(1)\n").decode()
    res = bridge.add_files_from_blobs([{"name": "a.py", "content": payload}])

    assert res["ok"] is True
    assert (a / "a.py").exists()
    assert len(bridge._pending_script_attachments) == 1


# ---------------------------------------------------------------------------
# 2b. add_files (native picker)
# ---------------------------------------------------------------------------

def test_native_picker_files_land_in_the_session_that_opened_it(
    tmp_path: Path,
) -> None:
    """Files picked via "+" must land in the session whose button was
    clicked, even if the researcher switches while the dialog is open.

    The old code captured ``self.cwd`` only after the dialog returned,
    so a focus switch while it sat open re-routed the whole selection
    — bytes on disk AND the inline script description — into the
    newly-focused session.
    """
    a = _mk_session(tmp_path, "A")
    b = _mk_session(tmp_path, "B")
    outside = tmp_path / "outside"
    outside.mkdir()
    data = outside / "A-data.csv"
    data.write_text("col1,col2\n1,2\n")
    script = outside / "A-analysis.py"
    script.write_text("import pandas as pd\nprint(secret_df.head())\n")

    bridge = NoraBridge(cwd=None)
    bridge._set_cwd(a)

    class _DialogSwitchesFocus:
        def create_file_dialog(self, *args: Any, **kwargs: Any):
            bridge._set_cwd(b)  # researcher clicks session B mid-dialog
            return [str(data), str(script)]

    bridge._window = _DialogSwitchesFocus()  # type: ignore[assignment]
    res = bridge.add_files()

    assert res["ok"] is True
    # The response names the session the files went to, so the
    # frontend can refuse to stage the returned images / receipts
    # onto the now-focused composer.
    assert Path(res["cwd"]).resolve() == a.resolve()
    assert (a / "A-data.csv").exists() and (a / "A-analysis.py").exists(), (
        "the picked files must land in the session that opened the picker"
    )
    assert not (b / "A-data.csv").exists() and not (b / "A-analysis.py").exists(), (
        "the files must NOT follow the focus switch into another session"
    )
    staged_a = bridge._runners[str(a.resolve())].pending_script_attachments
    assert [s["name"] for s in staged_a] == ["A-analysis.py"]
    assert bridge._runners[str(b.resolve())].pending_script_attachments == [], (
        "the session merely focused mid-dialog must not be handed "
        "another session's source"
    )


# ---------------------------------------------------------------------------
# 2c. attach_session_file (@-mention / Files panel)
# ---------------------------------------------------------------------------

def test_mentioned_file_stages_into_the_session_that_resolved_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An @-mentioned file must stage on the session it was resolved
    against, even if the researcher switches before staging happens.

    The old code resolved and read the file against the session
    focused at entry but asked for the FOCUSED runner at the end, so
    a switch in between appended A's script contents to B's pending
    attachments — B's next prompt carried source from an unrelated
    analysis.
    """
    a = _mk_session(tmp_path, "A")
    b = _mk_session(tmp_path, "B")
    (a / "A-secret.py").write_text("print('confidential')\n")

    bridge = NoraBridge(cwd=None)
    bridge._set_cwd(a)
    bridge._ensure_runner_for_cwd(b)

    # Seam: runs after the entry check, before resolution / staging.
    import nora.session_files as session_files
    real_visible = session_files.visible_run_dir_names
    fired = {"done": False}

    def visible_then_switch(*args: Any, **kwargs: Any):
        if not fired["done"]:
            fired["done"] = True
            bridge._set_cwd(b)  # researcher clicks session B mid-attach
        return real_visible(*args, **kwargs)

    monkeypatch.setattr(
        session_files, "visible_run_dir_names", visible_then_switch,
    )
    res = bridge.attach_session_file("A-secret.py")

    assert res["ok"] is True
    # The response names the session the file was staged on, so the
    # frontend can skip its receipt chip when focus already moved.
    assert Path(res["cwd"]).resolve() == a.resolve()
    staged_a = bridge._runners[str(a.resolve())].pending_script_attachments
    assert [s["name"] for s in staged_a] == ["A-secret.py"], (
        "the mention must stage on the session it was resolved against"
    )
    assert bridge._runners[str(b.resolve())].pending_script_attachments == [], (
        "the session merely focused mid-attach must not be handed "
        "another session's script contents"
    )
    assert bridge._runners[str(b.resolve())].pending_mentioned_files == []


# ---------------------------------------------------------------------------
# 2d. set_dataset_policy
# ---------------------------------------------------------------------------

def test_policy_change_saves_into_the_session_it_loaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A depth change must load, mutate, and save the SAME session's
    policy. The old code re-read ``self.cwd`` for the save, so a
    focus switch between ``load_policy`` and ``save_policy`` wrote
    session A's disclosure ceilings over session B's ``policy.json``
    — silently replacing B's permission table.
    """
    a = _mk_session(tmp_path, "A")
    b = _mk_session(tmp_path, "B")

    bridge = NoraBridge(cwd=None)
    bridge._set_cwd(a)

    import nora.policy as policy_mod
    from nora.policy import load_policy
    real_load = policy_mod.load_policy
    fired = {"done": False}

    def load_then_switch(cwd: Path):
        result = real_load(cwd)
        if not fired["done"]:
            fired["done"] = True
            bridge._set_cwd(b)  # researcher clicks session B mid-change
        return result

    monkeypatch.setattr(ui, "load_policy", load_then_switch)
    res = bridge.set_dataset_policy("A-panel.dta", "names_only")

    assert res["ok"] is True
    assert Path(res["cwd"]).resolve() == a.resolve(), (
        "the response must name the session whose policy changed"
    )
    pol_a = load_policy(a)
    assert "A-panel.dta" in pol_a.datasets, (
        "the change must land in the session it was made in"
    )
    assert not (b / ".nora" / "policy.json").exists(), (
        "the session merely focused mid-change must not have a "
        "policy file written for it"
    )


# ---------------------------------------------------------------------------
# 2e. delete_session_file
# ---------------------------------------------------------------------------

def test_deleting_a_file_unstages_from_its_own_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deleting a file must drop its staged attachment from the
    session the file lived in — not from whatever is focused when
    the (slow) listing gate and unlink finish.

    The old code looked the runner up at the end via
    ``_active_runner``: a switch mid-delete stripped same-named
    attachments from the newly-focused session while the deleted
    file's stale inline content stayed staged on its own — the next
    send there would inline a file the researcher just deleted.
    """
    a = _mk_session(tmp_path, "A")
    b = _mk_session(tmp_path, "B")
    target = a / "analysis.py"
    target.write_text("print('a')\n")
    (b / "analysis.py").write_text("print('b')\n")

    bridge = NoraBridge(cwd=None)
    bridge._set_cwd(a)
    bridge._ensure_runner_for_cwd(b)
    runner_a = bridge._runners[str(a.resolve())]
    runner_b = bridge._runners[str(b.resolve())]
    runner_a.pending_script_attachments = [
        {"name": "analysis.py", "path": str(target.resolve())},
    ]
    runner_b.pending_script_attachments = [
        {"name": "analysis.py", "path": str((b / "analysis.py").resolve())},
    ]

    # Seam: the panel-listing gate runs after the entry check and
    # before the unlink + unstage.
    import nora.session_files as session_files
    real_enum = session_files.enumerate_session_files
    fired = {"done": False}

    def enumerate_then_switch(*args: Any, **kwargs: Any):
        if not fired["done"]:
            fired["done"] = True
            bridge._set_cwd(b)  # researcher clicks session B mid-delete
        return real_enum(*args, **kwargs)

    monkeypatch.setattr(
        session_files, "enumerate_session_files", enumerate_then_switch,
    )
    res = bridge.delete_session_file(str(target))

    assert res["ok"] is True, res
    assert Path(res["cwd"]).resolve() == a.resolve()
    assert not target.exists()
    assert runner_a.pending_script_attachments == [], (
        "the deleted file's own session must lose its staged copy"
    )
    assert [s["name"] for s in runner_b.pending_script_attachments] == [
        "analysis.py"
    ], (
        "the session merely focused mid-delete must keep its "
        "same-named attachment"
    )
    assert (b / "analysis.py").exists()


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


def test_upload_paths_take_their_session_from_the_caller() -> None:
    """Both stagers must accept the session as a parameter, and the
    data one must forward it to the bridge.

    A default evaluated at CALL time means even a caller that names
    nothing gets the session focused when it called — never the one
    the researcher moved to during the read. Capturing it locally
    while the bridge still binds to ``self.cwd`` fixes nothing, so the
    forwarding is half the assertion.
    """
    code = _app_js_without_comments()
    assert "async function stageDataFile(file, targetCwd = currentCwd) {" in code
    assert "async function stageImageFile(file, targetCwd = currentCwd) {" in code
    assert "add_files_from_blobs([" in code and "], targetCwd);" in code, (
        "stageDataFile does not forward its session to the bridge"
    )


def test_upload_receipts_are_gated_on_focus() -> None:
    """Upload receipts paint the focused session, so skip them when
    the file landed somewhere else."""
    code = _app_js_without_comments()
    assert "const stillFocused = stillFocusedOn(targetCwd);" in code, (
        "the upload receipts must be gated on the session the upload "
        "was bound to"
    )


def test_drop_and_paste_capture_the_session_before_any_read() -> None:
    """One capture per batch, not per file.

    Each iteration awaits a full FileReader pass, so a capture taken
    inside the loop is already late for every file behind the first:
    drop a 900 MB .dta alongside a script, switch sessions while the
    .dta reads, and the script follows the researcher.
    """
    code = _app_js_without_comments()
    for handler, var in (
        ("form.addEventListener('drop'", "dropCwd"),
        ("input.addEventListener('paste'", "pasteCwd"),
    ):
        block = code.split(handler, 1)[1].split("\n  });", 1)[0]
        capture = f"const {var} = currentCwd;"
        assert capture in block, (
            f"{handler}: captures no session for the batch, so each file "
            f"binds to whatever is focused when it happens to land"
        )
        # Anchor on the first staging await rather than the first loop:
        # the paste handler runs a synchronous filter loop first, and a
        # capture after THAT is still early enough.
        first_stage = min(
            block.index("await stageImageFile("),
            block.index("await stageDataFile("),
        )
        assert block.index(capture) < first_stage, (
            f"{handler}: session captured after staging begins"
        )
        for stager in ("stageImageFile", "stageDataFile"):
            assert f"{stager}(f, {var})" in block or \
                   f"{stager}(file, {var})" in block, (
                f"{handler}: {stager} is not given the batch's session"
            )


def test_image_staging_checks_focus_before_touching_the_composer() -> None:
    """``stagedImages`` is cleared on switch precisely so an image
    staged in A can't ride along with B's next message. A push that
    lands after its own read arrives just past that clear, so it needs
    its own check — the file itself still reaches A through
    ``stageDataFile``, which reports where it went.
    """
    import re

    code = _app_js_without_comments()
    m = re.search(
        r"async function stageImageFile\(.*?\n\}\n", code, re.DOTALL
    )
    assert m is not None, "stageImageFile not found"
    body = m.group(0)

    assert body.index("stillFocusedOn(targetCwd)") < body.index(
        "stagedImages.push("
    ), (
        "stageImageFile pushes into the composer without checking that "
        "the drop's session is still focused"
    )


def test_add_files_button_gates_staging_on_focus() -> None:
    """The "+" handler must not stage the backend's returned images
    into the composer when focus moved during the dialog / copy.

    ``stagedImages`` is global composer state; an unconditional push
    after the ``add_files`` await lands just past the focus handler's
    clear and rides another session's image on the next message here.
    The backend pins the copy and names its session in ``res.cwd`` —
    the handler must compare that against the focused one.
    """
    code = _app_js_without_comments()
    block = code.split("addFilesBtn.addEventListener('click'", 1)[1]
    block = block.split("\n  });", 1)[0]

    assert "const requestCwd = currentCwd;" in block, (
        "the add-files handler must capture the focused session "
        "before its await"
    )
    assert "stillFocused" in block, (
        "the add-files handler must gate on focus being unchanged"
    )
    assert block.index("if (!stillFocused)") < block.index(
        "stagedImages.push("
    ), (
        "the add-files handler pushes into the composer before "
        "checking that the picker's session is still focused"
    )


def test_mention_receipt_is_gated_on_focus() -> None:
    """``stageMentionedFile``'s receipt chip paints the focused
    composer. The backend stages onto the session it names in
    ``res.cwd``; when that isn't the focused one any more, painting
    the chip advertises another session's attachment here."""
    import re

    code = _app_js_without_comments()
    m = re.search(
        r"async function stageMentionedFile\(.*?\n\}\n", code, re.DOTALL
    )
    assert m is not None, "stageMentionedFile not found"
    body = m.group(0)

    assert "const requestCwd = currentCwd;" in body, (
        "stageMentionedFile must capture the focused session before "
        "its await"
    )
    assert body.index("stillFocused") < body.index(
        "addStagedDataNotices("
    ), (
        "stageMentionedFile paints its receipt chip without checking "
        "that the mention's session is still focused"
    )


def test_switch_session_drops_superseded_responses() -> None:
    """Rapid A→B→C clicking puts two ``switch_session`` responses in
    flight at once, and they can settle in either order. Applying
    every response unconditionally lets a late loser repaint the UI
    to a session the researcher already left — visibly showing one
    session while the backend focuses another, so the next
    focus-routed call lands in the wrong one. Each switchSession call
    must take a ticket and only the NEWEST may apply its response."""
    import re

    code = _app_js_without_comments()
    m = re.search(
        r"async function switchSession\(.*?\n\}\n", code, re.DOTALL
    )
    assert m is not None, "switchSession not found"
    body = m.group(0)

    assert "const seq = ++switchSeq;" in body, (
        "switchSession must take a monotonic ticket before its await"
    )
    assert body.index("const seq = ++switchSeq;") < body.index(
        "await window.pywebview.api.switch_session("
    ), "the ticket must be taken before the switch await, not after"
    assert "seq !== switchSeq" in body, (
        "switchSession must compare its ticket after the await and "
        "drop superseded responses"
    )
    assert body.index("seq !== switchSeq") < body.index("showChat(res)"), (
        "the supersession check must run before the response is "
        "applied to the UI"
    )


def test_direct_sends_name_their_session() -> None:
    """The composer submit and the rewind resend must use the
    explicit-target ``_to_session`` send variants.

    The plain ``send_message`` routes to the bridge's focused cwd at
    the moment the RPC ARRIVES — so a session switch in flight when
    the researcher hits Enter (its response pending, or its backend
    focus change already landed) sends the message into the session
    being switched to while its bubble renders in the one on screen.
    The queue-flush path already routes explicitly for the same
    reason; the direct paths must too."""
    code = _app_js_without_comments()
    assert "send_message_to_session(sendCwd, text)" in code, (
        "the composer's text send must name the session captured at "
        "submit time"
    )
    assert "send_message_with_images_to_session(sendCwd, text, payload)" in code, (
        "the composer's image send must name the session captured at "
        "submit time"
    )
    assert "send_message_to_session(rewindCwd, newText)" in code, (
        "the rewind resend must name the session that was rewound"
    )


@pytest.mark.parametrize("fn", [
    "ensureMentionFiles",
    "refreshFilesChip",
    "loadModels",
    "triggerContextRecount",
])
def test_async_loaders_guard_the_focused_repaint(fn: str) -> None:
    """Every async loader that paints focused-session state (the
    mention cache, the Files chip/popup, the model chip, the context
    chip) must capture the session before its await and drop the
    response when focus moved.

    The mention cache was the sharpest of these: a fetch for A that
    resolved after a switch re-marked the cache fresh with A's rows,
    so the next "@" in B offered another session's file names and
    paths."""
    import re

    code = _app_js_without_comments()
    m = re.search(
        rf"async function {fn}\(.*?\n\}}\n", code, re.DOTALL
    )
    assert m is not None, f"{fn} not found"
    body = m.group(0)

    assert "const requestCwd = currentCwd;" in body, (
        f"{fn} must capture the focused session before its await"
    )
    assert "requestCwd !== currentCwd" in body, (
        f"{fn} must drop its response when focus moved during the fetch"
    )


def test_delete_and_policy_responses_are_gated_on_focus() -> None:
    """``deleteSessionFile``'s chip splice and the permission popup's
    ``set_dataset_policy`` repaint both edit focused-session UI from
    a response that names the session actually acted on — same
    ``res.cwd`` rule as setModel."""
    import re

    code = _app_js_without_comments()
    m = re.search(
        r"async function deleteSessionFile\(.*?\n\}\n", code, re.DOTALL
    )
    assert m is not None, "deleteSessionFile not found"
    body = m.group(0)
    assert "const requestCwd = currentCwd;" in body
    assert "res.cwd === currentCwd" in body
    assert body.index("stillFocused") < body.index("stagedDataNotices"), (
        "the composer-chip splice must be gated on the delete's "
        "session still being focused"
    )

    policy_window = code.split("set_dataset_policy(", 1)[1][:2000]
    assert "result.cwd === currentCwd" in policy_window, (
        "the policy-chip repaint must compare the changed session "
        "against the focused one"
    )
    assert "result.policy && stillFocused" in policy_window, (
        "updatePolicyChip must be skipped when focus moved during "
        "the policy change"
    )
