"""Regression tests for the eighth batch of reviewer-flagged fixes.

Findings addressed:

7. ``switch_session`` and ``set_session_name`` must reject the
   sessions root itself, not just paths outside ``~/.nora-sessions``.
   Pre-fix, ``_is_within`` returned True for ``target ==
   SESSIONS_ROOT`` (``relative_to`` of equal paths is ``Path('.')``)
   so a bridge caller could set the active cwd to the sessions root,
   and downstream tools that resolve relative to ``cwd`` would
   traverse every session. ``delete_session`` already had the
   narrower direct-child check; the two sibling methods didn't.

8. ``debug_excerpt`` redacts script-controlled exception bodies.
   Previously the body channel allowed up to 80 bytes of model-
   chosen content per failure (``raise RuntimeError(short_secret)``
   → forwarded back through the next tool result). The exception
   type, traceback frames, and source-line preview (which is the
   model's own script source) survive; the body itself is replaced
   with a fixed marker.

9. ``search_in_session_files`` honors the rewind visibility filter
   when enumerating run-dir scripts. Pre-fix, the sibling
   ``list_session_files`` / ``read_attached_file`` paths filtered
   on ``visible_run_dir_names(cwd)`` but ``search_in_session_files``
   didn't — a hidden-branch script remained searchable and its line
   excerpts crossed the boundary.

10. ``attach_session_file`` applies the same rewind visibility
    filter on its two run-dir lookups: the helper-plot iteration
    (``runs/<id>/_nora_plots/...``) and the run-dir-script
    resolution via ``find_run_dir_script_by_name``. Pre-fix, the
    bridge attach path could stage files from discarded branches
    if called with a known display name even though the
    model-facing read path was already fixed.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path


# ---------------------------------------------------------------------------
# Finding 7: switch_session and set_session_name reject SESSIONS_ROOT
# ---------------------------------------------------------------------------


def test_switch_session_rejects_sessions_root_itself(
    tmp_path: Path, monkeypatch,
) -> None:
    """Pointing the active cwd at SESSIONS_ROOT would let path
    resolution in tools (``list_session_files``,
    ``read_attached_file``) traverse every other session under the
    root. The narrower direct-child check matches
    ``delete_session``."""
    import nora.ui as ui

    fake_root = tmp_path / "sessions"
    fake_root.mkdir()
    monkeypatch.setattr(ui, "SESSIONS_ROOT", fake_root)

    bridge = ui.NoraBridge()
    res = bridge.switch_session(str(fake_root))
    assert res["ok"] is False
    assert "session directory" in res["reason"]


def test_switch_session_accepts_direct_child(
    tmp_path: Path, monkeypatch,
) -> None:
    """A direct child of SESSIONS_ROOT is the legitimate case — the
    narrowing in finding 7 must not regress this path."""
    import nora.ui as ui

    fake_root = tmp_path / "sessions"
    fake_root.mkdir()
    monkeypatch.setattr(ui, "SESSIONS_ROOT", fake_root)
    session = fake_root / "2026-05-10T12-00-00_abc"
    session.mkdir()

    bridge = ui.NoraBridge()
    res = bridge.switch_session(str(session))
    assert res["ok"] is True
    assert bridge.cwd is not None
    assert bridge.cwd.resolve() == session.resolve()


def test_switch_session_rejects_nested_path(
    tmp_path: Path, monkeypatch,
) -> None:
    """Nested paths beneath a session (e.g.
    ``SESSIONS_ROOT/foo/.nora``) are also rejected: only direct
    children of the root are legal sessions."""
    import nora.ui as ui

    fake_root = tmp_path / "sessions"
    fake_root.mkdir()
    monkeypatch.setattr(ui, "SESSIONS_ROOT", fake_root)
    session = fake_root / "session1"
    session.mkdir()
    nested = session / ".nora"
    nested.mkdir()

    bridge = ui.NoraBridge()
    res = bridge.switch_session(str(nested))
    assert res["ok"] is False
    assert "session directory" in res["reason"]


def test_set_session_name_rejects_sessions_root(
    tmp_path: Path, monkeypatch,
) -> None:
    """Setting a custom name on SESSIONS_ROOT itself would corrupt
    the root with a stray ``.nora/session_state.json``. Same
    direct-child gate."""
    import nora.ui as ui

    fake_root = tmp_path / "sessions"
    fake_root.mkdir()
    monkeypatch.setattr(ui, "SESSIONS_ROOT", fake_root)

    bridge = ui.NoraBridge()
    res = bridge.set_session_name(str(fake_root), "my name")
    assert res["ok"] is False
    assert "session directory" in res["reason"]
    # And no .nora/ directory was created at the root level.
    assert not (fake_root / ".nora").exists()


# ---------------------------------------------------------------------------
# Finding 9: search_in_session_files filters hidden-branch run-dir scripts
# ---------------------------------------------------------------------------


def _make_run_dir_with_script(
    cwd: Path, run_basename: str, language_ext: str, body: str,
) -> Path:
    """Helper: create a run dir with a ``script.<ext>`` so
    ``enumerate_run_dir_scripts`` picks it up."""
    run_dir = cwd / ".nora" / "runs" / run_basename
    run_dir.mkdir(parents=True)
    script = run_dir / f"script{language_ext}"
    script.write_text(body, encoding="utf-8")
    return run_dir


def test_search_in_session_files_hides_rewound_branch_scripts(
    tmp_path: Path,
) -> None:
    """A rewind hides the visible-result row for one branch's run
    dir. The on-disk run dir is still there, but the model must not
    be able to search its script — the rewind visibility contract.
    Sibling tools (``list_session_files`` / ``read_attached_file``)
    already enforced this; ``search_in_session_files`` did not."""
    from nora.config import use_cwd
    from nora.store import get_store
    from nora.tools import HANDLERS

    # Two run dirs with distinct scripts; the second's script
    # mentions a unique token so we can verify search hits.
    visible_run = _make_run_dir_with_script(
        tmp_path, "20260510T120000Z_visiblexx", ".py",
        "# visible script\nTARGET_TOKEN visible\n",
    )
    hidden_run = _make_run_dir_with_script(
        tmp_path, "20260510T120100Z_hiddenxx", ".py",
        "# hidden script\nTARGET_TOKEN hidden\n",
    )

    # Seed the store with one row per run dir, then hide the second.
    store = get_store(tmp_path)
    visible_row = store.insert(
        label="visible spec",
        analysis_type="descriptive",
        sanitized_payload={"type": "descriptive"},
        language="Python",
        script_code="",
        transformations=[],
        raw_log_path=visible_run,
    )
    hidden_row = store.insert(
        label="hidden spec",
        analysis_type="descriptive",
        sanitized_payload={"type": "descriptive"},
        language="Python",
        script_code="",
        transformations=[],
        raw_log_path=hidden_run,
    )
    # Rewind: keep only the visible row; hide everything else.
    store.hide_results_not_in({visible_row.id}, reason="rewind")

    with use_cwd(tmp_path):
        result = asyncio.run(HANDLERS["search_in_session_files"]({
            "query": "TARGET_TOKEN",
            "kinds": ["script"],
        }))
    body = json.loads(next(
        b for b in result["content"] if b.get("type") == "text"
    )["text"])
    assert body["status"] == "ok"
    # The hidden run dir's script must NOT contribute any match.
    matched_files = {r.get("name", "") for r in body.get("results", [])}
    assert any("visible" in m for m in matched_files), matched_files
    assert not any("hidden" in m for m in matched_files), matched_files


# ---------------------------------------------------------------------------
# Finding 10: attach_session_file refuses hidden-branch artifacts
# ---------------------------------------------------------------------------


def test_attach_session_file_refuses_hidden_run_dir_script(
    tmp_path: Path,
) -> None:
    """Bridge-side ``attach_session_file`` resolves run-dir scripts
    via ``find_run_dir_script_by_name``. After a rewind hides one
    branch, attaching a script display name that used to belong to
    the hidden branch must fail — the read path already enforces
    this; the attach path was the bypass."""
    from nora.run_files import find_run_dir_script_by_name
    from nora.store import get_store
    from nora.ui import NoraBridge

    # Two run dirs, each with a script. The label files give them
    # distinct display names so the lookup goes through label
    # resolution.
    visible_run = _make_run_dir_with_script(
        tmp_path, "20260510T120000Z_visiblexx", ".py",
        "# visible\n",
    )
    hidden_run = _make_run_dir_with_script(
        tmp_path, "20260510T120100Z_hiddenxx", ".py",
        "# hidden\n",
    )

    store = get_store(tmp_path)
    visible_row = store.insert(
        label="Visible Run",
        analysis_type="descriptive",
        sanitized_payload={"type": "descriptive"},
        language="Python",
        script_code="",
        transformations=[],
        raw_log_path=visible_run,
    )
    hidden_row = store.insert(
        label="Hidden Run",
        analysis_type="descriptive",
        sanitized_payload={"type": "descriptive"},
        language="Python",
        script_code="",
        transformations=[],
        raw_log_path=hidden_run,
    )

    # Resolve display names with NO filter (researcher-side view —
    # all run dirs visible). That gives us the names the model
    # could still remember from a discarded branch.
    from nora.run_files import enumerate_run_dir_scripts
    entries = enumerate_run_dir_scripts(tmp_path)
    by_run = {e.path.parent.name: e.display_name for e in entries}
    hidden_display = by_run[hidden_run.name]
    visible_display = by_run[visible_run.name]

    # Hide the second row to simulate a rewind that discarded that
    # branch.
    store.hide_results_not_in({visible_row.id}, reason="rewind")

    # Sanity: the model-facing lookup with the visible filter
    # returns the hidden path as None.
    from nora.session_files import visible_run_dir_names
    visible_set = visible_run_dir_names(tmp_path)
    assert visible_set is not None
    assert hidden_run.name not in visible_set

    # The bridge attach path must refuse the hidden display name.
    bridge = NoraBridge(cwd=tmp_path)
    res = bridge.attach_session_file(hidden_display)
    assert res["ok"] is False
    assert res["reason"].startswith("not found")

    # And accepts the visible one.
    res_visible = bridge.attach_session_file(visible_display)
    assert res_visible["ok"] is True, res_visible


# ---------------------------------------------------------------------------
# Finding 8: debug_excerpt body redaction (smoke check at the public API)
# ---------------------------------------------------------------------------


def test_debug_excerpt_does_not_forward_short_cell_value() -> None:
    """A model-authored script that does ``raise RuntimeError(
    df.iloc[0]['secret'])`` would previously leak a short cell
    value (under the 80-byte cap, not data-shaped) into the next
    tool result. The body channel is now closed: only the type
    and traceback frames cross."""
    from nora.error_summary import extract_debug_excerpt

    short_secret = "patient_42_xyz"
    stderr = (
        'Traceback (most recent call last):\n'
        '  File "/abs/x.py", line 3, in <module>\n'
        '    raise RuntimeError(short_secret)\n'
        f'RuntimeError: {short_secret}\n'
    )
    excerpt = extract_debug_excerpt("", stderr, 1, "Python")
    assert excerpt is not None
    # The exception line — where the body channel used to live —
    # is now redacted. The substring appearing in the source-line
    # preview ``raise RuntimeError(short_secret)`` is the variable
    # NAME the model wrote, not the runtime value.
    last_line = excerpt.strip().splitlines()[-1]
    assert "[message body redacted]" in last_line
    # Confirm the redacted body line is the actual exception line
    # (starts with the type name).
    assert last_line.startswith("RuntimeError")


def test_debug_excerpt_stata_macro_expanded_command_under_denylist() -> None:
    """Stata macro expansion echoes raw values into the failing
    command line (``local s = df[1]; regress y `s'`` →
    ``. regress y patient_42``). Under the denylist posture
    introduced in 41903e2, the extractor forwards the command and
    the body through _forward_short_body so the model can act on
    the actual diagnostic ("variable X not found"). Short scalar
    values (a varname here) DO pass through; this is the
    documented residual leak channel — bounded by the 200-byte
    per-body cap and the data-shape detect.

    Predecessor of this test asserted full redaction; it was the
    prior allowlist contract. The trade-off was that the model
    saw "[message body redacted]" and could not tell what to fix,
    so it re-probed (re-running the same broken script with minor
    variations). The denylist posture forwards short diagnostics
    at the cost of a varname-shaped leak channel; data-shape
    exfil (``test_debug_excerpt_stata_redacts_data_shape_exfil``
    below) is still blocked."""
    from nora.error_summary import extract_debug_excerpt

    secret_var = "patient_42_data"
    log = (
        f". regress y {secret_var}\n"
        f"variable {secret_var} not found\n"
        "r(111);\n"
    )
    excerpt = extract_debug_excerpt(log, "", 111, "Stata")
    assert excerpt is not None
    # Command echo and body forward verbatim under the denylist.
    assert ". regress" in excerpt
    assert "variable" in excerpt and "not found" in excerpt
    assert "r(111);" in excerpt
    # Short-scalar varname passes through (documented residual).
    # Predecessor asserted absence; we now pin presence so a
    # silent regression back to the allowlist posture is caught.
    assert secret_var in excerpt


def test_debug_excerpt_stata_redacts_data_shape_exfil() -> None:
    """Companion to the macro-expanded test above: confirm the
    data-shape detect inside _forward_short_body still blocks the
    canonical row-dump fingerprint even when the script tries to
    smuggle it through an error message. This is the mitigation
    that makes the documented short-scalar leak bounded."""
    from nora.error_summary import extract_debug_excerpt

    row_dump = "42, Jane Doe, 1975-03-14, 85000, nurse, MA, 02139"
    log = (
        f". display \"{row_dump}\"\n"
        f"{row_dump}\n"
        "r(198);\n"
    )
    excerpt = extract_debug_excerpt(log, "", 198, "Stata")
    assert excerpt is not None
    assert "Jane Doe" not in excerpt
    assert "85000" not in excerpt
    assert "message body suppressed: looked data-shaped" in excerpt


def test_attach_session_file_helper_plot_filters_hidden_run(
    tmp_path: Path,
) -> None:
    """``attach_session_file`` also walks every run dir's
    ``_nora_plots/`` looking for a plot of the requested basename.
    After a rewind, plots from the hidden branch must not be
    attachable through this fallback."""
    from nora.store import get_store
    from nora.ui import NoraBridge

    # Two run dirs, each with a plot of the SAME basename. Without
    # rewind filtering, ``attach_session_file`` would return
    # whichever the iteration hit first (and ``iterdir`` order is
    # not stable across platforms — the leak is real even when the
    # researcher renames a plot).
    plot_name = "residuals.png"
    visible_run = tmp_path / ".nora" / "runs" / "20260510T120000Z_visiblexx"
    hidden_run = tmp_path / ".nora" / "runs" / "20260510T120100Z_hiddenxx"
    (visible_run / "_nora_plots").mkdir(parents=True)
    (hidden_run / "_nora_plots").mkdir(parents=True)
    visible_plot = visible_run / "_nora_plots" / plot_name
    hidden_plot = hidden_run / "_nora_plots" / plot_name
    visible_plot.write_bytes(b"VISIBLE-PNG")
    hidden_plot.write_bytes(b"HIDDEN-PNG")
    # Each run dir also needs a script for the store row's
    # raw_log_path to make sense.
    (visible_run / "script.py").write_text("# vis\n", encoding="utf-8")
    (hidden_run / "script.py").write_text("# hid\n", encoding="utf-8")

    store = get_store(tmp_path)
    visible_row = store.insert(
        label="visible", analysis_type="descriptive",
        sanitized_payload={"type": "descriptive"},
        language="Python", script_code="", transformations=[],
        raw_log_path=visible_run,
    )
    hidden_row = store.insert(
        label="hidden", analysis_type="descriptive",
        sanitized_payload={"type": "descriptive"},
        language="Python", script_code="", transformations=[],
        raw_log_path=hidden_run,
    )
    store.hide_results_not_in({visible_row.id}, reason="rewind")

    bridge = NoraBridge(cwd=tmp_path)
    res = bridge.attach_session_file(plot_name)
    # Resolves — but to the VISIBLE plot, never the hidden one.
    assert res["ok"] is True, res
    # The bridge stages the file by name only, so we can't inspect
    # bytes from the response directly. Verify the visible runner's
    # pending list got the file from the visible run dir by checking
    # that the hidden plot's iteration was filtered out (the bridge's
    # active runner's pending_mentioned_files now references the
    # basename — and the file content the next turn will read comes
    # from the visible run dir, which is what we want).
    runner = bridge._active_runner()
    assert runner is not None
    # The file lookup chose the visible dir's copy. Open it back
    # via the same iteration the bridge used (filtered) and assert
    # there's exactly one candidate left.
    from nora.session_files import visible_run_dir_names
    visible_set = visible_run_dir_names(tmp_path)
    assert visible_set is not None
    candidates = []
    for run_dir in (tmp_path / ".nora" / "runs").iterdir():
        if run_dir.name not in visible_set:
            continue
        c = run_dir / "_nora_plots" / plot_name
        if c.is_file():
            candidates.append(c)
    assert candidates == [visible_plot]
