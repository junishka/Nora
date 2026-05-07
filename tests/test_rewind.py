"""Tests for the rewind feature.

Three layers:

1. ``ResultStore`` visibility — the ``hidden_at`` column, the default
   filter on ``list_all`` / ``get``, and ``hide_results_not_in``.
2. The bridge-level helpers in ``ui.py`` that locate the cut offset,
   collect kept result_ids from the prefix, and truncate the file —
   tested directly without spinning up a webview.
3. End-to-end: a synthetic ``chat_history.jsonl`` with a few turns,
   then a rewind, then a chat_history readback to verify the right
   bytes survived.

These don't reach for a real provider session — the rewind feature
itself is a pure file + sqlite operation. The "fire a new turn"
step is exercised separately through the existing send tests.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from nora.store import ResultStore
from nora.ui import (
    _find_user_message_offset,
    _result_ids_in_history_prefix,
    _truncate_at_offset,
)


# ---------------------------------------------------------------------------
# 1. Store visibility
# ---------------------------------------------------------------------------

def _fresh_store(tmp_path: Path) -> ResultStore:
    return ResultStore(tmp_path / "results.db")


def _insert_n(store: ResultStore, n: int) -> list[str]:
    ids: list[str] = []
    for i in range(n):
        row = store.insert(
            label=f"row {i}",
            analysis_type="t",
            sanitized_payload={"i": i},
            language="Python",
            script_code="x=1",
            transformations=[],
        )
        ids.append(row.id)
    return ids


def test_default_list_all_shows_only_visible(tmp_path: Path) -> None:
    store = _fresh_store(tmp_path)
    _insert_n(store, 5)
    store.hide_results_not_in({"M2", "M4"}, reason="rewind")
    visible = store.list_all()
    assert {r.id for r in visible} == {"M2", "M4"}
    assert all(r.hidden_at is None for r in visible)


def test_include_hidden_lists_everything(tmp_path: Path) -> None:
    store = _fresh_store(tmp_path)
    _insert_n(store, 5)
    store.hide_results_not_in({"M3"}, reason="rewind")
    rows = store.list_all(include_hidden=True)
    assert len(rows) == 5
    hidden = [r for r in rows if r.hidden_at is not None]
    assert {r.id for r in hidden} == {"M1", "M2", "M4", "M5"}
    assert all(r.hidden_reason == "rewind" for r in hidden)


def test_default_get_returns_none_for_hidden(tmp_path: Path) -> None:
    store = _fresh_store(tmp_path)
    _insert_n(store, 3)
    store.hide_results_not_in({"M2"}, reason="rewind")
    assert store.get("M1") is None
    assert store.get("M3") is None
    assert store.get("M2") is not None


def test_get_include_hidden_returns_row(tmp_path: Path) -> None:
    store = _fresh_store(tmp_path)
    _insert_n(store, 2)
    store.hide_results_not_in(set(), reason="rewind")
    row = store.get("M1", include_hidden=True)
    assert row is not None
    assert row.hidden_at is not None
    assert row.hidden_reason == "rewind"


def test_hide_results_not_in_idempotent(tmp_path: Path) -> None:
    """Re-running the same hide set MUST NOT re-stamp already-hidden
    rows. The rewind that originally hid them is the historical
    truth; a follow-up rewind targeting the same kept set should
    return zero changes."""
    store = _fresh_store(tmp_path)
    _insert_n(store, 4)
    first = store.hide_results_not_in({"M1"}, reason="rewind")
    assert sorted(first) == ["M2", "M3", "M4"]
    first_ts = store.get("M2", include_hidden=True).hidden_at
    second = store.hide_results_not_in({"M1"}, reason="rewind")
    assert second == []
    second_ts = store.get("M2", include_hidden=True).hidden_at
    assert first_ts == second_ts  # not re-stamped


def test_hide_with_empty_kept_set(tmp_path: Path) -> None:
    store = _fresh_store(tmp_path)
    _insert_n(store, 3)
    hidden = store.hide_results_not_in(set(), reason="rewind")
    assert sorted(hidden) == ["M1", "M2", "M3"]
    assert len(store.list_all()) == 0


def test_hide_with_no_hidings_returns_empty(tmp_path: Path) -> None:
    store = _fresh_store(tmp_path)
    _insert_n(store, 3)
    hidden = store.hide_results_not_in({"M1", "M2", "M3"}, reason="rewind")
    assert hidden == []


def test_unhide_results_restores_specific_rows(tmp_path: Path) -> None:
    """The rollback path: ``hide_results_not_in`` returns the ids
    it just hid; ``unhide_results`` restores exactly those, leaving
    any unrelated prior rewinds intact."""
    store = _fresh_store(tmp_path)
    _insert_n(store, 4)
    # First rewind hides M2-M4 (M1 kept).
    first = store.hide_results_not_in({"M1"}, reason="rewind")
    assert sorted(first) == ["M2", "M3", "M4"]

    # Now simulate a SECOND rewind that hides nothing more (M1 still
    # kept), then a rollback. unhide_results(first) should restore
    # M2-M4 only, NOT touch any unrelated rows.
    restored = store.unhide_results(first)
    assert restored == 3
    visible_ids = {r.id for r in store.list_all()}
    assert visible_ids == {"M1", "M2", "M3", "M4"}


def test_unhide_results_no_op_on_already_visible(tmp_path: Path) -> None:
    """``unhide_results`` is safe to call on rows that are already
    visible — the WHERE clause includes ``hidden_at IS NOT NULL``
    so it never re-clears or re-stamps anything."""
    store = _fresh_store(tmp_path)
    _insert_n(store, 3)
    # Nothing hidden. unhide_results should no-op cleanly.
    restored = store.unhide_results(["M1", "M2", "M3"])
    assert restored == 0


def test_unhide_results_empty_input_no_op(tmp_path: Path) -> None:
    store = _fresh_store(tmp_path)
    _insert_n(store, 2)
    assert store.unhide_results([]) == 0


# ---------------------------------------------------------------------------
# 2. Bridge helpers (offset, kept_ids, truncate)
# ---------------------------------------------------------------------------

def _write_history(path: Path, events: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")


def test_find_offset_first_user_message(tmp_path: Path) -> None:
    path = tmp_path / "chat_history.jsonl"
    _write_history(path, [
        {"type": "user_message", "text": "hello"},
        {"type": "assistant_text", "text": "hi"},
        {"type": "user_message", "text": "second"},
    ])
    off = _find_user_message_offset(path, 0)
    assert off == 0


def test_find_offset_second_user_message(tmp_path: Path) -> None:
    path = tmp_path / "chat_history.jsonl"
    _write_history(path, [
        {"type": "user_message", "text": "hello"},
        {"type": "assistant_text", "text": "hi"},
        {"type": "user_message", "text": "second"},
    ])
    off = _find_user_message_offset(path, 1)
    raw = path.read_bytes()
    # Truncating at the returned offset should leave only events
    # strictly before the second user_message.
    head = raw[:off]
    decoded = [json.loads(line) for line in head.splitlines() if line.strip()]
    assert [e["type"] for e in decoded] == ["user_message", "assistant_text"]


def test_find_offset_unicode_user_message(tmp_path: Path) -> None:
    """Multi-byte UTF-8 in user messages must NOT throw off the byte
    offset — this is exactly why the helper opens in binary mode."""
    path = tmp_path / "chat_history.jsonl"
    _write_history(path, [
        {"type": "user_message", "text": "résumé é à ñ 日本語"},
        {"type": "assistant_text", "text": "ok"},
        {"type": "user_message", "text": "回归分析"},
        {"type": "assistant_text", "text": "done"},
    ])
    off = _find_user_message_offset(path, 1)
    raw = path.read_bytes()
    head = raw[:off]
    # Each kept line must parse cleanly — proves the offset landed
    # at a record boundary and didn't cut a UTF-8 sequence in half.
    for line in head.splitlines():
        if not line.strip():
            continue
        json.loads(line)
    decoded = [json.loads(line) for line in head.splitlines() if line.strip()]
    assert [e["type"] for e in decoded] == ["user_message", "assistant_text"]
    assert decoded[0]["text"] == "résumé é à ñ 日本語"


def test_find_offset_out_of_range(tmp_path: Path) -> None:
    path = tmp_path / "chat_history.jsonl"
    _write_history(path, [
        {"type": "user_message", "text": "only one"},
        {"type": "assistant_text", "text": "reply"},
    ])
    assert _find_user_message_offset(path, 5) is None


def test_find_offset_skips_corrupt_lines(tmp_path: Path) -> None:
    path = tmp_path / "chat_history.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"type": "user_message", "text": "first"}) + "\n")
        f.write("{not valid json\n")
        f.write(json.dumps({"type": "user_message", "text": "second"}) + "\n")
    # Corrupt line is not a user_message — the second valid one is
    # therefore at index 1.
    off = _find_user_message_offset(path, 1)
    assert off is not None
    raw = path.read_bytes()
    # The corrupt line stays in the kept prefix (we only locate the
    # user_message offset; we don't repair the file).
    head = raw[:off]
    assert b"first" in head
    assert b"not valid json" in head
    assert b"second" not in head


def test_kept_ids_collects_from_tool_results(tmp_path: Path) -> None:
    path = tmp_path / "chat_history.jsonl"
    _write_history(path, [
        {"type": "user_message", "text": "regress"},
        {"type": "tool_call", "name": "submit_script", "input": {}},
        {
            "type": "tool_result",
            "text": json.dumps({
                "status": "ok",
                "results": [
                    {"result_id": "M1"},
                    {"result_id": "M2"},
                ],
            }),
        },
        {"type": "assistant_text", "text": "see M1, M2"},
        {"type": "user_message", "text": "expand"},
        {
            "type": "tool_result",
            "text": json.dumps({"status": "ok", "result_id": "M2"}),
        },
    ])
    raw = path.read_bytes()
    # Cut at the SECOND user_message — should keep M1 and M2 from
    # the first tool_result, drop the second tool_result entirely.
    off = _find_user_message_offset(path, 1)
    kept = _result_ids_in_history_prefix(path, off)
    assert kept == {"M1", "M2"}


def test_kept_ids_zero_offset_returns_empty(tmp_path: Path) -> None:
    """Cutting at offset 0 means nothing is kept — there's no prefix
    to scan. Edge case the rewind path hits when the researcher
    edits the very first message."""
    path = tmp_path / "chat_history.jsonl"
    _write_history(path, [
        {"type": "user_message", "text": "first"},
        {"type": "tool_result", "text": json.dumps({"result_id": "M1"})},
    ])
    kept = _result_ids_in_history_prefix(path, 0)
    assert kept == set()


def test_truncate_at_offset(tmp_path: Path) -> None:
    path = tmp_path / "chat_history.jsonl"
    _write_history(path, [
        {"type": "user_message", "text": "keep me"},
        {"type": "user_message", "text": "drop me"},
    ])
    off = _find_user_message_offset(path, 1)
    _truncate_at_offset(path, off)
    raw = path.read_bytes()
    decoded = [json.loads(line) for line in raw.splitlines() if line.strip()]
    assert [e["text"] for e in decoded] == ["keep me"]


# ---------------------------------------------------------------------------
# 3. End-to-end: rewind preserves kept rows, hides dropped ones
# ---------------------------------------------------------------------------

def test_rewind_round_trip_preserves_kept_results(tmp_path: Path) -> None:
    """Synthetic full session: 3 turns, each producing a result. A
    rewind cutting at turn 1 (0-indexed) keeps turn 0's result and
    hides turns 1 and 2's. The chat history is byte-truncated; the
    store's visible rows match what the model would see on the next
    warm-start."""
    store = _fresh_store(tmp_path)
    ids = _insert_n(store, 3)  # M1, M2, M3 — one per turn
    history_path = tmp_path / "chat_history.jsonl"
    _write_history(history_path, [
        {"type": "user_message", "text": "turn 0"},
        {"type": "tool_result", "text": json.dumps({"result_id": "M1"})},
        {"type": "assistant_text", "text": "M1 done"},
        {"type": "user_message", "text": "turn 1"},
        {"type": "tool_result", "text": json.dumps({"result_id": "M2"})},
        {"type": "assistant_text", "text": "M2 done"},
        {"type": "user_message", "text": "turn 2"},
        {"type": "tool_result", "text": json.dumps({"result_id": "M3"})},
        {"type": "assistant_text", "text": "M3 done"},
    ])

    # Cut at turn 1.
    off = _find_user_message_offset(history_path, 1)
    kept_ids = _result_ids_in_history_prefix(history_path, off)
    assert kept_ids == {"M1"}
    hidden_ids = store.hide_results_not_in(kept_ids, reason="rewind")
    _truncate_at_offset(history_path, off)

    # Visibility: M1 alone visible.
    visible = store.list_all()
    assert {r.id for r in visible} == {"M1"}
    assert sorted(hidden_ids) == ["M2", "M3"]

    # Audit path still sees the full set.
    audit = store.list_all(include_hidden=True)
    assert {r.id for r in audit} == {"M1", "M2", "M3"}

    # Truncated chat history retains exactly turn 0's events.
    raw = history_path.read_bytes()
    decoded = [json.loads(l) for l in raw.splitlines() if l.strip()]
    types = [e["type"] for e in decoded]
    assert types == ["user_message", "tool_result", "assistant_text"]
    assert decoded[0]["text"] == "turn 0"


# ---------------------------------------------------------------------------
# 4. NoraBridge.rewind_to integration
#
# These cover the full bridge entry point — store imports, runner
# state mutation, file truncation — to catch integration bugs the
# helper-level tests above don't exercise (e.g. a missing import in
# the endpoint body).
# ---------------------------------------------------------------------------

def test_rewind_to_endpoint_happy_path(tmp_path: Path) -> None:
    """End-to-end: bridge constructs, history is written, store has
    rows, ``rewind_to`` returns ok and mutates everything in lockstep.
    A unit-only test pass that doesn't exercise this path can hide
    integration bugs (missing imports, wrong field names) — see the
    'get_store not defined' regression this test was added to lock
    down."""
    from nora.ui import NoraBridge
    from nora.store import get_store

    bridge = NoraBridge(cwd=tmp_path)
    runner = bridge._active_runner()
    assert runner is not None

    # Two stored rows, only M1 referenced in the kept prefix.
    store = get_store(tmp_path)
    store.insert(
        label="kept", analysis_type="t",
        sanitized_payload={"i": 1}, language="Python",
        script_code="x=1", transformations=[],
    )
    store.insert(
        label="dropped", analysis_type="t",
        sanitized_payload={"i": 2}, language="Python",
        script_code="x=2", transformations=[],
    )

    # Synthetic chat_history.jsonl with two user turns.
    history_path = tmp_path / ".nora" / "chat_history.jsonl"
    _write_history(history_path, [
        {"type": "user_message", "text": "first"},
        {"type": "tool_result", "text": json.dumps({"result_id": "M1"})},
        {"type": "assistant_text", "text": "ok"},
        {"type": "user_message", "text": "second"},
        {"type": "tool_result", "text": json.dumps({"result_id": "M2"})},
    ])

    # Stage some pending attachments + plot images so we can verify
    # the cleanup happens.
    runner.pending_script_attachments.append({"name": "stale.py"})
    runner.pending_mentioned_files.append("stale.csv")
    runner.pending_mentioned_images.append({"name": "stale.png"})
    runner.pending_plot_images.append({"name": "old_plot.png"})

    res = bridge.rewind_to(1)

    assert res["ok"] is True, res
    assert res["truncated_from_index"] == 1
    assert res["hidden_count"] == 1

    # Store: M1 visible, M2 hidden. Audit query sees both.
    assert {r.id for r in store.list_all()} == {"M1"}
    assert {r.id for r in store.list_all(include_hidden=True)} == {"M1", "M2"}
    assert store.get("M2") is None
    assert store.get("M2", include_hidden=True) is not None

    # File truncated to exactly turn 0's events.
    raw = history_path.read_bytes()
    decoded = [json.loads(l) for l in raw.splitlines() if l.strip()]
    assert [e["type"] for e in decoded] == [
        "user_message", "tool_result", "assistant_text",
    ]

    # Runner cleanup: all four pending lists empty, warm-start flagged.
    assert runner.pending_script_attachments == []
    assert runner.pending_mentioned_files == []
    assert runner.pending_mentioned_images == []
    assert runner.pending_plot_images == []
    assert runner.needs_context_prefix is True


def test_rewind_to_rolls_back_hide_when_truncate_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If ``_truncate_at_offset`` raises mid-rewind, the rows that
    ``hide_results_not_in`` already marked must be restored. Without
    the rollback, the bridge returns ok=false to JS while the store
    has silently moved on — the model would see hidden rows on the
    next turn even though the UI says the rewind failed."""
    from nora.ui import NoraBridge
    from nora.store import get_store
    import nora.ui as ui_module

    bridge = NoraBridge(cwd=tmp_path)
    runner = bridge._active_runner()
    assert runner is not None

    store = get_store(tmp_path)
    store.insert(
        label="kept", analysis_type="t",
        sanitized_payload={"i": 1}, language="Python",
        script_code="x=1", transformations=[],
    )
    store.insert(
        label="dropped", analysis_type="t",
        sanitized_payload={"i": 2}, language="Python",
        script_code="x=2", transformations=[],
    )
    history_path = tmp_path / ".nora" / "chat_history.jsonl"
    _write_history(history_path, [
        {"type": "user_message", "text": "first"},
        {"type": "tool_result", "text": json.dumps({"result_id": "M1"})},
        {"type": "assistant_text", "text": "ok"},
        {"type": "user_message", "text": "second"},
        {"type": "tool_result", "text": json.dumps({"result_id": "M2"})},
    ])

    # Force truncate to fail. The rewind path should detect this,
    # roll the hide back, and surface ok=false.
    def _boom(*args, **kwargs):
        raise OSError("disk full (simulated)")

    monkeypatch.setattr(ui_module, "_truncate_at_offset", _boom)

    res = bridge.rewind_to(1)

    assert res["ok"] is False
    assert "could not truncate" in res["reason"]

    # Both rows must be visible again — the hide was rolled back.
    visible_ids = {r.id for r in store.list_all()}
    assert visible_ids == {"M1", "M2"}, (
        "rewind rollback failed: store mutated despite ok=false"
    )

    # And the chat history file is untouched.
    raw_lines = history_path.read_bytes().splitlines()
    assert len(raw_lines) == 5


def test_rewind_to_refuses_when_no_active_session(tmp_path: Path) -> None:
    from nora.ui import NoraBridge
    bridge = NoraBridge(cwd=None)
    res = bridge.rewind_to(0)
    assert res["ok"] is False
    assert "no active session" in res["reason"]


def test_rewind_to_refuses_when_runner_is_busy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stop-then-edit semantics: a running turn must block the rewind
    so the researcher can't truncate the history out from under an
    in-flight subprocess."""
    from nora.ui import NoraBridge
    bridge = NoraBridge(cwd=tmp_path)
    runner = bridge._active_runner()
    assert runner is not None
    monkeypatch.setattr(runner, "is_busy", lambda: True)
    res = bridge.rewind_to(0)
    assert res["ok"] is False
    assert "still running" in res["reason"]


def test_rewind_to_refuses_negative_index(tmp_path: Path) -> None:
    from nora.ui import NoraBridge
    bridge = NoraBridge(cwd=tmp_path)
    history_path = tmp_path / ".nora" / "chat_history.jsonl"
    _write_history(history_path, [
        {"type": "user_message", "text": "hello"},
    ])
    res = bridge.rewind_to(-1)
    assert res["ok"] is False
    assert "non-negative" in res["reason"]


def test_rewind_to_refuses_index_out_of_range(tmp_path: Path) -> None:
    from nora.ui import NoraBridge
    bridge = NoraBridge(cwd=tmp_path)
    history_path = tmp_path / ".nora" / "chat_history.jsonl"
    _write_history(history_path, [
        {"type": "user_message", "text": "hello"},
    ])
    res = bridge.rewind_to(5)
    assert res["ok"] is False
    assert "no user message at index 5" in res["reason"]


def test_rewind_to_refuses_when_history_missing(tmp_path: Path) -> None:
    from nora.ui import NoraBridge
    bridge = NoraBridge(cwd=tmp_path)
    res = bridge.rewind_to(0)
    assert res["ok"] is False
    assert "no chat history" in res["reason"]


def test_rewind_to_first_message_clears_everything(tmp_path: Path) -> None:
    """Cutting at index 0 means there's no kept prefix at all — every
    result row gets hidden and the chat_history.jsonl is empty.
    Makes the "edit my very first message" case work cleanly."""
    from nora.ui import NoraBridge
    from nora.store import get_store

    bridge = NoraBridge(cwd=tmp_path)
    store = get_store(tmp_path)
    store.insert(
        label="row", analysis_type="t",
        sanitized_payload={}, language="Python",
        script_code="", transformations=[],
    )
    history_path = tmp_path / ".nora" / "chat_history.jsonl"
    _write_history(history_path, [
        {"type": "user_message", "text": "only one"},
        {"type": "tool_result", "text": json.dumps({"result_id": "M1"})},
    ])
    res = bridge.rewind_to(0)
    assert res["ok"] is True
    assert res["hidden_count"] == 1
    assert history_path.read_bytes() == b""
    assert store.list_all() == []
