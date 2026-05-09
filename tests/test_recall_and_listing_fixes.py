"""Regression tests for four issues surfaced by audit:

1. ``recall_conversation`` AttributeError on ``use.result_id`` (the
   attribute was renamed to ``result_ids`` in commit f70a4e1; the
   recall renderer didn't get updated).
2. ``list_results`` was unbounded and returned chronological-ascending
   (oldest first) — wrong order for "what did we just run", and a
   long session shipped hundreds of rows in one call.
3. ``recall_conversation``'s soft ``max_chars`` budget didn't include
   the ``tools`` array or ``result_ids`` list size, so result-heavy
   turns drifted past the cap.
4. ``session_state.write_session_state`` paired the latest user
   message with the latest assistant reply independently — for an
   in-flight turn (user typed, assistant hasn't replied yet) this
   pulled the user from turn N and the assistant from turn N-1,
   showing a mismatched exchange in the sidebar summary.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from nora.config import use_cwd
from nora.session_state import write_session_state, read_session_state
from nora.store import StoredResult, get_store, reset_store_for_tests
from nora.tools import HANDLERS


def _mcp_text(payload: dict) -> dict:
    return json.loads(payload["content"][0]["text"])


def _write_jsonl(cwd: Path, events: list[dict]) -> None:
    p = cwd / ".nora" / "chat_history.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")


@pytest.fixture(autouse=True)
def _clear():
    reset_store_for_tests()
    yield
    reset_store_for_tests()


# ---------------------------------------------------------------------------
# Fix 1: recall_conversation must not AttributeError on tool_call turns
# ---------------------------------------------------------------------------

def test_recall_conversation_renders_multi_result_tool_call(
    tmp_path: Path,
) -> None:
    """A turn whose tool_result carries a multi-result submit_script
    payload (results: [{result_id: M1}, {result_id: M2}, ...]) must
    render through recall without raising. Previously this hit
    AttributeError because the renderer read ``use.result_id``
    against a ``ToolUse`` whose attribute is now ``result_ids``."""
    _write_jsonl(tmp_path, [
        {"type": "user_message", "text": "run the spec sweep"},
        {"type": "tool_call", "name": "mcp__nora__submit_script",
         "call_id": "abc",
         "input": {"language": "R", "label": "spec sweep"}},
        {"type": "tool_result", "call_id": "abc",
         "text": json.dumps({
             "status": "ok", "script_run_id": "R-deadbeef",
             "results": [
                 {"result_id": "M1", "status": "ok"},
                 {"result_id": "M2", "status": "ok"},
                 {"result_id": "M3", "status": "ok"},
             ],
         }),
         "is_error": False},
        {"type": "assistant_text", "text": "Done."},
    ])
    with use_cwd(tmp_path):
        res = asyncio.run(HANDLERS["recall_conversation"]({"tail": 5}))
    body = _mcp_text(res)
    assert body["status"] == "ok"
    turns = body["turns"]
    assert len(turns) == 1
    tools = turns[0]["tools"]
    assert len(tools) == 1
    # Multi-id case: emits ``result_ids`` (plural).
    assert tools[0]["result_ids"] == ["M1", "M2", "M3"]


def test_recall_conversation_emits_singular_result_id_when_one(
    tmp_path: Path,
) -> None:
    """A single-id tool result (expand_result, single-helper
    submit_script) emits ``result_id`` (singular) for compactness.
    """
    _write_jsonl(tmp_path, [
        {"type": "user_message", "text": "fetch M5"},
        {"type": "tool_call", "name": "mcp__nora__expand_result",
         "call_id": "abc", "input": {"result_id": "M5"}},
        {"type": "tool_result", "call_id": "abc",
         "text": '{"result_id": "M5", "status": "ok"}',
         "is_error": False},
    ])
    with use_cwd(tmp_path):
        res = asyncio.run(HANDLERS["recall_conversation"]({"tail": 5}))
    body = _mcp_text(res)
    assert body["status"] == "ok"
    tools = body["turns"][0]["tools"]
    assert tools[0]["result_id"] == "M5"
    assert "result_ids" not in tools[0]


# ---------------------------------------------------------------------------
# Fix 2: list_results is bounded + newest-first
# ---------------------------------------------------------------------------

def _seed_n_results(cwd: Path, n: int) -> list[StoredResult]:
    store = get_store(cwd)
    rows: list[StoredResult] = []
    for i in range(n):
        rows.append(store.insert(
            label=f"label-{i}",
            analysis_type="descriptive",
            sanitized_payload={
                "type": "descriptive", "variable": f"v{i}",
                "n": 10, "mean": float(i), "sd": 0.1, "missing_count": 0,
            },
            language="Python",
            script_code="x",
            transformations=[],
        ))
    return rows


def test_list_results_returns_newest_first_within_default_limit(
    tmp_path: Path,
) -> None:
    cwd = tmp_path / "session"
    cwd.mkdir()
    _seed_n_results(cwd, 5)
    with use_cwd(cwd):
        res = asyncio.run(HANDLERS["list_results"]({}))
    body = _mcp_text(res)
    assert body["status"] == "ok"
    assert body["total"] == 5
    assert body["count"] == 5
    assert body["truncated"] is False
    # Newest first: M5, M4, M3, M2, M1.
    assert [r["id"] for r in body["results"]] == ["M5", "M4", "M3", "M2", "M1"]


def test_list_results_caps_at_default_limit(tmp_path: Path) -> None:
    cwd = tmp_path / "session"
    cwd.mkdir()
    _seed_n_results(cwd, 75)
    with use_cwd(cwd):
        res = asyncio.run(HANDLERS["list_results"]({}))
    body = _mcp_text(res)
    assert body["total"] == 75
    assert body["count"] == 50
    assert body["limit"] == 50
    assert body["truncated"] is True
    # Newest are kept.
    assert body["results"][0]["id"] == "M75"


def test_list_results_honors_explicit_limit(tmp_path: Path) -> None:
    cwd = tmp_path / "session"
    cwd.mkdir()
    _seed_n_results(cwd, 20)
    with use_cwd(cwd):
        res = asyncio.run(HANDLERS["list_results"]({"limit": 5}))
    body = _mcp_text(res)
    assert body["count"] == 5
    assert body["limit"] == 5
    assert body["truncated"] is True
    assert [r["id"] for r in body["results"]] == ["M20", "M19", "M18", "M17", "M16"]


def test_list_results_clamps_above_hard_cap(tmp_path: Path) -> None:
    cwd = tmp_path / "session"
    cwd.mkdir()
    _seed_n_results(cwd, 10)
    with use_cwd(cwd):
        res = asyncio.run(HANDLERS["list_results"]({"limit": 99999}))
    body = _mcp_text(res)
    # Hard cap is 500; 10 rows fit either way.
    assert body["limit"] <= 500


# ---------------------------------------------------------------------------
# recall_conversation must clamp tail / max_chars to hard ceilings.
# Without these, a model that asked for tail=1_000_000 / max_chars=10_000_000
# would get essentially the full persisted conversation in one shot — both
# a memory/context blow and an unintended re-exposure path for content
# already trimmed earlier.
# ---------------------------------------------------------------------------

def test_recall_conversation_clamps_tail(tmp_path: Path) -> None:
    """An over-cap tail value clamps to MAX_TAIL and surfaces the
    clamp in the response."""
    events: list[dict] = []
    for i in range(50):
        events.append({"type": "user_message", "text": f"q{i}"})
        events.append({"type": "assistant_text", "text": f"a{i}"})
    _write_jsonl(tmp_path, events)
    with use_cwd(tmp_path):
        res = asyncio.run(
            HANDLERS["recall_conversation"]({"tail": 1_000_000})
        )
    body = _mcp_text(res)
    assert body["status"] == "ok"
    assert body.get("tail_clamped_to") == 200
    # turn_count is 50 (well under cap) so all returned, but tail's
    # clamp still surfaces so the model knows it asked for too much.
    assert body["returned"] <= 50


def test_recall_conversation_clamps_max_chars(tmp_path: Path) -> None:
    """An over-cap max_chars clamps to MAX_CHARS_CEILING and surfaces."""
    _write_jsonl(tmp_path, [
        {"type": "user_message", "text": "hi"},
        {"type": "assistant_text", "text": "hello"},
    ])
    with use_cwd(tmp_path):
        res = asyncio.run(
            HANDLERS["recall_conversation"]({"max_chars": 10_000_000})
        )
    body = _mcp_text(res)
    assert body["status"] == "ok"
    assert body.get("max_chars_clamped_to") == 64 * 1024


def test_recall_conversation_does_not_signal_clamp_when_within_bounds(
    tmp_path: Path,
) -> None:
    """No clamp metadata when the request is within ceilings."""
    _write_jsonl(tmp_path, [
        {"type": "user_message", "text": "hi"},
        {"type": "assistant_text", "text": "hello"},
    ])
    with use_cwd(tmp_path):
        res = asyncio.run(
            HANDLERS["recall_conversation"]({"tail": 5, "max_chars": 4000})
        )
    body = _mcp_text(res)
    assert body["status"] == "ok"
    assert "tail_clamped_to" not in body
    assert "max_chars_clamped_to" not in body


# ---------------------------------------------------------------------------
# Fix 3: recall_conversation budget includes tools / result_ids size
# ---------------------------------------------------------------------------

def test_recall_conversation_budget_counts_tool_array(
    tmp_path: Path,
) -> None:
    """A turn with many tool entries should consume budget proportional
    to the tools array's serialized size, not just user/assistant
    text. Construct a turn with ~24 result_ids and confirm
    max_chars stops it from overflowing."""
    events: list[dict] = [{"type": "user_message", "text": "spec sweep"}]
    # First turn: many tool calls so the rendered tool array is large.
    for i in range(24):
        events.append({
            "type": "tool_call",
            "name": "mcp__nora__submit_script",
            "call_id": f"c{i}",
            "input": {"language": "R", "label": f"spec {i}"},
        })
        events.append({
            "type": "tool_result",
            "call_id": f"c{i}",
            "text": json.dumps({"result_id": f"M{i+1}", "status": "ok"}),
            "is_error": False,
        })
    events.append({"type": "assistant_text", "text": "done"})

    # Second turn: small, but its presence proves the budget cut off
    # before reaching it when max_chars is tight.
    events.append({"type": "user_message", "text": "and again"})
    events.append({"type": "assistant_text", "text": "done"})
    _write_jsonl(tmp_path, events)

    # Tight budget: should drop the heavy turn from the response if
    # the budget math counts the tools array. With the old math
    # (user + assistant only) the heavy turn would slip in.
    with use_cwd(tmp_path):
        res = asyncio.run(HANDLERS["recall_conversation"](
            {"tail": 5, "max_chars": 400},
        ))
    body = _mcp_text(res)
    # We should end up with at most one turn under a tight budget.
    assert body["returned"] <= 1


# ---------------------------------------------------------------------------
# Fix 4: session_state pairs user/assistant from the same turn
# ---------------------------------------------------------------------------

def test_session_state_pairs_user_assistant_from_same_turn(
    tmp_path: Path,
) -> None:
    """In-flight turn: turn N has a user message but no assistant
    yet. Session state should pair the user from turn N with an
    empty assistant — NOT pair turn N's user with turn N-1's
    assistant."""
    cwd = tmp_path / "session"
    cwd.mkdir()
    # Two turns of completed exchange + one in-flight user.
    _write_jsonl(cwd, [
        {"type": "user_message", "text": "first question"},
        {"type": "assistant_text", "text": "first answer"},
        {"type": "user_message", "text": "second question"},
        {"type": "assistant_text", "text": "second answer"},
        {"type": "user_message", "text": "third question (in-flight)"},
    ])
    write_session_state(cwd)
    state = read_session_state(cwd)
    assert state is not None
    # The latest paired turn is the second one.
    assert state.last_user_message == "third question (in-flight)"
    # NOT "second answer" — that would be a mismatch with the new
    # in-flight user. Either empty (preferred) or the most recent
    # answer is acceptable; the bug we're fixing is the silent pair
    # of newest-user + older-assistant.
    assert state.last_assistant_summary == "", (
        f"expected empty assistant for in-flight turn, "
        f"got {state.last_assistant_summary!r}"
    )


def test_session_state_pairs_when_both_sides_present(
    tmp_path: Path,
) -> None:
    """Steady-state: latest turn has both user and assistant. The
    pair should be the latest turn's user + that turn's assistant."""
    cwd = tmp_path / "session"
    cwd.mkdir()
    _write_jsonl(cwd, [
        {"type": "user_message", "text": "first question"},
        {"type": "assistant_text", "text": "first answer"},
        {"type": "user_message", "text": "second question"},
        {"type": "assistant_text", "text": "second answer"},
    ])
    write_session_state(cwd)
    state = read_session_state(cwd)
    assert state is not None
    assert state.last_user_message == "second question"
    assert state.last_assistant_summary == "second answer"
