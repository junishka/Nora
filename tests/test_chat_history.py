"""Tests for the turn-grouped reader over chat_history.jsonl.

The reader underpins both the warm-start prefix (ui._build_context_prefix)
and the recall_conversation tool, so if it grouped events wrong both
memory paths would lie to Claude. These tests lock in the grouping
behavior, tool pairing, and robustness to malformed lines.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from dataclasses import dataclass as _dc

from nora.chat_history import (
    Turn,
    ToolUse,
    build_context_prefix,
    read_turns,
    summarize_tool_call,
)


@_dc
class _StubResult:
    """Minimal stand-in for nora.store.StoredResult — carries just
    the fields build_context_prefix reads."""
    id: str
    label: str
    analysis_type: str
    created_at: str


def _write_jsonl(tmp_path: Path, events: list[dict[str, Any]]) -> Path:
    """Set up a session cwd with a seeded chat_history.jsonl and
    return the cwd — not the file path. read_turns expects a cwd
    rooted at the session dir, not a bare log path."""
    (tmp_path / ".nora").mkdir()
    log = tmp_path / ".nora" / "chat_history.jsonl"
    with log.open("w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
    return tmp_path


def test_none_cwd_returns_empty():
    assert read_turns(None) == []


def test_missing_file_returns_empty(tmp_path: Path):
    assert read_turns(tmp_path) == []


def test_empty_file_returns_empty(tmp_path: Path):
    (tmp_path / ".nora").mkdir()
    (tmp_path / ".nora" / "chat_history.jsonl").write_text("")
    assert read_turns(tmp_path) == []


def test_single_turn_user_only(tmp_path: Path):
    cwd = _write_jsonl(tmp_path, [
        {"type": "user_message", "text": "hello"},
    ])
    turns = read_turns(cwd)
    assert len(turns) == 1
    assert turns[0].user == "hello"
    assert turns[0].assistant == ""
    assert turns[0].tools == []
    assert turns[0].index == 0


def test_single_turn_user_then_assistant(tmp_path: Path):
    cwd = _write_jsonl(tmp_path, [
        {"type": "user_message", "text": "hi"},
        {"type": "assistant_text", "text": "Hello there."},
    ])
    turns = read_turns(cwd)
    assert len(turns) == 1
    t = turns[0]
    assert t.user == "hi"
    assert t.assistant == "Hello there."


def test_assistant_text_blocks_joined(tmp_path: Path):
    """A single turn may contain multiple assistant_text blocks when
    tool use interleaves. They should collapse into one joined field."""
    cwd = _write_jsonl(tmp_path, [
        {"type": "user_message", "text": "run OLS"},
        {"type": "assistant_text", "text": "Sure, first let me check the schema."},
        {"type": "tool_call", "name": "mcp__nora__get_schema",
         "call_id": "c1", "input": {"dataset": "x.csv", "depth": "names_types"}},
        {"type": "tool_result", "call_id": "c1", "text": "{}", "is_error": False},
        {"type": "assistant_text", "text": "Coefficient is -0.15 (p=0.04)."},
    ])
    turns = read_turns(cwd)
    assert len(turns) == 1
    assert "first let me check" in turns[0].assistant
    assert "Coefficient is -0.15" in turns[0].assistant
    assert "\n\n" in turns[0].assistant  # blocks joined with blank line


def test_multiple_turns_separated_by_user_message(tmp_path: Path):
    cwd = _write_jsonl(tmp_path, [
        {"type": "user_message", "text": "Q1"},
        {"type": "assistant_text", "text": "A1"},
        {"type": "user_message", "text": "Q2"},
        {"type": "assistant_text", "text": "A2"},
        {"type": "user_message", "text": "Q3"},
    ])
    turns = read_turns(cwd)
    assert [t.user for t in turns] == ["Q1", "Q2", "Q3"]
    assert [t.assistant for t in turns] == ["A1", "A2", ""]
    assert [t.index for t in turns] == [0, 1, 2]


def test_tool_call_and_result_paired_by_call_id(tmp_path: Path):
    cwd = _write_jsonl(tmp_path, [
        {"type": "user_message", "text": "run it"},
        {"type": "tool_call", "name": "mcp__nora__submit_script",
         "call_id": "abc", "input": {"language": "R", "label": "OLS fit"}},
        {"type": "tool_result", "call_id": "abc",
         "text": '{"result_id": "r-42", "status": "ok"}',
         "is_error": False},
        {"type": "assistant_text", "text": "Done."},
    ])
    turns = read_turns(cwd)
    assert len(turns) == 1
    t = turns[0]
    assert len(t.tools) == 1
    assert t.tools[0].name == "submit_script"
    assert t.tools[0].label == "R: OLS fit"
    assert t.tools[0].result_id == "r-42"
    assert t.tools[0].is_error is False
    assert t.result_ids == ["r-42"]


def test_tool_call_error_result_flagged(tmp_path: Path):
    cwd = _write_jsonl(tmp_path, [
        {"type": "user_message", "text": "run bad script"},
        {"type": "tool_call", "name": "mcp__nora__submit_script",
         "call_id": "x", "input": {"language": "R", "label": "boom"}},
        {"type": "tool_result", "call_id": "x",
         "text": "execution failed", "is_error": True},
    ])
    turns = read_turns(cwd)
    assert turns[0].tools[0].is_error is True
    assert turns[0].tools[0].result_id is None  # no JSON → no id


def test_orphan_events_before_first_user_message_are_dropped(tmp_path: Path):
    """Older logs sometimes started with a banner event. Those should
    not become a phantom turn-0."""
    cwd = _write_jsonl(tmp_path, [
        {"type": "assistant_text", "text": "banner, ignore me"},
        {"type": "user_message", "text": "real first message"},
        {"type": "assistant_text", "text": "real reply"},
    ])
    turns = read_turns(cwd)
    assert len(turns) == 1
    assert turns[0].user == "real first message"


def test_malformed_lines_are_skipped(tmp_path: Path):
    (tmp_path / ".nora").mkdir()
    log = tmp_path / ".nora" / "chat_history.jsonl"
    log.write_text(
        '{"type": "user_message", "text": "ok"}\n'
        'not-json-at-all\n'
        '\n'
        '{"type": "assistant_text", "text": "reply"}\n'
    )
    turns = read_turns(tmp_path)
    assert len(turns) == 1
    assert turns[0].user == "ok"
    assert turns[0].assistant == "reply"


def test_assistant_thinking_captured_separately(tmp_path: Path):
    cwd = _write_jsonl(tmp_path, [
        {"type": "user_message", "text": "Q"},
        {"type": "assistant_thinking", "text": "considering options"},
        {"type": "assistant_text", "text": "A"},
    ])
    turns = read_turns(cwd)
    assert turns[0].thinking == "considering options"
    assert turns[0].assistant == "A"


def test_timestamp_preserved_when_present(tmp_path: Path):
    cwd = _write_jsonl(tmp_path, [
        {"type": "user_message", "text": "Q",
         "timestamp": "2026-04-24T17:30:00+00:00"},
        {"type": "assistant_text", "text": "A"},
    ])
    turns = read_turns(cwd)
    assert turns[0].timestamp == "2026-04-24T17:30:00+00:00"


def test_attachments_counted(tmp_path: Path):
    cwd = _write_jsonl(tmp_path, [
        {"type": "user_message", "text": "look at these",
         "attachments": 3},
    ])
    turns = read_turns(cwd)
    assert turns[0].attachments == 3


def test_script_attachment_names_counted_without_crash(tmp_path: Path):
    """Web-UI sessions persist attached script filenames as a list so
    replay can redraw the chips. The turn reader should treat that as
    a count, not crash when resume logic rebuilds prior context."""
    cwd = _write_jsonl(tmp_path, [
        {"type": "user_message", "text": "please inspect this",
         "attachments": ["analysis.py", "robustness.do"]},
        {"type": "assistant_text", "text": "I can take a look."},
    ])
    turns = read_turns(cwd)
    assert turns[0].attachments == 2
    assert turns[0].assistant == "I can take a look."


# --- summarize_tool_call ---------------------------------------------------

def test_summarize_tool_call_submit_script():
    assert summarize_tool_call(
        "submit_script",
        {"language": "R", "label": "OLS"},
    ) == "R: OLS"


def test_summarize_tool_call_submit_script_no_label():
    assert summarize_tool_call("submit_script", {"language": "R"}) == "R: (no label)"


def test_summarize_tool_call_get_schema():
    assert summarize_tool_call(
        "get_schema",
        {"dataset": "x.csv", "depth": "names_types"},
    ) == "x.csv at names_types"


def test_summarize_tool_call_unknown_returns_empty():
    assert summarize_tool_call("unknown_tool", {"foo": "bar"}) == ""


def test_summarize_tool_call_non_dict_input():
    assert summarize_tool_call("submit_script", None) == ""  # type: ignore[arg-type]


# --- build_context_prefix -------------------------------------------------
#
# The warm-start prefix is what Claude actually sees on session
# resume. These tests lock in:
# - Empty inputs produce an empty prefix (don't pollute the first
#   turn of a brand-new session).
# - The recent-results block appears when results exist, ordered
#   newest-first, capped, and explicitly annotated.
# - The turn blocks carry user / tool / assistant lines with
#   result_id pointers so Claude can expand_result into the payload.
# - The header reports omitted / shown / total counts honestly.

def test_build_prefix_empty_returns_empty(tmp_path: Path):
    assert build_context_prefix(tmp_path, results=[]) == ""


def test_build_prefix_none_cwd_returns_empty():
    assert build_context_prefix(None, results=[]) == ""


def test_build_prefix_with_only_results_no_turns(tmp_path: Path):
    """Unusual but valid: results.db has entries but chat_history
    is empty. Still emit a prefix so analytical memory isn't lost."""
    prefix = build_context_prefix(tmp_path, results=[
        _StubResult(
            id="r-1", label="OLS fit",
            analysis_type="linear_regression",
            created_at="2026-04-24T00:00:00+00:00",
        ),
    ])
    assert "Prior conversation context" in prefix
    assert "Recent analytical results" in prefix
    assert "r-1: OLS fit [linear_regression]" in prefix
    assert "End of prior context" in prefix


def test_build_prefix_turns_and_results_together(tmp_path: Path):
    cwd = _write_jsonl(tmp_path, [
        {"type": "user_message", "text": "run OLS"},
        {"type": "tool_call", "name": "mcp__nora__submit_script",
         "call_id": "c1",
         "input": {"language": "R", "label": "OLS of log(salary)"}},
        {"type": "tool_result", "call_id": "c1",
         "text": '{"result_id": "r-42", "status": "ok"}'},
        {"type": "assistant_text", "text": "Coefficient -0.15."},
    ])
    prefix = build_context_prefix(cwd, results=[
        _StubResult(
            id="r-42", label="OLS of log(salary)",
            analysis_type="linear_regression",
            created_at="2026-04-24T17:25:00+00:00",
        ),
    ])
    # Results block present.
    assert "r-42: OLS of log(salary) [linear_regression]" in prefix
    # Turn block present with tool line and result_id pointer.
    assert "[turn 0]" in prefix
    assert "user: run OLS" in prefix
    assert "tool: [submit_script] R: OLS of log(salary)" in prefix
    assert "result_id=r-42" in prefix
    assert "assistant: Coefficient -0.15." in prefix


def test_build_prefix_results_sorted_newest_first_and_capped(tmp_path: Path):
    _write_jsonl(tmp_path, [{"type": "user_message", "text": "hi"}])
    many = [
        _StubResult(
            id=f"r-{i}", label=f"result {i}",
            analysis_type="linear_regression",
            created_at=f"2026-04-{i:02d}T00:00:00+00:00",
        )
        for i in range(1, 16)  # 15 results
    ]
    prefix = build_context_prefix(tmp_path, results=many)
    # Newest first.
    assert prefix.index("r-15:") < prefix.index("r-14:")
    # Cap is 10 — expect r-15 down to r-6 in the prefix body.
    for i in range(6, 16):
        assert f"r-{i}:" in prefix
    # The oldest 5 should be summarized as "5 older results".
    assert "5 older results" in prefix
    assert "r-5:" not in prefix


def test_build_prefix_header_counts_omitted_turns(tmp_path: Path):
    events: list[dict] = []
    for i in range(30):  # 30 turns — 10 beyond MAX_TURNS=20 cap
        events.append({"type": "user_message", "text": f"Q{i}"})
        events.append({"type": "assistant_text", "text": f"A{i}"})
    cwd = _write_jsonl(tmp_path, events)

    prefix = build_context_prefix(cwd, results=[])
    assert "10 earlier turns omitted" in prefix
    assert "showing last 20 of 30 turns" in prefix
    # Newest turn should appear (chronological, so last in output).
    assert "Q29" in prefix
    assert "A29" in prefix
    # Oldest-kept turn should appear — Q10 is at the 20-turn tail.
    assert "Q10" in prefix
    # Anything before that should NOT.
    assert "Q9\n" not in prefix


def test_build_prefix_per_field_caps_truncate_long_text(tmp_path: Path):
    """A single enormous user message should be capped at the
    per-field limit with a truncation marker, but the turn itself
    still appears in the prefix."""
    long_text = "x" * 3000  # exceeds the 1500-char per-field cap
    cwd = _write_jsonl(tmp_path, [
        {"type": "user_message", "text": long_text},
        {"type": "assistant_text", "text": "noted"},
    ])
    prefix = build_context_prefix(cwd, results=[])
    assert "[turn 0]" in prefix
    assert "…[truncated]" in prefix
    assert long_text not in prefix  # full text must not be present
    assert "assistant: noted" in prefix


def test_build_prefix_results_without_timestamp_dropped(tmp_path: Path):
    """Results missing created_at can't be ordered; we drop them
    rather than guessing a position."""
    _write_jsonl(tmp_path, [{"type": "user_message", "text": "hi"}])
    prefix = build_context_prefix(tmp_path, results=[
        _StubResult(id="r-good", label="valid",
                    analysis_type="linear_regression",
                    created_at="2026-04-24T00:00:00+00:00"),
        _StubResult(id="r-bad", label="undated",
                    analysis_type="linear_regression",
                    created_at=""),
    ])
    assert "r-good" in prefix
    assert "r-bad" not in prefix
