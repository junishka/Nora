"""Tests for the per-session durable state file.

``session_state.json`` is Builder's "at a glance" summary of a
session — last active time, last user/assistant exchange, recent
analytic results, active model, datasets present. The file is the
foundation for the session-list preview, the warm-start prefix's
result-label enrichment, and any future UI that wants to show
session state without opening the full chat log.

These tests lock in:

- The writer produces a well-formed file with all expected fields.
- It derives "last user" / "last assistant" from the persisted chat
  log correctly (including joining multi-block assistant replies).
- Recent results are sorted newest-first and capped.
- Atomic replace: a prior state survives a write that produces an
  identical schema; the JSON on disk never appears half-written.
- Missing / corrupted / wrong-version files read back as ``None``
  rather than crashing callers.
- The writer never raises — a persistence failure shouldn't crash a
  chat turn.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from builder.session_state import (
    SESSION_STATE_FILENAME,
    SESSION_STATE_VERSION,
    RecentResult,
    SessionState,
    read_session_state,
    write_session_state,
)


# --- Lightweight stub for StoredResult, to avoid touching the SQLite
#     store in these tests. The writer only uses .id, .label,
#     .analysis_type, .created_at.

@dataclass
class _StubResult:
    id: str
    label: str
    analysis_type: str
    created_at: str


def _write_chat_log(cwd: Path, events: list[dict]) -> None:
    (cwd / ".builder").mkdir(exist_ok=True)
    with (cwd / ".builder" / "chat_history.jsonl").open("w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")


def _state_path(cwd: Path) -> Path:
    return cwd / ".builder" / SESSION_STATE_FILENAME


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

def test_write_creates_file_with_minimal_state(tmp_path: Path):
    _write_chat_log(tmp_path, [])
    state = write_session_state(tmp_path, model="sonnet-4-6", store_list=[])
    assert state is not None
    assert _state_path(tmp_path).exists()
    raw = json.loads(_state_path(tmp_path).read_text())
    assert raw["version"] == SESSION_STATE_VERSION
    assert raw["turn_count"] == 0
    assert raw["last_user_message"] == ""
    assert raw["active_model"] == "sonnet-4-6"


def test_write_captures_last_exchange(tmp_path: Path):
    _write_chat_log(tmp_path, [
        {"type": "user_message", "text": "Q1"},
        {"type": "assistant_text", "text": "A1"},
        {"type": "user_message", "text": "Q2"},
        {"type": "assistant_text", "text": "A2"},
    ])
    state = write_session_state(tmp_path, store_list=[])
    assert state.turn_count == 2
    assert state.last_user_message == "Q2"
    assert state.last_assistant_summary == "A2"


def test_write_joins_multi_block_assistant_reply(tmp_path: Path):
    """When a turn has multiple assistant_text blocks (because of tool
    interleaves), the state file captures the joined reply — matching
    what read_turns returns."""
    _write_chat_log(tmp_path, [
        {"type": "user_message", "text": "run it"},
        {"type": "assistant_text", "text": "First, schema check."},
        {"type": "tool_call", "name": "mcp__builder__get_schema",
         "call_id": "c1", "input": {"dataset": "x.csv", "depth": "names_types"}},
        {"type": "tool_result", "call_id": "c1", "text": "{}", "is_error": False},
        {"type": "assistant_text", "text": "Result: coefficient -0.15."},
    ])
    state = write_session_state(tmp_path, store_list=[])
    assert "First, schema check" in state.last_assistant_summary
    assert "coefficient -0.15" in state.last_assistant_summary


def test_recent_results_sorted_newest_first_and_capped(tmp_path: Path):
    _write_chat_log(tmp_path, [])
    results = [
        _StubResult(id=f"r-{i}", label=f"result {i}",
                    analysis_type="linear_regression",
                    created_at=f"2026-04-{i:02d}T00:00:00+00:00")
        for i in range(1, 16)  # 15 results, oldest first
    ]
    state = write_session_state(tmp_path, store_list=results)
    # Cap is 10; newest first
    assert len(state.recent_results) == 10
    assert state.recent_results[0].id == "r-15"
    assert state.recent_results[-1].id == "r-6"


def test_datasets_enumerated_from_cwd(tmp_path: Path):
    _write_chat_log(tmp_path, [])
    (tmp_path / "a.csv").write_text("x\n1\n")
    (tmp_path / "b.dta").write_bytes(b"fake")
    (tmp_path / "notes.txt").write_text("ignored — wrong extension")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "x.csv").write_text("ignored — not top-level")

    state = write_session_state(tmp_path, store_list=[])
    assert sorted(state.datasets) == ["a.csv", "b.dta"]


def test_write_never_raises_on_bad_cwd(tmp_path: Path):
    """A non-existent cwd returns None rather than raising."""
    result = write_session_state(tmp_path / "does-not-exist", store_list=[])
    assert result is None


def test_write_is_atomic(tmp_path: Path, monkeypatch):
    """A crash mid-write must leave the previous state intact."""
    _write_chat_log(tmp_path, [])

    # Seed a valid prior state.
    write_session_state(tmp_path, model="first", store_list=[])
    assert read_session_state(tmp_path).active_model == "first"
    prior_contents = _state_path(tmp_path).read_text()

    # Force os.replace to fail, simulating a crash after the temp
    # file was written. The writer should swallow the error and
    # leave the prior file untouched on disk.
    import builder.session_state as ss_mod
    original_replace = ss_mod.os.replace

    def _boom(src, dst):
        raise OSError("disk full, hypothetically")

    monkeypatch.setattr(ss_mod.os, "replace", _boom)
    write_session_state(tmp_path, model="second", store_list=[])
    monkeypatch.setattr(ss_mod.os, "replace", original_replace)

    # Prior state still there, not partial.
    assert _state_path(tmp_path).read_text() == prior_contents
    assert read_session_state(tmp_path).active_model == "first"


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------

def test_read_returns_none_when_file_missing(tmp_path: Path):
    assert read_session_state(tmp_path) is None


def test_read_returns_none_on_malformed_json(tmp_path: Path):
    (tmp_path / ".builder").mkdir()
    _state_path(tmp_path).write_text("{ not valid json")
    assert read_session_state(tmp_path) is None


def test_read_returns_none_on_wrong_version(tmp_path: Path):
    (tmp_path / ".builder").mkdir()
    _state_path(tmp_path).write_text(json.dumps({
        "version": 99,
        "last_active_at": "x",
        "turn_count": 0,
    }))
    assert read_session_state(tmp_path) is None


def test_round_trip(tmp_path: Path):
    _write_chat_log(tmp_path, [
        {"type": "user_message", "text": "hello"},
        {"type": "assistant_text", "text": "hi"},
    ])
    write_session_state(
        tmp_path,
        model="sonnet-4-6",
        store_list=[
            _StubResult(
                id="r-1", label="OLS fit",
                analysis_type="linear_regression",
                created_at="2026-04-24T00:00:00+00:00",
            ),
        ],
    )
    loaded = read_session_state(tmp_path)
    assert loaded is not None
    assert loaded.turn_count == 1
    assert loaded.last_user_message == "hello"
    assert loaded.last_assistant_summary == "hi"
    assert loaded.active_model == "sonnet-4-6"
    assert len(loaded.recent_results) == 1
    assert loaded.recent_results[0].id == "r-1"
    assert loaded.recent_results[0].label == "OLS fit"


def test_read_handles_missing_optional_fields(tmp_path: Path):
    """Older state files may not have every field the current
    SessionState dataclass defines. The reader should fill in
    empty defaults rather than crashing."""
    (tmp_path / ".builder").mkdir()
    _state_path(tmp_path).write_text(json.dumps({
        "version": SESSION_STATE_VERSION,
        "last_active_at": "2026-04-24T00:00:00+00:00",
        "turn_count": 3,
        # Everything else omitted.
    }))
    state = read_session_state(tmp_path)
    assert state is not None
    assert state.turn_count == 3
    assert state.last_user_message == ""
    assert state.recent_results == []
    assert state.datasets == []
    assert state.active_model is None
