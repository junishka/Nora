"""Tests for ``context_count`` and the bridge's ``count_next_context``.

The chip used to mix four signals at once (provider usage, cache
fields, post_turn_tokens, chars/4 of pending messages) and visibly
fluctuated. The new pipeline collapses that to one definition: the
size of the next request, measured the same way every time, by a
backend method JS calls on a small set of triggers. These tests pin:

  - Counter math respects every contributor (history, draft,
    attachments, system prompt, tool schemas, image kicker).
  - Recount after rewind sees a smaller history.
  - The bridge method threads ``request_id`` through unchanged so
    JS can drop stale responses.
"""

from __future__ import annotations

import json
from pathlib import Path

from nora.context_count import count_next_context, to_payload


def _write_history(cwd: Path, events: list[dict]) -> None:
    nora_dir = cwd / ".nora"
    nora_dir.mkdir(parents=True, exist_ok=True)
    path = nora_dir / "chat_history.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")


def test_count_includes_history_draft_and_kickers(tmp_path: Path) -> None:
    """Every contributor adds to the count; baseline starts at the
    sum of the system-prompt + tool-schema chars passed in."""
    _write_history(tmp_path, [
        {"type": "user_message", "text": "x" * 100},
    ])
    base = count_next_context(
        cwd=tmp_path, draft_text="", n_images=0,
        system_prompt_chars=10_000, tool_schema_chars=20_000,
        ceiling=1_000_000, request_id=1,
    )
    with_draft = count_next_context(
        cwd=tmp_path, draft_text="y" * 50, n_images=0,
        system_prompt_chars=10_000, tool_schema_chars=20_000,
        ceiling=1_000_000, request_id=2,
    )
    with_image = count_next_context(
        cwd=tmp_path, draft_text="", n_images=1,
        system_prompt_chars=10_000, tool_schema_chars=20_000,
        ceiling=1_000_000, request_id=3,
    )
    assert with_draft.tokens > base.tokens
    assert with_image.tokens > base.tokens
    # Draft contributes via chars/3.5; one image contributes 1500
    # tokens (the constant). Image > 50-char draft.
    assert with_image.tokens - base.tokens > with_draft.tokens - base.tokens


def test_count_after_rewind_shrinks(tmp_path: Path) -> None:
    """The whole point of recounting after a rewind: the assembled
    request is smaller because the chat history was truncated.
    Without this, the chip would keep showing the pre-rewind size
    until the next turn fired and the provider's usage caught up."""
    # Pre-rewind: long history.
    _write_history(tmp_path, [
        {"type": "user_message", "text": "long " * 1000},
        {"type": "assistant_text", "text": "reply " * 1000},
        {"type": "user_message", "text": "long " * 1000},
        {"type": "assistant_text", "text": "reply " * 1000},
    ])
    before = count_next_context(
        cwd=tmp_path, draft_text="", n_images=0,
        system_prompt_chars=10_000, tool_schema_chars=20_000,
        ceiling=1_000_000, request_id=10,
    )

    # Post-rewind: history truncated to one turn.
    _write_history(tmp_path, [
        {"type": "user_message", "text": "long " * 1000},
    ])
    after = count_next_context(
        cwd=tmp_path, draft_text="", n_images=0,
        system_prompt_chars=10_000, tool_schema_chars=20_000,
        ceiling=1_000_000, request_id=11,
    )

    assert after.tokens < before.tokens, (
        f"recount after rewind should drop: before={before.tokens}, "
        f"after={after.tokens}"
    )


def test_count_threads_request_id_unchanged(tmp_path: Path) -> None:
    """JS uses ``request_id`` to drop stale responses landing after
    a newer request. The backend just echoes it through — any
    transformation here would let an old response masquerade as
    fresh."""
    c = count_next_context(
        cwd=tmp_path, request_id=12345,
    )
    assert c.request_id == 12345
    assert to_payload(c)["request_id"] == 12345


def test_count_handles_missing_history_file(tmp_path: Path) -> None:
    """Fresh session, no chat history yet. Should not raise and
    should return a count from the other contributors only."""
    c = count_next_context(
        cwd=tmp_path,
        system_prompt_chars=10_000, tool_schema_chars=20_000,
        ceiling=1_000_000, request_id=1,
    )
    assert c.tokens > 0
    assert c.exact is False


def test_count_handles_none_cwd() -> None:
    """No active session — no history to read. Should not raise.
    The bridge layer also short-circuits this case with ``ok=False``,
    but the helper should be safe regardless."""
    c = count_next_context(
        cwd=None,
        system_prompt_chars=10_000, tool_schema_chars=20_000,
        ceiling=1_000_000, request_id=1,
    )
    assert c.tokens >= 0


def test_bridge_count_next_context_returns_ok_with_request_id(
    tmp_path: Path,
) -> None:
    """End-to-end: the bridge wraps the helper and exposes it to JS.
    The response shape must match what ``triggerContextRecount`` in
    app.js consumes (ok / tokens / exact / ceiling / request_id)."""
    from nora.ui import NoraBridge
    bridge = NoraBridge(cwd=tmp_path)
    res = bridge.count_next_context(
        draft_text="hello", n_images=0, n_pending_attachments=0,
        request_id=99,
    )
    assert res["ok"] is True
    assert res["request_id"] == 99
    assert isinstance(res["tokens"], int)
    assert res["tokens"] > 0
    assert "exact" in res
    assert "ceiling" in res


def test_bridge_count_next_context_no_session() -> None:
    from nora.ui import NoraBridge
    bridge = NoraBridge(cwd=None)
    res = bridge.count_next_context(request_id=5)
    assert res["ok"] is False
    assert res["request_id"] == 5


def test_count_includes_pending_attachment_chars(tmp_path: Path) -> None:
    """A 90 KB script staged for the next send must shift the chip
    proportional to its bytes, not by a constant per-attachment
    kicker. Pre-fix: the chip was nearly flat because
    ``pending_attachment_chars`` wasn't a parameter and JS sent 0
    for the count."""
    base = count_next_context(
        cwd=tmp_path, draft_text="", n_images=0,
        n_pending_attachments=0, pending_attachment_chars=0,
        system_prompt_chars=10_000, tool_schema_chars=20_000,
    )
    one_small = count_next_context(
        cwd=tmp_path, draft_text="", n_images=0,
        n_pending_attachments=1, pending_attachment_chars=200,
        system_prompt_chars=10_000, tool_schema_chars=20_000,
    )
    one_big = count_next_context(
        cwd=tmp_path, draft_text="", n_images=0,
        n_pending_attachments=1, pending_attachment_chars=90_000,
        system_prompt_chars=10_000, tool_schema_chars=20_000,
    )
    assert one_small.tokens > base.tokens
    # The 90 KB attachment must move the chip dramatically more than
    # a 200-byte one — proportional to bytes, not a flat kicker.
    assert one_big.tokens - base.tokens > 10 * (
        one_small.tokens - base.tokens
    )


def test_bridge_count_includes_runner_pending_scripts(tmp_path: Path) -> None:
    """End-to-end: the bridge reads ``runner.pending_script_attachments``
    when the JS side passes 0 for the count, so a script staged via
    the file picker (which lives only on the runner) is reflected
    in the chip the moment it's attached.

    Pre-fix: ``count_next_context`` only saw the JS-supplied
    ``n_pending_attachments`` (always 0 because
    ``pendingComposerScriptCount`` returned 0), so a 90 KB ``.do``
    file left the chip flat until the next turn committed.
    """
    from nora.ui import NoraBridge
    bridge = NoraBridge(cwd=tmp_path)

    # Baseline with no pending scripts.
    base = bridge.count_next_context(
        draft_text="", n_images=0, n_pending_attachments=0,
        request_id=1,
    )
    assert base["ok"] is True

    # Plant a 90 KB script directly on the runner's staging list, as
    # the bridge's stage path would.
    runner = bridge._active_runner()
    assert runner is not None
    runner.pending_script_attachments.append({
        "name": "analysis.do",
        "ext": ".do",
        "content": "x" * 90_000,
    })

    bumped = bridge.count_next_context(
        draft_text="", n_images=0, n_pending_attachments=0,
        request_id=2,
    )
    assert bumped["ok"] is True
    # Bytes ride into the count: ~90k chars / 3.5 ≈ 25k tokens.
    assert bumped["tokens"] - base["tokens"] > 20_000, (
        f"expected staged 90 KB script to shift the chip by ≥20k "
        f"tokens, got {bumped['tokens'] - base['tokens']}"
    )
