"""Regression tests for the sixth batch of reviewer-flagged fixes.

The behaviors pinned here:

1. ``rewind_to`` regenerates ``session_state.json`` after hiding
   results and truncating chat_history.jsonl. Pre-fix, the sidebar /
   session picker still showed the discarded branch's last
   user/assistant exchange and ``recent_results`` until the next
   successful turn rewrote the snapshot — across reloads, that meant
   the rewind looked like it didn't take.

2. ``_record_user_message`` persists image attachments inline as
   ``[{data, mime}]`` so a reload or session switch replays an
   image-bearing user prompt with its evidence intact. Pre-fix,
   only an integer ``image_count`` was stored, so replay surfaced
   the message as bare text.

3. ``schema.extract`` applies primary cell suppression to
   ``na_count`` at the summary depth. Pre-fix, a column with
   exactly one missing value (or one present value) leaked that
   subgroup count straight through ``get_schema``.

4. ``search_in_session_files`` honors a global response budget
   (total files, total matches, total chars). Pre-fix, every
   eligible file was scanned and every match appended; broad
   queries on a script-heavy session shipped megabytes of payload
   to the model.

5. ``list_results_global`` opens each session's result store via
   ``open_store_uncached`` and closes it after reading. Pre-fix,
   a single broad scan permanently retained a SQLite handle per
   visited session in the process-wide cache — file-descriptor
   pressure and stale handles in long-lived UI processes.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# 1. rewind regenerates session_state.json
# ---------------------------------------------------------------------------

def test_rewind_regenerates_session_snapshot(tmp_path: Path) -> None:
    """After ``rewind_to``, ``session_state.json`` must reflect the
    truncated history — not the discarded branch. Otherwise the
    sidebar / session picker keeps showing rewound content until the
    next turn rewrites the snapshot, which breaks the rewind promise
    across reloads."""
    from nora.session_state import (
        read_session_state,
        write_session_state,
    )
    from nora.ui import NoraBridge

    # Seed a chat history with two complete turns + an extra
    # user_message so we can rewind back to the first turn.
    history_dir = tmp_path / ".nora"
    history_dir.mkdir()
    history_path = history_dir / "chat_history.jsonl"
    records = [
        {"type": "user_message", "text": "first question"},
        {"type": "assistant_text", "text": "first answer"},
        {"type": "user_message", "text": "second question"},
        {"type": "assistant_text", "text": "second answer"},
    ]
    history_path.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n",
        encoding="utf-8",
    )

    # Seed a snapshot from the second turn — that's what was on disk
    # before the rewind.
    write_session_state(tmp_path, model="opus")
    pre = read_session_state(tmp_path)
    assert pre is not None
    assert pre.last_user_message == "second question"

    bridge = NoraBridge()
    bridge.cwd = tmp_path
    runner = bridge._ensure_runner_for_cwd(tmp_path)
    assert runner is not None

    res = bridge.rewind_to(1)  # truncate at the SECOND user_message
    assert res["ok"], res

    # Snapshot now reflects the truncated branch. Without the fix,
    # ``last_user_message`` still pointed at "second question" until
    # a real turn rewrote the file.
    post = read_session_state(tmp_path)
    assert post is not None
    assert post.last_user_message == "first question"
    assert post.last_assistant_summary == "first answer"


# ---------------------------------------------------------------------------
# 2. user_message images persist inline so replay shows them
# ---------------------------------------------------------------------------

def test_record_user_message_persists_image_blobs(tmp_path: Path) -> None:
    """Persisted ``user_message`` records carry the inline image
    blobs so reload/replay can render the original thumbnails. Pre-
    fix only ``image_count`` was stored, so replay produced a bare
    text bubble — researcher loses the audit trail of what was
    actually sent."""
    from nora.runner import SessionRunner
    from nora.ui import NoraBridge

    bridge = NoraBridge()
    bridge.cwd = tmp_path
    runner = SessionRunner(
        cwd=tmp_path, provider="anthropic", model="opus",
    )
    bridge._runners[str(tmp_path.resolve())] = runner

    # 1x1 transparent PNG, well under the 3 MB cap.
    tiny_png_b64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAA"
        "C0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
    )
    bridge._record_user_message(
        runner,
        "look at this plot",
        images=[{"data": tiny_png_b64, "mime": "image/png"}],
    )

    history = (tmp_path / ".nora" / "chat_history.jsonl").read_text(
        encoding="utf-8",
    ).strip().splitlines()
    record = json.loads(history[-1])
    assert record["type"] == "user_message"
    assert record["text"] == "look at this plot"
    assert record["image_count"] == 1
    # The persisted shape mirrors plot thumbnails: ``[{data, mime}]``.
    persisted = record.get("images")
    assert isinstance(persisted, list) and len(persisted) == 1
    assert persisted[0]["data"] == tiny_png_b64
    assert persisted[0]["mime"] == "image/png"


def test_record_user_message_skips_oversize_image(tmp_path: Path) -> None:
    """Images beyond the per-image persistence cap are dropped from
    the persisted record (``image_count`` still reflects the live
    send). Without this, a 50 MB attachment would inflate
    chat_history.jsonl on every reload."""
    from nora.runner import SessionRunner
    from nora.ui import NoraBridge

    bridge = NoraBridge()
    bridge.cwd = tmp_path
    runner = SessionRunner(
        cwd=tmp_path, provider="anthropic", model="opus",
    )
    bridge._runners[str(tmp_path.resolve())] = runner

    # A base64 string that decodes to roughly 4 MB — over the 3 MB cap.
    huge = "A" * (4 * 1024 * 1024 * 4 // 3)
    bridge._record_user_message(
        runner,
        "huge image",
        images=[{"data": huge, "mime": "image/png"}],
    )
    record = json.loads(
        (tmp_path / ".nora" / "chat_history.jsonl")
        .read_text(encoding="utf-8")
        .strip()
        .splitlines()[-1]
    )
    assert record["image_count"] == 1
    # No persisted blob — the live send still happened, but the
    # transcript falls back to "you sent an image" without bytes.
    assert "images" not in record


# ---------------------------------------------------------------------------
# 3. schema na_count cell suppression
# ---------------------------------------------------------------------------

def test_schema_summary_suppresses_rare_na_count(tmp_path: Path) -> None:
    """A column with exactly one missing value identifies that one
    observation. The schema summary used to publish ``na_count: 1``
    verbatim; now it goes through primary cell suppression and
    surfaces ``"<10"`` instead, same shape as
    :func:`nora.sdc.suppression_marker`."""
    import pandas as pd
    from nora import schema

    df = pd.DataFrame({
        "id": list(range(1, 101)),
        # 99 present, 1 missing — the rare-edge case the fix targets.
        "rare_missing": [1.0] * 99 + [None],
        # 50/50 — well above the threshold on both sides; should
        # surface its real count.
        "balanced": [None if i % 2 == 0 else 1.0 for i in range(100)],
        # All present — count of 0 is safe and passes through.
        "all_present": list(range(100)),
    })
    path = tmp_path / "study.csv"
    df.to_csv(path, index=False)

    out = schema.extract(path, "names_types_labels_summary")
    assert out["status"] == "ok"
    by_name = {v["name"]: v for v in out["variables"]}

    # Suppressed: rare missingness leaks identity.
    assert by_name["rare_missing"]["na_count"] == "<10"
    # Passed through: balanced count is well above the threshold.
    assert by_name["balanced"]["na_count"] == 50
    # Passed through: zero is informative and not disclosive.
    assert by_name["all_present"]["na_count"] == 0


def test_schema_summary_suppresses_rare_present_count(
    tmp_path: Path,
) -> None:
    """The rare-edge rule is symmetric: a column where ALL but a few
    are missing also re-identifies the few present values. Both
    sides go through the same primary-cell-suppression filter."""
    import pandas as pd
    from nora import schema

    df = pd.DataFrame({
        "mostly_missing": [1.0] + [None] * 99,
    })
    path = tmp_path / "sparse.csv"
    df.to_csv(path, index=False)

    out = schema.extract(path, "names_types_labels_summary")
    by_name = {v["name"]: v for v in out["variables"]}
    assert by_name["mostly_missing"]["na_count"] == "<10"


# ---------------------------------------------------------------------------
# 4. search_in_session_files global response cap
# ---------------------------------------------------------------------------

def test_search_in_session_files_truncates_with_total_files_cap(
    tmp_path: Path,
) -> None:
    """A broad query across many small scripts must stop at the
    global file cap and report ``truncated=true`` with a reason.
    Pre-fix, the per-file caps still let a wide search ship every
    eligible file's matches into one tool result."""
    from nora.config import use_cwd
    from nora.tools import HANDLERS, _SEARCH_FILES_TOTAL_FILES_CAP

    # Drop a few more scripts than the cap, all matching the query.
    n_files = _SEARCH_FILES_TOTAL_FILES_CAP + 5
    for i in range(n_files):
        (tmp_path / f"s{i:03d}.py").write_text(
            "TARGET_TOKEN here\n", encoding="utf-8",
        )
    # Stage the files so the search tool's SDC gate treats them as
    # researcher-known. In production the bridge runs ``initialize``
    # at session-open; tests that exercise the search path through
    # ``use_cwd`` bypass the bridge so do the same setup here.
    from nora.file_provenance import initialize as _init_staged
    _init_staged(tmp_path)

    with use_cwd(tmp_path):
        result = asyncio.run(HANDLERS["search_in_session_files"]({
            "query": "TARGET_TOKEN",
            "kinds": ["script"],
        }))
    body = json.loads(next(
        b for b in result["content"] if b.get("type") == "text"
    )["text"])
    assert body["status"] == "ok"
    assert body["truncated"] is True
    assert "truncated_reason" in body
    assert len(body["results"]) <= _SEARCH_FILES_TOTAL_FILES_CAP


def test_search_in_session_files_no_truncation_when_under_caps(
    tmp_path: Path,
) -> None:
    """A small search returns ``truncated=false`` and no
    ``truncated_reason``."""
    from nora.config import use_cwd
    from nora.tools import HANDLERS

    (tmp_path / "a.py").write_text("hello world\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("nothing here\n", encoding="utf-8")

    with use_cwd(tmp_path):
        result = asyncio.run(HANDLERS["search_in_session_files"]({
            "query": "hello",
            "kinds": ["script"],
        }))
    body = json.loads(next(
        b for b in result["content"] if b.get("type") == "text"
    )["text"])
    assert body["truncated"] is False
    assert "truncated_reason" not in body


# ---------------------------------------------------------------------------
# 5. list_results_global doesn't pin opened DBs in the cache
# ---------------------------------------------------------------------------

def test_list_results_global_does_not_pin_uncached_stores(
    tmp_path: Path, monkeypatch,
) -> None:
    """Global recall must not leave SQLite handles permanently
    cached for every session it touched. Walk three sessions; the
    cache must not grow as a result. Pre-fix, every visited
    session left a SQLite connection in the process-wide cache —
    long-lived UI processes would slowly burn through file
    descriptors and pin handles to deleted sessions."""
    import nora.store as store_module
    from nora.config import use_cwd
    from nora.tools import HANDLERS

    store_module.reset_store_for_tests()

    # Pretend three sessions live under SESSIONS_ROOT. Use
    # monkeypatch so we don't touch the real ~/.nora-sessions.
    sessions_root = tmp_path / "sessions"
    sessions_root.mkdir()
    monkeypatch.setattr("nora.ui.SESSIONS_ROOT", sessions_root)

    # Seed sessions WITHOUT going through ``get_store`` — that would
    # land in the cache and confuse the "did the global call grow
    # the cache?" measurement. Instead, open a fresh ResultStore
    # bound directly to each session's DB path, write the row, and
    # close. Mirrors what production code does when a long-lived
    # process visits a session for the first time.
    for i in range(3):
        sdir = sessions_root / f"session_{i}"
        sdir.mkdir()
        store = store_module.ResultStore(
            sdir / store_module.STORE_SUBDIR / store_module.DB_FILENAME,
        )
        store.insert(
            label=f"row {i}",
            analysis_type="linear_regression",
            sanitized_payload={"type": "linear_regression"},
            language="Stata",
            script_code="// ...",
            transformations=[],
            raw_log_path=None,
        )
        store.close()

    active = tmp_path / "active"
    active.mkdir()

    monkeypatch.setenv("NORA_ALLOW_CROSS_SESSION_RECALL", "1")
    cache_before = set(store_module._stores.keys())
    with use_cwd(active):
        result = asyncio.run(HANDLERS["list_results_global"]({}))
    body = json.loads(next(
        b for b in result["content"] if b.get("type") == "text"
    )["text"])
    assert body["status"] == "ok"
    assert body["total"] == 3

    cache_after = set(store_module._stores.keys())
    new_entries = cache_after - cache_before
    assert not new_entries, (
        f"global recall pinned new store handles in the cache: "
        f"{new_entries}"
    )

    store_module.reset_store_for_tests()
