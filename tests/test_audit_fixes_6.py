"""Regression tests for the fifth batch of reviewer-flagged fixes.

Pins behaviors across policy, runner, sanitizer, and run-file
listing — see ``test_policy.py`` and ``test_sanitizer.py`` for the
full coverage of those modules; this file collects the fixes that
sit between modules.

1. ``set_dataset_policy`` preserves ``non_disclosive_variables`` when
   the researcher changes the schema-depth ceiling. Pre-fix, the
   per-variable opt-in list was overwritten with an empty tuple on
   any depth change — including reverting to the file-wide default,
   which dropped the entry entirely.

2. ``enumerate_run_dir_scripts`` disambiguates run-dir script names
   against cwd top-level filenames. Pre-fix, a top-level file
   shadowed any run-dir script with the same display name in
   ``list_session_files`` (silently dropped) AND in
   ``read_attached_file`` (resolved to the top-level file), so the
   model lost access to prior runs that happened to share a name.

3. The runner restores carried context (warm-start prefix,
   dataset-diff state, plot attachments) when the provider yields a
   terminal ``TurnError`` / ``AuthFailure`` as a normal event, not
   only when one is raised. Pre-fix, an auth or context-overflow
   failure on a resumed turn lost the resume prefix because the
   restoration branches only ran for thrown exceptions.
"""

from __future__ import annotations

from pathlib import Path


# ---------------------------------------------------------------------------
# 1. set_dataset_policy preserves non_disclosive_variables
# ---------------------------------------------------------------------------

def test_set_dataset_policy_preserves_non_disclosive_on_depth_change(
    tmp_path: Path,
) -> None:
    """Changing the schema-depth chip must not silently erase a
    researcher's per-variable min/max disclosure opt-ins.

    The two policy axes are independent: schema depth is the
    ceiling on metadata Claude can request; the non-disclosive list
    is the per-variable consent for raw-extreme disclosure in
    descriptive results. Touching one shouldn't reset the other.
    """
    from nora.policy import (
        DatasetPolicy,
        NoraPolicy,
        load_policy,
        save_policy,
    )
    from nora.ui import NoraBridge

    # Seed the policy with a researcher who opted two safe variables
    # into min/max disclosure at the rich default depth.
    initial = NoraPolicy(
        default_max_depth="names_types_labels_summary",
        datasets={
            "study.csv": DatasetPolicy(
                max_depth="names_types_labels",
                set_at="2026-04-21T14:20:00+00:00",
                non_disclosive_variables=("age", "education_years"),
            ),
        },
    )
    save_policy(tmp_path, initial)

    bridge = NoraBridge()
    bridge.cwd = tmp_path

    # Tighten the depth to names_only — the opt-in list must survive.
    bridge.set_dataset_policy("study.csv", "names_only")
    after_tighten = load_policy(tmp_path)
    entry = after_tighten.datasets["study.csv"]
    assert entry.max_depth == "names_only"
    assert entry.non_disclosive_variables == ("age", "education_years")

    # Revert to the file-wide default — the entry used to disappear,
    # taking the opt-in list with it. Now the opt-in list keeps the
    # entry alive at the default depth.
    bridge.set_dataset_policy(
        "study.csv", after_tighten.default_max_depth,
    )
    after_revert = load_policy(tmp_path)
    assert "study.csv" in after_revert.datasets
    revert_entry = after_revert.datasets["study.csv"]
    assert revert_entry.max_depth == after_revert.default_max_depth
    assert revert_entry.non_disclosive_variables == (
        "age", "education_years",
    )


def test_set_dataset_policy_drops_entry_when_no_opt_in_at_default(
    tmp_path: Path,
) -> None:
    """When a dataset has no non_disclosive opt-ins and the
    researcher reverts to the file-wide default depth, the entry
    drops out entirely — keeps the policy file tidy and matches the
    'I chose the default' = 'I never changed it' mental model."""
    from nora.policy import (
        DatasetPolicy,
        NoraPolicy,
        load_policy,
        save_policy,
    )
    from nora.ui import NoraBridge

    initial = NoraPolicy(
        default_max_depth="names_types_labels_summary",
        datasets={
            "plain.csv": DatasetPolicy(
                max_depth="names_only",
                set_at="2026-04-21T14:20:00+00:00",
            ),
        },
    )
    save_policy(tmp_path, initial)

    bridge = NoraBridge()
    bridge.cwd = tmp_path
    bridge.set_dataset_policy(
        "plain.csv", initial.default_max_depth,
    )
    reloaded = load_policy(tmp_path)
    assert "plain.csv" not in reloaded.datasets


# ---------------------------------------------------------------------------
# 2. Run-dir scripts disambiguate against top-level filenames
# ---------------------------------------------------------------------------

def test_run_dir_script_disambiguates_against_top_level_collision(
    tmp_path: Path,
) -> None:
    """A top-level file with the same display name as a run-dir
    script used to shadow the script in the model-facing listing
    (silently dropped) and in ``read_attached_file`` (the top-level
    file resolved first). Now the run-dir script's display name
    gets ``(short_id)`` appended so each row is unique and
    addressable."""
    from nora.run_files import (
        cwd_top_level_display_names,
        enumerate_run_dir_scripts,
        find_run_dir_script_by_name,
    )

    # Top-level file the researcher uploaded.
    (tmp_path / "analysis.do").write_text("// uploaded\n", encoding="utf-8")

    # Run-dir script labeled "analysis" — same display name without
    # disambiguation.
    run_dir = tmp_path / ".nora" / "runs" / "20260101_120000_abcd1234"
    run_dir.mkdir(parents=True)
    (run_dir / "script.do").write_text(
        "// from a prior submit_script\n", encoding="utf-8",
    )
    (run_dir / "label.txt").write_text("analysis", encoding="utf-8")

    reserved = cwd_top_level_display_names(tmp_path)
    assert "analysis.do" in reserved

    entries = enumerate_run_dir_scripts(
        tmp_path, reserved_names=reserved,
    )
    assert len(entries) == 1
    # Suffix appended; the original "analysis.do" stays reserved
    # for the top-level file.
    assert entries[0].display_name != "analysis.do"
    assert "abcd1234" in entries[0].display_name
    assert entries[0].display_name.endswith(".do")

    # Round-trip: pass the disambiguated name back and the lookup
    # finds the run-dir script (not the top-level file).
    found = find_run_dir_script_by_name(
        tmp_path, entries[0].display_name,
        reserved_names=reserved,
    )
    assert found is not None
    assert found == run_dir / "script.do"


def test_run_dir_script_no_suffix_without_collision(
    tmp_path: Path,
) -> None:
    """No top-level collision → no disambiguation suffix. The
    label-derived display name stays clean for the common case."""
    from nora.run_files import (
        cwd_top_level_display_names,
        enumerate_run_dir_scripts,
    )

    run_dir = tmp_path / ".nora" / "runs" / "20260101_120000_abcd1234"
    run_dir.mkdir(parents=True)
    (run_dir / "script.do").write_text("// prior run\n", encoding="utf-8")
    (run_dir / "label.txt").write_text("analysis", encoding="utf-8")

    reserved = cwd_top_level_display_names(tmp_path)
    entries = enumerate_run_dir_scripts(
        tmp_path, reserved_names=reserved,
    )
    assert len(entries) == 1
    assert entries[0].display_name == "analysis.do"


def test_list_session_files_surfaces_both_top_level_and_run_dir_collision(
    tmp_path: Path,
) -> None:
    """End-to-end: when a top-level file and a run-dir script
    share a label, both rows appear in the model-facing listing
    under distinct names. Pre-fix, the run-dir row was silently
    dropped on dedupe."""
    import asyncio

    from nora.config import use_cwd
    from nora.tools import HANDLERS

    (tmp_path / "analysis.do").write_text("// uploaded\n", encoding="utf-8")
    run_dir = tmp_path / ".nora" / "runs" / "20260101_120000_abcd1234"
    run_dir.mkdir(parents=True)
    (run_dir / "script.do").write_text(
        "// prior run\n", encoding="utf-8",
    )
    (run_dir / "label.txt").write_text("analysis", encoding="utf-8")

    with use_cwd(tmp_path):
        result = asyncio.run(HANDLERS["list_session_files"]({
            "kinds": ["script"],
        }))
    import json
    text_block = next(
        b for b in result["content"] if b.get("type") == "text"
    )
    body = json.loads(text_block["text"])
    assert body["status"] == "ok"
    names = [r["name"] for r in body["files"]]
    # Both rows surface; the run-dir row carries the disambiguating
    # suffix so the model can address it.
    assert "analysis.do" in names
    assert any("abcd1234" in n for n in names), names


# ---------------------------------------------------------------------------
# 3. Runner restores context on yielded TurnError / AuthFailure
# ---------------------------------------------------------------------------

def test_runner_restores_carried_state_on_yielded_terminal_error(
    tmp_path: Path,
) -> None:
    """If the provider yields a terminal ``TurnError`` /
    ``AuthFailure`` instead of raising, the runner's restoration
    branches must still run. Pre-fix, those branches were inside
    ``except`` clauses, so a yielded failure on the first resumed
    turn ate the warm-start prefix and any pending plot
    attachments — the retry shipped less context than the
    researcher expected.

    We exercise the actual ``run_turn`` by stubbing out the
    provider session so it yields a single ``TurnError`` and
    then completes; the runner should mark the warm-start flag
    back to ``True`` and re-prepend any plots it had drained for
    the failed attempt.
    """
    import asyncio
    from nora.provider import TurnError
    from nora.runner import SessionRunner

    runner = SessionRunner(cwd=tmp_path, provider="anthropic", model="opus")

    # Pre-load the runner with the state a resumed turn would carry:
    # warm-start needed and a pending plot the model hasn't seen yet.
    runner.needs_context_prefix = True
    runner.pending_plot_images = [
        {"name": "fit.png", "data": "<base64>", "mime": "image/png",
         "kind": "plot"},
    ]

    class _StubSession:
        async def send(self, prompt, images=None):
            yield TurnError(message="provider rejected the request")

        async def aclose(self):
            return None

    async def _ensure(self=None):
        return _StubSession()

    runner.ensure_session = _ensure  # type: ignore[assignment]

    events: list[dict] = []

    def _on_event(payload: dict) -> None:
        events.append(payload)

    asyncio.run(runner.run_turn(
        text="hello",
        images=None,
        on_event=_on_event,
        build_context_prefix=lambda cwd: "WARM-START PREFIX",
        build_script_prefix=lambda items, cwd: "",
        turn_id="0123456789abcdef",
    ))

    # Restoration ran: the warm-start flag is back on AND the plot
    # the failed turn drained is queued for the next attempt.
    assert runner.needs_context_prefix is True
    assert runner.pending_plot_images and (
        runner.pending_plot_images[0]["name"] == "fit.png"
    )
    # The failure surfaced to the dispatcher.
    assert any(e.get("type") == "turn_error" for e in events)
