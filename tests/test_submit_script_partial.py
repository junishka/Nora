"""End-to-end test for the partial-success branch in ``submit_script``.

A script that emits N-1 valid payloads and then aborts on iteration N
must surface those N-1 payloads back to the model alongside the abort
debug_excerpt — not collapse to a single ``execution_failed`` with no
results. Without this, the model would defensively choose N separate
scripts to protect partial work, which negates the multi-result wire
format introduced in eb733d1.

Pinned properties (in order of importance):
- ``status`` is ``"execution_failed_partial"`` (not ``"execution_failed"``).
- The N-1 partials appear in ``results`` with their own ``result_id``s.
- All partials share the same ``script_run_id`` as the run.
- The abort cause reaches the model via ``debug_excerpt``.
- The stored rows under that ``script_run_id`` are recoverable from
  the on-disk store (so the researcher's audit path still works).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from nora.config import set_cwd
from nora.env_detect import detect_environment
from nora.store import get_store, reset_store_for_tests
from nora.tools import submit_script


def _python_ready() -> bool:
    e = detect_environment()
    if e.python is None or e.sandbox_exec is None:
        return False
    return not ({"pandas", "numpy"} & set(e.python.missing_packages))


_skip_no_python = pytest.mark.skipif(
    not _python_ready(),
    reason="needs python3 + pandas + numpy + sandbox-exec",
)


def _text_payload(response: dict) -> dict:
    text_block = next(
        b for b in response["content"] if b.get("type") == "text"
    )
    return json.loads(text_block["text"])


@_skip_no_python
def test_submit_script_resolves_source_row_count_once_per_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression test for the 20-minute lag observed on a 24-result
    multi-result run against a 3 GB .dta. Before the fix, the post-
    execution row-count audit re-loaded the source dataset on every
    iteration of the per-payload loop. With N results that meant N
    full pyreadstat reads of the same file. This test pins that the
    dataset row-count resolver is called at most ONCE per
    submit_script invocation regardless of how many payloads land.
    """
    set_cwd(tmp_path)
    reset_store_for_tests()

    # Build a tiny CSV the row-count audit can succeed against.
    import pandas as pd
    df = pd.DataFrame({"x": list(range(20))})
    src = tmp_path / "tiny.csv"
    df.to_csv(src, index=False)

    # Spy on _resolve_source_row_count to count invocations.
    calls = {"n": 0}
    from nora import tools as tools_mod
    real = tools_mod._resolve_source_row_count

    def counting(source_dataset):
        calls["n"] += 1
        return real(source_dataset)

    monkeypatch.setattr(tools_mod, "_resolve_source_row_count", counting)

    code = (
        "import nora\n"
        "for i in range(5):\n"
        "    nora.from_summarize(f'v{i}', n=10, mean=float(i), "
        "sd=0.1, missing_count=0)\n"
    )
    response = asyncio.run(submit_script.handler({
        "language": "Python",
        "code": code,
        "label": "row-count audit perf canary",
        "source_dataset": "tiny.csv",
    }))
    body = _text_payload(response)
    assert body["status"] == "ok"
    assert len(body["results"]) == 5
    # The whole point: ONE resolve, not five.
    assert calls["n"] == 1, (
        f"_resolve_source_row_count called {calls['n']} times for "
        f"a 5-result script — the per-payload loop should NOT trigger "
        f"a fresh row-count load on each iteration"
    )


@_skip_no_python
def test_submit_script_returns_phase_timings(tmp_path: Path) -> None:
    """The response carries a ``_phase_timings`` block so the model
    (and humans reading the audit trail) can tell where the wall
    clock went between the executor finishing and the tool returning.
    Without this, the slow row-count-audit regression hid behind
    ``duration_seconds`` which only reports the subprocess."""
    set_cwd(tmp_path)
    reset_store_for_tests()

    code = (
        "import nora\n"
        "nora.from_summarize('a', n=10, mean=1.0, sd=0.1, missing_count=0)\n"
    )
    response = asyncio.run(submit_script.handler({
        "language": "Python",
        "code": code,
        "label": "phase timings canary",
        "source_dataset": "",
    }))
    body = _text_payload(response)
    assert body["status"] == "ok"
    pt = body["_phase_timings"]
    for key in (
        "executor_seconds",
        "row_count_audit_seconds",
        "sanitize_seconds",
        "store_seconds",
    ):
        assert key in pt, f"missing phase timing: {key}"
        assert isinstance(pt[key], (int, float))
        assert pt[key] >= 0


def test_compact_payload_drops_vcov_and_vif_for_regressions() -> None:
    """The regression-specific trim drops the two largest collinearity-
    diagnostic fields, keeps the headline pattern. Same shape as
    ``expand_result(view="coefficients")``."""
    from nora.tools import _compact_payload
    full = {
        "type": "linear_regression",
        "n": 100,
        "coefficients": {"x1": 0.4, "x2": -0.1},
        "standard_errors": {"x1": 0.05, "x2": 0.04},
        "p_values": {"x1": 0.001, "x2": 0.06},
        "r_squared": 0.31,
        "condition_number": 4.2,
        "vif": {"x1": 1.05, "x2": 1.05},
        "vcov": {"x1": {"x1": 0.0025}, "x2": {"x2": 0.0016}},
    }
    out = _compact_payload(full)
    assert "vcov" not in out
    assert "vif" not in out
    assert out["coefficients"] == {"x1": 0.4, "x2": -0.1}
    assert out["r_squared"] == 0.31
    assert out["condition_number"] == 4.2


def test_compact_payload_passes_through_non_regression_types() -> None:
    """Non-regression payloads are already small; pass through unchanged."""
    from nora.tools import _compact_payload
    desc = {
        "type": "descriptive", "variable": "x",
        "n": 50, "mean": 1.0, "sd": 0.2, "missing_count": 0,
    }
    assert _compact_payload(desc) == desc


@_skip_no_python
def test_submit_script_inlines_canonical_markdown_per_result(
    tmp_path: Path,
) -> None:
    """Every ok-status result entry carries a ``markdown`` field
    rendered by ``nora.result_render.render_table``. The UI's
    canonical-tables panel reads this same field. Regression pin
    against the linter pass that twice removed this hookup
    (handoff would claim the field exists, code wouldn't deliver)."""
    set_cwd(tmp_path)
    reset_store_for_tests()

    code = (
        "import nora\n"
        "nora.from_summarize('a', n=20, mean=1.0, sd=0.1, missing_count=0)\n"
        "nora.from_summarize('b', n=20, mean=2.0, sd=0.2, missing_count=0)\n"
    )
    response = asyncio.run(submit_script.handler({
        "language": "Python",
        "code": code,
        "label": "markdown canary",
        "source_dataset": "",
    }))
    body = _text_payload(response)
    assert body["status"] == "ok"
    assert len(body["results"]) == 2
    for entry, var in zip(body["results"], ["a", "b"]):
        assert entry["status"] == "ok"
        md = entry.get("markdown")
        assert isinstance(md, str) and md, (
            f"expected canonical markdown for {var!r}, got {md!r}"
        )
        # Renderer's descriptive shape: Variable / n / Mean / SD / Missing.
        assert "Variable" in md
        assert var in md


@_skip_no_python
def test_submit_script_inlines_compact_payload_per_result(
    tmp_path: Path,
) -> None:
    """Each ok-status result entry carries its own ``payload`` field
    so the model can render coefficient tables directly from the
    submit_script response, instead of calling ``expand_result``
    once per result on a multi-result script. Inline payload is
    the same trim ``view="coefficients"`` applies — full sanitized
    data minus ``vcov`` / ``vif`` for regressions, full payload for
    other types."""
    set_cwd(tmp_path)
    reset_store_for_tests()

    code = (
        "import nora\n"
        "for i in range(3):\n"
        "    nora.from_summarize(f'v{i}', n=20+i, mean=float(i), "
        "sd=0.5, missing_count=0)\n"
    )
    response = asyncio.run(submit_script.handler({
        "language": "Python",
        "code": code,
        "label": "inline payload canary",
        "source_dataset": "",
    }))
    body = _text_payload(response)
    assert body["status"] == "ok"
    assert len(body["results"]) == 3
    for entry, expected_var, expected_n in zip(
        body["results"], ["v0", "v1", "v2"], [20, 21, 22]
    ):
        assert entry["status"] == "ok"
        payload = entry.get("payload")
        assert isinstance(payload, dict), entry
        # Descriptive type passes through full payload.
        assert payload.get("type") == "descriptive"
        assert payload.get("variable") == expected_var
        assert payload.get("n") == expected_n


def test_trim_oversize_inline_payloads_drops_payload_keeps_markdown() -> None:
    """When the per-result ``payload`` + ``markdown`` cost would push
    the assembled envelope past the budget, the helper drops only
    ``payload`` from each ok-status entry and reports the trim. The
    UI's per-result panels read ``markdown`` directly, so leaving it
    intact keeps the inline cards rendering — and the model can still
    quote tables off ``markdown`` without an ``expand_result`` round-
    trip. Pin against accidentally also dropping ``markdown``, which
    is what re-broke the user's 24-regression session."""
    from nora.tools import _INLINE_PAYLOAD_BUDGET, _trim_oversize_inline_payloads

    bulky_payload = {"type": "linear_regression", "n": 1000, "blob": "x" * 800}
    bulky_markdown = "| col | val |\n| --- | --- |\n" + ("| a | b |\n" * 30)
    results = [
        {
            "status": "ok",
            "result_id": f"M{i}",
            "label": f"spec {i}",
            "payload": dict(bulky_payload),
            "markdown": bulky_markdown,
        }
        for i in range(30)
    ]
    cost = (
        sum(len(json.dumps(r["payload"])) for r in results)
        + sum(len(r["markdown"]) for r in results)
    )
    assert cost > _INLINE_PAYLOAD_BUDGET, (
        f"sanity: synthesized {cost} chars must exceed budget "
        f"{_INLINE_PAYLOAD_BUDGET} for the trim to fire"
    )

    trimmed = _trim_oversize_inline_payloads(results)
    # New contract (two-stage trim): returns a dict of which stages
    # fired, not a single bool. Heavy payload + heavy markdown trips
    # both stages; this test pins stage 1 (payload drop) — assertion
    # is on payload_omitted, not on the whole dict shape, so a future
    # third stage flag can be added without rewriting the test.
    assert trimmed.get("payload_omitted") is True
    for entry in results:
        assert "payload" not in entry, f"payload should be dropped: {entry!r}"
        # When stage 2 also fires (heavy markdown after stage 1), each
        # entry's markdown becomes a stub pointing at ``expand_result``.
        # Confirm that case by checking for the stub marker — the bulky
        # original is gone, but the result_id is still discoverable in
        # the stub for round-tripping.
        if trimmed.get("markdown_omitted"):
            assert "expand_result" in entry["markdown"]
            assert entry["result_id"] in entry["markdown"]
        else:
            assert entry.get("markdown") == bulky_markdown, "markdown must stay"
        assert entry["status"] == "ok"
        assert entry["result_id"].startswith("M")


def test_trim_oversize_inline_payloads_no_trim_below_budget() -> None:
    """A handful of small results stays under the budget — the trim
    must NOT fire and ``payload`` must survive intact. Pin against
    accidentally trimming the small-N happy path (single regression,
    a few descriptives) where the model relies on inline payload to
    render the headline numbers without an extra round-trip."""
    from nora.tools import _trim_oversize_inline_payloads

    results = [
        {
            "status": "ok",
            "result_id": f"M{i}",
            "payload": {"type": "descriptive", "variable": f"v{i}", "n": 100},
            "markdown": "| Variable | n |\n| --- | --- |\n| v | 100 |",
        }
        for i in range(3)
    ]
    trimmed = _trim_oversize_inline_payloads(results)
    # Two-stage contract: dict with both flags False = no trim fired.
    assert trimmed == {"payload_omitted": False, "markdown_omitted": False}
    for entry in results:
        assert "payload" in entry
        assert isinstance(entry["payload"], dict)


@_skip_no_python
def test_submit_script_dedupes_shared_transformations(tmp_path: Path) -> None:
    """A multi-result script that emits N payloads typically generates
    the same SDC transformations (precision clamps, etc.) on each one.
    The response should hoist the common entries into a single
    ``transformations_summary`` at envelope level so the model isn't
    paying N copies of the same audit text. Per-result entries that
    differ stay per-result; the store keeps full lists for audit."""
    set_cwd(tmp_path)
    reset_store_for_tests()

    code = (
        "import nora\n"
        "nora.from_summarize('a', n=15, mean=1.0, sd=0.1, missing_count=0)\n"
        "nora.from_summarize('b', n=15, mean=2.0, sd=0.2, missing_count=0)\n"
        "nora.from_summarize('c', n=15, mean=3.0, sd=0.3, missing_count=0)\n"
    )
    response = asyncio.run(submit_script.handler({
        "language": "Python",
        "code": code,
        "label": "dedup canary",
        "source_dataset": "",
    }))
    body = _text_payload(response)
    assert body["status"] == "ok"
    assert len(body["results"]) == 3

    # If three identical SDC entries appeared on each result, dedup
    # should hoist them. The summary is non-empty whenever the three
    # results share at least one transformation.
    if "transformations_summary" in body:
        shared = body["transformations_summary"]
        assert isinstance(shared, list) and shared
        # No per-result list should still contain a hoisted entry.
        for r in body["results"]:
            for t in r.get("transformations", []):
                assert t not in shared, (
                    "shared entry not stripped from per-result list"
                )

    # The store keeps full transformation lists per row for audit
    # transparency, regardless of dedup in the response.
    store = get_store(tmp_path)
    grouped = store.list_by_script_run(body["script_run_id"])
    assert len(grouped) == 3
    # Each stored row carries its full list; if shared exists in the
    # response, those entries should still be present in the stored row.
    if "transformations_summary" in body:
        shared_set = set(body["transformations_summary"])
        for row in grouped:
            for t in shared_set:
                assert t in row.transformations, (
                    f"stored row {row.id} lost shared transformation {t!r}"
                )


@_skip_no_python
def test_submit_script_returns_partial_results_when_script_aborts(
    tmp_path: Path,
) -> None:
    set_cwd(tmp_path)
    reset_store_for_tests()

    code = (
        "import nora\n"
        "nora.from_summarize('a', n=10, mean=1.0, sd=0.1, missing_count=0)\n"
        "nora.from_summarize('b', n=20, mean=2.0, sd=0.2, missing_count=0)\n"
        "raise RuntimeError('thin cell on iteration 3')\n"
    )
    response = asyncio.run(submit_script.handler({
        "language": "Python",
        "code": code,
        "label": "partial-success canary",
        "source_dataset": "",
    }))
    body = _text_payload(response)

    # Envelope: partial-success, not bare failure.
    assert body["status"] == "execution_failed_partial", body
    assert body["script_run_id"], "missing script_run_id"
    assert "debug_excerpt" in body
    assert "thin cell on iteration 3" in body["debug_excerpt"]
    assert body["exit_code"] != 0

    # Two partials reached the model with their own result ids.
    results = body["results"]
    assert len(results) == 2, results
    assert all(r["status"] == "ok" for r in results)
    assert [r["analysis_type"] for r in results] == ["descriptive"] * 2
    assert all(r["result_id"] for r in results)

    # Both partials are persisted under the same script_run_id and
    # recoverable via the store; the researcher's audit path still
    # finds the abort context.
    store = get_store(tmp_path)
    grouped = store.list_by_script_run(body["script_run_id"])
    assert len(grouped) == 2
    assert {row.id for row in grouped} == {r["result_id"] for r in results}


@_skip_no_python
def test_status_is_failed_not_partial_when_emitted_payloads_all_rejected(
    tmp_path: Path,
) -> None:
    """The "execution_failed_partial" envelope is reserved for partial
    SUCCESS — at least one payload made it through SDC. When every
    emitted payload was rejected by the sanitizer AND the script
    also aborted, status is "execution_failed", because labelling
    the response "partial" would push the model to treat rejections
    as usable results.

    Both failure modes still surface to the model: the rejection
    rows stay in ``results`` (with their per-payload reasons) and
    the abort cause is in ``debug_excerpt``. The hint distinguishes
    the two so the model knows it's not just one problem.
    """
    set_cwd(tmp_path)
    reset_store_for_tests()

    # ``nora.result(type="totally_unknown_type")`` produces a payload
    # the sanitizer rejects as an unknown analysis type. Emit two of
    # those, then abort. No payload survives SDC.
    code = (
        "import nora\n"
        "nora.result(type='totally_unknown_type', variable='a')\n"
        "nora.result(type='totally_unknown_type', variable='b')\n"
        "raise RuntimeError('aborted after rejected emits')\n"
    )
    response = asyncio.run(submit_script.handler({
        "language": "Python",
        "code": code,
        "label": "all-rejected-then-aborted canary",
        "source_dataset": "",
    }))
    body = _text_payload(response)

    # NOT execution_failed_partial — the model would mistakenly read
    # rejection rows as usable partials.
    assert body["status"] == "execution_failed", body
    assert "debug_excerpt" in body
    assert "aborted after rejected emits" in body["debug_excerpt"]
    assert body["exit_code"] != 0

    # Rejection rows are still visible inline so the per-payload
    # reasons reach the model.
    results = body["results"]
    assert len(results) == 2, results
    assert all(r["status"] == "rejected_by_sanitizer" for r in results)

    # Hint distinguishes the two failure modes — generic "read
    # debug_excerpt before resubmit" alone would imply the
    # rejections were just symptoms of the abort.
    hint = body["hint"]
    assert "rejected" in hint.lower()
    assert "abort" in hint.lower()
    assert "independent" in hint.lower()

    # Diagnostic row is still persisted (the run dir must be
    # recoverable from the store even when only rejections came back).
    assert "result_id" in body, body


@_skip_no_python
def test_clean_exit_with_malformed_jsonl_line_stays_ok_with_warning(
    tmp_path: Path,
) -> None:
    """A clean-exit run (exit_code 0) that produces valid payloads
    AND one malformed JSONL line must surface as ``status="ok"``
    with the bad-line summary in ``warnings``, not as
    ``execution_failed_partial``.

    Before the fix, ``run_script`` would set ``error`` on any bad
    line, which flipped ``ok=False`` and demoted the envelope to
    ``execution_failed_partial`` — reading to the model as "the
    script aborted" even though the subprocess exited 0. The 24-
    spec / one-glitched-helper case is the canonical failure here:
    23 good results with one corrupt line should NOT make the model
    apologise about an abort that didn't happen.
    """
    set_cwd(tmp_path)
    reset_store_for_tests()

    # Real helper produces one valid token-stamped line; the trailing
    # write appends a non-JSON line directly to NORA_RESULT_PATH.
    # Script exits 0 — there is no abort, only a bad line.
    code = (
        "import os\n"
        "import nora\n"
        "nora.from_summarize('a', n=50, mean=1.0, sd=0.1, missing_count=0)\n"
        "open(os.environ['NORA_RESULT_PATH'], 'a').write("
        "'{\"type\":\"linear_regression\",\"n\":100,bogus,}\\n')\n"
    )
    response = asyncio.run(submit_script.handler({
        "language": "Python",
        "code": code,
        "label": "bad-line-but-clean-exit canary",
        "source_dataset": "",
    }))
    body = _text_payload(response)

    # Clean exit + at least one valid payload ⇒ envelope stays "ok".
    assert body["status"] == "ok", body
    assert body["exit_code"] == 0
    assert len(body["results"]) == 1
    assert body["results"][0]["status"] == "ok"

    # Bad-line context surfaces in ``warnings`` so the model can see
    # something went sideways without treating the whole run as failed.
    assert "warnings" in body, body
    assert any(
        "malformed result line" in w.lower() for w in body["warnings"]
    ), body["warnings"]
    # No debug_excerpt / reason — those are reserved for actual
    # execution failures.
    assert "reason" not in body
    assert "debug_excerpt" not in body
