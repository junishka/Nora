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
