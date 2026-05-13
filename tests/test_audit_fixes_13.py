"""Regression tests for the thirteenth batch of reviewer-flagged
fixes.

The behaviors pinned here:

1. The install-confirmation modal does NOT bind a global Enter ->
   respond(true) mapping. The prior code added a document-level
   keydown listener that approved on any Enter press while the
   modal was open; combined with the bubble-phase ordering of
   keydown propagation, this meant pressing Enter while the Deny
   button had focus approved the install (the document handler
   fired ``respond(true)`` before the button's default-click
   action got a chance, and once ``resolved=true`` the button's
   click respond(false) was a no-op). The fix removes the global
   mapping entirely and relies on native ``<button>`` Enter
   behavior, which fires the focused button's own click. Esc ->
   deny remains; that's safe regardless of focus.

2. ``nora.from_magnitude_table`` rejects ``**extra`` keys that
   would override helper-computed fields (``cells``,
   ``row_variable``, ``value_variable``, ``aggregation``, plus
   ``type`` / ``_via_helper``). Without this guard a caller could
   pass ``cells={"forged": ...}`` and ``fields.update(extra)``
   would replace the helper's raw-data computation; the
   ``_via_helper="from_magnitude_table"`` marker stamped at write
   time would then authenticate attacker-supplied values and the
   sanitizer (which trusts the marker to skip recomputing
   ``max_share``) would let a forged ``max_share=0`` bypass the
   dominance gate.

3. The R helper ``nora$from_magnitude_table`` mirrors the Python
   fix: extra-arg names colliding with helper-computed fields
   raise via ``stop()``. R uses ``c(list(computed), list(...))``
   so duplicate keys would emerge in JSON serialization;
   downstream ``json.loads`` keeps the last occurrence, which
   would again replace the helper-computed cells while leaving the
   marker authentic.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# 1. install-confirmation modal: no global Enter -> approve mapping
# ---------------------------------------------------------------------------


def test_install_confirmation_modal_has_no_global_enter_approve() -> None:
    """JS-only logic so the test is structural: read the source for
    the install-confirmation modal's keydown handler and assert
    Enter is NOT bound to ``respond(true)`` at the document level.
    Esc -> respond(false) must remain — that's the safe key
    binding regardless of which button has focus.

    The handler block is bounded by the ``const onKey = (e) => {``
    opening and the ``document.addEventListener('keydown', onKey)``
    line; this test isolates that block before pattern-matching so
    a similar Enter binding elsewhere in the file (eg a
    composer submit, image lightbox dismiss, etc.) doesn't shadow
    the assertion."""
    js_path = (
        Path(__file__).resolve().parents[1]
        / "src" / "nora" / "web" / "app.js"
    )
    src = js_path.read_text(encoding="utf-8")

    # Locate the install-confirmation modal's createInstallConfirmation
    # or showInstallConfirmation function via a stable nearby anchor.
    anchor = src.find("install-confirmation-overlay")
    assert anchor != -1, "install confirmation modal block not found"
    # Scan forward from the anchor to find the keydown handler block.
    onkey_open = src.find("const onKey = (e) => {", anchor)
    assert onkey_open != -1, "onKey handler not found near install modal"
    onkey_close = src.find(
        "document.addEventListener('keydown', onKey)", onkey_open,
    )
    assert onkey_close != -1, "onKey listener registration not found"
    onkey_block = src[onkey_open:onkey_close]

    # Esc -> respond(false) MUST be present.
    assert re.search(
        r"e\.key\s*===\s*'Escape'\s*\)\s*respond\(false\)",
        onkey_block,
    ), "Esc -> respond(false) binding is missing"

    # Enter -> respond(true) MUST NOT be present. Native button
    # Enter behavior on the focused element is the correct path.
    assert not re.search(
        r"e\.key\s*===\s*'Enter'[^}]*respond\(true\)",
        onkey_block,
    ), (
        "global Enter -> respond(true) mapping is present; it "
        "approves the install regardless of which button has focus "
        "because the document-level keydown listener fires on the "
        "bubble phase before the focused button's default-click "
        "action gets a chance."
    )


# ---------------------------------------------------------------------------
# 2. Python from_magnitude_table rejects reserved extras
# ---------------------------------------------------------------------------


def _run_helper(
    tmp_path: Path, script_body: str,
) -> tuple[int, str, list[dict]]:
    """Run a small Python script that imports the Nora runtime
    module and invokes the helper. ``nora.runtime.nora`` enforces
    ``NORA_RUN_TOKEN`` at import time, so we spawn a fresh
    subprocess instead of monkeypatching the parent's env. We
    prepend the runtime dir to ``sys.path`` (as the executor does
    in production via its preamble) so ``import nora`` resolves to
    the staged runtime file, not the top-level ``nora`` package.
    Returns (exit code, combined stdout+stderr, parsed result
    payloads).
    """
    import subprocess
    import sys
    result_path = tmp_path / "result.jsonl"
    env = os.environ.copy()
    env["NORA_RESULT_PATH"] = str(result_path)
    env["NORA_RUN_TOKEN"] = "test-token"
    runtime_dir = (
        Path(__file__).resolve().parents[1] / "src" / "nora" / "runtime"
    )
    preamble = (
        "import sys\n"
        f"sys.path.insert(0, {str(runtime_dir)!r})\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", preamble + script_body],
        env=env, capture_output=True, text=True, timeout=30,
    )
    payloads: list[dict] = []
    if result_path.exists():
        for line in result_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payloads.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return proc.returncode, proc.stdout + proc.stderr, payloads


def test_from_magnitude_table_rejects_reserved_extras_python(
    tmp_path: Path,
) -> None:
    """Calling the helper with ``cells=`` (or any other reserved
    key) in ``**extra`` raises ``ValueError`` before the payload is
    written. The marker would otherwise authenticate the forged
    cells, so this is the load-bearing defense for the helper-
    provenance contract."""
    pd = pytest.importorskip("pandas")
    del pd  # only used as availability gate
    # ``aggregation`` is a named keyword-only parameter on the helper
    # signature, so it can't reach ``**extra`` from Python's argument
    # binding rules — kept in the reject list as defense-in-depth
    # against a future signature refactor, but not testable here
    # through the public API. ``type`` / ``row_variable`` /
    # ``value_variable`` / ``cells`` / ``_via_helper`` all flow
    # through ``**extra`` and are the realistic attack vectors.
    for forbidden_kwarg in [
        'cells={"FORGED": {"value": 999.0, "n": 1, "max_share": 0.0}}',
        'type="magnitude_table"',
        'row_variable="spoofed"',
        'value_variable="spoofed"',
        '_via_helper="from_magnitude_table"',
    ]:
        code = (
            "import pandas as pd\n"
            "import nora\n"
            "df = pd.DataFrame({'g': ['a', 'a', 'b'], "
            "                   'v': [1.0, 2.0, 3.0]})\n"
            f"nora.from_magnitude_table(df, 'g', 'v', {forbidden_kwarg})\n"
        )
        rc, out, payloads = _run_helper(tmp_path, code)
        assert rc != 0, (
            f"helper should have rejected {forbidden_kwarg}; "
            f"output was: {out}"
        )
        assert "cannot override" in out, (
            f"expected reject message for {forbidden_kwarg}; "
            f"got: {out}"
        )
        # Nothing was written before the raise.
        assert payloads == [], (
            f"payload was written despite reject for "
            f"{forbidden_kwarg}: {payloads}"
        )
        # Wipe the result file between iterations.
        (tmp_path / "result.jsonl").unlink(missing_ok=True)


def test_from_magnitude_table_accepts_unreserved_extras_python(
    tmp_path: Path,
) -> None:
    """Unreserved keys (a ``label``, an analysis ``note``) pass
    through to the payload unchanged. The reject list must be
    exactly the helper-computed set, no broader."""
    pytest.importorskip("pandas")
    code = (
        "import pandas as pd\n"
        "import nora\n"
        "df = pd.DataFrame({'g': ['a', 'a', 'b'], "
        "                   'v': [1.0, 2.0, 3.0]})\n"
        "nora.from_magnitude_table(df, 'g', 'v', "
        "label='my label', note='some note')\n"
    )
    rc, out, payloads = _run_helper(tmp_path, code)
    assert rc == 0, f"helper crashed unexpectedly: {out}"
    assert len(payloads) == 1, f"expected one payload, got {payloads}"
    p = payloads[0]
    assert p["type"] == "magnitude_table"
    assert p["_via_helper"] == "from_magnitude_table"
    assert p["label"] == "my label"
    assert p["note"] == "some note"
    # Helper-computed fields survive.
    assert p["row_variable"] == "g"
    assert p["value_variable"] == "v"
    assert p["aggregation"] == "sum"
    assert "a" in p["cells"]
    assert p["cells"]["a"]["value"] == 3.0


# ---------------------------------------------------------------------------
# 3. R from_magnitude_table rejects reserved extras (structural)
# ---------------------------------------------------------------------------


def test_from_magnitude_table_rejects_reserved_extras_r_source() -> None:
    """R lives in nora.R and we can't reliably invoke it from the
    test environment without an R installation. Pin the guard
    structurally: the helper must (a) collect ``...`` extras into
    a named list and (b) ``stop()`` if any reserved name appears
    in that list before reaching the payload-construction step.
    """
    r_path = (
        Path(__file__).resolve().parents[1]
        / "src" / "nora" / "runtime" / "nora.R"
    )
    src = r_path.read_text(encoding="utf-8")

    # Isolate the from_magnitude_table function body.
    fn_open = src.find("nora$from_magnitude_table <- function(")
    assert fn_open != -1, "from_magnitude_table not found in nora.R"
    # Find the closing brace by counting (simple scan, no nested
    # functions in this helper).
    depth = 0
    body_start = src.find("{", fn_open)
    assert body_start != -1
    i = body_start
    while i < len(src):
        c = src[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
        i += 1
    body = src[fn_open:i + 1]

    # The guard collects extras into a list and rejects reserved
    # names before the payload concatenation.
    assert re.search(r"extras\s*<-\s*list\(\.\.\.\)", body), (
        "expected ``extras <- list(...)`` to capture the extra args"
    )
    assert re.search(
        r'reserved\s*<-\s*c\([^)]*"cells"[^)]*\)', body, re.DOTALL,
    ), "expected reserved-name vector including \"cells\""
    assert re.search(
        r"forbidden\s*<-\s*intersect\(names\(extras\),\s*reserved\)",
        body,
    ), "expected forbidden = intersect(names(extras), reserved)"
    assert re.search(
        r"if\s*\(\s*length\(forbidden\)\s*>\s*0\s*\)\s*\{\s*\n\s*"
        r"stop\(",
        body,
    ), "expected stop() when forbidden names appear"

    # The payload concatenation must use the validated ``extras``,
    # not ``list(...)`` directly (which would bypass the guard if a
    # future edit dropped the assignment).
    payload_block = body[body.find("payload <- c("):]
    assert "extras" in payload_block, (
        "payload concatenation must use validated ``extras``, "
        "not a fresh ``list(...)`` that bypasses the reject step"
    )


# ---------------------------------------------------------------------------
# 4. file_provenance content-binding integration through tools
# ---------------------------------------------------------------------------


def test_read_attached_file_rejects_overwritten_known_file(
    tmp_path: Path,
) -> None:
    """End-to-end: stage a file, overwrite it with different bytes
    (the SDC-bypass attack), and confirm
    ``read_attached_file`` refuses because the content fingerprint
    no longer matches. Closes the basename-only-trust gap."""
    from nora.config import set_cwd
    from nora.file_provenance import initialize
    set_cwd(tmp_path)

    # Researcher stages analysis.py.
    legit = tmp_path / "analysis.py"
    legit.write_text("import pandas as pd\n")
    initialize(tmp_path)

    # Sandbox-side script overwrites with raw-row bytes.
    legit.write_text("ROW_DATA = [(1, 'alice', 42000)]\n")

    # Recall path should reject. We hit the gate via the MCP
    # handler so any layer above the manifest is exercised too.
    import asyncio

    from nora.tools import HANDLERS
    payload = asyncio.run(
        HANDLERS["read_attached_file"]({"name": "analysis.py"}),
    )
    text = payload["content"][0]["text"]
    body = json.loads(text)
    assert body["status"] == "rejected"
