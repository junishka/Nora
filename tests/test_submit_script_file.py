"""Tests for the ``submit_script_file`` tool.

The motivation is the round-trip cost of attached scripts: a researcher
@-mentions a 12 KB .do file and the model otherwise has to re-emit the
bytes through ``submit_script(code=...)``. ``submit_script_file`` reads
the bytes from disk by basename so the tool input stays small.

Behavioural pins:
- Path safety mirrors ``read_attached_file``: basename only, no escapes.
- Extension allowlist (.do / .R / .Rmd / .py) — text files refused.
- Language inference from extension when ``language`` is omitted.
- Empty file refused (would otherwise be a confusing executor error).
- Forwards to ``submit_script`` so the response shape is identical.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from nora.config import set_cwd
from nora.env_detect import detect_environment
from nora.store import reset_store_for_tests
from nora.tools import HANDLERS


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


def _call(args: dict) -> dict:
    return _text_payload(asyncio.run(HANDLERS["submit_script_file"](args)))


def test_missing_name_returns_error(tmp_path: Path) -> None:
    set_cwd(tmp_path)
    body = _call({"name": ""})
    assert body["status"] == "error"
    assert "name" in body["reason"]


def test_path_escape_rejected(tmp_path: Path) -> None:
    set_cwd(tmp_path)
    # Even with a directory component, the basename is what gets
    # resolved against cwd. A non-existent basename returns not_found.
    body = _call({"name": "../escape.do"})
    assert body["status"] in ("error", "not_found")


def test_unknown_extension_rejected(tmp_path: Path) -> None:
    set_cwd(tmp_path)
    (tmp_path / "notes.txt").write_text("# not a script\n", encoding="utf-8")
    body = _call({"name": "notes.txt"})
    assert body["status"] == "error"
    assert "not a recognised script" in body["reason"]


def test_empty_script_rejected(tmp_path: Path) -> None:
    set_cwd(tmp_path)
    (tmp_path / "blank.py").write_text("", encoding="utf-8")
    body = _call({"name": "blank.py"})
    assert body["status"] == "error"
    assert "empty" in body["reason"]


def test_missing_file_returns_not_found(tmp_path: Path) -> None:
    set_cwd(tmp_path)
    body = _call({"name": "nonexistent.do"})
    assert body["status"] == "not_found"


def test_language_override_conflicts_with_extension_rejected(
    tmp_path: Path,
) -> None:
    """A model that hands a ``.do`` file with ``language="Python"``
    used to silently win the override and route the Stata-syntax
    script to the Python interpreter — undefined behaviour, often
    surfacing as an opaque syntax error from cpython on Stata
    locals/globals. Reject loudly instead so the model can either
    drop the override or rename the file.
    """
    set_cwd(tmp_path)
    (tmp_path / "regression.do").write_text("regress y x\n", encoding="utf-8")
    body = _call({"name": "regression.do", "language": "Python"})
    assert body["status"] == "error"
    reason = body.get("reason") or ""
    assert "conflicts with the file extension" in reason
    assert ".do" in reason
    assert "Stata" in reason


def test_language_inference_from_extension(tmp_path: Path) -> None:
    """When ``language`` is omitted, the extension drives the choice.
    Verified at the rejection path (no Python installed in CI's R-only
    test env etc.) where the executor refuses without reaching
    subprocess. We instead check the inference indirectly: a .py file
    with no ``language`` argument should not surface an
    "unsupported language" error from submit_script (which would
    indicate the inference failed)."""
    set_cwd(tmp_path)
    (tmp_path / "tiny.py").write_text("import nora\n", encoding="utf-8")
    body = _call({"name": "tiny.py"})
    # Whatever happens (script error, sanitizer rejection, etc.), the
    # error must NOT be the "unsupported language" reason that
    # submit_script returns for unknown languages.
    if body.get("status") == "error":
        assert "unsupported language" not in (body.get("reason") or "")


@_skip_no_python
def test_submit_script_file_round_trips_a_real_python_script(
    tmp_path: Path,
) -> None:
    """End-to-end: write a .py file to disk, submit it via
    submit_script_file (no ``language`` arg), confirm the response
    shape matches submit_script's success envelope."""
    set_cwd(tmp_path)
    reset_store_for_tests()

    script = (
        "import nora\n"
        "nora.from_summarize('outcome', n=42, mean=3.14, sd=0.5, "
        "missing_count=2)\n"
    )
    (tmp_path / "smoke.py").write_text(script, encoding="utf-8")

    body = _call({
        "name": "smoke.py",
        "label": "smoke from file",
        "source_dataset": "",
    })
    assert body["status"] == "ok", body
    assert len(body["results"]) == 1
    entry = body["results"][0]
    assert entry["analysis_type"] == "descriptive"
    assert entry["payload"]["variable"] == "outcome"
    assert entry["payload"]["n"] == 42


@_skip_no_python
def test_submit_script_file_falls_back_label_to_filename(
    tmp_path: Path,
) -> None:
    """Default label is the basename so a researcher who didn't pass
    one gets a recognisable row label in the store."""
    set_cwd(tmp_path)
    reset_store_for_tests()

    script = (
        "import nora\n"
        "nora.from_summarize('x', n=10, mean=0.0, sd=1.0, missing_count=0)\n"
    )
    (tmp_path / "named.py").write_text(script, encoding="utf-8")

    body = _call({"name": "named.py"})
    assert body["status"] == "ok"
    assert body["results"][0]["label"] == "named.py"
