"""End-to-end smoke tests for Python script execution.

Each test runs a tiny script under the real ``executor.run_script``
pipeline (subprocess, sandbox-exec, runtime library staging,
result-file authentication, payload parsing). Tests are gated on a
working python3 + pandas + numpy install so they self-skip in
environments where the executor would refuse anyway.

These tests are the Python analogue of ``test_executor_sandbox.py``
for R / Stata. Property-based sanitizer coverage lives in
``test_sanitizer.py`` and applies to all three languages once their
payloads land in the same JSON shapes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nora import env_detect, executor


# ---------------------------------------------------------------------------
# Skip-gate
# ---------------------------------------------------------------------------

def _python_ready() -> bool:
    e = env_detect.detect_environment()
    if e.python is None or e.sandbox_exec is None:
        return False
    # Hard requirements only — soft-recommended packages
    # (statsmodels, scipy) are checked per-test.
    hard = {"pandas", "numpy"} & set(e.python.missing_packages)
    return not hard


_skip_no_python = pytest.mark.skipif(
    not _python_ready(),
    reason=(
        "needs python3 + pandas + numpy + sandbox-exec; "
        "install missing pieces or run on macOS"
    ),
)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def test_python_is_detected_in_environment() -> None:
    """env_detect should at least find the binary on a Mac dev box."""
    e = env_detect.detect_environment()
    if e.python is not None:
        assert e.python.binary, "python.binary path should be populated"
        assert e.python.version is not None
        assert e.python.version.startswith("Python 3")
        # extra_read_paths should include sys.prefix (probed via subprocess).
        assert e.python.extra_read_paths, (
            "extra_read_paths should carry sys.prefix so the sandbox "
            "lets the interpreter read its own stdlib"
        )


# ---------------------------------------------------------------------------
# Refusal paths (don't need a real python install)
# ---------------------------------------------------------------------------

def test_run_script_refuses_python_without_interpreter(tmp_path: Path) -> None:
    """A research box without python3 should get a clear error
    pointing at how to install it — not a cryptic FileNotFoundError."""
    fake_env = env_detect.Environment(
        r=None,
        stata=None,
        python=None,
        sandbox_exec="/usr/bin/sandbox-exec",
    )
    res = executor.run_script(
        "Python", "import nora\nnora.result(type='descriptive')",
        tmp_path, env=fake_env,
    )
    assert not res.ok
    assert "python3 not found" in (res.error or "").lower()


def test_run_script_refuses_python_with_missing_hard_packages(
    tmp_path: Path,
) -> None:
    """If pandas is missing, the error should name the missing
    packages and the exact ``pip install`` to fix it. The runtime
    library imports pandas at module load, so without it the
    script would crash with a generic ImportError partway through —
    surfacing the install hint up-front is the friendlier path."""
    fake_env = env_detect.Environment(
        r=None,
        stata=None,
        python=env_detect.Tool(
            name="Python",
            binary="/usr/bin/python3",
            version="Python 3.12.0",
            missing_packages=("pandas", "numpy"),
        ),
        sandbox_exec="/usr/bin/sandbox-exec",
    )
    res = executor.run_script(
        "Python", "import nora\nnora.result(type='descriptive')",
        tmp_path, env=fake_env,
    )
    assert not res.ok
    err = (res.error or "").lower()
    assert "pandas" in err and "numpy" in err
    assert "pip install" in err


# ---------------------------------------------------------------------------
# Successful end-to-end paths
# ---------------------------------------------------------------------------

@_skip_no_python
def test_python_descriptive_round_trip(tmp_path: Path) -> None:
    """Smallest possible script: emit a descriptive payload via the
    runtime helper and confirm the executor returns the parsed,
    token-stripped dict."""
    code = (
        "import nora\n"
        "nora.from_summarize('outcome', n=42, mean=3.14, sd=0.5, "
        "missing_count=2)\n"
    )
    res = executor.run_script("Python", code, tmp_path, timeout_seconds=30)
    assert res.ok, f"expected success, got error={res.error!r}"
    assert res.exit_code == 0
    assert res.result_payloads
    assert res.result_payloads[0]["type"] == "descriptive"
    assert res.result_payloads[0]["variable"] == "outcome"
    assert res.result_payloads[0]["n"] == 42
    assert res.result_payloads[0]["mean"] == pytest.approx(3.14)
    assert res.result_payloads[0]["sd"] == pytest.approx(0.5)
    assert res.result_payloads[0]["missing_count"] == 2
    # Token must be stripped before the payload reaches us.
    assert "_token" not in res.result_payloads[0]


@_skip_no_python
def test_python_generic_result_round_trip(tmp_path: Path) -> None:
    """The generic ``nora.result(type=..., **fields)`` path must work
    too — that's how researchers emit non-standard analyses (custom
    bootstraps, sklearn-fit models, etc.)."""
    code = (
        "import nora\n"
        "nora.result(type='descriptive', variable='manual', "
        "n=10, mean=1.0, sd=0.1, missing_count=0)\n"
    )
    res = executor.run_script("Python", code, tmp_path, timeout_seconds=30)
    assert res.ok, res.error
    assert res.result_payloads[0]["variable"] == "manual"


@_skip_no_python
def test_python_script_without_runtime_call_is_rejected(
    tmp_path: Path,
) -> None:
    """A script that runs cleanly but emits no result should be
    flagged so the model knows to add a nora.* call. The error
    message should mention the runtime-library entry point so the
    fix is obvious."""
    code = "x = 1 + 1\n"
    res = executor.run_script("Python", code, tmp_path, timeout_seconds=30)
    assert not res.ok
    assert "result" in (res.error or "").lower()


@_skip_no_python
def test_python_multiple_helpers_one_script(tmp_path: Path) -> None:
    """A script can call nora.* helpers more than once. Each call
    appends a JSONL line and the executor returns the full list in
    emission order. Verifies the multi-result wire format end-to-end."""
    code = (
        "import nora\n"
        "nora.from_summarize('a', n=10, mean=1.0, sd=0.1, missing_count=0)\n"
        "nora.from_summarize('b', n=20, mean=2.0, sd=0.2, missing_count=0)\n"
        "nora.from_summarize('c', n=30, mean=3.0, sd=0.3, missing_count=0)\n"
    )
    res = executor.run_script("Python", code, tmp_path, timeout_seconds=30)
    assert res.ok, res.error
    assert len(res.result_payloads) == 3
    assert [p["variable"] for p in res.result_payloads] == ["a", "b", "c"]
    assert [p["n"] for p in res.result_payloads] == [10, 20, 30]
    # Tokens stripped from every line.
    assert all("_token" not in p for p in res.result_payloads)


@_skip_no_python
def test_python_handcrafted_payload_rejected(tmp_path: Path) -> None:
    """Authenticity guard: a script that writes JSON directly to
    NORA_RESULT_PATH (bypassing the runtime library) has no token
    and must be rejected. Same property the R / Stata paths enforce."""
    code = (
        "import os, json\n"
        "with open(os.environ['NORA_RESULT_PATH'], 'w') as f:\n"
        "    json.dump({'type': 'descriptive', 'variable': 'x', "
        "'n': 1, 'mean': 0, 'sd': 0, 'missing_count': 0}, f)\n"
    )
    res = executor.run_script("Python", code, tmp_path, timeout_seconds=30)
    assert not res.ok
    assert "token" in (res.error or "").lower()


@_skip_no_python
def test_python_subprocess_env_has_no_anthropic_key(tmp_path: Path) -> None:
    """The subprocess env-var allowlist must keep ANTHROPIC_API_KEY
    out of the script's reach. Same guarantee the R / Stata paths
    have — with an extra script-emits-the-env to prove it from
    inside the sandbox."""
    import os

    # Inject a fake key into the parent env so we can prove it
    # doesn't propagate. monkeypatch isn't available here without
    # the fixture; use os.environ manipulation in a try/finally.
    sentinel = "sk-ant-this-must-not-leak-1234567890"
    parent_had = "ANTHROPIC_API_KEY" in os.environ
    parent_value = os.environ.get("ANTHROPIC_API_KEY")
    os.environ["ANTHROPIC_API_KEY"] = sentinel
    try:
        code = (
            "import os, nora\n"
            "got = os.environ.get('ANTHROPIC_API_KEY', '__MISSING__')\n"
            "nora.from_summarize('leak', n=1, mean=0, sd=0, "
            "missing_count=0, transformations=[got])\n"
        )
        res = executor.run_script(
            "Python", code, tmp_path, timeout_seconds=30,
        )
    finally:
        if parent_had:
            os.environ["ANTHROPIC_API_KEY"] = parent_value or ""
        else:
            os.environ.pop("ANTHROPIC_API_KEY", None)

    assert res.ok, res.error
    # The "transformations" field on the payload is where the script
    # tried to surface the leaked secret. Should be the missing-marker,
    # never the sentinel.
    smuggled = res.result_payloads[0].get("transformations") or []
    assert sentinel not in (smuggled[0] if smuggled else "")
    assert "__MISSING__" in (smuggled[0] if smuggled else "")
