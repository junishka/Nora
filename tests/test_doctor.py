"""Tests for the ``nora --doctor`` health check.

The doctor exists to close the researcher-side diagnostic gap: when
Nora can't run a script, the researcher needs to see *what's broken*
and *how to fix it* without depending on the chat model to speculate.

The invariants here:

  * Each runtime gets one report; status is the worst applicable
    level (``ok`` / ``warning`` / ``blocked``).
  * Apple's xcrun-stub failure shape — interpreter present-but-
    sandbox-rejected — is identified specifically and the advice
    names the fix (install a real Python via Homebrew or
    python.org), not just "python3 broken".
  * "blocked" on the report maps to a non-zero CLI exit code so a
    shell-init wrapper can gate the .app launch on a clean check.
  * The report serialises cleanly to a JSON-safe dict for the
    bridge method ``NoraBridge.doctor_report``.
"""

from __future__ import annotations

import pytest

from nora import env_detect as _env_detect_mod
from nora.env_detect import (
    Environment,
    Tool,
    _SANDBOX_PROBE_CACHE,
)
from nora.doctor import (
    DoctorReport,
    RuntimeReport,
    main_cli,
    render_report_text,
    run_doctor,
)


@pytest.fixture(autouse=True)
def _clear_probe_caches():
    """Reset both caches before each test. The baseline cache survives
    across tests by default; null it explicitly so test ordering
    can't masquerade as a cached value."""
    _SANDBOX_PROBE_CACHE.clear()
    _env_detect_mod._SANDBOX_BASELINE_CACHE = None
    yield
    _SANDBOX_PROBE_CACHE.clear()
    _env_detect_mod._SANDBOX_BASELINE_CACHE = None


def _env(python=None, r=None, stata=None, sandbox="/usr/bin/sandbox-exec"):
    return Environment(python=python, r=r, stata=stata, sandbox_exec=sandbox)


# ---------------------------------------------------------------------------
# Per-runtime classification
# ---------------------------------------------------------------------------

def test_doctor_marks_sandbox_blocked_when_missing():
    """No sandbox-exec → no scripts can run, period. The whole report
    is blocked even if every interpreter is present."""
    env = _env(
        python=Tool(name="Python", binary="/p", version="Python 3.12"),
        sandbox=None,
    )
    report = run_doctor(env)
    sandbox = next(r for r in report.runtimes if r.runtime == "sandbox-exec")
    assert sandbox.status == "blocked"
    assert report.blocked is True


def test_doctor_marks_sandbox_blocked_when_baseline_fails():
    """sandbox-exec present but baseline profile won't apply.
    Distinct row from "not present" — the advice diverges (this is
    not an install problem). Without this distinction a researcher
    inside a nested sandbox harness would get pointed at Homebrew."""
    _env_detect_mod._SANDBOX_BASELINE_CACHE = (
        False, "sandbox-exec rejected a minimal allow-default profile.",
    )
    env = _env(
        python=Tool(name="Python", binary="/p", version="Python 3.12"),
        sandbox="/usr/bin/sandbox-exec",
    )
    report = run_doctor(env)
    sandbox = next(r for r in report.runtimes if r.runtime == "sandbox-exec")
    assert sandbox.status == "blocked"
    assert "cannot apply a minimal profile" in sandbox.detail
    # The advice must name the actual cause (nested sandbox or
    # customised macOS), not "install something".
    advice = " ".join(sandbox.advice)
    assert "sandbox" in advice.lower()
    assert "install" not in advice.lower()


def test_doctor_marks_python_ok_when_healthy():
    py = Tool(
        name="Python", binary="/opt/homebrew/bin/python3",
        version="Python 3.12.0",
        missing_packages=(),
        optional_missing_packages=(),
    )
    report = run_doctor(_env(python=py))
    python = next(r for r in report.runtimes if r.runtime == "Python")
    assert python.status == "ok"
    assert "healthy" in python.detail.lower()


def test_doctor_marks_python_blocked_when_hard_packages_missing():
    """``pandas`` and ``numpy`` are hard-required by the runtime
    library itself. Without them every script crashes at runtime
    import. The doctor must mark this as ``blocked`` and the
    advice must name the install command."""
    py = Tool(
        name="Python", binary="/p", version="Python 3.12",
        missing_packages=("pandas", "numpy"),
    )
    report = run_doctor(_env(python=py))
    python = next(r for r in report.runtimes if r.runtime == "Python")
    assert python.status == "blocked"
    assert "pandas" in python.detail
    assert any("pip install" in tip for tip in python.advice)


def test_doctor_marks_python_warning_for_soft_missing_packages():
    """``statsmodels`` / ``scipy`` missing doesn't block all runs —
    descriptive helpers still work. Status is ``warning``, with
    advice on how to enable the missing helpers."""
    py = Tool(
        name="Python", binary="/p", version="Python 3.12",
        missing_packages=("statsmodels", "scipy"),
        optional_missing_packages=("matplotlib",),
    )
    report = run_doctor(_env(python=py))
    python = next(r for r in report.runtimes if r.runtime == "Python")
    assert python.status == "warning"
    assert "statsmodels" in python.detail
    assert "matplotlib" in python.detail


# ---------------------------------------------------------------------------
# Apple-stub failure shape — present-but-sandbox-rejected
# ---------------------------------------------------------------------------

def test_doctor_identifies_apple_xcrun_stub_rejection():
    """The exact bug class this branch is fixing: ``find_python``
    returned None (so ``env.python`` is None) NOT because no python3
    exists but because the sandbox probe rejected the candidates.
    The doctor must distinguish this from "no python3" and surface
    the fix (install a real Python)."""
    _SANDBOX_PROBE_CACHE["/usr/bin/python3"] = (
        False,
        "xcrun: error: unable to load libxcrun ... blocked open()",
    )
    report = run_doctor(_env(python=None))
    python = next(r for r in report.runtimes if r.runtime == "Python")
    assert python.status == "blocked"
    # Detail names the actual problem — not "not found on PATH".
    assert "/usr/bin/python3" in python.detail
    assert "sandbox" in python.detail.lower()
    # Advice points at the fix: Homebrew or python.org, NOT
    # "reinstall python3" or "fix your PATH".
    advice_text = " ".join(python.advice)
    assert "Homebrew" in advice_text or "brew install" in advice_text
    assert "python.org" in advice_text
    # The rejected-candidates list carries the binary + stderr tail
    # separately so the UI banner can show it expandably.
    assert any(path == "/usr/bin/python3"
               for path, _ in report.rejected_python_candidates)


def test_doctor_distinguishes_no_python3_from_rejected_python3():
    """When the cache is empty and python is None, the message is
    "not found on PATH" — different fix path from "rejected by
    sandbox". Confusing the two is exactly what the model did in
    the original incident."""
    report = run_doctor(_env(python=None))
    python = next(r for r in report.runtimes if r.runtime == "Python")
    assert "not found on PATH" in python.detail


# ---------------------------------------------------------------------------
# Optional runtimes — Stata absence is a warning, not a block
# ---------------------------------------------------------------------------

def test_doctor_missing_stata_does_not_block_overall_report():
    """Stata is optional. A researcher who only writes Python or R
    must not have ``--doctor`` exit non-zero just because Stata
    isn't installed. Per-language status reads ``blocked`` (Stata
    scripts will fail), but the top-level ``blocked`` flag is False
    so long as some other language is usable."""
    env = _env(python=Tool(name="Python", binary="/p", version="Python 3.12"))
    report = run_doctor(env)
    stata = next(r for r in report.runtimes if r.runtime == "Stata")
    # Per-language: Stata scripts would fail, so its row is "blocked".
    assert stata.status == "blocked"
    # Overall: Python is usable, so Nora is not blocked.
    assert report.blocked is False


def test_doctor_blocks_only_when_every_language_is_blocked():
    """``blocked`` on the top-level report means script execution
    will fail. One working language + one missing one is NOT
    blocked — the working language is still usable."""
    env = _env(
        python=Tool(name="Python", binary="/p", version="Python 3.12"),
        # R and Stata absent.
    )
    report = run_doctor(env)
    assert report.blocked is False


def test_doctor_blocked_when_no_language_is_usable():
    """No interpreters at all → blocked. Researcher cannot run
    anything regardless of which language they pick."""
    report = run_doctor(_env())
    assert report.blocked is True


# ---------------------------------------------------------------------------
# Rendering — terminal output and CLI exit code
# ---------------------------------------------------------------------------

def test_render_text_includes_status_glyphs_and_advice():
    py = Tool(
        name="Python", binary="/p", version="Python 3.12",
        missing_packages=("pandas",),
    )
    text = render_report_text(run_doctor(_env(python=py)))
    assert "[fail]" in text  # blocked status uses [fail]
    # Advice line is indented under the runtime line.
    assert "    ->" in text


def test_main_cli_returns_nonzero_when_blocked(capsys):
    """``nora --doctor`` must exit non-zero on blocked so shell-init
    wrappers can gate launch. Cleaning the cache + clearing the
    detected env lets us force ``blocked=True`` deterministically."""
    from nora import doctor as _doctor
    # Force the report to a blocked state by stubbing detect_environment.
    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr(_doctor, "detect_environment",
                       lambda: Environment(python=None, r=None, stata=None,
                                           sandbox_exec=None))
        rc = main_cli()
    finally:
        monkey.undo()
    assert rc != 0
    out = capsys.readouterr().out
    assert "BLOCKED" in out


def test_main_cli_returns_zero_when_healthy(capsys):
    from nora import doctor as _doctor
    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr(
            _doctor, "detect_environment",
            lambda: Environment(
                python=Tool(name="Python", binary="/p",
                            version="Python 3.12.0"),
                r=None, stata=None,
                sandbox_exec="/usr/bin/sandbox-exec",
            ),
        )
        rc = main_cli()
    finally:
        monkey.undo()
    assert rc == 0
    out = capsys.readouterr().out
    assert "OK" in out


# ---------------------------------------------------------------------------
# Bridge wiring — UI-facing JSON shape
# ---------------------------------------------------------------------------

def test_bridge_doctor_report_returns_json_safe_shape():
    """``NoraBridge.doctor_report`` must return a primitive-only dict
    so pywebview's JSON serialiser doesn't choke on the
    dataclasses. Asserts the exact keys the UI banner will consume."""
    from nora.ui import NoraBridge
    bridge = NoraBridge(cwd=None)
    payload = bridge.doctor_report()
    assert set(payload.keys()) >= {
        "blocked", "runtimes", "rejected_python_candidates",
    }
    assert isinstance(payload["blocked"], bool)
    assert isinstance(payload["runtimes"], list)
    for r in payload["runtimes"]:
        assert set(r.keys()) >= {"runtime", "status", "detail", "advice"}
        assert isinstance(r["advice"], list)
