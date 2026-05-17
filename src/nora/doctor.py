"""Researcher-facing health check for the Nora runtime environment.

Why this module exists: the bug class this whole branch is addressing
(Apple xcrun stub fails inside the sandbox; every script silently
dies; the model speculates) is fundamentally a diagnostic-gap
problem. With the sandbox probe in ``env_detect``, the gap is closed
at detection time — Nora knows which interpreters work — but the
researcher still has no way to see that information without
submitting a script and watching it fail.

The doctor closes the researcher-side gap. It runs the same probes
the executor relies on, prints a short status per runtime, and
when something is wrong it names the *fix* (install Homebrew Python,
allow openssl.cnf, etc.) rather than the failure mode.

Two surfaces consume it:

  * ``nora --doctor`` (CLI) — for terminal launches and incident
    triage. Returns a non-zero exit code if any blocking issue
    exists, so a shell-init wrapper can refuse to launch the UI
    until the environment is healthy.
  * The UI banner — same data structure, rendered as a chat-side
    alert when ``Python`` (or whichever runtime the researcher's
    script needs) is unhealthy. Without this, the chat UI accepts
    the script, the executor refuses it, and the model paraphrases
    the refusal back to the researcher with no actionable next
    step.

Phase-safe by construction: every field this module produces is an
interpreter path, version string, package name, or launcher-stderr
tail. None of them touch researcher data, so the report can be
rendered anywhere (terminal stdout, UI banner, log file) without
the redaction posture the executor's runtime stderr needs.
"""

from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass, field
from typing import Literal

from nora.env_detect import (
    _PYTHON_REQUIRED_PACKAGES,
    Environment,
    detect_environment,
    python_sandbox_probe_results,
)


Status = Literal["ok", "warning", "blocked"]


@dataclass
class RuntimeReport:
    """Per-runtime block within ``DoctorReport``.

    ``status`` is the worst applicable level:
      * ``ok`` — usable as-is.
      * ``warning`` — usable, but a feature is degraded
        (e.g., Python without matplotlib means plots won't render).
      * ``blocked`` — present-but-unusable (probe failed, hard
        packages missing) or absent. Submitting a script in this
        language will fail.

    ``advice`` is the concrete next step. Phrased imperatively so the
    UI can render it as a button label or a copyable command.
    """
    runtime: str            # "Python" / "R" / "Stata" / "sandbox-exec"
    status: Status
    detail: str             # one-line human summary
    advice: list[str] = field(default_factory=list)


@dataclass
class DoctorReport:
    """Top-level report assembled by ``run_doctor``.

    ``blocked`` is True iff any RuntimeReport is at ``blocked``
    status AND that runtime is on Nora's required path (today:
    ``sandbox-exec``, plus at least one of R/Python/Stata being
    usable). The CLI's exit code derives from this.

    ``rejected_python_candidates`` is the list of (path, stderr_tail)
    entries from the sandbox-health probe cache — surfaced
    separately so a researcher who has a *working* Python alongside
    the rejected one still sees that Nora deliberately skipped the
    other binary (and why). The Apple-xcrun-stub case is the
    canonical example.
    """
    runtimes: list[RuntimeReport]
    rejected_python_candidates: list[tuple[str, str]]
    blocked: bool


def _python_report(env: Environment) -> RuntimeReport:
    py = env.python
    if py is None:
        # ``find_python`` returned None. Two distinct shapes: no
        # python3 on PATH at all, or every candidate failed the
        # sandbox probe. The probe cache disambiguates.
        rejected = [
            (path, stderr) for path, (ok, stderr)
            in python_sandbox_probe_results().items() if not ok
        ]
        if rejected:
            tail = (rejected[0][1] or "").strip().splitlines()
            stderr_summary = tail[-1] if tail else "(no stderr captured)"
            return RuntimeReport(
                runtime="Python",
                status="blocked",
                detail=(
                    f"python3 was found at {rejected[0][0]} but failed "
                    f"the sandbox-health probe: {stderr_summary}"
                ),
                advice=[
                    "Install a real Python via Homebrew "
                    "(``brew install python``) or python.org. "
                    "Apple's bundled /usr/bin/python3 is an xcselect "
                    "stub that cannot start under the Nora sandbox.",
                    "After installing, re-launch Nora.",
                ],
            )
        return RuntimeReport(
            runtime="Python",
            status="blocked",
            detail="python3 not found on PATH.",
            advice=[
                "Install Python 3 via Homebrew (``brew install python``) "
                "or python.org and re-launch Nora.",
            ],
        )

    hard_missing = sorted(
        set(py.missing_packages or ()) & set(_PYTHON_REQUIRED_PACKAGES[:2])
    )
    if hard_missing:
        return RuntimeReport(
            runtime="Python",
            status="blocked",
            detail=(
                f"Python at {py.binary} is missing required packages: "
                f"{', '.join(hard_missing)}"
            ),
            advice=[
                f"``{py.binary} -m pip install {' '.join(hard_missing)}``",
                "Then re-launch Nora.",
            ],
        )
    soft_missing = sorted(
        set(py.missing_packages or ()) - set(_PYTHON_REQUIRED_PACKAGES[:2])
    )
    optional_missing = sorted(py.optional_missing_packages or ())
    if soft_missing or optional_missing:
        notes = []
        if soft_missing:
            notes.append(
                "soft-required packages missing: "
                f"{', '.join(soft_missing)} (statsmodels / scipy helpers "
                "won't work without these)"
            )
        if optional_missing:
            notes.append(
                "optional packages missing: "
                f"{', '.join(optional_missing)} (plot helpers won't work "
                "without matplotlib)"
            )
        return RuntimeReport(
            runtime="Python",
            status="warning",
            detail=(
                f"Python at {py.binary} ({py.version}) is usable. "
                + "; ".join(notes)
            ),
            advice=[
                f"``{py.binary} -m pip install "
                f"{' '.join(soft_missing + optional_missing)}`` "
                "to enable the missing helpers."
            ] if (soft_missing or optional_missing) else [],
        )
    return RuntimeReport(
        runtime="Python",
        status="ok",
        detail=f"Python at {py.binary} ({py.version}) is healthy.",
    )


def _r_report(env: Environment) -> RuntimeReport:
    r = env.r
    if r is None:
        return RuntimeReport(
            runtime="R",
            status="blocked",
            detail="Rscript not found on PATH.",
            advice=[
                "Install R from https://cran.r-project.org and "
                "re-launch Nora, or submit scripts in Python / Stata."
            ],
        )
    optional_missing = sorted(r.optional_missing_packages or ())
    if optional_missing:
        return RuntimeReport(
            runtime="R",
            status="warning",
            detail=(
                f"R at {r.binary} ({r.version or 'version unknown'}) is "
                f"usable. Optional packages missing: "
                f"{', '.join(optional_missing)}"
            ),
            advice=[
                f"In R: ``install.packages(c({', '.join(repr(p) for p in optional_missing)}))``"
            ],
        )
    return RuntimeReport(
        runtime="R",
        status="ok",
        detail=f"R at {r.binary} ({r.version or 'version unknown'}) is healthy.",
    )


def _stata_report(env: Environment) -> RuntimeReport:
    st = env.stata
    if st is None:
        # Per-language status reflects usability of THIS language, not
        # whether Nora as a whole is unusable. A Python-only researcher
        # still sees Stata as blocked here (Stata scripts would fail)
        # but ``DoctorReport.blocked`` only fires when no language is
        # usable. Without this distinction, the per-language status
        # has to lie to keep the overall block calculation tractable —
        # confusing for anyone reading individual entries.
        return RuntimeReport(
            runtime="Stata",
            status="blocked",
            detail="Stata not found. Stata scripts will be refused.",
            advice=[
                "Install Stata from https://www.stata.com if you want to "
                "submit Stata scripts.",
                "Or submit in Python / R instead — Nora supports all three.",
            ],
        )
    return RuntimeReport(
        runtime="Stata",
        status="ok",
        detail=f"Stata at {st.binary} is healthy.",
    )


def _sandbox_report(env: Environment) -> RuntimeReport:
    if env.sandbox_exec is None:
        return RuntimeReport(
            runtime="sandbox-exec",
            status="blocked",
            detail=(
                "sandbox-exec not available on this machine — Nora "
                "refuses to run scripts unsandboxed."
            ),
            advice=[
                "On macOS this binary lives at /usr/bin/sandbox-exec "
                "and is always present. If you're on Linux or Windows, "
                "submit_script is not supported in this Nora version.",
            ],
        )
    # Distinct failure shape: sandbox-exec is present but won't apply
    # a minimal profile (SBPL compiler broken, OS doesn't honour
    # sandbox_apply, nested-sandbox harness). Surfaced as its own row
    # because the advice differs from "install something" — there's
    # nothing for the researcher to install. ``_check_sandbox_baseline``
    # is cached so this doesn't re-probe per ``--doctor`` call.
    from nora.env_detect import sandbox_baseline_result
    baseline_ok, baseline_err = sandbox_baseline_result()
    if not baseline_ok:
        return RuntimeReport(
            runtime="sandbox-exec",
            status="blocked",
            detail=(
                f"sandbox-exec at {env.sandbox_exec} cannot apply a "
                f"minimal profile: {baseline_err}"
            ),
            advice=[
                "If you're running Nora inside another sandbox or "
                "container, exit it and re-launch on the host.",
                "If you're on a heavily customised macOS variant, "
                "verify that ``sandbox-exec -p '(version 1)(allow "
                "default)' /usr/bin/true`` exits zero from a normal "
                "terminal — Nora needs at least that much to function.",
            ],
        )
    return RuntimeReport(
        runtime="sandbox-exec",
        status="ok",
        detail=f"sandbox-exec at {env.sandbox_exec}.",
    )


def run_doctor(env: Environment | None = None) -> DoctorReport:
    """Build a health report for the current Nora environment.

    Pass ``env`` to skip detection (testing / pre-computed state); by
    default this calls ``detect_environment()`` directly so probe
    caches inside ``env_detect`` populate as a side effect — meaning
    a subsequent ``submit_script`` in the same Nora process reuses
    them without re-probing.

    ``blocked`` on the returned report means script execution will
    fail. The CLI maps this to exit code 1; the UI banner maps it
    to disabling the chat input.
    """
    env = env or detect_environment()
    runtimes = [
        _sandbox_report(env),
        _python_report(env),
        _r_report(env),
        _stata_report(env),
    ]
    # A "blocked" report blocks the corresponding language. Nora is
    # blocked overall when sandbox is blocked (no scripts can run at
    # all) or when EVERY language is blocked (no script in any
    # language can run). One language blocked + another working is a
    # warning the UI surfaces but doesn't gate on — the researcher
    # can still proceed in the working language.
    sandbox_blocked = any(
        r.runtime == "sandbox-exec" and r.status == "blocked"
        for r in runtimes
    )
    language_runtimes = [r for r in runtimes if r.runtime != "sandbox-exec"]
    all_languages_blocked = (
        all(r.status == "blocked" for r in language_runtimes)
        if language_runtimes else True
    )
    blocked = sandbox_blocked or all_languages_blocked

    rejected = [
        (path, stderr)
        for path, (ok, stderr) in python_sandbox_probe_results().items()
        if not ok
    ]
    return DoctorReport(
        runtimes=runtimes,
        rejected_python_candidates=rejected,
        blocked=blocked,
    )


# ---------------------------------------------------------------------------
# CLI rendering
# ---------------------------------------------------------------------------

_STATUS_GLYPHS = {"ok": "[ok]", "warning": "[warn]", "blocked": "[fail]"}


def render_report_text(report: DoctorReport) -> str:
    """Render the report as plain text suitable for terminal output.

    Used by ``nora --doctor``. The UI banner consumes the
    DoctorReport directly and renders its own DOM, so this function
    is the terminal-only path.
    """
    lines: list[str] = []
    lines.append("Nora environment check")
    lines.append("=" * 30)
    for r in report.runtimes:
        glyph = _STATUS_GLYPHS.get(r.status, r.status)
        lines.append(f"{glyph} {r.runtime}: {r.detail}")
        for tip in r.advice:
            lines.append(f"    -> {tip}")
    if report.rejected_python_candidates:
        lines.append("")
        lines.append("Python candidates rejected by the sandbox probe:")
        for path, stderr in report.rejected_python_candidates:
            tail = (stderr or "").strip().splitlines()
            summary = tail[-1] if tail else "(no stderr captured)"
            lines.append(f"  {path}")
            lines.append(f"    {summary}")
    lines.append("")
    if report.blocked:
        lines.append(
            "Status: BLOCKED — script execution will fail until the "
            "issues above are fixed."
        )
    else:
        lines.append(
            "Status: OK — Nora is ready to run scripts."
        )
    return "\n".join(lines) + "\n"


def main_cli() -> int:
    """Entry point for ``nora --doctor``. Returns an exit code so the
    caller (typically ``ui.main``) can ``sys.exit`` on it; tests can
    call this directly and assert on the return value without
    intercepting ``sys.exit``."""
    report = run_doctor()
    sys.stdout.write(render_report_text(report))
    sys.stdout.flush()
    return 1 if report.blocked else 0
