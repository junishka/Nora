"""Nora — executor for R / Stata / Python scripts.

This is the only place in Nora that actually spawns subprocesses
against the researcher's data. Its job is narrow:

1. Stage the researcher's script and the Nora runtime library in a
   scoped scratch directory under ``<cwd>/.nora/runs/<run_id>/``.
2. Invoke the right interpreter (``Rscript``, ``stata-mp``, or
   ``python3``) against the script, with ``NORA_RESULT_PATH``
   pointing at a file inside the scratch dir.
3. Wrap the invocation in ``sandbox-exec`` with a profile that denies
   network access. (Defense in depth — the runtime library is still
   the only sanctioned I/O surface inside the script.)
4. Capture stdout/stderr (the researcher's raw log), read the structured
   result from ``NORA_RESULT_PATH``, and return both.

What this executor deliberately does NOT do:
- Enforce the sanitizer's SDC rules. Output goes to the caller (which
  routes through ``sanitizer.sanitize``). The executor just runs and
  reads.
- Inspect the script's contents. The AST verifier is a P2 idea only for
  Python; for R / Stata, the runtime-library-only-I/O rule plus the
  sandbox are the structural guarantees.
- Clean up scratch dirs on success. Keeping them around is useful for
  the researcher auditing what ran. A future step-7 cleanup task can
  trim old runs.
"""

from __future__ import annotations

import os
import secrets
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any, Callable, Literal

from nora.env_detect import Environment, detect_environment


Language = Literal["R", "Stata", "Python"]


# Hard-required Python packages — without these the runtime library
# itself won't import. Reflects what ``nora/runtime/nora.py`` needs
# at module load (NOT what the helpers need to do their job; e.g.,
# ``from_lm`` needs statsmodels but the rest of the library works
# without it). The executor refuses Python runs only when one of
# these is missing.
_PYTHON_HARD_REQUIRED: frozenset[str] = frozenset({"pandas", "numpy"})

# Per-script wall-clock cap. 300s (5 min) is the working ceiling: a
# Stata panel build plus a small batch of two-way-FE ``reghdfe``
# regressions fits comfortably (the prior 120s default forced
# researchers to artificially split scripts to stay under the wall),
# while staying tight enough that a runaway loop fails fast. For
# scripts that need more, the workflow Nora encourages is "split:
# build and save the analysis panel first, then run regressions in
# batches against the saved file" — that pattern keeps each call
# well under the cap and makes failure modes localizable.
#
# Override via ``NORA_SCRIPT_TIMEOUT_SECONDS`` for the unusual case
# where 5 min isn't enough (large simulations, bootstraps with many
# replications) or where you want a tighter cap (CI smoke tests).
# Bad values fall back to the default rather than crashing the
# bridge — a malformed env var should never strand the user — but
# emit a warning so a typo (``5min``, ``300s``) isn't silent. Cap
# at 24 hours: a value of e.g. ``2147483647`` would be accepted by
# the prior parser, letting a runaway script hang the runner for
# years; the upper bound prevents that without limiting any
# realistic research workload (a 24h script is already in
# "should be a batch job" territory, not "interactive analysis").
_TIMEOUT_FLOOR_SECONDS = 1
_TIMEOUT_CEILING_SECONDS = 24 * 60 * 60  # 24 hours


def _resolve_default_timeout() -> int:
    raw = os.environ.get("NORA_SCRIPT_TIMEOUT_SECONDS", "").strip()
    if not raw:
        return 300
    try:
        v = int(raw)
    except ValueError:
        import logging
        logging.getLogger("nora.executor").warning(
            "NORA_SCRIPT_TIMEOUT_SECONDS=%r is not an integer; "
            "falling back to default 300s", raw,
        )
        return 300
    if v < _TIMEOUT_FLOOR_SECONDS:
        import logging
        logging.getLogger("nora.executor").warning(
            "NORA_SCRIPT_TIMEOUT_SECONDS=%d is below floor %ds; "
            "falling back to default 300s", v, _TIMEOUT_FLOOR_SECONDS,
        )
        return 300
    if v > _TIMEOUT_CEILING_SECONDS:
        import logging
        logging.getLogger("nora.executor").warning(
            "NORA_SCRIPT_TIMEOUT_SECONDS=%d exceeds ceiling %ds (24h); "
            "clamping. A runaway script shouldn't hang the runner for "
            "longer than that.", v, _TIMEOUT_CEILING_SECONDS,
        )
        return _TIMEOUT_CEILING_SECONDS
    return v


DEFAULT_TIMEOUT_SECONDS = _resolve_default_timeout()

# Where per-run scratch dirs live, relative to cwd.
RUNS_SUBDIR = ".nora/runs"

# Name of the payload field the runtime library embeds to prove the
# payload came from the library (rather than being hand-crafted by
# Claude's script). Starts with underscore so it doesn't collide with
# a future analysis-schema field name.
RESULT_TOKEN_FIELD = "_token"

# Environment variable name the runtime library reads the token from.
# The R library reads this at source time and then unsets it; Stata
# `.ado` files read it on each invocation (Stata doesn't cleanly
# support env-unset from within the running process).
RUN_TOKEN_ENV_VAR = "NORA_RUN_TOKEN"


# Env vars we pass through to the R / Stata subprocess. Everything
# NOT in this set is stripped when we build ``subprocess_env``.
#
# Why an allowlist, not an ``os.environ`` inheritance: Claude-authored
# scripts can call ``Sys.getenv()`` (R) or read shell variables
# (Stata) and stuff the results into any allowed numeric / string
# field that reaches the sanitizer. If the parent process carries
# ``ANTHROPIC_API_KEY`` (it does — the SDK uses it to authenticate),
# AWS credentials, or any other secret, a prompt-injected script can
# exfiltrate them through e.g. a coefficient dict whose keys are
# "leak_bit_0", "leak_bit_1", … survived precisely by the
# dict_numeric sanitizer rule. The filesystem/network sandbox
# doesn't stop this — the bytes never leave the process boundary
# sandbox-exec protects.
#
# The explicit allowlist is PATH (so the interpreter can find
# system tools), HOME (R and Stata read config from here), LANG /
# LC_* (locale — determines number/date formatting), TMPDIR (R and
# Stata write scratch files here), USER / LOGNAME (some R packages
# read them), and SHELL / TERM (completeness; harmless). Everything
# else — including API keys, AWS creds, OpenAI tokens, whatever the
# researcher has in their shell — is dropped.
_SUBPROCESS_ENV_ALLOWLIST: frozenset[str] = frozenset({
    "PATH",
    "HOME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "LC_COLLATE",
    "LC_MESSAGES",
    "LC_MONETARY",
    "LC_NUMERIC",
    "LC_TIME",
    "TMPDIR",
    "TEMP",
    "TMP",
    "USER",
    "LOGNAME",
    "SHELL",
    "TERM",
    # macOS Homebrew Python / R sometimes need this to locate
    # dylibs; Stata needs STATATMP. Low-risk to pass through.
    "DYLD_FALLBACK_LIBRARY_PATH",
    "STATATMP",
    # R-specific: respect the researcher's existing R library
    # paths so Nora doesn't force-reinstall packages they
    # already have.
    "R_LIBS",
    "R_LIBS_USER",
    "R_LIBS_SITE",
})


def _filter_env(parent_env: dict[str, str]) -> dict[str, str]:
    """Return only the entries of ``parent_env`` on the allowlist.

    Kept as a separate function so tests can assert that secrets
    don't leak through, and so a future audit can point at one
    place for "what Claude's scripts can see from the shell".
    """
    return {k: v for k, v in parent_env.items() if k in _SUBPROCESS_ENV_ALLOWLIST}


def _generate_run_token() -> str:
    """Return a random per-run token embedded in every valid payload.

    256 bits via `secrets.token_hex(32)`. The runtime library copies it
    into every emitted payload under `_token`; the executor validates
    and strips it before passing the payload on. A malicious script
    that bypasses the runtime library (writing hand-crafted JSON
    straight to `NORA_RESULT_PATH`) has to either guess a fresh
    256-bit token or introspect the interpreter's loaded environment
    to recover it — the former is infeasible, the latter raises
    attacker cost meaningfully without being an absolute guarantee.
    See ``docs/direction.md`` "runtime-library contract" for the full
    threat-model discussion.
    """
    return secrets.token_hex(32)


def _runtime_call_hint(language: "Language") -> str:
    """Per-language hint for "your script didn't emit a structured
    result" errors. The fallback strings used to be a binary R-vs-
    Stata branch from before Python was added; without this helper
    a Python script that exits without a ``nora.*`` call gets told
    to call ``nora_result_regress in Stata``, which is unhelpful."""
    if language == "R":
        return "nora$result(...) or nora$from_lm(...) in R"
    if language == "Stata":
        return "nora_result_regress in Stata"
    return "nora.result(...) or nora.from_lm(...) in Python"


def _validate_and_strip_token(
    payload: dict[str, Any], expected_token: str
) -> tuple[dict[str, Any] | None, str | None]:
    """Check ``payload[RESULT_TOKEN_FIELD]`` matches ``expected_token``.

    Returns ``(cleaned_payload, None)`` on success or ``(None, error)``
    on mismatch / missing. The cleaned payload has the token field
    removed so downstream consumers (sanitizer, Claude) never see it.
    """
    if not isinstance(payload, dict):
        return None, (
            f"runtime-library payload must be a JSON object, got "
            f"{type(payload).__name__}"
        )
    got = payload.get(RESULT_TOKEN_FIELD)
    if got is None:
        return None, (
            f"runtime-library payload missing {RESULT_TOKEN_FIELD!r} "
            f"authenticity field — either the script bypassed the "
            f"Nora runtime library (writing JSON directly) or is "
            f"using a library version older than this executor. The "
            f"payload is rejected."
        )
    if not isinstance(got, str) or not secrets.compare_digest(
        got, expected_token
    ):
        return None, (
            f"runtime-library payload {RESULT_TOKEN_FIELD!r} did not "
            f"match the per-run token — the payload may have been "
            f"hand-crafted to bypass the runtime library. Rejected."
        )
    cleaned = {k: v for k, v in payload.items() if k != RESULT_TOKEN_FIELD}
    return cleaned, None


def _parse_result_jsonl(
    text: str, run_token: str
) -> tuple[list[dict[str, Any]], list[str]]:
    """Parse a JSONL result file line by line, skipping bad lines.

    A single corrupt line (e.g. a degenerate Stata fit that emitted a
    missing-value marker in ``f_statistic``, or a payload that fails
    authenticity-token validation) used to shadow every later valid
    line in the same batch — the parser would ``break`` and lose the
    rest. The current contract is "skip the bad line, keep going", so
    1 of 8 corrupt lines becomes "7 results + 1 documented error", not
    "0 results + 1 documented error".

    Returns ``(payloads, bad_line_messages)``. ``payloads`` carries
    every line that parsed AND token-validated, in emission order;
    ``bad_line_messages`` carries one short string per failed line for
    the caller to surface back to the model.
    """
    import json

    payloads: list[dict[str, Any]] = []
    bad_lines: list[str] = []
    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            raw_payload = json.loads(line)
        except json.JSONDecodeError as je:
            bad_lines.append(f"line {lineno}: {je.msg} at col {je.colno}")
            continue
        cleaned, auth_err = _validate_and_strip_token(raw_payload, run_token)
        if auth_err is not None:
            bad_lines.append(f"line {lineno}: {auth_err}")
            continue
        payloads.append(cleaned)
    return payloads, bad_lines


@lru_cache(maxsize=1)
def _cached_environment() -> Environment:
    """Return a process-local cached runtime probe.

    ``detect_environment()`` spawns multiple subprocesses (R package
    probe, Python package probe, prefix detection), which is fine at
    app startup but expensive to repeat on every ``submit_script``.
    Caching here keeps back-to-back regressions from paying that fixed
    tax every time.

    Callers with an already-known environment can still pass ``env=``
    to ``run_script`` and bypass this cache entirely.
    """
    return detect_environment()


def clear_environment_cache() -> None:
    """Drop the cached environment probe.

    Test hook today; also useful for a future explicit "refresh local
    runtimes" UI action if Nora grows one.
    """
    _cached_environment.cache_clear()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class ExecutionResult:
    """Outcome of running one script.

    - ``ok`` is ``True`` iff the subprocess exited 0 AND at least one
      valid JSON payload was written to the result file. Fatal
      deviations (timeout, non-zero exit, missing result file, empty
      result file, every line malformed) flip ``ok=False`` and fill
      ``error``.
    - ``warnings`` carries non-fatal advisories about the run — most
      commonly malformed JSONL lines that were skipped while OTHER
      lines parsed and validated cleanly. A run with 7 valid payloads
      and 1 malformed line stays ``ok=True`` and the malformed-line
      summary lands here, not in ``error``: the legitimate output
      should reach the caller without being demoted to
      "execution_failed". When NO payloads survive, the malformed-line
      message is upgraded into ``error`` instead, since it is then the
      only signal the caller has.
    - ``raw_stdout`` / ``raw_stderr`` are what the researcher sees in the
      TUI. Never routed to the sanitizer.
    - ``result_payloads`` is the list of parsed JSON payloads from the
      result file, in emission order, intended for ``sanitize()``. Still
      raw, no SDC rules applied here. The result file is JSONL: one
      object per line. A single-helper script produces one entry; a
      script that calls multiple ``nora_result_*`` helpers produces one
      entry per call. Empty list means no payload was emitted, which
      is treated as a script error in the success path.
    - ``run_dir`` and ``script_path`` are kept around for audit.
    - ``duration_seconds`` is wall-clock time inside the subprocess.
    """
    ok: bool
    language: Language
    raw_stdout: str
    raw_stderr: str
    exit_code: int | None
    result_payloads: list[dict]
    error: str | None
    run_dir: Path
    script_path: Path | None
    duration_seconds: float
    warnings: list[str] = field(default_factory=list)


def run_script(
    language: Language,
    code: str,
    cwd: Path,
    *,
    env: Environment | None = None,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    proc_register: "Callable[[subprocess.Popen[str]], None] | None" = None,
) -> ExecutionResult:
    """Stage + run + capture the researcher's script.

    Never raises for normal failure modes (interpreter missing, timeout,
    non-zero exit, bad result file). Those show up as ``ok=False`` with
    ``error`` populated so the calling tool can return a policy-shaped
    response to Claude. Programmer errors (invalid ``language``, etc.)
    still raise ``ValueError``.
    """
    env = env or _cached_environment()
    if language not in ("R", "Stata", "Python"):
        raise ValueError(
            f"unsupported language {language!r}; must be R, Stata, or Python"
        )

    # 1. Scratch dir.
    run_dir = _make_run_dir(cwd)
    result_path = run_dir / "result.json"
    script_path: Path | None = None

    # Preflight: sandbox required. On macOS `/usr/bin/sandbox-exec`
    # should always be present; if it isn't (or we're not on macOS at
    # all), we refuse to run rather than falling through to an
    # unsandboxed subprocess. The sandbox profile is the enforcement
    # of the data-boundary story — without it, a malicious script has
    # unrestricted local file access and could smuggle file contents
    # out through result fields the sanitizer forwards.
    if env.sandbox_exec is None:
        return ExecutionResult(
            ok=False, language=language, raw_stdout="", raw_stderr="",
            exit_code=None, result_payloads=[],
            error=(
                "sandbox-exec not available on this system — Nora "
                "refuses to run scripts unsandboxed. On macOS this "
                "binary lives at /usr/bin/sandbox-exec and is always "
                "present; if you're on Linux or Windows, submit_script "
                "is not supported in this Nora version."
            ),
            run_dir=run_dir, script_path=None, duration_seconds=0.0,
        )

    # Preflight: interpreter present.
    if language == "R" and env.r is None:
        return ExecutionResult(
            ok=False, language=language, raw_stdout="", raw_stderr="",
            exit_code=None, result_payloads=[],
            error=(
                "Rscript not found on this machine. Install R from "
                "https://cran.r-project.org and re-launch Nora, or "
                "submit the script in Stata instead."
            ),
            run_dir=run_dir, script_path=None, duration_seconds=0.0,
        )
    if language == "Stata" and env.stata is None:
        return ExecutionResult(
            ok=False, language=language, raw_stdout="", raw_stderr="",
            exit_code=None, result_payloads=[],
            error=(
                "Stata not found on this machine. Install Stata or submit "
                "the script in R or Python instead."
            ),
            run_dir=run_dir, script_path=None, duration_seconds=0.0,
        )
    if language == "Python":
        if env.python is None:
            return ExecutionResult(
                ok=False, language=language, raw_stdout="", raw_stderr="",
                exit_code=None, result_payloads=[],
                error=(
                    "python3 not found on PATH. Install Python 3 (the "
                    "official installer from python.org or via Homebrew, "
                    "``brew install python``) and re-launch Nora, or "
                    "submit the script in R or Stata instead."
                ),
                run_dir=run_dir, script_path=None, duration_seconds=0.0,
            )
        # Hard-required packages: the runtime library imports them
        # (or its emit-time encoder does). Soft-recommended ones
        # (statsmodels, scipy) are only needed by specific helpers
        # — refuse only on the hard set so a researcher who only
        # uses descriptive helpers doesn't have to install OLS deps.
        hard_missing = sorted(set(env.python.missing_packages) & _PYTHON_HARD_REQUIRED)
        if hard_missing:
            return ExecutionResult(
                ok=False, language=language, raw_stdout="", raw_stderr="",
                exit_code=None, result_payloads=[],
                error=(
                    "Python is installed at "
                    f"{env.python.binary} but the Nora runtime needs "
                    f"these packages: {', '.join(hard_missing)}. "
                    f"Install them with "
                    f"``{env.python.binary} -m pip install {' '.join(hard_missing)}`` "
                    f"and re-launch Nora."
                ),
                run_dir=run_dir, script_path=None, duration_seconds=0.0,
            )

    # 2. Stage runtime + script.
    lib_dir = _stage_runtime(run_dir, language)
    script_path = _write_script(run_dir, language, code)

    # 3. Compose the command + sandbox profile.
    extra_read_paths: tuple[str, ...] = ()
    if language == "R":
        cmd = _r_command(env.r.binary, lib_dir, script_path)  # type: ignore[union-attr]
    elif language == "Stata":
        cmd = _stata_command(env.stata.binary, lib_dir, script_path)  # type: ignore[union-attr]
    else:  # Python
        cmd = _python_command(env.python.binary, script_path)  # type: ignore[union-attr]
        # Allow the interpreter to read its own stdlib + site-packages
        # — critical for venv / pyenv / conda Pythons that live
        # outside the system trees the default sandbox already covers.
        extra_read_paths = env.python.extra_read_paths  # type: ignore[union-attr]

    # sandbox_exec presence is enforced above as a precondition.
    profile_path = _write_sandbox_profile(
        run_dir, cwd, extra_read_paths=extra_read_paths
    )
    cmd = [env.sandbox_exec, "-f", str(profile_path), *cmd]

    # 4. Run.
    # Subprocess cwd is language-specific:
    #   - R: the researcher's project dir, so `read.csv("survey.csv")`
    #     resolves naturally against relative paths.
    #   - Stata: the scratch dir, so Stata's batch-mode .log file lands
    #     there instead of in the researcher's project. The Stata wrapper
    #     `cd`s to the researcher's cwd before running the user's code,
    #     so relative paths in the user script still work.
    #   - Python: the researcher's project dir (same as R) — relative
    #     paths in ``pd.read_csv("survey.csv")`` resolve naturally.
    subprocess_cwd = run_dir if language == "Stata" else cwd
    # Generate a fresh per-run token. The runtime library reads it from
    # NORA_RUN_TOKEN, embeds it in every emitted payload, and (in R)
    # unsets the env var so user code loaded afterward can't read it
    # directly. See ``_validate_and_strip_token`` below.
    run_token = _generate_run_token()
    # Build the subprocess env from an explicit allowlist, not
    # ``{**os.environ}``. See _SUBPROCESS_ENV_ALLOWLIST above for
    # the rationale. Nora-specific vars are set last so a
    # pathological entry in the parent env can't shadow them.
    subprocess_env = {
        **_filter_env(dict(os.environ)),
        "NORA_RESULT_PATH": str(result_path),
        "NORA_LIB_DIR": str(lib_dir),
        "NORA_CWD": str(cwd),
        RUN_TOKEN_ENV_VAR: run_token,
    }

    start = time.monotonic()
    # Popen + communicate (instead of subprocess.run) so the async
    # caller can register the proc handle and ``proc.kill()`` it
    # when the asyncio task is cancelled. Without this, pressing
    # Stop while a long Stata regression / R fit / Python pipeline
    # is mid-run only cancels the Python coroutine; the subprocess
    # keeps running to completion (or to ``timeout_seconds``). From
    # the researcher's seat that looks identical to "Stop did
    # nothing".
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(subprocess_cwd),
            env=subprocess_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError as e:
        return ExecutionResult(
            ok=False, language=language, raw_stdout="", raw_stderr="",
            exit_code=None, result_payloads=[],
            error=f"interpreter not found: {e}",
            run_dir=run_dir, script_path=script_path, duration_seconds=0.0,
        )
    if proc_register is not None:
        try:
            proc_register(proc)
        except Exception:  # noqa: BLE001 — register is advisory, never fatal
            pass

    try:
        stdout, stderr = proc.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        # Same recovery as ``subprocess.run`` does: kill, drain,
        # report partial output. Without the second communicate(),
        # the .stdout/.stderr buffers stay attached to the killed
        # proc and the file descriptors leak into the run dir's
        # parent process.
        proc.kill()
        try:
            stdout, stderr = proc.communicate(timeout=2)
        except subprocess.TimeoutExpired:
            stdout, stderr = "", ""
        duration = time.monotonic() - start
        return ExecutionResult(
            ok=False, language=language,
            raw_stdout=stdout or "",
            raw_stderr=stderr or "",
            exit_code=None, result_payloads=[],
            error=f"script timed out after {timeout_seconds}s",
            run_dir=run_dir, script_path=script_path,
            duration_seconds=duration,
        )

    duration = time.monotonic() - start
    exit_code = proc.returncode

    # 5. Collect output. For Stata batch mode, real output lives in a
    # .log file next to the .do script rather than stdout.
    raw_stdout = stdout or ""
    raw_stderr = stderr or ""
    if language == "Stata":
        log_contents = _read_stata_log(script_path)
        if log_contents:
            raw_stdout = log_contents + (("\n" + raw_stdout) if raw_stdout else "")

    # Persist the raw subprocess output to the run dir so the researcher
    # TUI (and, if needed, later audit) can display what R / Stata / Python
    # actually said. The raw .log files NEVER cross to the model; there
    # is no file-read tool. The model can see a *short debug excerpt* on
    # script failure (see ``error_summary.extract_debug_excerpt``), which
    # is anchored on each language's error idiom, capped at 1 KB, and
    # passes through credential scrub plus path normalisation plus
    # dumpy-blob truncation. See ``test_error_summary_no_leak.py`` for
    # the SDC boundary regressions, and ``test_stderr_isolation.py`` for
    # the broader "no raw log file ever crosses" pin.
    try:
        (run_dir / "stdout.log").write_text(raw_stdout, encoding="utf-8")
        (run_dir / "stderr.log").write_text(raw_stderr, encoding="utf-8")
    except OSError:
        # Persistence failure isn't fatal — raw output still lives in the
        # ExecutionResult fields for in-process rendering.
        pass

    # 6. Parse result file. The runtime libraries write JSONL (one
    # payload per line). A script that calls a single helper produces
    # one line; one that calls N helpers produces N lines, in
    # emission order. Each line is independently token-validated.
    #
    # Partial-success surface: when a script aborts mid-loop after
    # emitting some helpers, we KEEP the payloads that parsed and
    # validated cleanly. The caller decides what to do with them
    # (submit_script returns them under status="execution_failed_partial").
    # Without this, a script doing 24 specs that hit a thin cell on
    # iteration #5 would lose the four good results — the same loss
    # mode multi-result was meant to fix.
    payloads: list[dict] = []
    error: str | None = None
    warnings: list[str] = []
    if not result_path.exists():
        error = (
            "script finished but did not emit a structured result — no file "
            f"was written to {result_path.name}. Make sure your script "
            f"calls the Nora runtime library ({_runtime_call_hint(language)})."
        )
    else:
        try:
            text = result_path.read_text(encoding="utf-8")
        except OSError as oe:
            error = f"could not read result file: {oe}"
        else:
            payloads, bad_lines = _parse_result_jsonl(text, run_token)
            if bad_lines:
                bad_msg = (
                    f"{len(bad_lines)} malformed result line(s) "
                    f"skipped ({len(payloads)} valid preserved): "
                    + "; ".join(bad_lines[:5])
                    + (" …" if len(bad_lines) > 5 else "")
                )
                # Two paths, two meanings:
                #   - Some payloads survived alongside bad lines: the
                #     bad lines are an advisory, not a failure. A
                #     24-spec script with one runtime-library glitch
                #     on spec #5 should stay ``ok=True`` and surface
                #     the 23 good results — the prior "any bad line
                #     ⇒ ok=False" behavior demoted these to
                #     "execution_failed_partial", which reads to the
                #     model as "the script aborted" even when the
                #     subprocess exited 0.
                #   - No payloads survived: the bad-line message IS
                #     the only signal we have (this is the bypass-
                #     attempt case — a forged JSON line that fails
                #     the auth-token check produces zero survivors).
                #     Keep it in ``error`` so the caller sees a
                #     fatal-shaped response and the security signal
                #     isn't buried under a non-blocking warning.
                if payloads:
                    warnings.append(bad_msg)
                else:
                    error = bad_msg
            if error is None and not payloads:
                error = (
                    "script finished but emitted an empty result file. "
                    "Make sure your script calls the Nora runtime "
                    f"library ({_runtime_call_hint(language)})."
                )

    ok = (exit_code == 0) and bool(payloads) and (error is None)
    if not ok and error is None:
        error = f"interpreter exited with non-zero code {exit_code}"

    return ExecutionResult(
        ok=ok, language=language,
        raw_stdout=raw_stdout, raw_stderr=raw_stderr,
        exit_code=exit_code, result_payloads=payloads,
        error=error,
        run_dir=run_dir, script_path=script_path,
        duration_seconds=duration,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Staging
# ---------------------------------------------------------------------------

def _make_run_dir(cwd: Path) -> Path:
    """Create a fresh per-run directory under <cwd>/.nora/runs/."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = uuid.uuid4().hex[:8]
    p = cwd / RUNS_SUBDIR / f"{timestamp}_{run_id}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _stage_runtime(run_dir: Path, language: Language) -> Path:
    """Copy the Nora runtime library for `language` into ``run_dir/lib``.

    Copying (rather than referencing) means the script sees a stable
    on-disk path regardless of whether Nora was installed as a wheel,
    frozen by PyInstaller, or run from source — and simplifies the
    Stata adopath case.
    """
    lib_dir = run_dir / "lib"
    lib_dir.mkdir(exist_ok=True)
    runtime_pkg = resources.files("nora.runtime")
    if language == "R":
        for name in ("nora.R",):
            src = runtime_pkg.joinpath(name)
            (lib_dir / name).write_text(src.read_text(encoding="utf-8"))
    elif language == "Stata":
        stata_ados = (
            "_nora_export_plot.ado",
            "nora_result_regress.ado",
            "nora_result_ttest.ado",
            "nora_ttest.ado",
            "nora_result_sum.ado",
            "nora_result_tab.ado",
            "nora_result_magnitude.ado",
            "nora_plot_residuals.ado",
            "nora_plot_coefficients.ado",
            "nora_plot_interaction.ado",
            "nora_plot_estimate_comparison.ado",
            "nora_safe_export.ado",
        )
        for name in stata_ados:
            src = runtime_pkg.joinpath(name)
            (lib_dir / name).write_text(src.read_text(encoding="utf-8"))
    else:  # Python
        # Single module — staged into lib_dir which the script's
        # subprocess sees on PYTHONPATH (set in subprocess_env).
        # Researchers ``import nora`` to reach the helpers exactly
        # the way the R library does ``nora$from_lm`` / Stata does
        # ``nora_result_regress``.
        src = runtime_pkg.joinpath("nora.py")
        (lib_dir / "nora.py").write_text(src.read_text(encoding="utf-8"))
    return lib_dir


def _write_script(run_dir: Path, language: Language, code: str) -> Path:
    """Persist Claude's code to disk.

    For Stata, prepends a small preamble that adds the Nora runtime
    library to the adopath and `cd`s into the researcher's working
    directory (so `use "data.dta"` in the user's code resolves against
    their project, not the scratch dir). The researcher's code and the
    preamble live in a single `.do` file with a visible separator — we
    used to split preamble and user code across two files with an
    internal `do "<user_path>"`, but that path was absolute and broke
    when the researcher's cwd contained spaces: Stata's own parser for
    the `-b do <path>` command line tokenizes on spaces and the nested
    `do` inherited the same risk. Concatenating avoids both issues and
    keeps the scratch dir easy to audit.
    """
    if language == "R":
        path = run_dir / "script.R"
        path.write_text(code, encoding="utf-8")
        return path
    if language == "Python":
        # Two-line preamble puts the staged ``nora.py`` on
        # ``sys.path`` so the researcher's ``import nora`` resolves
        # cleanly. We use a preamble (rather than PYTHONPATH) because
        # the interpreter is invoked with ``-I`` (isolated mode),
        # which ignores all PYTHON* env vars by design — keeps a
        # researcher's stray ``PYTHONSTARTUP`` from running before
        # their script. Same shape as the Stata adopath preamble:
        # explicit, visible in the scratch dir, easy to audit.
        lib_dir = run_dir / "lib"
        preamble = (
            "import sys as _nora_sys\n"
            f"_nora_sys.path.insert(0, {str(lib_dir)!r})\n"
            "del _nora_sys\n"
            "# ----- Nora preamble above; researcher code below -----\n"
            "\n"
        )
        path = run_dir / "script.py"
        path.write_text(preamble + code + "\n", encoding="utf-8")
        return path
    # Stata: single .do file. Preamble + separator + researcher code.
    # Quoted paths in Stata's `adopath +` and `cd` handle spaces fine
    # at the language level — only the command-line `-b do <path>`
    # has the tokenization bug (see _stata_command).
    preamble = (
        "local lib : env NORA_LIB_DIR\n"
        "adopath + \"`lib'\"\n"
        "local nora_cwd : env NORA_CWD\n"
        "cd \"`nora_cwd'\"\n"
        "\n"
        "*! ----- Nora preamble above; researcher code below -----\n"
        "\n"
    )
    script_path = run_dir / "script.do"
    script_path.write_text(preamble + code + "\n", encoding="utf-8")
    return script_path


# ---------------------------------------------------------------------------
# Command composition
# ---------------------------------------------------------------------------

def _python_command(python: str, script_path: Path) -> list[str]:
    """Compose a ``python3`` invocation for the user's script.

    ``-I`` (isolated mode) cuts the per-user site-packages dir and
    ``PYTHONSTARTUP`` out of the picture so the script runs against
    the interpreter's stdlib + the Nora-staged runtime + whatever's
    on ``PYTHONPATH`` (which the executor sets to ``lib_dir`` plus
    inherited paths). No ``site.USER_BASE`` reads, no surprise
    pre-script hooks.
    """
    return [python, "-I", str(script_path)]


def _r_command(rscript: str, lib_dir: Path, script_path: Path) -> list[str]:
    """Compose `Rscript` invocation that sources the runtime before the script.

    Using ``-e`` keeps the wrapper logic transparent: researchers reading
    the scratch dir see their code unchanged, and the Nora preamble
    is explicit in the process args.
    """
    source_lib = (lib_dir / "nora.R").as_posix()
    source_script = script_path.as_posix()
    bootstrap = (
        f'source({_r_quote(source_lib)}); '
        f'source({_r_quote(source_script)})'
    )
    return [rscript, "--vanilla", "-e", bootstrap]


def _r_quote(s: str) -> str:
    """R string literal with double quotes, backslash-escaped."""
    escaped = s.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _stata_command(stata: str, lib_dir: Path, script_path: Path) -> list[str]:
    """Compose `stata-mp -b do` invocation.

    Stata's batch mode writes output to a .log file next to the .do
    script (not stdout), so the executor reads that log after the
    process exits.

    Path handling: we pass the script as a **bare filename** and rely
    on the subprocess cwd being the scratch dir. Absolute paths don't
    work here — Stata's batch-mode argument parser tokenizes
    ``-b do <path>`` on spaces, so a researcher whose project lives
    under ``~/IESE Dropbox/...`` would hit ``file /Users/bb/IESE.do
    not found`` even though the shell passed a perfectly-quoted
    argument. See ``run_script`` for where subprocess_cwd is set to
    run_dir for Stata.
    """
    del lib_dir  # the .do script itself prepends adopath with $NORA_LIB_DIR
    return [stata, "-b", "-q", "do", script_path.name]


def _read_stata_log(script_path: Path | None) -> str:
    """Return the Stata batch log contents, or '' if missing."""
    if script_path is None:
        return ""
    log_path = script_path.with_suffix(".log")
    if not log_path.exists():
        return ""
    try:
        return log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


# ---------------------------------------------------------------------------
# Sandbox profile
# ---------------------------------------------------------------------------

def _sandbox_profile_string(
    run_dir: Path, cwd: Path, home: Path | None = None,
    extra_read_paths: tuple[str, ...] = (),
) -> str:
    """Build the SBPL profile text. Pure function — no I/O.

    Split from ``_write_sandbox_profile`` so unit tests can inspect the
    generated profile without needing sandbox-exec to be callable in
    the test environment. See ``test_executor_profile.py`` for the
    invariants the profile must satisfy.

    ``extra_read_paths`` is for runtimes whose interpreter lives
    outside the system trees the default profile already covers
    (e.g. a venv'd Python). Each entry is added as a read-subpath.
    """
    return _build_profile(
        run_dir, cwd, home or Path.home(), extra_read_paths,
    )


def _write_sandbox_profile(
    run_dir: Path, cwd: Path,
    extra_read_paths: tuple[str, ...] = (),
) -> Path:
    """Write a per-run sandbox-exec profile and return its path.

    Uses a ``(deny default)`` posture with explicit allowlists, so that
    a script spawned under the profile can read:

    - System and library trees needed by R/Stata to load their own code
      (``/System``, ``/Library``, ``/usr``, ``/bin``, ``/opt``, etc.).
    - The user's R package library under ``~/Library/R`` and Stata's
      user config under ``~/Library/Application Support/Stata`` plus
      the conventional ``~/ado`` adopath.
    - The researcher's working directory ``cwd`` — where data files
      live — and the scratch ``run_dir`` for the current invocation.

    and write only to:

    - The scratch ``run_dir`` (where ``NORA_RESULT_PATH`` lives and
      where Stata drops its batch ``.log``).
    - System temp directories (``/tmp``, ``/private/tmp``,
      ``/private/var/folders``) — R and Stata both stage scratch files
      under ``$TMPDIR`` which is normally a subpath of one of these.
    - Console device files (``/dev/null``, ``/dev/tty``, a pty) that
      shells / subprocess plumbing write to.

    Everything else — including the rest of the user's home dir, so
    ``~/.ssh``, ``~/.aws``, ``~/.gnupg``, Keychains — is denied.
    Network is denied entirely.

    This profile is the load-bearing enforcement of the data-boundary
    story. Without it, a malicious script could read arbitrary files
    and smuggle their contents out through ``label`` / coefficient-name
    fields in the result payload; with it, the script can only see
    ``cwd``, and the runtime-library-only I/O convention plus the
    sanitizer allowlist become the only paths to Claude's context.

    Notes on specific SBPL details:

    - ``(literal "/")`` is required in addition to the child subpaths
      because some metadata reads land on ``/`` itself (``stat("/")``)
      and a subpath of a child dir does not cover the parent.
    - ``file-read-metadata`` is allowed globally: ``stat()`` on paths
      outside the read allowlist is still permitted, which R/Stata's
      library-loading code paths rely on when probing for files. Only
      reading *contents* (``file-read-data``, open-for-read) is
      restricted.
    - The ``/dev/ttys*`` regex covers pseudo-terminals that subprocess
      pipes may briefly touch.
    """
    profile = _build_profile(run_dir, cwd, Path.home(), extra_read_paths)
    path = run_dir / "sandbox.sb"
    path.write_text(profile)
    return path


def _build_profile(
    run_dir: Path, cwd: Path, home: Path,
    extra_read_paths: tuple[str, ...] = (),
) -> str:
    """Construct the SBPL text. Separated so it can be unit-tested
    without filesystem I/O (see ``_sandbox_profile_string``).
    """
    r_user_lib = home / "Library" / "R"
    stata_user_config = home / "Library" / "Application Support" / "Stata"
    stata_user_ado = home / "ado"

    def _quote(p: Path | str) -> str:
        """Quote a path as an SBPL string literal.

        SBPL string literals are double-quoted with ``\\`` / ``"``
        escaping. Paths that contain either are extremely rare on macOS
        but we escape defensively so the profile can't be broken (or
        widened) by an unusual cwd.
        """
        s = str(p)
        return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'

    # Per-run write scope: the run's scratch dir (result file, Stata
    # .log, staged script), the researcher's cwd (so scripts can
    # ``save "panel.dta", replace`` / ``saveRDS`` / ``df.to_csv``),
    # plus system temp dirs that R/Stata use.
    #
    # The researcher's cwd is the analysis workspace. Every session
    # gets its own dir under ``~/.nora-sessions/`` so a script writing
    # there cannot reach personal files. The data-boundary the
    # sandbox enforces is the *network deny* and the *read* allowlist
    # (so a script can't slurp ``/etc/passwd`` or POST data to a
    # remote host); writes within the user-authorized cwd are part of
    # the normal Stata / R / Python workflow.
    write_subpaths = [
        _quote(run_dir),
        _quote(cwd),
        _quote("/private/tmp"),
        _quote("/private/var/folders"),
        _quote("/tmp"),
    ]
    write_literals = [
        _quote("/dev/null"),
        _quote("/dev/dtracehelper"),
        _quote("/dev/tty"),
    ]

    # Narrow /private/etc reads to the specific config files R/Stata
    # actually need. The previous `(subpath "/private/etc")` re-opened
    # reads on /etc/passwd (user GECOS, home dirs, shells), /etc/group,
    # /etc/sudoers.d, etc. — all of which a malicious script could
    # exfiltrate character-by-character through sanitizer-allowed label
    # fields. This list is an empirical starting point; expand if a
    # future R/Stata version probes another config file at startup.
    read_literals = [
        _quote("/private/etc/hosts"),
        _quote("/private/etc/localtime"),
        _quote("/private/etc/resolv.conf"),
        _quote("/private/etc/protocols"),
        _quote("/private/etc/services"),
        _quote("/private/etc/nsswitch.conf"),
    ]

    # Per-run read scope: system trees needed by R/Stata to bootstrap
    # themselves, plus the researcher's cwd (data) and the runtime
    # staging dir.
    #
    # The top-level dirs are deliberately NOT allowed as whole
    # subtrees. `/private` in particular would re-open reads on
    # `/private/var/log`, `/private/var/backups`, and other sensitive
    # subpaths that R/Stata don't need. `/Library` similarly contains
    # `/Library/Keychains` and `/Library/LaunchDaemons`. Each child is
    # listed explicitly so the boundary is "researcher cwd + runtime
    # dirs + only the system subpaths the interpreter actually needs."
    #
    # Everything under $HOME *except* the explicit R/Stata user
    # subpaths is denied, which is the whole point of the boundary.
    read_subpaths = [
        _quote("/System"),
        # /Library — only the subtrees R / Stata / Rosetta reach into.
        _quote("/Library/Apple"),
        _quote("/Library/Application Support"),
        _quote("/Library/Caches"),
        _quote("/Library/Frameworks"),
        _quote("/Library/Managed Preferences"),
        _quote("/Library/Preferences"),
        # /usr — skip /usr/sbin (R/Stata don't use privileged tools).
        _quote("/usr/bin"),
        _quote("/usr/lib"),
        _quote("/usr/libexec"),
        _quote("/usr/local"),
        _quote("/usr/share"),
        _quote("/bin"),
        _quote("/sbin"),
        _quote("/opt"),
        _quote("/dev"),
        _quote("/Applications"),
        # /private — only the subtrees needed for POSIX config,
        # user/group resolution, timezone data, and $TMPDIR scratch.
        # Notably: /private/etc is NOT a subpath — specific config
        # files are allowed via read_literals above. See the comment
        # on that list for why.
        _quote("/private/tmp"),
        _quote("/private/var/db/dslocal"),
        _quote("/private/var/db/timezone"),
        _quote("/private/var/folders"),
        _quote(r_user_lib),
        _quote(stata_user_config),
        _quote(stata_user_ado),
        _quote(cwd),
        _quote(run_dir),
    ]
    # Per-language extras (typically a Python interpreter's sys.prefix
    # so it can read its own stdlib + site-packages). Only added when
    # the executor is running a language that needs them; absent for
    # R / Stata runs.
    for p in extra_read_paths:
        if p:
            read_subpaths.append(_quote(p))

    return (
        "(version 1)\n"
        "(deny default)\n"
        "\n"
        "; Network — load-bearing. Scripts cannot exfiltrate data off-box.\n"
        "(deny network*)\n"
        "\n"
        "; Process / IPC / signal operations R and Stata expect.\n"
        "(allow process*)\n"
        "(allow mach*)\n"
        "(allow iokit*)\n"
        "(allow sysctl*)\n"
        "(allow ipc-posix*)\n"
        "(allow signal)\n"
        "\n"
        "; stat() is allowed anywhere — only reading file *contents* is\n"
        "; restricted below. R/Stata probe many paths during startup.\n"
        "(allow file-read-metadata)\n"
        "\n"
        "; File-content reads: system trees, user R/Stata config, and\n"
        "; the researcher's cwd. Everything else (including the rest\n"
        "; of the home dir) is implicitly denied.\n"
        "(allow file-read*\n"
        "    (literal \"/\")\n"
        + "".join(f"    (literal {p})\n" for p in read_literals)
        + "".join(f"    (subpath {p})\n" for p in read_subpaths)
        + ")\n"
        "\n"
        "; File writes restricted to the run's scratch dir and temp\n"
        "; paths used by R/Stata for internal staging.\n"
        "(allow file-write*\n"
        + "".join(f"    (subpath {p})\n" for p in write_subpaths)
        + "".join(f"    (literal {p})\n" for p in write_literals)
        + "    (regex #\"^/dev/ttys[0-9]+$\"))\n"
    )
