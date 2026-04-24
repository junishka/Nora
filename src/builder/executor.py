"""Builder — executor for R / Stata scripts.

This is the only place in Builder that actually spawns subprocesses
against the researcher's data. Its job is narrow:

1. Stage the researcher's script and the Builder runtime library in a
   scoped scratch directory under ``<cwd>/.builder/runs/<run_id>/``.
2. Invoke the right interpreter (``Rscript`` or ``stata-mp``) against
   the script, with ``BUILDER_RESULT_PATH`` pointing at a file inside
   the scratch dir.
3. Wrap the invocation in ``sandbox-exec`` with a profile that denies
   network access. (Defense in depth — the runtime library is still
   the only sanctioned I/O surface inside the script.)
4. Capture stdout/stderr (the researcher's raw log), read the structured
   result from ``BUILDER_RESULT_PATH``, and return both.

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
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path
from typing import Any, Literal

from builder.env_detect import Environment, detect_environment


Language = Literal["R", "Stata"]

DEFAULT_TIMEOUT_SECONDS = 120

# Where per-run scratch dirs live, relative to cwd.
RUNS_SUBDIR = ".builder/runs"

# Name of the payload field the runtime library embeds to prove the
# payload came from the library (rather than being hand-crafted by
# Claude's script). Starts with underscore so it doesn't collide with
# a future analysis-schema field name.
RESULT_TOKEN_FIELD = "_token"

# Environment variable name the runtime library reads the token from.
# The R library reads this at source time and then unsets it; Stata
# `.ado` files read it on each invocation (Stata doesn't cleanly
# support env-unset from within the running process).
RUN_TOKEN_ENV_VAR = "BUILDER_RUN_TOKEN"


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
    # paths so Builder doesn't force-reinstall packages they
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
    straight to `BUILDER_RESULT_PATH`) has to either guess a fresh
    256-bit token or introspect the interpreter's loaded environment
    to recover it — the former is infeasible, the latter raises
    attacker cost meaningfully without being an absolute guarantee.
    See ``docs/direction.md`` "runtime-library contract" for the full
    threat-model discussion.
    """
    return secrets.token_hex(32)


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
            f"Builder runtime library (writing JSON directly) or is "
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class ExecutionResult:
    """Outcome of running one script.

    - ``ok`` is ``True`` iff the subprocess exited 0 AND a valid JSON
      payload was written to the result file. Any deviation (timeout,
      non-zero exit, missing result, invalid JSON) flips ``ok=False``
      and fills ``error``.
    - ``raw_stdout`` / ``raw_stderr`` are what the researcher sees in the
      TUI. Never routed to the sanitizer.
    - ``result_payload`` is the parsed JSON from the result file, intended
      for ``sanitize()``. Still raw — no SDC rules applied here.
    - ``run_dir`` and ``script_path`` are kept around for audit.
    - ``duration_seconds`` is wall-clock time inside the subprocess.
    """
    ok: bool
    language: Language
    raw_stdout: str
    raw_stderr: str
    exit_code: int | None
    result_payload: dict | None
    error: str | None
    run_dir: Path
    script_path: Path | None
    duration_seconds: float


def run_script(
    language: Language,
    code: str,
    cwd: Path,
    *,
    env: Environment | None = None,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> ExecutionResult:
    """Stage + run + capture the researcher's script.

    Never raises for normal failure modes (interpreter missing, timeout,
    non-zero exit, bad result file). Those show up as ``ok=False`` with
    ``error`` populated so the calling tool can return a policy-shaped
    response to Claude. Programmer errors (invalid ``language``, etc.)
    still raise ``ValueError``.
    """
    env = env or detect_environment()
    if language not in ("R", "Stata"):
        raise ValueError(f"unsupported language {language!r}; must be R or Stata")

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
            exit_code=None, result_payload=None,
            error=(
                "sandbox-exec not available on this system — Builder "
                "refuses to run scripts unsandboxed. On macOS this "
                "binary lives at /usr/bin/sandbox-exec and is always "
                "present; if you're on Linux or Windows, submit_script "
                "is not supported in this Builder version."
            ),
            run_dir=run_dir, script_path=None, duration_seconds=0.0,
        )

    # Preflight: interpreter present.
    if language == "R" and env.r is None:
        return ExecutionResult(
            ok=False, language=language, raw_stdout="", raw_stderr="",
            exit_code=None, result_payload=None,
            error=(
                "Rscript not found on this machine. Install R from "
                "https://cran.r-project.org and re-launch Builder, or "
                "submit the script in Stata instead."
            ),
            run_dir=run_dir, script_path=None, duration_seconds=0.0,
        )
    if language == "Stata" and env.stata is None:
        return ExecutionResult(
            ok=False, language=language, raw_stdout="", raw_stderr="",
            exit_code=None, result_payload=None,
            error=(
                "Stata not found on this machine. Install Stata or submit "
                "the script in R instead."
            ),
            run_dir=run_dir, script_path=None, duration_seconds=0.0,
        )

    # 2. Stage runtime + script.
    lib_dir = _stage_runtime(run_dir, language)
    script_path = _write_script(run_dir, language, code)

    # 3. Compose the command + sandbox profile.
    if language == "R":
        cmd = _r_command(env.r.binary, lib_dir, script_path)  # type: ignore[union-attr]
    else:
        cmd = _stata_command(env.stata.binary, lib_dir, script_path)  # type: ignore[union-attr]

    # sandbox_exec presence is enforced above as a precondition.
    profile_path = _write_sandbox_profile(run_dir, cwd)
    cmd = [env.sandbox_exec, "-f", str(profile_path), *cmd]

    # 4. Run.
    # Subprocess cwd is language-specific:
    #   - R: the researcher's project dir, so `read.csv("survey.csv")`
    #     resolves naturally against relative paths.
    #   - Stata: the scratch dir, so Stata's batch-mode .log file lands
    #     there instead of in the researcher's project. The Stata wrapper
    #     `cd`s to the researcher's cwd before running the user's code,
    #     so relative paths in the user script still work.
    subprocess_cwd = cwd if language == "R" else run_dir
    # Generate a fresh per-run token. The runtime library reads it from
    # BUILDER_RUN_TOKEN, embeds it in every emitted payload, and (in R)
    # unsets the env var so user code loaded afterward can't read it
    # directly. See ``_validate_and_strip_token`` below.
    run_token = _generate_run_token()
    # Build the subprocess env from an explicit allowlist, not
    # ``{**os.environ}``. See _SUBPROCESS_ENV_ALLOWLIST above for
    # the rationale. Builder-specific vars are set last so a
    # pathological entry in the parent env can't shadow them.
    subprocess_env = {
        **_filter_env(dict(os.environ)),
        "BUILDER_RESULT_PATH": str(result_path),
        "BUILDER_LIB_DIR": str(lib_dir),
        "BUILDER_CWD": str(cwd),
        RUN_TOKEN_ENV_VAR: run_token,
    }

    start = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(subprocess_cwd),
            env=subprocess_env,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as te:
        duration = time.monotonic() - start
        return ExecutionResult(
            ok=False, language=language,
            raw_stdout=(te.stdout or b"").decode("utf-8", errors="replace")
                if isinstance(te.stdout, bytes) else (te.stdout or ""),
            raw_stderr=(te.stderr or b"").decode("utf-8", errors="replace")
                if isinstance(te.stderr, bytes) else (te.stderr or ""),
            exit_code=None, result_payload=None,
            error=f"script timed out after {timeout_seconds}s",
            run_dir=run_dir, script_path=script_path,
            duration_seconds=duration,
        )
    except FileNotFoundError as e:
        return ExecutionResult(
            ok=False, language=language, raw_stdout="", raw_stderr="",
            exit_code=None, result_payload=None,
            error=f"interpreter not found: {e}",
            run_dir=run_dir, script_path=script_path, duration_seconds=0.0,
        )

    duration = time.monotonic() - start

    # 5. Collect output. For Stata batch mode, real output lives in a
    # .log file next to the .do script rather than stdout.
    raw_stdout = proc.stdout or ""
    raw_stderr = proc.stderr or ""
    if language == "Stata":
        log_contents = _read_stata_log(script_path)
        if log_contents:
            raw_stdout = log_contents + (("\n" + raw_stdout) if raw_stdout else "")

    # Persist the raw subprocess output to the run dir so the researcher
    # TUI (and, if needed, later audit) can display what R / Stata
    # actually said. These files are INTENTIONALLY outside the
    # sanitization boundary — only the researcher ever sees them; they
    # never flow back to the frontier model. See
    # ``test_stderr_isolation.py`` for the regression that locks that in.
    try:
        (run_dir / "stdout.log").write_text(raw_stdout, encoding="utf-8")
        (run_dir / "stderr.log").write_text(raw_stderr, encoding="utf-8")
    except OSError:
        # Persistence failure isn't fatal — raw output still lives in the
        # ExecutionResult fields for in-process rendering.
        pass

    # 6. Parse result file.
    payload: dict | None = None
    error: str | None = None
    if not result_path.exists():
        error = (
            "script finished but did not emit a structured result — no file "
            f"was written to {result_path.name}. Make sure your script "
            f"calls the Builder runtime library "
            f"({'builder$result(...) or builder$from_lm(...) in R' if language == 'R' else 'builder_result_regress in Stata'})."
        )
    else:
        import json
        try:
            raw_payload = json.loads(result_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as je:
            error = (
                f"script emitted a result file that is not valid JSON: "
                f"{je.msg} at line {je.lineno} col {je.colno}"
            )
        except OSError as oe:
            error = f"could not read result file: {oe}"
        else:
            # Authenticity check: payload must carry the per-run token
            # the runtime library embeds. A script that wrote JSON
            # directly to BUILDER_RESULT_PATH (bypassing the library)
            # has no token and gets rejected here. The token is
            # stripped from the payload before it flows on to the
            # sanitizer, so downstream consumers don't see it.
            cleaned, auth_err = _validate_and_strip_token(
                raw_payload, run_token
            )
            if auth_err is not None:
                error = auth_err
            else:
                payload = cleaned

    exit_code = proc.returncode
    ok = (exit_code == 0) and (payload is not None) and (error is None)
    if not ok and error is None:
        error = f"interpreter exited with non-zero code {exit_code}"

    return ExecutionResult(
        ok=ok, language=language,
        raw_stdout=raw_stdout, raw_stderr=raw_stderr,
        exit_code=exit_code, result_payload=payload,
        error=error,
        run_dir=run_dir, script_path=script_path,
        duration_seconds=duration,
    )


# ---------------------------------------------------------------------------
# Staging
# ---------------------------------------------------------------------------

def _make_run_dir(cwd: Path) -> Path:
    """Create a fresh per-run directory under <cwd>/.builder/runs/."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = uuid.uuid4().hex[:8]
    p = cwd / RUNS_SUBDIR / f"{timestamp}_{run_id}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _stage_runtime(run_dir: Path, language: Language) -> Path:
    """Copy the Builder runtime library for `language` into ``run_dir/lib``.

    Copying (rather than referencing) means the script sees a stable
    on-disk path regardless of whether Builder was installed as a wheel,
    frozen by PyInstaller, or run from source — and simplifies the
    Stata adopath case.
    """
    lib_dir = run_dir / "lib"
    lib_dir.mkdir(exist_ok=True)
    runtime_pkg = resources.files("builder.runtime")
    if language == "R":
        for name in ("builder.R",):
            src = runtime_pkg.joinpath(name)
            (lib_dir / name).write_text(src.read_text(encoding="utf-8"))
    else:  # Stata
        stata_ados = (
            "builder_result_regress.ado",
            "builder_result_ttest.ado",
            "builder_result_sum.ado",
            "builder_result_tab.ado",
            "builder_result_magnitude.ado",
        )
        for name in stata_ados:
            src = runtime_pkg.joinpath(name)
            (lib_dir / name).write_text(src.read_text(encoding="utf-8"))
    return lib_dir


def _write_script(run_dir: Path, language: Language, code: str) -> Path:
    """Persist Claude's code to disk.

    For Stata, prepends a small preamble that adds the Builder runtime
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
    # Stata: single .do file. Preamble + separator + researcher code.
    # Quoted paths in Stata's `adopath +` and `cd` handle spaces fine
    # at the language level — only the command-line `-b do <path>`
    # has the tokenization bug (see _stata_command).
    preamble = (
        "local lib : env BUILDER_LIB_DIR\n"
        "adopath + \"`lib'\"\n"
        "local builder_cwd : env BUILDER_CWD\n"
        "cd \"`builder_cwd'\"\n"
        "\n"
        "*! ----- Builder preamble above; researcher code below -----\n"
        "\n"
    )
    script_path = run_dir / "script.do"
    script_path.write_text(preamble + code + "\n", encoding="utf-8")
    return script_path


# ---------------------------------------------------------------------------
# Command composition
# ---------------------------------------------------------------------------

def _r_command(rscript: str, lib_dir: Path, script_path: Path) -> list[str]:
    """Compose `Rscript` invocation that sources the runtime before the script.

    Using ``-e`` keeps the wrapper logic transparent: researchers reading
    the scratch dir see their code unchanged, and the Builder preamble
    is explicit in the process args.
    """
    source_lib = (lib_dir / "builder.R").as_posix()
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
    del lib_dir  # the .do script itself prepends adopath with $BUILDER_LIB_DIR
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
    run_dir: Path, cwd: Path, home: Path | None = None
) -> str:
    """Build the SBPL profile text. Pure function — no I/O.

    Split from ``_write_sandbox_profile`` so unit tests can inspect the
    generated profile without needing sandbox-exec to be callable in
    the test environment. See ``test_executor_profile.py`` for the
    invariants the profile must satisfy.
    """
    return _build_profile(run_dir, cwd, home or Path.home())


def _write_sandbox_profile(run_dir: Path, cwd: Path) -> Path:
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

    - The scratch ``run_dir`` (where ``BUILDER_RESULT_PATH`` lives and
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
    profile = _build_profile(run_dir, cwd, Path.home())
    path = run_dir / "sandbox.sb"
    path.write_text(profile)
    return path


def _build_profile(run_dir: Path, cwd: Path, home: Path) -> str:
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
    # .log, staged script) plus system temp dirs that R/Stata use.
    write_subpaths = [
        _quote(run_dir),
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
