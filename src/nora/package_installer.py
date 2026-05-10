"""Install / remove / reinstall language packages on the researcher's
machine, out-of-band from ``submit_script``.

Why this lives separately from the script executor:

``submit_script`` runs every R / Python / Stata user script under a
``sandbox-exec`` profile that pins ``(deny default)`` and explicitly
``(deny network*)``. That profile is the enforcement of the No Raw
Access guarantee — a script can't phone home with row contents because
the sandbox refuses every outbound socket. Package install needs the
opposite: outbound network to CRAN / PyPI / SSC, plus write access to
the language's user-library dir, neither of which the script sandbox
permits. So the install path runs OUTSIDE that sandbox.

Privacy posture is preserved by what an install does, not by what the
sandbox blocks: an install fetches public package code from a canonical
registry and writes it under the researcher's own user library. It
does not read the data files. The researcher already trusts whatever
they ``library()`` / ``import``; this module changes who initiates the
fetch, not the threat model.

Hardening:

  * Package names are validated against ``[A-Za-z0-9._-]+`` before
    they reach a shell. That rejects spaces, quotes, slashes, semicolons,
    backticks, and pip's ``pkg[extra]`` / ``pkg==1.2.3`` shapes — install
    whatever's on the canonical registry, no version pinning here.
  * Repos / index URLs are hard-coded. There is no parameter the model
    can pass to redirect to a custom mirror.
  * R installs target ``Sys.getenv("R_LIBS_USER")`` (auto-created on
    first install). Python installs go to ``nora_python_pkg_dir()``
    via ``pip install --target`` — NOT ``--user``. ``--user`` writes
    to a path the executor's ``-I`` mode and the sandbox both refuse
    to read; ``--target`` writes to a Nora-managed dir we explicitly
    add to the script preamble's ``sys.path`` and the sandbox read
    allowlist. ``--target`` also works inside a venv (``--user`` is
    rejected when user site-packages are not visible). Stata uses
    ``ssc install``. None require sudo.
  * Subprocess args go through ``argv``-form ``subprocess.run`` —
    no ``shell=True``, no string interpolation into a shell command.
"""

from __future__ import annotations

import asyncio
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


# Package-name allowlist. The same character set works for CRAN, PyPI,
# and SSC names: letters, digits, dot, underscore, dash. Crucially the
# FIRST character must be alphanumeric — without that anchor, a leading
# ``-`` (e.g. ``-r``, ``-e``, ``--no-index``) or a bare ``.`` / ``..``
# matches the original ``[A-Za-z0-9._-]+`` and pip happily interprets
# the value as an option or a local-path install rather than a registry
# package name. The tightened anchor + the ``--`` end-of-options
# separator we splice into the pip argv close that escape jointly.
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_NAME_MAX_LEN = 80


# Where Nora installs Python packages. Must satisfy two constraints
# the prior ``pip install --user`` path violated:
#
#   1. The executor invokes Python with ``-I`` (isolated mode), which
#      drops the per-user site-packages dir from ``sys.path``. Anything
#      installed via ``--user`` is therefore invisible to script runs.
#      The fix is to install to a directory we explicitly add to the
#      preamble's ``sys.path`` (see ``executor._write_script``) so the
#      install location matches the import location.
#
#   2. The script subprocess runs under ``sandbox-exec`` with a
#      tightly scoped read allowlist. ``~/.local`` (the typical
#      ``--user`` target) is not on it, so even without ``-I`` the
#      sandbox would deny reads of user-installed packages. The
#      sandbox profile re-allows reads under ``nora_python_pkg_dir()``
#      explicitly.
#
# Per-version subdir is required because pip ``--target`` writes
# wheel-version-specific code (numpy / pandas C extensions are
# tagged for a specific CPython ABI). Mixing 3.11 and 3.12 wheels in
# the same directory crashes at import.
_NORA_PKG_BASE_ENV_VAR = "NORA_PYTHON_PKG_BASE"


def _python_version_tag(binary: str) -> str:
    """Return a ``"3.11"`` / ``"3.12"`` tag for the given interpreter,
    or ``"unknown"`` if the probe fails. Used to namespace the install
    directory so wheels with C extensions don't collide across
    interpreter versions."""
    try:
        out = subprocess.run(
            [
                binary, "-c",
                "import sys; print(f'{sys.version_info.major}."
                "{sys.version_info.minor}')",
            ],
            capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    if out.returncode != 0:
        return "unknown"
    tag = (out.stdout or "").strip()
    return tag if re.fullmatch(r"\d+\.\d+", tag) else "unknown"


def nora_python_pkg_dir(binary: str) -> Path:
    """Return the absolute path to Nora's Python package install
    directory for the interpreter at ``binary``.

    Single source of truth — used by the installer (write target),
    the executor's preamble (sys.path entry), the sandbox profile
    (read allowlist), and the env-detect import probe (PYTHONPATH).
    Drift between any of those would re-open the original bug.

    Honors ``$NORA_PYTHON_PKG_BASE`` for tests that need a temp dir.
    """
    base_env = os.environ.get(_NORA_PKG_BASE_ENV_VAR)
    base = Path(base_env) if base_env else Path.home() / ".nora-packages" / "python"
    return base / _python_version_tag(binary)

# Per-action wall-clock cap. Network installs of ~10 packages can be
# slow on a cold cache; 5 minutes is generous enough for most real
# cases without hanging the agent indefinitely.
_INSTALL_TIMEOUT_SECONDS = 300

Action = Literal["install", "remove", "reinstall"]
Language = Literal["R", "Python", "Stata"]
_VALID_ACTIONS: tuple[str, ...] = ("install", "remove", "reinstall")
_VALID_LANGUAGES: tuple[str, ...] = ("R", "Python", "Stata")


@dataclass(frozen=True)
class PackageStatus:
    """Per-package outcome; one entry per name the caller passed in."""

    name: str
    status: Literal["ok", "failed", "skipped"]
    detail: str = ""


@dataclass(frozen=True)
class InstallResult:
    language: Language
    action: Action
    statuses: tuple[PackageStatus, ...]
    raw_stdout: str
    raw_stderr: str
    duration_seconds: float
    error: str | None = None


def _validate_names(packages: list[str]) -> tuple[list[str], list[PackageStatus]]:
    """Split inputs into (valid, rejected). Rejected entries come back
    as ``status='failed'`` with a name-shape detail so the model can
    correct itself."""
    valid: list[str] = []
    rejected: list[PackageStatus] = []
    for raw in packages:
        if not isinstance(raw, str):
            rejected.append(PackageStatus(
                name=str(raw), status="failed",
                detail="package name must be a string",
            ))
            continue
        name = raw.strip()
        if not name or len(name) > _NAME_MAX_LEN or not _NAME_RE.match(name):
            rejected.append(PackageStatus(
                name=raw, status="failed",
                detail=(
                    "rejected: package name must match [A-Za-z0-9._-]+ "
                    f"and be 1–{_NAME_MAX_LEN} chars (no version pins, "
                    "no extras, no URLs)"
                ),
            ))
            continue
        valid.append(name)
    return valid, rejected


# ---------------------------------------------------------------------------
# R: install.packages / remove.packages on the user library
# ---------------------------------------------------------------------------

def _r_command(binary: str, packages: list[str], action: Action) -> list[str]:
    """Build an Rscript invocation for the requested action.

    Uses the canonical CRAN mirror. Targets ``R_LIBS_USER`` (R's
    standard user-library env var); creates the directory if missing
    so a fresh-install machine doesn't error out on first call.
    """
    quoted = ", ".join(f'"{p}"' for p in packages)
    if action == "remove":
        body = (
            f"pkgs <- c({quoted}); "
            "lib <- Sys.getenv('R_LIBS_USER'); "
            "remove.packages(pkgs, lib = if (nzchar(lib)) lib else NULL)"
        )
    else:
        # install / reinstall — install.packages overwrites by default,
        # so 'reinstall' uses the same call shape. We just remove first
        # for reinstall to guarantee a clean reinstall.
        prelude = ""
        if action == "reinstall":
            prelude = (
                f"pkgs <- c({quoted}); "
                "lib <- Sys.getenv('R_LIBS_USER'); "
                "try(remove.packages(pkgs, lib = if (nzchar(lib)) lib else NULL), "
                "silent = TRUE); "
            )
        body = (
            prelude
            + f"pkgs <- c({quoted}); "
            "lib <- Sys.getenv('R_LIBS_USER'); "
            "if (nzchar(lib) && !dir.exists(lib)) dir.create(lib, recursive = TRUE, showWarnings = FALSE); "
            "install.packages(pkgs, "
            "lib = if (nzchar(lib)) lib else NULL, "
            "repos = 'https://cloud.r-project.org')"
        )
    return [binary, "--vanilla", "-e", body]


# ---------------------------------------------------------------------------
# Python: pip install --target <nora_python_pkg_dir> / pip uninstall
# ---------------------------------------------------------------------------

def _python_command(binary: str, packages: list[str], action: Action) -> list[str]:
    """Build a ``python -m pip ...`` invocation that writes to (and
    reads from) ``nora_python_pkg_dir(binary)``.

    Why ``--target`` instead of ``--user``: see the docstring on
    ``nora_python_pkg_dir``. ``--user`` writes to a path the executor's
    ``-I`` mode and the sandbox both refuse, so the prior install path
    looked like it succeeded but every subsequent ``submit_script``
    failed with ``ModuleNotFoundError``. ``--target`` writes to a path
    we explicitly own and grant access to.

    Why the ``--`` end-of-options terminator before package names:
    even with the tightened ``_NAME_RE``, defense-in-depth — pip stops
    interpreting tokens as flags after ``--``, so a future relaxation
    of the validator (or a bug in it) cannot smuggle ``-r requirements``
    / ``-e .`` etc. through.
    """
    target = nora_python_pkg_dir(binary)
    target.mkdir(parents=True, exist_ok=True)
    if action == "remove":
        # ``pip uninstall`` has no ``--target``; it locates the
        # package via ``sys.path``. The caller passes
        # ``PYTHONPATH=<target>`` in ``env`` so pip can see what we
        # installed under ``--target``.
        return [
            binary, "-m", "pip", "uninstall", "-y", "--", *packages,
        ]
    # ``--upgrade`` so a re-install of an already-present package
    # picks up newer wheels rather than no-op'ing. ``reinstall`` adds
    # ``--force-reinstall --no-deps`` so the model's refresh request
    # does not yank the entire dep tree.
    base: list[str] = [
        binary, "-m", "pip", "install",
        "--target", str(target),
        "--upgrade",
    ]
    if action == "reinstall":
        base += ["--force-reinstall", "--no-deps"]
    return [*base, "--", *packages]


# ---------------------------------------------------------------------------
# Stata: ssc install / ado uninstall
# ---------------------------------------------------------------------------

def _stata_command(binary: str, packages: list[str], action: Action) -> list[str]:
    """Build a Stata batch invocation. ``ssc install`` is the canonical
    SSC package fetch; ``ado uninstall`` removes.

    Commands are joined with NEWLINES, not semicolons — Stata's default
    line delimiter inside a do-file is ``\\n``. The previous ``;``
    join produced a single line like
    ``capture ado uninstall foo; ssc install foo, replace`` which Stata
    parsed as one command with a stray semicolon, failing even on
    a single ``reinstall`` request. Switching the separator (rather
    than emitting ``#delimit ;`` + trailing-``;``) is simpler and
    matches what every Nora-emitted .ado file already does.
    """
    if action == "remove":
        lines = [f"ado uninstall {p}" for p in packages]
    elif action == "reinstall":
        lines = []
        for p in packages:
            lines.append(f"capture ado uninstall {p}")
            lines.append(f"ssc install {p}, replace")
    else:
        lines = [f"ssc install {p}, replace" for p in packages]
    cmds = "\n".join(lines) + "\n"
    return [binary, "-b", "-q", "do", "/dev/stdin"], cmds  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def install_packages(
    language: str,
    packages: list[str],
    action: str = "install",
) -> InstallResult:
    """Run install / remove / reinstall for a list of packages."""
    if action not in _VALID_ACTIONS:
        return InstallResult(
            language=language,  # type: ignore[arg-type]
            action="install", statuses=(), raw_stdout="", raw_stderr="",
            duration_seconds=0.0,
            error=f"unknown action {action!r}; valid: {', '.join(_VALID_ACTIONS)}",
        )
    if language not in _VALID_LANGUAGES:
        return InstallResult(
            language="R",  # placeholder; error path
            action=action,  # type: ignore[arg-type]
            statuses=(), raw_stdout="", raw_stderr="", duration_seconds=0.0,
            error=(
                f"unknown language {language!r}; valid: "
                f"{', '.join(_VALID_LANGUAGES)}"
            ),
        )
    if not isinstance(packages, list) or not packages:
        return InstallResult(
            language=language,  # type: ignore[arg-type]
            action=action,  # type: ignore[arg-type]
            statuses=(), raw_stdout="", raw_stderr="", duration_seconds=0.0,
            error="packages must be a non-empty list of names",
        )

    valid, rejected = _validate_names(packages)
    if not valid:
        return InstallResult(
            language=language,  # type: ignore[arg-type]
            action=action,  # type: ignore[arg-type]
            statuses=tuple(rejected),
            raw_stdout="", raw_stderr="", duration_seconds=0.0,
            error="all package names rejected by the name validator",
        )

    # Resolve the interpreter binary lazily — env_detect is an
    # already-imported sibling module, so the import is cheap.
    from nora.env_detect import detect_environment
    from nora.executor import _filter_env

    env = detect_environment()
    # Build the installer subprocess env from the same allowlist the
    # executor uses for analysis scripts. Package install hooks
    # (pip's setup.py / wheel build steps; R's ``configure``,
    # ``.onLoad`` post-install scripts; Stata's much rarer ado-side
    # post-install) execute with network access OUTSIDE the script
    # sandbox, so any ``ANTHROPIC_API_KEY`` / ``OPENAI_API_KEY`` /
    # AWS credential / arbitrary user secret in the parent process
    # env was previously visible to whatever code those hooks ran.
    # The executor solved this for analysis runs via the same
    # ``_filter_env`` allowlist; the installer was the missing
    # symmetric path. Without this filter, ``install_packages`` is
    # the highest-leverage exfiltration channel in the codebase
    # (network-permitted, no sandbox, attacker controls the package
    # contents).
    base_env = _filter_env(dict(os.environ))
    subprocess_env: dict[str, str] = base_env
    if language == "R":
        if env.r is None:
            return InstallResult(
                language="R", action=action,  # type: ignore[arg-type]
                statuses=tuple(rejected), raw_stdout="", raw_stderr="",
                duration_seconds=0.0,
                error="Rscript not found on this machine",
            )
        cmd = _r_command(env.r.binary, valid, action)  # type: ignore[arg-type]
        proc_stdin: str | None = None
    elif language == "Python":
        if env.python is None:
            return InstallResult(
                language="Python", action=action,  # type: ignore[arg-type]
                statuses=tuple(rejected), raw_stdout="", raw_stderr="",
                duration_seconds=0.0,
                error="python3 not found on this machine",
            )
        cmd = _python_command(env.python.binary, valid, action)  # type: ignore[arg-type]
        proc_stdin = None
        # ``pip uninstall`` has no ``--target``; route the Nora pkg
        # dir through ``PYTHONPATH`` so pip resolves the package via
        # ``sys.path`` and removes the right copy. Install commands
        # are unaffected by this (``--target`` is explicit on argv)
        # but it's harmless to pass — pip ignores PYTHONPATH for
        # ``--target`` installs.
        target_dir = nora_python_pkg_dir(env.python.binary)
        existing = base_env.get("PYTHONPATH", "")
        new_pp = (
            f"{target_dir}{os.pathsep}{existing}" if existing else str(target_dir)
        )
        subprocess_env = {**base_env, "PYTHONPATH": new_pp}
    else:  # Stata
        if env.stata is None:
            return InstallResult(
                language="Stata", action=action,  # type: ignore[arg-type]
                statuses=tuple(rejected), raw_stdout="", raw_stderr="",
                duration_seconds=0.0,
                error="Stata not found on this machine",
            )
        cmd_pair = _stata_command(env.stata.binary, valid, action)  # type: ignore[arg-type]
        cmd, proc_stdin = cmd_pair  # type: ignore[assignment]

    # Run the subprocess in a thread so we don't block the asyncio
    # loop. ``subprocess.run`` with capture_output is plenty here —
    # we don't need streaming output for a one-shot install.
    started = time.monotonic()
    try:
        completed = await asyncio.to_thread(
            subprocess.run,
            cmd,
            input=proc_stdin,
            capture_output=True,
            text=True,
            timeout=_INSTALL_TIMEOUT_SECONDS,
            env=subprocess_env,
        )
    except subprocess.TimeoutExpired as e:
        return InstallResult(
            language=language,  # type: ignore[arg-type]
            action=action,  # type: ignore[arg-type]
            statuses=tuple(rejected) + tuple(
                PackageStatus(name=p, status="failed", detail="install timed out")
                for p in valid
            ),
            raw_stdout=(e.stdout or "") if isinstance(e.stdout, str) else "",
            raw_stderr=(e.stderr or "") if isinstance(e.stderr, str) else "",
            duration_seconds=time.monotonic() - started,
            error=f"install timed out after {_INSTALL_TIMEOUT_SECONDS}s",
        )
    except OSError as e:
        return InstallResult(
            language=language,  # type: ignore[arg-type]
            action=action,  # type: ignore[arg-type]
            statuses=tuple(rejected),
            raw_stdout="", raw_stderr="",
            duration_seconds=time.monotonic() - started,
            error=f"could not launch installer: {e}",
        )

    duration = time.monotonic() - started
    statuses = list(rejected)
    # Per-package result: trust the exit code as the headline. For a
    # multi-package call this is conservative — if pip succeeds on 4
    # of 5, exit is non-zero and we mark all 5 as failed at the headline
    # level. The raw stdout/stderr lets the model dig deeper. Keeping
    # per-name parsing language-specific is brittle; the simple rule
    # is honest about uncertainty.
    if completed.returncode == 0:
        statuses.extend(
            PackageStatus(name=p, status="ok", detail="") for p in valid
        )
        err = None
    else:
        statuses.extend(
            PackageStatus(name=p, status="failed",
                          detail=f"exit {completed.returncode}")
            for p in valid
        )
        err = (
            f"installer exited {completed.returncode} — see raw_stderr "
            "for details"
        )

    # Invalidate the executor's cached environment probe whenever a
    # successful run mutated the package set. The probe records which
    # optional packages are present (haven, ggplot2, statsmodels, …);
    # after install / remove / reinstall that snapshot is stale, and
    # the next ``run_script`` would dispatch with the pre-install view
    # — most visibly producing the wrong runtime block in the system
    # prompt for whichever next turn rebuilds it from cache. The
    # uncached ``detect_environment()`` call sites (system_prompt, the
    # web UI status pane) already pick up the change on their own; the
    # cache invalidation aligns the executor with them. Skipped on
    # failure so a non-mutating error doesn't trigger an unnecessary
    # re-probe (each one spawns the language interpreter).
    if completed.returncode == 0:
        from nora.executor import clear_environment_cache
        clear_environment_cache()

    return InstallResult(
        language=language,  # type: ignore[arg-type]
        action=action,  # type: ignore[arg-type]
        statuses=tuple(statuses),
        raw_stdout=completed.stdout or "",
        raw_stderr=completed.stderr or "",
        duration_seconds=duration,
        error=err,
    )


def _is_valid_name(s: str) -> bool:
    """Public-ish predicate so callers (tests, future tools) can probe
    the name policy without invoking the installer."""
    return (
        isinstance(s, str)
        and 0 < len(s) <= _NAME_MAX_LEN
        and bool(_NAME_RE.match(s))
    )
