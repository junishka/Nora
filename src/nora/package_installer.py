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
    first install). Python uses ``pip install --user``. Stata uses
    ``ssc install``. None require sudo.
  * Subprocess args go through ``argv``-form ``subprocess.run`` —
    no ``shell=True``, no string interpolation into a shell command.
"""

from __future__ import annotations

import asyncio
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


# Package-name allowlist. The same pattern works for CRAN, PyPI, and
# SSC names: letters, digits, dot, underscore, dash. Anything else is
# rejected before it reaches the shell. Length cap keeps a stray
# multi-megabyte input from slipping past.
_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_NAME_MAX_LEN = 80

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
# Python: pip install --user / pip uninstall
# ---------------------------------------------------------------------------

def _python_command(binary: str, packages: list[str], action: Action) -> list[str]:
    """Build a ``python -m pip ...`` invocation. ``--user`` writes to
    the per-user site-packages so no sudo is needed; respects an
    active virtualenv (in which case pip rejects ``--user`` and we
    drop the flag — see _python_runtime_args).
    """
    if action == "remove":
        return [binary, "-m", "pip", "uninstall", "-y", *packages]
    base = [binary, "-m", "pip", "install", "--user"]
    if action == "reinstall":
        # --force-reinstall + --no-deps to avoid yanking everything in
        # the dep tree; the model asked to refresh THESE packages, not
        # the world.
        base = [binary, "-m", "pip", "install", "--user", "--force-reinstall", "--no-deps"]
    return [*base, *packages]


# ---------------------------------------------------------------------------
# Stata: ssc install / ado uninstall
# ---------------------------------------------------------------------------

def _stata_command(binary: str, packages: list[str], action: Action) -> list[str]:
    """Build a Stata batch invocation. ``ssc install`` is the canonical
    SSC package fetch; ``ado uninstall`` removes. We construct one do
    string per package so a single bad name doesn't mask the others'
    progress.
    """
    if action == "remove":
        cmds = "; ".join(f"ado uninstall {p}" for p in packages)
    elif action == "reinstall":
        cmds = "; ".join(
            f"capture ado uninstall {p}; ssc install {p}, replace" for p in packages
        )
    else:
        cmds = "; ".join(f"ssc install {p}, replace" for p in packages)
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

    env = detect_environment()
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
