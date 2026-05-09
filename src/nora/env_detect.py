"""Nora — runtime environment detection.

Finds the researcher's installed `Rscript` / `stata` / `python3`
binaries so the executor knows what to invoke. Checks `PATH` and
common macOS install locations. No fuzziness: either we find an
executable or we don't.

The result is consulted at app startup so the banner can honestly tell
the researcher what Nora will and won't be able to run for them.

Python detection also probes for the scientific stack the runtime
library needs (``pandas`` + ``statsmodels``). A bare ``python3``
without those packages can still be discovered — ``find_python``
records what's missing so the UI can tell the researcher what to
``pip install`` rather than dropping a cryptic ``ModuleNotFoundError``
into the chat after the first script.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


# Common macOS install locations for Stata. `stata` / `stata-mp` /
# `stata-se` on PATH is preferred because users configure that themselves;
# falling back to /Applications paths means we find it even when PATH
# isn't set up.
_STATA_APP_LOCATIONS: tuple[str, ...] = (
    "/Applications/Stata/StataMP.app/Contents/MacOS/stata-mp",
    "/Applications/Stata/StataSE.app/Contents/MacOS/stata-se",
    "/Applications/Stata/Stata.app/Contents/MacOS/stata",
    "/Applications/StataMP.app/Contents/MacOS/stata-mp",
    "/Applications/StataSE.app/Contents/MacOS/stata-se",
    "/Applications/Stata.app/Contents/MacOS/stata",
)


@dataclass(frozen=True)
class Tool:
    """A discovered statistical runtime."""
    name: str        # Human-readable, e.g. "R" or "Stata"
    binary: str      # Absolute path to the executable
    version: str | None = None
    # Optional: which packages the discovered interpreter is missing,
    # for runtimes (Python today) where having the binary isn't enough.
    # Empty tuple means "ready to go." None means "not checked yet."
    missing_packages: tuple[str, ...] = ()
    # Optional packages whose absence DOES NOT block runs but DOES
    # disable specific features. ``matplotlib`` is the canonical
    # case: scripts that don't plot run fine without it, but
    # ``nora.plot_*`` helpers fail silently on import. The executor
    # surfaces these in the missing-deps hint so a researcher who
    # wants plot vision knows exactly what to install.
    optional_missing_packages: tuple[str, ...] = ()
    # Extra filesystem subpaths the executor's sandbox profile should
    # allow reads from when this interpreter runs. For Python this is
    # ``sys.prefix`` (and ``sys.exec_prefix`` if different) — the
    # interpreter needs to read its own stdlib and site-packages,
    # which can live outside the system trees the default profile
    # already covers (venvs, pyenv installs, conda envs, …).
    extra_read_paths: tuple[str, ...] = ()


# R packages we probe at startup. ``haven`` is needed to read .dta
# files (Stata's native format) — without it, R can't open any of
# the user's Stata datasets and the model's first attempt to
# ``library(haven)`` fails. ``ggplot2`` is the most common plotting
# library; helpers fall back to base graphics, but raw model
# scripts often reach for it. None of these are HARD requirements
# — base R can still run analyses without them — but advertising
# their availability in the system prompt lets the model pick the
# right path on the first try instead of discovering missing-
# package errors by failing.
_R_OPTIONAL_PACKAGES: tuple[str, ...] = (
    "haven",
    "ggplot2",
)


def find_r() -> Tool | None:
    """Return the discovered R runtime, or None."""
    path = shutil.which("Rscript")
    if path is None:
        return None
    optional_missing = _r_missing_packages(path, _R_OPTIONAL_PACKAGES)
    return Tool(
        name="R", binary=path, version=_r_version(path),
        optional_missing_packages=optional_missing,
    )


def _r_missing_packages(
    rscript: str, packages: tuple[str, ...],
) -> tuple[str, ...]:
    """Probe an R installation for ``packages``. Returns the names
    that aren't installed. Done in a single ``Rscript`` invocation
    (one subprocess per package would inflate startup time on
    machines with slow R). The probe writes a single boolean per
    package to stdout, separated by spaces.

    Failures (Rscript missing, weird R version, OS error) return
    "all missing" rather than the conservative "none missing" so
    the system prompt is honest about uncertainty.
    """
    if not packages:
        return ()
    expr = "; ".join(
        f"cat(requireNamespace(\"{pkg}\", quietly=TRUE), \" \")"
        for pkg in packages
    )
    try:
        out = subprocess.run(
            [rscript, "-e", expr],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return tuple(packages)
    flags = out.stdout.strip().split()
    if len(flags) != len(packages):
        return tuple(packages)
    return tuple(
        pkg for pkg, present in zip(packages, flags)
        if present.upper() != "TRUE"
    )


def find_stata() -> Tool | None:
    """Return the discovered Stata runtime, or None.

    Tries `stata-mp`, `stata-se`, `stata` on PATH first, then common macOS
    `.app` bundle paths.
    """
    for cmd in ("stata-mp", "stata-se", "stata"):
        path = shutil.which(cmd)
        if path:
            return Tool(name="Stata", binary=path)
    for p in _STATA_APP_LOCATIONS:
        if Path(p).is_file() and os.access(p, os.X_OK):
            return Tool(name="Stata", binary=p)
    return None


# Packages the Python runtime helpers (``nora.runtime.nora`` Python
# module) need to actually do work. Pandas is the lingua franca for
# data; statsmodels covers OLS / GLM / t-tests with the SE/CI fields
# the sanitizer expects. SciPy is pulled in transitively by
# statsmodels but checked explicitly so the missing-dep message is
# unambiguous.
_PYTHON_REQUIRED_PACKAGES: tuple[str, ...] = (
    "pandas",
    "numpy",
    "statsmodels",
    "scipy",
)

# Optional but feature-gating. ``matplotlib`` powers every plot
# helper; missing it means ``nora.plot_*`` calls fail silently and
# the model thinks it produced an image while the researcher sees
# nothing. Probed but NOT required so non-plotting scripts still
# run; the executor surfaces missing optionals in its hint text so
# a researcher who wanted plots knows what to install.
_PYTHON_OPTIONAL_PACKAGES: tuple[str, ...] = (
    "matplotlib",
)


def find_python() -> Tool | None:
    """Return the discovered Python 3 interpreter, or None.

    Tries ``python3`` then ``python`` on PATH. Probes the discovered
    interpreter for the scientific-stack packages the runtime library
    requires; missing ones are recorded on ``Tool.missing_packages``
    so the executor can refuse with a clear message rather than
    letting the script crash with ``ModuleNotFoundError`` after the
    sandbox is up.

    Refuses to consider Python 2 (still installed on some macOS
    setups via Homebrew) — the runtime library uses dataclasses and
    f-strings.
    """
    for cmd in ("python3", "python"):
        path = shutil.which(cmd)
        if not path:
            continue
        version = _python_version(path)
        if version is None or not version.startswith("Python 3"):
            continue
        missing = _python_missing_packages(path, _PYTHON_REQUIRED_PACKAGES)
        optional_missing = _python_missing_packages(path, _PYTHON_OPTIONAL_PACKAGES)
        prefixes = _python_prefixes(path)
        return Tool(
            name="Python",
            binary=path,
            version=version,
            missing_packages=missing,
            optional_missing_packages=optional_missing,
            extra_read_paths=prefixes,
        )
    return None


def find_sandbox_exec() -> str | None:
    """Return the path to macOS sandbox-exec, or None on non-macOS or if missing.

    sandbox-exec is deprecated by Apple but still functional through
    current macOS. On Linux and Windows it does not exist; the executor
    falls back to running unsandboxed with a prominent warning.
    """
    # Stable path on macOS. shutil.which may miss it if /usr/bin isn't
    # first on PATH in some shells.
    if Path("/usr/bin/sandbox-exec").is_file():
        return "/usr/bin/sandbox-exec"
    return shutil.which("sandbox-exec")


@dataclass(frozen=True)
class Environment:
    r: Tool | None
    stata: Tool | None
    python: Tool | None
    sandbox_exec: str | None

    def has_any_runtime(self) -> bool:
        # "Has the binary" — a Python install with missing packages
        # still counts here so the banner says "Python detected" and
        # the executor can refuse with a specific
        # please-install-pandas message. has_runnable_runtime() is
        # the strict variant.
        return any(t is not None for t in (self.r, self.stata, self.python))


def detect_environment() -> Environment:
    return Environment(
        r=find_r(),
        stata=find_stata(),
        python=find_python(),
        sandbox_exec=find_sandbox_exec(),
    )


# ---------------------------------------------------------------------------
# Version probing
# ---------------------------------------------------------------------------

def _r_version(binary: str) -> str | None:
    """Run `Rscript --version` and extract a short version string."""
    try:
        out = subprocess.run(
            [binary, "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    # R writes the version banner to stderr on some versions, stdout on
    # others. Check both.
    text = (out.stdout + out.stderr).strip()
    first_line = text.split("\n", 1)[0] if text else ""
    return first_line or None


def _python_version(binary: str) -> str | None:
    """Run ``python --version`` and return the first banner line."""
    try:
        out = subprocess.run(
            [binary, "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    text = (out.stdout + out.stderr).strip()
    first_line = text.split("\n", 1)[0] if text else ""
    return first_line or None


def _python_missing_packages(
    binary: str, required: tuple[str, ...],
) -> tuple[str, ...]:
    """Return the subset of ``required`` packages the interpreter at
    ``binary`` cannot import. An empty tuple means "all present."

    Runs a single ``python -c`` import probe rather than one
    subprocess per package (faster, and survives an interpreter
    that's slow to start). Any non-import-error failure
    (interpreter crash, timeout) is conservatively reported as
    "all packages missing" so the executor's missing-packages
    branch trips and surfaces a coherent error to the researcher.

    The probe runs with ``PYTHONPATH`` set to include the Nora pkg
    dir so packages installed via ``install_packages`` (which writes
    to ``--target <nora_python_pkg_dir>``) count as present. Without
    this the executor's preflight refuses runs even after a
    successful Nora install.
    """
    if not required:
        return ()
    probe = (
        "import json, sys\n"
        f"missing = []\n"
        f"for pkg in {list(required)!r}:\n"
        "    try:\n"
        "        __import__(pkg)\n"
        "    except Exception:\n"
        "        missing.append(pkg)\n"
        "sys.stdout.write(json.dumps(missing))\n"
    )
    # Lazy import: ``package_installer`` is a sibling module and
    # cheap to import, but keeping it lazy avoids any import-cycle
    # surprise if env_detect ever gets pulled in earlier in startup.
    from nora.package_installer import nora_python_pkg_dir
    pkg_dir = str(nora_python_pkg_dir(binary))
    existing = os.environ.get("PYTHONPATH", "")
    probe_env = {
        **os.environ,
        "PYTHONPATH": (
            f"{pkg_dir}{os.pathsep}{existing}" if existing else pkg_dir
        ),
    }
    try:
        out = subprocess.run(
            [binary, "-c", probe],
            capture_output=True,
            text=True,
            timeout=10,
            env=probe_env,
        )
    except (OSError, subprocess.TimeoutExpired):
        return tuple(required)
    if out.returncode != 0:
        return tuple(required)
    try:
        import json as _json
        result = _json.loads(out.stdout.strip() or "[]")
    except (ValueError, TypeError):
        return tuple(required)
    if not isinstance(result, list):
        return tuple(required)
    return tuple(str(x) for x in result)


def _python_prefixes(binary: str) -> tuple[str, ...]:
    """Return ``(sys.prefix,)`` (and ``sys.exec_prefix`` if it differs)
    for the given Python interpreter.

    These paths feed the executor's sandbox profile so the
    interpreter can read its own stdlib + site-packages — without
    them, a venv-based Python (which lives outside the system trees
    the default sandbox already covers) would fail to load even the
    standard library inside the sandbox.

    Best-effort: a probe failure returns an empty tuple. The
    sandbox falls back to the default system trees, which works for
    Apple's bundled python3 and Homebrew installs but breaks venvs.
    """
    try:
        out = subprocess.run(
            [binary, "-c", "import sys; print(sys.prefix); print(sys.exec_prefix)"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ()
    if out.returncode != 0:
        return ()
    lines = [ln.strip() for ln in (out.stdout or "").splitlines() if ln.strip()]
    # Dedupe while preserving order so the more general (sys.prefix)
    # entry comes first; some interpreters report the same path twice.
    seen: set[str] = set()
    result: list[str] = []
    for p in lines:
        if p and p not in seen:
            seen.add(p)
            result.append(p)
    return tuple(result)
