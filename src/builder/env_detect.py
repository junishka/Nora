"""Builder — runtime environment detection.

Finds the researcher's installed `Rscript` / `stata` binaries so the
executor knows what to invoke. Checks `PATH` and common macOS install
locations. No fuzziness: either we find an executable or we don't.

The result is consulted at app startup so the banner can honestly tell
the researcher what Builder will and won't be able to run for them.
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


def find_r() -> Tool | None:
    """Return the discovered R runtime, or None."""
    path = shutil.which("Rscript")
    if path is None:
        return None
    return Tool(name="R", binary=path, version=_r_version(path))


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
    sandbox_exec: str | None

    def has_any_runtime(self) -> bool:
        return self.r is not None or self.stata is not None


def detect_environment() -> Environment:
    return Environment(
        r=find_r(),
        stata=find_stata(),
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
