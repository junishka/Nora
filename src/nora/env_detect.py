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

    Uses ``system.file(package = pkg)`` rather than
    ``requireNamespace(pkg)``. ``requireNamespace`` LOADS the
    package namespace, which fires the package's ``.onLoad`` hook
    and runs arbitrary R code OUTSIDE the analysis sandbox (the
    probe is a vanilla ``Rscript`` invocation at app startup and
    after package installs). A malicious or compromised package
    named ``haven`` or ``ggplot2`` would execute code during the
    probe with the full parent environment (no env filter applied
    to ``Rscript`` here). ``system.file`` only checks the
    installed-package directory on disk and does NOT load
    namespaces — answers "is this package installed?" without
    executing package code.

    Failures (Rscript missing, weird R version, OS error) return
    "all missing" rather than the conservative "none missing" so
    the system prompt is honest about uncertainty.
    """
    if not packages:
        return ()
    expr = "; ".join(
        f"cat(nzchar(system.file(package=\"{pkg}\")), \" \")"
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

    Tries ``python3`` then ``python`` on PATH. For each candidate that
    exists and reports a Python 3 version, runs a sandbox-health probe
    (``_probe_sandbox_health``, cached in-memory) before accepting it.
    An interpreter that runs fine outside the sandbox but can't start
    under it — canonically Apple's ``/usr/bin/python3``, an
    ``xcselect`` stub that dlopens ``/Library/Developer/CommandLine
    Tools/usr/lib/libxcrun.dylib`` at startup, a path the executor's
    profile doesn't allow — is rejected so the executor doesn't
    accept a Python it will then fail to run.

    Missing required / optional packages are recorded on
    ``Tool.missing_packages`` so the executor can refuse with a
    clear message rather than letting the script crash with
    ``ModuleNotFoundError`` after the sandbox is up.

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
        # Sandbox-health probe before accepting. The probe lazily
        # populates ``_SANDBOX_PROBE_CACHE`` so the first ``find_python``
        # call pays ~200ms once, and downstream callers (system_prompt,
        # ui banner, executor's preamble lookup) hit the cache. Failed
        # candidates stay in the cache so ``python_sandbox_probe_results``
        # can explain the rejection to the doctor / UI layer.
        if path not in _SANDBOX_PROBE_CACHE:
            _SANDBOX_PROBE_CACHE[path] = _probe_sandbox_health(path)
        if not _SANDBOX_PROBE_CACHE[path][0]:
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


# ---------------------------------------------------------------------------
# Sandbox-health probe for candidate Python interpreters
# ---------------------------------------------------------------------------
#
# Why this exists: outside the sandbox, ``binary -c "print(1)"`` answers
# "does this interpreter respond to ``-c``" — and that's exactly the
# question Apple's ``/usr/bin/python3`` (an ``xcselect`` stub that
# dispatches via libxcrun) passes cleanly. Inside the sandbox, the
# stub dies before main() because libxcrun's dylib lives outside the
# read allowlist. The two checks have different threat models, and the
# gap is where every script-startup failure of this class hides.
#
# The probe closes that gap by running the candidate under the SAME
# profile builder the executor uses for real script runs
# (``executor._sandbox_profile_string``). Drift between probe and run
# would defeat the purpose; sharing the builder is the invariant.
#
# In-memory cache only, lazy on first call. Disk persistence doesn't
# earn its keep — a stale "good" cache pointing at an uninstalled
# interpreter is a worse failure than a 200ms cold-start probe, and
# probe cost is small enough that within-process caching covers the
# realistic cases (system_prompt, ui banner, the executor's preamble
# lookup all hit the same path within one Nora session).
_SANDBOX_PROBE_CACHE: dict[str, tuple[bool, str]] = {}

# Separate cache for the SANDBOX layer's own health (phase A of the
# probe). The interpreter probe runs ``sandbox-exec -f <profile>
# <binary> ...`` — if sandbox-exec or the profile compiler is itself
# broken, every interpreter candidate fails identically and the doctor
# would mis-attribute the failure to the interpreter. Caching the
# baseline check separately lets us report "sandbox is broken" without
# blaming the (possibly fine) Python install.
_SANDBOX_BASELINE_CACHE: "tuple[bool, str] | None" = None


def _check_sandbox_baseline() -> tuple[bool, str]:
    """Verify ``sandbox-exec`` can apply a minimal profile at all.

    The interpreter probe builds a real Nora profile and runs the
    candidate binary under it. That sequence can fail for two
    distinct reasons:

      * **Sandbox layer broken** — sandbox-exec is missing or
        misconfigured, the SBPL compiler rejects the profile, the
        OS doesn't honour ``sandbox_apply`` (nested-sandbox
        harnesses, exotic macOS variants). Every interpreter
        probed under such a sandbox fails identically.
      * **Interpreter rejected by a working sandbox** — the Apple
        xcrun stub class of failure, what the doctor was originally
        built to catch.

    Without distinguishing these, a researcher whose sandbox is
    broken would see "install Homebrew Python" advice that won't
    help. This function runs a minimal ``(allow default)`` profile
    against ``/usr/bin/true`` — the simplest possible sandbox
    invocation. If that fails, sandbox-exec itself is the problem,
    and ``_probe_sandbox_health`` short-circuits with the baseline
    error instead of running per-candidate probes.

    Cached in-memory for the lifetime of the process (the sandbox
    layer's health doesn't change mid-session). Non-macOS hosts
    return ``(True, "")`` — the executor doesn't sandbox there, so
    the baseline question is moot and the "sandbox not present"
    case is reported separately by the doctor's sandbox-runtime
    row.
    """
    global _SANDBOX_BASELINE_CACHE
    if _SANDBOX_BASELINE_CACHE is not None:
        return _SANDBOX_BASELINE_CACHE

    sandbox_exec = find_sandbox_exec()
    if sandbox_exec is None:
        _SANDBOX_BASELINE_CACHE = (True, "")
        return _SANDBOX_BASELINE_CACHE

    try:
        out = subprocess.run(
            [
                sandbox_exec, "-p", "(version 1)(allow default)",
                "/usr/bin/true",
            ],
            capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        _SANDBOX_BASELINE_CACHE = (
            False, f"sandbox-exec failed to launch: {e}",
        )
        return _SANDBOX_BASELINE_CACHE
    if out.returncode != 0:
        _SANDBOX_BASELINE_CACHE = (
            False,
            (
                "sandbox-exec rejected a minimal allow-default profile "
                f"(exit {out.returncode}). stderr: "
                f"{(out.stderr or '').strip() or '(empty)'}"
            ),
        )
        return _SANDBOX_BASELINE_CACHE
    _SANDBOX_BASELINE_CACHE = (True, "")
    return _SANDBOX_BASELINE_CACHE


def sandbox_baseline_result() -> tuple[bool, str]:
    """Public accessor for the sandbox-baseline cache.

    Used by the doctor's ``_sandbox_report`` to surface "sandbox
    layer broken" as a distinct failure from "sandbox-exec not
    present", and by tests that need to pre-seed the cache to
    simulate either branch.
    """
    return _check_sandbox_baseline()


def _probe_sandbox_health(binary: str) -> tuple[bool, str]:
    """Run a trivial ``binary -I -c "print(1)"`` under a representative
    Nora sandbox profile and report whether it succeeded.

    Returns ``(ok, stderr_excerpt)``:
      * ``ok=True`` and empty stderr when the sandboxed subprocess
        exited 0 with ``"1"`` on stdout.
      * ``ok=False`` with up to ~4 KB of the failing stderr (the tail,
        which is what carries the actual error). Callers — the doctor
        command and the executor's error path — pass this back to the
        researcher unredacted: at this phase no researcher data has
        been touched, so launcher / dlopen / sandbox-denial output is
        safe to surface.

    On non-macOS systems (no ``sandbox-exec``) the probe is a no-op
    and returns ``(True, "")``. The executor doesn't sandbox there, so
    probing for sandbox compatibility is moot.

    Profile shape: built by ``executor._sandbox_profile_string`` so
    the probe and the executor's per-run profile share one source of
    truth. The probe's ephemeral cwd / run_dir live under
    ``tempfile.mkdtemp()``; both are added to the read allowlist by
    the same code the real run uses, so a candidate that passes here
    starts cleanly under the real run too (modulo cwd-specific
    paths, which don't affect interpreter startup).
    """
    sandbox_exec = find_sandbox_exec()
    if sandbox_exec is None:
        return True, ""

    # Phase A: is the sandbox layer itself usable? If a minimal
    # ``(allow default)`` profile against ``/usr/bin/true`` already
    # fails, every interpreter probe would fail identically and the
    # diagnostic (and the doctor's downstream rendering) would
    # mis-attribute the failure to the interpreter rather than the
    # sandbox. Short-circuit here with the baseline error so the
    # rejection cache carries an explicit "sandbox layer broken"
    # signal instead of N copies of the same downstream symptom.
    baseline_ok, baseline_err = _check_sandbox_baseline()
    if not baseline_ok:
        return False, (
            "sandbox-exec itself is unusable; this rejection is not "
            "specific to the interpreter at "
            f"{binary}. {baseline_err}"
        )

    # Lazy local imports to avoid an import-cycle with executor /
    # package_installer at module load. Same pattern as
    # ``_python_missing_packages`` / ``_python_prefixes`` below.
    import tempfile
    from nora.executor import _filter_env, _sandbox_profile_string

    prefixes = _python_prefixes(binary)
    with tempfile.TemporaryDirectory(prefix="nora-probe-") as scratch_str:
        # ``.resolve()`` is required on macOS: /var/folders/... is
        # reached as /private/var/folders/... at the kernel level,
        # and SBPL matches resolved paths. Without resolve(), the
        # profile's ``(subpath "/var/folders/...")`` allow doesn't
        # cover the kernel-side path the probe actually accesses.
        scratch = Path(scratch_str).resolve()
        run_dir = scratch / ".nora" / "runs" / "probe"
        run_dir.mkdir(parents=True)
        profile_text = _sandbox_profile_string(
            run_dir=run_dir, cwd=scratch, extra_read_paths=prefixes,
        )
        profile_path = run_dir / "sandbox.sb"
        profile_path.write_text(profile_text)

        try:
            out = subprocess.run(
                [
                    sandbox_exec, "-f", str(profile_path),
                    binary, "-I", "-c", "print(1)",
                ],
                capture_output=True, text=True, timeout=10,
                env=_filter_env(dict(os.environ)),
                cwd=str(scratch),
            )
        except (OSError, subprocess.TimeoutExpired) as e:
            return False, f"probe could not launch: {e}"

    if out.returncode == 0 and (out.stdout or "").strip() == "1":
        return True, ""
    # Tail of stderr — most launcher / dlopen errors emit a single
    # short line, but Python's traceback formatter and SBPL's deny
    # messages can run long. Cap at ~4 KB so the UI / doctor output
    # doesn't balloon, but keep the tail (the proximate cause is
    # almost always the last line).
    return False, (out.stderr or "")[-4000:]


def python_sandbox_probe_results() -> dict[str, tuple[bool, str]]:
    """Return a snapshot of the in-memory sandbox-probe cache.

    Each key is a probed interpreter path; value is
    ``(ok, stderr_excerpt)``. Used by the ``nora doctor`` command and
    by the executor's "no python3 found" error path to explain why a
    candidate Python was rejected — without this, every script
    silently dies and the researcher has no clue why.
    """
    return dict(_SANDBOX_PROBE_CACHE)


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
    """Run ``python --version`` and return the first banner line.

    ``--version`` short-circuits CPython startup before ``site.py``
    runs, so ``sitecustomize`` / ``usercustomize`` and the inherited
    ``PYTHONPATH`` can't execute code via this probe. The filtered
    env + no ``-I`` here is intentional: ``-I`` doesn't compose
    with bare ``--version`` on all Python versions and the version
    flag doesn't load anything off sys.path anyway. The filtered
    env still strips parent-process secrets (``ANTHROPIC_API_KEY``,
    AWS creds) before the subprocess inherits them, matching the
    other probes.
    """
    from nora.executor import _filter_env
    try:
        out = subprocess.run(
            [binary, "--version"],
            capture_output=True,
            text=True,
            timeout=5,
            env=_filter_env(dict(os.environ)),
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

    Uses ``importlib.util.find_spec(pkg)`` rather than
    ``__import__(pkg)``. ``__import__`` executes the package's
    ``__init__.py`` (and any imports it triggers) OUTSIDE the
    analysis sandbox: this probe runs at app startup and after
    ``install_packages``, with full filesystem and network. A
    package masquerading as ``pandas`` / ``statsmodels`` would
    get code execution during detection. ``find_spec`` only
    consults sys.path finders — for top-level package names it
    returns metadata without importing the package, so no
    ``__init__.py`` runs.

    The probe is launched with ``-I`` (isolated mode) so the
    inherited ``PYTHONPATH`` and the user-site ``usercustomize.py``
    can't inject startup code. The Nora package dir is added to
    ``sys.path`` explicitly inside the probe rather than via
    ``PYTHONPATH`` env (which ``-I`` ignores). Without ``-I`` an
    attacker-controlled ``sitecustomize.py`` / ``usercustomize.py``
    on the inherited path would execute at every detection run.
    """
    if not required:
        return ()
    # Lazy import: ``package_installer`` and ``executor`` are sibling
    # modules and cheap to import, but keeping them lazy avoids any
    # import-cycle surprise if env_detect ever gets pulled in earlier
    # in startup.
    from nora.package_installer import nora_python_pkg_dir
    from nora.executor import _filter_env
    pkg_dir = str(nora_python_pkg_dir(binary))
    # ``find_spec`` returns ``None`` when the package isn't on
    # sys.path. Wrap in try/except so a finder that raises (rare,
    # but possible with broken namespace packages) doesn't mark
    # ALL packages missing — only the offending one.
    probe = (
        "import json, sys\n"
        f"sys.path.insert(0, {pkg_dir!r})\n"
        "import importlib.util\n"
        f"required = {list(required)!r}\n"
        "missing = []\n"
        "for pkg in required:\n"
        "    try:\n"
        "        spec = importlib.util.find_spec(pkg)\n"
        "    except Exception:\n"
        "        spec = None\n"
        "    if spec is None:\n"
        "        missing.append(pkg)\n"
        "sys.stdout.write(json.dumps(missing))\n"
    )
    # Filter the probe's env through the same allowlist the executor
    # uses for analysis scripts. The probe runs OUTSIDE the script
    # sandbox and with no network deny. Without the filter, the
    # interpreter inherits secrets like ``ANTHROPIC_API_KEY`` / AWS
    # credentials from the parent process env. The script executor's
    # ``_filter_env`` is the canonical allowlist; using it here keeps
    # the two surfaces aligned.
    probe_env = _filter_env(dict(os.environ))
    try:
        out = subprocess.run(
            [binary, "-I", "-c", probe],
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

    ``-I`` (isolated mode) + filtered env: the probe runs OUTSIDE
    the analysis sandbox at startup. Without ``-I``, an inherited
    ``PYTHONPATH`` pointing at attacker-controlled ``sitecustomize.py``
    would execute code during the probe. The probe only reads
    ``sys.prefix`` / ``sys.exec_prefix``, which are populated
    independently of the user/parent PYTHONPATH, so ``-I`` is safe
    here.

    Best-effort: a probe failure returns an empty tuple. The
    sandbox falls back to the default system trees, which works for
    Apple's bundled python3 and Homebrew installs but breaks venvs.
    """
    from nora.executor import _filter_env
    try:
        out = subprocess.run(
            [binary, "-I", "-c", "import sys; print(sys.prefix); print(sys.exec_prefix)"],
            capture_output=True,
            text=True,
            timeout=5,
            env=_filter_env(dict(os.environ)),
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
