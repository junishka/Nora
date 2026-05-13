"""Reviewer-flagged regressions for ``nora.package_installer``.

The reviewer batch covered four issues; the unit-testable three live
here:

1. Python install path mismatch: ``pip install --user`` writes to a
   directory the executor (``python -I``) and the sandbox both refuse
   to read from, so ``submit_script`` fails with
   ``ModuleNotFoundError`` after a "successful" install. The fix
   routes installs through ``--target <nora_python_pkg_dir>`` and
   wires that dir into the executor preamble + sandbox + env-detect
   probe. These tests pin the argv shape and the integration with
   the executor surfaces.

2. Package validator allowed pip option injection: the previous
   ``^[A-Za-z0-9._-]+$`` matched ``-r``, ``-e``, ``--no-index``,
   ``.``, ``..`` — pip parses any of those as flags / local-path
   installs rather than registry names. The validator now requires
   a leading alphanumeric char and the pip argv has a ``--``
   end-of-options separator before the package list as defense-in-
   depth.

3. Stata command joiner used ``;``: Stata's default delimiter inside
   a do-file is a newline, so the joined string was a syntax error
   even for a single-package reinstall. The joiner now emits one
   line per command.
"""
from __future__ import annotations

import os

import pytest


# ---------------------------------------------------------------------------
# 2. Package-name validator
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("hostile", [
    "-r",
    "-e",
    "--no-index",
    "--target",
    ".",
    "..",
    "-rrequirements.txt",
    "-",
])
def test_validate_names_rejects_pip_option_shapes(hostile: str) -> None:
    """The reviewer's option-injection list. Each of these would have
    been parsed by pip as a flag or a local-path install rather than a
    registry package name; the leading-alphanumeric anchor on the
    name regex closes the bypass."""
    from nora.package_installer import _validate_names

    valid, rejected = _validate_names([hostile])
    assert valid == []
    assert len(rejected) == 1
    assert rejected[0].status == "failed"
    assert "rejected" in rejected[0].detail


@pytest.mark.parametrize("legit", [
    "matplotlib",
    "scipy",
    "scikit-learn",
    "numpy",
    "statsmodels",
    "pandas",
    # Names with internal dots / dashes / underscores stay valid;
    # the anchor only forbids LEADING punctuation.
    "ggplot2",
    "data.table",
    "py4j",
    "google-cloud-storage",
])
def test_validate_names_accepts_canonical_registry_names(legit: str) -> None:
    """The tightened regex must not regress on real PyPI / CRAN /
    SSC names — those still start with an alphanumeric char."""
    from nora.package_installer import _validate_names

    valid, rejected = _validate_names([legit])
    assert valid == [legit]
    assert rejected == []


def test_validate_names_rejects_overlong() -> None:
    from nora.package_installer import _validate_names, _NAME_MAX_LEN

    valid, rejected = _validate_names(["a" * (_NAME_MAX_LEN + 1)])
    assert valid == []
    assert len(rejected) == 1


def test_python_remove_refuses_package_not_in_nora_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SDC + safety closure: ``pip uninstall`` has no ``--target`` —
    it resolves via ``sys.path`` and removes the first copy it
    finds. If the requested package is NOT in Nora's managed Python
    dir but IS in the researcher's system / user / venv
    site-packages, pip would happily remove it from there. A model
    that calls ``install_packages(action='remove', packages=['pandas'])``
    when Nora never installed pandas to its own dir could yank
    pandas from the researcher's broader Python environment —
    breaking Nora itself.

    The fix: before launching ``pip uninstall``, scan the Nora target
    dir's ``*.dist-info`` and ``*.egg-info`` and refuse any package
    name that isn't present there. The refusal surfaces as a
    ``skipped`` per-package status; the subprocess never starts if
    no eligible names remain.
    """
    import asyncio
    monkeypatch.setenv(
        "NORA_PYTHON_PKG_BASE", str(tmp_path / "nora-pkgs"),
    )
    from nora.package_installer import (
        InstallResult,
        install_packages,
        nora_python_pkg_dir,
    )
    # Stub the env-detect to a Python that resolves to a real
    # binary path (it never runs, since we'll intercept the
    # subprocess).
    import nora.env_detect as _env_detect
    from nora.env_detect import Environment, Tool
    fake_env = Environment(
        python=Tool(
            name="Python", binary="/usr/bin/python3",
            version="Python 3.12.0",
            missing_packages=(), optional_missing_packages=(),
            extra_read_paths=(),
        ),
        r=None, stata=None, sandbox_exec=None,
    )
    monkeypatch.setattr(_env_detect, "detect_environment", lambda: fake_env)

    # The Nora target dir EXISTS and contains ONE installed package
    # (``managed``) but NOT the package the model is asking to
    # remove (``site_only``).
    target = nora_python_pkg_dir("/usr/bin/python3")
    target.mkdir(parents=True, exist_ok=True)
    (target / "managed-1.2.3.dist-info").mkdir()
    (target / "managed-1.2.3.dist-info" / "METADATA").write_text(
        "Name: managed\n", encoding="utf-8",
    )

    # Intercept subprocess.Popen for pip invocations only. The
    # installer now uses Popen + communicate (so it can killpg the
    # process tree on timeout / cancel). ``nora_python_pkg_dir``
    # still uses subprocess.run for its version probe, which goes
    # through Popen internally — let those passthrough to the real
    # Popen by checking ``start_new_session`` (only the install path
    # sets it).
    pip_launched: list[list[str]] = []
    import subprocess as _subprocess
    real_popen = _subprocess.Popen

    class _FakePopen:
        def __init__(self, cmd, **kwargs):  # type: ignore[no-untyped-def]
            self._cmd = list(cmd) if isinstance(cmd, list) else cmd
            self.pid = -1
            self.returncode = 0

        def communicate(self, input=None, timeout=None):  # type: ignore[no-untyped-def]
            return ("", "")

        def kill(self) -> None:  # pragma: no cover — never reached
            pass

    def _fake_popen(cmd, **kwargs):  # type: ignore[no-untyped-def]
        is_install_call = kwargs.get("start_new_session")
        is_pip = (
            isinstance(cmd, list)
            and len(cmd) >= 3
            and cmd[1] == "-m"
            and cmd[2] == "pip"
        )
        if is_install_call and is_pip:
            pip_launched.append(list(cmd))
            return _FakePopen(cmd, **kwargs)
        return real_popen(cmd, **kwargs)
    monkeypatch.setattr(_subprocess, "Popen", _fake_popen)

    # Case 1: removing a package that ISN'T in Nora's target. The
    # request must fail cleanly without pip ever being launched.
    result = asyncio.run(install_packages(
        language="Python", packages=["site_only"], action="remove",
    ))
    assert isinstance(result, InstallResult)
    assert result.error is not None
    assert pip_launched == [], (
        "pip uninstall must NOT be launched for a package that "
        "isn't in Nora's target dir — otherwise pip would remove "
        "it from the researcher's broader Python environment"
    )
    statuses_by_name = {s.name: s for s in result.statuses}
    assert "site_only" in statuses_by_name
    assert statuses_by_name["site_only"].status == "skipped"

    # Case 2: a MIX of eligible + non-eligible names. The eligible
    # one goes through; the non-eligible one is skipped, never
    # reaches pip.
    pip_launched.clear()
    result2 = asyncio.run(install_packages(
        language="Python", packages=["managed", "site_only"],
        action="remove",
    ))
    assert len(pip_launched) == 1
    pip_argv = pip_launched[0]
    assert "managed" in pip_argv
    assert "site_only" not in pip_argv, (
        "the non-eligible package leaked into the pip argv — that "
        "would let pip uninstall it from site-packages"
    )
    statuses2_by_name = {s.name: s for s in result2.statuses}
    assert statuses2_by_name["site_only"].status == "skipped"


@pytest.mark.parametrize("name", [
    "nora",
    "Nora",
    "NORA",
])
def test_python_install_refuses_nora_distribution_name(
    name: str, tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Python ``nora`` helper module (``nora.from_lm``,
    ``nora.result``, ``nora.plot_*``) is staged onto every script's
    sys.path by the executor preamble. A model that calls
    ``install_packages(language='Python', packages=['nora'])`` is
    confused: there's nothing to fetch (the helpers are already
    importable), and the literal ``nora`` distribution on PyPI is an
    unrelated empty placeholder owned by another author (~1 KB,
    metadata only, no module code). Installing it does not provide
    the helpers and leaves a useless ``.dist-info`` in the Nora pkg
    dir.

    The guard rejects the name (PEP 503 normalised: case-folded,
    ``-``/``_``/``.`` collapsed) at the installer boundary BEFORE pip
    ever launches, and surfaces a ``skipped`` per-package status with
    the instruction to just ``import nora``.
    """
    import asyncio
    monkeypatch.setenv(
        "NORA_PYTHON_PKG_BASE", str(tmp_path / "nora-pkgs"),
    )
    from nora.package_installer import InstallResult, install_packages
    import nora.env_detect as _env_detect
    from nora.env_detect import Environment, Tool

    fake_env = Environment(
        python=Tool(
            name="Python", binary="/usr/bin/python3",
            version="Python 3.12.0",
            missing_packages=(), optional_missing_packages=(),
            extra_read_paths=(),
        ),
        r=None, stata=None, sandbox_exec=None,
    )
    monkeypatch.setattr(_env_detect, "detect_environment", lambda: fake_env)

    # Track every pip subprocess that gets launched — must remain
    # empty for the nora-only case.
    pip_launched: list[list[str]] = []
    import subprocess as _subprocess
    real_popen = _subprocess.Popen

    def _fake_popen(cmd, **kwargs):  # type: ignore[no-untyped-def]
        is_install_call = kwargs.get("start_new_session")
        is_pip = (
            isinstance(cmd, list)
            and len(cmd) >= 3
            and cmd[1] == "-m"
            and cmd[2] == "pip"
        )
        if is_install_call and is_pip:
            pip_launched.append(list(cmd))

            class _FakePopen:
                pid = -1
                returncode = 0

                def communicate(self, input=None, timeout=None):  # noqa: ARG002
                    return ("", "")

                def kill(self) -> None:  # pragma: no cover
                    pass

            return _FakePopen()
        return real_popen(cmd, **kwargs)
    monkeypatch.setattr(_subprocess, "Popen", _fake_popen)

    result = asyncio.run(install_packages(
        language="Python", packages=[name], action="install",
    ))

    assert isinstance(result, InstallResult)
    assert pip_launched == [], (
        f"pip must NOT launch for the {name!r} distribution name — "
        "the runtime helpers are preloaded by the executor preamble"
    )
    assert result.error is not None
    statuses_by_name = {s.name: s for s in result.statuses}
    assert name in statuses_by_name
    blocked = statuses_by_name[name]
    assert blocked.status == "skipped"
    assert "import nora" in blocked.detail
    assert "placeholder" in blocked.detail


def test_python_install_blocks_nora_alongside_legitimate_packages(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mixed-list case: a model request like ``['pandas', 'nora']``
    must let ``pandas`` through to pip while ``nora`` is filtered
    out as ``skipped``. Don't tank the entire call because of one
    bad name."""
    import asyncio
    monkeypatch.setenv(
        "NORA_PYTHON_PKG_BASE", str(tmp_path / "nora-pkgs"),
    )
    from nora.package_installer import install_packages
    import nora.env_detect as _env_detect
    from nora.env_detect import Environment, Tool

    fake_env = Environment(
        python=Tool(
            name="Python", binary="/usr/bin/python3",
            version="Python 3.12.0",
            missing_packages=(), optional_missing_packages=(),
            extra_read_paths=(),
        ),
        r=None, stata=None, sandbox_exec=None,
    )
    monkeypatch.setattr(_env_detect, "detect_environment", lambda: fake_env)

    pip_launched: list[list[str]] = []
    import subprocess as _subprocess
    real_popen = _subprocess.Popen

    def _fake_popen(cmd, **kwargs):  # type: ignore[no-untyped-def]
        is_install_call = kwargs.get("start_new_session")
        is_pip = (
            isinstance(cmd, list)
            and len(cmd) >= 3
            and cmd[1] == "-m"
            and cmd[2] == "pip"
        )
        if is_install_call and is_pip:
            pip_launched.append(list(cmd))

            class _FakePopen:
                pid = -1
                returncode = 0

                def communicate(self, input=None, timeout=None):  # noqa: ARG002
                    return ("", "")

                def kill(self) -> None:  # pragma: no cover
                    pass

            return _FakePopen()
        return real_popen(cmd, **kwargs)
    monkeypatch.setattr(_subprocess, "Popen", _fake_popen)

    result = asyncio.run(install_packages(
        language="Python", packages=["pandas", "nora"], action="install",
    ))

    assert len(pip_launched) == 1, (
        "exactly one pip invocation expected — for pandas only"
    )
    pip_argv = pip_launched[0]
    assert "pandas" in pip_argv
    assert "nora" not in pip_argv, (
        "the nora name leaked into the pip argv — guard must run "
        "before pip dispatch"
    )
    statuses_by_name = {s.name: s for s in result.statuses}
    assert statuses_by_name["nora"].status == "skipped"
    assert statuses_by_name["pandas"].status == "ok"


def test_python_remove_handles_pep503_name_normalization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pip normalises distribution names per PEP 503 — ``scikit_learn``
    on disk is ``scikit-learn`` to the consumer, and vice versa. The
    Nora-target presence check must match how pip wrote the dir, so
    a remove request with the underscore form still matches the
    dash form on disk (and vice versa).
    """
    monkeypatch.setenv(
        "NORA_PYTHON_PKG_BASE", str(tmp_path / "nora-pkgs"),
    )
    from nora.package_installer import (
        _python_packages_installed_in_nora_target,
        nora_python_pkg_dir,
    )
    target = nora_python_pkg_dir("/usr/bin/python3")
    target.mkdir(parents=True, exist_ok=True)
    # On-disk: ``scikit_learn-1.0.dist-info`` (underscore form).
    (target / "scikit_learn-1.0.dist-info").mkdir()
    # Request uses the dash form.
    found = _python_packages_installed_in_nora_target(
        target, ["scikit-learn"],
    )
    assert found == {"scikit-learn"}
    # And vice versa: dash on disk, underscore in request.
    (target / "another-pkg-2.0.dist-info").mkdir()
    found2 = _python_packages_installed_in_nora_target(
        target, ["another_pkg"],
    )
    assert found2 == {"another_pkg"}


def test_install_packages_tool_scrubs_credentials_from_raw_excerpts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SDC closure: the ``install_packages`` MCP tool returns
    ``raw_stdout_excerpt`` and ``raw_stderr_excerpt`` on failure so
    the model can diagnose. The script-sandbox path runs everything
    through ``error_summary.extract_debug_excerpt`` (language-
    anchored extraction + credential/path scrub) — the install path
    skips the language anchor but MUST NOT skip the scrub.

    The headline leak we lock against is the private pip index URL
    that pip echoes on every run from ``~/.pip/pip.conf`` (or
    ``PIP_INDEX_URL``):

        Looking in indexes: https://USER:TOKEN@private-pypi.acme.com/simple

    Without the scrub, the embedded user:token rides the failure
    response straight into the model's context. The fix pipes both
    raw excerpts through ``error_summary.scrub_raw_output`` before
    they reach the response payload.
    """
    import asyncio
    import json

    from nora.package_installer import InstallResult
    from nora.tools import HANDLERS
    import nora.tools as tools_mod

    fake_result = InstallResult(
        language="Python",
        action="install",
        statuses=(),
        raw_stdout=(
            "Looking in indexes: https://leaked_user:leaked_token@"
            "private-pypi.acme.com/simple\n"
            "Collecting pandas\n"
        ),
        raw_stderr=(
            "ERROR: HTTPSConnectionPool(host='private-pypi.acme.com', "
            "port=443): Max retries exceeded\n"
            "OPENAI_KEY=sk-abcdefghijklmnopqrstuvwxyz1234567890\n"
            "Failed at /Users/jdoe/.cache/pip/wheels/build.log\n"
        ),
        error="installer exited 1",
        duration_seconds=0.5,
    )

    async def _fake_install(language, packages, action, proc_register=None):  # type: ignore[no-untyped-def]
        return fake_result

    monkeypatch.setattr(
        tools_mod,
        "install_packages",
        tools_mod.install_packages,
    )
    # Patch the module-level import target that the tool handler
    # reaches for inside its body.
    import nora.package_installer as pkg_mod
    monkeypatch.setattr(pkg_mod, "install_packages", _fake_install)
    # The tool now gates installs behind a researcher-side
    # confirmation modal; stub the approval so the install runs and
    # we can assert on the failure-path scrub behavior.
    import nora.install_confirmation as ic_mod

    async def _approve(**_kwargs):
        return True
    monkeypatch.setattr(ic_mod, "request_confirmation", _approve)

    payload = asyncio.run(HANDLERS["install_packages"]({
        "language": "Python",
        "packages": ["pandas"],
        "action": "install",
    }))
    body = json.loads(next(
        b for b in payload["content"] if b.get("type") == "text"
    )["text"])
    assert body["status"] == "error"
    stdout_excerpt = body["raw_stdout_excerpt"]
    stderr_excerpt = body["raw_stderr_excerpt"]

    # Embedded URL credentials are gone; scheme + host are preserved
    # so a reader can still tell what was happening.
    assert "leaked_user" not in stdout_excerpt
    assert "leaked_token" not in stdout_excerpt
    assert "[redacted-credential]" in stdout_excerpt
    assert "https://" in stdout_excerpt

    # OpenAI-style key in stderr is redacted.
    assert "sk-abcdefghijklmnopqrstuvwxyz1234567890" not in stderr_excerpt
    assert "[redacted-credential]" in stderr_excerpt

    # Absolute path is reduced to its basename.
    assert "/Users/jdoe" not in stderr_excerpt
    assert "build.log" in stderr_excerpt


def test_install_packages_caps_list_length() -> None:
    """A massive package list shouldn't launch a single multi-hour
    invocation the researcher can't easily interrupt. The per-call
    cap rejects oversized batches before any subprocess fires.
    """
    import asyncio

    from nora.package_installer import install_packages as _do_install

    # 60 names — over the 50-package cap, all individually well-formed.
    names = [f"pkg{i:03d}" for i in range(60)]
    result = asyncio.run(_do_install("Python", names, "install"))

    assert result.error is not None
    assert "too many" in result.error.lower() or "cap" in result.error.lower()
    # No subprocess fired, so no statuses, no stdout, no stderr.
    assert result.statuses == ()
    assert result.raw_stdout == ""
    assert result.raw_stderr == ""


# ---------------------------------------------------------------------------
# 1. Python install command shape
# ---------------------------------------------------------------------------

def test_python_command_install_uses_target_not_user(tmp_path, monkeypatch) -> None:
    """``--user`` was the bug — it writes to a path the executor's
    isolated-mode interpreter and the sandbox both ignore. The fix
    is ``--target <nora_python_pkg_dir>``. Pin the argv shape."""
    monkeypatch.setenv("NORA_PYTHON_PKG_BASE", str(tmp_path))
    from nora.package_installer import _python_command, nora_python_pkg_dir

    cmd = _python_command("/usr/bin/python3", ["matplotlib", "scipy"], "install")
    expected_target = str(nora_python_pkg_dir("/usr/bin/python3"))

    assert "--user" not in cmd, "must not use --user; see docstring"
    assert "--target" in cmd
    assert cmd[cmd.index("--target") + 1] == expected_target
    assert "--" in cmd
    # Package names appear AFTER the ``--`` end-of-options separator.
    sep = cmd.index("--")
    assert cmd[sep + 1:] == ["matplotlib", "scipy"]


def test_python_command_reinstall_uses_force_reinstall_no_deps(
    tmp_path, monkeypatch,
) -> None:
    monkeypatch.setenv("NORA_PYTHON_PKG_BASE", str(tmp_path))
    from nora.package_installer import _python_command

    cmd = _python_command("/usr/bin/python3", ["matplotlib"], "reinstall")
    assert "--force-reinstall" in cmd
    assert "--no-deps" in cmd
    assert "--target" in cmd
    assert "--user" not in cmd
    assert "--" in cmd


def test_python_command_remove_uses_uninstall(tmp_path, monkeypatch) -> None:
    """``pip uninstall`` has no ``--target``; the runner separately
    sets PYTHONPATH so pip locates the package via sys.path. The
    argv shape here is just ``uninstall -y -- pkg…``."""
    monkeypatch.setenv("NORA_PYTHON_PKG_BASE", str(tmp_path))
    from nora.package_installer import _python_command

    cmd = _python_command("/usr/bin/python3", ["matplotlib"], "remove")
    assert "uninstall" in cmd
    assert "-y" in cmd
    # End-of-options separator is present even on uninstall — keeps
    # the package-name boundary identical across actions.
    assert "--" in cmd


def test_python_command_creates_target_dir(tmp_path, monkeypatch) -> None:
    """The ``--target`` dir must exist before pip writes; if it
    doesn't, pip fails with a confusing path error. Verify the
    command builder eagerly creates it."""
    monkeypatch.setenv("NORA_PYTHON_PKG_BASE", str(tmp_path))
    from nora.package_installer import _python_command, nora_python_pkg_dir

    expected = nora_python_pkg_dir("/usr/bin/python3")
    assert not expected.exists()
    _python_command("/usr/bin/python3", ["matplotlib"], "install")
    assert expected.is_dir()


# ---------------------------------------------------------------------------
# 4. Stata joiner — newlines, not semicolons
# ---------------------------------------------------------------------------

def test_stata_command_install_uses_newlines() -> None:
    from nora.package_installer import _stata_command

    cmd, stdin = _stata_command(
        "/Applications/Stata/StataMP.app/Contents/MacOS/StataMP",
        ["estout", "ftools"],
        "install",
    )
    # No semicolon separators (Stata's default delimiter is \n).
    # Internal commas are fine — they're option separators inside a
    # single ``ssc install pkg, replace`` command.
    assert ";" not in stdin
    assert stdin.count("\n") >= 2  # one per package, plus trailing
    assert "ssc install estout, replace" in stdin
    assert "ssc install ftools, replace" in stdin


def test_stata_command_single_package_reinstall_no_semicolons() -> None:
    """Even a single-package reinstall used to embed an internal
    ``;`` (``capture ado uninstall p; ssc install p, replace``),
    failing on its own — not just on multi-package joins. Pin the
    fix at the floor case the reviewer called out."""
    from nora.package_installer import _stata_command

    _, stdin = _stata_command("/path/to/stata", ["estout"], "reinstall")
    assert ";" not in stdin
    assert "capture ado uninstall estout" in stdin
    assert "ssc install estout, replace" in stdin
    # Each on its own line.
    lines = [ln for ln in stdin.splitlines() if ln.strip()]
    assert "capture ado uninstall estout" in lines
    assert "ssc install estout, replace" in lines


def test_stata_command_remove_uses_newlines() -> None:
    from nora.package_installer import _stata_command

    _, stdin = _stata_command("/path/to/stata", ["a", "b"], "remove")
    assert ";" not in stdin
    lines = [ln for ln in stdin.splitlines() if ln.strip()]
    assert "ado uninstall a" in lines
    assert "ado uninstall b" in lines


# ---------------------------------------------------------------------------
# Executor integration — the install dir reaches both surfaces
# ---------------------------------------------------------------------------

def test_python_preamble_adds_nora_pkg_dir_to_sys_path(
    tmp_path, monkeypatch,
) -> None:
    """The preamble must add the Nora pkg dir to ``sys.path`` —
    without this, ``-I`` mode never sees Nora-installed packages even
    if pip wrote them to the right place. This is half of the
    install/import alignment fix."""
    monkeypatch.setenv("NORA_PYTHON_PKG_BASE", str(tmp_path / "pkgs"))
    from nora.executor import _write_script
    from nora.package_installer import nora_python_pkg_dir
    from nora.env_detect import find_python

    py_tool = find_python()
    if py_tool is None:
        pytest.skip("python3 not on PATH in this test env")

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "lib").mkdir()
    script_path = _write_script(run_dir, "Python", "print('hi')\n")

    text = script_path.read_text(encoding="utf-8")
    expected_pkg_dir = str(nora_python_pkg_dir(py_tool.binary))
    assert expected_pkg_dir in text, (
        "preamble must insert nora_python_pkg_dir on sys.path so "
        "Nora-installed packages resolve under -I mode"
    )


def test_python_preamble_lib_dir_wins_over_pkg_dir(
    tmp_path, monkeypatch,
) -> None:
    """The staged Nora runtime (``<run_dir>/lib/nora.py``) must take
    precedence over anything in ``nora_python_pkg_dir`` on ``sys.path``.

    Concrete threat: a model-authored ``install_packages(["nora"])``
    call (or any third-party wheel that ships a top-level ``nora``
    module) writes to ``nora_python_pkg_dir``. If pkg_dir is on
    ``sys.path`` ahead of ``lib_dir``, the user script's
    ``import nora`` resolves to the installed (potentially malicious)
    module instead of the staged runtime — bypassing the
    ``NORA_RUN_TOKEN`` env-pop and the authenticity-token machinery
    the runtime owns. The preamble must therefore order the inserts
    so ``lib_dir`` ends up at ``sys.path[0]``.

    The same ordering also ensures ``pkg_dir`` sits after stdlib (via
    ``append`` rather than ``insert(0, ...)``), so an installed
    package can't shadow ``os`` / ``json`` / other stdlib modules.
    """
    monkeypatch.setenv("NORA_PYTHON_PKG_BASE", str(tmp_path / "pkgs"))
    from nora.executor import _write_script
    from nora.package_installer import nora_python_pkg_dir
    from nora.env_detect import find_python

    py_tool = find_python()
    if py_tool is None:
        pytest.skip("python3 not on PATH in this test env")

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    lib_dir = run_dir / "lib"
    lib_dir.mkdir()
    script_path = _write_script(run_dir, "Python", "print('hi')\n")

    text = script_path.read_text(encoding="utf-8")
    pkg_dir = str(nora_python_pkg_dir(py_tool.binary))
    # Both paths must be in the preamble.
    assert str(lib_dir) in text
    assert pkg_dir in text
    # ``lib_dir`` must be inserted at the front of sys.path (so it
    # wins over everything else). The literal we check for is the
    # ``insert(0, ...)`` form on lib_dir, paired with anything other
    # than ``insert(0, ...)`` on pkg_dir.
    assert f"insert(0, {str(lib_dir)!r})" in text, (
        "lib_dir must use insert(0, ...) so the staged runtime wins"
    )
    assert f"insert(0, {pkg_dir!r})" not in text, (
        "pkg_dir must NOT use insert(0, ...) — that would shadow "
        "lib_dir's nora.py with any package masquerading as 'nora', "
        "and shadow stdlib modules too"
    )


def test_sandbox_profile_grants_read_on_nora_pkg_dir(
    tmp_path, monkeypatch,
) -> None:
    """The other half of the alignment: even with the preamble's
    sys.path entry, the sandbox-exec read allowlist would still deny
    a script's import of ``nora_python_pkg_dir`` content. The
    executor must include it in ``extra_read_paths`` for Python
    runs."""
    monkeypatch.setenv("NORA_PYTHON_PKG_BASE", str(tmp_path / "pkgs"))
    from nora.executor import _sandbox_profile_string
    from nora.package_installer import nora_python_pkg_dir
    from nora.env_detect import find_python

    py_tool = find_python()
    if py_tool is None:
        pytest.skip("python3 not on PATH in this test env")

    pkg_dir = str(nora_python_pkg_dir(py_tool.binary))
    profile = _sandbox_profile_string(
        run_dir=tmp_path / "run",
        cwd=tmp_path / "cwd",
        home=tmp_path,
        extra_read_paths=(pkg_dir,),
    )
    # Path appears as a quoted ``(subpath …)`` entry in the read
    # allowlist; check the substring so we don't couple to the
    # exact escape form.
    assert pkg_dir in profile


# ---------------------------------------------------------------------------
# nora_python_pkg_dir — the new single source of truth
# ---------------------------------------------------------------------------

def test_nora_python_pkg_dir_namespaces_by_version(tmp_path, monkeypatch) -> None:
    """C-extension wheels (numpy, scipy, …) are tagged for a specific
    CPython ABI. Mixing 3.11 and 3.12 wheels in one ``--target`` dir
    crashes at import. Per-version subdir avoids this."""
    monkeypatch.setenv("NORA_PYTHON_PKG_BASE", str(tmp_path))
    import sys
    from nora.package_installer import nora_python_pkg_dir

    d = nora_python_pkg_dir(sys.executable)
    expected_tag = f"{sys.version_info.major}.{sys.version_info.minor}"
    assert d.parent == tmp_path
    assert d.name == expected_tag


# ---------------------------------------------------------------------------
# Subprocess env scrub — installer must not leak parent secrets
# ---------------------------------------------------------------------------

def test_install_subprocess_env_drops_parent_secrets(
    tmp_path, monkeypatch,
) -> None:
    """``install_packages`` runs pip / R / Stata installers with
    network access AND OUTSIDE the analysis sandbox, so any secret
    in the parent process env (API keys, AWS creds) is reachable
    by the installer's post-install hooks. The fix mirrors the
    executor's allowlist: the subprocess inherits only the env
    vars on ``_SUBPROCESS_ENV_ALLOWLIST``. Verify the symmetric
    contract by intercepting ``subprocess.run`` and inspecting
    the ``env`` argument it would have received.
    """
    import asyncio
    import sys as _sys

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-leaktest-anthropic")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-leaktest-openai")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "leaktest-aws")
    monkeypatch.setenv("PATH", os.environ.get("PATH", "/usr/bin"))
    monkeypatch.setenv("NORA_PYTHON_PKG_BASE", str(tmp_path / "pkgs"))

    captured: dict[str, dict[str, str] | None] = {"env": None}

    # Capture the install Popen call ONLY. ``_python_version_tag``
    # (called from ``nora_python_pkg_dir``) also uses subprocess.run
    # internally (which uses Popen under the hood); we let those
    # passthrough to the real Popen so the version probe still works.
    import subprocess as _sp
    real_popen = _sp.Popen

    class _FakePopen:
        def __init__(self, cmd, **kwargs):  # type: ignore[no-untyped-def]
            self._cmd = list(cmd) if isinstance(cmd, list) else cmd
            self.pid = -1
            self.returncode = 0
            # Only mark the install env — let probe calls go through
            # real Popen by raising NotImplementedError up through
            # the dispatch wrapper.
            captured["env"] = dict(kwargs.get("env") or {})

        def communicate(self, input=None, timeout=None):  # type: ignore[no-untyped-def]
            return ("", "")

        def kill(self) -> None:  # pragma: no cover
            pass

    def _dispatch_popen(cmd, **kwargs):  # type: ignore[no-untyped-def]
        # Identify the install command (pip / Rscript / Stata
        # install) — those have ``start_new_session=True`` set,
        # which the executor's run_script / package_installer use
        # but the version probe does not.
        if kwargs.get("start_new_session"):
            return _FakePopen(cmd, **kwargs)
        return real_popen(cmd, **kwargs)

    monkeypatch.setattr(_sp, "Popen", _dispatch_popen)
    # Run the install branch synchronously so the test doesn't
    # depend on the asyncio thread-pool executor seeing our patched
    # Popen (which it does, but reasoning about the capture order
    # is simpler this way).
    async def _sync_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)
    monkeypatch.setattr(asyncio, "to_thread", _sync_to_thread)

    # Patch ``detect_environment`` to a fixed fake Environment.
    # Without this, the real env_detect calls ``subprocess.run`` to
    # probe interpreters — but ``run`` is patched to a no-op
    # returning empty output, so the probe sees no python and
    # ``install_packages`` bails before reaching the audited path.
    # ``install_packages`` imports ``detect_environment`` lazily, so
    # the patch lives on ``nora.env_detect`` (the source module),
    # not on ``nora.package_installer``.
    import nora.env_detect as _env_detect
    from nora.env_detect import Environment, Tool

    fake_env = Environment(
        python=Tool(
            name="Python", binary="/usr/bin/python3",
            version="Python 3.12.0",
            missing_packages=(), optional_missing_packages=(),
            extra_read_paths=(),
        ),
        r=None, stata=None, sandbox_exec=None,
    )
    monkeypatch.setattr(_env_detect, "detect_environment", lambda: fake_env)

    from nora.package_installer import install_packages

    asyncio.run(install_packages(
        language="Python", packages=["pandas"], action="install",
    ))

    env = captured["env"]
    assert env is not None, "subprocess.Popen was not called with an env dict"
    # The three secrets we set MUST NOT cross to the installer.
    for leak in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "AWS_SECRET_ACCESS_KEY"):
        assert leak not in env, (
            f"{leak} leaked into installer env — package post-install "
            f"hooks run outside the analysis sandbox with network "
            f"access, so this is a credential-exfil channel"
        )
    # PATH (allowlisted) survives so pip can find python tooling.
    assert "PATH" in env
    # PYTHONPATH is the install-target injection — must point at our
    # nora pkg dir so ``pip uninstall`` resolves the right copy.
    assert "PYTHONPATH" in env
    assert str(tmp_path / "pkgs") in env["PYTHONPATH"]
