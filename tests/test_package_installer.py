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

    def fake_run(*args, **kwargs):
        captured["env"] = dict(kwargs.get("env") or {})

        class _R:
            returncode = 0
            stdout = ""
            stderr = ""
        return _R()

    import subprocess as _sp
    monkeypatch.setattr(_sp, "run", fake_run)
    # Run the install branch synchronously so the test doesn't
    # depend on the asyncio thread-pool executor seeing our patched
    # ``subprocess.run`` (which it does, but reasoning about the
    # capture order is simpler this way).
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
    assert env is not None, "subprocess.run was not called with an env dict"
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
