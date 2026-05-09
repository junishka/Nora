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
