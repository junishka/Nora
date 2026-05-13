"""Regression tests for the runtime-detection probes' safety posture.

Two attack surfaces are covered:

1. The Python package-installed probe used to call ``__import__(pkg)``,
   which executes the package's ``__init__.py`` OUTSIDE the analysis
   sandbox at startup and after every install. A package masquerading
   as ``pandas`` / ``statsmodels`` got code execution during detection.
   ``importlib.util.find_spec`` consults sys.path finders only and
   doesn't import the package, so no ``__init__.py`` runs.

2. The probes inherited ``PYTHONPATH`` / ``sitecustomize.py`` from
   the parent process, which let attacker-controlled startup code
   execute on every probe. ``-I`` (isolated mode) tells the
   interpreter to ignore PYTHON* env vars and user-site
   ``usercustomize.py``. Combined with explicit ``sys.path.insert``
   inside the probe code (so the Nora package dir still resolves
   without going through PYTHONPATH), the probes are immune to both
   injection paths.

The R probe equivalent — ``requireNamespace(pkg)`` loaded package
namespaces, firing ``.onLoad`` hooks. ``nzchar(system.file(package=))``
is a metadata check that doesn't load code. Covered here too in the
form that doesn't need a real Rscript on the test machine.
"""

from __future__ import annotations

import os
from pathlib import Path

from nora.env_detect import (
    _python_missing_packages,
    _python_prefixes,
    _python_version,
)
from nora.package_installer import _python_version_tag


def _python3() -> str:
    """Resolve the test machine's python3 — skip if missing."""
    import shutil
    p = shutil.which("python3")
    if p is None:
        import pytest
        pytest.skip("python3 not on PATH")
    return p


def test_find_spec_does_not_execute_fake_package(
    tmp_path: Path, monkeypatch
) -> None:
    """The Python missing-packages probe must NOT execute a fake
    package's ``__init__.py``. Verify by putting a sentinel package
    in ``nora_python_pkg_dir`` whose ``__init__.py`` writes a canary
    file. If the probe imports the package, the canary appears.
    """
    python = _python3()
    pkg_base = tmp_path / "pkg_base"
    monkeypatch.setenv("NORA_PYTHON_PKG_BASE", str(pkg_base))

    # The probe uses ``nora_python_pkg_dir(binary)`` which appends
    # the python version tag. Create the package under that
    # version-tagged dir.
    from nora.package_installer import nora_python_pkg_dir
    target_dir = nora_python_pkg_dir(python)
    target_dir.mkdir(parents=True, exist_ok=True)

    canary = tmp_path / "canary.txt"
    fake_pkg = target_dir / "tripwire_pkg"
    fake_pkg.mkdir()
    (fake_pkg / "__init__.py").write_text(
        f"# this should NEVER execute during the probe\n"
        f"open({str(canary)!r}, 'w').write('IMPORT EXECUTED')\n",
        encoding="utf-8",
    )

    missing = _python_missing_packages(python, ("tripwire_pkg",))
    # The probe must find the package (it's on the constructed path)
    # but must NOT have executed its __init__.py.
    assert missing == (), (
        "find_spec should report tripwire_pkg as installed"
    )
    assert not canary.exists(), (
        "__init__.py ran during probe — find_spec is not behaving "
        "as a non-loading check"
    )


def test_isolated_mode_ignores_inherited_pythonpath_sitecustomize(
    tmp_path: Path, monkeypatch
) -> None:
    """The probes run with ``-I`` so an attacker-controlled
    ``PYTHONPATH`` cannot inject a ``sitecustomize.py`` that
    executes on every probe. Set PYTHONPATH to a temp dir with a
    poisoned sitecustomize.py and verify the canary does not fire.
    """
    python = _python3()
    poisoned = tmp_path / "poisoned"
    poisoned.mkdir()
    canary = tmp_path / "sitecustom_canary.txt"
    (poisoned / "sitecustomize.py").write_text(
        f"# would run at every python startup if PYTHONPATH were honored\n"
        f"open({str(canary)!r}, 'w').write('SITECUSTOMIZE EXECUTED')\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PYTHONPATH", str(poisoned))

    # All three probes touch python -c paths that previously
    # inherited PYTHONPATH. None should fire the canary.
    _python_missing_packages(python, ("os",))
    _python_prefixes(python)
    _python_version_tag(python)
    # _python_version uses --version (short-circuits site loading)
    # but exercise it for completeness.
    _python_version(python)

    assert not canary.exists(), (
        f"sitecustomize.py ran via inherited PYTHONPATH: "
        f"isolated mode (-I) is not effective"
    )


def test_filter_env_strips_secrets_from_probe(
    tmp_path: Path, monkeypatch
) -> None:
    """The probes run with the executor's ``_filter_env`` so
    parent-process secrets don't reach the subprocess. Test by
    setting a fake credential and asserting the probe's child
    can't see it.
    """
    python = _python3()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-DO-NOT-LEAK")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-aws-DO-NOT-LEAK")

    # The probes don't return env state directly, but we can verify
    # by checking the filter contract head-on: every probe constructs
    # its env from _filter_env, and _filter_env's allowlist is the
    # canonical surface.
    from nora.executor import _filter_env
    filtered = _filter_env({
        "PATH": "/usr/bin",
        "ANTHROPIC_API_KEY": "sk-test-DO-NOT-LEAK",
        "AWS_SECRET_ACCESS_KEY": "test-aws-DO-NOT-LEAK",
    })
    assert "ANTHROPIC_API_KEY" not in filtered
    assert "AWS_SECRET_ACCESS_KEY" not in filtered
    # PATH should survive — interpreters need it.
    assert "PATH" in filtered


def test_r_probe_uses_system_file_not_requirenamespace() -> None:
    """The R probe must not call ``requireNamespace``, which loads
    the package namespace and fires ``.onLoad``. ``system.file`` is
    a metadata check. Inspect the constructed expression directly.
    """
    # Construct the same expression the probe builds, so we can
    # assert on its text without running Rscript.
    from nora import env_detect
    import subprocess

    captured: dict = {}

    def fake_run(*args, **kwargs):
        captured["argv"] = args[0]
        # Return a stub with empty stdout so the probe falls into
        # the "all missing" path.
        class _Out:
            stdout = ""
            stderr = ""
            returncode = 1
        return _Out()

    orig = subprocess.run
    subprocess.run = fake_run  # type: ignore[assignment]
    try:
        env_detect._r_missing_packages("Rscript", ("haven", "ggplot2"))
    finally:
        subprocess.run = orig  # type: ignore[assignment]

    argv = captured.get("argv") or []
    assert any("system.file" in str(a) for a in argv), (
        f"R probe should use system.file (non-loading), got argv={argv}"
    )
    assert not any("requireNamespace" in str(a) for a in argv), (
        f"R probe must not use requireNamespace (loads .onLoad), got argv={argv}"
    )
