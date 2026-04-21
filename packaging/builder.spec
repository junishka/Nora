# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec — builds the self-contained Builder binary.

Produces ``dist/builder/`` with the ``builder`` executable plus every
dependency (Python runtime, pandas, pyreadstat, claude-agent-sdk, etc.)
so researchers don't need Python/uv/pip installed. The ``.app``
wrapper in ``packaging/build_app.sh`` copies this bundle into
``Builder.app/Contents/Resources/builder/`` and installs an
AppleScript launcher that opens Terminal and runs it.

Run from the repo root:
    uv run pyinstaller packaging/builder.spec --clean --noconfirm

What needs explicit handling here and cannot be inferred by
PyInstaller:

- Runtime libraries (`builder.R`, `builder_result_*.ado`) are data
  files, not Python modules — PyInstaller won't pick them up
  without an explicit `datas` entry.
- ``pyreadstat`` is a C extension with a subdivided module layout;
  the default import scan sometimes misses submodules. Listed
  explicitly in ``hiddenimports``.
- ``claude_agent_sdk`` has dynamic imports for transport plugins
  (e.g. subprocess-based Claude Code CLI). Listed explicitly so
  they survive the tree-shake.
"""

from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules


# Repo root — PyInstaller runs with the spec file as its working ref,
# so we resolve relative to the spec's directory.
REPO_ROOT = Path(SPECPATH).parent  # type: ignore[name-defined]  # SPECPATH from PyInstaller

ENTRY = str(REPO_ROOT / "src" / "builder" / "__main__.py")

# Runtime libraries (R + Stata) are loaded via `importlib.resources`
# inside `executor._stage_runtime`. PyInstaller preserves the package
# layout when we list them explicitly.
RUNTIME_DIR = REPO_ROOT / "src" / "builder" / "runtime"
RUNTIME_DATAS = [
    (str(RUNTIME_DIR / "builder.R"), "builder/runtime"),
    (str(RUNTIME_DIR / "builder_result_regress.ado"), "builder/runtime"),
    (str(RUNTIME_DIR / "builder_result_ttest.ado"), "builder/runtime"),
    (str(RUNTIME_DIR / "builder_result_sum.ado"), "builder/runtime"),
    (str(RUNTIME_DIR / "builder_result_tab.ado"), "builder/runtime"),
    (str(RUNTIME_DIR / "builder_result_magnitude.ado"), "builder/runtime"),
    # __init__.py so `importlib.resources.files("builder.runtime")`
    # resolves as a package resource rather than a bare directory.
    (str(RUNTIME_DIR / "__init__.py"), "builder/runtime"),
]

# Hidden imports — things PyInstaller's static scan may miss because
# they're loaded dynamically.
HIDDEN_IMPORTS = [
    *collect_submodules("claude_agent_sdk"),
    *collect_submodules("pyreadstat"),
    # pandas and numpy are covered by PyInstaller's bundled hooks.
]


block_cipher = None

a = Analysis(
    [ENTRY],
    pathex=[str(REPO_ROOT / "src")],
    binaries=[],
    datas=RUNTIME_DATAS,
    hiddenimports=HIDDEN_IMPORTS,
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        # Shave weight — we don't ship notebooks, plotting, or tests.
        "tkinter",
        "matplotlib",
        "IPython",
        "notebook",
        "jupyter",
        "pytest",
        "hypothesis",
    ],
    noarchive=False,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="builder",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,  # Terminal app — we want stdout/stdin wired normally.
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="builder",
)
