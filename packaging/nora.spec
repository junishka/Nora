# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec — builds the self-contained Nora binary.

The bundle entry is ``__main__.py`` — a thin shim that calls
``nora.ui.main`` to bring up the pywebview-based UI directly.
Researchers double-click Nora.app and get the chat window with no
Terminal popup.

Produces ``dist/nora/`` with the ``nora`` executable plus every
dependency (Python runtime, pandas, pyreadstat, claude-agent-sdk,
pywebview, etc.) so researchers don't need Python/uv/pip installed.

Run from the repo root:
    uv run pyinstaller packaging/nora.spec --clean --noconfirm

What needs explicit handling here and cannot be inferred by
PyInstaller:

- Runtime libraries (`nora.R`, `nora_result_*.ado`) are data
  files, not Python modules — PyInstaller won't pick them up
  without an explicit `datas` entry.
- Web UI assets (HTML, JS, CSS, the cat-loading Lottie JSON, the
  bundled lottie-player) live under ``src/nora/web/``; same
  story, listed in `datas`.
- ``pyreadstat`` is a C extension with a subdivided module layout;
  the default import scan sometimes misses submodules. Listed
  explicitly in ``hiddenimports``.
- ``claude_agent_sdk`` has dynamic imports for transport plugins
  (e.g. subprocess-based Claude Code CLI). Listed explicitly so
  they survive the tree-shake.
- ``webview`` (pywebview) loads platform backends dynamically;
  ``webview.platforms.cocoa`` is the macOS one and won't be
  picked up by static analysis.
"""

from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules


# Repo root — PyInstaller runs with the spec file as its working ref,
# so we resolve relative to the spec's directory.
REPO_ROOT = Path(SPECPATH).parent  # type: ignore[name-defined]  # SPECPATH from PyInstaller

# UI entry. The shim re-exports ``nora.ui.main`` so ``uv run nora``
# (console-script) and ``python -m nora`` (package entry) both end up
# at the same place; PyInstaller targets the file rather than a
# console-script so it doesn't need entry-point metadata at build time.
ENTRY = str(REPO_ROOT / "src" / "nora" / "__main__.py")

# Runtime libraries (R + Stata + Python) are loaded via
# ``importlib.resources`` inside ``executor._stage_runtime``.
# PyInstaller preserves the package layout when we list them
# explicitly as data files.
#
# Glob, don't enumerate. The previous hand-maintained list missed
# 8 of the 13 .ado helpers (correlation, every plot helper,
# safe_export, plot_export, the standalone ttest) plus the Python
# runtime ``nora.py`` — every Stata script that used a plot helper
# or correlation, and every Python script entirely, crashed in
# .app builds with FileNotFoundError because
# ``importlib.resources.files("nora.runtime").joinpath(name)
# .read_text()`` returned nothing for the un-bundled files. The
# dev install (pip / uv from source) worked because the package
# dir on disk had every file; only the PyInstaller bundle was
# missing them. Globbing closes the door on this regression class:
# any new helper dropped into ``runtime/`` ships with the build
# without a spec edit. We include .py too because:
#   - ``__init__.py`` is required for ``importlib.resources.files
#     ("nora.runtime")`` to resolve as a package
#   - ``nora.py`` is the Python user-runtime that the executor
#     stages into every Python script's ``lib_dir`` and the user
#     script then imports — it has to be readable as a *file
#     resource*, not just importable as a module (PyInstaller's
#     bytecode-only archive doesn't satisfy resources.files's
#     ``.read_text()`` call)
#   - any other Python helper module in ``runtime/`` is also
#     picked up by PyInstaller's normal tree-shake; listing it as
#     data is harmless redundancy.
# Hidden / cache files (``__pycache__``, ``.DS_Store``) are
# skipped by ``is_file()`` + dot-prefix filter.
RUNTIME_DIR = REPO_ROOT / "src" / "nora" / "runtime"
RUNTIME_DATAS = [
    (str(p), "nora/runtime")
    for p in sorted(RUNTIME_DIR.iterdir())
    if p.is_file() and not p.name.startswith(".")
]

# Web UI assets (HTML + JS + CSS + the bundled Lottie player + the
# cat-loading animation JSON). The UI shell loads these from
# ``Path(__file__).parent / "web"`` at runtime — see ui.py near
# `webview.create_window`. PyInstaller's static import scan can't see
# static asset files; without an explicit datas entry the bundle
# ships without the UI.
#
# Globbed dynamically so anything dropped into web/ (a future asset,
# a different Lottie animation, etc.) gets bundled without spec
# edits — the rule is "everything in web/ goes into web/ in the
# bundle". Hidden files (``.DS_Store`` etc.) are skipped.
WEB_DIR = REPO_ROOT / "src" / "nora" / "web"
WEB_DATAS = [
    (str(p), "nora/web")
    for p in sorted(WEB_DIR.iterdir())
    if p.is_file() and not p.name.startswith(".")
]

# Hidden imports — things PyInstaller's static scan may miss because
# they're loaded dynamically.
HIDDEN_IMPORTS = [
    *collect_submodules("claude_agent_sdk"),
    *collect_submodules("pyreadstat"),
    # pywebview loads its window backend at runtime via
    # ``webview.platforms.<platform>``; the cocoa one is the macOS
    # backend. Without these the bundle starts up and immediately
    # complains that no GUI toolkit is available.
    *collect_submodules("webview"),
    "webview.platforms.cocoa",
    # pandas and numpy are covered by PyInstaller's bundled hooks.
]


block_cipher = None

a = Analysis(
    [ENTRY],
    pathex=[str(REPO_ROOT / "src")],
    binaries=[],
    datas=RUNTIME_DATAS + WEB_DATAS,
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
    name="nora",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    # Windowed (console=False): no Terminal popup when the .app
    # launcher execs the binary. pywebview opens its own native
    # WKWebView window, which is the whole point of the .app
    # path. stdout/stderr fall through to the parent process; the
    # launcher script tees them to a log file under
    # ``~/Library/Logs/Nora/`` for debugging.
    console=False,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(REPO_ROOT / "packaging" / "Nora.icns"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="nora",
)
