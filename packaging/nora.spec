# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec — builds the self-contained Nora binary.

The bundle entry is ``__main_ui__.py`` (the web shell), NOT
``__main__.py`` (the terminal CLI). The .app the build script
assembles around this bundle launches the pywebview-based UI
directly — researchers double-click Nora.app and get the
chat window with no Terminal popup. The terminal CLI stays
available from source via ``uv run nora``.

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

# Web UI entry. The terminal CLI's __main__.py is intentionally NOT
# the bundle entry — see the module docstring for the why. Keeping
# both Python entry-point modules in src/nora/ means
# ``uv run nora`` (CLI) and ``uv run nora-ui`` (web) both
# still work from source unchanged.
ENTRY = str(REPO_ROOT / "src" / "nora" / "__main_ui__.py")

# Runtime libraries (R + Stata) are loaded via `importlib.resources`
# inside `executor._stage_runtime`. PyInstaller preserves the package
# layout when we list them explicitly.
RUNTIME_DIR = REPO_ROOT / "src" / "nora" / "runtime"
RUNTIME_DATAS = [
    (str(RUNTIME_DIR / "nora.R"), "nora/runtime"),
    (str(RUNTIME_DIR / "nora_result_regress.ado"), "nora/runtime"),
    (str(RUNTIME_DIR / "nora_result_ttest.ado"), "nora/runtime"),
    (str(RUNTIME_DIR / "nora_result_sum.ado"), "nora/runtime"),
    (str(RUNTIME_DIR / "nora_result_tab.ado"), "nora/runtime"),
    (str(RUNTIME_DIR / "nora_result_magnitude.ado"), "nora/runtime"),
    # __init__.py so `importlib.resources.files("nora.runtime")`
    # resolves as a package resource rather than a bare directory.
    (str(RUNTIME_DIR / "__init__.py"), "nora/runtime"),
]

# Web UI assets (HTML + JS + CSS + the bundled Lottie player + the
# cat-loading animation JSON). The web shell (`nora-ui`) loads
# these from ``Path(__file__).parent / "web"`` at runtime — see
# ui.py near `webview.create_window`. PyInstaller's static import
# scan can't see static asset files; without an explicit datas
# entry the bundle ships without the UI.
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
