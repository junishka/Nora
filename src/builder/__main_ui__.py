"""Entry point for the .app-bundled web UI build.

Distinct from ``__main__.py`` (which launches the terminal CLI). The
PyInstaller spec uses this module as its entry so the bundled binary
inside Builder.app brings up the pywebview window directly. From-source
users still get the terminal via ``uv run builder``; the .app gives the
web UI without a Terminal popup.
"""

from builder.ui import main


if __name__ == "__main__":
    main()
