"""Package entry point: ``python -m nora`` brings up the pywebview window.

The .app launcher and the ``nora`` console-script (declared in
pyproject.toml) both ultimately call ``nora.ui.main``; this module
exists so ``python -m nora`` from a source checkout takes the same
path without going through the script shim.
"""

from nora.ui import main


if __name__ == "__main__":
    main()
