"""Runtime libraries (R, Stata) bundled as package data.

The executor locates these via `importlib.resources.files('nora.runtime')`
at run time and copies/sources them into the per-run scratch directory
before invoking the researcher's script.
"""
