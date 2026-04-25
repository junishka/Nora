"""Nora — process-wide configuration and path sandboxing.

The researcher points Nora at a working directory (their project dir /
data dir). Everything Claude can touch — via any of the five MCP tools —
lives inside that directory. This module holds the cwd and provides the
resolver that rejects escape attempts.

Setting the cwd is a one-shot operation done at startup from `app.py`. Tool
implementations call `resolve_in_cwd()` with the researcher-supplied
`dataset` string (which comes through Claude) and either get back a safe
absolute `Path`, or a `PathEscapeError` they should surface to Claude as a
policy denial.
"""

from __future__ import annotations

from pathlib import Path


class PathEscapeError(ValueError):
    """Raised when a tool input resolves to a path outside the working directory."""


# Module-level state. Intentionally simple: Nora is a single-process CLI,
# not a web server. The global is fine and makes tool implementations free of
# plumbing.
_cwd: Path = Path.cwd().resolve()


def set_cwd(path: Path) -> Path:
    """Set the process-wide working directory. Called once from app startup."""
    global _cwd
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise NotADirectoryError(f"not a directory: {resolved}")
    _cwd = resolved
    return _cwd


def get_cwd() -> Path:
    return _cwd


def resolve_in_cwd(user_path: str) -> Path:
    """Resolve a researcher/Claude-supplied path within the working directory.

    Accepts either a relative path (resolved against the cwd) or an absolute
    path (must be within the cwd). Follows symlinks via .resolve(). Rejects
    anything that ends up outside the cwd tree.

    Security note: this is the one authoritative place paths get validated.
    Every MCP tool that takes a path argument goes through this function.
    If a new tool touches paths without calling this, that's a regression.
    """
    if not user_path:
        raise PathEscapeError("empty path")
    p = Path(user_path).expanduser()
    if not p.is_absolute():
        p = _cwd / p
    resolved = p.resolve()
    cwd = _cwd
    # `is_relative_to` was added in Python 3.9; we require 3.10+.
    if not (resolved == cwd or resolved.is_relative_to(cwd)):
        raise PathEscapeError(
            f"path {user_path!r} resolves to {resolved}, which is outside "
            f"the working directory {cwd}. Nora tools can only access "
            f"files inside the working directory."
        )
    return resolved
