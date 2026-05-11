"""Nora — working-directory configuration and path sandboxing.

Every Nora MCP tool is gated by ``resolve_in_cwd()``: the
researcher/Claude-supplied path must resolve inside the active working
directory or the call is refused.

The active cwd is sourced in two layers:

1. **Per-task override** (``_cwd_var``) — a :class:`contextvars.ContextVar`
   bound by :func:`use_cwd` around any code that runs inside a specific
   session. This is what makes concurrent sessions safe: each
   :class:`~nora.runner.SessionRunner` runs its turn inside
   ``use_cwd(runner.cwd)`` so tool handlers (and any sub-tasks the SDK
   spawns) see the runner's cwd, regardless of which other runners are
   simultaneously executing turns. ContextVar binding is asyncio-task
   local — sister tasks see their own bindings.
2. **Process default** (``_cwd_default``) — the fallback used when no
   per-task override is in scope. Set by :func:`set_cwd` at startup
   and from tests. Web UI runners DO NOT update this; they bind via
   ``use_cwd`` only, so a focus switch in the UI doesn't trample tool
   execution in another session.

Anything that reads ``get_cwd()`` outside of a runner-bound context
(e.g., startup, test fixtures) gets the default. Anything inside a
runner-bound context gets the override.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Iterator


class PathEscapeError(ValueError):
    """Raised when a tool input resolves to a path outside the working directory."""


# Process-wide default. Used when no per-task ``use_cwd`` is in
# effect (startup, tests). Web UI runners override this via the
# ContextVar below; they do NOT mutate it, so concurrent runners
# don't trample each other.
_cwd_default: Path = Path.cwd().resolve()

# Per-asyncio-task override. ``ContextVar.set`` returns a Token that
# scopes the binding to the current task; sister tasks see the
# default. ``None`` means "no override — use the process default."
_cwd_var: ContextVar[Path | None] = ContextVar("nora_cwd", default=None)


def set_cwd(path: Path) -> Path:
    """Set the process-wide default working directory.

    Called from terminal startup (``ui.py``) and tests. Web UI runners
    bind their cwd via :func:`use_cwd` instead — see the module
    docstring for why.
    """
    global _cwd_default
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise NotADirectoryError(f"not a directory: {resolved}")
    _cwd_default = resolved
    return _cwd_default


def get_cwd() -> Path:
    """Return the active working directory.

    Returns the per-task override if one is in effect (set by
    :func:`use_cwd`), otherwise the process default.
    """
    val = _cwd_var.get()
    return val if val is not None else _cwd_default


@contextmanager
def use_cwd(path: Path) -> Iterator[Path]:
    """Bind the active cwd for the current asyncio task / context.

    Usage::

        with use_cwd(runner.cwd):
            # get_cwd() and resolve_in_cwd() see runner.cwd here,
            # regardless of any other concurrent runners.
            ...

    The binding is scoped to the calling task's context — sister
    tasks running concurrently see their own bindings (or the
    process default). Restores the previous value on exit.
    """
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise NotADirectoryError(f"not a directory: {resolved}")
    token = _cwd_var.set(resolved)
    try:
        yield resolved
    finally:
        _cwd_var.reset(token)


def ensure_private_nora_dir(cwd: Path) -> Path:
    """Create ``<cwd>/.nora`` (if missing) and force it to mode 0o700.

    Every Nora-owned file lives under ``.nora`` — chat history with
    verbatim user messages (including any credentials a researcher
    pasted), the SQLite results store with sanitized payloads and
    script source, the pre-SDC ``result.json`` for each run, the
    raw ``stdout.log`` / ``stderr.log`` for each script, the
    researcher-authored script source itself, and per-run scratch
    via ``TMPDIR=<run_dir>/tmp``. None of that should be readable
    by other users on the same machine.

    The default umask leaves files inside ``.nora`` at 0o644 —
    world-readable on any filesystem where the home directory's
    own mode permits traversal (HPC, NFS, university research
    servers). Gating the parent directory at 0o700 makes every
    descendant unreachable regardless of per-file modes: without
    execute on ``.nora`` itself, another user cannot ``open()``
    paths inside it even with a literal absolute path. So the
    one-line fix is also the load-bearing one — per-file chmods
    inside would just be defense-in-depth on top.

    Idempotent: returns the path whether the directory was just
    created or already existed; the chmod runs either way so a
    pre-existing 0o755 ``.nora`` from an earlier Nora run is
    upgraded.

    Errors during chmod are silently swallowed. On a filesystem
    that doesn't support POSIX modes (some network mounts, FAT
    volumes) we still want the rest of Nora to work — the
    application is just less hardened on that mount.
    """
    nora_dir = cwd / ".nora"
    nora_dir.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(nora_dir, 0o700)
    except OSError:
        # Best-effort: some network filesystems / Windows mounts
        # don't honor POSIX mode bits. Better to keep going than
        # block session creation.
        pass
    return nora_dir


def resolve_in_cwd(user_path: str) -> Path:
    """Resolve a researcher/Claude-supplied path within the working directory.

    Accepts either a relative path (resolved against the cwd) or an absolute
    path (must be within the cwd). Follows symlinks via .resolve(). Rejects
    anything that ends up outside the cwd tree.

    Security note: this is the one authoritative place paths get validated.
    Every MCP tool that takes a path argument goes through this function.
    If a new tool touches paths without calling this, that's a regression.

    Reads the cwd via :func:`get_cwd`, which honors the per-task
    override — so concurrent runners get sandboxed against THEIR cwd,
    not whichever session the UI happens to be focused on.
    """
    if not user_path:
        raise PathEscapeError("empty path")
    p = Path(user_path).expanduser()
    cwd = get_cwd()
    if not p.is_absolute():
        p = cwd / p
    resolved = p.resolve()
    # `is_relative_to` was added in Python 3.9; we require 3.10+.
    if not (resolved == cwd or resolved.is_relative_to(cwd)):
        raise PathEscapeError(
            f"path {user_path!r} resolves to {resolved}, which is outside "
            f"the working directory {cwd}. Nora tools can only access "
            f"files inside the working directory."
        )
    return resolved
