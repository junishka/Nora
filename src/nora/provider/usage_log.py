"""Shared ``.nora-usage.log`` writer with size cap.

Both provider implementations (Anthropic and OpenAI) append a
diagnostic line per round when ``NORA_DEBUG_USAGE=1`` is set. Without
a cap this file grows unboundedly — one ~200-byte line per round
per turn, plus whatever experimental fields a provider's usage dict
carries that we serialize verbatim. A researcher who toggles the
flag and forgets it on can fill their session disk in a few hours
of heavy work.

This module gives both providers a single ``append_usage_line``
function that:

- Writes one line atomically (the file is opened, a string is
  written, the file is closed; no partial-line interleavings between
  threads).
- Rotates when the file passes ``_MAX_USAGE_LOG_BYTES`` (default
  1 MiB, env-overridable). Rotation is the simplest possible scheme:
  the existing file is renamed to ``.nora-usage.log.1`` (overwriting
  any prior ``.1``) and a fresh log is started. One generation is
  enough for the diagnostic use case — ``.1`` is the "what we just
  rotated out," and the live file has the recent activity. Heavier
  schemes (logging.handlers.RotatingFileHandler with N backups)
  aren't justified for a diagnostic log.

Errors (disk full, read-only filesystem, permission) are swallowed
so the diagnostic stays exactly that — a diagnostic. The original
``except Exception: pass`` posture is preserved; the size cap just
keeps the file from being the cause of the disk-full state in the
first place.
"""

from __future__ import annotations

import os
from pathlib import Path


# Default 1 MiB cap. Big enough to keep ~2-3 hours of heavy
# diagnostic activity in the live log; small enough to not be the
# largest file in a session directory by orders of magnitude. Env-
# overridable for researchers who want longer windows or are
# debugging usage drift across days.
_DEFAULT_MAX_BYTES = 1 * 1024 * 1024


def _max_bytes() -> int:
    """Resolve the cap from ``NORA_USAGE_LOG_MAX_BYTES`` if set, else
    ``_DEFAULT_MAX_BYTES``. Invalid values fall back to the default
    rather than raising — a bad env value shouldn't break diagnostic
    logging.
    """
    raw = os.environ.get("NORA_USAGE_LOG_MAX_BYTES", "")
    if not raw:
        return _DEFAULT_MAX_BYTES
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return _DEFAULT_MAX_BYTES
    return n if n > 0 else _DEFAULT_MAX_BYTES


def append_usage_line(cwd: Path, line: str) -> None:
    """Append one diagnostic line to ``<cwd>/.nora-usage.log``.

    Rotates to ``.nora-usage.log.1`` (overwriting any prior backup)
    when the live log would exceed the cap. ``line`` is written
    verbatim with a trailing newline; callers shouldn't pre-append
    one.

    Errors (disk full, read-only, permission denied, missing parent)
    are swallowed. The whole point of the diagnostic is "see what
    the provider reported"; a writer that crashes the turn defeats
    that.
    """
    log_path = cwd / ".nora-usage.log"
    backup_path = cwd / ".nora-usage.log.1"
    payload = line + "\n"
    try:
        # Pre-flight rotation: check current size BEFORE the open()
        # so we don't end up with a single line that itself blows
        # the cap by a factor that depends on Python's buffer flush
        # timing. A line is at most a few hundred bytes; ~1 MiB cap
        # makes rotation a per-thousand-line event.
        cap = _max_bytes()
        try:
            current = log_path.stat().st_size
        except OSError:
            current = 0
        if current + len(payload) > cap and current > 0:
            try:
                # Atomic on POSIX; on Windows, ``replace`` works the
                # same way for our purposes (overwrites destination).
                os.replace(log_path, backup_path)
            except OSError:
                # Rotation failed (probably permission). Keep going —
                # we'd rather a stale-but-bounded log than crash.
                pass
        with log_path.open("a") as f:
            f.write(payload)
    except Exception:  # noqa: BLE001 — diagnostic must never crash a turn
        return
