"""Build a short, human-readable ``debug_excerpt`` of a failed script run
for the model to read on the next turn.

Why this exists
---------------
Before this module, when a researcher's R / Stata / Python script
failed, the model's tool result carried only ``status``, ``reason``
(a one-liner like "exit code 1"), and a hint to call
``expand_result`` - which itself has no raw-log access. The model
was effectively blind: it would re-iterate ad nauseum, propose
plausible-but-wrong fixes, and the researcher waited.

The fix is a new ``debug_excerpt`` field on the failure tool result:
~500-1000 chars of the actual error message, extracted from the
language's own error idiom. Researcher-style: not "KeyError raised
at line 42", but the literal `KeyError: 'a_yp0'` Nora needs to
know which column was missing.

SDC boundary
------------
This is the FIRST channel that ever forwards bytes from raw
stdout/stderr to the model. The boundary is preserved by:

  1. **Tightly anchored patterns.** We only forward what matches a
     known error idiom (R's "Error in ... :" block; Python's last
     traceback frame + exception line; Stata's "r(<code>);" with
     the echoed command above it). Arbitrary text never crosses.
  2. **stdout is read only for Stata** (because Stata batch mode
     puts everything in the .log file, which the executor merges
     into stdout). For R and Python, only stderr is scanned.
  3. **No `print(df)` leakage.** Because Stata's extractor anchors
     on `r(<code>);` and walks back to the most recent `. <cmd>`,
     intervening `display` / `list` output stays out of the
     excerpt. The 1 KB hard cap is the second line of defense.
  4. **Length-aware redaction.** Quoted args longer than 200 chars
     get truncated in place - covers the "ValueError with a 5KB
     pandas repr" foot-gun.
  5. **Credential scrub.** Regex out `sk-...`, `AKIA...`,
     three-segment JWTs. Catches the `print(os.environ)` foot-gun.
  6. **Path normalisation.** Absolute paths get reduced to their
     basename so the home-directory layout doesn't leak. Line
     numbers are preserved - that's what the model actually needs.

If extraction misses, we return ``None`` and the caller falls
back to a generic "script failed; inspect raw log in UI"
message. Better silent than wrong.

The extractor is exercised by ``test_error_summary.py`` (happy
paths) and ``test_error_summary_no_leak.py`` (SDC regression
tests with planted secrets in stdout / exception args).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional


# Hard caps. Intentionally generous on the per-arg side (so a
# KeyError with a 50-char column name passes through verbatim) and
# tight overall (so a single failure can't blow the budget).
MAX_EXCERPT_BYTES = 1000
MAX_QUOTED_ARG_BYTES = 200


# -----------------------------------------------------------------
# Public entry
# -----------------------------------------------------------------

def extract_debug_excerpt(
    stdout: str,
    stderr: str,
    exit_code: int | None,
    language: str,
) -> Optional[str]:
    """Return a short, human-readable error excerpt for the model
    or ``None`` if extraction couldn't locate a recognisable error.

    ``language`` is one of ``"R"`` / ``"Stata"`` / ``"Python"``.
    Casing is preserved as the executor uses it.

    Stdout is only consulted for Stata; for R and Python the
    excerpt comes from stderr exclusively. This keeps the SDC
    surface narrower (researchers ``print(df)`` in stdout, almost
    never in stderr).
    """
    raw: Optional[str] = None
    if language == "Python":
        raw = _extract_python(stderr or "")
    elif language == "R":
        raw = _extract_r(stderr or "")
    elif language == "Stata":
        # Stata's batch-mode log is merged into stdout by the
        # executor; stderr is essentially empty. The Stata
        # extractor is anchored on r(<code>); + command echo so
        # `display` / `list` content cannot bleed into the excerpt.
        raw = _extract_stata(stdout or "")
    if not raw:
        return None
    return _scrub_and_cap(raw)


# -----------------------------------------------------------------
# Python
# -----------------------------------------------------------------

# The last `File "..."`-prefixed line in a traceback marks the
# user-code frame at the point of failure. Capture its body so we
# can include it alongside the exception line.
_PY_FRAME_RE = re.compile(
    r'^\s*File "(?P<path>[^"]+)", line (?P<line>\d+)'
    r'(?:, in (?P<func>[^\n]+))?\s*$',
    re.MULTILINE,
)

# Python exception lines look like "ExceptionType: message" at the
# bottom of the traceback. The type is a dotted identifier; the
# message can include quotes, parens, anything.
_PY_EXC_RE = re.compile(
    r'^(?P<type>[A-Za-z_][\w\.]*(?:Error|Exception|Warning|Exit|Stop[a-zA-Z]*))'
    r'(?:: (?P<msg>.*))?$',
    re.MULTILINE,
)


def _extract_python(stderr: str) -> Optional[str]:
    """Pull the last user-code frame + final exception line.

    A typical Python traceback looks like::

        Traceback (most recent call last):
          File "/abs/path/script.py", line 17, in <module>
            df['typo']
          File "/.../pandas/core/frame.py", line 4090, in __getitem__
            indexer = self.columns.get_loc(key)
          File "/.../pandas/core/indexes/base.py", line 3812, in get_loc
            raise KeyError(key) from err
        KeyError: 'typo'

    We want: ``File "script.py", line 17, in <module>`` + the
    indented source line below it (if present) + ``KeyError: 'typo'``.
    Library frames (pandas internals) are dropped - they're noise
    and the path strip would lose context anyway.
    """
    frames = list(_PY_FRAME_RE.finditer(stderr))
    if not frames:
        # Bare exception (e.g. `raise SystemExit("x")` with no
        # traceback formatting). Still try to pluck the last
        # exception line.
        exc = _last_python_exception_line(stderr)
        return exc

    # Walk frames newest-first, drop site-packages / stdlib paths,
    # keep the first user-code one. "User code" heuristic:
    # any path that does NOT live under a site-packages /
    # dist-packages / typeshed / .venv / lib/python segment.
    LIB_PAT = re.compile(
        r"(?:/site-packages/|/dist-packages/|/lib/python[\d\.]+/"
        r"|/typeshed/|/\.venv/|/python\d+\.\d+/lib/)"
    )
    user_frame = None
    for m in reversed(frames):
        if not LIB_PAT.search(m.group("path")):
            user_frame = m
            break
    # If everything looks like library code (rare), still take the
    # newest frame - better some location than none.
    if user_frame is None:
        user_frame = frames[-1]

    # Grab the source-line below the frame (Python's traceback
    # formatter indents it by 4 spaces). The line might be absent
    # for tracebacks emitted via formatter overrides; that's fine.
    source_line = ""
    after = stderr[user_frame.end():]
    nl = after.find("\n")
    if nl != -1:
        candidate = after[nl + 1:].split("\n", 1)[0]
        if candidate.startswith("    ") and candidate.strip():
            source_line = candidate.strip()

    exc_line = _last_python_exception_line(stderr) or ""

    parts: list[str] = []
    path = Path(user_frame.group("path")).name
    line = user_frame.group("line")
    func = user_frame.group("func") or ""
    in_func = f", in {func}" if func else ""
    parts.append(f'File "{path}", line {line}{in_func}')
    if source_line:
        parts.append(f"    {source_line}")
    if exc_line:
        parts.append(exc_line)
    return "\n".join(parts) if parts else None


def _last_python_exception_line(stderr: str) -> Optional[str]:
    """Find the last line that matches the ExceptionType[: msg]
    shape. Python tracebacks always end with this line, and there
    can be multiple in a chained exception ("During handling of
    the above..."). The LAST one is the one that propagated."""
    matches = list(_PY_EXC_RE.finditer(stderr))
    if not matches:
        return None
    # Filter to lines that look like a top-level exception, not
    # something inside a code-line preview. Code preview lines are
    # indented; exception lines are flush-left.
    flush = [m for m in matches if not stderr[max(m.start() - 1, 0)].isspace()]
    chosen = flush[-1] if flush else matches[-1]
    msg = chosen.group("msg") or ""
    # Truncate a single dumpy arg in place. Length-aware: if the
    # whole message body is huge (a 5KB pandas repr), keep only
    # the head.
    if len(msg) > MAX_QUOTED_ARG_BYTES:
        msg = msg[:MAX_QUOTED_ARG_BYTES] + "…[truncated]"
    return f"{chosen.group('type')}: {msg}" if msg else chosen.group("type")


# -----------------------------------------------------------------
# R
# -----------------------------------------------------------------

# R's standard error idiom. The "Error in <call> :" prefix is
# emitted by R for every uncaught condition. The message body
# follows on the same line and may wrap onto continuation lines
# that are indented with whitespace (R's own formatter does this).
# The block ends at the first line that starts at column zero with
# a non-whitespace character - that's either the next "Error" /
# "Calls:" / "Execution halted" / "In addition:" / "Warning
# messages:" delimiter, or the next command / blank.
_R_ERROR_RE = re.compile(
    r"^Error(?: in (?P<call>.+?))? ?: ?(?P<msg>[^\n]*(?:\n[ \t]+[^\n]*)*)",
    re.MULTILINE,
)
# The "Calls:" chain that R prints right after the error message
# tells you the call site (`Calls: lm -> eval -> ...`). Useful for
# Nora to know which function blew up.
_R_CALLS_RE = re.compile(r"^Calls: .*$", re.MULTILINE)


def _extract_r(stderr: str) -> Optional[str]:
    """Pull the LAST ``Error in ... :`` block plus its ``Calls:``
    trailer if present.

    R errors look like::

        Error in eval(predvars, data, env) :
          object 'wage' not found
        Calls: lm -> eval -> eval

    We keep the whole block, including the "Calls:" chain, and
    drop "Execution halted" / "In addition: Warning..." trailers
    (they're rarely the cause and just spend budget).
    """
    matches = list(_R_ERROR_RE.finditer(stderr))
    if not matches:
        return None
    last = matches[-1]
    block = last.group(0).strip()

    # Look for a Calls: line in the slice immediately after the
    # error block (within the next 200 chars - the chain is always
    # right there).
    tail_window = stderr[last.end(): last.end() + 200]
    calls = _R_CALLS_RE.search(tail_window)
    if calls:
        block = block + "\n" + calls.group(0).strip()
    return block


# -----------------------------------------------------------------
# Stata
# -----------------------------------------------------------------

# Stata's batch-mode log echoes each command with a leading "." and
# prints the error message + "r(<code>);" on failure. The actual
# numeric exit-of-do-file `r(<code>);` appears at the bottom; the
# command-attached one is the line that triggered the abort.
_STATA_RC_RE = re.compile(r"^r\((?P<code>\d+)\);\s*$", re.MULTILINE)
# Echoed commands start with ". " at column zero in the .log.
_STATA_CMD_RE = re.compile(r"^\. (?P<cmd>.+?)\s*$", re.MULTILINE)
# The "end of do-file" trailer is just noise - it always follows
# the real error and only adds a duplicate r(<code>); line we
# want to ignore.
_STATA_EOF_RE = re.compile(r"^end of do-file\s*$", re.MULTILINE)


def _extract_stata(stdout: str) -> Optional[str]:
    """Pull the failing command + error message + r(<code>); line.

    Stata batch-mode log on a failed run looks like::

        . regress y x_missing
        variable x_missing not found
        r(111);

        end of do-file

        r(111);

    The closing ``r(111);`` after "end of do-file" is just an exit
    echo; the meaningful one is the inline ``r(111);`` that
    appears immediately after the error message. We anchor on the
    FIRST ``r(<code>);`` (after stripping the trailing exit echo)
    and walk back to the last ``. <cmd>`` line above it.
    """
    if not stdout:
        return None

    # Strip the trailing "end of do-file" + exit-echo so we don't
    # match the wrong r(<code>);.
    eof = _STATA_EOF_RE.search(stdout)
    scan = stdout[:eof.start()] if eof else stdout

    rc_matches = list(_STATA_RC_RE.finditer(scan))
    if not rc_matches:
        return None
    rc = rc_matches[-1]  # last error inside the executable region
    rc_code = rc.group("code")

    # Walk back from the rc line to find the most recent command
    # echo (`. <cmd>`). Anything between that echo and the rc line
    # is the error message - usually 1-4 lines.
    region = scan[:rc.start()]
    cmd_matches = list(_STATA_CMD_RE.finditer(region))
    if not cmd_matches:
        # No command echo found - return just the error message
        # bounded by the previous blank line, falling back to the
        # last 6 lines if no clear delimiter.
        lines = region.rstrip("\n").splitlines()
        # Walk back collecting non-blank lines until a blank one.
        msg_lines: list[str] = []
        for ln in reversed(lines):
            if not ln.strip():
                if msg_lines:
                    break
                continue
            msg_lines.append(ln)
            if len(msg_lines) >= 6:
                break
        msg_lines.reverse()
        msg = "\n".join(msg_lines)
        return (msg + f"\nr({rc_code});").strip() or None

    cmd = cmd_matches[-1]
    # Slice from the command echo to the rc line; that's the full
    # logical block.
    block = scan[cmd.start(): rc.end()].rstrip()
    return block


# -----------------------------------------------------------------
# Common scrub / cap
# -----------------------------------------------------------------

# Reduce absolute paths in messages / tracebacks to their basename.
# We keep "line N" markers because line numbers are exactly what
# the model needs. Two patterns: quoted Python-traceback paths and
# bare absolute paths in error messages.
_QUOTED_ABS_PATH_RE = re.compile(r'"((?:/[^"\n]+))"')
_BARE_ABS_PATH_RE = re.compile(r"(?<![\w/])/[A-Za-z0-9_.\-/]{4,}")

# Credential patterns we always strip. Scoped tight enough that
# they don't false-positive on column names or numeric IDs:
#   - Anthropic-style:  sk-ant-...   (>=20 alphanum/dash after sk-)
#   - OpenAI-style:     sk-...       (>=20 alphanum after sk-)
#   - AWS access keys:  AKIA + 16 alnum
#   - JWTs: three base64url segments separated by dots
_CRED_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"sk-(?:ant-)?[A-Za-z0-9_\-]{20,}"),
    re.compile(r"AKIA[A-Z0-9]{16}"),
    re.compile(
        r"\beyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\b"
    ),
)
_CRED_REDACTED = "[redacted-credential]"

# Long quoted blobs in error messages - heuristic for "researcher
# accidentally embedded a row in the error". The threshold is the
# same per-arg cap we apply to Python exceptions.
_LONG_QUOTED_RE = re.compile(
    r"(['\"])((?:(?!\1).){" + str(MAX_QUOTED_ARG_BYTES) + r",})\1",
    re.DOTALL,
)


def _scrub_and_cap(text: str) -> str:
    """Normalise paths, redact credentials, trim oversize quoted
    blobs, and cap the total to ``MAX_EXCERPT_BYTES``."""
    out = text

    # Path normalisation - quoted (Python tracebacks use these).
    def _quoted_basename(m: re.Match[str]) -> str:
        return f'"{Path(m.group(1)).name}"'
    out = _QUOTED_ABS_PATH_RE.sub(_quoted_basename, out)

    # Path normalisation - bare absolute paths in free-form
    # messages (R / Stata sometimes emit these).
    def _bare_basename(m: re.Match[str]) -> str:
        return Path(m.group(0)).name
    out = _BARE_ABS_PATH_RE.sub(_bare_basename, out)

    # Credential scrub. Order matters less than completeness - each
    # pattern is non-overlapping with the others.
    for pat in _CRED_PATTERNS:
        out = pat.sub(_CRED_REDACTED, out)

    # Long-quoted blob trim. Replace the inner body with a short
    # head + truncation marker so the model knows something was
    # there but doesn't see ~5 KB of pandas repr.
    def _trim_long_quoted(m: re.Match[str]) -> str:
        quote = m.group(1)
        body = m.group(2)
        return f"{quote}{body[:64]}…[truncated {len(body) - 64} chars]{quote}"
    out = _LONG_QUOTED_RE.sub(_trim_long_quoted, out)

    # Final hard cap. Keep the head - the most informative part of
    # any error block sits at the top (R: "Error in ... :"; Python:
    # the user-code frame; Stata: the failing command echo). The
    # truncation marker is included in the budget so the cap is a
    # genuine ceiling.
    if len(out) > MAX_EXCERPT_BYTES:
        marker = "\n…[excerpt truncated]"
        out = out[:MAX_EXCERPT_BYTES - len(marker)].rstrip() + marker
    return out
