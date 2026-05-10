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


# Overall hard caps. Generous on both axes so a verbose R error
# block or a many-frame Python traceback comes through intact — the
# model diagnoses from the FULL idiom, not a single line of it.
# The privacy guarantee comes from the tightly-anchored patterns
# (only matched error idioms forward, never arbitrary stdout); the
# cap is defense-in-depth, not the boundary itself, so it can sit
# well above typical error sizes without changing the threat model.
MAX_EXCERPT_BYTES = 8000
MAX_QUOTED_ARG_BYTES = 400
# Per-exception-body cap. Exception bodies are the one channel where
# script-controlled text crosses the SDC boundary verbatim — a
# script that calls ``raise RuntimeError(df.iloc[0].to_json())`` or
# ``stop(df$secret[1])`` would otherwise smuggle raw cell content
# through here. Legitimate exception messages ("'typo'",
# "object 'wage' not found", "[Errno 2] No such file") are well
# under 80 chars; longer bodies are usually data dumps and get
# truncated. The data-shape detector below handles the rest.
MAX_EXCEPTION_MSG_BYTES = 80


# Patterns that suggest an exception body is a data dump rather than
# a parser-owned diagnostic. Conservative on purpose — must not fire
# on common shapes like ``[Errno 2]`` or ``KeyError: ['col1','col2']``.
# The two patterns target the canonical exfil shapes:
#
#   * JSON dict from ``df.iloc[0].to_json()`` — at least one
#     ``key:value`` pair inside braces.
#   * Multi-cell row from ``df.to_csv()`` / ``str(row)`` —
#     six-or-more comma-separated tokens (low enough to catch
#     row dumps, high enough that ``KeyError: ['a','b','c','d']``
#     still passes through).
#
# Anything narrower than these patterns rides the
# ``MAX_EXCEPTION_MSG_BYTES`` cap. Documented residual risk in
# ``_scrub_exception_body``.
_DATA_SHAPED_RE = re.compile(
    r'\{[^{}\n]*:[^{}\n]*\}'             # JSON-ish dict (key:value)
    r'|(?:[^,\n]{1,40},\s*){5,}'         # 6+ comma-separated tokens
)
_REDACTED_DATA_BODY = "[message body suppressed: looked data-shaped]"


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
    """Pull the user-code call chain + final exception line.

    A typical Python traceback looks like::

        Traceback (most recent call last):
          File "/abs/path/script.py", line 17, in <module>
            run_analysis(df)
          File "/abs/path/script.py", line 9, in run_analysis
            df['typo']
          File "/.../pandas/core/frame.py", line 4090, in __getitem__
            indexer = self.columns.get_loc(key)
          File "/.../pandas/core/indexes/base.py", line 3812, in get_loc
            raise KeyError(key) from err
        KeyError: 'typo'

    We forward EVERY user-code frame (in source order) plus the final
    exception line so the model sees the full call chain. Library
    frames (site-packages, stdlib internals) are dropped — they're
    noise and the basename strip would lose their context anyway.
    Showing only the deepest user frame, as we used to, hid which
    callsite invoked the broken function and pushed the model to
    re-probe even when the chain was right there in the traceback.
    """
    frames = list(_PY_FRAME_RE.finditer(stderr))
    if not frames:
        # Bare exception (e.g. `raise SystemExit("x")` with no
        # traceback formatting). Still try to pluck the last
        # exception line.
        return _last_python_exception_line(stderr)

    # "User code" heuristic: any path that does NOT live under a
    # site-packages / dist-packages / typeshed / .venv / lib/python
    # segment.
    LIB_PAT = re.compile(
        r"(?:/site-packages/|/dist-packages/|/lib/python[\d\.]+/"
        r"|/typeshed/|/\.venv/|/python\d+\.\d+/lib/)"
    )
    user_frames = [m for m in frames if not LIB_PAT.search(m.group("path"))]
    if not user_frames:
        # All frames look like library code (rare — usually means the
        # script is a one-liner with no user-frame in the trace).
        # Fall back to the deepest frame so the model gets some
        # location instead of none.
        user_frames = [frames[-1]]

    parts: list[str] = []
    for frame in user_frames:
        path = Path(frame.group("path")).name
        line = frame.group("line")
        func = frame.group("func") or ""
        in_func = f", in {func}" if func else ""
        parts.append(f'File "{path}", line {line}{in_func}')
        # Grab the indented source-line that Python's formatter prints
        # below the frame, when present.
        after = stderr[frame.end():]
        nl = after.find("\n")
        if nl != -1:
            candidate = after[nl + 1:].split("\n", 1)[0]
            if candidate.startswith("    ") and candidate.strip():
                parts.append(f"    {candidate.strip()}")

    exc_line = _last_python_exception_line(stderr) or ""
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
    msg = _scrub_exception_body(msg)
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
    """Pull the LAST ``Error`` block's parser-owned framing plus its
    ``Calls:`` trailer if present.

    R errors look like::

        Error in eval(predvars, data, env) :
          object 'wage' not found
        Calls: lm -> eval -> eval

    The ``Error in <call> :`` portion and the message body are both
    script-controlled in the limit: ``<call>`` is R's deparse of the
    failing expression (and a script using ``do.call(fn, list(arg=
    secret))`` could put cell data in there), and the message body
    is whatever R's error formatter or ``stop()`` produced (with
    ``stop(value)`` allowing direct exfiltration of any short cell).
    We keep the ``Error`` anchor and the ``Calls:`` chain — function
    names from the call stack that the model already knows because
    it wrote the script — and discard both the call deparse and the
    body. The researcher still has the un-scrubbed message in the
    on-disk run log.
    """
    matches = list(_R_ERROR_RE.finditer(stderr))
    if not matches:
        return None
    last = matches[-1]
    # Drop both the `call` deparse AND the message body — both are
    # script-controlled in the limit, so the cleanest privacy
    # posture is to keep only the parser-owned anchor and the
    # ``Calls:`` chain (function names the model already knows
    # because it wrote the script). Earlier iterations scrubbed
    # the body via credential regexes / data-shape detection, but
    # the upstream "redact wholesale" approach is strictly safer:
    # no SSN / ID / short data value sits under any per-pattern
    # threshold the way it can in a body-included form. The
    # researcher still has the un-scrubbed message on disk.
    block = "Error : [message body redacted]"

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
    """Pull the failing command's VERB + r(<code>); line.

    Stata batch-mode log on a failed run looks like::

        . regress y x_missing
        variable x_missing not found
        r(111);

        end of do-file

        r(111);

    The closing ``r(111);`` after "end of do-file" is just an exit
    echo; the meaningful one is the inline ``r(111);`` that
    appears immediately after the error message.

    Previous versions returned the whole block from ``. <cmd>`` to
    ``r(<code>);``. That forwarded two script-controlled channels:
    the command echo (whose arguments can carry macro-expanded raw
    values — ``local secret = df[1]; regress y `secret'`` → echoed
    as ``. regress y patient_42``), and the Stata error message
    body (which embeds variable / file names that may be data-derived).
    Neither is parser-owned. We now keep only the command's VERB
    (first whitespace-separated token — ``regress``, ``summarize``,
    ``use``) and the ``r(<code>);`` line: enough framing for the
    model to know the kind of failure without exfiltrating any
    arguments or error text. The researcher's run log retains the
    full block for audit.
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
    # echo (`. <cmd>`); extract only the verb.
    region = scan[:rc.start()]
    cmd_matches = list(_STATA_CMD_RE.finditer(region))
    if not cmd_matches:
        # No command echo found — return the rc line alone. We don't
        # walk back over error-message lines (they may carry data).
        return f"[command body redacted]\nr({rc_code});"

    cmd_text = cmd_matches[-1].group("cmd").strip()
    # First whitespace-separated token is the command verb. Cap at
    # 32 chars as a defense-in-depth against an attacker-chosen
    # ``capture noisily`` prefix that combines the verb with an
    # inline data argument before whitespace. Stata verbs are short
    # identifiers; anything longer than 32 chars in the first token
    # almost certainly is an attempt to push data through.
    verb = cmd_text.split(None, 1)[0] if cmd_text else ""
    verb = verb[:32]
    # Validate against Stata's verb alphabet (lowercase letters and
    # underscores). If it doesn't match, redact entirely rather than
    # forwarding what an attacker constructed.
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", verb or ""):
        verb = ""
    if verb:
        return f". {verb} [args redacted]\n[message body redacted]\nr({rc_code});"
    return f"[command body redacted]\nr({rc_code});"


# -----------------------------------------------------------------
# Exception-body scrub
# -----------------------------------------------------------------

def _scrub_exception_body(msg: str) -> str:
    """Drop exception message bodies entirely.

    The exception type (``KeyError``) and traceback frame (file +
    line) are parser-owned — Python / R themselves emit those in a
    fixed shape we can verify. The body that follows the type, on
    the other hand, is whatever the script chose to put there.
    Previous versions allowed short non-data-shaped bodies through
    on the assumption that "column names and missing-file paths are
    exactly what the model needs", but a 1-cell, sub-cap body
    (``raise RuntimeError(df.iloc[0]['secret'])``) survived the cap
    and the shape detector — a deliberate SDC bypass available to
    any model-authored script.

    The threat model treats the model as untrusted with respect to
    the researcher's data. So the body cannot be forwarded: only the
    parser-owned framing (TYPE for Python, "Error :" template for R,
    ``r(<code>);`` for Stata, plus the function / call chain) is
    preserved. The model still gets enough framing to know what
    KIND of failure happened and where in the script source it
    fired — and it wrote the script, so it knows what the affected
    identifier was without us having to echo it back.

    Researchers can still read the un-scrubbed stderr / stdout from
    the on-disk run dir (``raw_log_path`` / the UI). This redaction
    only applies to the model-visible ``debug_excerpt`` channel.
    """
    return _REDACTED_BODY if msg else msg


_REDACTED_BODY = "[message body redacted]"


# -----------------------------------------------------------------
# Common scrub / cap
# -----------------------------------------------------------------

# Reduce absolute paths in messages / tracebacks to their basename.
# We keep "line N" markers because line numbers are exactly what
# the model needs. Three patterns, applied in order:
#
#   1. Quoted Python-traceback paths (`"/path/to/file.py"`).
#   2. Paths with spaces that end at a known data/script extension —
#      `"FileNotFoundError: ... /Users/John Smith/wages.csv"` would
#      otherwise leak `Smith/research/wages.csv` because the strict
#      bare-path regex stops at the first space. Anchoring on the
#      extension lets us absorb the username space without over-
#      matching into trailing prose.
#   3. Bare absolute paths in free-form messages with NO space
#      anywhere in the path (the strict legacy pattern, used as a
#      catch-all after the extension-anchored sweep).
_QUOTED_ABS_PATH_RE = re.compile(r'"((?:/[^"\n]+))"')
_PATH_WITH_EXT_RE = re.compile(
    r"(?<![\w/])"                      # not preceded by a word char or slash
    r"/[A-Za-z0-9_.\- /]+?"            # path body, allowing single spaces
    r"\.(?:csv|tsv|dta|rds|RData|rdata|parquet|jsonl|ndjson|"
    r"R|do|py|ipynb|sql|sas|sav|"
    r"xlsx|xls|txt|json|md|log|"
    r"png|pdf|svg|jpg|jpeg|html|"
    r"yaml|yml|toml)\b"                # known data / script / output extension
)
_BARE_ABS_PATH_RE = re.compile(r"(?<![\w/])/[A-Za-z0-9_.\-/]{4,}")

# Credential patterns we always strip. Scoped tight enough that
# they don't false-positive on column names or numeric IDs.
# Coverage roughly matches the secret types most likely to land in
# a researcher's `Sys.getenv(...)` / `os.environ[...]` and ride
# through into an error message (header dumps, library prints).
#   - Anthropic-style:    sk-ant-...     (>=20 alphanum/dash after sk-)
#   - OpenAI-style:       sk-...         (>=20 alphanum after sk-)
#   - Stripe / similar:   sk_live_..., sk_test_... (underscore variant
#                         that the hyphen-anchored OpenAI regex misses)
#   - AWS access keys:    AKIA + 16 alnum
#   - JWTs:               three base64url segments separated by dots
#   - GitHub PATs / OAuth / server tokens / fine-grained PATs:
#                         ghp_/gho_/ghs_/ghu_/ghr_ + 36+ alnum, plus
#                         the longer github_pat_ prefix
#   - Slack tokens:       xoxb-/xoxp-/xoxa-/xoxr-/xoxs- + numeric + alnum
#   - HuggingFace:        hf_ + 30+ alnum
_CRED_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"sk-(?:ant-)?[A-Za-z0-9_\-]{20,}"),
    re.compile(r"\bsk_(?:live|test)_[A-Za-z0-9]{20,}\b"),
    re.compile(r"AKIA[A-Z0-9]{16}"),
    re.compile(
        r"\beyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\b"
    ),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{60,}\b"),
    re.compile(r"\bxox[abprs]-[A-Za-z0-9\-]{10,}\b"),
    re.compile(r"\bhf_[A-Za-z0-9]{30,}\b"),
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

    # Path normalisation - extension-anchored sweep first so paths
    # with spaces (a username like "/Users/John Smith/research/x.csv")
    # collapse to the basename even though the strict regex below
    # would stop at the first space.
    def _bare_basename(m: re.Match[str]) -> str:
        return Path(m.group(0)).name
    out = _PATH_WITH_EXT_RE.sub(_bare_basename, out)
    # Path normalisation - bare absolute paths in free-form
    # messages (R / Stata sometimes emit these). Strict charset
    # (no spaces) catches paths the extension-anchored sweep
    # missed but leaves space-bearing ones to the sweep above.
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
