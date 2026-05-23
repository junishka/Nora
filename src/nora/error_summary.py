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

The fix is a new ``debug_excerpt`` field on the failure tool result
that carries the parser-owned framing of the language's error
idiom: the exception type and the user-code frame (with the
indented source-line preview) for Python; the literal ``Error :``
template plus the ``Calls:`` chain for R; the failing command's
verb plus ``r(<code>);`` for Stata.

What the model sees is intentionally thin: enough framing to know
what KIND of failure happened and where in the script source it
fired (the model wrote the script and already knows what each
identifier means), but nothing that crossed the SDC boundary in
the exception message body.

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
  3. **Per-language body posture.** The boundary is asymmetric by
     language because the leak surface and the model's alternatives
     differ:

       * **Python** — exception bodies are redacted wholesale on
         the user-code path. The exception TYPE and the user-code
         FRAME (file + line + source-line preview) are parser-owned
         and already give the model what it needs in the common
         case (a typo in ``df['col']`` shows up in the source-line
         preview, which is user-authored .py source). The body
         that follows the type is script-controlled and could
         exfiltrate any short cell value (``raise RuntimeError
         (df.iloc[0]['secret'])``); forwarding it would be redundant
         with the source line and open a covert channel.
       * **Stata + R** — bodies forward through ``_forward_short_body``
         (length cap + data-shape detect). The model has no
         equivalent of Python's source-line preview here: Stata's
         user-code extractor only sees the command echo, R's only
         sees the ``Error : ...`` block. Wholesale redaction left
         the model unable to act on common failures like "X
         invalid varname" or "object 'X' not found". The denylist
         posture forwards the actual diagnostic and bounds the
         residual scalar-leak channel to ~200 bytes per error.
         Stata command echoes go through the same scrub.
  4. **No `print(df)` leakage.** Because Stata's extractor anchors
     on `r(<code>);` and walks back to the most recent `. <cmd>`,
     intervening `display` / `list` output stays out of the
     excerpt. The ``MAX_EXCERPT_BYTES`` hard cap is defense in depth.
  5. **Length-aware redaction.** Quoted args longer than
     ``MAX_QUOTED_ARG_BYTES`` get truncated in place — covers the
     "ValueError with a 5KB pandas repr" foot-gun even though the
     body itself is already redacted.
  6. **Credential scrub.** Regex out `sk-...`, `AKIA...`,
     three-segment JWTs, GitHub / Slack / HF tokens, URL userinfo.
     Catches the `print(os.environ)` foot-gun and pip's echo of
     token-bearing index URLs from ``install_packages``.
  7. **Path normalisation.** Absolute paths get reduced to their
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

# Cap on Stata + R error bodies forwarded under the denylist posture
# (see ``_forward_short_body`` and module docstring). Legitimate
# error bodies are short — "highest_forprofit_title_pre_ceo_rank
# invalid varname" is 50 chars, "object 'wage' not found" is 22
# chars, "variable mpg not found" is 22. Bodies past 200 chars are
# almost always verbose multi-line estimator diagnostics or data
# dumps, both of which we'd rather truncate than forward whole.
# Larger than ``MAX_EXCEPTION_MSG_BYTES`` because Stata / R bodies
# can legitimately span two or three lines, where Python exception
# message bodies (still redacted wholesale on the user-code path)
# are always single-line.
MAX_FORWARDED_BODY_BYTES = 200


# Shapes that suggest an exception body is a data dump rather than
# a parser-owned diagnostic. Used by both ``_scrub_exception_body``
# (Python user-code phase) and ``_forward_short_body`` (Stata + R
# denylist mode).
#
# Two concerns the detector has to balance:
#
#   * Catch canonical exfil shapes: JSON dump from
#     ``df.iloc[0].to_json()`` (``{"k": "v", ...}``); CSV row from
#     ``df.to_csv()`` / ``str(row)`` (six-or-more comma-separated
#     mixed-shape tokens like ``patient_42, John Smith, 1985,
#     100000, doctor, NY``).
#
#   * Don't catch legitimate identifier lists embedded in real error
#     messages: Stata varlists (``mpg, price, weight, length,
#     displacement, gear_ratio``); R formula term lists; six-arg
#     function call sites (``pmin(a, b, c, d, e, f, g)``). A naïve
#     "6+ comma-separated tokens" rule trips on all of these,
#     stripping the model of the variable names it needs to act on
#     the error.
#
# The distinguishing signal is per-token shape: a pure-identifier
# token sequence is a legitimate varlist; a sequence with at least
# one non-identifier token (a quoted string, a number, a date, a
# name-with-space) is the row-dump shape. The check on the
# comma-separated branch verifies that AT LEAST ONE token in the
# run is non-identifier; an all-identifier run forwards.
#
# Identifier alphabet here is wider than ``safe_key``'s — it accepts
# a leading letter / underscore plus alphanumerics, underscores,
# periods (R ``package.name`` / ``data.frame`` style), dollar signs
# (R column accessors like ``df$col``), and parens (Stata function-
# style options like ``by(price``, R / Python function calls like
# ``pmin(a``, fixest formulas like ``i(x)``). Hyphen is excluded
# because a token with a hyphen is usually a date or a number
# (``2026-01-01``). The required-leading-letter-or-underscore is
# load-bearing: it forces token to LOOK like a name, so pure
# numeric tokens (``12345``, ``3.14``) and quoted strings (``"v"``)
# still fail the shape check and the row stays flagged.
_JSON_DICT_RE = re.compile(r'\{[^{}\n]*:[^{}\n]*\}')
_COMMA_LIST_RE = re.compile(r'[^,\n]{1,40}(?:\s*,\s*[^,\n]{1,40}){5,}')
_IDENTIFIER_TOKEN_RE = re.compile(r'^[A-Za-z_][A-Za-z0-9_.$()]*$')
_REDACTED_DATA_BODY = "[message body suppressed: looked data-shaped]"


def _body_looks_data_shaped(body: str) -> bool:
    """Return True if ``body`` matches a known data-exfil shape.

    Two shapes:
      1. JSON-dict (``{"k": "v"}``) — always redacted.
      2. Six-or-more comma-separated tokens where AT LEAST ONE
         MIDDLE token is non-identifier-shape. A pure-identifier
         run (Stata varlist, R formula args, ``pmin(a, b, c, d, e,
         f, g)`` call sites) is legitimate error context; mixed-
         shape runs are the canonical row dump.

    Boundary tokens (first / last) are excluded from the shape
    check because the greedy regex absorbs prefix / suffix prose
    into them — e.g., ``object x1, x2, x3, x4, x5, x6 not found``
    splits into ``['object x1', 'x2', 'x3', 'x4', 'x5', 'x6 not
    found']``. The first and last carry sentence context, not the
    token shape we care about; the inner tokens are bounded by
    commas on both sides and are the reliable signal. Since the
    regex requires 5+ continuations, every match has ≥ 6 raw
    tokens and ≥ 4 middle tokens — enough to distinguish.
    """
    if _JSON_DICT_RE.search(body):
        return True
    for run in _COMMA_LIST_RE.finditer(body):
        tokens = [t.strip() for t in run.group(0).split(',')]
        middle = tokens[1:-1]
        if any(t and not _IDENTIFIER_TOKEN_RE.match(t) for t in middle):
            return True
    return False


# -----------------------------------------------------------------
# Public entry
# -----------------------------------------------------------------

def extract_debug_excerpt(
    stdout: str,
    stderr: str,
    exit_code: int | None,
    language: str,
    *,
    run_dir: "Path | None" = None,
    pre_user_stderr: "str | None" = None,
    user_stderr: "str | None" = None,
) -> Optional[str]:
    """Return a short, human-readable error excerpt for the model
    or ``None`` if extraction couldn't locate a recognisable error.

    ``language`` is one of ``"R"`` / ``"Stata"`` / ``"Python"``.
    Casing is preserved as the executor uses it.

    Stdout is only consulted for Stata; for R and Python the
    excerpt comes from stderr exclusively. This keeps the SDC
    surface narrower (researchers ``print(df)`` in stdout, almost
    never in stderr).

    Phase-aware redaction for Python relies on the executor's
    buffer-split stderr capture:

      * ``pre_user_stderr`` is what came through before user code
        ran (phase 0 + phase A in ``executor._split_stderr_buffers``).
        By construction these bytes can't contain researcher data,
        so they're forwarded unredacted. This is the load-bearing
        invariant: it's enforced at the kernel level via the
        preamble's ``dup2``, NOT inferred from text shape. A
        segfault during user code that wrote to stderr beforehand
        used to look like ``pre_script`` to a traceback-based
        classifier; with the buffer split it correctly routes
        through the redacted path.
      * ``user_stderr`` is everything written after the marker swap
        (phase B). Python tracebacks from user code land here, as
        do any ``print(df.head(), file=sys.stderr)`` writes. The
        deepest traceback frame's path then decides whether the
        exception body is safe to forward (nora-owned ⇒ yes,
        researcher-authored ⇒ redacted).

    Legacy callers that pass only ``stderr`` (no buffer split)
    still work: the function treats the whole stderr as
    ``user_stderr``, which matches the pre-buffer-split SDC
    posture (full redaction).
    """
    raw: Optional[str] = None
    if language == "Python":
        # Reconcile new buffer-split inputs with the legacy single-
        # stderr signature. Callers that pass neither phase keyword
        # fall through to the conservative posture (whole stderr is
        # user-code phase, body always redacted).
        if pre_user_stderr is None and user_stderr is None:
            pre_user_stderr = ""
            user_stderr = stderr or ""
        else:
            pre_user_stderr = pre_user_stderr or ""
            user_stderr = user_stderr or ""
        raw = _extract_python(
            pre_user_stderr=pre_user_stderr,
            user_stderr=user_stderr,
            run_dir=run_dir,
        )
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


def _classify_python_phase(
    user_stderr: str,
    frames: "list[re.Match[str]]",
    run_dir: "Path | None",
) -> str:
    """Return ``"nora_owned"`` or ``"user_code"`` for a traceback
    found in the buffer-split ``user_stderr`` stream.

    The rule operates on the DEEPEST frame (the one that actually
    raised), not on whether any user-code frame appears in the
    chain. That distinction matters for the canonical "import nora"
    case: the user wrote ``import nora`` on script.py line N, but
    the failure raised inside the staged ``lib/nora.py``. The
    deepest frame is in Nora-controlled code, so the body
    (``"NORA_RUN_TOKEN not set"``, etc.) is Nora-authored and
    safe to forward. A frame-anywhere rule would mis-redact this
    body because line N counts as researcher-authored.

    Concrete classification:
      * ``run_dir`` is None → no way to identify user-code paths;
        fall back to ``user_code`` for safety.
      * Deepest frame's path is ``<run_dir>/script.py`` AND its
        line is past the preamble marker → ``user_code``.
      * Otherwise → ``nora_owned`` (deepest frame is in staged
        ``lib/nora.py``, library / stdlib code, or in the
        preamble lines of script.py).
    """
    if not frames or run_dir is None:
        return "user_code"

    # Python's traceback formatter prints frames in source order
    # (oldest to newest), so the LAST match is the one that raised.
    deepest = frames[-1]
    path = deepest.group("path")
    script_path_str = str(run_dir / "script.py")
    if path == script_path_str:
        try:
            line_no = int(deepest.group("line"))
        except (TypeError, ValueError):
            line_no = 0
        cutoff = _preamble_line_count(run_dir)
        if line_no > cutoff:
            return "user_code"
    return "nora_owned"


def _preamble_line_count(run_dir: "Path") -> int:
    """Return the highest line number in ``script.py`` still owned by
    Nora (i.e., before the researcher's code starts).

    The executor's ``_write_script`` prepends a short preamble plus a
    ``# ----- Nora preamble above; researcher code below -----``
    marker line. The marker itself is Nora-owned; everything after
    it is researcher-authored. Reading the marker line number is
    cheaper and more robust than re-deriving the preamble length
    from the executor's source: a future preamble change won't
    invalidate the classifier.

    Returns 0 on any read failure — the classifier then conservatively
    treats every script.py frame as user code, which matches the
    legacy (always-redact) behavior.
    """
    script = run_dir / "script.py"
    try:
        with script.open("r", encoding="utf-8", errors="replace") as fh:
            for idx, line in enumerate(fh, start=1):
                if "Nora preamble above; researcher code below" in line:
                    return idx
    except OSError:
        return 0
    return 0


def _render_pre_user_stderr(pre_user_stderr: str) -> Optional[str]:
    """Format the safe-by-construction pre-user-code stderr block.

    Returned when ``user_stderr`` is empty: by the buffer-split
    invariant, nothing in ``pre_user_stderr`` came from user code.
    Includes launcher output (libxcrun / xcselect / dyld), sandbox
    denials, and anything Python printed while running the preamble
    itself — all phase-safe.

    Returns ``None`` when the buffer is empty so the caller can fall
    through to the executor's generic "script failed" fallback.
    """
    s = (pre_user_stderr or "").strip()
    if not s:
        return None
    # Tail only — long launcher output rarely carries the proximate
    # cause in its header. Cap at ~2 KB so the model gets enough
    # context to diagnose but the response stays small.
    tail = s[-2000:]
    header = "(interpreter / preamble stderr, no user code executed):"
    return f"{header}\n{tail}"


def _extract_python(
    *,
    pre_user_stderr: str,
    user_stderr: str,
    run_dir: "Path | None" = None,
) -> Optional[str]:
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

    Buffer-split flow:
      * If ``user_stderr`` is empty, the failure was pre-user-code.
        Forward ``pre_user_stderr`` tail unredacted via
        ``_render_pre_user_stderr``. This is what closes the
        diagnostic loop for libxcrun-style launcher failures: the
        bytes came through before any user code, so by construction
        they can't carry researcher data.
      * If ``user_stderr`` has content, user code ran. We extract
        the traceback from ``user_stderr`` only — ``pre_user_stderr``
        is irrelevant once user code has started, because the
        traceback lives in phase B too. ``_classify_python_phase``
        looks at the deepest frame's path to decide whether the
        exception body is forwardable.
    """
    if not (user_stderr or "").strip():
        # No user-code-phase stderr at all. ``pre_user_stderr`` may
        # carry the launcher / preamble output that explains the
        # failure (libxcrun dlopen, sandbox-deny, preamble syntax
        # error). Forward it verbatim — buffer-split invariant
        # guarantees no researcher data is in there.
        return _render_pre_user_stderr(pre_user_stderr)

    frames = list(_PY_FRAME_RE.finditer(user_stderr))
    if not frames:
        # User-code phase has bytes but no parseable Python
        # traceback. Two shapes land here, both unsafe to forward:
        #   * Signal kill (segfault, OOM, SIGABRT) where Python
        #     never got the chance to format a traceback; the
        #     buffer may still contain ``print(df, file=sys.stderr)``
        #     output from before the crash.
        #   * ``os._exit()`` or ``sys.exit(int)`` with prior stderr
        #     writes; same shape.
        # The pre-user buffer is still safe to surface — it can name
        # the failure (launcher / preamble) even when user_stderr
        # has unsafe content.
        rendered = _render_pre_user_stderr(pre_user_stderr)
        if rendered is not None:
            return rendered
        # Last-ditch: try to pull an exception line from user_stderr.
        # Treated as ``user_code`` for redaction since we have no
        # frame to classify against.
        return _last_python_exception_line(user_stderr, phase="user_code")

    phase = _classify_python_phase(user_stderr, frames, run_dir)

    # "User code" heuristic: any path that does NOT live under a
    # site-packages / dist-packages / typeshed / .venv / lib/python
    # segment.
    LIB_PAT = re.compile(
        r"(?:/site-packages/|/dist-packages/|/lib/python[\d\.]+/"
        r"|/typeshed/|/\.venv/|/python\d+\.\d+/lib/)"
    )
    # Nora's own wrapper frames. ``executor.py`` runs the researcher's
    # script through ``_nora_wrapper.py`` → ``runpy.run_path``, which
    # produces stderr that begins with three or four bridge frames
    # before reaching ``script.py``:
    #
    #   File "<run_dir>/_nora_wrapper.py", line 12, in <module>
    #     _nora_runpy.run_path("script.py", run_name="__main__")
    #   File "<frozen runpy>", line 287, in run_path
    #   File "<frozen runpy>", line  98, in _run_module_code
    #   File "<frozen runpy>", line  88, in _run_code
    #   File "script.py", line 3, in <module>
    #     raise RuntimeError("boom")
    #
    # Neither lives under a site-packages-style path, so ``LIB_PAT``
    # doesn't drop them. Without an explicit filter the excerpt
    # leads with the Nora wrapper + four runpy frames, contradicting
    # the wrapper's documented intent at executor.py:1768
    # ("tracebacks reference script.py and the wrapper's ``_nora_*``
    # names never leak into user scope") and misdirecting the model
    # toward Nora internals when it diagnoses the failure.
    #
    # Match strategy mirrors LIB_PAT: a path-fragment regex run via
    # ``re.search`` over the captured path. ``_nora_wrapper.py`` is
    # the actual on-disk filename written by ``_write_python_wrapper``
    # in executor.py; ``<frozen runpy>`` is the literal path string
    # Python emits for the stdlib runpy module since 3.11.
    NORA_WRAPPER_PAT = re.compile(r"(?:/_nora_wrapper\.py$|^<frozen runpy>$)")
    user_frames = [
        m for m in frames
        if not LIB_PAT.search(m.group("path"))
        and not NORA_WRAPPER_PAT.search(m.group("path"))
    ]
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
        # below the frame, when present. Source line preview is
        # taken from ``user_stderr`` — the same buffer the frame was
        # parsed from — so we never accidentally splice content
        # across phase boundaries.
        after = user_stderr[frame.end():]
        nl = after.find("\n")
        if nl != -1:
            candidate = after[nl + 1:].split("\n", 1)[0]
            if candidate.startswith("    ") and candidate.strip():
                parts.append(f"    {candidate.strip()}")

    exc_line = _last_python_exception_line(user_stderr, phase=phase) or ""
    if exc_line:
        parts.append(exc_line)
    return "\n".join(parts) if parts else None


def _last_python_exception_line(
    stderr: str, *, phase: str = "user_code",
) -> Optional[str]:
    """Find the last line that matches the ExceptionType[: msg]
    shape. Python tracebacks always end with this line, and there
    can be multiple in a chained exception ("During handling of
    the above..."). The LAST one is the one that propagated.

    ``phase`` selects redaction posture for the message body. See
    ``_classify_python_phase`` for the three values and what each
    implies. Defaults to ``user_code`` so any caller that doesn't
    pass phase information keeps the conservative redacted behavior.
    """
    matches = list(_PY_EXC_RE.finditer(stderr))
    if not matches:
        return None
    # Filter to lines that look like a top-level exception, not
    # something inside a code-line preview. Code preview lines are
    # indented; exception lines are flush-left.
    flush = [m for m in matches if not stderr[max(m.start() - 1, 0)].isspace()]
    chosen = flush[-1] if flush else matches[-1]
    msg = chosen.group("msg") or ""
    msg = _scrub_exception_body(msg, phase=phase)
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
    """Pull the LAST ``Error`` block's call deparse + body, plus its
    ``Calls:`` trailer if present.

    R errors look like::

        Error in eval(predvars, data, env) :
          object 'wage' not found
        Calls: lm -> eval -> eval

    Under the denylist posture (see module docstring), both the
    ``<call>`` deparse and the message body forward through
    ``_forward_short_body`` (length cap + data-shape detect).
    Credential scrubs and path normalisation run later in
    ``_scrub_and_cap``. The previous posture redacted both
    wholesale; that bounded ``stop(df$secret[1])``-style exfil but
    left the model unable to read the specific diagnostic ("object
    'wage' not found" → "Error : [message body redacted]"). The
    denylist mitigations bound the residual short-scalar leak
    channel; see ``_forward_short_body``.
    """
    matches = list(_R_ERROR_RE.finditer(stderr))
    if not matches:
        return None
    last = matches[-1]
    call_text = (last.group("call") or "").strip()
    msg_text = (last.group("msg") or "").strip()
    call_excerpt = _forward_short_body(call_text)
    msg_excerpt = _forward_short_body(msg_text)
    if call_excerpt:
        block = f"Error in {call_excerpt} : {msg_excerpt}"
    else:
        block = f"Error : {msg_excerpt}"

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
# Phase-boundary marker the executor's Stata preamble ``display``s
# right before the researcher's code starts. First occurrence in
# the log marks the phase A → phase B boundary; any failing
# command whose echo line appears BEFORE this offset is preamble
# (nora_owned). The marker text is fixed-string and not token-
# bearing: the attack is "user code emits the same string to
# confuse classification", and the first-occurrence rule kills it
# because real-marker output always precedes user code.
_STATA_MARKER = "_NORA_STATA_PREAMBLE_END_MARKER_"


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

    # Phase-boundary classification. The preamble's ``display`` of
    # ``_NORA_STATA_PREAMBLE_END_MARKER_`` is the first occurrence
    # in a healthy run; any failing command whose echo appears
    # BEFORE that offset is preamble (nora_owned: no user code
    # has run, the failure body is Nora-authored). Any failing
    # command AFTER the marker is user-code (legacy redacted
    # posture). If the marker is missing, we conservatively treat
    # the whole log as user-code — that handles older Stata runs
    # without the marker line, scripts that bailed before the
    # ``display`` reached the log, and the SDC-safe fallback when
    # something interferes with the marker contract.
    marker_offset = scan.find(_STATA_MARKER)
    nora_owned_region = -1 if marker_offset < 0 else marker_offset

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

    cmd_match = cmd_matches[-1]
    cmd_text = cmd_match.group("cmd").strip()

    # Is the failing command in the nora-owned (preamble) region?
    # We compare the command echo's offset to the marker's offset.
    cmd_is_nora_owned = (
        nora_owned_region >= 0 and cmd_match.start() < nora_owned_region
    )

    if cmd_is_nora_owned:
        # Preamble failure: the failing command is one Nora wrote
        # into the .do file (``adopath``, ``local lib : env ...``,
        # one of the ``capture program drop nora_*`` lines). Its
        # arguments and the Stata error message are both
        # Nora-controlled — no researcher data has reached them.
        # Forward both verbatim, capped, so the model can read the
        # actual diagnostic instead of "[message body redacted]".
        body = _stata_error_body_between(scan, cmd_match.end(), rc.start())
        cmd_excerpt = cmd_text[:200]
        return f". {cmd_excerpt}\n{body}\nr({rc_code});"

    # User-code branch under denylist posture: forward the command
    # line and the error body so the model can diagnose the actual
    # failure ("X invalid varname", "object 'X' not found",
    # "joinby ein using `orgyears'"). Both pass through
    # ``_forward_short_body`` for length cap + data-shape detect;
    # credential scrubs, URL-userinfo collapse, and path
    # normalisation run later in ``_scrub_and_cap`` over the whole
    # excerpt.
    #
    # The previous posture redacted args + body wholesale on the
    # theory that macro expansion (``regress y `secret_macro'`` →
    # ``regress y patient_42`` in the log) and data-derived names
    # in error messages would leak cell content. That trade-off
    # was too aggressive: the common failure mode is a model-
    # authored identifier that hit a Stata constraint (>32-char
    # varname cap, unrecognised verb, missing column), and without
    # the body the model couldn't tell what to fix. The denylist
    # mitigations (data-shape detect + 200-char cap + downstream
    # credential / path scrubs) bound the residual short-scalar
    # leak channel; see module docstring and ``_forward_short_body``.
    body = _stata_error_body_between(scan, cmd_match.end(), rc.start())
    cmd_excerpt = _forward_short_body(cmd_text)
    body_excerpt = _forward_short_body(body)
    if cmd_excerpt:
        return f". {cmd_excerpt}\n{body_excerpt}\nr({rc_code});"
    return f"{body_excerpt}\nr({rc_code});"


def _stata_error_body_between(scan: str, start: int, end: int) -> str:
    """Return the Stata error-message body that sits between a
    failing command echo and the matching ``r(<code>);``.

    Stata prints the error message on the line(s) between the
    command echo and the rc line. There are no further command
    echoes in this region by construction — Stata aborts after the
    first failing command. Walk forward from ``start``, collect
    non-empty lines, stop at the rc line position. Cap at the
    standard exception-body length so a verbose Stata error
    (multi-line error blocks from certain estimators) doesn't
    inflate the response.
    """
    block = scan[start:end].strip("\n")
    lines = [
        line for line in block.splitlines()
        if line.strip() and not line.startswith("end of do-file")
    ]
    body = "\n".join(lines)
    if len(body) > MAX_EXCEPTION_MSG_BYTES * 4:
        body = body[: MAX_EXCEPTION_MSG_BYTES * 4] + "[...]"
    return body or "[no body emitted]"


# -----------------------------------------------------------------
# Exception-body scrub
# -----------------------------------------------------------------

def _scrub_exception_body(msg: str, *, phase: str = "user_code") -> str:
    """Redact exception message bodies that may carry researcher data.

    Three phases (see ``extract_debug_excerpt``):

      * ``user_code`` — researcher's script raised. Body is
        script-controlled and could exfiltrate any short cell value
        (``raise RuntimeError(df.iloc[0]['secret'])``), so we
        redact wholesale. This is the legacy posture and the
        default for any caller that didn't pass phase info.
      * ``nora_owned`` — the traceback fired inside the staged
        runtime, the preamble, or library / stdlib code that
        executed before user code touched data. The body is
        Nora-controlled (``"NORA_RUN_TOKEN not set"``,
        ``"fit.components_ missing"``, etc.) and cannot contain
        researcher data. Forward verbatim, but length-cap as
        defense in depth.
      * ``pre_script`` — should not normally reach here (handled
        upstream in ``_render_pre_script_stderr``), but mirror
        nora_owned's posture for safety: body is launcher /
        interpreter-startup output, no researcher data possible.

    The body cap (``MAX_EXCEPTION_MSG_BYTES``) applies to the safe
    phases too — defends against accidental verbose runtime
    messages that would otherwise inflate the response.
    """
    if not msg:
        return msg
    if phase in ("nora_owned", "pre_script"):
        if len(msg) > MAX_EXCEPTION_MSG_BYTES:
            return msg[:MAX_EXCEPTION_MSG_BYTES] + "[...]"
        return msg
    return _REDACTED_BODY


_REDACTED_BODY = "[message body redacted]"


def _forward_short_body(body: str) -> str:
    """Apply length cap + data-shape detect to an error body that
    the caller wants to forward to the model.

    This is the denylist posture used by Stata and R: forward the
    actual diagnostic ("X invalid varname", "object 'X' not found",
    "joinby ein using `orgyears'") so the model can act on the
    specific error without re-probing, with two targeted scrubs:

      * **Data-shape detect** — if the body matches a JSON-dict
        shape or a 6+ comma-separated-tokens shape (the canonical
        ``df.to_csv()`` / ``to_json()`` exfil patterns), drop it
        wholesale. Catches the obvious ``raise RuntimeError(df.iloc[0].to_json())``
        / ``stop(paste(df$row, collapse=","))`` channels.
      * **Length cap** — bodies past ``MAX_FORWARDED_BODY_BYTES``
        are usually verbose estimator output or data dumps; truncate
        to bound the per-error bandwidth.

    Credential scrubs, URL-userinfo collapse, and path normalisation
    run later in ``_scrub_and_cap`` over the full excerpt, so we
    don't repeat them here — keeping this helper a thin per-body
    pass keeps the layering legible.

    Residual risk under this posture: short scalar values (a single
    integer ID, a short string column value) embedded in a Stata or
    R error message can still pass through. This is the explicit
    SDC trade-off recorded in the module docstring — the model
    needs to read the actual error to fix it, and short scalars
    fall under the same bandwidth a script could exfiltrate through
    a single ``display`` / ``print`` call anyway. The cell-suppression
    threshold protects against tiny-count disclosure at the sanitizer
    layer; this channel is bounded to ~200 bytes per failed script.
    """
    if not body:
        return ""
    if _body_looks_data_shaped(body):
        return _REDACTED_DATA_BODY
    if len(body) > MAX_FORWARDED_BODY_BYTES:
        return body[:MAX_FORWARDED_BODY_BYTES] + "[...]"
    return body


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
#   - URL userinfo:       scheme://user:token@host (pip echoes
#                         ``Looking in indexes: https://USER:TOKEN@
#                         private-pypi.acme.com/simple`` from
#                         ~/.pip/pip.conf on every install, so the
#                         token-bearing URL would otherwise leak
#                         through ``install_packages``'s error path).
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
_URL_USERINFO_RE = re.compile(
    r"([a-zA-Z][a-zA-Z0-9+.\-]*://)[^/?#\s@]+:[^/?#\s@]+@",
)
_CRED_REDACTED = "[redacted-credential]"

# Long quoted blobs in error messages - heuristic for "researcher
# accidentally embedded a row in the error". The threshold is the
# same per-arg cap we apply to Python exceptions.
_LONG_QUOTED_RE = re.compile(
    r"(['\"])((?:(?!\1).){" + str(MAX_QUOTED_ARG_BYTES) + r",})\1",
    re.DOTALL,
)


def scrub_raw_output(text: str, cap_bytes: int) -> str:
    """Public wrapper around the credential / path scrubber for callers
    that need to forward verbatim subprocess output to the model.

    Use case: ``install_packages`` runs OUTSIDE the script sandbox
    (it needs network) and returns ``raw_stdout`` / ``raw_stderr``
    excerpts on failure so the model can diagnose. Those bytes
    skipped the language-anchored extraction in ``extract_debug_excerpt``,
    but they still must go through the same credential and path-
    normalisation pass — otherwise a private pip index URL (with
    embedded user:token), an AWS-style key dumped via ``echo``, or
    an internal absolute path leaks through this side door.

    ``cap_bytes`` overrides the default ``MAX_EXCERPT_BYTES`` cap
    because install excerpts have their own tighter budgets (1500
    for stdout, 3000 for stderr) — passing the cap explicitly keeps
    the chokepoint honest about how much can cross.
    """
    if not text:
        return text
    scrubbed = _scrub_and_cap(text)
    if len(scrubbed) > cap_bytes:
        marker = "\n…[excerpt truncated]"
        scrubbed = scrubbed[:cap_bytes - len(marker)].rstrip() + marker
    return scrubbed


def _scrub_and_cap(text: str) -> str:
    """Normalise paths, redact credentials, trim oversize quoted
    blobs, and cap the total to ``MAX_EXCERPT_BYTES``."""
    out = text

    # URL-embedded userinfo (user:token@host) MUST come before path
    # normalisation. The bare-abs-path regex matches the ``//USER``
    # in ``https://USER:TOKEN@host`` (the lookbehind passes because
    # the char before the first slash is ``:``, not ``/``) and
    # replaces it with the basename ``USER``, destroying the ``://``
    # anchor this regex relies on. Collapse the credentials first so
    # the path pass only ever sees ``https://[redacted-credential]@``
    # which the lookbehind correctly rejects.
    out = _URL_USERINFO_RE.sub(
        lambda m: f"{m.group(1)}{_CRED_REDACTED}@", out,
    )

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
