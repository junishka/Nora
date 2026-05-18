"""Per-language happy-path coverage for ``extract_debug_excerpt``.

These tests pin the *content* the model sees on a script failure -
the actual error line, the failing call site, the exit code marker.
The leak-boundary regressions live in ``test_error_summary_no_leak.py``;
this file is just "did the extractor find the right sentence."
"""

from __future__ import annotations

from nora.error_summary import (
    MAX_EXCERPT_BYTES,
    extract_debug_excerpt,
)


# ---------------------------------------------------------------------------
# Python
# ---------------------------------------------------------------------------

def test_python_keyerror_keeps_type_redacts_body() -> None:
    """The exception type and user-code frame survive; the body
    (here ``'typo'``) is redacted to close the script-controlled
    body channel. The model already wrote the script so it knows
    which ``df[X]`` line referenced the missing key — surfacing
    the actual key value is unnecessary and exfiltratable."""
    stderr = (
        'Traceback (most recent call last):\n'
        '  File "/abs/path/script.py", line 17, in <module>\n'
        "    df['typo']\n"
        '  File "/lib/python3.12/site-packages/pandas/core/frame.py", line 4090, in __getitem__\n'
        '    raise KeyError(key) from err\n'
        "KeyError: 'typo'\n"
    )
    excerpt = extract_debug_excerpt("", stderr, 1, "Python")
    assert excerpt is not None
    # Type preserved.
    assert "KeyError" in excerpt
    # The exception body is redacted on the exception line. The
    # source-line preview ``df['typo']`` IS the model's own script
    # source (Python's traceback formatter echoes the source line),
    # so the literal substring may also appear there — that's safe
    # (the model wrote that code). The leak channel is the body.
    last_line = excerpt.strip().splitlines()[-1]
    assert last_line.startswith("KeyError")
    assert "[message body redacted]" in last_line
    # The user-code frame survives; the pandas internals frame is dropped.
    assert 'line 17' in excerpt
    assert 'pandas' not in excerpt


def test_python_traceback_path_is_basenamed() -> None:
    """Absolute paths in tracebacks get reduced to basename - the
    researcher's home directory layout doesn't leak, but the line
    number does (which is what the model needs)."""
    stderr = (
        'Traceback (most recent call last):\n'
        '  File "/Users/you/.nora-sessions/2026/regression.py", line 42, in main\n'
        '    1 / 0\n'
        'ZeroDivisionError: division by zero\n'
    )
    excerpt = extract_debug_excerpt("", stderr, 1, "Python")
    assert excerpt is not None
    assert "/Users/you/.nora-sessions" not in excerpt
    assert '"regression.py"' in excerpt
    assert 'line 42' in excerpt


def test_python_includes_source_line() -> None:
    """The traceback's source-line preview (the indented line under
    the frame) is parser-emitted from the model's own script source,
    so it's safe to forward — the model already wrote that code.
    Only the exception body is data-controlled and redacted."""
    stderr = (
        'Traceback (most recent call last):\n'
        '  File "/abs/run.py", line 9, in <module>\n'
        '    result = compute_things(df)\n'
        'NameError: name \'compute_things\' is not defined\n'
    )
    excerpt = extract_debug_excerpt("", stderr, 1, "Python")
    assert excerpt is not None
    # Source-line preview preserved (it's the model's own code).
    assert "result = compute_things(df)" in excerpt
    # Type preserved, body redacted.
    assert "NameError" in excerpt
    assert "[message body redacted]" in excerpt
    assert "'compute_things' is not defined" not in excerpt


def test_python_chained_exceptions_keeps_last_one() -> None:
    """When Python raises during except-handling, both exceptions
    appear. The LAST one is the propagating one — that's the type
    that surfaces. Body still redacted."""
    stderr = (
        'Traceback (most recent call last):\n'
        '  File "/x.py", line 1, in <module>\n'
        '    1/0\n'
        'ZeroDivisionError: division by zero\n'
        '\n'
        'During handling of the above exception, another exception occurred:\n'
        '\n'
        'Traceback (most recent call last):\n'
        '  File "/x.py", line 3, in <module>\n'
        '    raise RuntimeError("retry failed")\n'
        'RuntimeError: retry failed\n'
    )
    excerpt = extract_debug_excerpt("", stderr, 1, "Python")
    assert excerpt is not None
    # The propagating type is RuntimeError; the body is redacted on
    # the final exception line. The literal ``"retry failed"`` is
    # in the model's own ``raise RuntimeError("retry failed")``
    # source line, which the traceback formatter echoes — that's
    # not a leak (it's the script source, which the model wrote).
    last_line = excerpt.strip().splitlines()[-1]
    assert last_line.startswith("RuntimeError")
    assert "[message body redacted]" in last_line


# ---------------------------------------------------------------------------
# R
# ---------------------------------------------------------------------------

def test_r_error_block_with_calls_chain() -> None:
    """R's ``Error`` anchor + the ``Calls:`` trailer survive, and
    under the denylist posture the call deparse and message body
    forward through ``_forward_short_body`` so the model can read
    the actual diagnostic ("object 'wage' not found") instead of a
    redacted placeholder."""
    stderr = (
        "Loading required package: stats\n"
        "Error in eval(predvars, data, env) : object 'wage' not found\n"
        "Calls: lm -> eval -> eval\n"
        "Execution halted\n"
    )
    excerpt = extract_debug_excerpt("", stderr, 1, "R")
    assert excerpt is not None
    # The "Error in <call> : <body>" framing and "Calls:" trailer
    # are now both forwarded. Call deparse helps the model locate
    # the failure ("in eval(predvars, ...)"); body names the missing
    # symbol ("object 'wage' not found").
    assert "Error in" in excerpt
    assert "eval(predvars" in excerpt
    assert "object 'wage' not found" in excerpt
    assert "Calls: lm -> eval -> eval" in excerpt
    # "Execution halted" trailer is noise — still dropped.
    assert "Execution halted" not in excerpt


def test_r_multiline_error_message_forwarded() -> None:
    """Multi-line error message bodies forward up to the cap. The
    Calls: trailer remains as parser-owned framing. Pre-denylist
    this was ``test_r_multiline_error_message_redacted`` and asserted
    that ``NA/NaN/Inf in 'x'`` did NOT appear; under the new posture
    the model needs to read that diagnostic to fix the design matrix."""
    stderr = (
        "Error in lm.fit(x, y, offset = offset, singular.ok = singular.ok, ...) : \n"
        "  NA/NaN/Inf in 'x'\n"
        "Calls: lm -> lm.fit\n"
    )
    excerpt = extract_debug_excerpt("", stderr, 1, "R")
    assert excerpt is not None
    assert "NA/NaN/Inf in 'x'" in excerpt
    assert "lm.fit(x, y" in excerpt
    assert "Calls: lm -> lm.fit" in excerpt
    assert "Error in" in excerpt


def test_r_only_last_error_is_returned() -> None:
    """If a script logs multiple errors (e.g., recovered errors
    inside ``tryCatch``), only the LAST top-level one's call +
    body + Calls trailer is returned. The earlier ``foo()`` error
    is dropped so the model focuses on the actual propagating
    failure."""
    stderr = (
        "Error in foo() : early problem\n"
        "Error in bar() : the actual cause\n"
        "Calls: bar -> baz\n"
    )
    excerpt = extract_debug_excerpt("", stderr, 1, "R")
    assert excerpt is not None
    # The LAST error's body and call survive; the earlier one is
    # dropped entirely (we extract only the last match).
    assert "the actual cause" in excerpt
    assert "early problem" not in excerpt
    assert "foo()" not in excerpt
    assert "bar()" in excerpt
    assert "Calls: bar -> baz" in excerpt


# ---------------------------------------------------------------------------
# Stata
# ---------------------------------------------------------------------------

# Stata batch-mode log on a typical failure. The leading dot-space
# lines are command echos; the bare "variable ... not found" is the
# Stata error message; "r(111);" is the rc.
_STATA_LOG_TYPICAL = """\
. set more off

. use "panel.dta", clear

. regress y x_missing
variable x_missing not found
r(111);

end of do-file

r(111);
"""


def test_stata_extract_anchors_on_rc_and_command_forwards_body() -> None:
    """The failing command echo + error body forward under the
    denylist posture so the model can read the actual diagnostic
    ("variable x_missing not found") and the command that triggered
    it ("regress y x_missing"). The extractor still anchors on
    ``r(<code>);`` and walks back to the most recent ``. <cmd>`` —
    unrelated earlier command echoes never reach the excerpt
    (that's what bounds the per-error bandwidth)."""
    excerpt = extract_debug_excerpt(_STATA_LOG_TYPICAL, "", 111, "Stata")
    assert excerpt is not None
    # Failing command line forwards in full.
    assert ". regress y x_missing" in excerpt
    # Error body forwards.
    assert "variable x_missing not found" in excerpt
    # rc line preserved.
    assert "r(111);" in excerpt
    # The unrelated `set more off` / `use ...` echos must NOT be
    # in the excerpt - they're not the failing command.
    assert "set more off" not in excerpt
    assert "use \"panel.dta\"" not in excerpt


def test_stata_excludes_end_of_dofile_trailer_rc() -> None:
    """The closing ``r(<code>);`` after "end of do-file" is just
    an exit echo. The extractor must anchor on the inline one
    (the real error), not the exit echo."""
    excerpt = extract_debug_excerpt(_STATA_LOG_TYPICAL, "", 111, "Stata")
    assert excerpt is not None
    # The whole excerpt sits BEFORE "end of do-file" in the log;
    # the exit echo is excluded entirely.
    assert "end of do-file" not in excerpt


def test_stata_modifier_wrapped_command_forwards_full_line() -> None:
    """Stata wrappers like ``capture``, ``quietly``, ``noisily`` take
    another command as their body. The old verb-only extractor
    unwrapped these so the verb-only output named the inner command
    (``regress`` rather than ``capture``). Under the denylist
    posture the FULL command line forwards, so the model sees both
    the wrapper and the inner command verbatim — no unwrap needed."""
    log_capture_noisily = (
        ". capture noisily regress y x_missing\n"
        "variable x_missing not found\n"
        "r(111);\n"
        "\n"
        "end of do-file\n"
        "\n"
        "r(111);\n"
    )
    excerpt = extract_debug_excerpt(log_capture_noisily, "", 111, "Stata")
    assert excerpt is not None
    # Whole command line surfaces, including wrappers.
    assert ". capture noisily regress y x_missing" in excerpt
    assert "variable x_missing not found" in excerpt
    assert "r(111);" in excerpt


def test_stata_short_form_modifier_command_forwards_full_line() -> None:
    """Same forwarding for the short-form wrappers (``cap``, ``qui``,
    ``noi``). The full command line including the wrapper passes
    through; the body names the missing variable."""
    log = (
        ". qui summarize bad_var\n"
        "variable bad_var not found\n"
        "r(111);\n"
        "\n"
        "end of do-file\n"
        "\n"
        "r(111);\n"
    )
    excerpt = extract_debug_excerpt(log, "", 111, "Stata")
    assert excerpt is not None
    assert ". qui summarize bad_var" in excerpt
    assert "variable bad_var not found" in excerpt


def test_stata_no_command_echo_returns_body_and_rc() -> None:
    """If the executor truncated the log such that the failing
    command isn't present, the body still forwards alongside the
    rc line — the model gets at least the diagnostic text
    ("syntax error") even without a command line to anchor on."""
    log = (
        "syntax error\n"
        "r(198);\n"
    )
    excerpt = extract_debug_excerpt(log, "", 198, "Stata")
    assert excerpt is not None
    assert "r(198);" in excerpt


def test_stata_invalid_varname_surfaces_long_identifier() -> None:
    """The denylist posture's main motivating case. A model-
    constructed varname that overflows Stata's 32-char cap
    triggers ``r(198) invalid varname`` and the failing identifier
    must be visible so the model can shorten it without re-probing.
    Before the relaxation the model saw ``. joinby [args redacted]``
    + ``[message body redacted]`` and could not tell which name to
    fix; now both the command line and the body forward."""
    log = (
        "_NORA_STATA_PREAMBLE_END_MARKER_\n\n"
        ". use orgyears.dta, clear\n\n"
        ". joinby ein using `orgyears'\n"
        "highest_forprofit_title_pre_ceo_rank invalid varname\n"
        "r(198);\n\n"
        "end of do-file\n\n"
        "r(198);\n"
    )
    excerpt = extract_debug_excerpt(log, "", 198, "Stata")
    assert excerpt is not None
    # Failing command line forwards in full.
    assert ". joinby ein using `orgyears'" in excerpt
    # The 36-char identifier the model needs to shorten is now
    # visible in the body.
    assert "highest_forprofit_title_pre_ceo_rank" in excerpt
    assert "invalid varname" in excerpt
    assert "r(198);" in excerpt


def test_r_long_identifier_in_body_surfaces() -> None:
    """The R analogue of the Stata invalid-varname case. A
    ``object 'X' not found`` body with a long identifier X must
    forward so the model can rename / add the column. Pre-denylist
    the body was redacted wholesale and the model only saw
    ``Error : [message body redacted]``."""
    stderr = (
        "Error in eval(predvars, data, env) : "
        "object 'highest_forprofit_title_pre_ceo_rank' not found\n"
        "Calls: lm -> eval -> eval\n"
    )
    excerpt = extract_debug_excerpt("", stderr, 1, "R")
    assert excerpt is not None
    assert "highest_forprofit_title_pre_ceo_rank" in excerpt
    assert "object" in excerpt and "not found" in excerpt
    assert "Calls: lm -> eval -> eval" in excerpt


def test_stata_body_length_cap_truncates_verbose_diagnostic() -> None:
    """Stata estimators occasionally emit multi-line diagnostics
    that run to several hundred chars. The 200-char per-body cap
    in ``_forward_short_body`` truncates without losing the head
    of the message — the model still sees the failure type."""
    long_diag = "convergence not achieved; " + ("residual deviance increased; " * 30)
    log = (
        ". glm y x, family(poisson)\n"
        f"{long_diag}\n"
        "r(430);\n"
        "\n"
        "end of do-file\n"
        "\n"
        "r(430);\n"
    )
    excerpt = extract_debug_excerpt(log, "", 430, "Stata")
    assert excerpt is not None
    # The first chunk of the diagnostic is present.
    assert "convergence not achieved" in excerpt
    # The truncation marker is present.
    assert "[...]" in excerpt
    # The whole long diagnostic is NOT echoed — the cap fired.
    assert long_diag not in excerpt


def test_r_data_shaped_body_redacted() -> None:
    """``stop(paste(df$row, collapse=","))``-style exfil where the
    body matches the multi-cell row shape gets caught by
    ``_body_looks_data_shaped`` and replaced with the data-shape
    marker. The detector flags the run because at least one token
    is non-identifier-shape (numbers, quoted strings, a
    name-with-space, a date) — the canonical row-dump fingerprint.
    Pure-identifier lists pass through, see
    ``test_r_identifier_list_body_forwards`` below."""
    # Realistic ``str(df.iloc[0])`` shape: mixed numbers, names,
    # dates, identifiers. At least one token violates the identifier
    # alphabet (the space in "John Smith", the dash in "1985-01-01",
    # the leading digits "42" and "100000").
    row_dump = "42, John Smith, 1985-01-01, 100000, doctor, NY, 12345"
    stderr = f"Error in stop(...) : {row_dump}\n"
    excerpt = extract_debug_excerpt("", stderr, 1, "R")
    assert excerpt is not None
    assert "John Smith" not in excerpt
    assert "100000" not in excerpt
    assert "message body suppressed: looked data-shaped" in excerpt


def test_r_identifier_list_body_forwards() -> None:
    """The refinement to ``_body_looks_data_shaped``: a comma-
    separated list of pure-identifier tokens is legitimate error
    context — a Stata varlist, an R formula term list, a six-arg
    function call — not a row dump. Forwarding it gives the model
    the variable names it needs to fix the script. The data-shape
    rule fires only when at least one token in the list is
    non-identifier-shape (the canonical row-dump fingerprint)."""
    # Realistic R error with a many-arg function call. Before the
    # refinement, the 6+ comma-separated-tokens rule fired on this
    # and the model saw "[message body suppressed: looked data-shaped]"
    # instead of the missing-arg diagnostic.
    stderr = (
        "Error in pmin(a, b, c, d, e, f, g) : "
        "object 'a' not found\n"
    )
    excerpt = extract_debug_excerpt("", stderr, 1, "R")
    assert excerpt is not None
    # The call args forward — model can see which function was called
    # and what args were supplied.
    assert "pmin(a, b, c, d, e, f, g)" in excerpt
    assert "object 'a' not found" in excerpt
    assert "message body suppressed" not in excerpt


def test_stata_identifier_list_body_forwards() -> None:
    """Same refinement at the Stata layer. A comma-separated varlist
    in an error message must reach the model — these are exactly
    the variable names the script tried to use and the model has
    to read to fix the issue."""
    log = (
        ". collapse (mean) mpg, by(price, weight, length, displacement, gear_ratio, foreign)\n"
        "invalid: mpg, price, weight, length, displacement, gear_ratio, foreign\n"
        "r(198);\n\n"
        "end of do-file\n\n"
        "r(198);\n"
    )
    excerpt = extract_debug_excerpt(log, "", 198, "Stata")
    assert excerpt is not None
    # The body forwards even though it carries a 7-element
    # comma-separated identifier list.
    assert "invalid: mpg, price, weight" in excerpt
    assert "message body suppressed" not in excerpt


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_unrecognised_output_returns_none() -> None:
    """If we can't find any of the language's error patterns, the
    extractor must return None so the caller falls back to the
    generic message - better silent than wrong."""
    assert extract_debug_excerpt("", "", 1, "Python") is None
    assert extract_debug_excerpt("", "totally unrelated text", 1, "R") is None
    assert extract_debug_excerpt("just a log", "", 1, "Stata") is None


def test_unknown_language_returns_none() -> None:
    """Future-proofing: if a new language gets added without a
    matching extractor, we don't crash, we just return None."""
    assert extract_debug_excerpt("", "Error: foo", 1, "Julia") is None


def test_empty_inputs_return_none() -> None:
    assert extract_debug_excerpt("", "", 0, "Python") is None
    assert extract_debug_excerpt("", "", 0, "R") is None
    assert extract_debug_excerpt("", "", 0, "Stata") is None


def test_excerpt_respects_overall_cap() -> None:
    """Even if a language emits a huge error block, the final
    excerpt must respect the 1 KB hard cap."""
    huge = "Error in foo : " + ("padding " * 500)
    excerpt = extract_debug_excerpt("", huge, 1, "R")
    assert excerpt is not None
    assert len(excerpt) <= MAX_EXCERPT_BYTES
