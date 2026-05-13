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
    """R's ``Error`` anchor + the ``Calls:`` trailer survive. The
    call deparse (``in eval(predvars, data, env)``) and message
    body are both script-controlled and now redacted."""
    stderr = (
        "Loading required package: stats\n"
        "Error in eval(predvars, data, env) : object 'wage' not found\n"
        "Calls: lm -> eval -> eval\n"
        "Execution halted\n"
    )
    excerpt = extract_debug_excerpt("", stderr, 1, "R")
    assert excerpt is not None
    # The "Error :" anchor and "Calls:" trailer are parser-owned.
    assert "Error :" in excerpt
    assert "Calls: lm -> eval -> eval" in excerpt
    # Call deparse and body redacted.
    assert "eval(predvars" not in excerpt
    assert "object 'wage' not found" not in excerpt
    assert "[message body redacted]" in excerpt
    # "Execution halted" trailer is noise — still dropped.
    assert "Execution halted" not in excerpt


def test_r_multiline_error_message_redacted() -> None:
    """Multi-line error message bodies are redacted regardless of
    their wrap. The Calls: trailer remains as parser-owned framing."""
    stderr = (
        "Error in lm.fit(x, y, offset = offset, singular.ok = singular.ok, ...) : \n"
        "  NA/NaN/Inf in 'x'\n"
        "Calls: lm -> lm.fit\n"
    )
    excerpt = extract_debug_excerpt("", stderr, 1, "R")
    assert excerpt is not None
    assert "NA/NaN/Inf in 'x'" not in excerpt
    assert "lm.fit(x, y" not in excerpt
    assert "Calls: lm -> lm.fit" in excerpt
    assert "Error :" in excerpt


def test_r_only_last_error_is_returned() -> None:
    """If a script logs multiple errors (e.g., recovered errors
    inside ``tryCatch``), only the LAST top-level one's Calls
    trailer remains. Both error bodies are redacted."""
    stderr = (
        "Error in foo() : early problem\n"
        "Error in bar() : the actual cause\n"
        "Calls: bar -> baz\n"
    )
    excerpt = extract_debug_excerpt("", stderr, 1, "R")
    assert excerpt is not None
    # Both bodies redacted.
    assert "the actual cause" not in excerpt
    assert "early problem" not in excerpt
    # The Calls trailer of the LAST error survives.
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


def test_stata_extract_anchors_on_rc_and_command_verb() -> None:
    """The failing command's verb (``regress``) and the rc line are
    parser-owned framing and pass through. The command's arguments
    (``y x_missing``) and the error message body (``variable
    x_missing not found``) are script-controlled and redacted —
    a macro-expanded raw value in the args used to ride that
    channel directly to the model."""
    excerpt = extract_debug_excerpt(_STATA_LOG_TYPICAL, "", 111, "Stata")
    assert excerpt is not None
    # Verb survives.
    assert ". regress" in excerpt
    # Args dropped.
    assert "x_missing" not in excerpt
    # Error body dropped.
    assert "variable x_missing not found" not in excerpt
    assert "[message body redacted]" in excerpt
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


def test_stata_unwraps_modifier_prefixes_to_underlying_verb() -> None:
    """Stata wrappers like ``capture``, ``quietly``, ``noisily`` take
    another command as their body. Reporting the wrapper as the
    failing verb (``capture``) is useless to the model — the
    actionable information is the inner command (``regress``).
    The extractor unwraps known modifier prefixes iteratively
    before picking the verb."""
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
    assert ". regress" in excerpt, (
        f"verb should unwrap to the inner command, got: {excerpt!r}"
    )
    assert "capture" not in excerpt
    assert "noisily" not in excerpt
    # Still no args / message body.
    assert "x_missing" not in excerpt
    assert "[message body redacted]" in excerpt
    assert "r(111);" in excerpt


def test_stata_unwraps_short_form_modifier_prefixes() -> None:
    """Stata accepts short forms ``cap`` / ``qui`` / ``noi``. The
    unwrapper covers those too — a script that hits an error via
    ``qui summarize x`` should report ``summarize``, not ``qui``."""
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
    assert ". summarize" in excerpt
    assert "qui" not in excerpt
    assert "bad_var" not in excerpt


def test_stata_modifier_only_command_redacts_completely() -> None:
    """A pathological echo with ONLY a wrapper and nothing after
    (the script body got cut by extraction) should fail closed:
    no verb gets through. Without this guard, the wrapper itself
    would survive as the verb after the strip loop empties the
    token list."""
    log = (
        ". capture\n"
        "r(198);\n"
        "\n"
        "end of do-file\n"
        "\n"
        "r(198);\n"
    )
    excerpt = extract_debug_excerpt(log, "", 198, "Stata")
    assert excerpt is not None
    # No verb survives.
    assert ". " not in excerpt or "[command body redacted]" in excerpt
    assert "capture" not in excerpt
    assert "r(198);" in excerpt


def test_stata_no_command_echo_returns_rc_only() -> None:
    """If the executor truncated the log such that the failing
    command isn't present, return the rc line only. The error
    message body (``syntax error``) is data-controlled too and
    is no longer forwarded — the rc code carries enough framing
    for the model to know the failure class."""
    log = (
        "syntax error\n"
        "r(198);\n"
    )
    excerpt = extract_debug_excerpt(log, "", 198, "Stata")
    assert excerpt is not None
    assert "syntax error" not in excerpt
    assert "r(198);" in excerpt


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
