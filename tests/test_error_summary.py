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

def test_python_keyerror_keeps_column_name() -> None:
    """The whole point of forwarding the message - a researcher
    typo'd a column name, and Nora needs to know which one to
    propose the fix."""
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
    assert "KeyError: 'typo'" in excerpt
    # The user-code frame survives; the pandas internals frame is dropped.
    assert 'line 17' in excerpt
    assert 'pandas' not in excerpt


def test_python_traceback_path_is_basenamed() -> None:
    """Absolute paths in tracebacks get reduced to basename - the
    researcher's home directory layout doesn't leak, but the line
    number does (which is what the model needs)."""
    stderr = (
        'Traceback (most recent call last):\n'
        '  File "/Users/bb/.nora-sessions/2026/regression.py", line 42, in main\n'
        '    1 / 0\n'
        'ZeroDivisionError: division by zero\n'
    )
    excerpt = extract_debug_excerpt("", stderr, 1, "Python")
    assert excerpt is not None
    assert "/Users/bb/.nora-sessions" not in excerpt
    assert '"regression.py"' in excerpt
    assert 'line 42' in excerpt


def test_python_includes_source_line() -> None:
    """The traceback's source-line preview (the indented line under
    the frame) is the most actionable thing in the excerpt - keep
    it."""
    stderr = (
        'Traceback (most recent call last):\n'
        '  File "/abs/run.py", line 9, in <module>\n'
        '    result = compute_things(df)\n'
        'NameError: name \'compute_things\' is not defined\n'
    )
    excerpt = extract_debug_excerpt("", stderr, 1, "Python")
    assert excerpt is not None
    assert "result = compute_things(df)" in excerpt
    assert "NameError: name 'compute_things' is not defined" in excerpt


def test_python_chained_exceptions_keeps_last_one() -> None:
    """When Python raises during except-handling, both exceptions
    appear. The LAST one is the propagating one - that's what the
    model should see."""
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
    assert 'RuntimeError: retry failed' in excerpt


# ---------------------------------------------------------------------------
# R
# ---------------------------------------------------------------------------

def test_r_error_block_with_calls_chain() -> None:
    """R's ``Error in ... :`` block + the ``Calls:`` trailer is
    exactly what shows on a researcher's R console. Keep both."""
    stderr = (
        "Loading required package: stats\n"
        "Error in eval(predvars, data, env) : object 'wage' not found\n"
        "Calls: lm -> eval -> eval\n"
        "Execution halted\n"
    )
    excerpt = extract_debug_excerpt("", stderr, 1, "R")
    assert excerpt is not None
    assert "Error in eval(predvars, data, env)" in excerpt
    assert "object 'wage' not found" in excerpt
    assert "Calls: lm -> eval -> eval" in excerpt
    # "Execution halted" trailer is noise - drop it.
    assert "Execution halted" not in excerpt


def test_r_multiline_error_message_preserved() -> None:
    """Some R errors wrap onto a second line. The extractor must
    pick up the whole logical block."""
    stderr = (
        "Error in lm.fit(x, y, offset = offset, singular.ok = singular.ok, ...) : \n"
        "  NA/NaN/Inf in 'x'\n"
        "Calls: lm -> lm.fit\n"
    )
    excerpt = extract_debug_excerpt("", stderr, 1, "R")
    assert excerpt is not None
    assert "NA/NaN/Inf in 'x'" in excerpt
    assert "Calls: lm -> lm.fit" in excerpt


def test_r_only_last_error_is_returned() -> None:
    """If a script logs multiple errors (e.g., recovered errors
    inside ``tryCatch``), only the LAST top-level one matters -
    that's what propagated."""
    stderr = (
        "Error in foo() : early problem\n"
        "Error in bar() : the actual cause\n"
        "Calls: bar -> baz\n"
    )
    excerpt = extract_debug_excerpt("", stderr, 1, "R")
    assert excerpt is not None
    assert "the actual cause" in excerpt
    # The earlier "early problem" shouldn't be in the excerpt.
    assert "early problem" not in excerpt


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


def test_stata_extract_anchors_on_rc_and_command_echo() -> None:
    excerpt = extract_debug_excerpt(_STATA_LOG_TYPICAL, "", 111, "Stata")
    assert excerpt is not None
    assert ". regress y x_missing" in excerpt
    assert "variable x_missing not found" in excerpt
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


def test_stata_no_command_echo_falls_back_to_message_lines() -> None:
    """If the executor truncated the log such that the failing
    command isn't present, still surface the error message + rc
    so the model has something to work with."""
    log = (
        "syntax error\n"
        "r(198);\n"
    )
    excerpt = extract_debug_excerpt(log, "", 198, "Stata")
    assert excerpt is not None
    assert "syntax error" in excerpt
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
