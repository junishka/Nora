"""SDC boundary regressions for the new ``debug_excerpt`` channel.

The debug_excerpt is the FIRST channel that ever forwards
stdout/stderr bytes from a researcher's script to the model. Every
test in this file plants a recognisable secret in the raw output
and asserts the extractor does NOT forward it.

Categories covered:

  - **Stdout leakage (Stata).** Stata batch-mode merges its log
    into stdout, which is the one stream the Stata extractor
    reads. ``display`` / ``list`` / pandas-style data prints
    BEFORE the failing command must not appear in the excerpt.
  - **Long quoted blobs.** A ``ValueError`` whose message embeds
    a 5 KB pandas repr (the row that failed to parse) must be
    truncated, not forwarded verbatim.
  - **Credentials in tracebacks / error messages.** The classic
    ``print(os.environ)`` foot-gun: an Anthropic / OpenAI / AWS
    key in stderr must be redacted.
  - **Absolute path normalisation.** Researcher's home-directory
    layout (and any path prefix) reduced to basename.
  - **Stdout NEVER read for R / Python.** Even if a researcher
    prints data to stdout and the script then crashes, the R /
    Python extractors only see stderr.

Each test names the threat in its docstring so a future
contributor weakening the regex sees what break.
"""

from __future__ import annotations

from nora.error_summary import (
    MAX_EXCERPT_BYTES,
    MAX_QUOTED_ARG_BYTES,
    extract_debug_excerpt,
)


# Recognisable secrets we plant in raw output. None of these
# should appear in the extractor's return value.
_PII_ROW = "patient_42_id=12345 ssn=123-45-6789 dob=1980-01-15"
_PII_VALUE = "George Washington Adams III"  # plausible PII, short
_LARGE_BLOB = "row=" + ("X" * 4000)
_OPENAI_KEY = "sk-proj-aBcDeFgHiJkLmNoPqRsTuV0123456789"
_ANTHROPIC_KEY = "sk-ant-api03-aBcDeFgHiJkLmNoPqRsTuV0123456789"
_AWS_KEY = "AKIAIOSFODNN7EXAMPLE"
_JWT = (
    "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NSIsIm5hbWUiOiJKb2huIn0."
    "abc123def456ghijklmnop"
)


# ---------------------------------------------------------------------------
# Stdout leakage - Stata's `display` / `list` content before failure
# ---------------------------------------------------------------------------

def test_stata_display_output_before_failure_is_not_forwarded() -> None:
    """A researcher who runs ``display var`` followed by a
    failing regression must not have the displayed values bleed
    into the excerpt. The Stata extractor anchors on the LAST
    ``r(<code>);`` and walks back to the most-recent ``. <cmd>``
    - anything earlier is excluded by construction."""
    log = f"""\
. set more off

. use "panel.dta", clear

. display "secret_token: {_PII_ROW}"
secret_token: {_PII_ROW}

. list patient_id age in 1/3
{_PII_VALUE}, 45
{_PII_VALUE}, 52
{_PII_VALUE}, 38

. regress y x_missing
variable x_missing not found
r(111);

end of do-file

r(111);
"""
    excerpt = extract_debug_excerpt(log, "", 111, "Stata")
    assert excerpt is not None
    assert _PII_ROW not in excerpt
    assert _PII_VALUE not in excerpt
    assert "secret_token" not in excerpt
    # Sanity: the actual error IS forwarded.
    assert "x_missing not found" in excerpt
    assert "r(111);" in excerpt


def test_stata_list_dump_inside_failing_command_block_is_bounded() -> None:
    """Even if a researcher's failing command line spans a
    multi-row dump (Stata sometimes echoes data right next to a
    syntax error), the cap keeps blast radius bounded. We don't
    promise zero inclusion of "between echo and rc" content -
    that's where the real error message lives - but we do
    promise the 1 KB ceiling."""
    long_dump = "\n".join(f"row{i}: secret_data_value_{i}" for i in range(200))
    log = f". list y x in 1/200\n{long_dump}\nr(2000);\n"
    excerpt = extract_debug_excerpt(log, "", 2000, "Stata")
    assert excerpt is not None
    assert len(excerpt) <= MAX_EXCERPT_BYTES


# ---------------------------------------------------------------------------
# Long quoted blob - Python ValueError with a 5 KB repr arg
# ---------------------------------------------------------------------------

def test_python_oversized_quoted_arg_gets_truncated() -> None:
    """A ValueError whose message embeds a multi-KB repr (because
    pandas formatted the offending row) must not be forwarded
    verbatim - the in-place truncation rule kicks in."""
    blob = "X" * (MAX_QUOTED_ARG_BYTES * 5)
    stderr = (
        'Traceback (most recent call last):\n'
        '  File "/abs/script.py", line 5, in <module>\n'
        '    parse(row)\n'
        f"ValueError: could not convert string '{blob}' to float\n"
    )
    excerpt = extract_debug_excerpt("", stderr, 1, "Python")
    assert excerpt is not None
    # The full blob must not appear.
    assert blob not in excerpt
    # ValueError type still surfaces.
    assert "ValueError" in excerpt
    # And the truncation is visible so the model knows something
    # was elided rather than guessing.
    assert "truncated" in excerpt


def test_extreme_oversize_message_capped_overall() -> None:
    """Belt-and-braces: even if the per-arg trim missed (say a
    pathological message without any quotes), the overall 1 KB
    cap is hit."""
    stderr = "Error in foo : " + ("data " * 5000)
    excerpt = extract_debug_excerpt("", stderr, 1, "R")
    assert excerpt is not None
    assert len(excerpt) <= MAX_EXCERPT_BYTES


# ---------------------------------------------------------------------------
# Exception-body data exfil — the channel a malicious script could use
# to ship cell content out via a hand-crafted ``raise``.
# ---------------------------------------------------------------------------

def test_python_to_json_exfil_via_runtimeerror_is_redacted() -> None:
    """``raise RuntimeError(df.iloc[0].to_json())`` would otherwise
    forward a JSON dump of a row through the exception body. The
    data-shape detector replaces the body with a redaction marker."""
    stderr = (
        'Traceback (most recent call last):\n'
        '  File "/abs/x.py", line 3, in <module>\n'
        '    raise RuntimeError(df.iloc[0].to_json())\n'
        'RuntimeError: {"patient_id":42,"ssn":"123-45-6789","name":"Alice"}\n'
    )
    excerpt = extract_debug_excerpt("", stderr, 1, "Python")
    assert excerpt is not None
    assert "patient_id" not in excerpt
    assert "123-45-6789" not in excerpt
    assert "Alice" not in excerpt
    # The exception type still surfaces so the model knows what
    # happened; only the body is suppressed.
    assert "RuntimeError" in excerpt


def test_r_stop_exfil_long_message_is_truncated() -> None:
    """``stop(paste(df$secret, collapse=','))`` would otherwise
    pour a row of comma-separated cell values into the message
    body. The data-shape detector + per-body cap together limit
    the leak."""
    secret_row = ",".join(f"value_{i}" for i in range(20))
    stderr = f"Error in stop(...) : {secret_row}\n"
    excerpt = extract_debug_excerpt("", stderr, 1, "R")
    assert excerpt is not None
    # The full 20-cell row is not present.
    assert secret_row not in excerpt
    # The framing IS — model still sees the call site.
    assert "Error in stop" in excerpt


def test_python_long_unquoted_message_is_capped() -> None:
    """A message body longer than ``MAX_EXCEPTION_MSG_BYTES`` is
    truncated even when it doesn't trip the data-shape detector
    (e.g. a long English sentence assembled from row values)."""
    long_body = "the offending row contains " + ("extremely_long_cell_value " * 20)
    stderr = (
        'Traceback (most recent call last):\n'
        '  File "/abs/x.py", line 1, in <module>\n'
        '    raise ValueError(msg)\n'
        f'ValueError: {long_body}\n'
    )
    excerpt = extract_debug_excerpt("", stderr, 1, "Python")
    assert excerpt is not None
    assert long_body not in excerpt
    assert "truncated" in excerpt


def test_short_python_keyerror_passes_through() -> None:
    """The cap must NOT break the legitimate use case: a short
    ``KeyError: 'typo'`` should still surface the column name so
    the model can fix the script."""
    stderr = (
        'Traceback (most recent call last):\n'
        '  File "/abs/x.py", line 3, in <module>\n'
        "    df['typo']\n"
        "KeyError: 'typo'\n"
    )
    excerpt = extract_debug_excerpt("", stderr, 1, "Python")
    assert excerpt is not None
    assert "'typo'" in excerpt
    assert "KeyError" in excerpt


def test_short_r_object_not_found_passes_through() -> None:
    """Same legitimate-use guarantee for R: a short
    ``object 'wage' not found`` body must still reach the model."""
    stderr = (
        "Error in eval(predvars, data, env) : object 'wage' not found\n"
    )
    excerpt = extract_debug_excerpt("", stderr, 1, "R")
    assert excerpt is not None
    assert "'wage'" in excerpt
    assert "Error in eval" in excerpt


def test_python_filenotfound_errno_pattern_passes_through() -> None:
    """``FileNotFoundError: [Errno 2] No such file: 'foo.csv'`` is
    a common, legitimate pattern. The data-shape regex must not
    treat ``[Errno 2]`` as a data dump."""
    stderr = (
        'Traceback (most recent call last):\n'
        '  File "/abs/x.py", line 1, in <module>\n'
        "    open('foo.csv')\n"
        "FileNotFoundError: [Errno 2] No such file or directory: 'foo.csv'\n"
    )
    excerpt = extract_debug_excerpt("", stderr, 1, "Python")
    assert excerpt is not None
    assert "FileNotFoundError" in excerpt
    assert "foo.csv" in excerpt


# ---------------------------------------------------------------------------
# Credentials embedded in raw output
# ---------------------------------------------------------------------------

def test_openai_key_in_traceback_is_redacted() -> None:
    """``print(os.environ)`` followed by a crash is the canonical
    foot-gun. The extractor's credential scrub must catch
    Anthropic / OpenAI / AWS / JWT shapes regardless of where they
    appear in the stderr text."""
    stderr = (
        'Traceback (most recent call last):\n'
        '  File "/abs/x.py", line 3, in <module>\n'
        '    raise RuntimeError(os.environ["OPENAI_API_KEY"])\n'
        f"RuntimeError: {_OPENAI_KEY}\n"
    )
    excerpt = extract_debug_excerpt("", stderr, 1, "Python")
    assert excerpt is not None
    assert _OPENAI_KEY not in excerpt
    assert "[redacted-credential]" in excerpt


def test_anthropic_key_redacted() -> None:
    stderr = f'RuntimeError: leaked key {_ANTHROPIC_KEY} via env\n'
    # Wrap in a minimal traceback so the python extractor pulls it.
    stderr = (
        'Traceback (most recent call last):\n'
        '  File "/x.py", line 1, in <module>\n'
        '    raise RuntimeError("leaked")\n'
        + stderr
    )
    excerpt = extract_debug_excerpt("", stderr, 1, "Python")
    assert excerpt is not None
    assert _ANTHROPIC_KEY not in excerpt


def test_aws_access_key_redacted() -> None:
    stderr = (
        'Traceback (most recent call last):\n'
        '  File "/x.py", line 1, in <module>\n'
        '    raise ValueError("env: " + os.environ["AWS_ACCESS_KEY_ID"])\n'
        f'ValueError: env: {_AWS_KEY}\n'
    )
    excerpt = extract_debug_excerpt("", stderr, 1, "Python")
    assert excerpt is not None
    assert _AWS_KEY not in excerpt


def test_jwt_redacted() -> None:
    stderr = (
        'Traceback (most recent call last):\n'
        '  File "/x.py", line 1, in <module>\n'
        '    raise RuntimeError("token")\n'
        f'RuntimeError: token={_JWT}\n'
    )
    excerpt = extract_debug_excerpt("", stderr, 1, "Python")
    assert excerpt is not None
    assert _JWT not in excerpt


def test_credential_scrub_runs_for_r_and_stata_too() -> None:
    """The credential scrub is in the common ``_scrub_and_cap``
    pipeline; it must apply regardless of which language the
    error came from."""
    r_stderr = f'Error in foo : credential = "{_OPENAI_KEY}"\n'
    excerpt = extract_debug_excerpt("", r_stderr, 1, "R")
    assert excerpt is not None
    assert _OPENAI_KEY not in excerpt

    stata_log = (
        f'. display "credential={_AWS_KEY}"\n'
        f'credential={_AWS_KEY}\n'
        f'. regress y "{_AWS_KEY}"\n'
        f'invalid varname\n'
        f'r(198);\n'
    )
    excerpt = extract_debug_excerpt(stata_log, "", 198, "Stata")
    assert excerpt is not None
    assert _AWS_KEY not in excerpt


# ---------------------------------------------------------------------------
# Path normalisation
# ---------------------------------------------------------------------------

def test_python_traceback_strips_home_directory_prefix() -> None:
    """Researcher home-directory layout shouldn't leak. We keep
    the basename and line number - that's what the model needs to
    locate the failure."""
    stderr = (
        'Traceback (most recent call last):\n'
        '  File "/Users/jdoe/.nora-sessions/2026/regression.py", line 17, in <module>\n'
        '    df["x"] / 0\n'
        'ZeroDivisionError: division by zero\n'
    )
    excerpt = extract_debug_excerpt("", stderr, 1, "Python")
    assert excerpt is not None
    assert "/Users/jdoe" not in excerpt
    assert ".nora-sessions" not in excerpt
    assert '"regression.py"' in excerpt
    assert "line 17" in excerpt


def test_bare_absolute_path_in_message_is_basenamed() -> None:
    """R / Stata error messages sometimes embed a bare absolute
    path (e.g., the .dta file that couldn't open). The
    home-directory prefix is dropped; the filename remains."""
    stderr = 'Error in read_dta : file /Users/jdoe/private/secrets.dta not found\n'
    excerpt = extract_debug_excerpt("", stderr, 1, "R")
    assert excerpt is not None
    assert "/Users/jdoe/private" not in excerpt
    assert "secrets.dta" in excerpt


# ---------------------------------------------------------------------------
# stdout NEVER read for R / Python
# ---------------------------------------------------------------------------

def test_python_extractor_ignores_stdout_entirely() -> None:
    """Whatever the researcher printed to stdout - including the
    full contents of a sensitive DataFrame - must never reach the
    excerpt. The Python extractor only reads stderr."""
    stdout = f"the row that broke things: {_PII_ROW}\n"
    # stderr has the actual traceback, no canary.
    stderr = (
        'Traceback (most recent call last):\n'
        '  File "/abs/x.py", line 1, in <module>\n'
        '    1/0\n'
        'ZeroDivisionError: division by zero\n'
    )
    excerpt = extract_debug_excerpt(stdout, stderr, 1, "Python")
    assert excerpt is not None
    assert _PII_ROW not in excerpt
    assert "ZeroDivisionError" in excerpt


def test_r_extractor_ignores_stdout_entirely() -> None:
    """Same boundary for R: a ``cat()`` / ``print(df)`` to stdout
    must never bleed into the excerpt."""
    stdout = f"printed row: {_PII_ROW}\n"
    stderr = "Error in lm.fit : NA in design matrix\n"
    excerpt = extract_debug_excerpt(stdout, stderr, 1, "R")
    assert excerpt is not None
    assert _PII_ROW not in excerpt
    assert "NA in design matrix" in excerpt
