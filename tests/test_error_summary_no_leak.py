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
    into the excerpt. The Stata extractor now keeps only the
    failing command's verb plus the ``r(<code>);`` line, so neither
    pre-failure display output nor the post-cmd error message can
    cross."""
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
    # The failing command's args and error body are redacted; only
    # the verb and rc line survive.
    assert "x_missing" not in excerpt
    assert "variable x_missing not found" not in excerpt
    assert ". regress" in excerpt
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

def test_python_oversized_quoted_arg_is_redacted() -> None:
    """A ValueError whose message embeds a multi-KB repr (because
    pandas formatted the offending row) must not be forwarded
    verbatim. The exception body is now redacted wholesale — the
    blob never reaches the excerpt at all, no per-arg truncation
    needed."""
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
    # Body redacted.
    assert "[message body redacted]" in excerpt


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


def test_r_stop_exfil_message_is_redacted() -> None:
    """``stop(paste(df$secret, collapse=','))`` would otherwise
    pour a row of comma-separated cell values into the message
    body. The body is now redacted wholesale — no shape detection
    or cap-based partial leak."""
    secret_row = ",".join(f"value_{i}" for i in range(20))
    stderr = f"Error in stop(...) : {secret_row}\n"
    excerpt = extract_debug_excerpt("", stderr, 1, "R")
    assert excerpt is not None
    # The full 20-cell row is not present.
    assert secret_row not in excerpt
    # The "Error :" anchor is the parser-owned framing — call
    # deparse ``in stop(...)`` is also script-controlled and dropped.
    assert "Error :" in excerpt
    assert "stop(" not in excerpt


def test_python_long_unquoted_message_is_redacted() -> None:
    """Long unquoted message bodies are redacted wholesale — no
    per-arg truncation or quote-shape required."""
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
    assert "ValueError" in excerpt
    assert "[message body redacted]" in excerpt


def test_short_python_keyerror_body_redacted() -> None:
    """Previous versions allowed short non-data-shaped exception
    bodies to pass through the cap. A model-authored script could
    ``raise KeyError(df.loc[0, 'name'])`` and read the cell back
    under the 80-byte ceiling. The cap is gone; the exception body
    is now redacted on its line. (The source-line preview ``df[
    'typo']`` is the model's own script source — Python's traceback
    formatter echoes it back, and the model wrote it.)"""
    stderr = (
        'Traceback (most recent call last):\n'
        '  File "/abs/x.py", line 3, in <module>\n'
        "    df['typo']\n"
        "KeyError: 'typo'\n"
    )
    excerpt = extract_debug_excerpt("", stderr, 1, "Python")
    assert excerpt is not None
    last_line = excerpt.strip().splitlines()[-1]
    # Body redacted on the exception line — the key value 'typo'
    # no longer surfaces THROUGH the exception body channel.
    assert "'typo'" not in last_line
    assert last_line.startswith("KeyError")
    assert "[message body redacted]" in last_line


def test_short_r_object_not_found_body_redacted() -> None:
    """Same posture for R: a short ``object 'wage' not found`` body
    used to pass the cap. It now doesn't reach the excerpt — the
    model has the script source and knows which symbol was
    referenced without us forwarding it."""
    stderr = (
        "Error in eval(predvars, data, env) : object 'wage' not found\n"
    )
    excerpt = extract_debug_excerpt("", stderr, 1, "R")
    assert excerpt is not None
    assert "'wage'" not in excerpt
    assert "Error :" in excerpt
    assert "[message body redacted]" in excerpt


def test_python_filenotfound_body_redacted() -> None:
    """``FileNotFoundError: [Errno 2] No such file: 'foo.csv'``
    surfaces the type only — the filename in the body is data-
    controlled (a model can ``open(df.loc[0, 'secret'])``)."""
    stderr = (
        'Traceback (most recent call last):\n'
        '  File "/abs/x.py", line 1, in <module>\n'
        "    open('foo.csv')\n"
        "FileNotFoundError: [Errno 2] No such file or directory: 'foo.csv'\n"
    )
    excerpt = extract_debug_excerpt("", stderr, 1, "Python")
    assert excerpt is not None
    assert "FileNotFoundError" in excerpt
    # The literal 'foo.csv' that appeared in the body is gone;
    # the source-line preview that shows ``open('foo.csv')`` is
    # the model's own script source, so its quoted-path basename
    # form remains.
    assert "[message body redacted]" in excerpt


# ---------------------------------------------------------------------------
# Credentials embedded in raw output
# ---------------------------------------------------------------------------

def test_openai_key_in_traceback_is_redacted() -> None:
    """``print(os.environ)`` followed by a crash is the canonical
    foot-gun. With the new body-redaction posture, the key never
    even reaches the credential scrub — it's dropped wholesale
    with the rest of the exception body. The earlier marker
    ``[redacted-credential]`` is therefore not generated; the
    important guarantee is that the key string itself is gone."""
    stderr = (
        'Traceback (most recent call last):\n'
        '  File "/abs/x.py", line 3, in <module>\n'
        '    raise RuntimeError(os.environ["OPENAI_API_KEY"])\n'
        f"RuntimeError: {_OPENAI_KEY}\n"
    )
    excerpt = extract_debug_excerpt("", stderr, 1, "Python")
    assert excerpt is not None
    assert _OPENAI_KEY not in excerpt
    # The body is now redacted wholesale, so the credential never
    # reaches the per-token credential scrub.
    assert "[message body redacted]" in excerpt


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


def test_bare_absolute_path_in_message_is_redacted() -> None:
    """R / Stata error messages sometimes embed a bare absolute
    path (e.g., the .dta file that couldn't open). With body
    redaction, the entire path — including the filename, which is
    data-derived (a model can construct a script that ``read_dta(
    df.loc[0, 'secret'])``) — never reaches the excerpt."""
    stderr = 'Error in read_dta : file /Users/jdoe/private/secrets.dta not found\n'
    excerpt = extract_debug_excerpt("", stderr, 1, "R")
    assert excerpt is not None
    assert "/Users/jdoe/private" not in excerpt
    assert "secrets.dta" not in excerpt
    assert "Error :" in excerpt


def test_path_with_space_in_username_is_fully_scrubbed() -> None:
    """macOS / Windows users with names containing spaces are common
    (``John Smith``, ``Mary O'Brien Lopez``). The bare-path regex
    used to stop at the first space and only basename ``/Users/John``,
    leaving ``Smith/research/wages.csv`` in the excerpt — leaking the
    surname AND the substantive path. With the wholesale exception-body
    redaction the leak is closed at a different layer (the body is
    gone entirely), but the regex fix in ``_scrub_and_cap`` still
    matters as defense-in-depth for any surface that survives the
    redaction (R ``Calls:`` chain, residual frame lines)."""
    stderr = (
        "FileNotFoundError: [Errno 2] No such file or directory: "
        "/Users/John Smith/research/wages.csv\n"
    )
    # Wrap so the Python extractor anchors on a real traceback.
    stderr = (
        "Traceback (most recent call last):\n"
        '  File "/abs/x.py", line 1, in <module>\n'
        "    pd.read_csv(path)\n"
        + stderr
    )
    excerpt = extract_debug_excerpt("", stderr, 1, "Python")
    assert excerpt is not None
    # No part of the user-controlled path may survive — both the
    # surname-as-directory-component AND the substantive path body.
    assert "/Users/John" not in excerpt
    assert "Smith" not in excerpt
    assert "research" not in excerpt
    assert "wages.csv" not in excerpt
    # The exception type still surfaces so the model knows what
    # happened; only the body (which carried the path) is redacted.
    assert "FileNotFoundError" in excerpt


def test_github_personal_access_token_redacted() -> None:
    """A GitHub PAT in stderr (e.g., a request library printed the
    auth header on a 401) must be redacted. The classic ``ghp_``
    prefix slipped past the original credential pattern set, which
    only covered ``sk-...``, ``AKIA``, and JWTs.

    Two layers protect now: (1) the wholesale exception-body
    redaction drops the body entirely (so the PAT goes with it),
    and (2) ``_CRED_PATTERNS`` still applies to any surface that
    survives — Stata logs, R ``Calls:`` chains. Either layer alone
    is enough; both together is defense in depth."""
    pat = "ghp_" + "A" * 36
    stderr = (
        "Traceback (most recent call last):\n"
        '  File "/x.py", line 1, in <module>\n'
        '    raise RuntimeError(headers["Authorization"])\n'
        f"RuntimeError: token {pat}\n"
    )
    excerpt = extract_debug_excerpt("", stderr, 1, "Python")
    assert excerpt is not None
    assert pat not in excerpt


def test_slack_token_redacted() -> None:
    token = "xoxb-1234567890-1234567890-AbCdEfGhIjKlMnOpQrStUv"
    stderr = (
        "Traceback (most recent call last):\n"
        '  File "/x.py", line 1, in <module>\n'
        '    raise RuntimeError("posted")\n'
        f"RuntimeError: webhook {token}\n"
    )
    excerpt = extract_debug_excerpt("", stderr, 1, "Python")
    assert excerpt is not None
    assert token not in excerpt


def test_huggingface_token_redacted() -> None:
    token = "hf_" + "A" * 30
    stderr = (
        "Traceback (most recent call last):\n"
        '  File "/x.py", line 1, in <module>\n'
        '    raise RuntimeError("hf")\n'
        f"RuntimeError: token={token}\n"
    )
    excerpt = extract_debug_excerpt("", stderr, 1, "Python")
    assert excerpt is not None
    assert token not in excerpt


def test_stripe_underscore_key_redacted() -> None:
    """The original ``sk-`` regex requires a hyphen; Stripe uses
    underscores (``sk_live_...`` / ``sk_test_...``). Pin the
    underscore variant so a researcher's STRIPE_KEY env var
    leaking via a printed header doesn't slip through."""
    key = "sk_live_" + "A" * 24
    stderr = (
        "Traceback (most recent call last):\n"
        '  File "/x.py", line 1, in <module>\n'
        '    raise RuntimeError("billing")\n'
        f"RuntimeError: STRIPE_KEY={key}\n"
    )
    excerpt = extract_debug_excerpt("", stderr, 1, "Python")
    assert excerpt is not None
    assert key not in excerpt


def test_r_call_argument_credential_is_redacted() -> None:
    """R errors of the form ``Error in some_func("user-supplied") : ...``
    interpolate the call argument verbatim. Earlier extractor versions
    forwarded the call group raw, leaking short data values that sit
    under the long-quoted-arg threshold (SSNs, IDs, GH PATs from
    header dumps). The current extractor drops the whole body
    wholesale ("Error : [message body redacted]"), so the call group
    — and any credential it carried — never reaches the excerpt."""
    pat = "ghp_" + "B" * 36
    r_stderr = f'Error in httr::GET("api", token = "{pat}") : 401 Unauthorized\n'
    excerpt = extract_debug_excerpt("", r_stderr, 1, "R")
    assert excerpt is not None
    assert pat not in excerpt
    # The whole body (including the call group's argument) is
    # redacted; the parser-owned anchor still surfaces.
    assert "[message body redacted]" in excerpt
    assert "Error" in excerpt


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
    must never bleed into the excerpt. The error message body
    (``NA in design matrix``) is also dropped under the new
    posture — only ``Error :`` framing remains."""
    stdout = f"printed row: {_PII_ROW}\n"
    stderr = "Error in lm.fit : NA in design matrix\n"
    excerpt = extract_debug_excerpt(stdout, stderr, 1, "R")
    assert excerpt is not None
    assert _PII_ROW not in excerpt
    assert "Error :" in excerpt
    assert "NA in design matrix" not in excerpt


def test_url_embedded_userinfo_credentials_are_redacted() -> None:
    """SDC closure for the install-path leak: pip echoes the index URL
    from ``~/.pip/pip.conf`` (or ``PIP_INDEX_URL``) on every run,
    embedding any ``user:token@`` segment that file contains. The
    scrubber must redact the userinfo while preserving the scheme so
    a reader still sees "this was a URL" without seeing the
    credentials. The same regex also protects an R / Python error
    message that happens to print a token-bearing repository URL.

    The check runs through ``extract_debug_excerpt`` (the public
    entry that ``install_packages``' wrapper ``scrub_raw_output``
    shares) to keep the contract pinned at the chokepoint, not at
    the regex.
    """
    from nora.error_summary import scrub_raw_output
    raw = (
        "ERROR: Could not find a version\n"
        "Looking in indexes: https://user_alice:tok-abc123XYZ@"
        "private-pypi.acme.com/simple\n"
    )
    out = scrub_raw_output(raw, cap_bytes=500)
    assert "user_alice" not in out
    assert "tok-abc123XYZ" not in out
    assert "[redacted-credential]" in out
    # Scheme and host survive — they're useful for diagnosis and
    # neither is a credential.
    assert "https://" in out
    assert "private-pypi.acme.com" in out
