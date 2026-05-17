"""Tests for phase-aware redaction in ``error_summary.extract_debug_excerpt``.

The boundary is enforced by the executor's buffer-split stderr
capture: the preamble ``dup2``'s fd 2 onto ``stderr.phase_a`` at
startup and onto ``stderr.phase_b`` just before user code. The
extractor consumes the two buffers via the
``pre_user_stderr`` / ``user_stderr`` kwargs, which is what these
tests exercise.

Three combinations of buffer state matter:

  * ``user_stderr`` empty, ``pre_user_stderr`` populated: pre-user-
    code failure (libxcrun, sandbox-deny, preamble error). Whole
    pre-user buffer forwards unredacted by construction.
  * ``user_stderr`` has a traceback whose DEEPEST frame is in
    Nora-controlled code (staged ``lib/nora.py``, library /
    stdlib, preamble lines of script.py): exception body is
    Nora-authored, safe to forward.
  * ``user_stderr`` has a traceback whose deepest frame is in
    researcher-authored script.py lines: body redacted.

Buffer-split is the load-bearing invariant. The classifier never
infers phase from text shape — a segfault during user code that
prints to stderr before dying produces no traceback, but its
bytes land in ``user_stderr`` and are NOT forwarded as if they
were pre-script content. That regression is what made the buffer-
split refactor necessary.

Legacy callers that pass only ``stderr`` get the conservative
treatment (whole stream as ``user_stderr``, full redaction).
"""

from __future__ import annotations

from pathlib import Path

from nora.error_summary import extract_debug_excerpt


_PREAMBLE = (
    "import sys as _nora_sys\n"
    "_nora_sys.path.append('/tmp/pkg')\n"
    "_nora_sys.path.insert(0, '/tmp/lib')\n"
    "del _nora_sys\n"
    "# ----- Nora preamble above; researcher code below -----\n"
    "\n"
)


def _stage(tmp_path: Path, user_code: str) -> Path:
    """Materialise a script.py mirroring what the executor writes."""
    script = tmp_path / "script.py"
    script.write_text(_PREAMBLE + user_code, encoding="utf-8")
    return tmp_path


# ---------------------------------------------------------------------------
# pre-user phase: empty user_stderr → pre_user_stderr forwarded
# ---------------------------------------------------------------------------

def test_pre_user_phase_forwards_launcher_stderr(tmp_path: Path):
    """The libxcrun-stub failure puts its error in phase 0 (the pipe
    stderr the executor captured before the preamble's first
    dup2), which lands in ``pre_user_stderr``. ``user_stderr`` is
    empty because no user code ever ran. The extractor must
    forward ``pre_user_stderr`` verbatim — that's the diagnostic
    payload the original "import nora" incident needed."""
    run_dir = _stage(tmp_path, "x = 1\n")
    libxcrun_stderr = (
        "xcrun: error: unable to load libxcrun "
        "(dlopen(/Library/Developer/CommandLineTools/usr/lib/"
        "libxcrun.dylib, 0x0005): file system sandbox blocked open())."
    )
    out = extract_debug_excerpt(
        stdout="", stderr="", exit_code=1,
        language="Python", run_dir=run_dir,
        pre_user_stderr=libxcrun_stderr,
        user_stderr="",
    )
    assert out is not None
    assert "libxcrun" in out
    assert "no user code executed" in out
    # The redaction sentinel must NOT appear — that's the whole win.
    assert "[message body redacted]" not in out


def test_pre_user_phase_handles_empty_buffers(tmp_path: Path):
    """Both buffers empty = nothing to forward. Caller falls through
    to the executor's generic "script failed" fallback."""
    run_dir = _stage(tmp_path, "x = 1\n")
    out = extract_debug_excerpt(
        stdout="", stderr="", exit_code=1,
        language="Python", run_dir=run_dir,
        pre_user_stderr="", user_stderr="",
    )
    assert out is None


# ---------------------------------------------------------------------------
# Buffer split closes the segfault-during-user-code leak
# ---------------------------------------------------------------------------

def test_segfault_with_prior_user_stderr_writes_does_not_leak(tmp_path: Path):
    """The canonical reason for the buffer-split refactor: user code
    writes data to stderr then crashes WITHOUT raising. The old
    "no traceback → pre_script" classifier would have forwarded the
    user-printed bytes as if they were launcher output. With the
    buffer split, those bytes are in ``user_stderr`` (because the
    preamble already dup2'd fd 2 before user code ran), so the
    extractor will NOT forward them unredacted."""
    run_dir = _stage(tmp_path, "x = 1\n")
    secret = "DEADBEEF-secret-cell-value-DEADBEEF"
    # No traceback — segfault / OOM / SIGKILL produces this shape.
    user_buf = f"row 0: {secret}\nrow 1: more-data\n"
    out = extract_debug_excerpt(
        stdout="", stderr="", exit_code=-11,  # -11 = SIGSEGV
        language="Python", run_dir=run_dir,
        pre_user_stderr="",
        user_stderr=user_buf,
    )
    # Either no excerpt (None) or one that does NOT contain the
    # secret. Both are acceptable; what's NOT acceptable is the
    # secret reaching the model.
    if out is not None:
        assert secret not in out


# ---------------------------------------------------------------------------
# nora_owned: traceback in runtime/preamble → body forwarded
# ---------------------------------------------------------------------------

def test_nora_owned_phase_forwards_runtime_exception_body(tmp_path: Path):
    """When the staged ``lib/nora.py`` raises (e.g., its module-init
    ``RuntimeError("NORA_RUN_TOKEN not set")``), the traceback's
    deepest frame is in Nora-controlled code. The body is Nora's
    own diagnostic — naming the runtime as the failure point — and
    must reach the model unredacted. The deepest-frame rule is
    what makes this work even when a user-script frame appears
    higher in the chain (the user wrote ``import nora`` on a real
    line of script.py)."""
    run_dir = _stage(tmp_path, "import nora\n")
    nora_path = run_dir / "lib" / "nora.py"
    user_stderr = (
        "Traceback (most recent call last):\n"
        f'  File "{run_dir}/script.py", line 8, in <module>\n'
        "    import nora\n"
        f'  File "{nora_path}", line 55, in <module>\n'
        '    raise RuntimeError(\n'
        "RuntimeError: NORA_RUN_TOKEN not set. This script must be run "
        "through the Nora executor.\n"
    )
    out = extract_debug_excerpt(
        stdout="", stderr="", exit_code=1,
        language="Python", run_dir=run_dir,
        pre_user_stderr="", user_stderr=user_stderr,
    )
    assert out is not None
    # Deepest frame is lib/nora.py → nora_owned → body forwards.
    assert "NORA_RUN_TOKEN not set" in out
    assert "[message body redacted]" not in out


def test_nora_owned_phase_forwards_when_only_library_frames(tmp_path: Path):
    """A traceback that only touches stdlib / site-packages and never
    enters the staged script.py is by definition nora-owned — no
    researcher data has flowed through any of those frames yet."""
    run_dir = _stage(tmp_path, "x = 1\n")
    user_stderr = (
        "Traceback (most recent call last):\n"
        '  File "/opt/homebrew/lib/python3.12/runpy.py", line 196, in _run_module_as_main\n'
        "    return _run_code(...)\n"
        "RuntimeError: bootstrap failure\n"
    )
    out = extract_debug_excerpt(
        stdout="", stderr="", exit_code=1,
        language="Python", run_dir=run_dir,
        pre_user_stderr="", user_stderr=user_stderr,
    )
    assert out is not None
    assert "bootstrap failure" in out


# ---------------------------------------------------------------------------
# user_code: any frame past the preamble marker → body redacted
# ---------------------------------------------------------------------------

def test_user_code_phase_redacts_exception_body(tmp_path: Path):
    """User-code deepest frame triggers redaction — the body could
    carry researcher data exfiltrated via
    ``raise RuntimeError(df.iloc[0]['secret'])``."""
    user_code = "df = read_csv()\nraise RuntimeError(df['secret'][0])\n"
    run_dir = _stage(tmp_path, user_code)
    script_path = run_dir / "script.py"
    # User code starts after the (now longer) preamble; line numbers
    # past the marker line classify as user_code regardless of
    # exact preamble length.
    user_stderr = (
        "Traceback (most recent call last):\n"
        f'  File "{script_path}", line 20, in <module>\n'
        "    raise RuntimeError(df['secret'][0])\n"
        "RuntimeError: 4242-deadbeef-secret-cell-value\n"
    )
    out = extract_debug_excerpt(
        stdout="", stderr="", exit_code=1,
        language="Python", run_dir=run_dir,
        pre_user_stderr="", user_stderr=user_stderr,
    )
    assert out is not None
    assert "4242-deadbeef-secret-cell-value" not in out
    assert "[message body redacted]" in out


def test_user_code_redaction_preserved_when_buffers_omitted(tmp_path: Path):
    """Legacy callers that don't pass the buffer-split kwargs must
    keep the conservative redacted behavior: the whole ``stderr``
    is treated as ``user_stderr``, so any body is redacted. Without
    this fallback every legacy call site would silently relax SDC."""
    stderr = (
        "Traceback (most recent call last):\n"
        '  File "/abs/path/script.py", line 20, in <module>\n'
        "    raise RuntimeError(df['secret'][0])\n"
        "RuntimeError: cell-content-here\n"
    )
    out = extract_debug_excerpt(
        stdout="", stderr=stderr, exit_code=1,
        language="Python",
        # pre_user_stderr / user_stderr / run_dir all intentionally
        # omitted — legacy single-buffer signature.
    )
    assert out is not None
    assert "cell-content-here" not in out
    assert "[message body redacted]" in out


# ---------------------------------------------------------------------------
# Preamble marker boundary
# ---------------------------------------------------------------------------

def test_preamble_line_is_classified_as_nora_owned(tmp_path: Path):
    """A frame in script.py BEFORE the preamble marker (e.g., a
    syntax error in the Nora-authored preamble itself) must classify
    as nora_owned — the lines didn't come from the researcher.
    Verified via the deepest-frame rule: if the failure raised
    inside the preamble, no user code has run."""
    run_dir = _stage(tmp_path, "x = 1\n")
    script_path = run_dir / "script.py"
    # Line 3 is well within the preamble.
    user_stderr = (
        "Traceback (most recent call last):\n"
        f'  File "{script_path}", line 3, in <module>\n'
        "    _nora_sys.path.insert(0, ...)\n"
        "ValueError: preamble-internal-message\n"
    )
    out = extract_debug_excerpt(
        stdout="", stderr="", exit_code=1,
        language="Python", run_dir=run_dir,
        pre_user_stderr="", user_stderr=user_stderr,
    )
    assert out is not None
    assert "preamble-internal-message" in out
    assert "[message body redacted]" not in out


def test_unreadable_script_falls_back_to_user_code(tmp_path: Path):
    """If the script can't be read (deleted between run and report),
    the classifier conservatively treats every script.py frame as
    user code so the body is redacted. Without this fallback a
    cleanup race could expose the body."""
    # Don't stage script.py — make the read fail.
    run_dir = tmp_path
    script_path = run_dir / "script.py"
    user_stderr = (
        "Traceback (most recent call last):\n"
        f'  File "{script_path}", line 8, in <module>\n'
        "    bad()\n"
        "RuntimeError: data-shaped-content-that-must-not-leak\n"
    )
    out = extract_debug_excerpt(
        stdout="", stderr="", exit_code=1,
        language="Python", run_dir=run_dir,
        pre_user_stderr="", user_stderr=user_stderr,
    )
    assert out is not None
    assert "data-shaped-content-that-must-not-leak" not in out
