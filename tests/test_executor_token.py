"""Unit + integration tests for the per-run authenticity token.

The token is the executor's defense against a malicious script writing
hand-crafted JSON directly to ``NORA_RESULT_PATH`` and bypassing
the runtime library. The runtime library embeds a random per-run
token in every emitted payload; the executor validates it and strips
it before the payload reaches the sanitizer.

Two kinds of tests here:

- **Unit tests** for ``_validate_and_strip_token`` — pure, portable,
  no sandbox-exec needed. These lock in the contract: payloads without
  the token are rejected, payloads with the wrong token are rejected,
  payloads with the right token get the token stripped before flowing
  on.

- **Integration test** that a real R script bypassing the nora
  library fails. Gated on ``Rscript`` + ``sandbox-apply`` preflight,
  same as the other sandbox integration tests. This is the
  end-to-end proof that the defense actually works against the
  original attack shape.

See also ``docs/direction.md`` "runtime-library contract" for the
deliberate limits of this measure (it raises attacker cost, does not
provide a structural guarantee against in-process introspection).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from nora.env_detect import find_sandbox_exec
from nora.executor import (
    RESULT_TOKEN_FIELD,
    _format_bad_lines_summary,
    _generate_run_token,
    _parse_result_jsonl,
    _validate_and_strip_token,
    run_script,
)


# ---------------------------------------------------------------------------
# Unit tests — pure, no subprocess
# ---------------------------------------------------------------------------

def test_token_generator_produces_long_random_hex():
    """The token must be long enough that a random guess has vanishing
    probability of matching. 256 bits is the target; hex-encoded
    that's 64 characters."""
    t = _generate_run_token()
    assert isinstance(t, str)
    assert len(t) == 64
    # All hex, and two successive generations don't collide.
    int(t, 16)  # raises if non-hex
    assert _generate_run_token() != t


def test_validate_strips_token_on_success():
    token = "a" * 64
    payload = {
        "type": "linear_regression",
        "n": 100,
        "r_squared": 0.5,
        RESULT_TOKEN_FIELD: token,
    }
    cleaned, err = _validate_and_strip_token(payload, token)
    assert err is None
    assert cleaned is not None
    assert RESULT_TOKEN_FIELD not in cleaned, (
        "token must be stripped so downstream consumers never see it"
    )
    assert cleaned["type"] == "linear_regression"
    assert cleaned["n"] == 100


def test_validate_rejects_missing_token():
    payload = {"type": "linear_regression", "n": 100}
    cleaned, err = _validate_and_strip_token(payload, "a" * 64)
    assert cleaned is None
    assert err is not None
    assert "missing" in err.lower() or "authenticity" in err.lower()


def test_validate_rejects_wrong_token():
    payload = {
        "type": "linear_regression",
        "n": 100,
        RESULT_TOKEN_FIELD: "b" * 64,
    }
    cleaned, err = _validate_and_strip_token(payload, "a" * 64)
    assert cleaned is None
    assert err is not None
    assert "match" in err.lower() or "bypass" in err.lower()


def test_validate_rejects_non_string_token():
    payload = {
        "type": "linear_regression",
        "n": 100,
        RESULT_TOKEN_FIELD: 12345,  # not a string
    }
    cleaned, err = _validate_and_strip_token(payload, "a" * 64)
    assert cleaned is None
    assert err is not None


def test_validate_rejects_non_dict_payload():
    cleaned, err = _validate_and_strip_token(["not", "a", "dict"], "a" * 64)  # type: ignore[arg-type]
    assert cleaned is None
    assert err is not None


# ---------------------------------------------------------------------------
# Unit tests — JSONL parser preserves valid lines past a corrupt one
# ---------------------------------------------------------------------------

def test_parse_jsonl_preserves_valid_lines_past_a_bad_one():
    """A degenerate Stata fit can emit a missing-value marker (``.``) in
    the middle of an otherwise valid JSON line — the line is unparseable
    and the parser used to ``break``, silently losing every later valid
    line in the same multi-result batch. Pin: bad lines are skipped and
    the rest land in ``payloads``, with a per-line error message in the
    ``bad_lines`` return."""
    token = "a" * 64
    good = (
        '{"type":"linear_regression","n":100,"r_squared":0.5,'
        f'"_token":"{token}"' + '}'
    )
    bad_json = '{"type":"linear_regression","n":100,"f_statistic":.,'\
        f'"_token":"{token}"' + '}'
    text = "\n".join([good, bad_json, good, "", good])

    payloads, bad_lines = _parse_result_jsonl(text, token)
    assert len(payloads) == 3, (
        f"expected 3 valid payloads past the corrupt line, got "
        f"{len(payloads)}"
    )
    assert len(bad_lines) == 1
    assert "line 2" in bad_lines[0]
    # Bad line message names the JSON failure, not just "skipped".
    assert "Expecting" in bad_lines[0] or "valid" in bad_lines[0].lower()
    # Tokens stripped from the surviving payloads.
    for p in payloads:
        assert RESULT_TOKEN_FIELD not in p


def test_parse_jsonl_preserves_valid_lines_past_a_token_failure():
    """The same skip-and-keep-going contract applies when a line parses
    as JSON but fails token validation (e.g., a hand-crafted bypass
    payload from a malicious script). Earlier valid payloads survive,
    later valid payloads survive, the bad payload is dropped with an
    explanatory message."""
    token = "a" * 64
    wrong = "b" * 64
    valid_line = (
        '{"type":"linear_regression","n":100,'
        f'"_token":"{token}"' + '}'
    )
    forged_line = (
        '{"type":"linear_regression","n":100,'
        f'"_token":"{wrong}"' + '}'
    )
    text = "\n".join([valid_line, forged_line, valid_line])

    payloads, bad_lines = _parse_result_jsonl(text, token)
    assert len(payloads) == 2
    assert len(bad_lines) == 1
    assert "line 2" in bad_lines[0]


def test_bad_lines_summary_surfaces_linenos_past_first_five():
    """Eight corrupt lines: details for the first 5, then the line
    numbers of the remaining 3. Plain ``…`` used to hide that
    information, forcing the researcher to read the result file by
    hand to find the rest of the failures."""
    bad_lines = [f"line {i}: invalid json" for i in (3, 5, 7, 9, 11, 13, 17, 22)]
    msg = _format_bad_lines_summary(bad_lines, payload_count=2)
    assert "8 malformed result line(s) skipped (2 valid preserved)" in msg
    # Each of the first 5 line numbers appears in detail form.
    for ln in (3, 5, 7, 9, 11):
        assert f"line {ln}: invalid json" in msg
    # Each of the remaining 3 line numbers appears in the tail.
    assert "and lines 13,17,22 also failed" in msg


def test_bad_lines_summary_exact_five_omits_tail():
    """Boundary: 5 entries fits in detail form, no tail needed."""
    bad_lines = [f"line {i}: invalid json" for i in (1, 2, 3, 4, 5)]
    msg = _format_bad_lines_summary(bad_lines, payload_count=0)
    assert "5 malformed result line(s) skipped (0 valid preserved)" in msg
    assert "…" not in msg
    assert "also failed" not in msg


def test_parse_jsonl_returns_no_bad_lines_for_clean_input():
    """The happy path: a multi-result file with three clean payloads
    parses to three results and zero bad lines."""
    token = "a" * 64
    line = (
        '{"type":"linear_regression","n":50,'
        f'"_token":"{token}"' + '}'
    )
    text = "\n".join([line, line, line])
    payloads, bad_lines = _parse_result_jsonl(text, token)
    assert len(payloads) == 3
    assert bad_lines == []


# ---------------------------------------------------------------------------
# Integration test — real R subprocess bypassing the library
# ---------------------------------------------------------------------------

_RSCRIPT = shutil.which("Rscript")


def _sandbox_apply_works() -> bool:
    exe = find_sandbox_exec()
    if exe is None:
        return False
    try:
        r = subprocess.run(
            [exe, "-p", "(version 1)(allow default)", "/usr/bin/true"],
            capture_output=True, timeout=5, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return r.returncode == 0


requires_rscript = pytest.mark.skipif(
    _RSCRIPT is None, reason="Rscript not on PATH"
)

requires_sandbox_apply = pytest.mark.skipif(
    sys.platform != "darwin" or not _sandbox_apply_works(),
    reason="sandbox-exec cannot apply a profile in this environment",
)


@requires_sandbox_apply
@requires_rscript
def test_bypass_without_token_is_rejected(tmp_path: Path):
    """A script that writes hand-crafted JSON to NORA_RESULT_PATH
    without going through the runtime library must be rejected.

    This is the core defense the token adds. Before the token, the
    sanitizer would happily process the hand-crafted payload (the
    shape is valid) and Claude would see whatever the attacker
    encoded. With the token, the executor rejects the payload at the
    authenticity check before it reaches the sanitizer.
    """
    # Note: this R script explicitly does NOT call anything from the
    # `nora` runtime. It just opens the result file and writes a
    # plausible-looking regression payload.
    code = (
        'result_path <- Sys.getenv("NORA_RESULT_PATH")\n'
        'con <- file(result_path, open = "w", encoding = "UTF-8")\n'
        'writeLines(paste0(\n'
        '  \'{"type":"linear_regression","n":100,\',\n'
        '  \'"response_variable":"y","predictor_variables":["x"],\',\n'
        '  \'"coefficients":{"x":1.0},"standard_errors":{"x":0.1},\',\n'
        '  \'"r_squared":0.5}\'\n'
        '), con)\n'
        'close(con)\n'
    )
    r = run_script("R", code, tmp_path)
    assert not r.ok, (
        "bypass should fail authenticity check; instead got ok=True"
    )
    assert r.error is not None
    err = r.error.lower()
    assert (
        "_token" in r.error
        or "authenticity" in err
        or "missing" in err
    ), f"expected token-related error, got: {r.error!r}"
    assert not r.result_payloads, (
        "unauthenticated payload must not be returned to the caller"
    )


@requires_sandbox_apply
@requires_rscript
def test_legitimate_script_with_token_succeeds(tmp_path: Path):
    """Counterpart to the bypass test: a script that uses the
    library normally must still work. Sanity check that the token
    validation doesn't reject legitimate payloads."""
    code = (
        'df <- data.frame(x = 1:12, y = (1:12) * 2)\n'
        'nora$from_lm(lm(y ~ x, data = df), label = "legit")\n'
    )
    r = run_script("R", code, tmp_path)
    assert r.ok, f"legitimate script failed: {r.error}"
    assert r.result_payloads
    assert r.result_payloads[0]["type"] == "linear_regression"
    # The token must be stripped from what the caller receives.
    assert RESULT_TOKEN_FIELD not in r.result_payloads[0]


# ---------------------------------------------------------------------------
# ``ok`` / ``warnings`` semantics across run_script
# ---------------------------------------------------------------------------


@requires_sandbox_apply
@requires_rscript
def test_run_script_ok_true_with_warning_when_bad_line_alongside_valid(
    tmp_path: Path,
):
    """A clean-exit script that writes ONE valid token-stamped payload
    AND ONE malformed JSONL line must come back with ``ok=True`` and
    a non-empty ``warnings`` list — the prior behavior flipped
    ``ok=False`` and routed the bad-line summary into ``error``,
    which demoted the envelope to ``execution_failed_partial`` even
    though the subprocess exited 0.

    Bypasses the runtime library on purpose so the test pins the
    executor's parser path independently of the helper modules. Uses
    the same ``Sys.getenv`` shape as the existing bypass test.
    """
    # ``nora$.run_token`` is captured at runtime-library load (the
    # ``source(nora.R)`` bootstrap that wraps every R run). The env
    # var itself is unset right after, so reading
    # ``Sys.getenv("NORA_RUN_TOKEN")`` here returns "". Using the
    # in-memory copy is the same path the helpers themselves take.
    code = (
        'token <- nora$.run_token\n'
        'path <- Sys.getenv("NORA_RESULT_PATH")\n'
        'con <- file(path, open = "w", encoding = "UTF-8")\n'
        'valid <- paste0(\n'
        '  \'{"type":"linear_regression","n":100,\',\n'
        '  \'"response_variable":"y","predictor_variables":["x"],\',\n'
        '  \'"coefficients":{"x":0.5},"standard_errors":{"x":0.05},\',\n'
        '  \'"_token":"\', token, \'"}\'\n'
        ')\n'
        'writeLines(valid, con)\n'
        'writeLines("{not valid json,}", con)\n'
        'close(con)\n'
    )
    r = run_script("R", code, tmp_path)

    assert r.ok, (
        f"clean-exit run with one valid + one bad line should stay ok=True; "
        f"got ok={r.ok}, error={r.error!r}"
    )
    assert r.exit_code == 0
    assert len(r.result_payloads) == 1, r.result_payloads
    assert r.error is None, r.error
    assert r.warnings, "expected a malformed-line warning"
    assert any(
        "malformed result line" in w.lower() for w in r.warnings
    ), r.warnings


@requires_sandbox_apply
@requires_rscript
def test_run_script_ok_false_when_only_bad_lines_no_survivors(tmp_path: Path):
    """When NO payload survives — every line is malformed or fails
    token validation — the bad-line message is upgraded into
    ``error`` and ``ok=False``. The bypass-attempt case (a forged
    line with no legitimate output to keep) must still surface as a
    fatal-shaped response so the security signal isn't buried."""
    code = (
        'path <- Sys.getenv("NORA_RESULT_PATH")\n'
        'con <- file(path, open = "w", encoding = "UTF-8")\n'
        'writeLines("{not json,}", con)\n'
        'close(con)\n'
    )
    r = run_script("R", code, tmp_path)

    assert not r.ok
    assert r.error is not None
    assert "malformed" in r.error.lower(), r.error
    assert r.result_payloads == []
