"""Unit + integration tests for the per-run authenticity token.

The token is the executor's defense against a malicious script writing
hand-crafted JSON directly to ``BUILDER_RESULT_PATH`` and bypassing
the runtime library. The runtime library embeds a random per-run
token in every emitted payload; the executor validates it and strips
it before the payload reaches the sanitizer.

Two kinds of tests here:

- **Unit tests** for ``_validate_and_strip_token`` — pure, portable,
  no sandbox-exec needed. These lock in the contract: payloads without
  the token are rejected, payloads with the wrong token are rejected,
  payloads with the right token get the token stripped before flowing
  on.

- **Integration test** that a real R script bypassing the builder
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

from builder.env_detect import find_sandbox_exec
from builder.executor import (
    RESULT_TOKEN_FIELD,
    _generate_run_token,
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
    """A script that writes hand-crafted JSON to BUILDER_RESULT_PATH
    without going through the runtime library must be rejected.

    This is the core defense the token adds. Before the token, the
    sanitizer would happily process the hand-crafted payload (the
    shape is valid) and Claude would see whatever the attacker
    encoded. With the token, the executor rejects the payload at the
    authenticity check before it reaches the sanitizer.
    """
    # Note: this R script explicitly does NOT call anything from the
    # `builder` runtime. It just opens the result file and writes a
    # plausible-looking regression payload.
    code = (
        'result_path <- Sys.getenv("BUILDER_RESULT_PATH")\n'
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
    assert r.result_payload is None, (
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
        'builder$from_lm(lm(y ~ x, data = df), label = "legit")\n'
    )
    r = run_script("R", code, tmp_path)
    assert r.ok, f"legitimate script failed: {r.error}"
    assert r.result_payload is not None
    assert r.result_payload["type"] == "linear_regression"
    # The token must be stripped from what the caller receives.
    assert RESULT_TOKEN_FIELD not in r.result_payload
