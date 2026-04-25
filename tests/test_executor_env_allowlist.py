"""Regression tests for the subprocess env-var allowlist.

Nora scripts inherit only the env vars on an explicit allowlist
(see ``executor._SUBPROCESS_ENV_ALLOWLIST``). The reason: a prompt-
injected Claude can call ``Sys.getenv()`` from R and stuff results
into any allowed numeric / string field that reaches the sanitizer,
so any secret in the parent env is in scope for exfiltration. The
sandbox doesn't stop this — the bytes never leave the process
boundary sandbox-exec protects.

These tests lock in:

- Shell secrets (API keys, AWS creds, arbitrary undocumented vars)
  are NOT visible in the subprocess env.
- The allowlisted vars ARE forwarded (so R / Stata keep working).
- Nora-specific vars (NORA_RUN_TOKEN, NORA_RESULT_PATH,
  NORA_LIB_DIR, NORA_CWD) take precedence over any
  pathological same-named entry in the parent env.

If you add a new env var Nora needs, update both the allowlist
and these tests — they're the canonical spec.
"""

from __future__ import annotations

from nora.executor import (
    RUN_TOKEN_ENV_VAR,
    _SUBPROCESS_ENV_ALLOWLIST,
    _filter_env,
)


def test_api_key_style_vars_are_dropped():
    """The specific vars a secret-hunting prompt-injection would
    target must not pass through. If you find yourself adding one
    of these to the allowlist, stop and think about why."""
    parent = {
        "ANTHROPIC_API_KEY": "sk-ant-XXXX",
        "OPENAI_API_KEY": "sk-XXXX",
        "AWS_ACCESS_KEY_ID": "AKIA...",
        "AWS_SECRET_ACCESS_KEY": "secret",
        "GITHUB_TOKEN": "ghp_XXXX",
        "PATH": "/usr/bin",  # allowed — sanity baseline
    }
    filtered = _filter_env(parent)
    assert "ANTHROPIC_API_KEY" not in filtered
    assert "OPENAI_API_KEY" not in filtered
    assert "AWS_ACCESS_KEY_ID" not in filtered
    assert "AWS_SECRET_ACCESS_KEY" not in filtered
    assert "GITHUB_TOKEN" not in filtered
    assert filtered["PATH"] == "/usr/bin"


def test_arbitrary_vars_are_dropped():
    """An arbitrary marker var shouldn't survive either — the
    allowlist is the ONLY way through, not a denylist."""
    parent = {
        "NORA_TEST_SECRET": "TOPSECRET123",
        "RANDOM_SECRET_1234": "leak",
        "HOME": "/home/user",
    }
    filtered = _filter_env(parent)
    assert "NORA_TEST_SECRET" not in filtered
    assert "RANDOM_SECRET_1234" not in filtered
    assert filtered["HOME"] == "/home/user"


def test_allowlist_forwards_what_it_says():
    """Every entry on the allowlist must round-trip through
    _filter_env. Guards against a typo (e.g. accidentally writing
    `"path"` lowercase) that would silently drop a needed var."""
    parent = {name: f"value-of-{name}" for name in _SUBPROCESS_ENV_ALLOWLIST}
    # Add a sprinkle of non-allowlisted noise so we exercise the
    # filter, not just an identity pass-through.
    parent["SECRET_NOT_FORWARDED"] = "nope"
    filtered = _filter_env(parent)
    for name in _SUBPROCESS_ENV_ALLOWLIST:
        assert name in filtered, f"{name} on allowlist but filtered out"
        assert filtered[name] == f"value-of-{name}"
    assert "SECRET_NOT_FORWARDED" not in filtered


def test_locale_and_path_are_allowlisted():
    """Sanity: the things R / Stata actually need to run locally
    are all on the list. If someone trims the allowlist in a
    future cleanup without reading the docstring, this catches it."""
    assert "PATH" in _SUBPROCESS_ENV_ALLOWLIST
    assert "HOME" in _SUBPROCESS_ENV_ALLOWLIST
    assert "LANG" in _SUBPROCESS_ENV_ALLOWLIST
    assert "LC_ALL" in _SUBPROCESS_ENV_ALLOWLIST
    assert "TMPDIR" in _SUBPROCESS_ENV_ALLOWLIST


def test_nora_run_token_var_is_not_on_allowlist():
    """NORA_RUN_TOKEN is set explicitly by the executor per run.
    If it were on the allowlist, a pathological parent-env entry
    could pre-seed a token the attacker knows — defeating the
    whole runtime-library authenticity check. Keep it OUT of the
    allowlist; the executor merges its own value in after the
    allowlist filter."""
    assert RUN_TOKEN_ENV_VAR not in _SUBPROCESS_ENV_ALLOWLIST
    assert "NORA_RUN_TOKEN" not in _SUBPROCESS_ENV_ALLOWLIST
    assert "NORA_RESULT_PATH" not in _SUBPROCESS_ENV_ALLOWLIST
    assert "NORA_LIB_DIR" not in _SUBPROCESS_ENV_ALLOWLIST
    assert "NORA_CWD" not in _SUBPROCESS_ENV_ALLOWLIST


def test_empty_parent_env_yields_empty_filtered():
    assert _filter_env({}) == {}
