"""Regression tests for the fourth batch of reviewer-flagged fixes.

Pins the python-side behaviors:

1. ``count_next_context`` no longer accepts a default ``ceiling``;
   forgetting to pass the active model's window would otherwise
   silently anchor the chip's denominator at 1M for every model
   regardless of its real context size.

2. ``auth.get_credential`` does NOT cache transient backend errors
   as ``None``. A securityd hiccup or momentarily-locked keychain
   used to be cached for the rest of the process, so a follow-up
   call after the backend recovered still reported "no creds"
   until restart.

The shell-script fixes (release.sh pgrep target, build_dmg.sh
poll-first / status handling, ditto vs cp, etc.) live in packaging/
and don't have a python harness; their correctness is exercised by
the release pipeline and code review.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# count_next_context: ceiling is required
# ---------------------------------------------------------------------------

def test_count_next_context_requires_ceiling() -> None:
    """No default — every caller must pass the active model's window
    so a forgotten parameter doesn't silently render with a 1M
    denominator on a model that's actually 200k."""
    from nora.context_count import count_next_context

    sig = inspect.signature(count_next_context)
    ceiling_param = sig.parameters["ceiling"]
    assert ceiling_param.default is inspect.Parameter.empty, (
        "ceiling must be required so callers can't silently inherit "
        "a 1M denominator on smaller-window models"
    )

    # And calling without ceiling raises TypeError.
    with pytest.raises(TypeError, match="ceiling"):
        count_next_context(cwd=None)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# auth keyring cache: don't cache transient backend errors
# ---------------------------------------------------------------------------

class _FlakyKeyring:
    """get_password raises the first time, succeeds on retry."""

    def __init__(self, eventual_value: str | None) -> None:
        self.calls = 0
        self.eventual = eventual_value
        self.store: dict[tuple[str, str], str] = {}

    def get_password(self, service: str, username: str) -> str | None:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("securityd transient")
        return self.eventual

    def set_password(self, service: str, username: str, password: str) -> None:
        self.store[(service, username)] = password

    def delete_password(self, service: str, username: str) -> None:
        self.store.pop((service, username), None)


def test_get_credential_does_not_cache_transient_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A keyring backend error must NOT be cached as 'no creds' for
    the rest of the process. Pre-fix, the second call returned the
    cached None even though the backend had recovered, so the auth
    screen kept claiming the credential was gone until restart.

    The recovery path is gated by ``_ERROR_BACKOFF_SECONDS`` (5s by
    default — prevents prompt-storm during a single auth-screen
    render where 4+ ``has_credential`` calls fire in microseconds).
    Within that window the second call still returns ``None`` to
    suppress re-prompts; once the window elapses the next call hits
    the backend and recovers. This test rewinds the recorded error
    timestamp to simulate "enough time has passed" so the recovery
    path is exercised deterministically."""
    import time
    from nora import auth

    auth._CRED_CACHE.clear()
    auth._CRED_ERROR_AT.clear()
    flaky = _FlakyKeyring(eventual_value="sk-ant-recovered")
    monkeypatch.setattr(auth, "_keyring", flaky)

    # First call: backend raises → returns None, but the failure
    # must NOT be cached as a definitive "no creds".
    first = auth.get_credential("anthropic")
    assert first is None
    assert "anthropic" not in auth._CRED_CACHE, (
        "transient errors must not poison the cache"
    )

    # Rewind the error timestamp so the backoff window is past — the
    # next call will retry the backend instead of returning the
    # within-window suppressed ``None``.
    auth._CRED_ERROR_AT["anthropic"] = (
        time.monotonic() - auth._ERROR_BACKOFF_SECONDS - 1
    )

    # Second call: backend has recovered, value comes through.
    second = auth.get_credential("anthropic")
    assert second == "sk-ant-recovered"
    # Successful reads ARE cached so subsequent UI-render calls
    # don't re-hit the backend.
    assert auth._CRED_CACHE["anthropic"] == "sk-ant-recovered"


def test_get_credential_caches_definitive_misses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Negative regression: a real "no credential stored" answer
    (keyring returns None without raising) IS still cached — the
    auth screen renders this case repeatedly and we don't want to
    re-prompt on every read on unsigned builds."""
    from nora import auth

    auth._CRED_CACHE.clear()

    class _EmptyKeyring:
        def __init__(self) -> None:
            self.calls = 0

        def get_password(self, service: str, username: str) -> str | None:
            self.calls += 1
            return None

        def set_password(self, service: str, username: str, password: str) -> None:
            pass

        def delete_password(self, service: str, username: str) -> None:
            pass

    backend = _EmptyKeyring()
    monkeypatch.setattr(auth, "_keyring", backend)

    assert auth.get_credential("openai") is None
    assert auth.get_credential("openai") is None
    assert backend.calls == 1, (
        "definitive 'no credential' answers should be cached so "
        "redundant reads don't re-hit the backend"
    )
