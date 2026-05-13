"""Tests for the keyring-backed credential store.

These tests use an in-memory fake keyring backend so they can run
in CI environments where there is no Keychain / D-Bus / Windows
Credential Manager available. The fake mirrors the methods Nora's
``auth`` module actually calls: ``set_password``, ``get_password``,
``delete_password``.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from nora import auth


class _FakeKeyring:
    """Minimal in-memory ``keyring`` backend stand-in."""

    def __init__(self) -> None:
        self.store: dict[tuple[str, str], str] = {}

    def get_password(self, service: str, username: str) -> str | None:
        return self.store.get((service, username))

    def set_password(self, service: str, username: str, password: str) -> None:
        self.store[(service, username)] = password

    def delete_password(self, service: str, username: str) -> None:
        # keyring raises ``PasswordDeleteError`` for missing entries;
        # mimic that so ``auth.delete_credential`` exercises its
        # idempotent except branch.
        if (service, username) not in self.store:
            raise KeyError("not found")
        del self.store[(service, username)]


@pytest.fixture(autouse=True)
def _clear_cred_cache() -> None:
    """The auth module caches credential reads across the process to
    avoid redundant Keychain prompts. Tests need a fresh cache each
    case or fakes from the previous test bleed through. The error
    backoff dict mirrors the cache and must be cleared too."""
    auth._CRED_CACHE.clear()
    auth._CRED_ERROR_AT.clear()


@pytest.fixture
def fake_keyring(monkeypatch: pytest.MonkeyPatch) -> _FakeKeyring:
    fake = _FakeKeyring()
    monkeypatch.setattr(auth, "_keyring", fake)
    return fake


def test_get_credential_missing_returns_none(fake_keyring: _FakeKeyring) -> None:
    assert auth.get_credential("openai") is None
    assert auth.has_credential("openai") is False


def test_set_then_get_credential_roundtrip(fake_keyring: _FakeKeyring) -> None:
    res = auth.set_credential("openai", "sk-test-123")
    assert res == {"ok": True, "provider": "openai"}
    assert auth.get_credential("openai") == "sk-test-123"
    assert auth.has_credential("openai") is True


def test_set_credential_strips_whitespace(fake_keyring: _FakeKeyring) -> None:
    """Researchers paste keys with stray newlines / spaces from
    websites. Strip them so the credential actually authenticates."""
    auth.set_credential("openai", "  sk-trimmed  \n")
    assert auth.get_credential("openai") == "sk-trimmed"


def test_set_credential_rejects_empty(fake_keyring: _FakeKeyring) -> None:
    res = auth.set_credential("openai", "")
    assert res["ok"] is False
    assert "empty" in res["reason"]
    res2 = auth.set_credential("openai", "   ")
    assert res2["ok"] is False


def test_set_credential_rejects_unknown_provider(
    fake_keyring: _FakeKeyring,
) -> None:
    res = auth.set_credential("acme-corp", "k")
    assert res["ok"] is False
    assert "unknown provider" in res["reason"]


def test_delete_credential_removes_entry(fake_keyring: _FakeKeyring) -> None:
    auth.set_credential("openai", "sk-1")
    res = auth.delete_credential("openai")
    assert res["ok"] is True
    assert auth.get_credential("openai") is None


def test_delete_credential_is_idempotent(fake_keyring: _FakeKeyring) -> None:
    """Deleting a missing credential succeeds; deleting twice does too."""
    res1 = auth.delete_credential("openai")
    assert res1["ok"] is True
    res2 = auth.delete_credential("openai")
    assert res2["ok"] is True


def test_list_authed_providers_returns_in_canonical_order(
    fake_keyring: _FakeKeyring,
) -> None:
    auth.set_credential("openai", "sk-o")
    auth.set_credential("anthropic", "sk-a")
    listed = auth.list_authed_providers()
    # Order matches KNOWN_PROVIDERS, not insertion order.
    assert listed == list(auth.KNOWN_PROVIDERS)


def test_list_authed_providers_partial(fake_keyring: _FakeKeyring) -> None:
    auth.set_credential("openai", "sk-o")
    assert auth.list_authed_providers() == ["openai"]


def test_auth_summary_default_covers_all_known(
    fake_keyring: _FakeKeyring,
) -> None:
    auth.set_credential("openai", "sk-o")
    summary = auth.auth_summary()
    assert summary == {"anthropic": False, "openai": True}


def test_keyring_backend_unavailable_returns_none_no_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If keyring isn't installed at all, every read returns None and
    every write returns ``ok: False`` — no exceptions cross the
    boundary. Critical because the bridge calls into auth on the UI
    thread; a raise would crash the page."""
    monkeypatch.setattr(auth, "_keyring", None)
    assert auth.get_credential("openai") is None
    assert auth.has_credential("openai") is False
    res = auth.set_credential("openai", "sk-1")
    assert res["ok"] is False
    assert "not available" in res["reason"]
    # Delete is still idempotent OK — there's nothing to remove.
    assert auth.delete_credential("openai") == {"ok": True, "provider": "openai"}


def test_keyring_read_error_returns_none(
    monkeypatch: pytest.MonkeyPatch, fake_keyring: _FakeKeyring,
) -> None:
    """Keyring backends can raise mid-read (locked Keychain, denied
    permission). Treat any error as 'no credential' so the auth
    screen surfaces a re-prompt rather than a crash."""

    def _raise(*_a: Any, **_kw: Any) -> None:
        raise RuntimeError("backend locked")

    monkeypatch.setattr(fake_keyring, "get_password", _raise)
    assert auth.get_credential("openai") is None


def test_keyring_error_does_not_poison_cache_permanently(
    monkeypatch: pytest.MonkeyPatch, fake_keyring: _FakeKeyring,
) -> None:
    """A transient backend failure (locked Keychain, denied prompt,
    securityd hiccup) must not lock the process into a no-creds view.
    Once the backoff window elapses and the backend recovers, the
    next read picks up the real credential — without restarting the
    app.

    Regression: previously ``get_credential`` cached ``None`` on any
    keyring exception, so a single denied prompt made every later
    call return ``None`` even after the user granted access via
    Keychain Access.
    """
    # Pre-populate the fake's store as if a credential was set
    # before the process even started.
    fake_keyring.store[(auth.KEYRING_SERVICE, "openai")] = "sk-real"

    calls = {"n": 0}
    real_get = fake_keyring.get_password

    def _flaky(service: str, username: str) -> str | None:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("backend locked")
        return real_get(service, username)

    monkeypatch.setattr(fake_keyring, "get_password", _flaky)

    # First call — backend errors → returns None.
    assert auth.get_credential("openai") is None

    # Within the backoff window, repeated calls do NOT re-hit keyring
    # (so the user isn't prompt-stormed during a single render burst).
    # Burst suppression: 5 calls produce 1 backend hit total.
    for _ in range(5):
        assert auth.get_credential("openai") is None
    assert calls["n"] == 1

    # Simulate enough time passing for the backoff to expire by
    # rewinding the recorded error timestamp. The next call retries
    # the backend and recovers.
    auth._CRED_ERROR_AT["openai"] = (
        time.monotonic() - auth._ERROR_BACKOFF_SECONDS - 1
    )
    assert auth.get_credential("openai") == "sk-real"
    assert calls["n"] == 2


def test_set_credential_clears_error_backoff(
    monkeypatch: pytest.MonkeyPatch, fake_keyring: _FakeKeyring,
) -> None:
    """Once the user successfully writes a credential, the error
    backoff for that provider is cleared — a successful write proves
    the backend is reachable, so future reads must not be artificially
    delayed by a stale error timestamp."""
    auth._CRED_ERROR_AT["openai"] = time.monotonic()
    res = auth.set_credential("openai", "sk-new")
    assert res["ok"] is True
    assert "openai" not in auth._CRED_ERROR_AT


def test_delete_credential_backend_failure_returns_error(
    monkeypatch: pytest.MonkeyPatch, fake_keyring: _FakeKeyring,
) -> None:
    """A keyring backend that raises during delete (locked Keychain,
    denied prompt, securityd error) must surface ``ok: False``.
    Reporting success would lie to the UI: the secret is still in the
    OS store, the user thinks it's gone, and it reappears on next
    launch.

    Regression: previously every delete exception was swallowed and
    delete_credential returned ``{"ok": True}``.
    """
    fake_keyring.store[(auth.KEYRING_SERVICE, "openai")] = "sk-real"

    def _raise_on_delete(*_a: Any, **_kw: Any) -> None:
        raise RuntimeError("keychain locked")

    monkeypatch.setattr(fake_keyring, "delete_password", _raise_on_delete)

    res = auth.delete_credential("openai")
    assert res["ok"] is False
    assert "delete failed" in res["reason"]
    # Cache must NOT be poisoned with None — that would hide the
    # still-present credential from the auth screen.
    assert auth._CRED_CACHE.get("openai") == "sk-real"


def test_delete_credential_missing_entry_is_idempotent(
    monkeypatch: pytest.MonkeyPatch, fake_keyring: _FakeKeyring,
) -> None:
    """Deleting a credential that isn't there should still report
    success — and must not even attempt a backend delete (so no
    spurious ACL prompt for a no-op)."""
    delete_calls = {"n": 0}
    real_delete = fake_keyring.delete_password

    def _counting(*a: Any, **kw: Any) -> None:
        delete_calls["n"] += 1
        return real_delete(*a, **kw)

    monkeypatch.setattr(fake_keyring, "delete_password", _counting)

    res = auth.delete_credential("openai")
    assert res["ok"] is True
    # Pre-check via get_password saw None; no delete attempted.
    assert delete_calls["n"] == 0


def test_delete_credential_during_backoff_returns_error(
    monkeypatch: pytest.MonkeyPatch, fake_keyring: _FakeKeyring,
) -> None:
    """If a recent read error is still inside the backoff window, we
    can't confidently claim deletion — surface the unreachable backend
    rather than silently no-op'ing the user's request."""

    def _raise(*_a: Any, **_kw: Any) -> None:
        raise RuntimeError("backend locked")

    monkeypatch.setattr(fake_keyring, "get_password", _raise)
    # Trigger the error path on a read so the backoff is set.
    auth.get_credential("openai")
    assert "openai" in auth._CRED_ERROR_AT

    res = auth.delete_credential("openai")
    assert res["ok"] is False
    assert "unavailable" in res["reason"]


# ---------------------------------------------------------------------------
# Cache freshness (``force_refresh=True``). The auth-screen render
# path needs to notice credentials that were deleted directly in
# Keychain Access (outside Nora) between launches, otherwise the
# in-process credential cache keeps showing the provider as
# configured until the next process start.
# ---------------------------------------------------------------------------

def test_get_credential_force_refresh_picks_up_external_deletion(
    fake_keyring: _FakeKeyring,
) -> None:
    """A researcher who deletes the keyring entry via Keychain Access
    (or another tool, bypassing Nora's bridge) must have that
    deletion reflected on the next auth-screen render. Without
    ``force_refresh``, the in-process cache would keep returning the
    stale value for the rest of the process — and the docstring on
    ``ui_ready`` claims the opposite, so the bug surfaces as the auth
    screen continuing to show a provider as configured even though
    the credential is gone from the OS store."""
    auth.set_credential("openai", "sk-stored")
    # First read populates the cache.
    assert auth.get_credential("openai") == "sk-stored"
    # External deletion: bypass ``auth.delete_credential`` entirely,
    # mirroring "researcher used Keychain Access directly."
    del fake_keyring.store[(auth.KEYRING_SERVICE, "openai")]
    # Cached value still surfaces on a normal call.
    assert auth.get_credential("openai") == "sk-stored"
    # ``force_refresh`` drops the cache and re-reads from the
    # backend — now sees the empty store.
    assert auth.get_credential("openai", force_refresh=True) is None
    # And the cache is updated, so the next plain call also returns
    # None (no need to keep passing ``force_refresh``).
    assert auth.get_credential("openai") is None


def test_force_refresh_clears_stale_error_backoff(
    monkeypatch: pytest.MonkeyPatch, fake_keyring: _FakeKeyring,
) -> None:
    """``force_refresh`` should not be blocked by a stale error
    backoff: the caller is explicitly asking for a fresh read, and
    the backoff window's job is burst-suppression, not recovery
    suppression."""
    fake_keyring.store[(auth.KEYRING_SERVICE, "openai")] = "sk-real"

    calls = {"n": 0}
    real_get = fake_keyring.get_password

    def _flaky(service: str, username: str) -> str | None:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("backend locked")
        return real_get(service, username)

    monkeypatch.setattr(fake_keyring, "get_password", _flaky)
    # First call errors; backoff is set.
    assert auth.get_credential("openai") is None
    assert "openai" in auth._CRED_ERROR_AT
    # Plain call within the backoff window does NOT re-hit the
    # backend.
    assert auth.get_credential("openai") is None
    assert calls["n"] == 1
    # ``force_refresh`` ignores the backoff and recovers.
    assert auth.get_credential("openai", force_refresh=True) == "sk-real"
    assert calls["n"] == 2


# ---------------------------------------------------------------------------
# Tri-state credential state. Boolean ``has_credential`` cannot
# distinguish "no credential" from "keyring locked / denied". The
# auth screen rendered both as "Not configured" before, leading a
# researcher whose Keychain prompt was denied to re-paste a key that
# was already stored (or to assume the credential was gone). The
# tri-state distinguishes the two cases for UI rendering.
# ---------------------------------------------------------------------------

def test_credential_state_configured(fake_keyring: _FakeKeyring) -> None:
    auth.set_credential("openai", "sk-1")
    assert auth.credential_state("openai") == auth.AUTH_STATE_CONFIGURED


def test_credential_state_missing(fake_keyring: _FakeKeyring) -> None:
    assert auth.credential_state("openai") == auth.AUTH_STATE_MISSING


def test_credential_state_keyring_unavailable(
    monkeypatch: pytest.MonkeyPatch, fake_keyring: _FakeKeyring,
) -> None:
    """When the keyring backend raises, ``credential_state`` must
    report the third state — not silently downgrade to ``missing``."""
    def _raise(*_a: Any, **_kw: Any) -> None:
        raise RuntimeError("backend locked")

    monkeypatch.setattr(fake_keyring, "get_password", _raise)
    assert auth.credential_state("openai") == auth.AUTH_STATE_KEYRING_UNAVAILABLE


def test_credential_state_keyring_none_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If ``_keyring`` itself is ``None`` (the package failed to
    import), no backend errors fire and the state is plain ``missing``.
    The auth screen for that case routes to "paste a key" — there's
    no Keychain prompt to retry, so ``keyring_unavailable`` would be
    misleading."""
    monkeypatch.setattr(auth, "_keyring", None)
    assert auth.credential_state("openai") == auth.AUTH_STATE_MISSING
