"""Tests for the keyring-backed credential store.

These tests use an in-memory fake keyring backend so they can run
in CI environments where there is no Keychain / D-Bus / Windows
Credential Manager available. The fake mirrors the methods Nora's
``auth`` module actually calls: ``set_password``, ``get_password``,
``delete_password``.
"""

from __future__ import annotations

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
    case or fakes from the previous test bleed through."""
    auth._CRED_CACHE.clear()


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
