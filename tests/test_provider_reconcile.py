"""Tests for ``NoraBridge._reconcile_active_provider_with_auth`` and
the ``clear_injected_env`` cleanup on Anthropic credential delete.

Two distinct production failures these tests pin:

1. **OpenAI-only onboarding** — bridge defaults ``_provider`` to
   ``"anthropic"`` at construction. Without reconcile, a researcher
   who only authenticates OpenAI hits the chat with the Anthropic
   default and the first turn fails (no Claude credential).

2. **Anthropic credential-delete stickiness** — once
   ``_ensure_anthropic_env`` copies the keyring credential into
   ``ANTHROPIC_API_KEY``, deleting the keyring entry leaves the env
   var behind and ``detect_auth()`` keeps reporting ``api_key``
   until the app restarts.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from nora import auth, provider
from nora.ui import NoraBridge


@pytest.fixture(autouse=True)
def _clear_cred_cache() -> None:
    """The auth module caches credential reads across the process to
    avoid redundant Keychain prompts on unsigned builds. Tests need
    a fresh cache each case or fakes from the previous test bleed
    through (a ``has_credential`` call in test 1 caches ``None``,
    test 2's ``fake_keyring.set_password`` writes to the fake but
    the cache short-circuits the read back to None — looks like
    test-order flakiness)."""
    auth._CRED_CACHE.clear()


@pytest.fixture(autouse=True)
def isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each test runs without ANTHROPIC_API_KEY / OPENAI_API_KEY in
    the parent shell, so detect_auth's behaviour is determined
    purely by the (mocked) keyring."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    # And nothing in ~/.claude.json subscription path either —
    # otherwise the test suite picks up the developer's real auth.
    monkeypatch.setattr(
        "nora.provider.anthropic.Path",
        # Drop in a Path that always reports .claude.json missing.
        type("FakePath", (), {
            "home": staticmethod(lambda: type("HM", (), {
                "__truediv__": lambda _self, _other: type("M", (), {
                    "is_file": lambda _s: False,
                })(),
            })()),
        }),
    )


class _FakeKeyring:
    """Drop-in replacement for the ``keyring`` module attribute on
    ``nora.auth``. Stores creds in process memory so tests don't
    need a real Keychain / D-Bus."""

    def __init__(self) -> None:
        self.store: dict[tuple[str, str], str] = {}

    def get_password(self, service: str, username: str) -> str | None:
        return self.store.get((service, username))

    def set_password(self, service: str, username: str, password: str) -> None:
        self.store[(service, username)] = password

    def delete_password(self, service: str, username: str) -> None:
        if (service, username) not in self.store:
            raise KeyError("missing")
        del self.store[(service, username)]


@pytest.fixture
def fake_keyring(monkeypatch: pytest.MonkeyPatch) -> _FakeKeyring:
    fake = _FakeKeyring()
    monkeypatch.setattr(auth, "_keyring", fake)
    return fake


# ---------------------------------------------------------------------------
# OpenAI-only onboarding
# ---------------------------------------------------------------------------

def test_save_credential_for_openai_when_no_anthropic_switches_active(
    tmp_path: Path, fake_keyring: _FakeKeyring,
) -> None:
    """A fresh bridge defaults to Anthropic. If the researcher saves
    an OpenAI key (and has no Anthropic auth at all), the bridge
    must promote OpenAI to the active provider — otherwise the
    first turn opens an Anthropic session with no credential."""
    bridge = NoraBridge(cwd=tmp_path)
    assert bridge._provider == "anthropic"  # default at construction

    res = bridge.save_credential("openai", "sk-test-openai-1234")

    assert res["ok"] is True
    assert bridge._provider == "openai"
    # Model also gets bumped to OpenAI's default (otherwise we'd
    # be on a Claude model id under an OpenAI session, which the
    # provider catalog would reject).
    from nora.provider.catalog import PROVIDER_DEFAULTS
    assert bridge._model == PROVIDER_DEFAULTS["openai"]


def test_save_credential_for_openai_when_anthropic_already_authed_keeps_anthropic(
    tmp_path: Path, fake_keyring: _FakeKeyring,
) -> None:
    """A researcher who configured Claude first and adds OpenAI
    second shouldn't get silently switched to OpenAI. Reconcile
    only acts when the active provider is unauthed."""
    fake_keyring.set_password("nora", "anthropic", "sk-ant-existing")
    bridge = NoraBridge(cwd=tmp_path)
    # Re-reconcile to pick up the keyring-stashed Anthropic key —
    # the constructor doesn't run reconcile on its own.
    bridge._reconcile_active_provider_with_auth()
    assert bridge._provider == "anthropic"

    bridge.save_credential("openai", "sk-test-openai-1234")
    assert bridge._provider == "anthropic", (
        "saving an additional credential must not steal the active "
        "provider from a Claude-first researcher"
    )


def test_ui_ready_reconciles_provider(
    tmp_path: Path, fake_keyring: _FakeKeyring,
) -> None:
    """If the bridge somehow ended up on an unauthed provider (stale
    state, manual mutation), ``ui_ready`` is the catch-all that
    fixes it before the JS reads ``current_provider`` from
    ``list_models``."""
    fake_keyring.set_password("nora", "openai", "sk-openai-only")
    bridge = NoraBridge(cwd=tmp_path)
    # Manually wedge the bridge into a bad state.
    bridge._provider = "anthropic"
    bridge._model = "claude-sonnet-4-6[1m]"

    bridge.ui_ready()
    assert bridge._provider == "openai"


# ---------------------------------------------------------------------------
# Anthropic credential delete stickiness
# ---------------------------------------------------------------------------

def test_delete_anthropic_credential_clears_injected_env(
    tmp_path: Path, fake_keyring: _FakeKeyring,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When _ensure_anthropic_env copied the keyring credential into
    ANTHROPIC_API_KEY, deleting the keyring entry must also clear
    the env var. Without this, detect_auth keeps reporting
    ``api_key`` and the auth screen claims Anthropic is still
    configured until the app restarts."""
    fake_keyring.set_password("nora", "anthropic", "sk-ant-injected")
    # Trigger the env injection — this is what AnthropicSession.open
    # does in production.
    from nora.provider import anthropic as a_mod
    a_mod._ENV_INJECTED_BY_NORA = False  # reset module state
    a_mod._ensure_anthropic_env()
    assert os.environ.get("ANTHROPIC_API_KEY") == "sk-ant-injected"
    assert a_mod._ENV_INJECTED_BY_NORA is True

    # Now delete the credential through the bridge.
    bridge = NoraBridge(cwd=tmp_path)
    bridge.delete_credential("anthropic")

    # Env var cleared; flag reset.
    assert "ANTHROPIC_API_KEY" not in os.environ
    assert a_mod._ENV_INJECTED_BY_NORA is False


def test_delete_does_not_clear_env_set_by_user_shell(
    tmp_path: Path, fake_keyring: _FakeKeyring,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the researcher exported ANTHROPIC_API_KEY in their shell
    BEFORE Nora started, _ensure_anthropic_env should never touch
    it — so deleting the keyring entry must not clear the
    user-set env either."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-from-shell")
    from nora.provider import anthropic as a_mod
    a_mod._ENV_INJECTED_BY_NORA = False  # clean state
    a_mod._ensure_anthropic_env()
    # No injection flag — env was already set.
    assert a_mod._ENV_INJECTED_BY_NORA is False

    bridge = NoraBridge(cwd=tmp_path)
    bridge.delete_credential("anthropic")

    # User's shell-exported env stays.
    assert os.environ.get("ANTHROPIC_API_KEY") == "sk-ant-from-shell"
