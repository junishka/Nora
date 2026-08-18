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

import json
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
    bridge._model = "claude-sonnet-5[1m]"

    bridge.ui_ready()
    assert bridge._provider == "openai"


def test_reconcile_skips_swap_when_active_runner_busy(
    tmp_path: Path, fake_keyring: _FakeKeyring,
) -> None:
    """``_reconcile_active_provider_with_auth`` must NOT swap the
    active runner mid-stream. ``swap_model`` closes and reopens the
    provider session under the hood, which would tear down a live
    turn — and this is exactly the case ``delete_credential``
    deliberately spares (it leaves busy runners alone). Without this
    guard, a credential delete + page reload would race ahead of
    that policy and replace the very session ``delete_credential``
    refused to touch.
    """
    # Set up: researcher has Anthropic configured, then OpenAI gets
    # added, then Anthropic credential gets removed mid-turn.
    fake_keyring.set_password("nora", "openai", "sk-openai-only")
    bridge = NoraBridge(cwd=tmp_path)
    # Pretend the active runner is on Anthropic and currently busy.
    active = bridge._active_runner()
    assert active is not None
    active.provider = "anthropic"
    active.model = "claude-sonnet-5[1m]"

    class _BusyTask:
        def done(self) -> bool: return False
        def cancel(self) -> None: pass
        def get_loop(self): return None

    active._current_turn_task = _BusyTask()  # type: ignore[assignment]
    active._current_turn_id = "t-busy"

    swap_called = False
    real_swap = active.swap_model

    async def _spy(*args, **kwargs):
        nonlocal swap_called
        swap_called = True
        return await real_swap(*args, **kwargs)

    active.swap_model = _spy  # type: ignore[assignment]

    # ui_ready triggers reconciliation.
    bridge.ui_ready()

    # Bridge defaults still updated (they don't touch the live
    # session).
    assert bridge._default_provider == "openai"
    # But the in-flight runner is left alone.
    assert active.provider == "anthropic"
    assert swap_called is False, (
        "swap_model must not be called on a busy runner — closing "
        "the provider session mid-turn would tear down the live "
        "stream that delete_credential deliberately spared."
    )

    # Cleanup before pytest tears down so close() doesn't trip.
    active._current_turn_task = None
    active._current_turn_id = None


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


# ---------------------------------------------------------------------------
# Subscription wins over a stale keyring credential
# ---------------------------------------------------------------------------

def test_ensure_env_skips_keyring_when_subscription_present(
    tmp_path: Path, fake_keyring: _FakeKeyring,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``detect_auth`` checks ``~/.claude.json`` BEFORE the keyring,
    so a researcher with a valid Claude subscription AND a stale
    keyring API key gets ``subscription`` reported. But
    ``_ensure_anthropic_env`` then injected the keyring credential
    into ``ANTHROPIC_API_KEY`` anyway — and the SDK prefers an
    explicit env API key over OAuth, silently routing the
    researcher's traffic through the stale (possibly-defunct) key
    and bypassing the subscription. Verify the injection is now
    suppressed when subscription is detected.
    """
    # Subscription configured: a ``~/.claude.json`` with an OAuth
    # account. Override the outer fixture's "always missing" Path
    # stub for this test.
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    claude_json = fake_home / ".claude.json"
    claude_json.write_text(json.dumps({
        "oauthAccount": {"accountUuid": "abc-123-uuid"},
    }), encoding="utf-8")

    class _StubHome:
        @staticmethod
        def home() -> Path:
            return fake_home

    monkeypatch.setattr("nora.provider.anthropic.Path", _StubHome)

    # Plus a stale keyring credential lingering from a prior session.
    fake_keyring.set_password("nora", "anthropic", "sk-ant-STALE")

    from nora.provider import anthropic as a_mod
    a_mod._ENV_INJECTED_BY_NORA = False  # reset module state

    # Sanity check: detect_auth must agree subscription is the
    # configured path — otherwise this test is testing the wrong
    # precondition.
    assert a_mod.detect_auth() == "subscription"

    a_mod._ensure_anthropic_env()

    # The injection must NOT have happened. With the bug, this
    # assertion fails: ANTHROPIC_API_KEY would be "sk-ant-STALE",
    # overriding the subscription path on the next SDK call.
    assert "ANTHROPIC_API_KEY" not in os.environ, (
        "subscription must take priority over a stale keyring "
        "credential; injecting the keyring key would silently "
        "route traffic through the (possibly-defunct) API key "
        "and bypass the researcher's active Claude subscription"
    )
    assert a_mod._ENV_INJECTED_BY_NORA is False
