"""Shared fixtures for the Nora test suite."""

from __future__ import annotations

import pytest

from nora import auth


@pytest.fixture
def anthropic_authed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend this machine has a working Anthropic credential (and no
    OpenAI one) without touching the real keyring, env vars, or
    ``~/.claude.json``.

    Auth detection is machine state, not fixture state: bridge tests
    that assert ``ready``-state routing or per-session model restore
    were green only on machines whose keyring (or Claude CLI config)
    held a real credential, and failed on a fresh checkout where
    ``ui_ready`` returned ``needs_auth`` and ``_initial_model_for_session``
    skipped the restore. Tests exercising authed behaviour must take
    this fixture instead of relying on the developer's credentials.

    ``detect_auth`` is the single seam every authed-or-not decision
    flows through (``_authed_providers``, ``_auth_status_payload``,
    ``_reconcile_active_provider_with_auth`` all import it at call
    time), so one stub covers routing. The keyring handle and caches
    are neutralised too so the auth-screen payload's secondary fields
    (``has_keyring_entry``, ``status``) don't issue live Keychain
    reads — which on macOS can pop access prompts mid-test.
    """
    import nora.provider as provider_mod

    monkeypatch.setattr(
        provider_mod, "detect_auth",
        lambda p: "api_key" if p == "anthropic" else "unknown",
    )
    monkeypatch.setattr(auth, "_keyring", None)
    monkeypatch.setattr(auth, "_CRED_CACHE", {})
    monkeypatch.setattr(auth, "_CRED_ERROR_AT", {})
