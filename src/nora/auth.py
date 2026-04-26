"""Per-provider credential storage backed by the OS keyring.

Nora keeps API keys in the OS-native secret store (macOS Keychain via
``keyring``) so they never sit in cleartext on disk and survive across
upgrades / reinstalls. The auth screen writes here; each provider's
session reads from here at open time.

Provider IDs in this module match those used elsewhere
(``"anthropic"``, ``"openai"``). The keyring entries live under
service name ``KEYRING_SERVICE`` (``"nora"``) with the provider id as
the keyring username — so a researcher can spot Nora entries in
Keychain Access by searching for "nora".

The Anthropic CLI subscription path (``~/.claude.json``) lives outside
this module — only API keys flow through here. ``detect_auth`` in
``provider/anthropic.py`` consults both sources.
"""

from __future__ import annotations

from typing import Iterable

try:
    import keyring as _keyring
except ImportError:  # pragma: no cover — keyring is in dependencies
    _keyring = None  # type: ignore[assignment]


KEYRING_SERVICE = "nora"

# Providers Nora knows how to store credentials for. Kept in sync with
# ``provider.SUPPORTED_PROVIDERS``; defined separately to avoid an
# import cycle (``auth`` is imported from ``provider/__init__.py``).
KNOWN_PROVIDERS: tuple[str, ...] = ("anthropic", "openai")


def get_credential(provider: str) -> str | None:
    """Return the stored API key for ``provider``, or ``None`` if no
    credential is stored or the keyring backend is unavailable.

    Never raises — auth lookup must always have a definite answer
    (set / unset) so callers can route to the right UI without a
    try/except dance.
    """
    if _keyring is None:
        return None
    try:
        value = _keyring.get_password(KEYRING_SERVICE, provider)
    except Exception:  # noqa: BLE001 — backend errors mean "no creds"
        return None
    if not value:
        return None
    return value


def set_credential(provider: str, api_key: str) -> dict[str, object]:
    """Store an API key for ``provider``. Returns
    ``{"ok": True, "provider": ...}`` on success or
    ``{"ok": False, "reason": ...}`` on failure.

    Empty / whitespace-only values are rejected — saving a blank key
    is almost always a UI bug.
    """
    if provider not in KNOWN_PROVIDERS:
        return {"ok": False, "reason": f"unknown provider: {provider!r}"}
    if not api_key or not api_key.strip():
        return {"ok": False, "reason": "API key is empty"}
    if _keyring is None:
        return {"ok": False, "reason": "keyring backend not available"}
    try:
        _keyring.set_password(KEYRING_SERVICE, provider, api_key.strip())
    except Exception as e:  # noqa: BLE001 — surface to caller
        return {"ok": False, "reason": f"keyring write failed: {e}"}
    return {"ok": True, "provider": provider}


def delete_credential(provider: str) -> dict[str, object]:
    """Remove the stored credential for ``provider``. Idempotent —
    deleting a missing entry is success, not failure."""
    if _keyring is None:
        return {"ok": True, "provider": provider}
    try:
        _keyring.delete_password(KEYRING_SERVICE, provider)
    except Exception:  # noqa: BLE001 — typically "no such entry"; idempotent OK
        pass
    return {"ok": True, "provider": provider}


def has_credential(provider: str) -> bool:
    """Quick yes/no check. Equivalent to ``get_credential(...) is not None``
    but conveys intent more clearly at call sites."""
    return get_credential(provider) is not None


def list_authed_providers() -> list[str]:
    """Return the providers with a stored credential, in the order
    given by ``KNOWN_PROVIDERS``. Used by the auth screen to render
    "already configured" badges and by the model picker to filter
    rows by what the researcher can actually use."""
    return [p for p in KNOWN_PROVIDERS if has_credential(p)]


def auth_summary(providers: Iterable[str] | None = None) -> dict[str, bool]:
    """Render ``{provider: bool}`` of credential presence. The bridge
    surfaces this to the auth screen so it can show check / cross
    badges next to each provider row in one round-trip."""
    if providers is None:
        providers = KNOWN_PROVIDERS
    return {p: has_credential(p) for p in providers}
