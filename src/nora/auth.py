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

import time
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

# Process-lifetime cache of resolved credentials. The auth screen's
# payload builder calls ``has_credential`` and ``detect_auth`` per
# provider per render, and ``detect_auth`` itself calls
# ``has_credential`` internally — without this cache an unsigned build
# triggers a fresh macOS Keychain prompt on every redundant lookup,
# which on first launch presents as 4+ "allow access" dialogs in a row.
# A signed build's Keychain ACL covers all reads from the same binary,
# but the cache helps even there by avoiding repeated IPC to securityd.
#
# Mutations (set/delete) write through to the cache so callers see a
# coherent view immediately; otherwise the auth screen would show
# stale "configured" badges right after a save until the next launch.
#
# Only successful reads land here. Backend errors (locked Keychain,
# denied prompt, transient IPC failure) go through ``_CRED_ERROR_AT``
# instead — caching ``None`` on error would conflate "definitely
# missing" with "couldn't tell" and lock the app into a no-creds view
# for the rest of the process even after the user grants access.
_CRED_CACHE: dict[str, str | None] = {}

# Monotonic timestamp of the last keyring exception per provider. While
# a timestamp is within ``_ERROR_BACKOFF_SECONDS`` of now, ``get_credential``
# returns ``None`` without re-hitting the backend. This preserves the
# burst-suppression that the cache provides during a single auth-screen
# render (4 redundant ``has_credential`` calls in microseconds → one
# keyring hit, no prompt storm), while letting the read recover within
# seconds once the underlying issue (locked keychain, denied prompt) is
# resolved — instead of staying poisoned until process restart.
_CRED_ERROR_AT: dict[str, float] = {}
_ERROR_BACKOFF_SECONDS = 5.0


def get_credential(provider: str, *, force_refresh: bool = False) -> str | None:
    """Return the stored API key for ``provider``, or ``None`` if no
    credential is stored or the keyring backend is unavailable.

    ``force_refresh=True`` drops the cached entry and re-reads from
    the OS keyring. Use it from the auth-screen render path so a
    credential that was deleted directly in Keychain Access (outside
    Nora) is noticed on the next page load instead of waiting until
    the next process start. Default behavior preserves the burst-
    suppression cache the auth-screen render relies on.

    Never raises — auth lookup must always have a definite answer
    (set / unset) so callers can route to the right UI without a
    try/except dance.
    """
    if force_refresh:
        # Drop the cached value AND any stale backoff for this provider
        # so we actually re-hit the backend. The cache will be
        # rewritten below with whatever the fresh read produces.
        _CRED_CACHE.pop(provider, None)
        _CRED_ERROR_AT.pop(provider, None)
    if provider in _CRED_CACHE:
        return _CRED_CACHE[provider]
    if _keyring is None:
        _CRED_CACHE[provider] = None
        return None
    last_error = _CRED_ERROR_AT.get(provider)
    if (
        last_error is not None
        and time.monotonic() - last_error < _ERROR_BACKOFF_SECONDS
    ):
        # Recent backend error — return None without re-hitting keyring
        # to avoid prompt-storming the user during this render. Will
        # retry once the backoff window elapses.
        return None
    try:
        value = _keyring.get_password(KEYRING_SERVICE, provider)
    except Exception:  # noqa: BLE001 — backend errors mean "couldn't tell"
        # Record the error timestamp instead of caching ``None``.
        # Caching ``None`` on a transient backend hiccup conflated
        # "definitely missing" with "couldn't tell" — the auth
        # screen reported the credential gone, the researcher saved
        # a new key, and the UI still showed the provider as unauthed
        # until restart. ``_CRED_ERROR_AT`` + ``_ERROR_BACKOFF_SECONDS``
        # gives us both: prompt-storm suppression within a single
        # render (multiple ``has_credential`` calls in microseconds
        # → one keyring hit), AND automatic recovery once the backoff
        # window elapses (no restart needed).
        _CRED_ERROR_AT[provider] = time.monotonic()
        return None
    # Successful read — clear any stale error backoff and cache the
    # value (or true-miss ``None``) for the rest of the process.
    _CRED_ERROR_AT.pop(provider, None)
    resolved = value if value else None
    _CRED_CACHE[provider] = resolved
    return resolved


# Status tokens returned by ``credential_state``. Boolean-only
# ``has_credential`` cannot distinguish "definitely no credential" from
# "keyring is locked / denied / unavailable, so we don't know": the
# auth screen would render the same "Not configured" badge for both,
# and a researcher whose Keychain prompt was denied would re-paste a
# key that's already present (or worse, dismiss the prompt thinking
# the credential was forgotten when it's actually still there). The
# tri-state separates those cases so the UI can render the right copy.
AUTH_STATE_CONFIGURED = "configured"
AUTH_STATE_MISSING = "missing"
AUTH_STATE_KEYRING_UNAVAILABLE = "keyring_unavailable"


def credential_state(
    provider: str, *, force_refresh: bool = False,
) -> str:
    """Tri-state credential check. Returns one of
    ``"configured"``, ``"missing"``, ``"keyring_unavailable"``.

    Callers that need to render an auth surface — and only those —
    should use this. Internal hot-path callers (``provider/openai``,
    ``provider/anthropic``, the model picker) can stay on
    ``has_credential`` since they only need the boolean.

    The third state fires when the most recent keyring read raised
    an exception within the backoff window. Outside the window, a
    transient backend recovery resets the state — the UI is meant
    to reflect "I can't currently tell" rather than "definitely
    gone."
    """
    value = get_credential(provider, force_refresh=force_refresh)
    if value is not None:
        return AUTH_STATE_CONFIGURED
    if provider in _CRED_ERROR_AT:
        return AUTH_STATE_KEYRING_UNAVAILABLE
    return AUTH_STATE_MISSING


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
    cleaned = api_key.strip()
    try:
        _keyring.set_password(KEYRING_SERVICE, provider, cleaned)
    except Exception as e:  # noqa: BLE001 — surface to caller
        _CRED_ERROR_AT[provider] = time.monotonic()
        return {"ok": False, "reason": f"keyring write failed: {e}"}
    _CRED_CACHE[provider] = cleaned
    _CRED_ERROR_AT.pop(provider, None)
    return {"ok": True, "provider": provider}


def delete_credential(provider: str) -> dict[str, object]:
    """Remove the stored credential for ``provider``. Idempotent for
    a genuinely-absent entry (delete-of-nothing succeeds), but
    backend failures (locked Keychain, denied prompt, securityd
    error) MUST surface as ``ok: False`` — silently swallowing them
    leaves the secret in the OS store while the UI shows it as
    forgotten, ready to reappear on the next launch.
    """
    if _keyring is None:
        _CRED_CACHE[provider] = None
        return {"ok": True, "provider": provider}

    # Pre-check so we can distinguish "entry was already absent →
    # idempotent OK" from "delete itself failed".
    # ``keyring.delete_password`` wraps every backend error in
    # ``PasswordDeleteError`` regardless of cause (item-not-found vs.
    # backend locked vs. permission denied), so the exception type
    # alone can't tell us which we hit. Reading first is portable.
    #
    # ``force_refresh=True`` so the answer reflects the live keyring,
    # not a cached value from a prior auth-screen render. Without
    # this, a researcher who deletes the credential directly in
    # Keychain Access while Nora's auth panel is still open hits
    # this path with a stale cache: pre-check sees the old key,
    # ``delete_password`` raises ``PasswordDeleteError`` because the
    # entry's already gone, and we surface ``ok=False`` even though
    # the target state ("credential is absent") is already met.
    # Delete is rare and user-initiated, so paying one extra keyring
    # hit per call is cheap and worth the correctness.
    try:
        existing = get_credential(provider, force_refresh=True)
    except Exception:  # noqa: BLE001 — get_credential is non-raising; defensive
        existing = None

    if existing is None:
        # Either really absent, or a recent backend error left the
        # backoff window active. In the backoff case we can't
        # confidently claim success — but if the read errored
        # transiently, ``_CRED_ERROR_AT`` was set by ``get_credential``.
        # Surface that as a failure rather than a silent no-op so the
        # UI doesn't report a deletion that didn't happen.
        if provider in _CRED_ERROR_AT:
            return {
                "ok": False,
                "reason": (
                    "keyring backend is unavailable. Could not confirm "
                    "current credential state. Try again once the "
                    "system store is reachable."
                ),
            }
        _CRED_CACHE[provider] = None
        return {"ok": True, "provider": provider}

    try:
        _keyring.delete_password(KEYRING_SERVICE, provider)
    except Exception as e:  # noqa: BLE001 — backend failure, not idempotent
        _CRED_ERROR_AT[provider] = time.monotonic()
        return {"ok": False, "reason": f"keyring delete failed: {e}"}
    _CRED_CACHE[provider] = None
    _CRED_ERROR_AT.pop(provider, None)
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
