"""Backend cap on ``NoraBridge.send_feedback``.

The feedback modal is the one carve-out where text deliberately
leaves the researcher's machine — it POSTs to
``api.web3forms.com`` which forwards to the maintainer's inbox.
The modal copy ("only this note leaves your machine. Chat history
and datasets stay local") is only honest if "this note" stays
bounded. A researcher in a hurry can paste a CSV chunk, a full
log, or a credentials snippet into the textarea; without a cap
that's exactly what gets POSTed.

The cap lives in three places that MUST stay in lockstep:

  * ``ui._FEEDBACK_MAX_MESSAGE_CHARS`` — the Python-side hard cap
    enforced before any network call.
  * ``index.html`` textarea ``maxlength`` — stops normal typing.
  * ``app.js`` ``FEEDBACK_MESSAGE_CAP`` — drives the live counter.

These tests pin the Python side. The HTML maxlength is a UX
affordance; the JS counter is read-only feedback for the user.
The backend rejection is the load-bearing privacy guardrail —
catches any path that reaches the bridge endpoint outside the
modal (direct page-JS, future automation, etc.).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from nora.ui import NoraBridge, _FEEDBACK_MAX_MESSAGE_CHARS


def _bridge() -> NoraBridge:
    """Bridge with no active cwd — ``send_feedback`` doesn't need
    a session and shouldn't read from one."""
    return NoraBridge(cwd=None)


def test_empty_message_rejected_without_network() -> None:
    """Blank feedback never hits Web3Forms — no payload to send."""
    with patch("urllib.request.urlopen") as mock_urlopen:
        result = _bridge().send_feedback(message="")
    assert result["ok"] is False
    assert "empty" in result["reason"].lower()
    mock_urlopen.assert_not_called()


def test_whitespace_only_message_rejected() -> None:
    """``"   \\n\\t  "`` strips to empty; same path as blank."""
    with patch("urllib.request.urlopen") as mock_urlopen:
        result = _bridge().send_feedback(message="   \n\t  ")
    assert result["ok"] is False
    mock_urlopen.assert_not_called()


def test_oversized_message_rejected_before_network_call() -> None:
    """The privacy guardrail: an over-cap message must be rejected
    by the bridge BEFORE any bytes leave the machine. Confirm both
    the rejection AND that ``urlopen`` was never called — the cap
    is meaningless if the message rides through anyway."""
    oversized = "x" * (_FEEDBACK_MAX_MESSAGE_CHARS + 1)
    with patch("urllib.request.urlopen") as mock_urlopen:
        result = _bridge().send_feedback(message=oversized)
    assert result["ok"] is False
    assert "too long" in result["reason"].lower()
    # Surface the limit so the JS layer (or a manual caller) can
    # show a useful error.
    assert str(_FEEDBACK_MAX_MESSAGE_CHARS) in result["reason"]
    # Load-bearing assertion: NO bytes hit web3forms.com.
    mock_urlopen.assert_not_called()


def test_message_at_exact_cap_is_accepted() -> None:
    """The cap is INCLUSIVE — ``len == cap`` passes the gate.
    Confirms the boundary check uses ``> cap`` not ``>= cap``;
    inverting that would reject a perfectly legitimate maximum-
    length report."""
    at_cap = "x" * _FEEDBACK_MAX_MESSAGE_CHARS
    # Stub urlopen with a fake 200 response so the network branch
    # of send_feedback doesn't actually fire. The cap acceptance
    # path then reaches the multipart-build + urlopen call.
    class _Fake200:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return b'{"success": true, "message": "ok"}'
    with patch("urllib.request.urlopen", return_value=_Fake200()):
        result = _bridge().send_feedback(message=at_cap)
    # Accepted at the cap — even though Web3Forms is faked. The
    # `ok` flag reflects the upstream response parse; what we're
    # pinning here is that the size check did NOT reject.
    assert result["ok"] is True


def test_subject_still_capped_at_120() -> None:
    """Subject is the OTHER text field that crosses to Web3Forms;
    its cap is enforced separately (already in place before this
    fix) and applies independently of the message cap. Pin both
    so a future refactor that touches one doesn't drop the other."""
    long_subject = "S" * 500
    class _Fake200:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return b'{"success": true, "message": "ok"}'
    # Capture the request body so we can inspect what would be POSTed.
    captured: dict = {}
    def _stub_urlopen(req, timeout=20):
        captured["body"] = req.data
        return _Fake200()
    with patch("urllib.request.urlopen", side_effect=_stub_urlopen):
        _bridge().send_feedback(
            message="hi", subject=long_subject,
        )
    # The 120-char subject cap means at most 120 ``S`` characters
    # land in the multipart body's subject field — search for the
    # subject delimiter and verify the value length.
    body = captured.get("body") or b""
    # Multipart bodies have ``name="subject"`` followed by the
    # header terminator and the value; search-and-extract is
    # enough for a sanity check without parsing the whole
    # multipart structure.
    needle = b'name="subject"'
    pos = body.find(needle)
    assert pos != -1, "subject field missing from POST body"
    # Skip to the value (after the blank line that follows the
    # header). The exact bytes-after-needle depend on multipart
    # formatting; just count consecutive ``S`` characters anywhere
    # in the body and bound them by the cap.
    s_runs = max(
        (len(run) for run in body.split(b"S") if False),
        default=0,
    )
    # Simpler: count total ``S`` bytes; the cap means there are
    # at most 120 of them coming from the subject field. The
    # message body is ``"hi"`` (no ``S``), the access key and
    # boundary contain none, so total ``S`` count equals the
    # forwarded subject length.
    assert body.count(b"S") == 120


@pytest.mark.parametrize(
    "size_offset",
    [
        -1,   # one under the cap
        0,    # exactly at the cap
    ],
)
def test_message_under_and_at_cap_does_not_short_circuit(size_offset: int) -> None:
    """The size-check branch in ``send_feedback`` must allow
    messages of length ``cap + size_offset`` for ``size_offset in
    {-1, 0}``. Anything in this range should at minimum reach the
    network call (which the test mocks). Catches accidental
    ``>=`` typos that would reject legitimate maximum reports."""
    text = "x" * (_FEEDBACK_MAX_MESSAGE_CHARS + size_offset)
    class _Fake200:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return b'{"success": true, "message": "ok"}'
    with patch("urllib.request.urlopen", return_value=_Fake200()) as mock_open:
        result = _bridge().send_feedback(message=text)
    mock_open.assert_called_once()
    assert result["ok"] is True
