"""Pre-flight context counter for the chip.

The chip used to mix four signals at once — provider-reported usage
from ``turn_done``, cache fields, ``post_turn_tokens``, and a
chars/4 estimate of pending messages — which produced visible
fluctuation that didn't correspond to any single useful question.

This module collapses that to one definition: **the assembled size
of the next request, measured the same way for every chip update.**
The bridge calls ``count_next_context`` on a small, well-defined set
of triggers (session open, rewind success, turn complete, attachment
add/remove); the JS chip only refreshes when the backend returns. No
intermediate estimates flicker through the UI.

Accuracy tier (``exact`` field on the response):
- OpenAI: planned to use ``tiktoken`` (local, exact). Today this
  module returns a chars/3.5 approximation for both providers; the
  exact path lands in a follow-up that adds the dep.
- Anthropic: no public local tokenizer matches server-side counting.
  When ``ANTHROPIC_API_KEY`` is set we'll route to
  ``messages/count_tokens`` (exact, costs an API call). Today: same
  chars/3.5 approximation. The chip text itself stays clean — no
  ``~`` prefix even when ``exact=False`` — because today every
  render is approximate, so a permanent prefix conveys no signal.
  The tooltip spells out that the value is approximate; once exact
  tokenization lands the tooltip wording switches and the chip text
  stays the same.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


# Average bytes per token for English-leaning prose with code mixed
# in. 3.5 sits between Claude's empirical 3.6-3.8 (per Anthropic's
# own published rule of thumb) and the chars/4 figure the OpenAI
# docs cite, biased slightly toward over-estimation so we don't
# under-report headroom. Replaced by exact tokenization once the
# tiktoken / count_tokens paths land.
_CHARS_PER_TOKEN_FALLBACK = 3.5

# Each image submitted as a vision content block costs roughly this
# many tokens at the providers' standard resolutions. A more precise
# accounting depends on the image's pixel dimensions; this constant
# is a safe over-estimate for the typical Stata / R plot.
_IMAGE_TOKEN_ESTIMATE = 1500


@dataclass
class ContextCount:
    """Result of one pre-flight count.

    ``tokens``: integer count for the chip's numerator.
    ``exact``: True when the count came from a provider-matching
       tokenizer (tiktoken for OpenAI, Anthropic's count_tokens for
       Claude); False for the chars/3.5 approximation. The chip
       prefixes ``~`` when False.
    ``ceiling``: model context window in tokens — the chip's
       denominator. Sourced from the model registry, not derived
       here, so this struct stays decoupled from per-provider model
       metadata.
    ``request_id``: monotonic id passed through from the caller so
       JS can reject stale responses landing after a newer request.
    """
    tokens: int
    exact: bool
    ceiling: int
    request_id: int


def count_next_context(
    cwd: Path | None,
    *,
    draft_text: str = "",
    n_images: int = 0,
    n_pending_attachments: int = 0,
    system_prompt_chars: int = 0,
    tool_schema_chars: int = 0,
    ceiling: int = 1_000_000,
    request_id: int = 0,
) -> ContextCount:
    """Count the size of the next request the bridge would assemble.

    Inputs cover every contributor the chip needs to reflect:

    - ``cwd``: location of ``.nora/chat_history.jsonl``. The full
      file's text bytes are summed — this is the conversation chain
      that re-rides on every turn (Anthropic) or that the warm-start
      prefix re-injects on session resume.
    - ``draft_text``: composer contents the next send would carry.
      Pass ``""`` when the chip is being recounted between turns
      (no in-flight draft).
    - ``n_images``: count of pending image attachments — counted via
      ``_IMAGE_TOKEN_ESTIMATE`` since chars-based math doesn't apply.
    - ``n_pending_attachments``: pending script attachments. Their
      bytes already live in the chat history if the previous turn
      committed them; this kicker adds a small constant so the chip
      moves immediately when a researcher attaches a script before
      send.
    - ``system_prompt_chars`` / ``tool_schema_chars``: caller passes
      lengths of the assembled system prompt and tool schemas (both
      provider-specific). Caller does this to avoid this module
      having to import provider modules and ride a circular-import
      risk.

    Returns ``ContextCount`` with ``exact=False`` until the
    tiktoken / count_tokens paths land — caller should branch on
    ``exact`` to decide whether to prefix the chip with ``~``.
    """
    history_chars = 0
    if cwd is not None:
        history_path = cwd / ".nora" / "chat_history.jsonl"
        if history_path.is_file():
            try:
                history_chars = history_path.stat().st_size
            except OSError:
                history_chars = 0

    # Draft attachments aren't included in history yet; the kicker
    # accounts for the framing the bridge will wrap them in (
    # ``[Attached file: name]\n<bytes>\n``). Real bytes ride in once
    # the turn commits and the next recount picks them up from the
    # history file.
    attachment_kicker_chars = n_pending_attachments * 200

    total_chars = (
        system_prompt_chars
        + tool_schema_chars
        + history_chars
        + len(draft_text)
        + attachment_kicker_chars
    )
    text_tokens = int(total_chars / _CHARS_PER_TOKEN_FALLBACK)
    image_tokens = n_images * _IMAGE_TOKEN_ESTIMATE
    return ContextCount(
        tokens=text_tokens + image_tokens,
        exact=False,
        ceiling=ceiling,
        request_id=request_id,
    )


def to_payload(count: ContextCount) -> dict[str, Any]:
    """Serialize for the JS bridge response. Keep field names stable
    — JS reads them directly into the chip render path."""
    return {
        "tokens": count.tokens,
        "exact": count.exact,
        "ceiling": count.ceiling,
        "request_id": count.request_id,
    }
