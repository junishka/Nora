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

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# Fields the bridge enriches into ``tool_result`` records for UI
# replay (raw stdout/stderr captures, base64 plot thumbnails, plot
# diagnostic strings) but that NEVER ride into the next provider
# request. Stripping them at counting time gives the chip a
# realistic estimate of next-turn size — without this, plot- and
# script-heavy sessions overcounted dramatically because a single
# tool_result line could be 1-2 MB of base64 plot data the model
# will never see.
_HISTORY_UI_ONLY_FIELDS: frozenset[str] = frozenset({
    "raw_stdout",
    "raw_stderr",
    "plots",
    "plot_diagnostic",
})


def _model_facing_history_chars(history_path: Path) -> int:
    """Sum the bytes of the persisted chat log MINUS UI-only fields.

    The persisted ``chat_history.jsonl`` mixes model-facing record
    bodies (``user_message`` text, ``assistant_text``, ``tool_call``
    args, sanitized ``tool_result`` payloads) with UI-only
    enrichments (full raw stdout/stderr captures, base64 plot
    thumbnails). The model never sees the UI-only fields, but
    ``stat()`` on the file counts them — the chip's denominator
    pressure (raw bytes / chars-per-token) was therefore badly
    inflated on plot-heavy or script-heavy sessions.

    Streaming JSON parse, one line at a time. A 100-MB log is the
    realistic upper bound (50+ tool calls each carrying ~1-2 MB of
    plot bytes); on that size this scan takes ~1-2s. The chip's
    triggers are infrequent enough that the cost is acceptable;
    caching by file mtime is the obvious follow-up if it bites.
    """
    if not history_path.is_file():
        return 0
    total = 0
    try:
        with history_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    # Unparseable lines shouldn't happen in normal
                    # operation. Count their full length defensively
                    # — better to over-report than under-report when
                    # we genuinely don't know what they contain.
                    total += len(line) + 1
                    continue
                if not isinstance(rec, dict):
                    total += len(line) + 1
                    continue
                ui_only_present = any(
                    k in rec for k in _HISTORY_UI_ONLY_FIELDS
                )
                if not ui_only_present:
                    # Fast path: no enrichment to strip. Use the
                    # original line length (saves a json.dumps).
                    total += len(line) + 1
                    continue
                stripped = {
                    k: v for k, v in rec.items()
                    if k not in _HISTORY_UI_ONLY_FIELDS
                }
                total += len(json.dumps(stripped, ensure_ascii=False)) + 1
    except OSError:
        return 0
    return total


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
    pending_attachment_chars: int = 0,
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
    - ``n_pending_attachments``: pending script attachment count. A
      small per-attachment kicker covers the framing the bridge
      wraps each one in (header + fence). Their *content* bytes
      ride separately in ``pending_attachment_chars`` because the
      chip needs to reflect a 90 KB ``.do`` file the moment the
      researcher attaches it, not only after the next turn commits.
    - ``pending_attachment_chars``: summed length of inlined script
      content the next turn will prepend (post per-file truncation
      and aggregate cap). Caller computes this against the runner's
      staging list so the count matches what
      ``_build_script_attachment_prefix`` will actually emit.
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
        # Project to model-facing fields only. The persisted log
        # carries raw stdout/stderr captures and base64 plot
        # thumbnails for UI replay; those never ride into the next
        # provider request, so a stat() of the whole file would
        # overcount badly on plot- or script-heavy sessions.
        history_chars = _model_facing_history_chars(history_path)

    # Draft attachments aren't in history yet. Two contributions:
    #   - per-file kicker for the header / fence framing the bridge
    #     wraps each attachment in, so the chip moves the moment a
    #     file is attached even if its bytes are tiny;
    #   - the actual inlined content bytes (post-truncation, post-
    #     aggregate-cap) so a 90 KB ``.do`` file shifts the chip
    #     proportionally instead of looking like a 200-char nudge.
    # Real bytes also ride in once the turn commits and the next
    # recount picks them up from the history file — at that point
    # the runner's staging list is empty and these contributions
    # drop out.
    attachment_kicker_chars = (
        n_pending_attachments * 200 + pending_attachment_chars
    )

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
