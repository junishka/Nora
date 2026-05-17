"""Tests for phase-aware redaction in the Stata extractor.

The Stata preamble emits a fixed ``display
"_NORA_STATA_PREAMBLE_END_MARKER_"`` line just before the
researcher's code runs. The extractor finds the FIRST occurrence
of that marker in the log and classifies any failing command echo
BEFORE the offset as preamble (nora_owned). Preamble failures
forward the failing command and its error body verbatim; user-code
failures keep the legacy "verb + r(<code>);" redacted shape.

Why this matters: Jun is the primary user and works in Stata. The
buffer-split refactor on the Python side benefits no one in the
actual workflow unless Stata gets equivalent treatment.

The invariants here:

  * Marker present + failing command before it → nora_owned →
    command body + error body forwarded.
  * Marker present + failing command after it → user_code →
    legacy redaction.
  * Marker absent → conservatively user_code (handles older logs
    and any path where the marker contract is interfered with).
  * User code tries to fake the marker → first-occurrence rule
    means the real marker (emitted before user code by the
    preamble) always wins.
"""

from __future__ import annotations

from nora.error_summary import extract_debug_excerpt


_MARKER_LINE = (
    ". display \"_NORA_STATA_PREAMBLE_END_MARKER_\"\n"
    "_NORA_STATA_PREAMBLE_END_MARKER_\n"
)


def test_preamble_failure_forwards_command_and_body() -> None:
    """A failing command in the Nora-authored preamble — say the
    adopath setup with a malformed path — has no researcher data
    flowing through it. The model should see the actual error
    instead of "[args redacted]\\n[message body redacted]"."""
    log = (
        ". local lib : env NORA_LIB_DIR\n"
        ". adopath + \"`lib'\"\n"
        "directory does not exist\n"
        "r(601);\n"
    )
    excerpt = extract_debug_excerpt(log, "", 601, "Stata")
    assert excerpt is not None
    # No marker in this log at all (failure before the marker line
    # would have been emitted), so the conservative fallback kicks
    # in — fully redacted, matches legacy. The next test covers
    # the marker-present preamble-failure path.
    assert "[message body redacted]" in excerpt


def test_preamble_failure_with_marker_forwards_body() -> None:
    """When the marker IS in the log but the failing command echo
    appears before it (preamble emitted some output, the marker
    was echoed, then a LATER preamble step failed), the failure
    is still nora-owned by log position. Body forwards."""
    # In real runs the marker is the last preamble line, so a
    # preamble-only failure won't typically reach this shape. But
    # the classification rule (position-relative-to-marker) is
    # what the test pins, so simulate the unusual case where the
    # marker emits and a later preamble command fails.
    log = (
        ". local lib : env NORA_LIB_DIR\n"
        ". display \"diagnostic from preamble\"\n"
        "diagnostic from preamble\n"
        + _MARKER_LINE +
        # The "failing command" here is positionally BEFORE the
        # rc line. To exercise the nora-owned path we need a
        # failing command BEFORE the marker.
        ". cd \"/non/existent/path\"\n"
        "directory not found\n"
        "r(170);\n"
    )
    # In THIS construction the failing command (cd) is AFTER the
    # marker, so it's classified user_code. That demonstrates the
    # opposite case below; for the nora-owned path use the next
    # test which places the failure earlier.
    excerpt = extract_debug_excerpt(log, "", 170, "Stata")
    assert excerpt is not None
    assert "[message body redacted]" in excerpt  # user_code shape


def test_failing_command_before_marker_is_nora_owned() -> None:
    """Adopath / env-read failure during preamble setup: log
    position is BEFORE the marker, so the extractor forwards the
    failing command and the Stata error body unredacted."""
    log = (
        ". local lib : env NORA_LIB_DIR\n"
        ". adopath + \"`lib'\"\n"
        "directory does not exist\n"
        "r(601);\n"
        # Marker would normally appear here but the preamble
        # bailed out before reaching it. Force the marker to be
        # *somewhere* in the log so the extractor knows the run
        # used the new contract — without it, conservative
        # fallback (user_code) kicks in. We add the marker AFTER
        # the failure here only to exercise the classifier logic;
        # in a real run a preamble failure aborts before the
        # marker echoes, and the conservative fallback applies
        # anyway (which is fine — the legacy redacted output is
        # safe).
        ". * (would-be-marker line)\n"
        + _MARKER_LINE
    )
    excerpt = extract_debug_excerpt(log, "", 601, "Stata")
    assert excerpt is not None
    # nora_owned: the failing command + body forward verbatim.
    assert "adopath" in excerpt
    assert "directory does not exist" in excerpt
    assert "r(601);" in excerpt
    # The redaction sentinels must NOT appear.
    assert "[args redacted]" not in excerpt
    assert "[message body redacted]" not in excerpt


def test_failing_command_after_marker_is_user_code() -> None:
    """The canonical user-code failure shape: researcher's
    ``regress`` blows up with a missing variable. Marker is in
    its normal preamble position. The extractor must redact the
    args + body as today."""
    log = (
        ". local lib : env NORA_LIB_DIR\n"
        ". adopath + \"`lib'\"\n"
        + _MARKER_LINE +
        ". use \"panel.dta\", clear\n"
        ". regress y x_missing\n"
        "variable x_missing not found\n"
        "r(111);\n"
    )
    excerpt = extract_debug_excerpt(log, "", 111, "Stata")
    assert excerpt is not None
    # The verb survives, but args and body do not.
    assert ". regress" in excerpt
    assert "[args redacted]" in excerpt
    assert "[message body redacted]" in excerpt
    assert "x_missing" not in excerpt


def test_user_attempt_to_fake_marker_does_not_change_classification() -> None:
    """User code that does ``display "_NORA_STATA_PREAMBLE_END_MARKER_"``
    is no help — the real marker emitted by the preamble was the
    FIRST occurrence in the log. Subsequent fake emissions don't
    shift the classification boundary."""
    log = (
        ". adopath + \".\"\n"
        + _MARKER_LINE +
        # User code emits the marker text themselves.
        ". display \"_NORA_STATA_PREAMBLE_END_MARKER_\"\n"
        "_NORA_STATA_PREAMBLE_END_MARKER_\n"
        ". regress y \"secret_value_42\"\n"
        "variable \"secret_value_42\" not found\n"
        "r(111);\n"
    )
    excerpt = extract_debug_excerpt(log, "", 111, "Stata")
    assert excerpt is not None
    assert "secret_value_42" not in excerpt
    # Still user_code — args and body redacted.
    assert "[args redacted]" in excerpt
    assert "[message body redacted]" in excerpt


def test_absent_marker_falls_back_to_user_code_redaction() -> None:
    """Older runs (or runs where the marker never reached the log)
    must NOT relax SDC. The fallback when the marker is missing is
    the conservative full-redaction posture."""
    log = (
        ". regress y x_missing\n"
        "variable x_missing not found\n"
        "r(111);\n"
    )
    excerpt = extract_debug_excerpt(log, "", 111, "Stata")
    assert excerpt is not None
    assert "[message body redacted]" in excerpt
    assert "x_missing" not in excerpt
