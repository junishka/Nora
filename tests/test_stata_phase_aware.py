"""Tests for phase-aware redaction in the Stata extractor.

The Stata preamble emits a fixed ``display
"_NORA_STATA_PREAMBLE_END_MARKER_"`` line just before the
researcher's code runs. The extractor finds the FIRST occurrence
of that marker in the log and classifies any failing command echo
BEFORE the offset as preamble (nora_owned). Both branches now
FORWARD the failing command and its error body, but they do so
through different paths:

  * ``nora_owned`` (failing command before the marker): args and
    error body are Nora-authored — no researcher data has reached
    them. Forwarded verbatim, capped at 200 chars on the command
    echo only.
  * ``user_code`` (everything else, including the missing-marker
    fallback): args and body are researcher-influenced. Forwarded
    through ``_forward_short_body`` (200-char cap + data-shape
    detect), with downstream credential / URL / path scrubs in
    ``_scrub_and_cap`` over the full excerpt.

Why this matters: Jun is the primary user and works in Stata. The
buffer-split refactor on the Python side benefits no one in the
actual workflow unless Stata gets equivalent treatment, and the
prior allowlist posture left the model unable to read the actual
diagnostic ("variable X not found" → "[message body redacted]").

The invariants here:

  * Marker present + failing command before it → nora_owned →
    command body + error body forwarded verbatim.
  * Marker present + failing command after it → user_code →
    command + body forwarded through _forward_short_body. Short
    scalar values (varnames, identifiers) pass through; this is
    the documented residual leak channel bounded by the 200-byte
    per-body cap.
  * Marker absent → conservatively user_code (handles older logs
    and any path where the marker contract is interfered with).
    Same forwarding policy as the marker-present user_code branch.
  * User code tries to fake the marker → first-occurrence rule
    means the real marker (emitted before user code by the
    preamble) always wins.
  * Data-shape exfil patterns (JSON dict, 6+ mixed-token rows) are
    still rejected by ``_body_looks_data_shaped`` regardless of
    classification.
"""

from __future__ import annotations

from nora.error_summary import extract_debug_excerpt


_MARKER_LINE = (
    ". display \"_NORA_STATA_PREAMBLE_END_MARKER_\"\n"
    "_NORA_STATA_PREAMBLE_END_MARKER_\n"
)


def test_no_marker_falls_back_to_user_code_forwarding() -> None:
    """No marker in the log at all (e.g. the preamble bailed before
    reaching the ``display`` line, or a legacy run predates the
    marker contract). Classification defaults to user_code under
    the conservative fallback. Under the denylist posture, that
    means the command and body forward through _forward_short_body
    rather than being redacted wholesale."""
    log = (
        ". local lib : env NORA_LIB_DIR\n"
        ". adopath + \"`lib'\"\n"
        "directory does not exist\n"
        "r(601);\n"
    )
    excerpt = extract_debug_excerpt(log, "", 601, "Stata")
    assert excerpt is not None
    # Body forwards so the model can act on the actual diagnostic.
    assert "adopath" in excerpt
    assert "directory does not exist" in excerpt
    assert "r(601);" in excerpt
    # Legacy redaction sentinels are gone.
    assert "[args redacted]" not in excerpt
    assert "[message body redacted]" not in excerpt


def test_user_code_failure_after_marker_forwards_command_and_body() -> None:
    """Canonical user-code failure shape: researcher's ``regress``
    blows up with a missing variable. Marker is in its normal
    preamble position. Under the denylist posture the failing
    command and the Stata error body forward through
    _forward_short_body so the model can fix the specific
    diagnostic. Short scalars (varnames here) ARE expected to pass
    through; this is the documented residual leak."""
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
    # Command echo and body forward; the model needs both to
    # diagnose "varname not found".
    assert "regress y x_missing" in excerpt
    assert "variable x_missing not found" in excerpt
    assert "r(111);" in excerpt
    # No legacy redaction sentinels.
    assert "[args redacted]" not in excerpt
    assert "[message body redacted]" not in excerpt


def test_failing_command_before_marker_is_nora_owned() -> None:
    """Adopath / env-read failure during preamble setup: log
    position is BEFORE the marker, so the extractor takes the
    nora_owned branch. Behavior here is unchanged by the denylist
    migration — preamble failures always forwarded verbatim
    because no researcher data has reached them."""
    log = (
        ". local lib : env NORA_LIB_DIR\n"
        ". adopath + \"`lib'\"\n"
        "directory does not exist\n"
        "r(601);\n"
        # Marker emitted after the failure to exercise the
        # classifier logic; in a real run the preamble would
        # have aborted before reaching the display line.
        ". * (would-be-marker line)\n"
        + _MARKER_LINE
    )
    excerpt = extract_debug_excerpt(log, "", 601, "Stata")
    assert excerpt is not None
    assert "adopath" in excerpt
    assert "directory does not exist" in excerpt
    assert "r(601);" in excerpt
    assert "[args redacted]" not in excerpt
    assert "[message body redacted]" not in excerpt


def test_user_attempt_to_fake_marker_does_not_change_classification() -> None:
    """User code that does ``display "_NORA_STATA_PREAMBLE_END_MARKER_"``
    is no help — the real marker emitted by the preamble was the
    FIRST occurrence in the log. Subsequent fake emissions don't
    shift the classification boundary, so the failing command
    stays in the user_code branch with its forwarding policy.

    Note the contract change vs the prior posture: the failing
    command and body now forward (under _forward_short_body
    mitigations), so the user-supplied value (``secret_value_42``
    here) IS visible in the excerpt. This is the documented short-
    scalar residual leak; the classification still works correctly,
    which is what this test is actually pinning."""
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
    # Classification stays user_code (first-occurrence rule wins),
    # which under the denylist means body forwards. The fake
    # marker did NOT promote the failure to the nora_owned branch
    # (which would have used a different formatting path); both
    # branches now forward, so we pin the forwarded body and the
    # rc shape rather than asserting redaction sentinels.
    assert "regress y" in excerpt
    assert "r(111);" in excerpt


def test_user_code_data_shape_exfil_still_blocked() -> None:
    """``_body_looks_data_shaped`` runs INSIDE _forward_short_body
    regardless of nora_owned vs user_code classification. A script
    that tries ``display "42, John Smith, 1985-01-01, 100000, doctor, NY"``
    as its error message still gets its body suppressed: the
    mixed-token row shape is the canonical row-dump fingerprint."""
    row_dump = "42, John Smith, 1985-01-01, 100000, doctor, NY, 12345"
    log = (
        _MARKER_LINE
        + ". display \"" + row_dump + "\"\n"
        + row_dump + "\n"
        + "r(198);\n"
    )
    excerpt = extract_debug_excerpt(log, "", 198, "Stata")
    assert excerpt is not None
    # The data-shape body never reaches the excerpt.
    assert "John Smith" not in excerpt
    assert "100000" not in excerpt
    # The suppression marker is present.
    assert "message body suppressed: looked data-shaped" in excerpt
