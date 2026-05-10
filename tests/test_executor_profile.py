"""Pure unit tests on the generated sandbox profile string.

These assert the invariants the profile MUST satisfy, without needing
``sandbox-exec`` to actually run. That makes them portable across
environments where the outer harness blocks nested sandbox application
(as happened during a recent review, where integration tests
``test_executor_sandbox.py`` failed with ``sandbox_apply: Operation
not permitted``).

The profile string is the whole contract — if a reviewer wants to
tighten or loosen what scripts can reach, the audit surface lives
here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nora.executor import _sandbox_profile_string


@pytest.fixture
def example_profile() -> str:
    """Render the profile for a plausible run to inspect."""
    return _sandbox_profile_string(
        run_dir=Path("/private/var/folders/ab/cdefg/T/nora/run-1234"),
        cwd=Path("/Users/testuser/project"),
        home=Path("/Users/testuser"),
    )


# ---------------------------------------------------------------------------
# Header / posture
# ---------------------------------------------------------------------------

def test_profile_is_deny_default(example_profile: str):
    assert "(deny default)" in example_profile


def test_profile_denies_network(example_profile: str):
    assert "(deny network*)" in example_profile


def test_profile_version_1(example_profile: str):
    assert example_profile.startswith("(version 1)")


# ---------------------------------------------------------------------------
# The /private hole — reviewer feedback 2026-04-20
# ---------------------------------------------------------------------------

def test_no_broad_private_read_subpath(example_profile: str):
    """`(subpath "/private")` would re-open reads on /private/var/log,
    /private/var/backups, /private/var/db/* (other than the narrow
    subtrees we need), and other sensitive paths. It must not appear."""
    assert '(subpath "/private")' not in example_profile


def test_narrow_private_subpaths_present(example_profile: str):
    """The specific /private subtrees R/Stata actually need must be
    allowed — otherwise the interpreter can't resolve user/group IDs
    and load timezone data.

    NB: `/private/etc` is NOT a subpath here — specific config files
    are allowed via `read_literals` instead (see
    `test_narrow_private_etc_literals_present` below). Locking these
    subpaths in prevents accidental over-narrowing that breaks
    R/Stata startup.
    """
    required = [
        '(subpath "/private/var/db/dslocal")',
        '(subpath "/private/var/db/timezone")',
    ]
    for entry in required:
        assert entry in example_profile, f"missing required allow: {entry}"


def test_broad_temp_subpaths_absent(example_profile: str):
    """`/private/tmp` and `/private/var/folders` must NOT be in the
    read or write allowlist as broad subpaths. Those trees hold
    scratch files from every other app the same user is running
    (Slack, Cursor, Chrome cache); granting subpath read there let
    a model-authored script grep cross-app secrets and smuggle
    excerpts through any surviving channel. The executor sets
    TMPDIR=<run_dir>/tmp for the subprocess so R/Stata/Python's
    tempfile module lands inside the run-dir allow without needing
    the broad temp roots.
    """
    forbidden = [
        '(subpath "/private/tmp")',
        '(subpath "/private/var/folders")',
        '(subpath "/tmp")',
    ]
    for entry in forbidden:
        assert entry not in example_profile, f"forbidden broad temp allow: {entry}"


def test_no_broad_private_etc_subpath(example_profile: str):
    """`(subpath "/private/etc")` would re-open reads on /etc/passwd
    (user GECOS + home + shell), /etc/group, /etc/sudoers.d, and
    similar mildly-sensitive config files. Specific files R/Stata
    need are allowed as literals below, not as a whole-subtree
    subpath.
    """
    assert '(subpath "/private/etc")' not in example_profile


def test_narrow_private_etc_literals_present(example_profile: str):
    """The specific /private/etc config files R/Stata probe at startup
    must be allowed as literals. Expand this list if a future version
    of R/Stata needs another file; never widen back to the whole
    /private/etc subpath.
    """
    required_literals = [
        '(literal "/private/etc/hosts")',
        '(literal "/private/etc/localtime")',
        '(literal "/private/etc/resolv.conf")',
        '(literal "/private/etc/protocols")',
        '(literal "/private/etc/services")',
        '(literal "/private/etc/nsswitch.conf")',
    ]
    for entry in required_literals:
        assert entry in example_profile, f"missing required literal: {entry}"


def test_private_etc_passwd_not_readable(example_profile: str):
    """`/etc/passwd` contains user real names (GECOS), home dirs, and
    shells. It must not be in the read allowlist — neither as a subpath
    ancestor nor as an explicit literal. Letting a script read it lets
    Claude exfil those strings through sanitizer-allowed label fields.
    """
    assert '(literal "/private/etc/passwd")' not in example_profile
    assert '(literal "/etc/passwd")' not in example_profile


def test_no_broad_private_var_db(example_profile: str):
    """Only /private/var/db/dslocal and /private/var/db/timezone — not
    the whole /private/var/db subtree (which holds various user-level
    caches and history)."""
    assert '(subpath "/private/var/db")' not in example_profile


def test_no_private_var_log(example_profile: str):
    """System logs can contain sensitive operational data. Not needed
    by R/Stata startup."""
    assert '(subpath "/private/var/log")' not in example_profile


# ---------------------------------------------------------------------------
# /Library narrowing — same principle as /private
# ---------------------------------------------------------------------------

def test_no_broad_library_read_subpath(example_profile: str):
    """`(subpath "/Library")` would re-open reads on
    /Library/Keychains (encrypted but still credential-bearing) and
    /Library/LaunchDaemons / LaunchAgents (system-service config)."""
    assert '(subpath "/Library")' not in example_profile


def test_no_library_keychains_read(example_profile: str):
    assert '(subpath "/Library/Keychains")' not in example_profile


def test_narrow_library_subpaths_present(example_profile: str):
    """R / Stata / Rosetta need these specific /Library subtrees."""
    required = [
        '(subpath "/Library/Apple")',
        '(subpath "/Library/Application Support")',
        '(subpath "/Library/Frameworks")',
    ]
    for entry in required:
        assert entry in example_profile, f"missing required allow: {entry}"


# ---------------------------------------------------------------------------
# HOME — only specific R/Stata subtrees allowed
# ---------------------------------------------------------------------------

def test_home_root_not_read_allowed(example_profile: str):
    """The home dir itself must not be allowed — only specific R/Stata
    subtrees. This is what keeps ~/.ssh, ~/.aws, ~/.gnupg, Keychains,
    ~/Documents out of a script's reach."""
    assert '(subpath "/Users/testuser")' not in example_profile


def test_narrow_home_r_subpath_present(example_profile: str):
    assert '(subpath "/Users/testuser/Library/R")' in example_profile


def test_narrow_home_stata_subpaths_present(example_profile: str):
    assert (
        '(subpath "/Users/testuser/Library/Application Support/Stata")'
        in example_profile
    )
    assert '(subpath "/Users/testuser/ado")' in example_profile


# ---------------------------------------------------------------------------
# cwd + run_dir
# ---------------------------------------------------------------------------

def test_cwd_is_read_allowed(example_profile: str):
    assert '(subpath "/Users/testuser/project")' in example_profile


def test_run_dir_is_read_and_write_allowed(example_profile: str):
    run_entry = '(subpath "/private/var/folders/ab/cdefg/T/nora/run-1234")'
    # Should appear in BOTH the read and write clauses.
    assert example_profile.count(run_entry) >= 2


def test_cwd_is_write_allowed(example_profile: str):
    """The researcher's cwd must be writable so scripts can
    ``save "panel.dta", replace`` / ``saveRDS`` / ``df.to_csv``.
    These are the standard Stata / R / Python output idioms.
    Without this the sandbox returns ``r(603); file ... could not
    be opened`` on every save, and analyses can't persist their
    intermediate panels.

    The data-boundary is preserved by the network deny and the read
    allowlist (no ``/etc/passwd`` reads, no outbound network); a
    write to the user-authorized session cwd is part of the normal
    workflow, not an exfiltration channel.
    """
    cwd_entry = '(subpath "/Users/testuser/project")'
    write_section = example_profile.split("(allow file-write*")[1]
    assert cwd_entry in write_section, (
        "researcher cwd must be writable so save / saveRDS / to_csv "
        "land in the analysis workspace"
    )


# ---------------------------------------------------------------------------
# Writes — very narrow
# ---------------------------------------------------------------------------

def test_no_broad_write_to_system_paths(example_profile: str):
    """Writes outside the run scratch dir and /tmp-style locations
    must not be permitted."""
    forbidden_write_prefixes = [
        '(subpath "/Users/testuser")',
        '(subpath "/Library")',
        '(subpath "/Applications")',
        '(subpath "/System")',
        '(subpath "/usr")',
        '(subpath "/etc")',
        '(subpath "/private/etc")',
        '(subpath "/private/var/db")',
    ]
    # Extract just the file-write clause so we don't false-match against
    # read allows.
    write_section = example_profile.split("(allow file-write*")[1]
    for entry in forbidden_write_prefixes:
        assert entry not in write_section, (
            f"write clause should not include {entry}"
        )


def test_dev_null_writable(example_profile: str):
    assert '(literal "/dev/null")' in example_profile


def test_pty_regex_present(example_profile: str):
    """pseudo-terminals that subprocess plumbing may briefly touch."""
    assert '(regex #"^/dev/ttys[0-9]+$")' in example_profile


# ---------------------------------------------------------------------------
# Paths injected into the profile are SBPL-escaped
# ---------------------------------------------------------------------------

def test_nora_subtree_denied_for_reads(example_profile: str):
    """``<cwd>/.nora`` must NOT be readable by scripts. It holds
    chat_history.jsonl, results.db, prior run logs, and helper
    manifests — exactly the raw/pre-sanitizer material the tool
    layer keeps out of model-visible context. A script that could
    read this directory would smuggle excerpts back through label
    fields, helper error bodies, or any other channel that survives
    sanitization."""
    deny_line = '(deny file-read* (subpath "/Users/testuser/project/.nora"))'
    assert deny_line in example_profile, (
        f"missing deny for .nora reads: profile must include\n  {deny_line}"
    )


def test_nora_subtree_denied_for_writes(example_profile: str):
    """Same carve-out for writes — a script must not modify
    Nora's session state (results.db / chat_history.jsonl) to
    influence future turns by tampering with persisted records."""
    deny_line = '(deny file-write* (subpath "/Users/testuser/project/.nora"))'
    assert deny_line in example_profile, (
        f"missing deny for .nora writes: profile must include\n  {deny_line}"
    )


def test_run_dir_re_allowed_after_nora_deny(example_profile: str):
    """The current ``run_dir`` lives under ``<cwd>/.nora/runs/<id>/``,
    so the .nora deny would block reading the runtime library and
    writing result.json. Re-allow rules for the run_dir must come
    AFTER the deny in profile order — SBPL takes the last matching
    rule, so deny-then-allow gives "allow run_dir, deny everything
    else under .nora"."""
    nora_deny_idx = example_profile.find(
        '(deny file-read* (subpath "/Users/testuser/project/.nora"))'
    )
    run_dir_allow_idx = example_profile.find(
        '(allow file-read* (subpath '
        '"/private/var/folders/ab/cdefg/T/nora/run-1234"))'
    )
    assert nora_deny_idx >= 0 and run_dir_allow_idx >= 0
    assert run_dir_allow_idx > nora_deny_idx, (
        "run_dir re-allow must come AFTER the .nora deny, otherwise "
        "the deny overrides the allow and result.json becomes "
        "unreadable"
    )

    # Same precedence requirement on the write side.
    nora_write_deny_idx = example_profile.find(
        '(deny file-write* (subpath "/Users/testuser/project/.nora"))'
    )
    run_dir_write_allow_idx = example_profile.find(
        '(allow file-write* (subpath '
        '"/private/var/folders/ab/cdefg/T/nora/run-1234"))'
    )
    assert nora_write_deny_idx >= 0 and run_dir_write_allow_idx >= 0
    assert run_dir_write_allow_idx > nora_write_deny_idx


def test_paths_with_special_chars_are_escaped():
    """A cwd containing a double-quote or backslash (extremely unusual
    on macOS, but defensible) must not break out of the SBPL string."""
    profile = _sandbox_profile_string(
        run_dir=Path("/tmp/normal"),
        cwd=Path('/Users/test/weird"dir'),
        home=Path("/Users/test"),
    )
    # The literal " must be backslash-escaped in the SBPL string.
    assert r'\"dir' in profile
    # No unescaped " that would end the SBPL string mid-path.
    # We check by counting balanced quotes on each subpath line.
    for line in profile.splitlines():
        if '"' not in line:
            continue
        # Count unescaped quotes: total quotes minus escaped ones.
        total = line.count('"')
        escaped = line.count('\\"')
        unescaped = total - escaped
        assert unescaped % 2 == 0, f"unbalanced quotes on line: {line!r}"
