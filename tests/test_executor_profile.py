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

from builder.executor import _sandbox_profile_string


@pytest.fixture
def example_profile() -> str:
    """Render the profile for a plausible run to inspect."""
    return _sandbox_profile_string(
        run_dir=Path("/private/var/folders/ab/cdefg/T/builder/run-1234"),
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
    allowed — otherwise the interpreter can't read passwd/hosts,
    resolve user/group IDs, load timezone data, or stage $TMPDIR
    scratch files. Locking these in prevents accidental over-narrowing
    that breaks R/Stata startup."""
    required = [
        '(subpath "/private/etc")',
        '(subpath "/private/tmp")',
        '(subpath "/private/var/db/dslocal")',
        '(subpath "/private/var/db/timezone")',
        '(subpath "/private/var/folders")',
    ]
    for entry in required:
        assert entry in example_profile, f"missing required allow: {entry}"


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
    run_entry = '(subpath "/private/var/folders/ab/cdefg/T/builder/run-1234")'
    # Should appear in BOTH the read and write clauses.
    assert example_profile.count(run_entry) >= 2


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
