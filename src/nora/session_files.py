"""Shared file-kind taxonomy used by the model-facing
``list_session_files`` tool and the researcher-facing Files panel +
chat-bubble renderers.

Three call sites historically each defined their own ext→kind map.
They agreed on script/log/graph but disagreed on whether to include
datasets (the model-facing path excludes them — datasets have their
own discovery surface gated by SDC) and on the unknown-extension
default. ``classify_ext`` parameterises both decisions so the same
constants drive every call site.
"""

from __future__ import annotations

from nora.schema import DATA_EXTENSIONS


# Source scripts the runtime can re-execute end-to-end. ``.ipynb``
# (Jupyter) is included because the executor accepts it; whether to
# inline its contents into a user message is a separate decision (see
# ``ui._INLINE_SCRIPT_EXTS``).
SCRIPT_EXTS: frozenset[str] = frozenset({
    ".py", ".do", ".r", ".rmd", ".ipynb",
})

# Stata's native graph format + raster/vector image outputs. PDF and
# EPS land here because researchers iterate on plots in those formats
# even though they're more general — within Nora's scope, anything
# matching this extension set IS a plot.
GRAPH_EXTS: frozenset[str] = frozenset({
    ".gph", ".png", ".jpg", ".jpeg", ".pdf", ".eps",
})

# Stata's text logs (.log) and SMCL logs (.smcl).
LOG_EXTS: frozenset[str] = frozenset({
    ".log", ".smcl",
})


def classify_ext(
    ext: str,
    *,
    include_data: bool = False,
    default: str | None = None,
) -> str | None:
    """Map a file extension to one of ``{"script", "graph", "log",
    "data"}`` or ``default``.

    Parameters
    ----------
    ext:
        File extension including the leading dot, lowercase. Empty
        strings and unknown extensions return ``default``.
    include_data:
        When True (Files panel, chat rendering), extensions in
        ``DATA_EXTENSIONS`` map to ``"data"``. When False
        (model-facing ``list_session_files`` tool), data extensions
        return ``default`` so they don't surface through the model's
        discovery path — datasets are listed in the system prompt's
        cwd enumeration and gated by the SDC schema-depth policy
        instead.
    default:
        Returned for unrecognised extensions (and for data exts when
        ``include_data`` is False). The model-facing path passes
        ``None`` to filter; the chat-bubble path passes ``"data"`` to
        treat every unknown extension as a generic data row.
    """
    if ext in SCRIPT_EXTS:
        return "script"
    if ext in GRAPH_EXTS:
        return "graph"
    if ext in LOG_EXTS:
        return "log"
    if include_data and ext in DATA_EXTENSIONS:
        return "data"
    return default


# All non-data kind labels, useful for argument validation in the
# model-facing tool (which rejects 'data' as a kind filter).
NON_DATA_KINDS: frozenset[str] = frozenset({"script", "graph", "log"})


# ---------------------------------------------------------------------------
# Filesystem enumeration
# ---------------------------------------------------------------------------
#
# The Files-panel bridge method previously did extension taxonomy,
# filesystem scan, dedup, run-dir traversal, and thumbnail/base64
# encoding all in one body. ``enumerate_session_files`` covers the
# first four; the bridge layer adds thumbnail enrichment afterward
# (since base64 + PDF rasterisation are UI-only concerns and shouldn't
# be in a module that will eventually be importable from the
# model-facing tool too).

from pathlib import Path
from typing import Any

# Files-panel kind sort order: data first (the analysis subjects),
# then scripts, then graphs, then logs. Used by the panel renderer
# and the enumerate helper below; the model-facing tool ignores it.
KIND_PRIORITY: dict[str, int] = {
    "data": 0, "script": 1, "graph": 2, "log": 3,
}


def enumerate_session_files(
    cwd: Path,
    *,
    include_data: bool = True,
    include_run_scripts: bool = True,
) -> list[dict[str, Any]]:
    """Walk a session cwd and return one dict per known file.

    Two roots are scanned (both non-recursively, so we don't descend
    into deep subtrees):

      1. ``cwd`` itself — researcher uploads, Stata's ``graph export``
         writes, direct ``ggsave`` / ``plt.savefig`` with bare names.
      2. Each ``cwd/.nora/runs/<id>/_nora_plots/`` — helper-produced
         plots; listing them here lets the Files panel act as a
         session-wide gallery regardless of which run wrote each plot.

    When ``include_run_scripts`` is True, also surfaces the script
    Nora wrote on each ``submit_script`` (lives at
    ``<run_dir>/script.{do,R,py,ipynb}``) under its analytic label.

    Each entry carries: ``name`` (display name; rewritten to the
    analytic label for run-dir scripts), ``kind`` (one of script /
    graph / log, plus data when ``include_data`` is True), ``priority``
    (kind sort order), ``size`` (bytes), ``ext`` (lowercase, with
    leading dot), ``mtime`` (POSIX float), ``path`` (string of the
    actual on-disk path). Thumbnail bytes (``data`` / ``mime``) are
    NOT included — the bridge layer adds those for the Files panel.

    Generated PDF→PNG sidecars (filenames ending in ``.nora.png``)
    are filtered out so they don't appear alongside the originals.

    Returns rows sorted by (priority asc, mtime desc, name asc).
    Caller can re-sort if a different order is needed.
    """
    if cwd is None or not cwd.is_dir():
        return []

    rows: list[dict[str, Any]] = []
    seen_paths: set[Path] = set()

    def _add(child: Path) -> None:
        if child.name.endswith(".nora.png"):
            return
        ext = child.suffix.lower()
        kind = classify_ext(ext, include_data=include_data)
        if kind is None:
            return
        priority = KIND_PRIORITY[kind]
        try:
            stat = child.stat()
        except OSError:
            return
        try:
            resolved = child.resolve()
        except OSError:
            return
        if resolved in seen_paths:
            return
        seen_paths.add(resolved)
        rows.append({
            "name": child.name,
            "kind": kind,
            "priority": priority,
            "size": stat.st_size,
            "ext": ext,
            "mtime": stat.st_mtime,
            "path": str(child),
        })

    try:
        for child in cwd.iterdir():
            if child.is_file() and not child.is_symlink():
                _add(child)
    except OSError:
        return []

    runs_root = cwd / ".nora" / "runs"
    if runs_root.is_dir():
        try:
            for run_dir in runs_root.iterdir():
                plots_dir = run_dir / "_nora_plots"
                if plots_dir.is_dir():
                    try:
                        for plot in plots_dir.iterdir():
                            if plot.is_file() and not plot.is_symlink():
                                _add(plot)
                    except OSError:
                        pass
        except OSError:
            pass

        if include_run_scripts:
            from nora.run_files import enumerate_run_dir_scripts
            for entry in enumerate_run_dir_scripts(cwd):
                ext = entry.path.suffix.lower()
                kind = classify_ext(ext, include_data=include_data) or "script"
                priority = KIND_PRIORITY.get(kind, KIND_PRIORITY["script"])
                try:
                    resolved = entry.path.resolve()
                except OSError:
                    continue
                if resolved in seen_paths:
                    continue
                seen_paths.add(resolved)
                rows.append({
                    "name": entry.display_name,
                    "kind": kind,
                    "priority": priority,
                    "size": entry.size_bytes,
                    "ext": ext,
                    "mtime": entry.mtime,
                    "path": str(entry.path),
                })

    rows.sort(
        key=lambda r: (r["priority"], -r.get("mtime", 0), r["name"].lower())
    )
    return rows
