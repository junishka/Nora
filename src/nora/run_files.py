"""Shared helpers for surfacing Nora-written scripts from run dirs.

The same enumeration ran in three places at growing risk of drift:
the bridge's Files panel (``ui._files_listing``), the model-facing
``list_session_files`` tool, and the resolution path inside
``read_attached_file``. Centralising it here keeps the labeled-name
contract identical across the UI and the tools — what the panel
shows is exactly what the model can ask back for by name.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


_LABEL_FILENAME_MAX = 100

_SCRIPT_EXTS: tuple[str, ...] = (".do", ".R", ".r", ".py", ".ipynb")


def label_to_filename_stem(label: str) -> str:
    """Render a ``submit_script`` label as a Files-panel filename stem.

    Collapses whitespace, drops control / path characters, strips a
    redundant trailing extension, and caps at ``_LABEL_FILENAME_MAX``.
    Returns ``""`` for the ``(unlabeled)`` placeholder, bare
    ``[error]`` / ``[rejected]`` markers, or anything that cleans to
    empty — the caller then falls back to ``script_<short_id>``.

    Diagnostic prefixes carrying real label content (``[error] H2
    main``) are KEPT — the tag is informative and the trailing label
    is the only useful name; stripping both would leave a crashed run
    indistinguishable from any other.
    """
    if not label:
        return ""
    s = label.strip()
    if not s or s == "(unlabeled)":
        return ""
    if s in {"[error]", "[rejected]"}:
        return ""
    if s in {"[error] (unlabeled)", "[rejected] (unlabeled)"}:
        return ""
    cleaned: list[str] = []
    for ch in s:
        if ord(ch) < 0x20 or ch in {"\x7f"}:
            cleaned.append(" ")
        elif ch in {"/", "\\", "\x00"}:
            cleaned.append(" ")
        else:
            cleaned.append(ch)
    s = "".join(cleaned)
    s = " ".join(s.split())
    for ext in _SCRIPT_EXTS:
        if s.lower().endswith(ext.lower()):
            s = s[: -len(ext)].rstrip()
            break
    if not s:
        return ""
    if len(s) > _LABEL_FILENAME_MAX:
        s = s[: _LABEL_FILENAME_MAX - 1].rstrip() + "…"
    return s


def _labels_by_run_basename(cwd: Path) -> dict[str, str]:
    """Build a basename → cleaned-label map for the Files panel.

    Two-pass priority: prefer ``<run_dir>/label.txt`` (the SCRIPT-
    level label the model passed to ``submit_script``), fall back to
    the first per-helper label in ``results.db``. The umbrella label
    is what describes the run as a whole; per-helper labels are
    per-cell and make a poor file name for a multi-result script.

    Falling back to the store also keeps backwards compatibility
    with runs created before label.txt was being written, and with
    runs whose tools layer crashed before the label.txt write
    landed.

    Keys by run-dir basename rather than full path so resolved-vs-
    unresolved cwd symlinks don't silently miss every entry. Hidden
    store rows (rewinds) still contribute — the file on disk hasn't
    moved and the panel is the model's only path back to it after a
    rewind invalidates the chat history.
    """
    out: dict[str, str] = {}

    # Pass 1: walk run dirs for label.txt — the umbrella label.
    runs_root = cwd / ".nora" / "runs"
    if runs_root.is_dir():
        try:
            for run_dir in runs_root.iterdir():
                if not run_dir.is_dir() or run_dir.is_symlink():
                    continue
                label_file = run_dir / "label.txt"
                if not label_file.is_file():
                    continue
                try:
                    raw = label_file.read_text(encoding="utf-8")
                except OSError:
                    continue
                cleaned = label_to_filename_stem(raw)
                if cleaned:
                    out[run_dir.name] = cleaned
        except OSError:
            pass

    # Pass 2: fill in any basename still missing from the store's
    # per-helper labels. Pre-label.txt runs land here, and so do
    # runs where the label.txt write happened to fail.
    try:
        from nora.store import get_store
        for srow in get_store(cwd).list_all(include_hidden=True):
            if not srow.raw_log_path:
                continue
            basename = Path(srow.raw_log_path).name
            if not basename or basename in out:
                continue
            cleaned = label_to_filename_stem(srow.label)
            if cleaned:
                out[basename] = cleaned
    except Exception:  # noqa: BLE001 — store missing/corrupt is fine
        pass
    return out


@dataclass
class RunDirScript:
    """One run-dir script file with its display name and metadata."""
    path: Path           # absolute path to the on-disk script.{do,R,py}
    display_name: str    # surfaced filename (label-derived or fallback)
    short_id: str        # 8-char run id (last segment of run_dir.name)
    mtime: float         # for newest-first sorting + display
    size_bytes: int


def cwd_top_level_display_names(cwd: Path) -> frozenset[str]:
    """Return the ``safe_text``-projected basenames of cwd top-level files.

    These are the names ``read_attached_file`` resolves first; a
    run-dir script that produces the same display name would be
    shadowed (the top-level file wins on lookup, and the listing
    used to silently drop the run-dir script entirely). Used by
    ``enumerate_run_dir_scripts`` and ``find_run_dir_script_by_name``
    to disambiguate run-script names against this set so each row in
    the model-facing listing has a unique, resolvable handle.
    """
    from nora.text_safety import safe_text
    out: set[str] = set()
    try:
        for child in cwd.iterdir():
            try:
                if not child.is_file() or child.is_symlink():
                    continue
            except OSError:
                continue
            cleaned = safe_text(child.name)
            if cleaned:
                out.add(cleaned)
    except OSError:
        pass
    return frozenset(out)


def enumerate_run_dir_scripts(
    cwd: Path, *, max_count: int = 12,
    visible_run_dirs: set[str] | None = None,
    reserved_names: frozenset[str] | None = None,
) -> list[RunDirScript]:
    """Return the ``max_count`` most recently-modified run-dir scripts.

    Each entry carries a display name suitable for surfacing in the
    Files panel and as the lookup key for ``read_attached_file`` —
    label-derived when the model passed one, ``script_<short_id>``
    otherwise. Same-name collisions get the short_id appended in
    parens. Symlinks are skipped both at the run dir and at the
    script file level — only real files in real run dirs participate.

    ``visible_run_dirs``: when supplied, only run dirs whose
    basename is in the set are considered. Used by the model-facing
    ``list_session_files`` / ``read_attached_file`` paths to enforce
    chat-rewind boundaries — a rewind hides results from the store
    but the on-disk run dirs remain, and without filtering the
    model can still discover scripts from a discarded conversation
    branch. ``None`` (the default) leaves every run dir visible
    (researcher-only Files panel).

    ``reserved_names``: when supplied, any run-script whose display
    name (after ``safe_text``) collides with a name in the set has
    the short_id appended to disambiguate. The intended set is the
    cwd top-level filenames — ``read_attached_file`` resolves those
    first, so without this disambiguation a top-level file shadows
    the run-dir script in lookup AND the listing used to drop the
    run-dir row entirely (deduped against top-level names). Pass
    ``cwd_top_level_display_names(cwd)`` from any model-facing call
    site to keep the listing and the lookup consistent.
    """
    runs_root = cwd / ".nora" / "runs"
    if not runs_root.is_dir():
        return []

    labels = _labels_by_run_basename(cwd)
    from nora.text_safety import safe_text

    candidates: list[tuple[float, Path, str, str, int]] = []
    try:
        for run_dir in runs_root.iterdir():
            if not run_dir.is_dir() or run_dir.is_symlink():
                continue
            if (visible_run_dirs is not None
                    and run_dir.name not in visible_run_dirs):
                continue
            for ext in _SCRIPT_EXTS:
                cand = run_dir / f"script{ext}"
                try:
                    if not cand.is_file() or cand.is_symlink():
                        continue
                    st = cand.stat()
                except OSError:
                    continue
                short_id = run_dir.name.rsplit("_", 1)[-1][:8] or "run"
                cleaned = labels.get(run_dir.name, "")
                base = cleaned if cleaned else f"script_{short_id}"
                display = f"{base}{ext}"
                candidates.append(
                    (st.st_mtime, cand, display, short_id, st.st_size),
                )
                break  # one script per run dir
    except OSError:
        pass

    candidates.sort(key=lambda t: -t[0])
    top = candidates[:max_count]

    name_counts: dict[str, int] = {}
    for _, _path, display, _sid, _sz in top:
        name_counts[display] = name_counts.get(display, 0) + 1

    reserved = reserved_names or frozenset()

    out: list[RunDirScript] = []
    for mtime, path, display, short_id, size in top:
        # Two reasons to disambiguate: another run-dir script in this
        # batch produced the same display name (existing behavior), or
        # a cwd top-level file already owns the name (new — without
        # this, ``list_session_files`` dropped the run-script row and
        # ``read_attached_file`` resolved to the top-level file
        # silently, hiding prior runs from the model).
        collides_with_sibling = name_counts.get(display, 0) > 1
        collides_with_top_level = (
            display in reserved or safe_text(display) in reserved
        )
        if collides_with_sibling or collides_with_top_level:
            ext = path.suffix.lower()
            stem = display[: -len(ext)] if ext else display
            display = f"{stem} ({short_id}){ext}"
        out.append(RunDirScript(
            path=path,
            display_name=display,
            short_id=short_id,
            mtime=mtime,
            size_bytes=size,
        ))
    return out


def find_run_dir_script_by_name(
    cwd: Path, name: str,
    *,
    visible_run_dirs: set[str] | None = None,
    reserved_names: frozenset[str] | None = None,
) -> Path | None:
    """Resolve a Files-panel display name back to its on-disk path.

    Used by ``read_attached_file`` when the basename the model passed
    isn't in cwd top-level or any ``_nora_plots/`` dir — those are
    Nora-written scripts living under ``<cwd>/.nora/runs/<id>/``.
    Walks the same enumeration the panel uses so the model can pass
    the same names it sees in ``list_session_files`` output.

    The listing tool surfaces ``safe_text(display_name)``; for labels
    longer than ~120 chars or carrying embedded whitespace, that's
    not the same string as ``entry.display_name``. Match on both the
    raw display name AND its sanitised form so the round-trip works
    regardless of which side a researcher's label hit the cap on.

    ``visible_run_dirs`` mirrors ``enumerate_run_dir_scripts``: pass
    a set to restrict to rewind-visible runs. The model-facing
    ``read_attached_file`` MUST pass the visible set so a rewound
    script can't be re-fetched by the name the model still
    remembers from the discarded chat branch.

    ``reserved_names``: must match the value passed to
    ``enumerate_run_dir_scripts`` at listing time. The disambiguation
    suffix this enumeration applies depends on the reserved set, so
    if the model saw "foo (ab12cd34).do" in the listing, the lookup
    has to enumerate with the same reserved set to reproduce that
    name. Defaults to the cwd top-level names when not supplied —
    the same default any model-facing path should use.
    """
    if not name:
        return None
    from nora.text_safety import safe_text
    if reserved_names is None:
        reserved_names = cwd_top_level_display_names(cwd)
    for entry in enumerate_run_dir_scripts(
        cwd, max_count=64, visible_run_dirs=visible_run_dirs,
        reserved_names=reserved_names,
    ):
        if entry.display_name == name or safe_text(entry.display_name) == name:
            return entry.path
    return None
