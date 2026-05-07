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


def enumerate_run_dir_scripts(
    cwd: Path, *, max_count: int = 12,
) -> list[RunDirScript]:
    """Return the ``max_count`` most recently-modified run-dir scripts.

    Each entry carries a display name suitable for surfacing in the
    Files panel and as the lookup key for ``read_attached_file`` —
    label-derived when the model passed one, ``script_<short_id>``
    otherwise. Same-name collisions get the short_id appended in
    parens. Symlinks are skipped both at the run dir and at the
    script file level — only real files in real run dirs participate.
    """
    runs_root = cwd / ".nora" / "runs"
    if not runs_root.is_dir():
        return []

    labels = _labels_by_run_basename(cwd)

    candidates: list[tuple[float, Path, str, str, int]] = []
    try:
        for run_dir in runs_root.iterdir():
            if not run_dir.is_dir() or run_dir.is_symlink():
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

    out: list[RunDirScript] = []
    for mtime, path, display, short_id, size in top:
        if name_counts.get(display, 0) > 1:
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
) -> Path | None:
    """Resolve a Files-panel display name back to its on-disk path.

    Used by ``read_attached_file`` when the basename the model passed
    isn't in cwd top-level or any ``_nora_plots/`` dir — those are
    Nora-written scripts living under ``<cwd>/.nora/runs/<id>/``.
    Walks the same enumeration the panel uses so the model can pass
    the same names it sees in ``list_session_files`` output.
    """
    if not name:
        return None
    for entry in enumerate_run_dir_scripts(cwd, max_count=64):
        if entry.display_name == name:
            return entry.path
    return None
