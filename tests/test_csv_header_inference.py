"""CSV/TSV header inference must look below the first row.

The original heuristic classified on the first record alone: any text
cell meant "header", an all-numeric record meant "data". That silently
reshaped real datasets in both directions — headerless categorical
data beginning ``control,north`` lost its first observation to the
column names, while numeric year headers over numeric data gained a
phantom observation — and the misread propagated consistently through
``load_data``, schema extraction, and ``row_count`` (the single-
source-of-truth peek keeps the surfaces agreeing, so they all agreed
on the wrong answer).

``_records_look_like_header`` now compares the first record against a
bounded sample of the records below it, column by column. Cases with
no signal (an all-numeric header over numeric data is byte-identical
to a headerless dump) fall back to the old single-record posture and
are surfaced to the model via ``header_note`` in the schema payload
so a wrong guess is visible instead of silent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nora.schema import (
    _csv_has_header,
    _records_look_like_header,
    extract,
    load_data,
    row_count,
)


def _write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


# Each case: (file content, expected has_header, expected data rows).
# The expectations spell out the contract; the docstrings on the
# ambiguous ones say WHY the call goes the way it does.
CASES = [
    pytest.param(
        "control,north\ntreatment,south\ncontrol,south\ntreatment,north\n",
        False, 4,
        id="headerless-categorical-values-recur",
        # First-row values reappear below — header names don't recur
        # as data values, so this is data. The old any-text-is-a-
        # header rule ate the first observation and named the columns
        # ``control`` / ``north``.
    ),
    pytest.param(
        "arm,region\ncontrol,north\ntreatment,south\n",
        True, 2,
        id="text-header-over-categorical",
        # Names never recur in the body — no data signal, so the
        # text-first-row-is-a-header posture holds.
    ),
    pytest.param(
        "id,score\n1,9.5\n2,8.1\n",
        True, 2,
        id="text-header-over-numeric",
    ),
    pytest.param(
        "1,2,3\n4,5,6\n",
        False, 2,
        id="headerless-numeric-dump",
        # The documented raw-dump posture: all-numeric first row over
        # numeric data stays data.
    ),
    pytest.param(
        "2020,2021\n1.5,2.3\n4.1,0.9\n",
        False, 3,
        id="numeric-header-over-numeric-UNDECIDABLE",
        # Genuinely indistinguishable from a headerless dump by
        # content alone — every mainstream reader guesses here too
        # (pandas guesses header, fread and this heuristic guess
        # data). The guess is pinned AND surfaced via header_note so
        # the model can flag it instead of silently analysing a
        # dataset with a phantom observation.
    ),
    pytest.param(
        "2020,2021\nFrance,Spain\nItaly,Greece\n",
        True, 2,
        id="numeric-header-over-text",
        # A numeric name over a text column IS decidable — years over
        # country names. The old rule called row 1 data.
    ),
    pytest.param(
        "country,2020,2021\nFrance,1.5,2.3\nSpain,4.1,0.9\n",
        True, 2,
        id="wide-panel-mixed-header",
        # The common wide-format shape. The numeric-over-numeric year
        # columns abstain; the tie falls back to the text rule, which
        # reads ``country`` as a name.
    ),
    pytest.param(
        "control,1.5\ntreatment,2.3\ncontrol,0.9\n",
        False, 3,
        id="headerless-mixed-with-recurring-categorical",
        # Realistic headerless export: a recurring categorical next
        # to measurements. The old rule saw text and ate row 1.
    ),
    pytest.param(
        ",name,score\n0,ann,9.5\n1,bob,8.1\n",
        True, 2,
        id="pandas-index-column-empty-header-cell",
        # pandas writes an empty header cell for the index column —
        # the empty cell must abstain, not vote.
    ),
    pytest.param(
        "a,b\n",
        True, 0,
        id="single-record-text-fallback",
    ),
    pytest.param(
        "1,2\n",
        False, 1,
        id="single-record-numeric-fallback",
    ),
]


@pytest.mark.parametrize("content, want_header, want_rows", CASES)
def test_inference_and_row_count(
    tmp_path: Path, content: str, want_header: bool, want_rows: int,
) -> None:
    p = _write(tmp_path, "data.csv", content)
    assert _csv_has_header(p, ",") is want_header
    assert row_count(p) == want_rows


@pytest.mark.parametrize("content, want_header, want_rows", CASES)
def test_all_surfaces_agree(
    tmp_path: Path, content: str, want_header: bool, want_rows: int,
) -> None:
    """load_data, the names_only fast path, the full extract, and
    row_count must all see the same shape — a disagreement means one
    surface reports variables the other counts as an observation."""
    p = _write(tmp_path, "data.csv", content)
    df = load_data(p)
    assert len(df) == want_rows
    names_only = extract(p, "names_only")
    assert names_only["observation_count"] == want_rows
    assert [v["name"] for v in names_only["variables"]] == [
        str(c) for c in df.columns
    ]
    full = extract(p, "names_types")
    assert full["observation_count"] == want_rows


def test_tsv_uses_the_same_inference(tmp_path: Path) -> None:
    p = _write(
        tmp_path, "data.tsv",
        "control\tnorth\ntreatment\tsouth\ncontrol\tsouth\n",
    )
    assert _csv_has_header(p, "\t") is False
    assert row_count(p) == 3
    payload = extract(p, "names_only")
    assert payload["observation_count"] == 3


def test_header_note_travels_with_the_schema(tmp_path: Path) -> None:
    """Both inference outcomes must be named in the payload the model
    reads, so the undecidable cases fail loud instead of silent."""
    headered = _write(tmp_path, "h.csv", "id,score\n1,9.5\n")
    headerless = _write(tmp_path, "d.csv", "2020,2021\n1.5,2.3\n")
    for depth in ("names_only", "names_types"):
        note_h = extract(headered, depth)["header_note"]
        assert "column names" in note_h
        note_d = extract(headerless, depth)["header_note"]
        assert "No header row inferred" in note_d
        assert "0..N-1" in note_d


def test_empty_and_unreadable_files_default_to_header(
    tmp_path: Path,
) -> None:
    """Posture preserved from the single-record version: no record
    (or no file) defaults to "has header" so the downstream pandas
    read uses its default rather than an unexpected option."""
    empty = _write(tmp_path, "empty.csv", "")
    assert _csv_has_header(empty, ",") is True
    assert _csv_has_header(tmp_path / "missing.csv", ",") is True


def test_body_aware_inference_direct() -> None:
    """Unit-level checks on the voting function itself, including the
    majority rule across mixed signals."""
    # One header vote (text over numeric) beats one abstain.
    assert _records_look_like_header(
        ["id", "note"], [["1", "x"], ["2", "y"]],
    ) is True
    # Recurring text beats a no-signal column.
    assert _records_look_like_header(
        ["control", "9.9"], [["treatment", "1.5"], ["control", "2.0"]],
    ) is False
    # No body rows falls back to the single-record rule.
    assert _records_look_like_header(["a", "b"], []) is True
    assert _records_look_like_header(["1", "2"], []) is False
