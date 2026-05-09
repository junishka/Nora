"""Schema extractor tests for the formats added alongside Python
support: .parquet, .jsonl/.ndjson, .tsv.

Each test writes a small synthetic dataset to ``tmp_path``, calls
``schema.extract`` at the strictest depth (``names_types_labels_summary``),
and asserts the row/column shape, type taxonomy, NA counts, and
file_type tag come out as expected. Same fixtures double as
data-shape regression tests for the underlying pandas readers.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from nora import schema
from nora.schema import DATA_EXTENSIONS


# ---------------------------------------------------------------------------
# Centralised allowlist
# ---------------------------------------------------------------------------

def test_data_extensions_includes_new_formats():
    """Every format the extractor dispatches on must be in the
    centralised allowlist — the bridge, session_state, and the file
    dialog filter all read from this single tuple."""
    assert ".parquet" in DATA_EXTENSIONS
    assert ".jsonl" in DATA_EXTENSIONS
    assert ".ndjson" in DATA_EXTENSIONS
    assert ".tsv" in DATA_EXTENSIONS
    # Original three still there.
    assert ".csv" in DATA_EXTENSIONS
    assert ".dta" in DATA_EXTENSIONS
    assert ".rds" in DATA_EXTENSIONS


def test_scan_datasets_picks_up_every_supported_extension(tmp_path: Path) -> None:
    """The scanner that feeds the system-prompt's dataset listing AND
    the permission panel must pick up every extension Nora claims to
    support. Regression for the bug where a researcher uploaded a
    parquet file but the model couldn't see it (because the scanner
    was hardcoded to ``.csv/.dta/.rds``) and the permission panel
    rendered empty."""
    from nora.system_prompt import scan_datasets

    # Touch one file per extension so missing parsers don't trip up
    # the scanner — it's a directory walk, not a content read.
    for ext in DATA_EXTENSIONS:
        (tmp_path / f"sample{ext}").write_text("")

    found = {p.suffix.lower() for p in scan_datasets(tmp_path)}
    missing = set(DATA_EXTENSIONS) - found
    assert not missing, (
        f"scan_datasets misses extensions in DATA_EXTENSIONS: {sorted(missing)}"
    )


# ---------------------------------------------------------------------------
# Parquet
# ---------------------------------------------------------------------------

def test_extract_parquet_round_trip(tmp_path: Path) -> None:
    # Use enough rows that the "few distinct strings → categorical"
    # heuristic in _pandas_type actually fires (it requires
    # nunique <= n // 20, so n must be ~40+ for two distinct levels).
    n = 60
    df = pd.DataFrame({
        "subject_id": list(range(1, n + 1)),
        "treatment": ["A" if i % 2 == 0 else "B" for i in range(n)],
        "outcome": [None if i == 0 else float(i) / 10 for i in range(n)],
    })
    path = tmp_path / "trial.parquet"
    df.to_parquet(path)

    out = schema.extract(path, "names_types_labels_summary")

    assert out["status"] == "ok"
    assert out["file_type"] == "parquet"
    assert out["dataset"] == "trial.parquet"
    assert out["observation_count"] == n
    names = [v["name"] for v in out["variables"]]
    assert names == ["subject_id", "treatment", "outcome"]
    by_name = {v["name"]: v for v in out["variables"]}
    assert by_name["subject_id"]["type"] == "integer"
    # Two distinct values across 60 rows comfortably fits the
    # categorical heuristic (nunique <= 20 AND nunique <= n // 20).
    assert by_name["treatment"]["type"] == "categorical"
    assert by_name["outcome"]["type"] == "numeric"
    # na_count of 1 is a re-identification channel (it points at the
    # single missing observation), so the schema summary suppresses
    # rare counts via the same primary-cell-suppression rule used
    # elsewhere. Marker shape mirrors ``nora.sdc.suppression_marker``.
    assert by_name["outcome"]["na_count"] == "<10"


def test_extract_parquet_load_data_returns_dataframe(tmp_path: Path) -> None:
    """data_request and friends call schema.load_data, which has its
    own dispatch — make sure parquet is wired there too."""
    df_in = pd.DataFrame({"x": [10, 20, 30]})
    path = tmp_path / "tiny.parquet"
    df_in.to_parquet(path)

    df_out = schema.load_data(path)
    assert list(df_out["x"]) == [10, 20, 30]


# ---------------------------------------------------------------------------
# JSONL / NDJSON
# ---------------------------------------------------------------------------

def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_extract_jsonl(tmp_path: Path) -> None:
    records = [
        {"id": 1, "score": 0.83, "label": "ok"},
        {"id": 2, "score": 0.91, "label": "ok"},
        {"id": 3, "score": None, "label": "fail"},
    ]
    path = tmp_path / "events.jsonl"
    _write_jsonl(path, records)

    out = schema.extract(path, "names_types_labels_summary")

    assert out["status"] == "ok"
    assert out["file_type"] == "jsonl"
    assert out["observation_count"] == 3
    by_name = {v["name"]: v for v in out["variables"]}
    assert set(by_name) == {"id", "score", "label"}
    assert by_name["id"]["type"] == "integer"
    assert by_name["score"]["type"] == "numeric"
    # Suppressed: only 3 rows total, 1 missing. The rare-count
    # filter fires on either side. See _suppress_rare_count.
    assert by_name["score"]["na_count"] == "<10"


def test_extract_ndjson_alias(tmp_path: Path) -> None:
    """``.ndjson`` is the same format as ``.jsonl``; both must extract
    identically."""
    records = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
    path = tmp_path / "events.ndjson"
    _write_jsonl(path, records)

    out = schema.extract(path, "names_types")
    assert out["status"] == "ok"
    assert out["file_type"] == "jsonl"
    assert out["observation_count"] == 2


# ---------------------------------------------------------------------------
# TSV
# ---------------------------------------------------------------------------

def test_extract_tsv(tmp_path: Path) -> None:
    """Tab-separated — same dtypes as CSV but reads via ``sep='\\t'``."""
    path = tmp_path / "table.tsv"
    path.write_text(
        "id\tname\tweight\n"
        "1\talpha\t12.5\n"
        "2\tbeta\t13.7\n"
        "3\tgamma\t11.2\n",
        encoding="utf-8",
    )

    out = schema.extract(path, "names_types_labels_summary")

    assert out["status"] == "ok"
    assert out["file_type"] == "tsv"
    assert out["observation_count"] == 3
    by_name = {v["name"]: v for v in out["variables"]}
    assert by_name["id"]["type"] == "integer"
    assert by_name["weight"]["type"] == "numeric"


# ---------------------------------------------------------------------------
# Unsupported extension still rejects with a clear error
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Stata value-label caps
# ---------------------------------------------------------------------------

def test_stata_value_labels_capped_per_variable(tmp_path: Path) -> None:
    """A codebook-heavy .dta (industry codes, geographic codes) used
    to forward every value-label entry into the schema response. Cap
    per-variable at 50 entries with a ``value_labels_truncated``
    flag so the model knows the codebook is partial."""
    pyreadstat = pytest.importorskip("pyreadstat")

    # Build a NAICS-shaped value-label set: 1000 entries.
    n = 1000
    df = pd.DataFrame({
        "industry": [i % n for i in range(2 * n)],
        "y": [float(i) for i in range(2 * n)],
    })
    variable_value_labels = {
        "industry": {i: f"industry-{i:04d}" for i in range(n)},
    }
    path = tmp_path / "codebook_heavy.dta"
    pyreadstat.write_dta(
        df, str(path),
        variable_value_labels=variable_value_labels,
    )

    out = schema.extract(path, "names_types_labels")
    assert out["status"] == "ok"
    by_name = {v["name"]: v for v in out["variables"]}
    industry = by_name["industry"]
    assert "value_labels" in industry
    assert len(industry["value_labels"]) <= 50
    assert industry.get("value_labels_truncated") is True
    assert industry.get("value_labels_total") == n


def test_stata_value_labels_total_cap_across_variables(tmp_path: Path) -> None:
    """Beyond the per-variable cap, the total across all variables
    is also bounded at 500 so a file with many medium-sized label
    sets can't spend the whole context."""
    pyreadstat = pytest.importorskip("pyreadstat")

    # 20 variables, each with a 40-entry label set. Per-variable cap
    # (50) wouldn't trip; only the total (500) does.
    n_vars = 20
    n_labels = 40
    cols: dict[str, list] = {}
    variable_value_labels: dict[str, dict] = {}
    for v in range(n_vars):
        col = f"var_{v}"
        cols[col] = [i % n_labels for i in range(80)]
        variable_value_labels[col] = {
            i: f"v{v}-label-{i}" for i in range(n_labels)
        }
    df = pd.DataFrame(cols)
    path = tmp_path / "many_codebooks.dta"
    pyreadstat.write_dta(
        df, str(path),
        variable_value_labels=variable_value_labels,
    )

    out = schema.extract(path, "names_types_labels")
    assert out["status"] == "ok"
    total_emitted = sum(
        len(v.get("value_labels", {})) for v in out["variables"]
    )
    assert total_emitted <= 500
    # At least one variable should be marked truncated since
    # 20 vars × 40 entries = 800 > 500 budget.
    truncated_count = sum(
        1 for v in out["variables"]
        if v.get("value_labels_truncated") is True
    )
    assert truncated_count >= 1


def test_unsupported_format_lists_all_supported(tmp_path: Path) -> None:
    """A researcher who drops a ``.xlsx`` should get a message that
    names what Nora actually accepts — not just a ``KeyError`` from
    pandas. Pin the message text loosely so the dispatch list change
    propagates here without breaking the test."""
    path = tmp_path / "evil.xlsx"
    path.write_text("not really excel", encoding="utf-8")

    with pytest.raises(ValueError) as exc:
        schema.extract(path, "names_types")
    msg = str(exc.value)
    assert ".parquet" in msg
    assert ".jsonl" in msg
    assert ".tsv" in msg
