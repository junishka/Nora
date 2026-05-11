"""Nora — schema extractor.

Produces a structural summary of a dataset (variable names, types, labels,
observation count, optionally NA counts / distinct counts) from files on
the researcher's machine. Supported formats: `.dta` (Stata), `.rds` (R),
`.csv`, `.tsv`, `.parquet`, `.jsonl` / `.ndjson`.

No individual observation values are ever returned. The only potentially
disclosive pieces of output are:
- Variable labels (short human-readable strings; may contain sensitive
  descriptions like "Primary diagnosis" — exposed at depth >=
  names_types_labels).
- Value labels, i.e. the level-name dictionary for a categorical variable
  (e.g. `{1: "Control", 2: "Treatment"}` — exposed at depth >=
  names_types_labels, same tier as variable labels).

Neither is a per-observation value; both are metadata attached to the
variable. The researcher can dial depth down if their column labels or
value labels are themselves sensitive. Step 5 (real sanitizer with SDC
rules) will add per-variable controls.

Depths (graded from conservative → permissive):
  names_only
  names_types
  names_types_labels
  names_types_labels_summary     (+ NA count, distinct count for categoricals)

The ``na_count`` field at the summary depth is subject to primary
cell suppression (see ``_suppress_rare_count``): a count below the
threshold on the rarer side is replaced with a ``<N`` marker, same
shape as :func:`nora.sdc.suppression_marker`. Without this, a
column with exactly one missing value would re-identify that
observation through ``get_schema`` before the stricter
``request_data`` / result-sanitizer paths ever ran.

Not included, ever, at any depth:
- Actual observation values
- Min / max / mean / quantiles on numerics (these are individual values)
- Frequency tables (step 5 territory — requires SDC cell-suppression)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from nora.text_safety import safe_key, safe_text


Depth = Literal[
    "names_only",
    "names_types",
    "names_types_labels",
    "names_types_labels_summary",
]

_VALID_DEPTHS: frozenset[str] = frozenset(
    ("names_only", "names_types", "names_types_labels", "names_types_labels_summary")
)


# Centralised data-file extension allowlist. Imported by every module
# that scans a session dir for "the researcher's datasets" (the bridge,
# session_state, drop-zone hint copy, file dialog filters). Adding a
# new format means: (a) add the extension here, (b) add a dispatch
# branch in ``extract()``/``load_data()`` below, (c) make sure any
# parsing dependency is in pyproject.toml.
#
# ``.jsonl`` and ``.ndjson`` are aliases — pandas reads both with
# ``read_json(lines=True)`` and researchers ship under either name.
DATA_EXTENSIONS: tuple[str, ...] = (
    ".csv",
    ".dta",
    ".rds",
    ".parquet",
    ".jsonl",
    ".ndjson",
    ".tsv",
)

# Our coarse taxonomy. Step 4+ analysis-result schemas consume these; keep
# the vocabulary small and stable.
_TYPE_NUMERIC = "numeric"
_TYPE_INTEGER = "integer"
_TYPE_CATEGORICAL = "categorical"
_TYPE_STRING = "string"
_TYPE_BOOLEAN = "boolean"
_TYPE_DATETIME = "datetime"
_TYPE_UNKNOWN = "unknown"


# Primary cell-suppression threshold for schema summary metadata. The
# value here mirrors :class:`nora.sanitizer.SDCConfig.cell_suppression_threshold`
# (10) but is kept inline so this module doesn't take a runtime
# dependency on the regression sanitizer. Schema summary publishes
# ``na_count`` per variable at the richest depth tier; without
# suppression a column with exactly one missing value (or one present
# value, in a mostly-empty column) would re-identify that observation
# directly — the same disclosive concern that primary cell
# suppression solves for frequency tables. ``request_data`` /
# regression-result paths apply their own SDC, but ``get_schema`` is
# the first surface the model can call against a dataset and runs
# before either of those, so the suppression has to live here too.
_SCHEMA_SUMMARY_THRESHOLD = 10


def _suppress_rare_count(value: int, n: int, threshold: int) -> int | str:
    """Return ``value`` unchanged when it sits comfortably above the
    threshold on both sides; otherwise return the suppression marker.

    "Both sides" is the symmetric edge case: a column with
    ``na_count == 1`` identifies the one missing observation; a
    column with ``n - na_count == 1`` identifies the one present
    observation. Either is a re-identification channel, so suppress
    when the rarer side falls below ``threshold``. ``value == 0`` and
    ``value == n`` are both safe (no rare subgroup) and pass
    through.
    """
    if value < 0 or n < 0 or threshold <= 0:
        return value
    rarer = min(value, n - value) if n >= value else value
    if rarer == 0:
        return value
    if rarer < threshold:
        # ``<10``-style marker, same shape as
        # :func:`nora.sdc.suppression_marker`.
        return f"<{threshold}"
    return value


def load_data(dataset_path: Path) -> Any:
    """Load the dataset at `dataset_path` as a pandas DataFrame.

    Dispatches by extension (.dta / .rds / .csv). Used by
    ``extract()`` at the summary depth, and by ``data_request`` for
    computing bounded sanitized facts about a variable.

    Returns the full data as a DataFrame. Callers are responsible for
    never returning raw values — in Nora, only ``data_request``
    and schema extraction load data, and both sanitize before emitting.
    """
    import pandas as pd

    suffix = dataset_path.suffix.lower()
    if suffix == ".dta":
        import pyreadstat
        df, _meta = pyreadstat.read_dta(str(dataset_path))
        return df
    if suffix == ".rds":
        import pyreadr
        result = pyreadr.read_r(str(dataset_path))
        if not result:
            raise SchemaExtractError(f".rds file contains no objects: {dataset_path}")
        obj = next(iter(result.values()))
        if not hasattr(obj, "columns"):
            raise SchemaExtractError(
                f".rds at {dataset_path} does not contain a data frame; "
                f"got {type(obj).__name__}"
            )
        return obj
    if suffix == ".csv":
        return pd.read_csv(dataset_path, low_memory=False)
    if suffix == ".tsv":
        return pd.read_csv(dataset_path, sep="\t", low_memory=False)
    if suffix == ".parquet":
        # pandas dispatches to pyarrow (preferred) or fastparquet —
        # pyarrow is a declared dep of nora so this works out of
        # the box. Parquet preserves dtypes so column types come
        # back exactly as the writer set them.
        return pd.read_parquet(dataset_path)
    if suffix in (".jsonl", ".ndjson"):
        # Line-delimited JSON: one record per line. Top-level JSON
        # arrays of arbitrary shape are NOT supported (each row
        # would have to be a flat object for the schema extractor
        # to make sense of it); researchers can convert with `jq`
        # if needed.
        return pd.read_json(dataset_path, lines=True)
    raise SchemaExtractError(
        f"unsupported format: {suffix!r}. Nora reads "
        ".dta, .rds, .csv, .tsv, .parquet, .jsonl, .ndjson."
    )


def _record_looks_like_header(record: list[str]) -> bool:
    """Heuristic: does a parsed CSV/TSV record look like a header row
    (column names) or a data row?

    Takes an already-parsed list of cells from ``csv.reader`` (which
    correctly handles quoted, multi-line fields) rather than raw
    bytes. True (header) when at least one cell can't be parsed as a
    number. Same edge-case posture as the previous bytes-based
    version:

    - Empty or single-cell record: treated as a header (a single-
      column file with a name like ``"id"`` is the common case).
    - Every cell is numeric (raw sensor dump etc.): treated as data,
      no header offset applied.
    """
    if not record:
        return True
    saw_any = False
    for cell in record:
        s = (cell or "").strip()
        if not s:
            continue
        saw_any = True
        try:
            float(s)
        except ValueError:
            return True
    if not saw_any:
        return True
    return False


def row_count(dataset_path: Path) -> int | None:
    """Return the row count for the dataset at ``dataset_path`` without
    materialising the values where the format allows it.

    Used by ``submit_script``'s row-count audit (one comparison per
    multi-result call against the source dataset's N). The audit doesn't
    need any column data — just the row count — so we deliberately
    avoid the full ``load_data`` path which can take 60+ seconds on a
    multi-GB .dta.

    Per format:
    - ``.dta``: ``pyreadstat.read_dta(metadataonly=True).meta.number_rows``.
      0.5s on a 3 GB file vs ~60s for a full load.
    - ``.parquet``: ``pyarrow.parquet.ParquetFile(...).metadata.num_rows``.
      Reads only the footer.
    - ``.csv`` / ``.tsv``: byte-streamed line count, minus 1 if the
      first line looks like a header (any non-numeric token in the
      first row). The previous unconditional ``-1`` was wrong for
      headerless dumps (raw instrument files, anonymous panel data,
      log files renamed to .csv) — it produced an audit count off
      by one and the row-count check then false-flagged scripts
      that correctly counted the headerless row. Heuristic-only;
      a CSV whose header happens to be numeric (rare but possible
      — column names like ``"123"``) still gets misclassified, but
      the audit treats ``None`` and "off by 1" the same way (best-
      effort flag, not a gate).
    - ``.jsonl`` / ``.ndjson``: byte-streamed line count.
    - ``.rds``: no light path available without spinning up R; falls
      back to ``load_data`` and counts.

    Returns ``None`` on any failure — the audit is a best-effort signal,
    not a gate, so callers should treat ``None`` as "skip the check"
    rather than raise.
    """
    suffix = dataset_path.suffix.lower()
    try:
        if suffix == ".dta":
            import pyreadstat
            _df, meta = pyreadstat.read_dta(
                str(dataset_path), metadataonly=True,
            )
            return int(meta.number_rows)
        if suffix == ".parquet":
            import pyarrow.parquet as pq
            return int(pq.ParquetFile(str(dataset_path)).metadata.num_rows)
        if suffix == ".csv" or suffix == ".tsv":
            # Streamed parse via ``csv.reader`` so quoted fields with
            # embedded newlines count as ONE record, not multiple.
            # The previous byte-streamed line counter treated every
            # physical ``\n`` as a row boundary, so a valid CSV like
            #   id,note\n
            #   1,"hello\n
            #   world"\n
            # was counted as 3 lines (- 1 for header) = 2 rows even
            # though the actual analysis sees 1 row, false-flagging
            # the row-count audit. ``csv.reader`` honours RFC 4180
            # quoting and is memory-efficient (it streams the file
            # without materialising the whole thing). The header
            # heuristic operates on the parsed first record so it
            # works through quotes correctly.
            import csv
            delimiter = "," if suffix == ".csv" else "\t"
            n_records = 0
            first_record: list[str] | None = None
            with open(
                dataset_path, "r", encoding="utf-8",
                errors="replace", newline="",
            ) as f:
                reader = csv.reader(f, delimiter=delimiter)
                for record in reader:
                    if first_record is None:
                        first_record = record
                    n_records += 1
            if n_records == 0:
                return 0
            has_header = _record_looks_like_header(first_record or [])
            return max(0, n_records - 1 if has_header else n_records)
        if suffix in (".jsonl", ".ndjson"):
            n_lines = 0
            with open(dataset_path, "rb") as f:
                for line in f:
                    if line.strip():
                        n_lines += 1
            return n_lines
        if suffix == ".rds":
            # No metadata-only path; fall back to full load.
            return int(len(load_data(dataset_path)))
    except Exception:  # noqa: BLE001 — audit is best-effort
        return None
    return None


def extract(dataset_path: Path, depth: str) -> dict[str, Any]:
    """Return a structured schema summary for the file at `dataset_path`.

    Dispatches by suffix. Raises ValueError for unsupported formats or
    invalid depth; lets underlying library errors propagate (tool layer
    catches them and returns a policy-shaped error payload).
    """
    if depth not in _VALID_DEPTHS:
        raise SchemaExtractError(
            f"invalid depth: {depth!r}; valid: {sorted(_VALID_DEPTHS)}"
        )
    suffix = dataset_path.suffix.lower()
    if suffix == ".dta":
        return _extract_stata(dataset_path, depth)
    if suffix == ".rds":
        return _extract_rds(dataset_path, depth)
    if suffix == ".csv":
        return _extract_csv(dataset_path, depth)
    if suffix == ".tsv":
        return _extract_tsv(dataset_path, depth)
    if suffix == ".parquet":
        return _extract_parquet(dataset_path, depth)
    if suffix in (".jsonl", ".ndjson"):
        return _extract_jsonl(dataset_path, depth)
    raise SchemaExtractError(
        f"unsupported format: {suffix!r}. Nora currently reads "
        ".dta (Stata), .rds (R), .csv, .tsv, .parquet, .jsonl, .ndjson. "
        "Other formats (.rda, .xlsx, .sav, .feather) are not supported yet."
    )


# ---------------------------------------------------------------------------
# Stata — .dta via pyreadstat
# ---------------------------------------------------------------------------

# Per-variable cap on emitted value-label entries. A codebook-heavy
# .dta (e.g. an industry classification with thousands of NAICS codes)
# would otherwise pour every label into the schema response, blowing
# context and creating a large data-origin text channel even after
# per-string sanitization. We surface the count and a hint so the
# model can request the full codebook through a different path if
# it actually needs it.
_MAX_VALUE_LABELS_PER_VAR = 50
# Total cap across all variables in one schema response, so a file
# with many medium-sized label sets can't blow the budget either.
_MAX_VALUE_LABELS_TOTAL = 500


class SchemaExtractError(ValueError):
    """Raised by ``schema.extract`` for parser-OWNED validation errors.

    These messages are crafted by this module (unsupported format,
    invalid depth, .rds-without-dataframe, etc.) and are safe to
    forward verbatim — they do not embed row content or library
    diagnostics. The tool layer relies on the class to distinguish
    them from data-leak-prone pandas / pyreadstat / pyreadr
    exceptions (some of which are also ``ValueError`` subclasses,
    notably ``pandas.errors.ParserError`` whose message quotes the
    offending CSV row).
    """


def _extract_stata(path: Path, depth: str) -> dict[str, Any]:
    import pyreadstat

    wants_summary = depth == "names_types_labels_summary"
    # metadataonly is fast and avoids loading any values. We only load the
    # DataFrame when we need summary stats.
    if wants_summary:
        df, meta = pyreadstat.read_dta(str(path))
    else:
        df, meta = pyreadstat.read_dta(str(path), metadataonly=True)

    variables: list[dict[str, Any]] = []
    labels_emitted_total = 0
    for idx, name in enumerate(meta.column_names):
        # Variable names originate in the data file and are forwarded to
        # Claude — pass through the injection defense.
        safe_name = safe_key(str(name))
        var: dict[str, Any] = {"name": safe_name}

        if depth != "names_only":
            var["type"] = _stata_type(meta, name)

        if depth in ("names_types_labels", "names_types_labels_summary"):
            col_labels = meta.column_labels or []
            if idx < len(col_labels):
                label = col_labels[idx]
                if label:
                    # Variable labels are free-text — the longest injection
                    # surface in a typical .dta. Sanitize aggressively.
                    var["label"] = safe_text(str(label))
            # Value labels, if this column is tied to a label set. Both
            # the codes (keys) and labels (values) originate in the data,
            # so we sanitize each entry AND cap the count: a codebook-
            # heavy file (industry classifications, geographic codes)
            # could otherwise emit thousands of labels per variable
            # and tens of thousands across the file, spending the
            # context window and providing a wide data-origin text
            # channel.
            label_set = meta.variable_to_label.get(name)
            if label_set and label_set in meta.value_labels:
                raw = meta.value_labels[label_set]
                total_in_set = len(raw)
                budget_remaining = max(
                    0, _MAX_VALUE_LABELS_TOTAL - labels_emitted_total
                )
                effective_cap = min(
                    _MAX_VALUE_LABELS_PER_VAR, budget_remaining
                )
                # Stable insertion order from pyreadstat; take the
                # first ``effective_cap`` entries so repeated calls
                # against the same file return the same view.
                items = list(raw.items())[:effective_cap]
                var["value_labels"] = {
                    safe_key(str(k)): safe_text(str(v))
                    for k, v in items
                }
                labels_emitted_total += len(items)
                if total_in_set > len(items):
                    var["value_labels_total"] = total_in_set
                    var["value_labels_truncated"] = True

        # Use the ORIGINAL name as the key when reading df (to match pandas),
        # but report the SANITIZED name to Claude.
        if wants_summary and df is not None and name in df.columns:
            series = df[name]
            n_obs = int(len(series))
            raw_na = int(series.isna().sum())
            var["na_count"] = _suppress_rare_count(
                raw_na, n_obs, _SCHEMA_SUMMARY_THRESHOLD,
            )
            if var.get("type") == _TYPE_CATEGORICAL:
                var["distinct_count"] = int(series.nunique(dropna=True))

        variables.append(var)

    return {
        "status": "ok",
        # Filename crosses to Claude as text, so it's a prompt-injection
        # surface: a file named with embedded newlines or fake system
        # markers would land in the model's context verbatim. safe_text
        # strips control chars, flattens whitespace, and caps length —
        # same chokepoint we apply to variable labels above. "filename
        # only — no path injection surface" was the old comment; it
        # covered path-traversal but NOT prompt injection.
        "dataset": safe_text(path.name),
        "file_type": "stata",
        "depth": depth,
        "observation_count": int(meta.number_rows),
        "variables": variables,
    }


def _stata_type(meta: Any, name: str) -> str:
    """Map pyreadstat's Stata metadata to our coarse type taxonomy."""
    # Value labels → treat as categorical (regardless of underlying numeric
    # encoding — that's how Stata users think about them).
    if meta.variable_to_label.get(name):
        return _TYPE_CATEGORICAL
    typ = (meta.readstat_variable_types or {}).get(name, "").lower()
    if typ == "string":
        return _TYPE_STRING
    if typ in ("int8", "int16", "int32"):
        return _TYPE_INTEGER
    if typ in ("float", "double"):
        return _TYPE_NUMERIC
    return _TYPE_UNKNOWN


# ---------------------------------------------------------------------------
# R — .rds via pyreadr
# ---------------------------------------------------------------------------

def _extract_rds(path: Path, depth: str) -> dict[str, Any]:
    import pyreadr

    result = pyreadr.read_r(str(path))
    # .rds holds a single object. pyreadr returns an OrderedDict keyed
    # either by the saved name or by `None` if nameless. Take the first.
    if not result:
        raise SchemaExtractError(f".rds file contains no objects: {path}")
    obj_key = next(iter(result))
    df = result[obj_key]
    # pyreadr can return non-DataFrame objects (lists, etc.). We only
    # handle data frames in v0.
    if not hasattr(df, "columns"):
        raise SchemaExtractError(
            f".rds file at {path} does not contain a data frame; got "
            f"{type(df).__name__}. Convert to a data frame in R and "
            f"re-save with saveRDS()."
        )

    return _extract_from_pandas(
        df,
        depth=depth,
        dataset_name=path.name,
        file_type="r_rds",
    )


# ---------------------------------------------------------------------------
# names_only fast path — column names only, no full data load
# ---------------------------------------------------------------------------

def _names_only_payload(
    column_names: list[str], path: Path, file_type: str,
) -> dict[str, Any]:
    """Build a names_only response from a list of column names and the
    path. Used by the CSV/TSV/Parquet/JSONL fast paths to avoid
    loading the whole dataset when the model only asked for the
    variable list.

    Observation count comes from ``row_count(path)`` (metadata- or
    streaming-only for these formats). A ``None`` return falls back
    to ``0`` so the response stays well-formed; the caller can decide
    to escalate to the full-load path if they need a real count.
    """
    obs = row_count(path) or 0
    variables = [{"name": safe_key(str(c))} for c in column_names]
    return {
        "status": "ok",
        # See _extract_stata for the prompt-injection rationale on
        # filenames passing through safe_text.
        "dataset": safe_text(path.name),
        "file_type": file_type,
        "depth": "names_only",
        "observation_count": int(obs),
        "variables": variables,
    }


# ---------------------------------------------------------------------------
# CSV — pandas
# ---------------------------------------------------------------------------

def _extract_csv(path: Path, depth: str) -> dict[str, Any]:
    import pandas as pd

    if depth == "names_only":
        # Fast path: ``nrows=0`` makes pandas read only the header
        # row and return an empty DataFrame with the correct columns.
        # For a multi-GB CSV this is constant-time + a single read of
        # the first line, vs a full pass that materialises every
        # column in memory for type inference. A bare "what columns
        # does this dataset have" call should not OOM the app.
        header_df = pd.read_csv(path, nrows=0)
        return _names_only_payload(
            list(header_df.columns), path, "csv",
        )
    # low_memory=False gives a single-pass type inference — more accurate
    # for columns where the type isn't obvious from the first chunk. For
    # genuinely huge CSVs this will be slow; step 7 can add streaming.
    df = pd.read_csv(path, low_memory=False)
    return _extract_from_pandas(
        df, depth=depth, dataset_name=path.name, file_type="csv"
    )


# ---------------------------------------------------------------------------
# TSV — same as CSV, tab-separated
# ---------------------------------------------------------------------------

def _extract_tsv(path: Path, depth: str) -> dict[str, Any]:
    import pandas as pd

    if depth == "names_only":
        # See _extract_csv for the fast-path rationale.
        header_df = pd.read_csv(path, sep="\t", nrows=0)
        return _names_only_payload(
            list(header_df.columns), path, "tsv",
        )
    df = pd.read_csv(path, sep="\t", low_memory=False)
    return _extract_from_pandas(
        df, depth=depth, dataset_name=path.name, file_type="tsv"
    )


# ---------------------------------------------------------------------------
# Parquet — pyarrow-backed via pandas
# ---------------------------------------------------------------------------

def _extract_parquet(path: Path, depth: str) -> dict[str, Any]:
    """Parquet preserves column dtypes natively (unlike CSV, which the
    extractor has to infer). Schema extraction is therefore a thin
    wrapper around ``read_parquet``: pandas hands back a DataFrame
    whose dtypes already match what the writer set.

    Heavy datasets: pyarrow streams the file rather than reading it
    whole into memory like the CSV path does, but we still call
    ``read_parquet`` (no streaming yet). For Parquet files much
    larger than RAM this will OOM — out of scope for the current
    pilot, where datasets fit comfortably."""
    import pandas as pd

    if depth == "names_only":
        # Fast path: pyarrow exposes the Parquet schema in the file
        # footer (constant-time regardless of file size). The full-
        # load fallback below covers any depth that actually needs
        # column data.
        try:
            import pyarrow.parquet as pq
            pf = pq.ParquetFile(str(path))
            names = list(pf.schema_arrow.names)
            return _names_only_payload(names, path, "parquet")
        except Exception:  # noqa: BLE001 — fall through to full load
            pass
    df = pd.read_parquet(path)
    return _extract_from_pandas(
        df, depth=depth, dataset_name=path.name, file_type="parquet"
    )


# ---------------------------------------------------------------------------
# Line-delimited JSON — .jsonl / .ndjson
# ---------------------------------------------------------------------------

def _extract_jsonl(path: Path, depth: str) -> dict[str, Any]:
    """One JSON object per line. Each object is treated as a row;
    pandas infers column types from the union of keys.

    Top-level JSON arrays (a single ``[{...}, {...}]`` document) are
    NOT supported here. They're shape-arbitrary — a record could
    contain nested objects or arrays — and don't fit the tabular
    model the rest of the pipeline assumes. Researchers can convert
    with ``jq -c '.[]'`` if needed.
    """
    import pandas as pd

    if depth == "names_only":
        # Fast path: read only the first record to discover keys.
        # JSONL has no separate schema, so we can't avoid reading at
        # least one line; ``nrows=1`` keeps memory bounded.
        # Note: this only sees keys present in the first record. A
        # downstream consumer that needs the full column union must
        # use a deeper depth (which loads the file).
        header_df = pd.read_json(path, lines=True, nrows=1)
        return _names_only_payload(
            list(header_df.columns), path, "jsonl",
        )
    df = pd.read_json(path, lines=True)
    return _extract_from_pandas(
        df, depth=depth, dataset_name=path.name, file_type="jsonl"
    )


# ---------------------------------------------------------------------------
# Generic pandas → schema
# ---------------------------------------------------------------------------

def _extract_from_pandas(
    df: Any,
    *,
    depth: str,
    dataset_name: str,
    file_type: str,
) -> dict[str, Any]:
    variables: list[dict[str, Any]] = []
    for name in df.columns:
        # Original name for DataFrame indexing; sanitized for the payload.
        col_name = str(name)
        safe_name = safe_key(col_name)
        var: dict[str, Any] = {"name": safe_name}
        if depth != "names_only":
            var["type"] = _pandas_type(df[name])
        # CSV / R data frames typically have no variable labels or value
        # labels metadata to expose at the names_types_labels depth. If
        # future formats (SPSS etc.) carry labels, we'll add them here.
        if depth == "names_types_labels_summary":
            series = df[name]
            n_obs = int(len(series))
            raw_na = int(series.isna().sum())
            var["na_count"] = _suppress_rare_count(
                raw_na, n_obs, _SCHEMA_SUMMARY_THRESHOLD,
            )
            if var.get("type") == _TYPE_CATEGORICAL:
                var["distinct_count"] = int(series.nunique(dropna=True))
        variables.append(var)
    return {
        "status": "ok",
        # See the identical note in _extract_stata_dta — filename is a
        # prompt-injection surface when echoed unsanitized.
        "dataset": safe_text(dataset_name),
        "file_type": file_type,
        "depth": depth,
        "observation_count": int(len(df)),
        "variables": variables,
    }


def _pandas_type(series: Any) -> str:
    """Map a pandas dtype to our coarse taxonomy."""
    import pandas as pd

    dtype = series.dtype
    if pd.api.types.is_bool_dtype(dtype):
        return _TYPE_BOOLEAN
    if pd.api.types.is_integer_dtype(dtype):
        return _TYPE_INTEGER
    if pd.api.types.is_float_dtype(dtype):
        return _TYPE_NUMERIC
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return _TYPE_DATETIME
    if isinstance(dtype, pd.CategoricalDtype):
        return _TYPE_CATEGORICAL
    # Object columns in pandas are usually strings, but can also be
    # heterogeneous. We call them strings unless the unique count is small
    # enough to treat as categorical — matches how researchers actually
    # model them.
    if pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype):
        try:
            n = len(series)
            if n == 0:
                return _TYPE_STRING
            nunique = series.nunique(dropna=True)
            # Heuristic: fewer than 20 distinct values and fewer than 5% of
            # total → categorical. Arbitrary but useful; researchers can
            # always override in their analysis.
            if nunique <= 20 and nunique <= max(1, n // 20):
                return _TYPE_CATEGORICAL
        except (TypeError, ValueError):
            pass
        return _TYPE_STRING
    return _TYPE_UNKNOWN
