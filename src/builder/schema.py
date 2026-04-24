"""Builder — schema extractor.

Produces a structural summary of a dataset (variable names, types, labels,
observation count, optionally NA counts / distinct counts) from files on
the researcher's machine. Supported formats: `.dta` (Stata), `.rds` (R),
`.csv`.

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

Not included, ever, at any depth:
- Actual observation values
- Min / max / mean / quantiles on numerics (these are individual values)
- Frequency tables (step 5 territory — requires SDC cell-suppression)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from builder.text_safety import safe_key, safe_text


Depth = Literal[
    "names_only",
    "names_types",
    "names_types_labels",
    "names_types_labels_summary",
]

_VALID_DEPTHS: frozenset[str] = frozenset(
    ("names_only", "names_types", "names_types_labels", "names_types_labels_summary")
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


def load_data(dataset_path: Path) -> Any:
    """Load the dataset at `dataset_path` as a pandas DataFrame.

    Dispatches by extension (.dta / .rds / .csv). Used by
    ``extract()`` at the summary depth, and by ``data_request`` for
    computing bounded sanitized facts about a variable.

    Returns the full data as a DataFrame. Callers are responsible for
    never returning raw values — in Builder, only ``data_request``
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
            raise ValueError(f".rds file contains no objects: {dataset_path}")
        obj = next(iter(result.values()))
        if not hasattr(obj, "columns"):
            raise ValueError(
                f".rds at {dataset_path} does not contain a data frame; "
                f"got {type(obj).__name__}"
            )
        return obj
    if suffix == ".csv":
        return pd.read_csv(dataset_path, low_memory=False)
    raise ValueError(
        f"unsupported format: {suffix!r}. Builder reads .dta, .rds, .csv."
    )


def extract(dataset_path: Path, depth: str) -> dict[str, Any]:
    """Return a structured schema summary for the file at `dataset_path`.

    Dispatches by suffix. Raises ValueError for unsupported formats or
    invalid depth; lets underlying library errors propagate (tool layer
    catches them and returns a policy-shaped error payload).
    """
    if depth not in _VALID_DEPTHS:
        raise ValueError(
            f"invalid depth: {depth!r}; valid: {sorted(_VALID_DEPTHS)}"
        )
    suffix = dataset_path.suffix.lower()
    if suffix == ".dta":
        return _extract_stata(dataset_path, depth)
    if suffix == ".rds":
        return _extract_rds(dataset_path, depth)
    if suffix == ".csv":
        return _extract_csv(dataset_path, depth)
    raise ValueError(
        f"unsupported format: {suffix!r}. Builder currently reads "
        ".dta (Stata), .rds (R), and .csv. Other formats (.rda, .xlsx, "
        ".sav, .parquet) are not supported yet."
    )


# ---------------------------------------------------------------------------
# Stata — .dta via pyreadstat
# ---------------------------------------------------------------------------

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
            # the codes (keys) and labels (values) originate in the data.
            label_set = meta.variable_to_label.get(name)
            if label_set and label_set in meta.value_labels:
                raw = meta.value_labels[label_set]
                var["value_labels"] = {
                    safe_key(str(k)): safe_text(str(v))
                    for k, v in raw.items()
                }

        # Use the ORIGINAL name as the key when reading df (to match pandas),
        # but report the SANITIZED name to Claude.
        if wants_summary and df is not None and name in df.columns:
            series = df[name]
            var["na_count"] = int(series.isna().sum())
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
        raise ValueError(f".rds file contains no objects: {path}")
    obj_key = next(iter(result))
    df = result[obj_key]
    # pyreadr can return non-DataFrame objects (lists, etc.). We only
    # handle data frames in v0.
    if not hasattr(df, "columns"):
        raise ValueError(
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
# CSV — pandas
# ---------------------------------------------------------------------------

def _extract_csv(path: Path, depth: str) -> dict[str, Any]:
    import pandas as pd

    # low_memory=False gives a single-pass type inference — more accurate
    # for columns where the type isn't obvious from the first chunk. For
    # genuinely huge CSVs this will be slow; step 7 can add streaming.
    df = pd.read_csv(path, low_memory=False)
    return _extract_from_pandas(
        df, depth=depth, dataset_name=path.name, file_type="csv"
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
            var["na_count"] = int(series.isna().sum())
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
