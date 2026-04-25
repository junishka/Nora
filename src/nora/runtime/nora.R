# Nora runtime library for R.
#
# Sourced at the top of every R script Nora runs. Provides the single
# sanctioned I/O surface for emitting structured results:
#   nora$result(type = "linear_regression", ...)
#   nora$from_lm(model, ...)
#   nora$from_t_test(res, ...)
#   nora$from_summarize(var_name, n, mean, sd, missing_count)
#   nora$from_table(var_name, counts, n, missing_count)
#
# The script writes structured payloads to the path in $NORA_RESULT_PATH.
# Raw stdout / stderr are captured by the executor as the "raw log" the
# researcher sees in the TUI; only the structured JSON reaches the sanitizer
# (and from there, Claude).
#
# Ships a pure-R JSON serializer so researchers don't need to install
# jsonlite. Handles the subset of types Nora needs: null, TRUE/FALSE,
# numbers, strings, unnamed lists/vectors (→ arrays), named lists (→
# objects). That's enough for every v0 analysis payload.
#
# NOTE: floats are emitted at full precision. The Python sanitizer clamps
# precision per-type using sigfigs_for_n; the runtime library does NOT
# pre-round, so the clamp is applied consistently regardless of which
# language produced the result.

nora <- new.env(parent = emptyenv())


# ---------------------------------------------------------------------------
# Per-run authenticity token
# ---------------------------------------------------------------------------
# Read the token once at source-time, stash it in the library's private
# env, and clear the env var so user code loaded after this file can't
# read it via Sys.getenv. Claude's script still CAN find it via
# environment introspection (`ls(nora)`, `get("token", ..., envir =
# nora)`) — R closures are open — but doing so requires code that
# clearly shows up in the executed script the researcher reviews. That
# raises attacker cost without pretending to be a structural
# guarantee. See `docs/direction.md` "runtime-library contract" for
# the deliberate limits of this measure.

nora$.run_token <- Sys.getenv("NORA_RUN_TOKEN")
if (!nzchar(nora$.run_token)) {
  stop(
    "NORA_RUN_TOKEN not set. This script must be run through the ",
    "Nora executor; direct `Rscript` invocation of user code that ",
    "emits result payloads isn't supported."
  )
}
Sys.unsetenv("NORA_RUN_TOKEN")


# ---------------------------------------------------------------------------
# Pure-R JSON serializer
# ---------------------------------------------------------------------------

nora$.json_escape_str <- function(s) {
  s <- as.character(s)
  # Order matters: backslash first so we don't double-escape ours.
  s <- gsub("\\", "\\\\", s, fixed = TRUE)
  s <- gsub('"', '\\"', s, fixed = TRUE)
  s <- gsub("\n", "\\n", s, fixed = TRUE)
  s <- gsub("\r", "\\r", s, fixed = TRUE)
  s <- gsub("\t", "\\t", s, fixed = TRUE)
  paste0('"', s, '"')
}

nora$.to_json <- function(x) {
  if (is.null(x)) return("null")

  # Scalars first — auto-unbox for length-1 atomics.
  if (is.logical(x) && length(x) == 1 && !is.na(x)) {
    return(if (x) "true" else "false")
  }
  if (is.numeric(x) && length(x) == 1) {
    if (is.na(x) || !is.finite(x)) return("null")
    # `digits = 17` preserves full IEEE-754 precision — the Python
    # sanitizer is responsible for N-appropriate clamping, so the
    # runtime emits as-is. Allow scientific notation (the default) so
    # extremely small / large numbers get compact valid-JSON literals
    # like `1.7858e-41` instead of ugly long decimals.
    return(format(x, digits = 17, trim = TRUE))
  }
  if (is.character(x) && length(x) == 1 && !is.na(x)) {
    return(nora$.json_escape_str(x))
  }

  # Vectors → JSON array, element-wise.
  if (is.atomic(x)) {
    parts <- vapply(seq_along(x), function(i) nora$.to_json(x[[i]]),
                    character(1))
    return(paste0("[", paste(parts, collapse = ","), "]"))
  }

  # Lists.
  if (is.list(x)) {
    nms <- names(x)
    if (is.null(nms) || any(!nzchar(nms))) {
      parts <- vapply(x, nora$.to_json, character(1))
      return(paste0("[", paste(parts, collapse = ","), "]"))
    }
    parts <- vapply(seq_along(x), function(i) {
      paste0(nora$.json_escape_str(nms[i]), ":",
             nora$.to_json(x[[i]]))
    }, character(1))
    return(paste0("{", paste(parts, collapse = ","), "}"))
  }

  stop("nora.R: unsupported type for JSON: ", class(x)[1])
}


# ---------------------------------------------------------------------------
# Core emit
# ---------------------------------------------------------------------------

nora$.write_result <- function(payload) {
  result_path <- Sys.getenv("NORA_RESULT_PATH")
  if (!nzchar(result_path)) {
    stop(
      "NORA_RESULT_PATH not set. This script must be run through ",
      "Nora — direct `Rscript` invocation isn't supported."
    )
  }
  # Embed the per-run authenticity token. The executor validates this
  # and strips it before the payload reaches the sanitizer.
  payload[["_token"]] <- nora$.run_token
  con <- file(result_path, open = "w", encoding = "UTF-8")
  on.exit(close(con), add = TRUE)
  writeLines(nora$.to_json(payload), con)
  invisible(NULL)
}

nora$result <- function(type, ...) {
  payload <- c(list(type = type), list(...))
  nora$.write_result(payload)
}


# ---------------------------------------------------------------------------
# Convenience helpers that pull structured payloads out of common R objects
# ---------------------------------------------------------------------------

#' From an `lm` fit, emit a linear_regression payload.
#'
#' The helper covers the fields the Nora linear_regression schema
#' accepts. Any extra kwargs passed via `...` are included too (dropped
#' by the sanitizer if not whitelisted).
#'
#' Also prints R's native `summary(model)` table to stdout so the
#' researcher sees the familiar regression output in the TUI's raw
#' log panel. Stdout never reaches Claude (executor strips it before
#' anything returns to the sanitizer), so printing here is only for
#' the researcher's benefit.
nora$from_lm <- function(model, ...) {
  print(summary(model))
  s <- summary(model)
  ce <- as.data.frame(s$coefficients)
  coefs <- as.list(ce[, "Estimate"])
  names(coefs) <- rownames(ce)
  ses <- as.list(ce[, "Std. Error"])
  names(ses) <- rownames(ce)
  tvals <- as.list(ce[, "t value"])
  names(tvals) <- rownames(ce)
  pvals <- as.list(ce[, "Pr(>|t|)"])
  names(pvals) <- rownames(ce)

  response <- as.character(attr(model$terms, "variables")[[2]])
  # Wrap in `as.list(...)` so a single-predictor model serializes as the
  # JSON array `["x"]` not the bare string `"x"` — the sanitizer's
  # `predictor_variables` field expects a list of strings.
  predictors <- as.list(as.character(attr(model$terms, "term.labels")))

  f_value <- if (!is.null(s$fstatistic)) unname(s$fstatistic["value"]) else NULL
  f_pvalue <- if (!is.null(s$fstatistic)) {
    pf(s$fstatistic["value"], s$fstatistic["numdf"], s$fstatistic["dendf"],
       lower.tail = FALSE)
  } else NULL
  if (!is.null(f_pvalue)) f_pvalue <- unname(f_pvalue)

  nora$result(
    type = "linear_regression",
    n = as.integer(nobs(model)),
    response_variable = response,
    predictor_variables = predictors,
    coefficients = coefs,
    standard_errors = ses,
    t_statistics = tvals,
    p_values = pvals,
    r_squared = s$r.squared,
    adj_r_squared = s$adj.r.squared,
    f_statistic = f_value,
    f_p_value = f_pvalue,
    degrees_of_freedom = as.integer(s$df[2]),
    residual_std_error = s$sigma,
    ...
  )
}


#' From a `t.test` result, emit a t_test payload.
#'
#' Also prints the native t.test output — R formats this nicely
#' (test name, CI, p-value, sample means) so the researcher sees
#' the conventional view in the raw log panel.
nora$from_t_test <- function(res, ...) {
  print(res)
  # t.test returns different shapes depending on one-sample vs two-sample.
  is_two_sample <- grepl("two sample", res$method, ignore.case = TRUE)
  is_welch      <- grepl("welch",       res$method, ignore.case = TRUE)
  is_paired     <- grepl("paired",      res$method, ignore.case = TRUE)

  subtype <- if (is_welch) "welch"
             else if (is_paired) "paired"
             else if (is_two_sample) "two_sample"
             else "one_sample"

  # res$estimate can be length 1 (one_sample / paired) or length 2.
  ests <- res$estimate
  mean1 <- unname(ests[1])
  mean2 <- if (length(ests) >= 2) unname(ests[2]) else NULL

  # Sample sizes: for t.test, these come from the original data length,
  # which the test object doesn't carry — researcher must pass n1/n2
  # explicitly.
  args <- list(...)
  if (is.null(args$n1)) {
    stop("nora$from_t_test: n1 must be provided via `n1 = length(x)`. ",
         "The t.test object doesn't carry sample sizes.")
  }
  if (subtype %in% c("two_sample", "welch") && is.null(args$n2)) {
    stop("nora$from_t_test: n2 must be provided for ", subtype,
         " tests via `n2 = length(y)`.")
  }

  ci <- res$conf.int
  ci_list <- if (!is.null(ci) && length(ci) == 2) list(ci[1], ci[2]) else NULL

  payload <- c(
    list(
      type = "t_test",
      test_type = subtype,
      mean1 = mean1,
      t_statistic = unname(res$statistic),
      p_value = unname(res$p.value),
      degrees_of_freedom = unname(res$parameter),
      alternative = res$alternative
    ),
    args
  )
  if (!is.null(mean2)) payload$mean2 <- mean2
  if (!is.null(ci_list)) payload$confidence_interval <- ci_list

  do.call(nora$result, payload)
}


#' Emit a descriptive payload for a single variable.
#'
#' Prints a compact one-variable summary to stdout so the researcher
#' sees "variable: n=X, mean=Y, sd=Z, missing=M" in the raw log
#' panel. The caller provides the numbers; we don't recompute.
nora$from_summarize <- function(variable, n, mean, sd, missing_count,
                                   distinct_count = NULL, ...) {
  cat(sprintf(
    "%s: n=%d, mean=%.6g, sd=%.6g, missing=%d",
    variable, as.integer(n), mean, sd, as.integer(missing_count)
  ))
  if (!is.null(distinct_count)) {
    cat(sprintf(", distinct=%d", as.integer(distinct_count)))
  }
  cat("\n")
  nora$result(
    type = "descriptive",
    variable = variable,
    n = as.integer(n),
    mean = mean,
    sd = sd,
    missing_count = as.integer(missing_count),
    distinct_count = if (is.null(distinct_count)) NULL else as.integer(distinct_count),
    ...
  )
}


#' Emit a magnitude_table payload (sum or mean of `value_var` by `group_var`).
#'
#' For each group, the helper computes three quantities:
#'   - value: the aggregate (sum or mean)
#'   - n: number of non-missing observations contributing
#'   - max_share: the largest single contributor's share of the total
#'
#' `max_share` is the dominance metric the sanitizer consults to apply
#' the (1, k)-dominance rule. It's required by the schema but NOT
#' forwarded to the frontier model — the sanitizer strips it after
#' consulting. Computed as `max(abs(x)) / sum(abs(x))` on the group's
#' values so mixed signs don't produce meaningless shares.
nora$from_magnitude_table <- function(df, group_var, value_var,
                                          aggregation = "sum", ...) {
  # Native-R preview for the researcher's raw log panel. Use the same
  # aggregation the payload will report so the printed table matches
  # what the sanitized result says.
  tryCatch({
    na_mask <- !is.na(df[[group_var]]) & !is.na(df[[value_var]])
    if (any(na_mask)) {
      agg_fn <- if (aggregation == "sum") sum else mean
      agg_df <- aggregate(
        df[[value_var]][na_mask],
        by = list(df[[group_var]][na_mask]),
        FUN = agg_fn
      )
      names(agg_df) <- c(group_var, paste0(aggregation, "_", value_var))
      cat(sprintf("Magnitude table: %s of %s by %s\n",
                  aggregation, value_var, group_var))
      print(agg_df)
    }
  }, error = function(e) {
    cat(sprintf("(native preview skipped: %s)\n", conditionMessage(e)))
  })

  if (!(aggregation %in% c("sum", "mean"))) {
    stop('nora$from_magnitude_table: aggregation must be "sum" or "mean", ',
         'got ', aggregation)
  }
  if (!(group_var %in% names(df))) {
    stop('nora$from_magnitude_table: group_var ', group_var,
         ' not in data')
  }
  if (!(value_var %in% names(df))) {
    stop('nora$from_magnitude_table: value_var ', value_var,
         ' not in data')
  }

  groups <- unique(df[[group_var]])
  # Drop NA group labels — they don't form a meaningful cell.
  groups <- groups[!is.na(groups)]

  cells <- list()
  for (g in groups) {
    vals <- df[[value_var]][df[[group_var]] == g & !is.na(df[[value_var]])]
    n <- length(vals)
    if (n == 0) {
      # No non-missing observations for this group — emit a zero cell
      # so the sanitizer suppresses it on n grounds.
      cells[[as.character(g)]] <- list(value = 0, n = 0L, max_share = 0)
      next
    }
    total_abs <- sum(abs(vals))
    if (aggregation == "sum") {
      value <- sum(vals)
    } else {
      value <- mean(vals)
    }
    # Guard against all-zero groups: max_share undefined; set to 0
    # (no contributor dominates because there's no magnitude).
    max_share <- if (total_abs == 0) 0 else max(abs(vals)) / total_abs
    cells[[as.character(g)]] <- list(
      value = as.numeric(value),
      n = as.integer(n),
      max_share = as.numeric(max_share)
    )
  }

  nora$result(
    type = "magnitude_table",
    row_variable = as.character(group_var),
    value_variable = as.character(value_var),
    aggregation = aggregation,
    cells = cells,
    ...
  )
}


#' From a 2D table / matrix, emit a crosstab payload.
#'
#' Input can be:
#'   - a 2D `table` object (from `table(x, y)`)
#'   - a numeric matrix with `dimnames`
#'
#' The helper copies over the dimension names as row/column variable
#' labels if present, falling back to `"row"` / `"col"` otherwise.
#' Crosstabs emit cells only — never margins. The sanitizer will drop
#' any margin-ish field by name if one slips in here.
nora$from_crosstab <- function(tbl, row_variable = NULL, col_variable = NULL,
                                  missing_count = 0L, ...) {
  if (!(is.table(tbl) || is.matrix(tbl))) {
    stop("nora$from_crosstab: expected a 2D table or matrix, got ",
         class(tbl)[1])
  }
  # Native preview for the raw log panel.
  print(tbl)
  if (length(dim(tbl)) != 2) {
    stop("nora$from_crosstab: expected a 2D structure, got ",
         length(dim(tbl)), " dimension(s). Use nora$from_table for 1D.")
  }

  rows <- dimnames(tbl)[[1]]
  cols <- dimnames(tbl)[[2]]
  if (is.null(rows)) rows <- paste0("row", seq_len(nrow(tbl)))
  if (is.null(cols)) cols <- paste0("col", seq_len(ncol(tbl)))

  dn <- names(dimnames(tbl))
  if (is.null(row_variable)) {
    row_variable <- if (!is.null(dn) && nzchar(dn[1])) dn[1] else "row"
  }
  if (is.null(col_variable)) {
    col_variable <- if (!is.null(dn) && length(dn) > 1 && nzchar(dn[2])) dn[2] else "col"
  }

  # Build the nested dict counts[row][col] = integer.
  counts <- list()
  for (r in rows) {
    inner <- list()
    for (c in cols) {
      inner[[c]] <- as.integer(tbl[r, c])
    }
    counts[[r]] <- inner
  }

  nora$result(
    type = "crosstab",
    row_variable = as.character(row_variable),
    col_variable = as.character(col_variable),
    counts = counts,
    missing_count = as.integer(missing_count),
    ...
  )
}


#' Emit a frequency_table payload.
nora$from_table <- function(variable, counts, n = NULL, missing_count = 0L, ...) {
  # `counts` should be a named integer vector / list. Normalize to a
  # named list of ints.
  if (is.table(counts)) {
    # Native preview — R formats a 1D table nicely as a two-row block.
    cat(sprintf("Frequency table: %s\n", variable))
    print(counts)
    d <- as.list(as.integer(counts))
    names(d) <- names(counts)
    counts <- d
  } else if (is.list(counts) || !is.null(names(counts))) {
    cat(sprintf("Frequency table: %s\n", variable))
    for (lvl in names(counts)) {
      cat(sprintf("  %s: %s\n", lvl, counts[[lvl]]))
    }
  }
  if (is.null(n)) {
    n <- sum(unlist(counts)) + missing_count
  }
  nora$result(
    type = "frequency_table",
    variable = variable,
    counts = counts,
    n = as.integer(n),
    missing_count = as.integer(missing_count),
    ...
  )
}


# ---------------------------------------------------------------------------
# Smoke test — skipped automatically in production because the env var is
# expected to be set by the executor. Researchers running `source("nora.R")`
# manually (e.g., for exploration) can ignore this file — calling result()
# without the env var raises a clear error.
# ---------------------------------------------------------------------------

invisible(NULL)
