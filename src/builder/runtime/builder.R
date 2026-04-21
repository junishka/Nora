# Builder runtime library for R.
#
# Sourced at the top of every R script Builder runs. Provides the single
# sanctioned I/O surface for emitting structured results:
#   builder$result(type = "linear_regression", ...)
#   builder$from_lm(model, ...)
#   builder$from_t_test(res, ...)
#   builder$from_summarize(var_name, n, mean, sd, missing_count)
#   builder$from_table(var_name, counts, n, missing_count)
#
# The script writes structured payloads to the path in $BUILDER_RESULT_PATH.
# Raw stdout / stderr are captured by the executor as the "raw log" the
# researcher sees in the TUI; only the structured JSON reaches the sanitizer
# (and from there, Claude).
#
# Ships a pure-R JSON serializer so researchers don't need to install
# jsonlite. Handles the subset of types Builder needs: null, TRUE/FALSE,
# numbers, strings, unnamed lists/vectors (→ arrays), named lists (→
# objects). That's enough for every v0 analysis payload.
#
# NOTE: floats are emitted at full precision. The Python sanitizer clamps
# precision per-type using sigfigs_for_n; the runtime library does NOT
# pre-round, so the clamp is applied consistently regardless of which
# language produced the result.

builder <- new.env(parent = emptyenv())


# ---------------------------------------------------------------------------
# Per-run authenticity token
# ---------------------------------------------------------------------------
# Read the token once at source-time, stash it in the library's private
# env, and clear the env var so user code loaded after this file can't
# read it via Sys.getenv. Claude's script still CAN find it via
# environment introspection (`ls(builder)`, `get("token", ..., envir =
# builder)`) — R closures are open — but doing so requires code that
# clearly shows up in the executed script the researcher reviews. That
# raises attacker cost without pretending to be a structural
# guarantee. See `docs/direction.md` "runtime-library contract" for
# the deliberate limits of this measure.

builder$.run_token <- Sys.getenv("BUILDER_RUN_TOKEN")
if (!nzchar(builder$.run_token)) {
  stop(
    "BUILDER_RUN_TOKEN not set. This script must be run through the ",
    "Builder executor; direct `Rscript` invocation of user code that ",
    "emits result payloads isn't supported."
  )
}
Sys.unsetenv("BUILDER_RUN_TOKEN")


# ---------------------------------------------------------------------------
# Pure-R JSON serializer
# ---------------------------------------------------------------------------

builder$.json_escape_str <- function(s) {
  s <- as.character(s)
  # Order matters: backslash first so we don't double-escape ours.
  s <- gsub("\\", "\\\\", s, fixed = TRUE)
  s <- gsub('"', '\\"', s, fixed = TRUE)
  s <- gsub("\n", "\\n", s, fixed = TRUE)
  s <- gsub("\r", "\\r", s, fixed = TRUE)
  s <- gsub("\t", "\\t", s, fixed = TRUE)
  paste0('"', s, '"')
}

builder$.to_json <- function(x) {
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
    return(builder$.json_escape_str(x))
  }

  # Vectors → JSON array, element-wise.
  if (is.atomic(x)) {
    parts <- vapply(seq_along(x), function(i) builder$.to_json(x[[i]]),
                    character(1))
    return(paste0("[", paste(parts, collapse = ","), "]"))
  }

  # Lists.
  if (is.list(x)) {
    nms <- names(x)
    if (is.null(nms) || any(!nzchar(nms))) {
      parts <- vapply(x, builder$.to_json, character(1))
      return(paste0("[", paste(parts, collapse = ","), "]"))
    }
    parts <- vapply(seq_along(x), function(i) {
      paste0(builder$.json_escape_str(nms[i]), ":",
             builder$.to_json(x[[i]]))
    }, character(1))
    return(paste0("{", paste(parts, collapse = ","), "}"))
  }

  stop("builder.R: unsupported type for JSON: ", class(x)[1])
}


# ---------------------------------------------------------------------------
# Core emit
# ---------------------------------------------------------------------------

builder$.write_result <- function(payload) {
  result_path <- Sys.getenv("BUILDER_RESULT_PATH")
  if (!nzchar(result_path)) {
    stop(
      "BUILDER_RESULT_PATH not set. This script must be run through ",
      "Builder — direct `Rscript` invocation isn't supported."
    )
  }
  # Embed the per-run authenticity token. The executor validates this
  # and strips it before the payload reaches the sanitizer.
  payload[["_token"]] <- builder$.run_token
  con <- file(result_path, open = "w", encoding = "UTF-8")
  on.exit(close(con), add = TRUE)
  writeLines(builder$.to_json(payload), con)
  invisible(NULL)
}

builder$result <- function(type, ...) {
  payload <- c(list(type = type), list(...))
  builder$.write_result(payload)
}


# ---------------------------------------------------------------------------
# Convenience helpers that pull structured payloads out of common R objects
# ---------------------------------------------------------------------------

#' From an `lm` fit, emit a linear_regression payload.
#'
#' The helper covers the fields the Builder linear_regression schema
#' accepts. Any extra kwargs passed via `...` are included too (dropped
#' by the sanitizer if not whitelisted).
builder$from_lm <- function(model, ...) {
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

  builder$result(
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
builder$from_t_test <- function(res, ...) {
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
    stop("builder$from_t_test: n1 must be provided via `n1 = length(x)`. ",
         "The t.test object doesn't carry sample sizes.")
  }
  if (subtype %in% c("two_sample", "welch") && is.null(args$n2)) {
    stop("builder$from_t_test: n2 must be provided for ", subtype,
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

  do.call(builder$result, payload)
}


#' Emit a descriptive payload for a single variable.
builder$from_summarize <- function(variable, n, mean, sd, missing_count,
                                   distinct_count = NULL, ...) {
  builder$result(
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
builder$from_magnitude_table <- function(df, group_var, value_var,
                                          aggregation = "sum", ...) {
  if (!(aggregation %in% c("sum", "mean"))) {
    stop('builder$from_magnitude_table: aggregation must be "sum" or "mean", ',
         'got ', aggregation)
  }
  if (!(group_var %in% names(df))) {
    stop('builder$from_magnitude_table: group_var ', group_var,
         ' not in data')
  }
  if (!(value_var %in% names(df))) {
    stop('builder$from_magnitude_table: value_var ', value_var,
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

  builder$result(
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
builder$from_crosstab <- function(tbl, row_variable = NULL, col_variable = NULL,
                                  missing_count = 0L, ...) {
  if (!(is.table(tbl) || is.matrix(tbl))) {
    stop("builder$from_crosstab: expected a 2D table or matrix, got ",
         class(tbl)[1])
  }
  if (length(dim(tbl)) != 2) {
    stop("builder$from_crosstab: expected a 2D structure, got ",
         length(dim(tbl)), " dimension(s). Use builder$from_table for 1D.")
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

  builder$result(
    type = "crosstab",
    row_variable = as.character(row_variable),
    col_variable = as.character(col_variable),
    counts = counts,
    missing_count = as.integer(missing_count),
    ...
  )
}


#' Emit a frequency_table payload.
builder$from_table <- function(variable, counts, n = NULL, missing_count = 0L, ...) {
  # `counts` should be a named integer vector / list. Normalize to a
  # named list of ints.
  if (is.table(counts)) {
    d <- as.list(as.integer(counts))
    names(d) <- names(counts)
    counts <- d
  }
  if (is.null(n)) {
    n <- sum(unlist(counts)) + missing_count
  }
  builder$result(
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
# expected to be set by the executor. Researchers running `source("builder.R")`
# manually (e.g., for exploration) can ignore this file — calling result()
# without the env var raises a clear error.
# ---------------------------------------------------------------------------

invisible(NULL)
