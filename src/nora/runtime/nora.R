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
# Plot helpers — model-output visualizations only
# ---------------------------------------------------------------------------
#
# Plots produced via these helpers are surfaced to the model on the
# next turn as image attachments. RAW-DATA plots (a histogram of an
# observed variable, a scatter of all rows, a density of a column)
# are NOT covered here on purpose — they would expose the data
# itself, which is the privacy line Nora is built to keep.
#
# What IS covered:
#   - Residual diagnostics (plot_residuals): residuals vs fitted,
#     QQ plot, scale-location, histogram of residuals. All
#     functions of model output, not raw observations.
#   - Interaction / predicted-value plots (plot_interaction):
#     model's predicted response across a variable's range, holding
#     other predictors at their means. Shows model behavior, not
#     data points.
#
# Future helpers (coefficient forest plots, marginal effects)
# follow the same principle: things the model produced from the fit,
# not visualizations of the rows.
#
# Mechanism: helpers write a PNG into <run_dir>/_nora_plots/ and
# append a JSONL entry to <run_dir>/_nora_plots/manifest.jsonl. The
# bridge reads ONLY the manifest — anything else dropped into the
# directory by ggsave / plain plot() / etc. is invisible to the
# model regardless. That's the allowlist; mirrors how the sanitizer
# works for textual results.

nora$.plots_dir <- function() {
  result_path <- Sys.getenv("NORA_RESULT_PATH")
  if (!nzchar(result_path)) return(NULL)
  d <- file.path(dirname(result_path), "_nora_plots")
  dir.create(d, showWarnings = FALSE, recursive = TRUE)
  d
}

nora$.append_plot_manifest <- function(file, kind, label) {
  d <- nora$.plots_dir()
  if (is.null(d)) return(invisible(NULL))
  entry <- list(file = file, kind = kind)
  if (!is.null(label) && nzchar(label)) entry$label <- label
  line <- nora$.to_json(entry)
  con <- file(file.path(d, "manifest.jsonl"), open = "a", encoding = "UTF-8")
  on.exit(close(con), add = TRUE)
  writeLines(line, con)
  invisible(NULL)
}

# Record a structured plot-helper failure so submit_script can surface
# it in the tool result the MODEL receives. Without this, helper
# failures only land in stderr and the model says "thumbnail should be
# visible above" while the researcher sees nothing.
nora$.append_plot_helper_error <- function(helper, message) {
  d <- nora$.plots_dir()
  if (is.null(d)) return(invisible(NULL))
  msg <- as.character(message)
  fix <- NULL
  lower <- tolower(msg)
  if (grepl("haven", lower) || grepl("could not find function .*read_dta", lower)) {
    fix <- "install.packages(\"haven\")"
  } else if (grepl("ggplot2", lower) || grepl("could not find function .*ggplot", lower)) {
    fix <- "install.packages(\"ggplot2\")"
  }
  entry <- list(helper = helper, error = "R error", message = msg)
  if (!is.null(fix)) entry$fix <- fix
  line <- nora$.to_json(entry)
  con <- file(file.path(d, "helper_errors.jsonl"),
              open = "a", encoding = "UTF-8")
  on.exit(close(con), add = TRUE)
  writeLines(line, con)
  invisible(NULL)
}

#' Write the four standard residual-diagnostic panels for an `lm`
#' (or anything `plot()` accepts as a fit) and register them with
#' the manifest so the model can see them on the next turn.
#'
#' Failures inside the helper are surfaced via `message()` (visible
#' to the researcher in the raw-log panel) but never raise — a
#' broken plot helper must not break the analysis script around it.
nora$plot_residuals <- function(model, label = NULL) {
  d <- nora$.plots_dir()
  if (is.null(d)) return(invisible(NULL))
  fname <- "residuals.png"
  res <- tryCatch({
    grDevices::png(file.path(d, fname),
                   width = 900, height = 700, res = 110)
    on.exit(grDevices::dev.off(), add = TRUE)
    op <- graphics::par(mfrow = c(2, 2))
    on.exit(graphics::par(op), add = TRUE)
    plot(model)
    TRUE
  }, error = function(e) {
    message("nora$plot_residuals failed: ", conditionMessage(e))
    nora$.append_plot_helper_error("plot_residuals", conditionMessage(e))
    FALSE
  })
  if (isTRUE(res)) {
    nora$.append_plot_manifest(
      fname, "residuals",
      if (is.null(label)) "Residual diagnostics" else label
    )
  }
  invisible(NULL)
}

#' Predicted-response curve across a single predictor, with other
#' predictors held at their means (numeric) or first level (factor).
#' Confidence bands are 1.96 * SE of the predicted mean.
#'
#' Optional ``xlab`` / ``ylab`` / ``title`` override the defaults
#' (which fall back to the variable name and "Predicted response").
#' If ``ggplot2`` is available the helper uses it for a cleaner
#' filled-ribbon CI; otherwise it falls back to base graphics.
#'
#' This is a model-output plot — the predicted line and bands come
#' from the fit's variance / coefficient estimates, not from the
#' data rows themselves. Same privacy posture as plot_residuals.
nora$plot_interaction <- function(model, var, label = NULL,
                                   xlab = NULL, ylab = NULL,
                                   title = NULL) {
  d <- nora$.plots_dir()
  if (is.null(d)) return(invisible(NULL))
  if (!is.character(var) || length(var) != 1) {
    message("nora$plot_interaction: `var` must be a single name string")
    return(invisible(NULL))
  }
  fname <- paste0("interaction_", make.names(var), ".png")
  xtitle <- if (is.null(xlab)) var else xlab
  ytitle <- if (is.null(ylab)) "Predicted response" else ylab
  ptitle <- if (is.null(title)) paste0("Predicted response by ", var) else title
  res <- tryCatch({
    md <- model$model
    if (is.null(md) || !var %in% names(md)) {
      stop("variable '", var, "' is not in the model frame")
    }
    template <- md[1, , drop = FALSE]
    for (col in names(template)) {
      v <- md[[col]]
      if (is.numeric(v))      template[[col]] <- mean(v, na.rm = TRUE)
      else if (is.factor(v))  template[[col]] <- levels(v)[1]
      else                    template[[col]] <- v[1]
    }
    xs <- md[[var]]
    if (is.numeric(xs)) {
      grid <- seq(min(xs, na.rm = TRUE), max(xs, na.rm = TRUE), length.out = 100)
    } else if (is.factor(xs)) {
      grid <- factor(levels(xs), levels = levels(xs))
    } else {
      grid <- unique(xs)
    }
    new <- template[rep(1, length(grid)), , drop = FALSE]
    new[[var]] <- grid
    pr <- stats::predict(model, newdata = new, se.fit = TRUE)
    lo <- pr$fit - 1.96 * pr$se.fit
    hi <- pr$fit + 1.96 * pr$se.fit

    have_gg <- requireNamespace("ggplot2", quietly = TRUE)
    if (have_gg && is.numeric(grid)) {
      # ggplot2 path: filled ribbon CI, theme_minimal, decent
      # default font sizes. Continuous-x only; for factor x we
      # fall through to a base barplot below.
      df <- data.frame(x = grid, fit = pr$fit, lo = lo, hi = hi)
      p <- ggplot2::ggplot(df, ggplot2::aes(x = x, y = fit)) +
        ggplot2::geom_ribbon(
          ggplot2::aes(ymin = lo, ymax = hi),
          fill = "#4C78A8", alpha = 0.20
        ) +
        ggplot2::geom_line(color = "#1F4E79", linewidth = 1) +
        ggplot2::labs(title = ptitle, x = xtitle, y = ytitle) +
        ggplot2::theme_minimal(base_size = 12) +
        ggplot2::theme(
          plot.title = ggplot2::element_text(face = "bold"),
          panel.grid.minor = ggplot2::element_blank()
        )
      ggplot2::ggsave(file.path(d, fname), plot = p,
                      width = 8, height = 5, dpi = 110)
    } else {
      # Base graphics fallback: still cleaner than the previous
      # default — filled polygon for the CI band, colored line,
      # margin-aware labels.
      grDevices::png(file.path(d, fname),
                     width = 900, height = 560, res = 110)
      on.exit(grDevices::dev.off(), add = TRUE)
      op <- graphics::par(mar = c(4.5, 4.5, 2.5, 1.5))
      on.exit(graphics::par(op), add = TRUE)
      if (is.numeric(grid)) {
        ylim <- range(c(lo, hi), na.rm = TRUE)
        plot(grid, pr$fit, type = "n",
             xlab = xtitle, ylab = ytitle, main = ptitle,
             ylim = ylim)
        graphics::polygon(
          c(grid, rev(grid)), c(lo, rev(hi)),
          col = grDevices::adjustcolor("#4C78A8", alpha.f = 0.20),
          border = NA
        )
        graphics::lines(grid, pr$fit, lwd = 2, col = "#1F4E79")
      } else {
        labels <- as.character(grid)
        graphics::barplot(
          pr$fit, names.arg = labels,
          ylab = ytitle, main = ptitle, col = "#4C78A8",
          border = "#1F4E79"
        )
      }
    }
    TRUE
  }, error = function(e) {
    message("nora$plot_interaction failed: ", conditionMessage(e))
    nora$.append_plot_helper_error("plot_interaction", conditionMessage(e))
    FALSE
  })
  if (isTRUE(res)) {
    nora$.append_plot_manifest(
      fname, "interaction",
      if (is.null(label)) paste0("Predicted response by ", var) else label
    )
  }
  invisible(NULL)
}


#' Forest plot of coefficient estimates with 95% CIs.
#'
#' Operates only on `coef(model)` and `confint(model)` — pure
#' functions of model output, never the raw data. The helper IS
#' the gate; there is no escape-hatch path that accepts an
#' arbitrary file. That would let a histogram of raw rows pose
#' as a coefficient plot — bypassing the privacy line the entire
#' system rests on.
nora$plot_coefficients <- function(model, label = NULL) {
  d <- nora$.plots_dir()
  if (is.null(d)) return(invisible(NULL))
  fname <- "coefficients.png"
  res <- tryCatch({
    cf <- coef(model)
    ci <- stats::confint(model)
    nms <- names(cf)
    if (is.null(nms)) nms <- as.character(seq_along(cf))
    # Drop the intercept by default — almost never on the same
    # scale as predictors. Researchers who want it can call
    # plot_coefficients on a fit without an intercept term.
    keep <- !(tolower(nms) %in% c("(intercept)", "intercept", "_cons"))
    if (!any(keep)) {
      stop("nothing to plot after dropping the intercept")
    }
    cf <- cf[keep]
    ci <- ci[keep, , drop = FALSE]
    nms <- nms[keep]
    # Order top-to-bottom matching the names vector — y axis goes
    # downward so we plot index 1 at the top.
    y <- seq_along(cf)
    grDevices::png(file.path(d, fname),
                   width = 900,
                   height = max(220, 60 * length(cf) + 120),
                   res = 110)
    on.exit(grDevices::dev.off(), add = TRUE)
    op <- graphics::par(mar = c(4.5, 7, 2, 2))
    on.exit(graphics::par(op), add = TRUE)
    xlim <- range(c(ci[, 1], ci[, 2]), finite = TRUE)
    plot(NA, xlim = xlim, ylim = c(length(cf) + 0.5, 0.5),
         yaxt = "n", xlab = "Coefficient (95% CI)", ylab = "",
         main = "Coefficients")
    graphics::abline(v = 0, lty = 2, col = "gray60")
    graphics::segments(ci[, 1], y, ci[, 2], y, lwd = 2, col = "#4C78A8")
    graphics::points(cf, y, pch = 16, cex = 1.4, col = "#4C78A8")
    graphics::axis(2, at = y, labels = nms, las = 1)
    TRUE
  }, error = function(e) {
    message("nora$plot_coefficients failed: ", conditionMessage(e))
    nora$.append_plot_helper_error("plot_coefficients", conditionMessage(e))
    FALSE
  })
  if (isTRUE(res)) {
    nora$.append_plot_manifest(
      fname, "coefficients",
      if (is.null(label)) "Coefficient estimates with 95% CIs" else label
    )
  }
  invisible(NULL)
}


#' Forest plot comparing one coefficient across multiple model fits.
#'
#' Use case: "Female gap, before vs after controls" — fit two
#' models, plot their named coefficient with CIs side-by-side.
#' ``models`` is a NAMED list of fits (the names become y-axis
#' labels). ``coef`` is the coefficient to extract from each fit
#' via ``coef(m)`` + ``vcov(m)``. SEs come from the diagonal of
#' the variance-covariance matrix.
#'
#' Without this helper, comparison plots forced cross-language
#' workflows: the model would extract estimates from R / Stata,
#' switch to Python or back to R, and hand-roll a forest plot —
#' often three attempts before one landed.
nora$plot_estimate_comparison <- function(models, coef, label = NULL) {
  d <- nora$.plots_dir()
  if (is.null(d)) return(invisible(NULL))
  fname <- "estimate_comparison.png"
  res <- tryCatch({
    if (!is.list(models) || length(models) < 2) {
      stop("`models` must be a list of at least 2 model fits")
    }
    if (!is.character(coef) || length(coef) != 1) {
      stop("`coef` must be a single coefficient name")
    }
    nms <- names(models)
    if (is.null(nms) || any(!nzchar(nms))) {
      nms <- paste0("Model ", seq_along(models))
    }

    n_models <- length(models)
    ests <- numeric(n_models)
    ses  <- numeric(n_models)
    for (i in seq_len(n_models)) {
      m <- models[[i]]
      cf <- stats::coef(m)
      if (!coef %in% names(cf)) {
        stop("coefficient '", coef, "' not in model: ", nms[i])
      }
      ests[i] <- cf[[coef]]
      vc <- stats::vcov(m)
      if (!coef %in% rownames(vc)) {
        stop("coefficient '", coef, "' not in vcov of model: ", nms[i])
      }
      ses[i] <- sqrt(vc[coef, coef])
    }
    los <- ests - 1.96 * ses
    his <- ests + 1.96 * ses

    grDevices::png(file.path(d, fname),
                   width = 900,
                   height = max(220, 80 * n_models + 100),
                   res = 110)
    on.exit(grDevices::dev.off(), add = TRUE)
    op <- graphics::par(mar = c(4.5, 8, 2.5, 2))
    on.exit(graphics::par(op), add = TRUE)
    xlim <- range(c(los, his), finite = TRUE)
    y <- seq_len(n_models)
    plot(NA, xlim = xlim, ylim = c(n_models + 0.5, 0.5),
         yaxt = "n", xlab = paste0(coef, " (95% CI)"), ylab = "",
         main = paste0("Estimate comparison: ", coef))
    graphics::abline(v = 0, lty = 2, col = "gray60")
    graphics::segments(los, y, his, y, lwd = 2, col = "#4C78A8")
    graphics::points(ests, y, pch = 16, cex = 1.4, col = "#4C78A8")
    graphics::axis(2, at = y, labels = nms, las = 1)
    TRUE
  }, error = function(e) {
    message("nora$plot_estimate_comparison failed: ", conditionMessage(e))
    nora$.append_plot_helper_error(
      "plot_estimate_comparison", conditionMessage(e)
    )
    FALSE
  })
  if (isTRUE(res)) {
    nora$.append_plot_manifest(
      fname, "coefficients",
      if (is.null(label)) paste0("Estimate comparison: ", coef) else label
    )
  }
  invisible(NULL)
}


# ---------------------------------------------------------------------------
# Smoke test — skipped automatically in production because the env var is
# expected to be set by the executor. Researchers running `source("nora.R")`
# manually (e.g., for exploration) can ignore this file — calling result()
# without the env var raises a clear error.
# ---------------------------------------------------------------------------

invisible(NULL)
