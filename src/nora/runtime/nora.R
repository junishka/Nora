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
  # RFC 8259 §7 requires that ALL U+0000..U+001F appear as escape
  # sequences inside JSON strings. Without this pass, a value-label or
  # variable-label byte like \x01 (real automated-export datasets do
  # ship these) reaches the wire as a raw control character — which
  # Python's `json.loads` rejects with "Invalid control character",
  # and the executor's JSONL parser drops every line of the payload
  # silently. Stata's helpers have the same gap; matching changes
  # land in the .ado files.
  # Codepoints 9 (\t), 10 (\n), 13 (\r) are already escaped via the
  # named-escape gsubs above. NUL (codepoint 0) cannot occur in an R
  # character vector at all -- R rejects strings with embedded nulls
  # at every ingestion boundary -- so 1..31 minus the three handled
  # is the full range we need to walk here.
  for (cp in setdiff(1:31, c(9L, 10L, 13L))) {
    s <- gsub(intToUtf8(cp), sprintf("\\u%04x", cp), s, fixed = TRUE)
  }
  paste0('"', s, '"')
}

nora$.to_json <- function(x) {
  if (is.null(x)) return("null")

  # Coerce factors to their character labels BEFORE any of the scalar
  # / vector branches. A factor is `is.atomic` and `length(x) == 1`
  # for length-1 cases, but is neither `is.numeric` nor
  # `is.character`, so it falls through every scalar guard into the
  # atomic-vector branch. That branch recurses on `x[[1]]`, and for a
  # factor `x[[1]]` is *another factor* equal to `x` itself —
  # infinite recursion → node stack overflow. Concretely: any
  # `factor(...)` value (very common with `read.csv` defaults pre-
  # R 4.0, and with `stringsAsFactors = TRUE`) crashes the script
  # before any payload reaches disk. Convert to character early so
  # the rest of the serializer treats labels like ordinary strings.
  if (is.factor(x)) x <- as.character(x)

  # NA-of-any-flavor is JSON null. Catch this BEFORE the atomic-vector
  # branch below: a length-1 NA passes `is.atomic`, and `x[[1]]` for an
  # NA is identical to NA itself, so without this guard the recursion
  # never terminates and the script crashes with a node stack overflow.
  # Concretely: any helper field that ends up NA upstream (e.g.
  # `attr(x, "label")` returning NA, an upstream `mean(x)` over an
  # all-NA vector) would crash the whole script before any payload
  # reached disk.
  if (length(x) == 1 && is.atomic(x) && is.na(x)) return("null")

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
    # `decimal.mark = "."` is required: `format()` honors the
    # locale-dependent OutDec option, so a researcher script that ran
    # `options(OutDec = ",")` (German/French/Spanish locales) would
    # otherwise emit `3,14...` and break JSON parsing for every line
    # that follows. Stata's helper uses `strofreal(..., "%21.17e")`
    # which is locale-independent — match that guarantee here.
    return(format(x, digits = 17, trim = TRUE, decimal.mark = "."))
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
    # Mixed naming (some named, some positional) — `list(a = 1, 2)` —
    # used to silently fall back to array-mode and DROP the named
    # entries. That's a quiet data loss: a researcher passing
    # mixed args (e.g. via `do.call`) gets a payload whose shape no
    # longer matches schema validation, but no error is raised.
    # Keep array-mode only for the *fully-positional* case (no names
    # at all). Any partial naming becomes object-mode, with empty
    # names auto-numbered so the JSON is still well-formed.
    if (is.null(nms)) {
      parts <- vapply(x, nora$.to_json, character(1))
      return(paste0("[", paste(parts, collapse = ","), "]"))
    }
    if (any(!nzchar(nms))) {
      idx <- which(!nzchar(nms))
      nms[idx] <- paste0("_", idx)
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
  # Embed the per-run authenticity token, then APPEND a single JSONL
  # line so multiple emit calls in one script all reach the executor.
  # Single-helper scripts produce one line; multi-helper scripts
  # produce N lines in emission order.
  payload[["_token"]] <- nora$.run_token
  con <- file(result_path, open = "a", encoding = "UTF-8")
  on.exit(close(con), add = TRUE)
  writeLines(nora$.to_json(payload), con)
  invisible(NULL)
}

nora$result <- function(type, ...) {
  # Strip the sanitizer-side helper-provenance marker. Typed
  # helpers (e.g. nora$from_magnitude_table) write directly via
  # .write_result with `_via_helper` set, proving they computed
  # disclosure metrics from raw data. Allowing it through here
  # would let a script forge the marker through the generic API
  # and bypass the dominance gate on magnitude_table.
  payload <- c(list(type = type), list(...))
  payload[["_via_helper"]] <- NULL
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
  s <- summary(model)
  print(s)

  # Per-class dispatch — the previous version assumed an lm/glm shape
  # ("Estimate" / "Std. Error" columns) and aborted with "undefined
  # columns selected" on Cox (coxph) and fixest fits, both of which
  # the comment block above claims to support. Now we detect class
  # explicitly and route extraction.
  is_cox    <- inherits(model, "coxph")
  is_fixest <- inherits(model, "fixest")
  is_mixed  <- inherits(model, "merMod")  # lmerMod, glmerMod, nlmerMod
  is_glm    <- inherits(model, "glm") && !is_mixed
  # ``glmerMod`` inherits from both glm and merMod; treat as mixed.
  is_lm     <- inherits(model, "lm") && !is_glm && !is_cox && !is_fixest && !is_mixed

  # Coefficient table location: fixest puts it in $coeftable, not
  # $coefficients (the $coefficients slot on a fixest summary is the
  # point-estimate vector).
  ce <- if (is_fixest) {
    as.data.frame(s$coeftable)
  } else {
    as.data.frame(s$coefficients)
  }
  ce_cols <- colnames(ce)

  # Estimate column: "Estimate" (lm/glm/fixest), "coef" (coxph), or
  # positional fallback to col 1.
  est_col <- if ("Estimate" %in% ce_cols) "Estimate" else
             if ("coef"     %in% ce_cols) "coef"     else
             ce_cols[1]
  # SE column: "Std. Error" (lm/glm/fixest), "se(coef)" (coxph), or
  # positional fallback to col 2 (Cox has exp(coef) in col 2 — that's
  # wrong, but we never reach this branch because "se(coef)" is the
  # only hit on coxph).
  se_col  <- if ("Std. Error" %in% ce_cols) "Std. Error" else
             if ("se(coef)"   %in% ce_cols) "se(coef)"   else
             ce_cols[2]
  # Test-stat column: t value (lm/fixest), z value (glm), z (coxph).
  stat_col <- if ("t value" %in% ce_cols) "t value" else
              if ("z value" %in% ce_cols) "z value" else
              if ("z"       %in% ce_cols) "z"       else
              if (ncol(ce) >= 3) ce_cols[3] else NA_character_
  # p-value column: prefer named matches, then fall back to the LAST
  # column when there are at least 4 columns (Cox has 5 with
  # Pr(>|z|) at position 5; lm/glm have 4 with Pr at position 4).
  # ``lmer`` without ``lmerTest`` emits only 3 columns (no p-value
  # at all) — the unconditional ``ce_cols[ncol(ce)]`` fallback used
  # to land on "t value" and mis-stamp it as p_values. The
  # ncol >= 4 guard refuses the fallback for the 3-column shape so
  # the helper omits p_values rather than misreports them.
  p_col <- if ("Pr(>|t|)" %in% ce_cols) "Pr(>|t|)" else
           if ("Pr(>|z|)" %in% ce_cols) "Pr(>|z|)" else
           if (ncol(ce) >= 4) ce_cols[ncol(ce)] else NA_character_

  coefs <- as.list(ce[, est_col]); names(coefs) <- rownames(ce)
  ses   <- as.list(ce[, se_col]);  names(ses)   <- rownames(ce)
  tvals <- if (!is.na(stat_col)) {
    v <- as.list(ce[, stat_col]); names(v) <- rownames(ce); v
  } else NULL
  pvals <- if (!is.na(p_col)) {
    v <- as.list(ce[, p_col]); names(v) <- rownames(ce); v
  } else NULL

  # Response / predictors. For Cox the LHS is Surv(time, event) — a
  # call, not a symbol. ``all.vars()`` returns the variable names in
  # order, so all.vars(Surv(t_obs, cens))[1] = "t_obs" is the time
  # variable, which is the right thing to report as the response.
  lhs <- attr(terms(model), "variables")[[2]]
  response <- all.vars(lhs)[1]
  if (is_mixed) {
    # ``term.labels`` for a merMod includes the random-effect
    # grouping factors (e.g. "school" in ``y ~ x + (1 | school)``)
    # alongside the fixed-effect predictors. fixef() returns just
    # the fixed-effect coefficients; their names are the predictor
    # surface the model thinks of as the "regressors of interest".
    fe_names <- tryCatch(names(lme4::fixef(model)),
                         error = function(e) character(0))
    predictors <- as.list(fe_names[!fe_names %in%
                                   c("(Intercept)", "intercept", "const")])
  } else {
    predictors <- as.list(as.character(attr(terms(model), "term.labels")))
  }

  # Sample size: nobs(coxph) returns the number of events, not records.
  # m$n is records; m$nevent is failures. Use m$n for n on Cox so the
  # SDC min-N gate sees the sample size, not the event count.
  n_val <- if (is_cox) as.integer(model$n) else as.integer(nobs(model))

  # Aggregate diagnostics — collinearity / numerical stability. Pure
  # aggregates over the design matrix, no per-row leak. Failures are
  # silent — the field is omitted rather than blowing up the emit.
  vif_list    <- tryCatch(nora$.compute_vif(model),               error = function(e) NULL)
  cond_num    <- tryCatch(nora$.compute_condition_number(model),  error = function(e) NULL)
  vcov_nested <- tryCatch(nora$.compute_vcov(model),              error = function(e) NULL)

  args <- list(
    # ``coefficient_table_with_fit_stats`` is the canonical bucket
    # name (covers OLS / glm / coxph / fixest — anything that emits
    # a coefficient table). ``linear_regression`` is kept as a
    # legacy alias in the sanitizer's dispatch table for back-compat
    # with stored payloads; new emissions use the descriptive name
    # so the model sees a type that matches what's in the bucket.
    type = "coefficient_table_with_fit_stats",
    n = n_val,
    response_variable = response,
    predictor_variables = predictors,
    coefficients = coefs,
    standard_errors = ses,
    t_statistics = tvals,
    p_values = pvals
  )

  # Estimator-appropriate fit metrics. Each branch only emits fields
  # that are meaningful for its class — emitting r_squared on a glm
  # fit (where summary()$r.squared is NULL) used to ride along as a
  # null and force the sanitizer to drop it with a transformation
  # note on every GLM payload.
  if (is_lm) {
    args$r_squared          <- s$r.squared
    args$adj_r_squared      <- s$adj.r.squared
    args$residual_std_error <- s$sigma
    args$degrees_of_freedom <- as.integer(s$df[2])
    if (!is.null(s$fstatistic)) {
      args$f_statistic <- unname(s$fstatistic["value"])
      args$f_p_value   <- unname(pf(s$fstatistic["value"],
                                    s$fstatistic["numdf"],
                                    s$fstatistic["dendf"],
                                    lower.tail = FALSE))
    }
  }
  if (is_glm) {
    # McFadden-equivalent for GLM: 1 - residual_deviance/null_deviance.
    # Deviance ratio matches McFadden when the link is canonical; for
    # non-canonical links it's the conventional GLM pseudo-R² reported
    # by Stata's ``glm`` and statsmodels' ``.prsquared``.
    if (!is.null(model$null.deviance) && !is.null(model$deviance) &&
        is.finite(model$null.deviance) && is.finite(model$deviance) &&
        model$null.deviance > 0) {
      args$pseudo_r_squared <- 1 - model$deviance / model$null.deviance
      chi2 <- model$null.deviance - model$deviance
      df_chi <- tryCatch(
        as.integer(model$df.null - model$df.residual),
        error = function(e) NA_integer_
      )
      if (!is.na(df_chi) && df_chi > 0 && chi2 >= 0) {
        args$chi_squared <- as.numeric(chi2)
        args$chi_squared_p_value <- as.numeric(
          pchisq(chi2, df = df_chi, lower.tail = FALSE)
        )
      }
    }
    if (!is.null(model$df.residual)) {
      args$degrees_of_freedom <- as.integer(model$df.residual)
    }
    ll <- tryCatch(as.numeric(logLik(model)), error = function(e) NULL)
    if (!is.null(ll) && is.finite(ll)) args$log_likelihood <- ll
    aic_v <- tryCatch(as.numeric(AIC(model)), error = function(e) NULL)
    if (!is.null(aic_v) && is.finite(aic_v)) args$aic <- aic_v
    bic_v <- tryCatch(as.numeric(BIC(model)), error = function(e) NULL)
    if (!is.null(bic_v) && is.finite(bic_v)) args$bic <- bic_v
  }
  if (is_cox) {
    # Cox PH: subject + failure counts, Harrell's C, LR test, log-lik.
    if (!is.null(model$n))      args$n_subjects <- as.integer(model$n)
    if (!is.null(model$nevent)) args$n_failures <- as.integer(model$nevent)
    cs <- tryCatch(s$concordance, error = function(e) NULL)
    if (!is.null(cs) && length(cs) >= 1 && is.finite(cs[1])) {
      args$concordance <- as.numeric(cs[1])
    }
    lr <- tryCatch(s$logtest, error = function(e) NULL)
    if (!is.null(lr) && length(lr) >= 3 &&
        is.finite(lr["test"]) && is.finite(lr["pvalue"])) {
      args$chi_squared <- as.numeric(lr["test"])
      args$chi_squared_p_value <- as.numeric(lr["pvalue"])
    }
    ll <- tryCatch(as.numeric(logLik(model)), error = function(e) NULL)
    if (!is.null(ll) && is.finite(ll)) args$log_likelihood <- ll
    aic_v <- tryCatch(as.numeric(AIC(model)), error = function(e) NULL)
    if (!is.null(aic_v) && is.finite(aic_v)) args$aic <- aic_v
    bic_v <- tryCatch(as.numeric(BIC(model)), error = function(e) NULL)
    if (!is.null(bic_v) && is.finite(bic_v)) args$bic <- bic_v
  }
  if (is_mixed) {
    # lme4::lmer / glmer / nlmer. Fixed-effects coefficient table is
    # already extracted via the standard ``summary(m)$coefficients``
    # path (Estimate / Std. Error / t-or-z / optional p). What's
    # specific to mixed models: variance components, per-level
    # group counts, REML vs ML fit method, ICC for one-level fits.
    #
    # Variance components live in ``VarCorr(model)`` — a list of
    # per-group covariance matrices plus an ``sc`` attribute for
    # residual SD. Each matrix's diagonal entries are variances
    # (random-intercept variance, random-slope variance). We emit
    # *variances* — sqrt'd values would duplicate signal and the
    # intercept-slope covariance stays inside ``vcov`` for callers
    # that genuinely need it.
    vc <- tryCatch(lme4::VarCorr(model), error = function(e) NULL)
    re_var <- list()
    if (!is.null(vc)) {
      for (grp_name in names(vc)) {
        mat <- vc[[grp_name]]
        if (!is.matrix(mat)) next
        rn <- rownames(mat)
        if (is.null(rn)) next
        for (i in seq_along(rn)) {
          v <- as.numeric(mat[i, i])
          if (!is.finite(v) || v < 0) next
          key <- if (rn[i] %in% c("(Intercept)", "intercept"))
                    grp_name else paste0(grp_name, ".", rn[i])
          re_var[[key]] <- v
        }
      }
      sc <- attr(vc, "sc")
      if (!is.null(sc) && length(sc) == 1 && is.finite(sc) && sc >= 0) {
        re_var[["residual"]] <- as.numeric(sc^2)
      }
    }
    if (length(re_var) > 0) args$random_effects_variance <- re_var

    # Per-level group counts. lme4::ngrps() returns a named integer
    # vector keyed by grouping-factor name. Same disclosure profile
    # as ``fixed_effects`` and ``n_clusters`` — column name +
    # cardinality, no level identities.
    ng <- tryCatch(lme4::ngrps(model), error = function(e) NULL)
    if (!is.null(ng) && length(ng) > 0) {
      ng_dict <- list()
      for (i in seq_along(ng)) {
        ng_dict[[names(ng)[i]]] <- as.integer(ng[i])
      }
      args$n_groups_per_level <- ng_dict
    }

    # Fit method. ``isREML(m)`` for lmer, FALSE for glmer (always ML).
    # ``getME(model, "is_REML")`` works across the merMod hierarchy.
    is_reml <- tryCatch(lme4::isREML(model), error = function(e) NULL)
    if (!is.null(is_reml)) {
      args$fit_method <- if (isTRUE(is_reml)) "REML" else "ML"
    }

    # Intraclass correlation — only well-defined for one-grouping,
    # intercept-only random-effect specifications. Compute as
    # sigma_u² / (sigma_u² + sigma_e²) when there's exactly one
    # group + a residual term in ``random_effects_variance``.
    if (length(re_var) == 2 && "residual" %in% names(re_var)) {
      grp_var_name <- setdiff(names(re_var), "residual")
      if (length(grp_var_name) == 1) {
        s_u2 <- re_var[[grp_var_name]]
        s_e2 <- re_var[["residual"]]
        if (is.finite(s_u2) && is.finite(s_e2) && (s_u2 + s_e2) > 0) {
          args$icc <- as.numeric(s_u2 / (s_u2 + s_e2))
        }
      }
    }

    ll <- tryCatch(as.numeric(logLik(model)), error = function(e) NULL)
    if (!is.null(ll) && is.finite(ll)) args$log_likelihood <- ll
    aic_v <- tryCatch(as.numeric(AIC(model)), error = function(e) NULL)
    if (!is.null(aic_v) && is.finite(aic_v)) args$aic <- aic_v
    bic_v <- tryCatch(as.numeric(BIC(model)), error = function(e) NULL)
    if (!is.null(bic_v) && is.finite(bic_v)) args$bic <- bic_v
    n_obs <- tryCatch(as.integer(nobs(model)), error = function(e) NULL)
    if (!is.null(n_obs)) args$n <- n_obs
    df_resid <- tryCatch(as.integer(df.residual(model)), error = function(e) NULL)
    if (!is.null(df_resid) && !is.na(df_resid)) args$degrees_of_freedom <- df_resid
  }
  if (is_fixest) {
    # fixest::feols (and family). Coefficients and SE columns are
    # already extracted above; here we add fit metrics, absorbed-FE
    # cardinality, and cluster-robust metadata. Critically: emit
    # FE-dimension SIZES, never the level identifiers themselves —
    # listing the 1,247 firms is not OK, reporting "firm FE absorbed,
    # 1,247 levels" is. Same rule for cluster cardinalities.
    r2 <- tryCatch(
      fixest::fitstat(model, type = "r2", verbose = FALSE)$r2,
      error = function(e) NULL
    )
    if (!is.null(r2) && is.finite(r2)) args$r_squared <- as.numeric(r2)
    ar2 <- tryCatch(
      fixest::fitstat(model, type = "ar2", verbose = FALSE)$ar2,
      error = function(e) NULL
    )
    if (!is.null(ar2) && is.finite(ar2)) args$adj_r_squared <- as.numeric(ar2)
    ll <- tryCatch(as.numeric(logLik(model)), error = function(e) NULL)
    if (!is.null(ll) && is.finite(ll)) args$log_likelihood <- ll
    aic_v <- tryCatch(as.numeric(AIC(model)), error = function(e) NULL)
    if (!is.null(aic_v) && is.finite(aic_v)) args$aic <- aic_v
    bic_v <- tryCatch(as.numeric(BIC(model)), error = function(e) NULL)
    if (!is.null(bic_v) && is.finite(bic_v)) args$bic <- bic_v
    fv <- model$fixef_vars
    fs <- model$fixef_sizes
    if (!is.null(fv) && !is.null(fs) && length(fv) == length(fs) && length(fv) > 0) {
      fe_summary <- list()
      for (i in seq_along(fv)) {
        fe_summary[[as.character(fv[i])]] <- as.integer(fs[i])
      }
      args$fixed_effects <- fe_summary
    }
    # Cluster-robust SE metadata. fixest stores the cluster formula
    # in m$call$cluster (e.g. ``~g`` or ``~g + h`` for two-way).
    # ``attr(summary(m)$cov.scaled, "G")`` carries per-dimension
    # cluster counts as an integer vector when single-dim; for
    # multi-way it collapses to a single value (the minimum count)
    # so we can't always recover per-dim counts cleanly. Emit the
    # NAME list always; emit ``n_clusters`` only when the count
    # vector matches the name vector in length (single-dim case).
    cl_call <- tryCatch(model$call$cluster, error = function(e) NULL)
    if (!is.null(cl_call)) {
      cl_names <- tryCatch(all.vars(cl_call), error = function(e) character(0))
      if (length(cl_names) > 0) {
        args$cluster_variables <- as.list(cl_names)
        args$robust_se_type <- "cluster"
        Gvec <- tryCatch(
          attr(s$cov.scaled, "G"),
          error = function(e) NULL
        )
        if (!is.null(Gvec) && length(Gvec) == length(cl_names)) {
          nc <- list()
          for (i in seq_along(cl_names)) {
            nc[[cl_names[i]]] <- as.integer(Gvec[i])
          }
          args$n_clusters <- nc
        }
      }
    }
  }

  if (!is.null(vif_list) && length(vif_list) > 0) args$vif <- vif_list
  if (!is.null(cond_num)) args$condition_number <- cond_num
  if (!is.null(vcov_nested) && length(vcov_nested) > 0) args$vcov <- vcov_nested

  do.call(nora$result, c(args, list(...)))
}


# vcov(model): full variance-covariance matrix of the coefficient
# estimates. Diagonals equal SE^2; off-diagonals enable Wald tests
# / joint significance / linear-combination CIs. Pure aggregate
# from sigma^2 * (X'X)^-1 with no per-row leak. Returns a nested
# named list keyed by coefficient name; ``NULL`` on any error.
nora$.compute_vcov <- function(model) {
  v <- tryCatch(stats::vcov(model), error = function(e) NULL)
  if (is.null(v) || !is.matrix(v)) return(NULL)
  rn <- rownames(v); cn <- colnames(v)
  if (is.null(rn) || is.null(cn)) return(NULL)
  out <- list()
  for (i in seq_along(rn)) {
    inner <- list()
    for (j in seq_along(cn)) {
      val <- v[i, j]
      if (is.finite(val)) inner[[cn[j]]] <- as.numeric(val)
    }
    if (length(inner) > 0) out[[rn[i]]] <- inner
  }
  if (length(out) == 0) NULL else out
}


# VIF per predictor: regress each predictor on the others, return
# 1 / (1 - R^2_aux). The intercept is excluded; perfectly collinear
# predictors are omitted (R^2_aux >= 1) so the caller treats their
# absence as "VIF undefined" rather than emitting Inf.
nora$.compute_vif <- function(model) {
  X <- tryCatch(model.matrix(model), error = function(e) NULL)
  if (is.null(X) || ncol(X) < 2 || nrow(X) < 2) return(NULL)
  cols <- colnames(X)
  intercept_alias <- c("(Intercept)", "intercept", "const")
  drop_intercept <- cols %in% intercept_alias
  out <- list()
  for (i in seq_along(cols)) {
    if (drop_intercept[i]) next
    name <- cols[i]
    xi <- X[, i]
    X_others <- X[, -i, drop = FALSE]
    if (ncol(X_others) == 0) next
    fit_aux <- tryCatch(
      stats::lm.fit(X_others, xi),
      error = function(e) NULL
    )
    if (is.null(fit_aux)) next
    ss_tot <- sum((xi - mean(xi))^2)
    ss_res <- sum(fit_aux$residuals^2)
    if (ss_tot <= 0 || ss_res < 0) next
    r2_aux <- 1 - ss_res / ss_tot
    if (r2_aux >= 1 || r2_aux < 0) next
    out[[name]] <- 1 / (1 - r2_aux)
  }
  if (length(out) == 0) NULL else out
}


# kappa(X): condition number of the design matrix. Higher values flag
# numerical instability that VIF (single-column at a time) can miss
# when collinearity is spread across many predictors.
nora$.compute_condition_number <- function(model) {
  X <- tryCatch(model.matrix(model), error = function(e) NULL)
  if (is.null(X)) return(NULL)
  k <- tryCatch(kappa(X, exact = TRUE), error = function(e) NULL)
  if (is.null(k) || !is.finite(k)) NULL else as.numeric(k)
}


#' From a clustering fit (``stats::kmeans`` or ``stats::hclust``),
#' emit a cluster_analysis payload.
#'
#' Dispatches on class:
#'   * ``kmeans`` — read cluster sizes, centroids, within-SS from
#'     the fit directly; no extra data needed.
#'   * ``hclust`` — hierarchical fits don't store the data or a
#'     cluster assignment (just the dendrogram). The caller must
#'     pass ``data`` (the matrix the dendrogram was built on) and
#'     ``k`` (the cut point), and the helper computes assignments
#'     via ``cutree(fit, k=k)`` plus centroids and within-SS from
#'     the data.
#'
#' DBSCAN / HDBSCAN intentionally aren't supported by this helper
#' — their inference-adequacy story (density parameters, noise-
#' point handling, no centroids by construction) is a separate
#' design pass. The helper raises with a clear pointer to the
#' generic ``nora$result(type="cluster_analysis", method="dbscan",
#' ...)`` path until a dedicated helper ships.
#'
#' Per-observation cluster assignments are NOT emitted on any path
#' — they're per-row data and have no slot on the allowlist. The
#' sanitizer's whole-cluster suppression gate fires on sizes below
#' ``min_n_descriptive`` and per-cluster precision clamping fires
#' on surviving centroids.
#'
#' Examples:
#'   # k-means
#'   m <- kmeans(df[, c("age","income","tenure")], centers = 4, nstart = 10)
#'   nora$from_cluster(m, variables = c("age","income","tenure"),
#'                     label = "customer segmentation")
#'
#'   # Hierarchical (Ward) — data + k required
#'   d <- dist(df[, c("age","income","tenure")])
#'   h <- hclust(d, method = "ward.D2")
#'   nora$from_cluster(h, data = df[, c("age","income","tenure")],
#'                     k = 4,
#'                     variables = c("age","income","tenure"),
#'                     linkage = "ward",
#'                     label = "ward clustering")
nora$from_cluster <- function(fit, variables = NULL, data = NULL,
                              k = NULL, linkage = NULL, label = NULL) {
  if (inherits(fit, "kmeans")) {
    return(nora$.from_kmeans_impl(fit, variables = variables, label = label))
  }
  if (inherits(fit, "hclust")) {
    if (is.null(data) || is.null(k)) {
      stop(
        "nora$from_cluster: hierarchical fits need ``data`` (the matrix ",
        "the dendrogram was built on) and ``k`` (the cut point) — hclust ",
        "doesn't store either."
      )
    }
    return(nora$.from_hclust_impl(fit, data = data, k = k,
                                  variables = variables,
                                  linkage = linkage, label = label))
  }
  if (inherits(fit, "dbscan") || inherits(fit, "hdbscan")) {
    stop(
      "nora$from_cluster: dedicated DBSCAN / HDBSCAN helper not yet ",
      "shipped. Construct the payload via ",
      "``nora$result(type=\"cluster_analysis\", method=\"dbscan\", ",
      "cluster_sizes=..., n_noise_points=..., variables=..., ...)`` ",
      "from the script — the cluster_analysis shape accepts dbscan ",
      "with centroids absent."
    )
  }
  stop(
    "nora$from_cluster: unknown clustering class ", class(fit)[1],
    ". Supported: kmeans, hclust. DBSCAN-family: use generic ",
    "``nora$result(type=\"cluster_analysis\", ...)`` until a ",
    "dedicated helper ships."
  )
}


# kmeans extraction — internal implementation of from_cluster's
# kmeans branch.
nora$.from_kmeans_impl <- function(fit, variables = NULL, label = NULL) {
  print(fit)
  centers <- fit$centers
  if (is.null(centers) || !is.matrix(centers)) {
    stop("nora$from_kmeans: fit$centers is missing or not a matrix")
  }
  n_clusters <- nrow(centers)
  n_features <- ncol(centers)

  # Variable names: prefer centers's column names; else accept the
  # caller's list; else fall back to feature_i.
  if (is.null(variables)) {
    vnames <- colnames(centers)
    if (is.null(vnames) || any(!nzchar(vnames))) {
      variables <- paste0("feature_", seq_len(n_features))
    } else {
      variables <- vnames
    }
  } else {
    variables <- as.character(variables)
    if (length(variables) != n_features) {
      stop(sprintf(
        "nora$from_kmeans: variables has %d entries but kmeans was fit on %d features",
        length(variables), n_features
      ))
    }
  }

  cluster_labels <- paste0("cluster_", seq_len(n_clusters))
  cluster_sizes <- list()
  centroids <- list()
  within_cluster_ss <- list()
  for (i in seq_len(n_clusters)) {
    cl <- cluster_labels[i]
    cluster_sizes[[cl]] <- as.integer(fit$size[i])
    centroid_row <- list()
    for (j in seq_len(n_features)) {
      centroid_row[[variables[j]]] <- as.numeric(centers[i, j])
    }
    centroids[[cl]] <- centroid_row
    if (!is.null(fit$withinss) && length(fit$withinss) >= i) {
      within_cluster_ss[[cl]] <- as.numeric(fit$withinss[i])
    }
  }

  # Total N is the sum of cluster sizes.
  n_obs <- sum(fit$size)
  totss <- fit$totss
  twss <- fit$tot.withinss
  bss <- fit$betweenss

  args <- list(
    type = "cluster_analysis",
    method = "kmeans",
    distance_metric = "euclidean",
    n_observations = as.integer(n_obs),
    n_clusters = as.integer(n_clusters),
    n_features = as.integer(n_features),
    variables = as.list(variables),
    cluster_labels = as.list(cluster_labels),
    cluster_sizes = cluster_sizes,
    centroids = centroids
  )
  if (length(within_cluster_ss) > 0) args$within_cluster_ss <- within_cluster_ss
  if (!is.null(twss) && is.finite(twss)) args$total_within_ss <- as.numeric(twss)
  if (!is.null(twss) && is.finite(twss)) args$inertia <- as.numeric(twss)
  if (!is.null(bss) && is.finite(bss)) args$between_cluster_ss <- as.numeric(bss)
  if (!is.null(totss) && is.finite(totss)) args$total_ss <- as.numeric(totss)
  if (!is.null(totss) && is.finite(totss) && totss > 0) {
    args$ss_ratio <- as.numeric(bss / totss)
  }
  if (!is.null(fit$iter) && is.finite(fit$iter)) {
    args$n_iterations <- as.integer(fit$iter)
  }
  if (!is.null(label)) args$label <- as.character(label)
  do.call(nora$result, args)
}


# Hierarchical extraction — internal implementation of from_cluster's
# hclust branch. hclust stores only the dendrogram (merge matrix +
# heights), not the data or any cluster assignment. The caller
# passes ``data`` (the matrix the dendrogram was built on) and
# ``k`` (the cut point); the helper computes cluster assignments
# via ``cutree(fit, k = k)`` and centroids + within-SS from the
# data + assignments.
#
# Privacy carve-out: the linkage matrix (``fit$merge``) and merge
# heights (``fit$height``) are NOT emitted — they're per-merge
# records over the data, structurally absent from the
# cluster_analysis allowlist. The dendrogram lives on the
# researcher's local R session.
nora$.from_hclust_impl <- function(fit, data, k, variables = NULL,
                                   linkage = NULL, label = NULL) {
  if (!is.numeric(k) || length(k) != 1 || k < 2 || k != as.integer(k)) {
    stop("nora$from_cluster: ``k`` must be a positive integer >= 2")
  }
  data <- as.matrix(data)
  if (nrow(data) < 2 || ncol(data) < 1) {
    stop("nora$from_cluster: ``data`` must be a non-degenerate matrix")
  }
  k <- as.integer(k)
  if (is.null(variables)) {
    vnames <- colnames(data)
    if (is.null(vnames) || any(!nzchar(vnames))) {
      variables <- paste0("feature_", seq_len(ncol(data)))
    } else {
      variables <- vnames
    }
  } else {
    variables <- as.character(variables)
    if (length(variables) != ncol(data)) {
      stop(sprintf(
        "nora$from_cluster: variables has %d entries but data has %d columns",
        length(variables), ncol(data)
      ))
    }
  }
  print(fit)

  # cutree returns an integer vector of length nrow(data) with
  # cluster ids 1..k. NEVER emitted — per-observation assignments
  # are structurally absent from the allowlist.
  assignments <- stats::cutree(fit, k = k)
  cluster_labels <- paste0("cluster_", seq_len(k))
  cluster_sizes <- list()
  centroids <- list()
  within_cluster_ss <- list()
  total_within_ss <- 0
  grand_mean <- colMeans(data)
  for (i in seq_len(k)) {
    cl <- cluster_labels[i]
    members <- which(assignments == i)
    n_i <- length(members)
    cluster_sizes[[cl]] <- as.integer(n_i)
    if (n_i == 0) next
    sub <- data[members, , drop = FALSE]
    centroid_row <- list()
    centroid_vec <- colMeans(sub)
    for (j in seq_len(ncol(data))) {
      centroid_row[[variables[j]]] <- as.numeric(centroid_vec[j])
    }
    centroids[[cl]] <- centroid_row
    # within-cluster SS for cluster i = sum over members of
    # ||x - centroid||² (squared Euclidean distance).
    deviations <- sweep(sub, 2, centroid_vec, FUN = "-")
    wss_i <- sum(deviations^2)
    within_cluster_ss[[cl]] <- as.numeric(wss_i)
    total_within_ss <- total_within_ss + wss_i
  }
  # Total SS (centered at grand mean); between-cluster SS = total - within.
  total_ss <- sum(sweep(data, 2, grand_mean, FUN = "-")^2)
  between_ss <- total_ss - total_within_ss

  # Cut height: the merge height at which exactly k clusters
  # remain. hclust's heights are in fit$height (length n-1).
  cut_height <- NA_real_
  if (!is.null(fit$height) && length(fit$height) >= (nrow(data) - k)) {
    # The cut for k clusters is between the (n-k)-th and (n-k+1)-th
    # merge heights; report the (n-k+1)-th as the threshold height
    # (the height ABOVE which only k clusters remain).
    idx <- length(fit$height) - k + 1
    if (idx >= 1 && idx <= length(fit$height)) {
      cut_height <- fit$height[idx]
    }
  }

  args <- list(
    type = "cluster_analysis",
    method = "hierarchical",
    distance_metric = if (!is.null(fit$dist.method)) fit$dist.method else "euclidean",
    n_observations = as.integer(nrow(data)),
    n_clusters = k,
    n_features = as.integer(ncol(data)),
    variables = as.list(variables),
    cluster_labels = as.list(cluster_labels),
    cluster_sizes = cluster_sizes,
    centroids = centroids
  )
  if (length(within_cluster_ss) > 0) {
    args$within_cluster_ss <- within_cluster_ss
  }
  args$total_within_ss <- as.numeric(total_within_ss)
  args$inertia <- as.numeric(total_within_ss)
  args$between_cluster_ss <- as.numeric(between_ss)
  args$total_ss <- as.numeric(total_ss)
  if (total_ss > 0) {
    args$ss_ratio <- as.numeric(between_ss / total_ss)
  }
  # Linkage method: prefer caller's argument; else read from the
  # fit's ``method`` slot (hclust stores ``"ward.D"`` /
  # ``"ward.D2"`` / ``"complete"`` / ``"average"`` / ``"single"`` /
  # ``"centroid"`` / ``"median"``). Normalize ward.D / ward.D2 to
  # ``"ward"`` since the sanitizer's enum doesn't distinguish.
  if (is.null(linkage)) linkage <- fit$method
  if (!is.null(linkage)) {
    linkage <- as.character(linkage)
    if (grepl("^ward", linkage)) linkage <- "ward"
    args$linkage <- linkage
  }
  if (is.finite(cut_height)) args$cut_height <- as.numeric(cut_height)
  if (!is.null(label)) args$label <- as.character(label)
  do.call(nora$result, args)
}


# Back-compat: ``nora$from_kmeans`` was the public name in earlier
# releases. The class-dispatched ``from_cluster`` is the new
# canonical entry point. Keep both during the transition.
nora$from_kmeans <- function(fit, variables = NULL, label = NULL) {
  nora$from_cluster(fit, variables = variables, label = label)
}


#' From a ``stats::prcomp`` fit, emit a factor_decomposition payload.
#'
#' Wraps base R's ``prcomp`` (eigen-decomposition of the centered
#' and optionally scaled data matrix). The payload carries the
#' loadings matrix (variable × component), explained-variance
#' ratios, cumulative variance, eigenvalues, and PCA-derived
#' communalities. The full row × component factor-scores matrix
#' (``fit$x``) is researcher-only by construction — no field in
#' the sanitizer's ``factor_decomposition`` allowlist accepts it.
#'
#' By default we report all components prcomp computed (one per
#' input variable). The ``n_components`` argument trims to the top-k
#' for parsimony; the dropped components remain researcher-visible
#' on the original fit object.
#'
#' Example:
#'   m <- prcomp(df[, c("v1","v2","v3","v4","v5")], scale. = TRUE)
#'   nora$from_pca(m, label = "five-variable PCA")
#'
#' To trim to the top three components:
#'   nora$from_pca(m, n_components = 3)
nora$from_pca <- function(fit, n_components = NULL, label = NULL) {
  if (!inherits(fit, "prcomp")) {
    stop("nora$from_pca: ``fit`` must be a prcomp object.")
  }
  print(fit)
  rotation <- fit$rotation
  if (is.null(rotation) || !is.matrix(rotation)) {
    stop("nora$from_pca: fit$rotation is missing or not a matrix")
  }
  variables <- rownames(rotation)
  comp_labels_full <- colnames(rotation)
  total_k <- ncol(rotation)
  if (is.null(n_components)) n_components <- total_k
  n_components <- as.integer(min(n_components, total_k))
  comp_labels <- comp_labels_full[seq_len(n_components)]

  # nobs(fit) is the post-fit sample size. ``fit$x`` (the scores
  # matrix) carries it row-wise; we never emit ``$x``, only the
  # row count.
  n_obs <- if (!is.null(fit$x)) nrow(fit$x) else NA_integer_

  # Build loadings as {variable: {component: value}}.
  loadings <- list()
  for (v in variables) {
    row <- list()
    for (i in seq_len(n_components)) {
      row[[comp_labels[i]]] <- as.numeric(rotation[v, i])
    }
    loadings[[v]] <- row
  }

  # Explained variance: sdev² is the variance per component (the
  # eigenvalues when scale.=TRUE, since we work on the correlation
  # matrix). Ratio = eigenvalue / sum(eigenvalues). Cumulative is
  # the running sum.
  eigenvalues_full <- as.numeric(fit$sdev^2)
  total_var <- sum(eigenvalues_full)
  ev_ratio_full <- eigenvalues_full / total_var
  cum_var_full <- cumsum(ev_ratio_full)

  eigenvalues <- list()
  explained_variance <- list()
  explained_variance_ratio <- list()
  cumulative_variance <- list()
  for (i in seq_len(n_components)) {
    eigenvalues[[comp_labels[i]]] <- as.numeric(eigenvalues_full[i])
    explained_variance[[comp_labels[i]]] <- as.numeric(eigenvalues_full[i])
    explained_variance_ratio[[comp_labels[i]]] <- as.numeric(ev_ratio_full[i])
    cumulative_variance[[comp_labels[i]]] <- as.numeric(cum_var_full[i])
  }

  # Communalities: sum of squared loadings across the retained
  # components, per variable. = 1 when all components retained.
  communalities <- list()
  for (v in variables) {
    h2 <- sum(rotation[v, seq_len(n_components)]^2)
    communalities[[v]] <- as.numeric(h2)
  }

  args <- list(
    type = "factor_decomposition",
    method = "pca",
    rotation = "none",
    n_observations = as.integer(n_obs),
    n_variables = as.integer(length(variables)),
    n_components = n_components,
    variables = as.list(variables),
    components = as.list(comp_labels),
    loadings = loadings,
    explained_variance = explained_variance,
    explained_variance_ratio = explained_variance_ratio,
    cumulative_variance = cumulative_variance,
    eigenvalues = eigenvalues,
    communalities = communalities
  )
  if (!is.null(label)) args$label <- as.character(label)
  do.call(nora$result, args)
}


#' From a Sun-Abraham interaction-weighted event-study fit, emit a
#' did_event_study payload.
#'
#' Wraps ``fixest::feols()`` with ``fixest::sunab(cohort, time)`` in
#' the formula — the interaction-weighted estimator from Sun &
#' Abraham (2021). The estimator's natural output is one ATT per
#' event-time (already aggregated across cohorts via IW weights),
#' not per (cohort, event-time). We package this as a
#' did_event_study payload with a single synthetic cohort ``"all"``
#' whose ATT series IS the event-time aggregate; the
#' ``estimator: "sun_abraham"`` field tells the model the
#' aggregation happened inside the estimator.
#'
#' The caller passes ``n_treated`` (total treated units across the
#' cohorts that fed the IW weights) so the cohort-N gate has its
#' input. The helper can't recover this from the feols result
#' without re-walking the data — better to make it explicit.
#'
#' Example:
#'   m <- feols(y ~ sunab(cohort, period) | id + period,
#'              data = df, cluster = ~id)
#'   n_treated <- length(unique(df$id[df$cohort <= max(df$period)]))
#'   nora$from_sun_abraham(m, n_treated = n_treated,
#'                         outcome_variable = "y",
#'                         treatment_variable = "cohort")
nora$from_sun_abraham <- function(fit,
                                  n_treated,
                                  outcome_variable = NULL,
                                  treatment_variable = NULL,
                                  label = NULL,
                                  event_time_pattern = "(period|event_time|rel_time|et)::([^:]+)") {
  if (!inherits(fit, "fixest")) {
    stop("nora$from_sun_abraham: ``fit`` must be a fixest::feols result")
  }
  if (missing(n_treated) || !is.numeric(n_treated) || n_treated < 0) {
    stop("nora$from_sun_abraham: ``n_treated`` (total treated units) is required")
  }
  print(fit)

  # fixest's aggregate() collapses the cohort-by-event-time
  # interactions to per-event-time ATTs via IW weights. The
  # pattern argument captures the event-time portion of the
  # coefficient name (sunab produces names like "period::-3").
  agg <- tryCatch(
    aggregate(fit, event_time_pattern),
    error = function(e) {
      stop("nora$from_sun_abraham: aggregate(fit, ...) failed — was the fit ",
           "produced via feols(y ~ sunab(cohort, time) | ...)? ",
           "Error: ", conditionMessage(e))
    }
  )
  if (is.null(agg) || nrow(agg) == 0) {
    stop("nora$from_sun_abraham: aggregated coefficient table is empty")
  }

  # Extract event-time integer from each coefficient name.
  coef_names <- rownames(agg)
  m_extract <- regmatches(coef_names, regexec(event_time_pattern, coef_names))
  # Use the LAST capture group as the event-time integer, regardless
  # of how many groups the user supplied (default pattern has 2 groups
  # — prefix + integer; caller may pass a pattern with 1 group).
  # ``unname()`` strips any names ``sapply`` carries through from the
  # matched coefficient names — without it, ``as.list(event_times)``
  # would build a NAMED list, which the JSON serializer emits as an
  # OBJECT, breaking the sanitizer's "event_times must be a list of
  # finite numbers" check.
  event_times <- unname(sapply(m_extract, function(x) {
    if (length(x) >= 2) as.integer(x[length(x)]) else NA_integer_
  }))
  keep <- !is.na(event_times)
  if (!any(keep)) {
    stop("nora$from_sun_abraham: no event-time coefficients matched pattern")
  }
  event_times <- event_times[keep]
  agg <- agg[keep, , drop = FALSE]

  att_all <- list()
  se_all  <- list()
  p_all   <- list()
  ci_lo   <- list()
  ci_hi   <- list()
  for (i in seq_along(event_times)) {
    et_lab <- as.character(event_times[i])
    est <- as.numeric(agg[i, "Estimate"])
    se  <- as.numeric(agg[i, "Std. Error"])
    pv  <- as.numeric(agg[i, "Pr(>|t|)"])
    att_all[[et_lab]] <- est
    se_all[[et_lab]]  <- se
    p_all[[et_lab]]   <- pv
    if (is.finite(est) && is.finite(se) && se > 0) {
      ci_lo[[et_lab]] <- est - 1.96 * se
      ci_hi[[et_lab]] <- est + 1.96 * se
    }
  }

  args <- list(
    type = "did_event_study",
    estimator = "sun_abraham",
    aggregation_method = "dynamic",
    groups = list("all"),
    event_times = as.list(sort(event_times)),
    att = list(all = att_all),
    standard_errors = list(all = se_all),
    p_values = list(all = p_all),
    ci_lower = list(all = ci_lo),
    ci_upper = list(all = ci_hi),
    n_treated_per_group = list(all = as.integer(n_treated))
  )
  if (!is.null(outcome_variable))   args$outcome_variable   <- as.character(outcome_variable)
  if (!is.null(treatment_variable)) args$treatment_variable <- as.character(treatment_variable)
  if (!is.null(label))              args$label              <- as.character(label)
  do.call(nora$result, args)
}


#' From a TWFE event-study regression (any feols / lm with i(rel_time,
#' treated, ref=0) style interactions), emit a did_event_study payload.
#'
#' Differs from Sun-Abraham only in the estimator label and the
#' identification assumptions the model needs to know about
#' (TWFE-ES is biased under treatment-effect heterogeneity; the
#' Sun-Abraham IW estimator is the heterogeneity-robust version).
#' Same single-synthetic-cohort payload shape.
#'
#' Caller passes ``event_time_pattern`` matching the coefficient
#' names (regex with one capture group for the event-time integer).
#' Default matches ``rel_time::N`` / ``event_time::N`` / ``period::N``.
#'
#' Example:
#'   m <- feols(y ~ i(rel_time, treated, ref=-1) | id + period, data = df)
#'   nora$from_twfe_event_study(m, n_treated = sum(df$treated > 0),
#'                              outcome_variable = "y",
#'                              event_time_pattern = "rel_time::([^:]+):")
nora$from_twfe_event_study <- function(fit,
                                       n_treated,
                                       outcome_variable = NULL,
                                       treatment_variable = NULL,
                                       label = NULL,
                                       event_time_pattern = "(rel_time|event_time|period|et)::([^:]+)") {
  if (!inherits(fit, "fixest") && !inherits(fit, "lm")) {
    stop("nora$from_twfe_event_study: ``fit`` must be feols or lm")
  }
  if (missing(n_treated) || !is.numeric(n_treated) || n_treated < 0) {
    stop("nora$from_twfe_event_study: ``n_treated`` is required")
  }
  print(fit)

  ct <- tryCatch(
    if (inherits(fit, "fixest")) coeftable(fit) else as.data.frame(summary(fit)$coefficients),
    error = function(e) stop("nora$from_twfe_event_study: coefficient table unreachable: ", conditionMessage(e))
  )
  if (is.null(ct) || nrow(ct) == 0) {
    stop("nora$from_twfe_event_study: empty coefficient table")
  }

  coef_names <- rownames(ct)
  m_extract <- regmatches(coef_names, regexec(event_time_pattern, coef_names))
  # Use the LAST capture group as the event-time integer, regardless
  # of how many groups the user supplied (default pattern has 2 groups
  # — prefix + integer; caller may pass a pattern with 1 group).
  # ``unname()`` strips any names ``sapply`` carries through from the
  # matched coefficient names — without it, ``as.list(event_times)``
  # would build a NAMED list, which the JSON serializer emits as an
  # OBJECT, breaking the sanitizer's "event_times must be a list of
  # finite numbers" check.
  event_times <- unname(sapply(m_extract, function(x) {
    if (length(x) >= 2) as.integer(x[length(x)]) else NA_integer_
  }))
  keep <- !is.na(event_times)
  if (!any(keep)) {
    stop("nora$from_twfe_event_study: no event-time coefficients matched ",
         "the pattern. The default pattern matches rel_time::N / ",
         "event_time::N / period::N — pass ``event_time_pattern`` if ",
         "your design uses a different naming convention.")
  }
  event_times <- event_times[keep]
  ct <- ct[keep, , drop = FALSE]

  ce_cols <- colnames(ct)
  est_col <- if ("Estimate" %in% ce_cols) "Estimate" else ce_cols[1]
  se_col  <- if ("Std. Error" %in% ce_cols) "Std. Error" else ce_cols[2]
  p_col   <- if ("Pr(>|t|)" %in% ce_cols) "Pr(>|t|)" else
             if ("Pr(>|z|)" %in% ce_cols) "Pr(>|z|)" else
             ce_cols[ncol(ct)]

  att_all <- list(); se_all <- list(); p_all <- list()
  ci_lo <- list(); ci_hi <- list()
  for (i in seq_along(event_times)) {
    et_lab <- as.character(event_times[i])
    est <- as.numeric(ct[i, est_col])
    se  <- as.numeric(ct[i, se_col])
    pv  <- as.numeric(ct[i, p_col])
    att_all[[et_lab]] <- est
    se_all[[et_lab]]  <- se
    p_all[[et_lab]]   <- pv
    if (is.finite(est) && is.finite(se) && se > 0) {
      ci_lo[[et_lab]] <- est - 1.96 * se
      ci_hi[[et_lab]] <- est + 1.96 * se
    }
  }

  args <- list(
    type = "did_event_study",
    estimator = "twfe_event_study",
    aggregation_method = "dynamic",
    groups = list("all"),
    event_times = as.list(sort(event_times)),
    att = list(all = att_all),
    standard_errors = list(all = se_all),
    p_values = list(all = p_all),
    ci_lower = list(all = ci_lo),
    ci_upper = list(all = ci_hi),
    n_treated_per_group = list(all = as.integer(n_treated))
  )
  if (!is.null(outcome_variable))   args$outcome_variable   <- as.character(outcome_variable)
  if (!is.null(treatment_variable)) args$treatment_variable <- as.character(treatment_variable)
  if (!is.null(label))              args$label              <- as.character(label)
  do.call(nora$result, args)
}


#' From a ``did::att_gt`` MP object, emit a did_event_study payload.
#'
#' Wraps the Callaway-Sant'Anna heterogeneous-treatment DiD estimator.
#' The MP object carries pre-aggregation ATT(g, t) — one estimate per
#' (cohort g, calendar time t). The helper:
#'   * Pivots ATT(g, t) → ATT(g, event_time) where event_time = t - g
#'   * Pulls per-cohort treated counts from
#'     ``mp$DIDparams$cohort_counts`` (data.table with cohort + size)
#'   * Optionally runs ``aggte(mp, type="dynamic")`` to add the
#'     aggregate ATT and event-time-aggregated series
#'   * Tags ``estimator: "callaway_santanna"``
#'
#' Privacy carve-out: the sanitizer's cohort-N gate fires on
#' ``n_treated_per_group``. Cohorts below ``min_n_did_cohort`` are
#' dropped *whole* (entire cohort row stripped from the ATT matrix);
#' partial-cell publication would leak the cohort size through
#' which cells survived. The helper emits the raw cohort sizes
#' unchanged — the suppression decision belongs to the sanitizer.
#'
#' Example:
#'   mp <- att_gt(yname = "y", tname = "period", idname = "id",
#'                gname = "G", data = df,
#'                control_group = "nevertreated")
#'   nora$from_callaway_santanna(mp,
#'     outcome_variable = "y", treatment_variable = "G",
#'     label = "headline DiD")
nora$from_callaway_santanna <- function(mp,
                                        outcome_variable = NULL,
                                        treatment_variable = NULL,
                                        aggregation_method = "dynamic",
                                        label = NULL, ...) {
  if (!inherits(mp, "MP")) {
    stop(
      "nora$from_callaway_santanna: ``mp`` must be a did::att_gt result ",
      "(an ``MP`` object). Run ``att_gt(...)`` first and pass the result."
    )
  }
  print(mp)

  # Cohort labels (treated cohorts in mp$group; sorted ascending).
  cohorts <- sort(unique(mp$group))
  if (length(cohorts) == 0) {
    stop("nora$from_callaway_santanna: no treated cohorts found in mp$group")
  }
  cohort_labels <- as.character(cohorts)

  # Event-time grid: union of (t - g) across all (g, t) entries.
  event_times_all <- mp$t - mp$group
  event_time_grid <- sort(unique(event_times_all))

  # Build ATT(g, e) and SE(g, e) nested dicts.
  att_dict <- list()
  se_dict <- list()
  ci_lo_dict <- list()
  ci_hi_dict <- list()
  for (g in cohorts) {
    g_lab <- as.character(g)
    att_dict[[g_lab]] <- list()
    se_dict[[g_lab]] <- list()
    ci_lo_dict[[g_lab]] <- list()
    ci_hi_dict[[g_lab]] <- list()
    idx <- which(mp$group == g)
    # mp$c is the critical value for the CS uniform CI (one scalar);
    # multiply by se to get per-cell CI half-widths.
    crit <- if (!is.null(mp$c) && length(mp$c) >= 1 && is.finite(mp$c[1])) mp$c[1] else 1.96
    for (i in idx) {
      e <- mp$t[i] - mp$group[i]
      e_lab <- as.character(e)
      att_dict[[g_lab]][[e_lab]] <- as.numeric(mp$att[i])
      se_dict[[g_lab]][[e_lab]] <- as.numeric(mp$se[i])
      ci_lo_dict[[g_lab]][[e_lab]] <- as.numeric(mp$att[i] - crit * mp$se[i])
      ci_hi_dict[[g_lab]][[e_lab]] <- as.numeric(mp$att[i] + crit * mp$se[i])
    }
  }

  # Per-cohort treated counts. mp$DIDparams$cohort_counts is a
  # data.table with rows {cohort, cohort_size}. Skip the never-
  # treated row (cohort == Inf) and any cohort not in mp$group.
  cc <- mp$DIDparams$cohort_counts
  n_treated_per_group <- list()
  if (!is.null(cc)) {
    for (i in seq_len(nrow(cc))) {
      g_val <- cc$cohort[i]
      if (is.finite(g_val) && g_val %in% cohorts) {
        n_treated_per_group[[as.character(g_val)]] <-
          as.integer(cc$cohort_size[i])
      }
    }
  }

  args <- list(
    type = "did_event_study",
    estimator = "callaway_santanna",
    groups = as.list(cohort_labels),
    event_times = as.list(event_time_grid),
    att = att_dict,
    standard_errors = se_dict,
    ci_lower = ci_lo_dict,
    ci_upper = ci_hi_dict,
    n_treated_per_group = n_treated_per_group,
    aggregation_method = as.character(aggregation_method)
  )
  if (!is.null(outcome_variable))   args$outcome_variable   <- as.character(outcome_variable)
  if (!is.null(treatment_variable)) args$treatment_variable <- as.character(treatment_variable)
  if (!is.null(label))              args$label              <- as.character(label)

  # Pass through CS config so the model knows the identification
  # assumptions the estimator ran under.
  ctrl_grp <- mp$DIDparams$control_group
  if (!is.null(ctrl_grp) && nzchar(ctrl_grp)) {
    args$comparison_group <- as.character(ctrl_grp)
  }
  antic <- mp$DIDparams$anticipation
  if (!is.null(antic) && length(antic) == 1 && is.finite(antic)) {
    args$anticipation_periods <- as.integer(antic)
  }
  bp <- mp$DIDparams$base_period
  if (!is.null(bp) && nzchar(bp)) {
    args$base_period <- as.character(bp)
  }

  # Aggregate scalars via aggte() — wrap in tryCatch because some
  # data shapes (single cohort, balanced-only requests) can fail
  # inside aggte; omit aggregate fields rather than blowing up the
  # whole emit.
  es <- tryCatch(
    did::aggte(mp, type = aggregation_method),
    error = function(e) NULL
  )
  if (!is.null(es)) {
    if (!is.null(es$overall.att) && is.finite(es$overall.att)) {
      args$aggregate_att <- as.numeric(es$overall.att)
    }
    if (!is.null(es$overall.se) && is.finite(es$overall.se)) {
      args$aggregate_se <- as.numeric(es$overall.se)
      # Two-sided z-test p-value for the aggregate.
      if (es$overall.se > 0) {
        z <- abs(es$overall.att / es$overall.se)
        args$aggregate_p_value <- as.numeric(2 * pnorm(-z))
        crit <- if (!is.null(es$crit.val.egt) && length(es$crit.val.egt) >= 1
                    && is.finite(es$crit.val.egt[1])) es$crit.val.egt[1] else 1.96
        args$aggregate_ci_lower <- as.numeric(es$overall.att - crit * es$overall.se)
        args$aggregate_ci_upper <- as.numeric(es$overall.att + crit * es$overall.se)
      }
    }
  }

  do.call(nora$result, c(args, list(...)))
}


#' From an ``rdrobust`` fit, emit an ``rdd`` payload.
#'
#' Wraps the rdrobust package (CCT 2014) — the standard cross-language
#' implementation maintained by Calonico-Cattaneo-Titiunik. The
#' payload carries the three-flavor τ table (conventional, bias-
#' corrected, robust), bandwidth(s), kernel, polynomial order,
#' effective N per side, and the bandwidth selector name. For fuzzy
#' RDD pass ``fuzzy_treatment_variable`` (the endogenous treatment
#' indicator) so the estimator is tagged ``fuzzy_2sls``.
#'
#' Privacy carve-out is structural: the helper signature does not
#' accept density / binscatter / mccrary arguments at all (not even
#' to drop them). McCrary density tests and binscatter near the
#' cutoff are visual diagnostics for the researcher; they have no
#' field on the ``rdd`` shape's allowlist either, so even hand-
#' crafted payloads through ``nora$result(type="rdd", ...)`` cannot
#' smuggle them.
#'
#' Example:
#'   m <- rdrobust(y, x, c = 50000)
#'   nora$from_rdd(m, running_variable = "income",
#'                 outcome_variable = "voted", label = "headline RDD")
#'
#'   # Fuzzy RDD: pass the treatment-receipt indicator.
#'   m <- rdrobust(y, x, c = 50000, fuzzy = takeup)
#'   nora$from_rdd(m, running_variable = "income",
#'                 outcome_variable = "voted",
#'                 fuzzy_treatment_variable = "takeup")
nora$from_rdd <- function(fit,
                          running_variable = NULL,
                          outcome_variable = NULL,
                          fuzzy_treatment_variable = NULL,
                          first_stage_f = NULL,
                          label = NULL, ...) {
  if (!inherits(fit, "rdrobust")) {
    stop(
      "nora$from_rdd: ``fit`` must be an rdrobust object. ",
      "Use rdrobust::rdrobust(y, x, c = cutoff) and pass the result."
    )
  }
  # Privacy carve-out: refuse density/binscatter arguments if a
  # researcher tries to pass them through ``...`` — these are
  # researcher-only diagnostics for an RDD, not analytical fields.
  extra <- list(...)
  banned <- c("mccrary_density_curve", "mccrary_density",
              "binscatter_bins", "binscatter", "density_curve")
  for (b in banned) {
    if (!is.null(extra[[b]])) {
      stop(
        "nora$from_rdd: ``", b, "`` is a visual diagnostic for the ",
        "researcher and is not allowed on the rdd payload. The model ",
        "sees the analytical fields (tau / bandwidth / effective N); ",
        "ask the researcher qualitatively about manipulation evidence ",
        "if it bears on the design."
      )
    }
  }
  print(fit)

  args <- list(
    type = "rdd",
    estimator = if (!is.null(fuzzy_treatment_variable))
                  "fuzzy_2sls" else "local_polynomial"
  )
  if (!is.null(running_variable))  args$running_variable  <- as.character(running_variable)
  if (!is.null(outcome_variable))  args$outcome_variable  <- as.character(outcome_variable)
  if (!is.null(label))             args$label             <- as.character(label)

  # rdrobust's ``Estimate`` row is [tau.us, tau.bc, se.us, se.rb]:
  #   tau.us = conventional point estimate
  #   tau.bc = bias-corrected point estimate (also used for robust)
  #   se.us  = conventional SE
  #   se.rb  = robust SE
  # rdrobust's ``se`` / ``pv`` / ``ci`` rows map to:
  #   [1] Conventional, [2] Bias-Corrected, [3] Robust
  if (!is.null(fit$Estimate) && is.matrix(fit$Estimate) && ncol(fit$Estimate) >= 4) {
    args$tau_conventional   <- as.numeric(fit$Estimate[1, 1])
    args$tau_bias_corrected <- as.numeric(fit$Estimate[1, 2])
    args$tau_robust         <- as.numeric(fit$Estimate[1, 2])
  }
  if (!is.null(fit$se) && length(fit$se) >= 3) {
    args$se_conventional   <- as.numeric(fit$se[1])
    args$se_bias_corrected <- as.numeric(fit$se[2])
    args$se_robust         <- as.numeric(fit$se[3])
  }
  if (!is.null(fit$pv) && length(fit$pv) >= 3) {
    args$p_conventional   <- as.numeric(fit$pv[1])
    args$p_bias_corrected <- as.numeric(fit$pv[2])
    args$p_robust         <- as.numeric(fit$pv[3])
  }
  if (!is.null(fit$ci) && is.matrix(fit$ci) && nrow(fit$ci) >= 3 && ncol(fit$ci) >= 2) {
    args$ci_lower_conventional   <- as.numeric(fit$ci[1, 1])
    args$ci_upper_conventional   <- as.numeric(fit$ci[1, 2])
    args$ci_lower_bias_corrected <- as.numeric(fit$ci[2, 1])
    args$ci_upper_bias_corrected <- as.numeric(fit$ci[2, 2])
    args$ci_lower_robust         <- as.numeric(fit$ci[3, 1])
    args$ci_upper_robust         <- as.numeric(fit$ci[3, 2])
  }

  # Bandwidth(s): h = main, b = bias-correction. rdrobust stores
  # them as a 2x2 matrix (rows = h/b, cols = left/right).
  if (!is.null(fit$bws) && is.matrix(fit$bws) && nrow(fit$bws) >= 2 && ncol(fit$bws) >= 2) {
    args$bandwidth_left  <- as.numeric(fit$bws[1, 1])
    args$bandwidth_right <- as.numeric(fit$bws[1, 2])
    args$bandwidth_bias_correction_left  <- as.numeric(fit$bws[2, 1])
    args$bandwidth_bias_correction_right <- as.numeric(fit$bws[2, 2])
  }

  # Effective N inside the main bandwidth — the SDC-relevant counts
  # that gate the rdd payload. rdrobust's ``N_h`` is a 2-vector
  # [left, right].
  if (!is.null(fit$N_h) && length(fit$N_h) >= 2) {
    args$effective_n_left  <- as.integer(fit$N_h[1])
    args$effective_n_right <- as.integer(fit$N_h[2])
  }
  args$polynomial_order <- as.integer(fit$p)
  args$cutoff           <- as.numeric(fit$c)

  if (!is.null(fit$bwselect)) {
    args$bandwidth_selector <- as.character(fit$bwselect)
  }
  # rdrobust reports kernel capitalized ("Triangular"); the sanitizer
  # accepts lowercase only.
  if (!is.null(fit$kernel)) {
    args$kernel <- tolower(as.character(fit$kernel))
  }

  # Fuzzy first-stage F. rdrobust doesn't compute this automatically;
  # the caller passes it via the kwarg (compute first-stage F by
  # regressing the endogenous-treatment indicator on the running-
  # variable polynomial inside the bandwidth, then F-test the cutoff
  # dummy).
  if (!is.null(first_stage_f) && is.finite(as.numeric(first_stage_f))) {
    args$first_stage_f <- as.numeric(first_stage_f)
  }

  do.call(nora$result, args)
}


#' From a survival::survfit fit, emit a kaplan_meier payload (safe form).
#'
#' The sanitizer's ``kaplan_meier`` shape ships median survival (with
#' CI) plus survival at preset canonical horizons (``1y`` / ``3y`` /
#' ``5y`` / ``10y``) — each gated by per-horizon ``n_at_risk_h``.
#' The full step function (S(t) at every event time) is researcher-only
#' by construction; this helper does NOT emit it.
#'
#' Time-unit translation is the caller's responsibility. The
#' ``horizons`` argument is a named numeric vector mapping canonical
#' labels to the numeric time in whatever units the fit was built in:
#'
#'   # Data measured in years
#'   nora$from_kaplan_meier(fit, horizons = c("1y" = 1, "3y" = 3, "5y" = 5),
#'                          time_variable = "t_obs", event_variable = "cens")
#'
#'   # Data measured in months
#'   nora$from_kaplan_meier(fit, horizons = c("1y" = 12, "3y" = 36),
#'                          time_variable = "follow_up_months",
#'                          event_variable = "dead")
#'
#' For grouped / log-rank inference, pass the unstratified ``survfit``
#' for the horizon scalars and the ``survdiff`` result separately so
#' the helper can extract the chi² and df.
#'
#'   sd <- survdiff(Surv(t, e) ~ arm, data = df)
#'   nora$from_kaplan_meier(survfit(Surv(t, e) ~ 1, data = df),
#'                          horizons = c("1y" = 1, "3y" = 3),
#'                          time_variable = "t", event_variable = "e",
#'                          group_variable = "arm", survdiff = sd)
nora$from_kaplan_meier <- function(fit, horizons = NULL,
                                   time_variable = NULL,
                                   event_variable = NULL,
                                   group_variable = NULL,
                                   survdiff = NULL,
                                   label = NULL, ...) {
  if (!inherits(fit, "survfit")) {
    stop("nora$from_kaplan_meier: ``fit`` must be a survival::survfit object")
  }
  print(fit)
  if (!is.null(fit$strata)) {
    stop(
      "nora$from_kaplan_meier: ``fit`` is stratified. Fit an UNSTRATIFIED ",
      "survfit (e.g. ``survfit(Surv(t, e) ~ 1, data = df)``) for the ",
      "horizon scalars and pass the ``survdiff`` result separately for ",
      "log-rank inference."
    )
  }

  args <- list(type = "kaplan_meier")
  if (!is.null(time_variable))   args$time_variable   <- as.character(time_variable)
  if (!is.null(event_variable))  args$event_variable  <- as.character(event_variable)
  if (!is.null(group_variable))  args$group_variable  <- as.character(group_variable)
  if (!is.null(label))           args$label           <- as.character(label)

  args$n_subjects <- as.integer(fit$n)
  args$n_failures <- as.integer(sum(fit$n.event))

  # Median + CI. ``quantile.survfit`` returns NA when the curve doesn't
  # cross 0.5 (heavily censored studies); omit those fields gracefully
  # rather than emitting null and provoking a sanitizer transformation
  # note on every KM payload.
  med <- tryCatch(
    quantile(fit, 0.5, conf.int = TRUE),
    error = function(e) NULL
  )
  if (!is.null(med)) {
    mq <- med$quantile; ml <- med$lower; mh <- med$upper
    if (length(mq) >= 1 && is.finite(mq[1])) args$median_survival_time <- as.numeric(mq[1])
    if (length(ml) >= 1 && is.finite(ml[1])) args$median_survival_ci_lower <- as.numeric(ml[1])
    if (length(mh) >= 1 && is.finite(mh[1])) args$median_survival_ci_upper <- as.numeric(mh[1])
  }

  # Per-horizon S(t) and n.risk. ``horizons`` maps canonical labels
  # to numeric time values in the fit's units. ``summary(fit, times=)``
  # with ``extend = TRUE`` ensures lookups past the last event time
  # return NA rather than dropping the row.
  if (!is.null(horizons) && length(horizons) > 0) {
    labels <- names(horizons)
    if (is.null(labels) || any(!nzchar(labels))) {
      stop(
        "nora$from_kaplan_meier: ``horizons`` must be a NAMED vector ",
        "mapping canonical labels (\"1y\", \"3y\", \"5y\", \"10y\") to ",
        "numeric time values in the fit's units. Unnamed entries would ",
        "ship without horizon labels the sanitizer can recognise."
      )
    }
    times_at <- as.numeric(unname(horizons))
    s <- tryCatch(
      summary(fit, times = times_at, extend = TRUE),
      error = function(e) NULL
    )
    if (!is.null(s)) {
      for (i in seq_along(times_at)) {
        lab <- labels[i]
        s_val <- s$surv[i]
        n_val <- s$n.risk[i]
        if (!is.na(s_val) && is.finite(s_val)) {
          args[[paste0("survival_at_", lab)]] <- as.numeric(s_val)
        }
        if (!is.na(n_val) && is.finite(n_val)) {
          args[[paste0("n_at_risk_", lab)]] <- as.integer(n_val)
        }
        if (!is.null(s$lower)) {
          lo <- s$lower[i]
          if (!is.na(lo) && is.finite(lo)) {
            args[[paste0("survival_at_", lab, "_ci_lower")]] <- as.numeric(lo)
          }
        }
        if (!is.null(s$upper)) {
          hi <- s$upper[i]
          if (!is.na(hi) && is.finite(hi)) {
            args[[paste0("survival_at_", lab, "_ci_upper")]] <- as.numeric(hi)
          }
        }
      }
    }
  }

  # Log-rank χ² across groups. ``survdiff`` from survival reports
  # the chi² stat as ``$chisq`` and the df as ``length($n) - 1``
  # (one DF per group beyond the reference). Compute the p-value
  # via ``pchisq`` (chi² distribution upper tail).
  if (!is.null(survdiff)) {
    chi2 <- tryCatch(as.numeric(survdiff$chisq), error = function(e) NULL)
    if (!is.null(chi2) && is.finite(chi2)) {
      args$logrank_chi_squared <- chi2
      df <- length(survdiff$n) - 1L
      if (df >= 1) {
        args$logrank_p_value <- as.numeric(
          pchisq(chi2, df = df, lower.tail = FALSE)
        )
      }
      args$n_groups <- as.integer(length(survdiff$n))
    }
  }

  do.call(nora$result, c(args, list(...)))
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
                                   distinct_count = NULL,
                                   ...) {
  cat(sprintf(
    "%s: n=%d, mean=%.6g, sd=%.6g, missing=%d",
    variable, as.integer(n), mean, sd, as.integer(missing_count)
  ))
  if (!is.null(distinct_count)) {
    cat(sprintf(", distinct=%d", as.integer(distinct_count)))
  }
  cat("\n")
  # min_value / max_value are no longer accepted: the sanitizer drops
  # them in every payload because nothing in the payload binds the
  # reported values to the named variable's actual column. A future
  # Nora-owned bounds path (request_data extension) is the correct
  # surface; emit aggregates only here.
  args <- list(
    type = "descriptive",
    variable = variable,
    n = as.integer(n),
    mean = mean,
    sd = sd,
    missing_count = as.integer(missing_count),
    distinct_count = if (is.null(distinct_count)) NULL else as.integer(distinct_count)
  )
  do.call(nora$result, c(args, list(...)))
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

  # Reject `...` arguments that would override fields the helper
  # computes from raw data. Without this guard, a caller could pass
  # `cells=list(...)` (or row_variable=..., etc.) and the
  # `c(list(computed), list(...))` concatenation below would emit
  # duplicate keys whose JSON serialization a downstream parser
  # resolves by last-occurrence, replacing the helper's computation.
  # The `_via_helper` marker (stamped at the end) would then
  # authenticate attacker-supplied values, and the sanitizer (which
  # trusts the marker to skip recomputing `max_share`) would let a
  # forged max_share=0 bypass the dominance gate.
  extras <- list(...)
  reserved <- c("type", "row_variable", "value_variable",
                "aggregation", "cells", "_via_helper")
  forbidden <- intersect(names(extras), reserved)
  if (length(forbidden) > 0) {
    stop("nora$from_magnitude_table: cannot override helper-",
         "computed fields via extra arguments: ",
         paste(sort(forbidden), collapse = ", "),
         ". These are computed from the data and bound to the ",
         "_via_helper provenance marker.")
  }

  # Bypass nora$result and write directly so the helper-provenance
  # marker (`_via_helper`) survives. The generic nora$result strips
  # `_via_helper` from caller-passed kwargs so a script can't forge
  # this marker through the public API. The sanitizer requires the
  # marker for magnitude_table because cell-level `max_share` is
  # consulted-only and stripped; without proof that max_share came
  # from raw-data computation a malicious script could publish a
  # dominance-violating value with `max_share=0` and skip the gate.
  payload <- c(
    list(
      type = "magnitude_table",
      row_variable = as.character(group_var),
      value_variable = as.character(value_var),
      aggregation = aggregation,
      cells = cells
    ),
    extras
  )
  payload[["_via_helper"]] <- "from_magnitude_table"
  nora$.write_result(payload)
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


#' Pairwise correlation matrix from a data frame.
#'
#' By default correlates every numeric column; pass `variables` to
#' restrict to a named subset. `method` is `"pearson"` (default),
#' `"spearman"`, or `"kendall"`. Sample size N is the number of
#' COMPLETE rows over the chosen variables — emitting pairwise N
#' would let off-diagonals draw on different samples and make joint
#' inference dishonest.
#'
#' Also prints the matrix to stdout for the researcher's raw log.
nora$from_correlation <- function(df, variables = NULL,
                                   method = "pearson", ...) {
  valid_methods <- c("pearson", "spearman", "kendall")
  if (!(method %in% valid_methods)) {
    stop('nora$from_correlation: method must be one of ',
         paste(shQuote(valid_methods), collapse = ", "), ", got ",
         shQuote(method))
  }
  if (!is.data.frame(df)) {
    stop("nora$from_correlation: expected a data.frame, got ",
         class(df)[1])
  }
  if (is.null(variables)) {
    # Pick numeric / logical columns. Match the Python helper's
    # default: skip character / factor / Date.
    is_num_col <- vapply(df, function(col) {
      is.numeric(col) || is.logical(col)
    }, logical(1))
    variables <- names(df)[is_num_col]
  }
  if (length(variables) == 0) {
    stop("nora$from_correlation: no numeric columns found and no ",
         "`variables` provided")
  }
  missing_cols <- setdiff(variables, names(df))
  if (length(missing_cols) > 0) {
    stop("nora$from_correlation: variables not in df: ",
         paste(missing_cols, collapse = ", "))
  }
  sub <- df[, variables, drop = FALSE]
  complete <- stats::complete.cases(sub)
  n_complete <- sum(complete)
  missing_count <- nrow(sub) - n_complete
  cm <- stats::cor(sub[complete, , drop = FALSE], method = method)
  print(cm)
  # Build nested list-of-lists in declared variable order so the
  # JSON output is symmetric and stable regardless of `cor()`'s
  # internal column ordering.
  correlations <- list()
  for (rv in variables) {
    inner <- list()
    for (cv in variables) {
      v <- cm[rv, cv]
      if (is.finite(v)) inner[[cv]] <- as.numeric(v)
    }
    if (length(inner) > 0) correlations[[rv]] <- inner
  }
  nora$result(
    type = "correlation_matrix",
    n = as.integer(n_complete),
    variables = as.list(variables),
    method = method,
    correlations = correlations,
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

# Return `base` if no file by that name exists in `d`; otherwise
# append _2, _3, ... before the extension. Without this, calling
# `plot_coefficients(fit1)` then `plot_coefficients(fit2)` in one
# script would (a) overwrite `coefficients.png` with fit2's image
# and (b) append a second manifest row pointing at the SAME file,
# so the model sees two "different" plots that are both fit2.
# `plot_interaction` already side-steps this by suffixing the
# variable name into the filename; the others need a counter.
nora$.unique_plot_name <- function(d, base) {
  if (!file.exists(file.path(d, base))) return(base)
  parts <- tools::file_path_sans_ext(base)
  ext <- tools::file_ext(base)
  i <- 2L
  repeat {
    candidate <- if (nzchar(ext)) paste0(parts, "_", i, ".", ext) else paste0(parts, "_", i)
    if (!file.exists(file.path(d, candidate))) return(candidate)
    i <- i + 1L
  }
}

# Scrub a categorical tick label that will be rendered into a
# model-visible plot image. Mirrors the Python helper's use of
# ``safe_text`` on raw category names: strip C0/C1 control chars,
# bidi overrides, zero-width chars, and BOM/word-joiner; cap to
# 24 chars; fall back to "[redacted]" if the scrub leaves nothing.
# Plot images bypass the JSON/text path's safety gate, so any raw
# string that reaches an axis label has to be cleaned here.
nora$.safe_tick_label <- function(v) {
  s <- tryCatch(as.character(v), error = function(e) "")
  if (length(s) != 1 || is.na(s)) s <- ""
  # C0 controls (U+0001-U+001F) and DEL (U+007F). R strings cannot
  # carry a literal U+0000 (the C-string representation forbids
  # it), so the NUL byte never reaches this function and doesn't
  # need a strip pass — and ``\x00`` in a string literal is a
  # parse error anyway.
  s <- gsub("[\x01-\x1f\x7f]", "", s, perl = TRUE, useBytes = FALSE)
  # C1 controls (U+0080-U+009F).
  s <- gsub("[-]", "", s, perl = TRUE, useBytes = FALSE)
  # Bidi overrides + isolates and zero-width chars: LRM/RLM,
  # LRE/RLE/PDF, LRO/RLO, LRI/RLI/FSI/PDI, ZWSP/ZWNJ/ZWJ, word
  # joiner, BOM, invisible-times / invisible-separator.
  s <- gsub(
    paste0(
      "[\u200b\u200c\u200d\u200e\u200f",
      "\u202a\u202b\u202c\u202d\u202e",
      "\u2060\u2062\u2063",
      "\u2066\u2067\u2068\u2069",
      "\ufeff]"
    ),
    "", s, perl = TRUE, useBytes = FALSE,
  )
  # Cap at 24 chars (matches the Python helper's per-tick limit).
  if (nchar(s) > 24) s <- paste0(substr(s, 1, 23), "…")
  if (!nzchar(s)) s <- "[redacted]"
  s
}


nora$.append_plot_manifest <- function(file, kind, label) {
  d <- nora$.plots_dir()
  if (is.null(d)) return(invisible(NULL))
  # Stamp every entry with the per-run token. The executor validates
  # this field after the script finishes and drops any entry whose
  # token is missing or wrong; that strips manifest rows a script
  # could otherwise have appended directly (saving a raw-data plot
  # under _nora_plots/ and labeling it "coefficients" to slip past
  # the disclosure-control allowlist for vision attachment). Same
  # posture as the result-payload _token field.
  entry <- list(file = file, kind = kind)
  if (!is.null(label) && nzchar(label)) entry$label <- label
  entry[["_token"]] <- nora$.run_token
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
  fname <- nora$.unique_plot_name(d, "residuals.png")
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
    # Disclosure-control: the rendered PNG is allowlisted for model
    # vision (kind="interaction"), so anything legible on the x-axis
    # crosses the SDC boundary. The previous version used min/max
    # (numeric) and full level lists (factor/categorical), which
    # surfaced raw extrema and rare-level identities the JSON
    # sanitizer would have refused. Match the Python helper's
    # disclosure-safe grid:
    #   - numeric: mean ± 2*sd; equivalent to the descriptive
    #     sanitizer's already-allowed mean+sd disclosure. Refuse
    #     when N is below threshold or variance is zero.
    #   - factor / categorical: drop levels with count below
    #     threshold; refuse when none remain.
    .CELL_SUPP_THRESH <- 10L
    .CAT_LEVEL_CAP <- 20L
    .suppression_note <- NULL
    if (is.numeric(xs)) {
      .clean <- xs[!is.na(xs)]
      if (length(.clean) < .CELL_SUPP_THRESH) {
        stop("variable '", var, "' has fewer than ", .CELL_SUPP_THRESH,
             " non-missing observations; below the disclosure threshold")
      }
      .mu <- mean(.clean)
      .sd <- stats::sd(.clean)
      if (!is.finite(.sd) || .sd <= 0) {
        stop("variable '", var,
             "' has zero variance — interaction plot would expose ",
             "the constant value")
      }
      grid <- seq(.mu - 2 * .sd, .mu + 2 * .sd, length.out = 100)
    } else {
      .clean <- xs[!is.na(xs)]
      .counts <- table(.clean)
      .visible <- .counts[.counts >= .CELL_SUPP_THRESH]
      if (length(.visible) == 0) {
        stop("variable '", var,
             "': no level meets the disclosure threshold (n >= ",
             .CELL_SUPP_THRESH, "); refusing to plot")
      }
      # Sort by frequency desc, keep top-K for readability.
      .visible <- sort(.visible, decreasing = TRUE)
      if (length(.visible) > .CAT_LEVEL_CAP) {
        .visible <- .visible[seq_len(.CAT_LEVEL_CAP)]
      }
      .keep_levels <- names(.visible)
      .dropped <- as.integer(sum(.counts < .CELL_SUPP_THRESH))
      if (.dropped > 0) {
        .suppression_note <- paste0(
          .dropped, " rare level(s) with count < ",
          .CELL_SUPP_THRESH, " suppressed"
        )
      }
      if (is.factor(xs)) {
        grid <- factor(.keep_levels, levels = levels(xs))
      } else {
        grid <- .keep_levels
      }
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
        # Run tick labels through the same text-safety primitive
        # the Python helper applies. ``grid`` values come straight
        # from levels in the raw data, so without scrubbing, a
        # frequent category name with embedded control chars,
        # bidi overrides, zero-width chars, or prompt-like text
        # would render straight into the model-visible image,
        # bypassing the JSON/text safety gate. Empty after scrub
        # becomes "[redacted]" so the bar stays identifiable.
        labels <- vapply(grid, function(v) {
          nora$.safe_tick_label(v)
        }, character(1))
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
  fname <- nora$.unique_plot_name(d, "coefficients.png")
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
  fname <- nora$.unique_plot_name(d, "estimate_comparison.png")
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
      fname, "estimate_comparison",
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
