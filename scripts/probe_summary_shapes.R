#!/usr/bin/env Rscript
# Probe what summary()$coefficients (or equivalent) looks like for
# each estimator class, so the helper rewrite is based on what R
# actually returns, not what I think it returns.

set.seed(42)
n <- 200
df <- data.frame(
  x1 = rnorm(n),
  x2 = rnorm(n),
  group = factor(sample(letters[1:3], n, replace = TRUE))
)
df$y_cont <- 1 + 0.5 * df$x1 - 0.3 * df$x2 + rnorm(n)
df$y_bin  <- as.integer(plogis(0.3 * df$x1 - 0.5 * df$x2) > runif(n))
df$y_count <- rpois(n, lambda = exp(0.2 + 0.1 * df$x1))
df$t_event <- rexp(n, rate = exp(-0.3 + 0.4 * df$x1))
df$cens <- as.integer(df$t_event < 2)
df$t_obs <- pmin(df$t_event, 2)

probe <- function(label, m) {
  cat("\n===", label, "===\n")
  cat("class:", paste(class(m), collapse=","), "\n")
  s <- summary(m)
  ce <- tryCatch(as.data.frame(s$coefficients), error = function(e) NULL)
  if (is.null(ce)) {
    cat("s$coefficients: <not a data.frame-able object>\n")
    # fixest uses $coeftable
    ct <- tryCatch(s$coeftable, error = function(e) NULL)
    if (!is.null(ct)) {
      cat("s$coeftable cols:", paste(colnames(ct), collapse=" | "), "\n")
    }
  } else {
    cat("s$coefficients cols:", paste(colnames(ce), collapse=" | "), "\n")
  }
  cat("model$df.residual:", m$df.residual %||% NA, "\n")
  cat("model$deviance:", m$deviance %||% NA, "\n")
  cat("model$null.deviance:", m$null.deviance %||% NA, "\n")
  cat("model$aic (attr):", m$aic %||% NA, "\n")
  cat("model$nevent:", m$nevent %||% NA, "\n")
  cat("model$n:", m$n %||% NA, "\n")
  cat("nobs(m):", nobs(m), "\n")
  cat("logLik(m):", tryCatch(as.numeric(logLik(m)), error = function(e) NA), "\n")
  cat("AIC(m):",   tryCatch(AIC(m), error = function(e) NA), "\n")
  cat("BIC(m):",   tryCatch(BIC(m), error = function(e) NA), "\n")
  cs <- tryCatch(s$concordance, error = function(e) NULL)
  cat("summary$concordance:", paste(cs, collapse=","), "\n")
  lr <- tryCatch(s$logtest, error = function(e) NULL)
  cat("summary$logtest:", paste(names(lr), collapse=","), "=", paste(lr, collapse=","), "\n")
  # fixest-specific
  for (nm in c("fixef_vars", "fixef_sizes", "nobs", "ssr", "tss", "fml")) {
    val <- m[[nm]]
    if (!is.null(val)) cat(sprintf("m$%s: %s\n", nm,
      if(is.atomic(val) && length(val) <= 10) paste(val, collapse=",") else paste("<", paste(class(val), collapse=","), ">")))
  }
}

`%||%` <- function(a, b) if (is.null(a) || length(a) == 0) b else a

probe("ols",     lm(y_cont ~ x1 + x2, data = df))
probe("logit",   glm(y_bin ~ x1 + x2, family = binomial, data = df))
probe("poisson", glm(y_count ~ x1 + x2, family = poisson, data = df))
suppressMessages({
  library(MASS)
  probe("negbin", MASS::glm.nb(y_count ~ x1 + x2, data = df))
})
suppressMessages({
  library(survival)
  probe("cox", coxph(Surv(t_obs, cens) ~ x1 + x2, data = df))
})
suppressMessages({
  library(fixest)
  probe("feols_fe", feols(y_cont ~ x1 + x2 | group, data = df, cluster = ~group))
})
