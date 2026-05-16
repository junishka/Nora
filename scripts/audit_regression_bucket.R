#!/usr/bin/env Rscript
# Audit: fit each estimator the linear_regression bucket claims to
# support, call nora$from_lm, and capture the emitted JSONL. The
# Python side of the audit then sanitizes and diffs each payload
# against an "inference-adequate" minimum field set per estimator.

Sys.setenv(NORA_RUN_TOKEN = "audit-run-token-not-secret")
result_path <- "scripts/audit_payloads.jsonl"
if (file.exists(result_path)) file.remove(result_path)
Sys.setenv(NORA_RESULT_PATH = result_path)

source("src/nora/runtime/nora.R")

set.seed(42)
n <- 400
df <- data.frame(
  x1 = rnorm(n),
  x2 = rnorm(n),
  group = factor(sample(c("a", "b", "c"), n, replace = TRUE))
)
df$y_cont <- 1 + 0.5 * df$x1 - 0.3 * df$x2 + rnorm(n)
df$y_bin  <- as.integer(plogis(0.3 * df$x1 - 0.5 * df$x2) > runif(n))
df$y_count <- rpois(n, lambda = exp(0.2 + 0.1 * df$x1))
# Survival data: exponential times, right-censoring at t=2
df$t_event <- rexp(n, rate = exp(-0.3 + 0.4 * df$x1))
df$cens <- as.integer(df$t_event < 2)
df$t_obs <- pmin(df$t_event, 2)

cat("\n=== AUDIT: from_lm against real fits ===\n", file = stderr())

audit <- function(label, expr) {
  cat(sprintf("\n--- %s ---\n", label), file = stderr())
  res <- tryCatch(eval(expr), error = function(e) {
    cat(sprintf("  HELPER ERROR: %s\n", conditionMessage(e)), file = stderr())
    NULL
  })
  # Tag the emitted line so the Python side can match payload to estimator.
  # Append a sentinel line to the JSONL with the label.
  con <- file(result_path, open = "a", encoding = "UTF-8")
  writeLines(sprintf('{"_audit_label": "%s"}', label), con)
  close(con)
  invisible(NULL)
}

# 1. OLS (baseline)
audit("ols", quote(
  nora$from_lm(lm(y_cont ~ x1 + x2, data = df))
))

# 2. Logit
audit("logit", quote(
  nora$from_lm(glm(y_bin ~ x1 + x2, family = binomial, data = df))
))

# 3. Probit
audit("probit", quote(
  nora$from_lm(glm(y_bin ~ x1 + x2, family = binomial(link = "probit"), data = df))
))

# 4. Poisson
audit("poisson", quote(
  nora$from_lm(glm(y_count ~ x1 + x2, family = poisson, data = df))
))

# 5. Negative binomial (MASS)
audit("negbin", quote({
  library(MASS, quietly = TRUE)
  nora$from_lm(MASS::glm.nb(y_count ~ x1 + x2, data = df))
}))

# 6. Cox PH (survival)
audit("cox", quote({
  library(survival, quietly = TRUE)
  nora$from_lm(coxph(Surv(t_obs, cens) ~ x1 + x2, data = df))
}))

# 7. Fixed effects via fixest::feols (absorbed group FE + clustered SE)
audit("feols_fe", quote({
  library(fixest, quietly = TRUE)
  nora$from_lm(feols(y_cont ~ x1 + x2 | group, data = df, cluster = ~group))
}))

cat(sprintf("\nWrote audit payloads to: %s\n", result_path), file = stderr())
cat(result_path, "\n")  # to stdout for the caller
