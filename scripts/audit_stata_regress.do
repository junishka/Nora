* Audit: fit each estimator nora_result_regress claims to support,
* emit through the helper, capture JSONL for sanitization in Python.
*
* Mirrors scripts/audit_regression_bucket.R but for Stata. Run via:
*
*   NORA_RUN_TOKEN=audit-token NORA_RESULT_PATH=$PWD/scripts/audit_stata.jsonl \
*     stata-mp -b do scripts/audit_stata_regress.do

clear all
set more off
adopath ++ "src/nora/runtime"
local _path : env NORA_RESULT_PATH
* Drop the existing JSONL so this run starts clean. Stata's `erase`
* errors if the file doesn't exist; capture so the first run works.
capture erase "`_path'"

* Build a small dataset that supports OLS / logit / Poisson / Cox /
* xtreg-fe / areg without external data dependencies.
clear
set obs 600
set seed 42
gen id = mod(_n - 1, 60) + 1            // 60 panels of 10 obs each
gen t = floor((_n - 1) / 60) + 1
gen x1 = rnormal()
gen x2 = rnormal()
gen y_cont = 1 + 0.5*x1 - 0.3*x2 + rnormal()
gen y_bin = (invlogit(0.3*x1 - 0.5*x2) > runiform())
gen y_count = rpoisson(exp(0.2 + 0.1*x1))
gen u = runiform()
gen t_event = -log(u) / exp(-0.3 + 0.4*x1)
gen cens = (t_event < 2)
gen t_obs = min(t_event, 2)

xtset id t

* Emit a sentinel line BEFORE each fit so the Python side can map
* payload to estimator (helper appends payload after our sentinel).
* Format: {"_audit_label": "ols"}
local labels "ols logit poisson stcox xtreg_fe areg"

* OLS
file open ah using "`_path'", write text append
file write ah `"{"_audit_label":"ols"}"' _newline
file close ah
quietly regress y_cont x1 x2
nora_result_regress

* Logit
file open ah using "`_path'", write text append
file write ah `"{"_audit_label":"logit"}"' _newline
file close ah
quietly logit y_bin x1 x2
nora_result_regress

* Poisson
file open ah using "`_path'", write text append
file write ah `"{"_audit_label":"poisson"}"' _newline
file close ah
quietly poisson y_count x1 x2
nora_result_regress

* Cox PH
file open ah using "`_path'", write text append
file write ah `"{"_audit_label":"stcox"}"' _newline
file close ah
quietly stset t_obs, failure(cens)
quietly stcox x1 x2
nora_result_regress

* Panel FE (xtreg, fe)
file open ah using "`_path'", write text append
file write ah `"{"_audit_label":"xtreg_fe"}"' _newline
file close ah
quietly xtreg y_cont x1 x2, fe
nora_result_regress

* areg (alternative FE absorption)
file open ah using "`_path'", write text append
file write ah `"{"_audit_label":"areg"}"' _newline
file close ah
quietly areg y_cont x1 x2, absorb(id)
nora_result_regress

* OLS with cluster-robust SE — pins the auto-emission of
* cluster_variables / n_clusters / robust_se_type via e(vce) ==
* "cluster", e(clustvar), e(N_clust).
file open ah using "`_path'", write text append
file write ah `"{"_audit_label":"ols_clustered"}"' _newline
file close ah
quietly regress y_cont x1 x2, vce(cluster id)
nora_result_regress

* Mixed-effects (single-grouping random intercept). Pins emission of
* random_effects_variance, n_groups_per_level, fit_method, icc on the
* Stata side — matches the contract R lme4 and Python
* statsmodels.MixedLM ship through nora$from_lm / nora.from_lm.
* Build a larger panel with real between-cluster variance so `mixed`
* actually has variance components to estimate. Build per-school
* effects first, then expand to 30 obs per school for 1500 total.
preserve
clear
set obs 50
set seed 20260516
gen school = _n
gen school_eff = 0.6 * rnormal()
expand 30
bysort school: gen sx = rnormal()
gen sy = school_eff + 0.4 * sx + 0.4 * rnormal()

file open ah using "`_path'", write text append
file write ah `"{"_audit_label":"mixed_re_intercept"}"' _newline
file close ah
* `, reml` pinned explicitly. Stata's `mixed` default flipped to ML
* around Stata 18, but the cross-language sanitizer contract pinned by
* the R lme4 / Python statsmodels.MixedLM real-fit tests uses REML.
* Without this option the test would assert fit_method=="REML" against
* an ML fit on newer Stata installs.
quietly mixed sy sx || school:, reml
nora_result_regress

* meglm logistic with random intercept. Same variance-components
* shape as `mixed`, but no residual variance (non-Gaussian), so
* icc must NOT appear and the residual key must not be present in
* random_effects_variance — pins that gate.
gen sy_bin = (sy > 0)
file open ah using "`_path'", write text append
file write ah `"{"_audit_label":"meglm_logit_re"}"' _newline
file close ah
quietly meglm sy_bin sx || school:, family(binomial) link(logit)
nora_result_regress
restore

display "audit complete: " "`_path'"
