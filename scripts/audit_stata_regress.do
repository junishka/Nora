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

display "audit complete: " "`_path'"
