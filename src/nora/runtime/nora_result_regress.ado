*! version 0.0.5  Nora runtime: emit a coefficient_table_with_fit_stats payload from e().
*!
*! Call after a regression command (regress, logit, probit, poisson,
*! stcox, xtreg fe, areg, ivregress, mixed, meglm — anything that
*! populates e(b), e(V), e(N), e(depvar)). Writes the structured
*! payload to the path in $NORA_RESULT_PATH so the Nora executor
*! can pick it up and route through the sanitizer.
*!
*! Usage:
*!   regress y x1 x2
*!   nora_result_regress, label("OLS income ~ edu+age")
*!
*!   mixed y x || school:
*!   nora_result_regress, label("random-intercept by school")
*!
*! Notes:
*! - `version 13` for broad Stata compatibility (Stata 13, 14, 15, 16, 17, 18).
*! - Matrix indexing uses integer positions, not name-indexing — Stata
*!   rejects matname[1, "name"] in scalar context on some versions.
*! - Float values are formatted as `%21.17e` (scientific, full precision)
*!   so the output is always valid JSON — Stata's default number
*!   formatting drops the leading zero on values < 1 (".123" vs "0.123")
*!   and JSON requires the leading zero.

program define nora_result_regress
    version 13
    syntax [, label(string) ]

    * JSON-escape `label` (the only Claude-controllable free-text field
    * in this emitter). Variable names from e(depvar) / colnames are
    * Stata-syntax-shape-constrained and don't need escaping. Order of
    * subinstr steps matters: backslash first, so its escape isn't
    * double-escaped; quote next; then collapse CR/LF/TAB to spaces
    * (JSON disallows literal control chars in string values).
    local label : subinstr local label "\" "\\", all
    local label : subinstr local label `"""' `"\""', all
    local label : subinstr local label "`=char(10)'" " ", all
    local label : subinstr local label "`=char(13)'" " ", all
    local label : subinstr local label "`=char(9)'" " ", all

    if "`e(cmd)'" == "" {
        display as error "nora_result_regress: no regression results in memory. Run a regression command first."
        exit 198
    }

    local path : env NORA_RESULT_PATH
    if "`path'" == "" {
        display as error "NORA_RESULT_PATH not set. This script must be run through Nora."
        exit 198
    }

    * Per-run authenticity token — the executor validates this before
    * the payload reaches the sanitizer. Stata cannot cleanly unset
    * environment variables from within the running process, so a user
    * script CAN also read NORA_RUN_TOKEN. The protection this
    * provides is that naive "write hand-crafted JSON" bypasses fail,
    * and any sophisticated bypass must include token-extraction logic
    * that is visible in the executed script the researcher reviews.
    local _nora_token : env NORA_RUN_TOKEN
    if "`_nora_token'" == "" {
        display as error "NORA_RUN_TOKEN not set. Run through Nora."
        exit 198
    }

    * Integer-position indexing into e(b) / e(V) — name indexing fails
    * in scalar context on Stata 13/15. Column names come from colnames.
    tempname bmat Vmat
    matrix `bmat' = e(b)
    matrix `Vmat' = e(V)
    local vnames : colnames `bmat'
    local k = colsof(`bmat')

    * Mixed-effects models (mixed, meglm) post e(b) with TWO blocks of
    * columns: fixed-effect parameters under the dep-var equation
    * prefix ("y:x", "y:_cons", ...) followed by transformed variance
    * components under prefixes like "lns1_1_1:_cons" (log-SD of school
    * intercept), "lnc1_1_1_2:_cons" (atanh of intercept-slope
    * correlation), "lnsig_e:_cons" (log residual SD). e(k_f) reports
    * the count of fixed-effect parameters — restrict bmat / Vmat /
    * vnames / k to that leading submatrix so the existing coefficient
    * / SE / p-value / vcov blocks operate only on the fixed effects.
    * The variance components are emitted separately further down via
    * estat recovariance (natural-scale matrices, one per RE level).
    * Without this guard the helper would emit a "coefficient" called
    * "lns1_1_1:_cons" and the sanitizer would reject the whole
    * payload because predictor names must be plain identifiers.
    *
    * The leading "<depvar>:" equation prefix is also stripped from
    * vnames here so predictor names match the dataset schema
    * (sanitizer expects "x", not "y:x").
    * meglm in current Stata is wired through gsem internally: the
    * outer ``e(cmd)`` reports "gsem" and the "real" command name lives
    * in ``e(cmd2)``. Check both so the random-effects branch fires
    * on meglm regardless of which surface stored the name.
    local _is_mixed_re = ("`e(cmd)'" == "mixed") | ///
        ("`e(cmd)'" == "meglm") | ("`e(cmd2)'" == "meglm")
    if `_is_mixed_re' & "`e(k_f)'" != "" & !missing(`=e(k_f)') & `=e(k_f)' > 0 {
        local _k_f = `=e(k_f)'
        matrix `bmat' = e(b)[1, 1..`_k_f']
        matrix `Vmat' = e(V)[1..`_k_f', 1..`_k_f']
        local _raw_names : colnames `bmat'
        local k = `_k_f'
        * Strip equation prefix "y:" from each colname. tokenize
        * splits on whitespace so this is a simple per-name loop.
        local _stripped ""
        forvalues i = 1/`k' {
            local _full : word `i' of `_raw_names'
            local _colon = strpos("`_full'", ":")
            if `_colon' > 0 {
                local _term = substr("`_full'", `_colon' + 1, .)
            }
            else {
                local _term = "`_full'"
            }
            local _stripped "`_stripped' `_term'"
        }
        local vnames "`_stripped'"
    }

    tempname fh
    file open `fh' using `"`path'"', write text append

    * ``coefficient_table_with_fit_stats`` is the canonical bucket
    * name. ``linear_regression`` remains a back-compat alias in the
    * sanitizer dispatch so stored payloads keep working.
    file write `fh' `"{"type":"coefficient_table_with_fit_stats""'
    file write `fh' `","_token":"`_nora_token'""'
    if `"`label'"' != "" {
        file write `fh' `","label":"`label'""'
    }

    * Integer fields: n, df.
    * ``e(N)`` can be ``.`` (Stata missing) for some estimators
    * (e.g., ``xtlogit, pa`` doesn't populate it). Without a
    * ``missing()`` guard we'd write a literal ``.`` into the
    * JSON, which Python's ``json.loads`` rejects as not a number
    * — every line that follows in the JSONL stream then drops
    * silently. Emit ``null`` instead and let the sanitizer
    * decide.
    if missing(`=e(N)') {
        file write `fh' `","n":null"'
    }
    else {
        file write `fh' `","n":`=e(N)'"'
    }

    local dv = "`e(depvar)'"
    file write `fh' `","response_variable":"`dv'""'

    * predictor_variables — drop _cons.
    file write `fh' `","predictor_variables":["'
    local first = 1
    forvalues i = 1/`k' {
        local v : word `i' of `vnames'
        if "`v'" != "_cons" {
            if !`first' file write `fh' ","
            file write `fh' `""`v'""'
            local first = 0
        }
    }
    file write `fh' "]"

    * coefficients (named dict, integer-indexed into e(b)). Missing
    * values (degenerate fits, perfect collinearity in some
    * estimators) are written as JSON null — strofreal(., ...)
    * returns "." which would corrupt the entire JSON line and
    * break the executor's line-by-line parser.
    file write `fh' `","coefficients":{"'
    local first = 1
    forvalues i = 1/`k' {
        local v : word `i' of `vnames'
        if !`first' file write `fh' ","
        if missing(`bmat'[1, `i']) {
            file write `fh' `""`v'":null"'
        }
        else {
            local coef_str = strofreal(`bmat'[1, `i'], "%21.17e")
            file write `fh' `""`v'":`coef_str'"'
        }
        local first = 0
    }
    file write `fh' "}"

    * standard_errors (named dict, from sqrt(diag(V))). Same
    * missing → null treatment as coefficients above.
    file write `fh' `","standard_errors":{"'
    local first = 1
    forvalues i = 1/`k' {
        local v : word `i' of `vnames'
        if !`first' file write `fh' ","
        if missing(`Vmat'[`i', `i']) | `Vmat'[`i', `i'] < 0 {
            * Negative diagonals can arise from numerical noise in
            * robust SE computation; ``sqrt`` of a negative yields
            * Stata-missing ``.`` and ``strofreal(.)`` emits a
            * literal ``.`` into the JSON — invalid number, every
            * downstream JSONL line drops silently. Guard the
            * sign too so robust-SE artifacts produce ``null``
            * rather than a corrupt payload.
            file write `fh' `""`v'":null"'
        }
        else {
            local se_str = strofreal(sqrt(`Vmat'[`i', `i']), "%21.17e")
            file write `fh' `""`v'":`se_str'"'
        }
        local first = 0
    }
    file write `fh' "}"

    * p_values (named dict). Two cases by estimator family:
    *   OLS (regress): e(df_r) is populated → two-sided t-test
    *     against that df_r.
    *   GLMs / survival (logit, probit, poisson, stcox, ...):
    *     e(df_r) is empty → fall back to the asymptotic Wald
    *     z-test, p = 2 * (1 - normal(|b/se|)). Stata's display
    *     output for these estimators uses the same z-test, so the
    *     emitted values match what the researcher sees printed.
    * Without the z-fallback, every non-OLS regression silently
    * shipped with no per-coefficient p-values and the renderer
    * dropped the p-value column entirely — making logit / probit /
    * Poisson / Cox cards look like "estimator doesn't compute
    * p-values" when in fact the Wald test is well-defined.
    * Dropped/collinear terms have SE=0, which makes b/se missing;
    * emit JSON `null` for those rather than Stata's "." (not valid
    * JSON). When neither b nor V is meaningfully populated (a
    * non-regression e()), the dict is omitted entirely.
    if `k' > 0 {
        local _has_dfr = ("`e(df_r)'" != "" & !missing(`=e(df_r)'))
        tempname _se _b _t _p
        scalar `_se' = .
        scalar `_b' = .
        scalar `_t' = .
        scalar `_p' = .
        file write `fh' `","p_values":{"'
        local first = 1
        forvalues i = 1/`k' {
            local v : word `i' of `vnames'
            if !`first' file write `fh' ","
            scalar `_se' = sqrt(`Vmat'[`i', `i'])
            scalar `_b' = `bmat'[1, `i']
            if `_se' == 0 | missing(`_se') | missing(`_b') {
                file write `fh' `""`v'":null"'
            }
            else {
                scalar `_t' = abs(`_b' / `_se')
                if `_has_dfr' {
                    scalar `_p' = 2 * ttail(`=e(df_r)', `_t')
                }
                else {
                    * 2 * normal(-|t|) is numerically stable for
                    * large |t|; (1 - normal(|t|)) loses precision
                    * once |t| > 8 or so and rounds to 0.
                    scalar `_p' = 2 * normal(-`_t')
                }
                local _pstr = strofreal(`_p', "%21.17e")
                file write `fh' `""`v'":`_pstr'"'
            }
            local first = 0
        }
        file write `fh' "}"
    }

    * Optional fit statistics. Missing e() macros mean the command didn't
    * populate them (e.g. robust SE paths change what's in e()) — omit
    * the field. ``e(F)`` and ``e(r2_a)`` can also be SET to missing
    * under degenerate fits (perfect fit, df_r = 0, FE absorption);
    * the ``& !missing(...)`` guard catches that case too. Without it
    * ``strofreal(., "%21.17e")`` returns "." and the JSON line is
    * malformed — the executor's parser breaks at that line and
    * silently loses every later result in the same script.
    if "`e(r2)'" != "" & !missing(`=e(r2)') {
        local _x = strofreal(`=e(r2)', "%21.17e")
        file write `fh' `","r_squared":`_x'"'
    }
    if "`e(r2_a)'" != "" & !missing(`=e(r2_a)') {
        local _x = strofreal(`=e(r2_a)', "%21.17e")
        file write `fh' `","adj_r_squared":`_x'"'
    }
    * F-test from ``regress``. Stata's ``regress`` reports the
    * F-test p-value in its display output (``Prob > F``) but does
    * NOT populate ``e(p)`` for OLS (verified empirically against
    * Stata 17/18). The chi2-gated branch further down therefore
    * does not silently emit ``chi_squared_p_value`` from an OLS
    * run — ``e(p)`` is empty, so the gate is satisfied either way.
    * If a future Stata version starts populating ``e(p)`` for
    * regress, the chi2 gate (``e(chi2)`` non-empty) keeps the
    * fields disjoint: the value would be available as ``e(p)``
    * but would not flow into ``chi_squared_p_value`` because
    * regress doesn't set ``e(chi2)``. To close the cross-language
    * consistency gap (R's ``lm()`` and Python's statsmodels OLS
    * both ship the F p-value via ``from_lm``), compute the F
    * p-value here ourselves from ``e(F)``, ``e(df_m)``, ``e(df_r)``
    * via ``Ftail`` whenever all three are populated. Without this
    * the model could not tell "missing-by-design" from "missing-
    * by-error" on a Stata OLS card and would silently lose a
    * field every other regression card carries.
    if "`e(F)'" != "" & !missing(`=e(F)') {
        local _x = strofreal(`=e(F)', "%21.17e")
        file write `fh' `","f_statistic":`_x'"'
        if "`e(df_m)'" != "" & !missing(`=e(df_m)') ///
                & "`e(df_r)'" != "" & !missing(`=e(df_r)') {
            local _fp = Ftail(`=e(df_m)', `=e(df_r)', `=e(F)')
            if !missing(`_fp') {
                local _x = strofreal(`_fp', "%21.17e")
                file write `fh' `","f_p_value":`_x'"'
            }
        }
    }
    if "`e(rmse)'" != "" & !missing(`=e(rmse)') {
        local _x = strofreal(`=e(rmse)', "%21.17e")
        file write `fh' `","residual_std_error":`_x'"'
    }
    if "`e(df_r)'" != "" & !missing(`=e(df_r)') {
        file write `fh' `","degrees_of_freedom":`=e(df_r)'"'
    }

    * Non-OLS fit metrics. Emitted only when the underlying command
    * populated them — guards mirror the r2 / F pattern above. Maps
    * Stata's e() vocabulary to the sanitizer's field names so the
    * model sees the right numbers regardless of which regression
    * command produced them:
    *
    *   logit/probit/poisson  -> e(r2_p) -> pseudo_r_squared
    *                            e(ll)   -> log_likelihood
    *                            e(chi2) -> chi_squared
    *                            e(p)    -> chi_squared_p_value
    *   stcox                  -> e(ll)     -> log_likelihood
    *                            e(chi2)   -> chi_squared
    *                            e(p)      -> chi_squared_p_value
    *                            e(N_sub)  -> n_subjects
    *                            e(N_fail) -> n_failures
    *
    * Concordance for stcox isn't in e() automatically — it requires
    * a follow-up `estat concordance` call. We don't emit it here;
    * the researcher's script can call estat and stash the value into
    * the next nora_result_regress invocation if they want it surfaced.
    if "`e(r2_p)'" != "" & !missing(`=e(r2_p)') {
        local _x = strofreal(`=e(r2_p)', "%21.17e")
        file write `fh' `","pseudo_r_squared":`_x'"'
    }
    if "`e(ll)'" != "" & !missing(`=e(ll)') {
        local _x = strofreal(`=e(ll)', "%21.17e")
        file write `fh' `","log_likelihood":`_x'"'
    }
    * Chi-squared omnibus test (LR / Wald) — populated by logit /
    * probit / Poisson / stcox via ``e(chi2)`` and its p-value
    * via ``e(p)``. Gate ``chi_squared_p_value`` on ``e(chi2)``
    * being populated, NOT on ``e(p)`` alone — Stata's ``regress``
    * (OLS) ALSO populates ``e(p)`` with the F-test p-value, so a
    * lone ``e(p)`` guard would emit ``chi_squared_p_value`` from
    * an OLS run that has no chi-squared test, misleading the
    * reader. The OLS path captures ``e(p)`` as ``f_p_value`` in
    * the F-statistic block above; this one only fires when a
    * chi2 test actually ran.
    if "`e(chi2)'" != "" & !missing(`=e(chi2)') {
        local _x = strofreal(`=e(chi2)', "%21.17e")
        file write `fh' `","chi_squared":`_x'"'
        if "`e(p)'" != "" & !missing(`=e(p)') {
            local _x = strofreal(`=e(p)', "%21.17e")
            file write `fh' `","chi_squared_p_value":`_x'"'
        }
    }
    if "`e(N_sub)'" != "" & !missing(`=e(N_sub)') {
        file write `fh' `","n_subjects":`=e(N_sub)'"'
    }
    if "`e(N_fail)'" != "" & !missing(`=e(N_fail)') {
        file write `fh' `","n_failures":`=e(N_fail)'"'
    }

    * Collinearity diagnostics. R and Python emit ``vif`` and
    * ``condition_number`` automatically; the Stata side used to
    * skip them entirely on the rationale that ``estat vif`` is a
    * display-only command (no clean r() vocabulary). That left
    * Stata regression cards missing diagnostics R/Python users see,
    * with no signal to the model that they're available — just
    * silent absence.
    *
    * Both are now derived from e() directly, no estat parsing:
    *
    *   condition_number = sqrt(λ_max / λ_min) of e(V). Scaling by
    *     σ² doesn't change eigenvalue ratios, so this equals the
    *     Belsley-Kuh-Welsch condition index of X'X. Computed via
    *     ``matrix symeigen`` (real eigenvalues for the symmetric
    *     V); guarded against non-PD edge cases (the capture
    *     swallows symeigen's failure on degenerate fits and we
    *     simply omit the field).
    *
    *   vif (per non-intercept predictor) = SE²_j · TSS_j / σ²,
    *     where TSS_j = Var(x_j) · (N_j - 1) under the regression
    *     sample. Algebraically identical to ``estat vif`` but
    *     reads off e() and the live data — no command output to
    *     parse. Only meaningful for OLS, so gated on
    *     ``e(cmd) == "regress"``; logit / probit / Poisson use
    *     pseudo-R² and chi² omnibus stats instead.
    * Condition number on the ESTIMABLE submatrix only.
    *
    * Factor-variable models (``regress y i.foreign mpg``) populate
    * e(V) with a row/column for the structurally omitted base level
    * whose variance is exactly 0. Running symeigen across the full
    * e(V) puts a 0 eigenvalue into the spectrum, the ``_emin > 0``
    * guard fails, and condition_number is silently dropped — even
    * though the *estimable* design has a finite condition number.
    * The fix is to build a square submatrix of e(V) restricted to
    * columns whose diagonal variance is strictly positive, then run
    * symeigen on that. Non-estimable / dropped / base columns drop
    * out cleanly and the resulting eigenvalue spectrum reflects the
    * actual design.
    *
    * Restricted to classical OLS. The "eigenvalues of e(V) ratio
    * equals the Belsley-Kuh-Welsch condition index of X'X"
    * argument relies on e(V) being proportional to (X'X)^-1 —
    * which only holds for ``regress`` without a robust / cluster
    * VCE. With ``vce(robust)`` or ``vce(cluster ...)``, e(V) is
    * the sandwich/cluster-robust covariance and its eigenvalues
    * are NOT the design-matrix eigenvalues. For GLMs
    * (logit/probit/Poisson) e(V) comes from the score Hessian,
    * also unrelated to (X'X)^-1. Publishing
    * ``condition_number`` in either case would silently report
    * the wrong number rather than omit. ``e(vce)`` is empty for
    * plain ``regress`` and "ols" if vce(ols) was explicit; both
    * forms count as classical.
    local _classical_ols = ("`e(cmd)'" == "regress") & ///
        (("`e(vce)'" == "") | ("`e(vce)'" == "ols"))
    local _kest = 0
    forvalues i = 1/`k' {
        if !missing(`Vmat'[`i', `i']) & `Vmat'[`i', `i'] > 0 {
            local _kest = `_kest' + 1
            local _eidx`_kest' = `i'
        }
    }
    if `_classical_ols' & `_kest' >= 1 {
        tempname _Vsub
        matrix `_Vsub' = J(`_kest', `_kest', 0)
        forvalues a = 1/`_kest' {
            local _ia = `_eidx`a''
            forvalues b = 1/`_kest' {
                local _ib = `_eidx`b''
                matrix `_Vsub'[`a', `b'] = `Vmat'[`_ia', `_ib']
            }
        }
        tempname _evals _evecs
        capture matrix symeigen `_evecs' `_evals' = `_Vsub'
        if !_rc {
            local _ne = colsof(`_evals')
            if `_ne' > 0 {
                local _emax = `_evals'[1, 1]
                local _emin = `_evals'[1, 1]
                forvalues j = 2/`_ne' {
                    if !missing(`_evals'[1, `j']) {
                        if `_evals'[1, `j'] > `_emax' local _emax = `_evals'[1, `j']
                        if `_evals'[1, `j'] < `_emin' local _emin = `_evals'[1, `j']
                    }
                }
                if !missing(`_emin') & !missing(`_emax') & `_emin' > 0 {
                    local _cn = sqrt(`_emax' / `_emin')
                    local _x = strofreal(`_cn', "%21.17e")
                    file write `fh' `","condition_number":`_x'"'
                }
            }
        }
    }

    * VIF: same restriction as condition_number. ``SE_j² · TSS_j /
    * σ²`` reproduces ``estat vif`` only when e(V) = σ² (X'X)^-1.
    * Under vce(robust) or vce(cluster ...), e(V) is the sandwich
    * covariance and the diagonal entries no longer factor as
    * ``σ² / (1 - R²_j) / TSS_j``. Omit VIF in those cases rather
    * than publish a wrong number.
    if `_classical_ols' & "`e(rmse)'" != "" & !missing(`=e(rmse)') & `=e(rmse)' > 0 {
        local _rmse2 = `=e(rmse)'^2
        * Build the field even if every predictor falls through to
        * "skip"; the renderer is happy with an empty dict and
        * shipping nothing here would suggest VIF wasn't computed
        * at all on a regression where it's well-defined.
        file write `fh' `","vif":{"'
        local _vfirst = 1
        forvalues i = 1/`k' {
            local v : word `i' of `vnames'
            if "`v'" == "_cons" continue
            if missing(`Vmat'[`i', `i']) continue
            * Factor-variable expansions like ``1.foreign`` or
            * ``1.year#2.sector`` are NOT valid Stata variable
            * references — ``count if !missing(1.foreign)`` errors,
            * which would halt the helper before the payload is
            * written. ``regress y i.x z`` is mainstream usage, so
            * the helper must skip these gracefully instead of
            * crashing. A name containing ``.`` or ``#`` is a
            * factor expansion; numeric/letter-only names are
            * ordinary variables that count/summarize can handle.
            if regexm("`v'", "[.#]") continue
            capture quietly count if e(sample) & !missing(`v')
            if _rc continue
            local _nj = r(N)
            if `_nj' < 2 continue
            capture quietly summarize `v' if e(sample)
            if _rc continue
            local _varj = r(Var)
            if missing(`_varj') | `_varj' == 0 continue
            local _tssj = `_varj' * (`_nj' - 1)
            local _vifj = `Vmat'[`i', `i'] * `_tssj' / `_rmse2'
            if missing(`_vifj') continue
            if !`_vfirst' file write `fh' ","
            local _x = strofreal(`_vifj', "%21.17e")
            file write `fh' `""`v'":`_x'"'
            local _vfirst = 0
        }
        file write `fh' "}"
    }

    * AIC / BIC via ``estat ic``. Stata stores them in a 1x6 matrix
    * ``r(S)`` with columns [N, ll0, ll, df, AIC, BIC] — neither e()
    * nor ``estat ic``'s scalar returns expose them directly. R and
    * Python both ship aic/bic on every regression card; without
    * this block Stata payloads drop both fields and the model can't
    * compare model fit across estimators that don't share R².
    * Placed AFTER the VIF block: ``count`` and ``summarize`` inside
    * VIF write to ``r()`` and ``estat ic`` would clobber that
    * state. Inside ``capture`` because ``estat ic`` errors on a
    * handful of estimators (no log-likelihood, perfect fit, etc.);
    * the helper omits the field rather than aborting the script.
    tempname _icS
    capture quietly estat ic
    if !_rc {
        capture matrix `_icS' = r(S)
        if !_rc & rowsof(`_icS') >= 1 & colsof(`_icS') >= 6 {
            if !missing(`_icS'[1, 5]) {
                local _x = strofreal(`_icS'[1, 5], "%21.17e")
                file write `fh' `","aic":`_x'"'
            }
            if !missing(`_icS'[1, 6]) {
                local _x = strofreal(`_icS'[1, 6], "%21.17e")
                file write `fh' `","bic":`_x'"'
            }
        }
    }

    * Harrell's C-index for stcox. ``concordance`` is not in
    * ``e()`` after ``stcox`` — Stata requires the follow-up
    * ``estat concordance`` call, which sets ``r(C)``. R's
    * coxph emits this via ``summary()$concordance``; without it
    * the Stata Cox card was missing the standard "C =" diagnostic
    * researchers report alongside hazard ratios.
    * ``stcox`` posts ``e(cmd) == "cox"`` (the "st" is the
    * survival-time prefix syntax, not the e(cmd) value). Gating
    * on "stcox" would silently skip every Cox fit. Verified
    * empirically against Stata 19.5.
    if "`e(cmd)'" == "cox" {
        capture quietly estat concordance
        if !_rc & "`r(C)'" != "" & !missing(`=r(C)') {
            local _x = strofreal(`=r(C)', "%21.17e")
            file write `fh' `","concordance":`_x'"'
        }
    }

    * Cluster-robust SE metadata. ``vce(cluster id)`` populates
    * ``e(vce) == "cluster"``, ``e(clustvar)`` (single name), and
    * ``e(N_clust)`` (cluster count). Same disclosure profile as
    * the fixed-effects block: emit the variable NAME (already in
    * the schema) and the cluster CARDINALITY; do NOT emit the
    * cluster labels themselves. Helper-side decision pinned in
    * ``docs/direction.md`` — bounded aggregates go in the existing
    * allowlist (``cluster_variables`` plural list,
    * ``n_clusters`` dict-of-counts), not in a new sub-shape.
    *
    * Stata exposes single-dimension clustering through these
    * macros; Cameron-Gelbach-Miller two-way is available via
    * ``cgmreg`` or ``reghdfe`` (third-party), not core ``regress``.
    * The helper handles the single-cluster case from core Stata;
    * two-way emission from third-party commands would follow the
    * same pattern when populated.
    if "`e(vce)'" == "cluster" & "`e(clustvar)'" != "" {
        file write `fh' `","robust_se_type":"cluster""'
        file write `fh' `","cluster_variables":["`e(clustvar)'"]"'
        if "`e(N_clust)'" != "" & !missing(`=e(N_clust)') {
            file write `fh' `","n_clusters":{"`e(clustvar)'":`=e(N_clust)'}"'
        }
    }
    else {
        * Non-cluster variance estimators. Map Stata's ``e(vce)`` /
        * ``e(cmd)`` vocabulary onto the sanitizer's canonical
        * ``robust_se_type`` enum so the model can tell at a glance
        * which variance flavour produced the SEs. Classical /
        * unset / OLS is omitted (absence already implies
        * model-based SEs). Each emit goes through a tiny
        * if-else ladder rather than an interpolated local so a
        * future spelling drift doesn't leak free text past the
        * sanitizer's enum gate.
        local _rse = ""
        if "`e(cmd)'" == "newey" {
            local _rse = "hac_newey_west"
        }
        else if "`e(vce)'" == "robust" | "`e(vce)'" == "hc1" {
            local _rse = "hc1"
        }
        else if "`e(vce)'" == "hc2" {
            local _rse = "hc2"
        }
        else if "`e(vce)'" == "hc3" {
            local _rse = "hc3"
        }
        else if "`e(vce)'" == "bootstrap" {
            local _rse = "bootstrap"
        }
        if "`_rse'" != "" {
            file write `fh' `","robust_se_type":"`_rse'""'
        }
    }

    * fixed_effects — absorbed FE dimension cardinality.
    *
    *   xtreg, fe:  e(ivar) names the panel id, e(N_g) carries the
    *               panel-dim cardinality (= the absorbed FE count).
    *   areg:       e(absvar) names the absorbed variable, e(df_a)
    *               carries (N_groups - 1) after dropping one base
    *               level, so we report e(df_a) + 1 as the level count.
    *
    * Listing the actual levels is forbidden — only cardinality
    * crosses the boundary. This mirrors the fixest-FE handling in
    * R's from_lm (helper emits sizes via model$fixef_sizes; never
    * the level labels).
    if "`e(cmd)'" == "xtreg" & "`e(ivar)'" != "" & "`e(N_g)'" != "" & !missing(`=e(N_g)') {
        file write `fh' `","fixed_effects":{"`e(ivar)'":`=e(N_g)'}"'
    }
    else if "`e(cmd)'" == "areg" & "`e(absvar)'" != "" & "`e(df_a)'" != "" & !missing(`=e(df_a)') {
        local _nlevels = `=e(df_a)' + 1
        file write `fh' `","fixed_effects":{"`e(absvar)'":`_nlevels'}"'
    }

    * Panel-data post-estimation diagnostics. ``xtreg, fe`` stores
    * the F-test on the joint significance of the panel-level FE
    * in ``e(F_f)`` with its denominator df in ``e(df_a)``. The
    * test is "are the unit fixed effects jointly zero" — a small
    * F means pooled OLS suffices; a big F says the FE matter.
    * Mirrors R's ``plm::pFtest`` and the sanitizer's
    * ``f_test_fe_chi2`` slot.
    *
    * Breusch-Pagan LM (RE vs pooled) and Wooldridge AR(1) come
    * from ``xttest0`` and ``xtserial`` post-estimation commands;
    * those reset ``r()`` and would clobber prior state, so the
    * researcher runs them in their script and passes the chi² + p
    * via this helper's caller — same pattern Cox concordance uses
    * with ``estat concordance``. Future enhancement: run them
    * inside ``capture quietly`` here, after the VIF / estat-ic
    * blocks finish their state writes.
    if "`e(cmd)'" == "xtreg" & "`e(F_f)'" != "" & !missing(`=e(F_f)') {
        local _x = strofreal(`=e(F_f)', "%21.17e")
        file write `fh' `","f_test_fe_chi2":`_x'"'
        * Compute the p-value via Ftail when df components are
        * available. ``e(df_a)`` is the absorbed-FE df (numerator
        * minus 1); ``e(df_r)`` is the residual df.
        if "`e(df_a)'" != "" & !missing(`=e(df_a)') ///
                & "`e(df_r)'" != "" & !missing(`=e(df_r)') {
            local _fp = Ftail(`=e(df_a)', `=e(df_r)', `=e(F_f)')
            if !missing(`_fp') {
                local _x = strofreal(`_fp', "%21.17e")
                file write `fh' `","f_test_fe_p":`_x'"'
            }
        }
    }

    * Mixed-effects variance components, group counts, fit method, ICC.
    *
    * Matches the contract R's from_lm (lme4::merMod) and Python's
    * from_lm (statsmodels MixedLMResultsWrapper) emit through the same
    * sanitizer bucket:
    *   random_effects_variance: {group: var_intercept,
    *                             group.slope_term: var_slope,
    *                             residual: sigma_e^2}
    *   n_groups_per_level:      {group: count}
    *   fit_method:              "REML" / "ML"
    *   icc:                     single-grouping intercept-only case
    *
    * Why we read e(b) column metadata instead of `estat recovariance`:
    *   - `estat recovariance` posts r(Cov<N>) matrices in newer Stata
    *     (the underscored r(Cov_<N>) form earlier versions of this
    *     helper used does not exist on Stata 18/19 — the capture
    *     swallowed the rc and the random_effects_variance field
    *     silently disappeared).
    *   - `estat recovariance` is "not valid" after meglm (gsem-backed
    *     in current Stata), so a single uniform path can't be built
    *     on it anyway.
    *   - e(sigma_e) is empty on at least StataNow 19.5 even for
    *     plain Gaussian `mixed` — the residual SD lives in
    *     e(b)["lnsig_e":_cons] as log(sigma_e). Reading e(b) directly
    *     avoids the version drift entirely.
    *
    * Parameterization in e(b):
    *   mixed (and gsem when called as a non-mixed sem path):
    *     equation = "lns<L>_<E>_<T>"  → log-SD of the T-th term in
    *     equation E at level L; variance = exp(2 * coef). Level 1
    *     is innermost grouping; L = n_grouping_levels is outermost.
    *     equation = "lnsig_e"         → log(sigma_e); residual
    *     variance = exp(2 * coef).
    *     equation = "atr<...>"        → atanh of a correlation; we
    *     skip these (sanitizer contract is variances only).
    *   meglm (gsem internal):
    *     equation = "/"  with column-name pattern "var(<term>[<group>])"
    *     → variance DIRECTLY (no transform). Random slopes show up
    *     as "var(<slopevar>[<group>])"; correlations as
    *     "cov(<term1>[<group>],<term2>[<group>])" — skipped.
    *
    * Term-to-key mapping:
    *   intercept (T=1 for mixed; "_cons" inside var(...) for meglm) →
    *       group name, e.g. "school".
    *   slope (T>1 for mixed; slope var name for meglm) →
    *       "group.term", e.g. "school.x".
    * For mixed we can't recover the slope variable name from
    * "lns<L>_<E>_<T>" alone (T is just an index into the level's
    * random-effect equation). For the intercept-only case the audit
    * actually covers, T=1 maps to the group name cleanly; if random
    * slopes are present we fall back to "group.term<T>" so the key
    * is at least disambiguating rather than colliding.
    if `_is_mixed_re' {
        local _ivars = "`e(ivars)'"
        local _n_levels : word count `_ivars'

        * Walk e(b) once, classifying each column by its equation
        * prefix. Buffer pairs into a local macro so we can omit the
        * random_effects_variance field entirely when nothing
        * survives (matches R/Python which never emit empty dicts).
        local _re_pairs ""
        local _residual_var = .
        if `_n_levels' > 0 {
            local _coleq : coleq e(b)
            local _colnm : colnames e(b)
            local _ncols : word count `_coleq'
            forvalues i = 1/`_ncols' {
                local _eq : word `i' of `_coleq'
                local _cn : word `i' of `_colnm'

                * mixed: lns<L>_<E>_<T> — log-SD parameter.
                if regexm("`_eq'", "^lns([0-9]+)_([0-9]+)_([0-9]+)$") {
                    local _L = real(regexs(1))
                    local _T = real(regexs(3))
                    if missing(e(b)[1, `i']) continue
                    local _coef = e(b)[1, `i']
                    local _v = exp(2 * `_coef')
                    if missing(`_v') | `_v' <= 0 continue
                    * Map level L (innermost=1) to outermost-first
                    * e(ivars) word. word_idx = n_levels - L + 1.
                    local _word_idx = `_n_levels' - `_L' + 1
                    if `_word_idx' < 1 | `_word_idx' > `_n_levels' continue
                    local _gname : word `_word_idx' of `_ivars'
                    local _key = "`_gname'"
                    if `_T' > 1 {
                        local _key = "`_gname'.term`_T'"
                    }
                    local _x = strofreal(`_v', "%21.17e")
                    local _re_pairs `"`_re_pairs',"`_key'":`_x'"'
                    continue
                }

                * mixed: lnsig_e — log-SD of residual.
                if "`_eq'" == "lnsig_e" {
                    if missing(e(b)[1, `i']) continue
                    local _v = exp(2 * e(b)[1, `i'])
                    if missing(`_v') | `_v' <= 0 continue
                    local _residual_var = `_v'
                    continue
                }

                * meglm (gsem-backed): "/" equation with var(...) name
                * is the natural-scale variance, no transform.
                if "`_eq'" == "/" & regexm("`_cn'", "^var\(([^\[]+)\[([^\]]+)\]\)$") {
                    local _term = regexs(1)
                    local _grp = regexs(2)
                    if missing(e(b)[1, `i']) continue
                    local _v = e(b)[1, `i']
                    if `_v' <= 0 continue
                    local _key = "`_grp'"
                    if "`_term'" != "_cons" {
                        local _key = "`_grp'.`_term'"
                    }
                    local _x = strofreal(`_v', "%21.17e")
                    local _re_pairs `"`_re_pairs',"`_key'":`_x'"'
                    continue
                }
            }
        }
        if !missing(`_residual_var') {
            local _x = strofreal(`_residual_var', "%21.17e")
            local _re_pairs `"`_re_pairs',"residual":`_x'"'
        }
        if `"`_re_pairs'"' != "" {
            local _re_body = substr(`"`_re_pairs'"', 2, .)
            file write `fh' `","random_effects_variance":{`_re_body'}"'
        }

        * n_groups_per_level — same disclosure profile as fixed_effects
        * (column name + cardinality, never the level identities).
        if `_n_levels' > 0 {
            tempname _ng
            capture matrix `_ng' = e(N_g)
            if !_rc & colsof(`_ng') >= `_n_levels' {
                local _ng_pairs ""
                forvalues lev = 1/`_n_levels' {
                    local _gname : word `lev' of `_ivars'
                    if missing(`_ng'[1, `lev']) continue
                    local _n = `_ng'[1, `lev']
                    local _ng_pairs `"`_ng_pairs',"`_gname'":`=`_n''"'
                }
                if `"`_ng_pairs'"' != "" {
                    local _ng_body = substr(`"`_ng_pairs'"', 2, .)
                    file write `fh' `","n_groups_per_level":{`_ng_body'}"'
                }
            }
        }

        * Fit method. mixed: e(method) is "REML" or "ML" (set by the
        * default vs , mle option — but the default flipped to ML
        * somewhere around Stata 18, so the audit do-file pins ,reml
        * to keep the contract deterministic). meglm: nonlinear-link
        * mixed models are always ML (REML isn't defined there), so
        * hardcode regardless of what e(method) returns (meglm's
        * e(method) reports the integration scheme, not REML/ML).
        local _fm = ""
        if "`e(cmd)'" == "mixed" {
            if "`e(method)'" == "REML" | "`e(method)'" == "ML" {
                local _fm = "`e(method)'"
            }
        }
        else if "`e(cmd)'" == "meglm" | "`e(cmd2)'" == "meglm" {
            local _fm = "ML"
        }
        if "`_fm'" != "" {
            file write `fh' `","fit_method":"`_fm'""'
        }

        * ICC — only well-defined for single-grouping, intercept-only
        * random-effect specifications, AND a residual variance to
        * divide into. Gate on n_levels == 1 (estat icc on multilevel
        * fits posts r(icc1), r(icc2), ... which mean something
        * different) and on having found a residual variance above
        * (Gaussian only — meglm has no residual term, and the gate
        * via _residual_var is the right signal regardless of which
        * Stata version populated e(sigma_e)).
        if `_n_levels' == 1 & !missing(`_residual_var') {
            capture quietly estat icc
            if !_rc {
                * Stata 19's estat icc returns r(icc<N>) where N is
                * the model level: r(icc2) for a single-grouping
                * fit (level 1 is the residual). Older Stata used
                * r(icc1) for the same quantity. Try the modern
                * shape first; fall back so the helper stays
                * version-stable.
                local _icc = .
                if "`r(icc2)'" != "" & !missing(`=r(icc2)') {
                    local _icc = `=r(icc2)'
                }
                else if "`r(icc1)'" != "" & !missing(`=r(icc1)') {
                    local _icc = `=r(icc1)'
                }
                if !missing(`_icc') {
                    local _x = strofreal(`_icc', "%21.17e")
                    file write `fh' `","icc":`_x'"'
                }
            }
        }
    }

    * vcov — full variance-covariance matrix of the coefficient
    * estimates. Diagonals equal SE²; off-diagonals enable Wald /
    * joint-significance tests the model can run itself. Pure
    * aggregate from sigma² · (X'X)^-1 (or the sandwich estimator
    * under robust/cluster) — no per-observation leak. R and Python
    * both ship it; Stata used to drop it entirely.
    *
    * Emitted as a nested {row: {col: value}} dict so the sanitizer's
    * dict-of-dict vcov handler can clamp precision and run the
    * symmetry / diagonal=SE² invariant check. Missing diagonals
    * (collinear / dropped columns) are written as ``null`` per the
    * same convention coefficients / standard_errors use.
    if `k' > 0 {
        file write `fh' `","vcov":{"'
        local _rfirst = 1
        forvalues i = 1/`k' {
            local vi : word `i' of `vnames'
            if !`_rfirst' file write `fh' ","
            file write `fh' `""`vi'":{"'
            local _cfirst = 1
            forvalues j = 1/`k' {
                local vj : word `j' of `vnames'
                if !`_cfirst' file write `fh' ","
                if missing(`Vmat'[`i', `j']) {
                    file write `fh' `""`vj'":null"'
                }
                else {
                    local _x = strofreal(`Vmat'[`i', `j'], "%21.17e")
                    file write `fh' `""`vj'":`_x'"'
                }
                local _cfirst = 0
            }
            file write `fh' "}"
            local _rfirst = 0
        }
        file write `fh' "}"
    }

    file write `fh' "}" _newline
    file close `fh'

    display as text "nora_result_regress: wrote result to " as result "`path'"
end
