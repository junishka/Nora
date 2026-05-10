*! version 0.0.4  Nora runtime: emit a linear_regression payload from e().
*!
*! Call after a regression command (regress, logit, etc. — anything that
*! populates e(b), e(V), e(N), e(depvar)). Writes the structured payload
*! to the path in $NORA_RESULT_PATH so the Nora executor can pick
*! it up and route through the sanitizer.
*!
*! Usage:
*!   regress y x1 x2
*!   nora_result_regress, label("OLS income ~ edu+age")
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

    tempname fh
    file open `fh' using `"`path'"', write text append

    file write `fh' `"{"type":"linear_regression""'
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
    local _kest = 0
    forvalues i = 1/`k' {
        if !missing(`Vmat'[`i', `i']) & `Vmat'[`i', `i'] > 0 {
            local _kest = `_kest' + 1
            local _eidx`_kest' = `i'
        }
    }
    if `_kest' >= 1 {
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

    if "`e(cmd)'" == "regress" & "`e(rmse)'" != "" & !missing(`=e(rmse)') & `=e(rmse)' > 0 {
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

    file write `fh' "}" _newline
    file close `fh'

    display as text "nora_result_regress: wrote result to " as result "`path'"
end
