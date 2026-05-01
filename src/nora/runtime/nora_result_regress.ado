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
    file write `fh' `","n":`=e(N)'"'

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
        if missing(`Vmat'[`i', `i']) {
            file write `fh' `""`v'":null"'
        }
        else {
            local se_str = strofreal(sqrt(`Vmat'[`i', `i']), "%21.17e")
            file write `fh' `""`v'":`se_str'"'
        }
        local first = 0
    }
    file write `fh' "}"

    * p_values (named dict, two-sided t-test against e(df_r)). Skipped
    * when the estimator didn't populate e(df_r) — the renderer drops
    * the p-value column when the dict is absent. Dropped/collinear
    * terms have SE=0, which makes b/se missing; emit JSON `null` for
    * those rather than Stata's "." (not valid JSON).
    if "`e(df_r)'" != "" {
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
                scalar `_p' = 2 * ttail(`=e(df_r)', `_t')
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
    if "`e(F)'" != "" & !missing(`=e(F)') {
        local _x = strofreal(`=e(F)', "%21.17e")
        file write `fh' `","f_statistic":`_x'"'
    }
    if "`e(rmse)'" != "" & !missing(`=e(rmse)') {
        local _x = strofreal(`=e(rmse)', "%21.17e")
        file write `fh' `","residual_std_error":`_x'"'
    }
    if "`e(df_r)'" != "" & !missing(`=e(df_r)') {
        file write `fh' `","degrees_of_freedom":`=e(df_r)'"'
    }

    file write `fh' "}" _newline
    file close `fh'

    display as text "nora_result_regress: wrote result to " as result "`path'"
end
