*! version 0.0.2  Nora runtime: emit a linear_regression payload from e().
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
    file open `fh' using `"`path'"', write text replace

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

    * coefficients (named dict, integer-indexed into e(b)).
    file write `fh' `","coefficients":{"'
    local first = 1
    forvalues i = 1/`k' {
        local v : word `i' of `vnames'
        if !`first' file write `fh' ","
        local coef_str = strofreal(`bmat'[1, `i'], "%21.17e")
        file write `fh' `""`v'":`coef_str'"'
        local first = 0
    }
    file write `fh' "}"

    * standard_errors (named dict, from sqrt(diag(V))).
    file write `fh' `","standard_errors":{"'
    local first = 1
    forvalues i = 1/`k' {
        local v : word `i' of `vnames'
        if !`first' file write `fh' ","
        local se_str = strofreal(sqrt(`Vmat'[`i', `i']), "%21.17e")
        file write `fh' `""`v'":`se_str'"'
        local first = 0
    }
    file write `fh' "}"

    * Optional fit statistics. Missing e() macros mean the command didn't
    * populate them (e.g. robust SE paths change what's in e()), so we
    * only emit fields we actually have. All floats go through the
    * same scientific-notation formatter for JSON-safety.
    if "`e(r2)'" != "" {
        local _x = strofreal(`=e(r2)', "%21.17e")
        file write `fh' `","r_squared":`_x'"'
    }
    if "`e(r2_a)'" != "" {
        local _x = strofreal(`=e(r2_a)', "%21.17e")
        file write `fh' `","adj_r_squared":`_x'"'
    }
    if "`e(F)'" != "" {
        local _x = strofreal(`=e(F)', "%21.17e")
        file write `fh' `","f_statistic":`_x'"'
    }
    if "`e(rmse)'" != "" {
        local _x = strofreal(`=e(rmse)', "%21.17e")
        file write `fh' `","residual_std_error":`_x'"'
    }
    if "`e(df_r)'" != "" {
        file write `fh' `","degrees_of_freedom":`=e(df_r)'"'
    }

    file write `fh' "}"
    file close `fh'

    display as text "nora_result_regress: wrote result to " as result "`path'"
end
