*! version 0.0.1  Nora runtime: emit a correlation_matrix payload.
*!
*! Usage:
*!   nora_result_correlation x1 x2 x3, label("X correlations") method("pearson")
*!
*! Picks the correlation method from the syntax option; runs the
*! corresponding Stata command (correlate / spearman / ktau) on the
*! supplied variables, and writes the resulting matrix to
*! NORA_RESULT_PATH so the executor can route it through the
*! sanitizer.
*!
*! Method dispatch:
*!   pearson  -> ``correlate``           -> r(C) is the matrix
*!   spearman -> ``spearman, stats(rho)`` -> r(Rho) is the matrix
*!   kendall  -> ``ktau, stats(taub)``    -> r(Tau_b) is the matrix
*!
*! Sample N is the count of rows where ALL listed variables are
*! observed (pairwise / casewise differs across the three commands;
*! we recompute it with ``count if e(sample)`` analog so the
*! reported N is honest regardless of method).
*!
*! Caveats vs R / Python helpers:
*! - ``ktau`` on Stata is O(N²) and slow on large samples; that's a
*!   Stata constraint, not a Nora one.
*! - Stata ``spearman`` reports rho only when stats(rho) is set; we
*!   pass that explicitly so r() carries the matrix.

program define nora_result_correlation
    version 13
    syntax varlist(min=2 numeric) [, label(string) method(string)]

    if "`method'" == "" {
        local method "pearson"
    }
    if !inlist("`method'", "pearson", "spearman", "kendall") {
        display as error "nora_result_correlation: method must be pearson / spearman / kendall (got `method')"
        exit 198
    }

    * Same JSON-escape sequence used in nora_result_regress for the
    * label field. Variable names are Stata-syntax-shape-constrained
    * so they don't need escaping.
    local label : subinstr local label "\" "\\", all
    local label : subinstr local label `"""' `"\""', all
    local label : subinstr local label "`=char(10)'" " ", all
    local label : subinstr local label "`=char(13)'" " ", all
    local label : subinstr local label "`=char(9)'" " ", all

    local path : env NORA_RESULT_PATH
    if "`path'" == "" {
        display as error "NORA_RESULT_PATH not set. This script must be run through Nora."
        exit 198
    }
    local _nora_token : env NORA_RUN_TOKEN
    if "`_nora_token'" == "" {
        display as error "NORA_RUN_TOKEN not set. Run through Nora."
        exit 198
    }

    * Honest N: count rows where ALL listed variables are observed,
    * regardless of which Stata correlation command we invoke. The
    * three commands handle missing differently (pairwise vs
    * casewise vs filter); reporting the casewise N keeps the
    * payload's N honest.
    local _allobs ""
    foreach v of varlist `varlist' {
        local _allobs "`_allobs' & !missing(`v')"
    }
    * Trim leading " & ".
    local _allobs = substr("`_allobs'", 4, .)
    quietly count if `_allobs'
    local _ncomplete = r(N)
    quietly count
    local _ntotal = r(N)
    local _missing = `_ntotal' - `_ncomplete'

    * Compute the correlation matrix per method. For consistency
    * across methods, we restrict each command to the
    * casewise-complete sample so the off-diagonal cells are all
    * computed on the same N. Stata's ``correlate`` defaults to
    * casewise already; ``spearman`` and ``ktau`` default to
    * pairwise but we pass the if-clause explicitly to override.
    tempname Cmat
    if "`method'" == "pearson" {
        capture noisily quietly correlate `varlist' if `_allobs'
        if _rc {
            display as error "nora_result_correlation: correlate failed (rc=`_rc')"
            exit `_rc'
        }
        matrix `Cmat' = r(C)
    }
    else if "`method'" == "spearman" {
        capture noisily quietly spearman `varlist' if `_allobs', stats(rho)
        if _rc {
            display as error "nora_result_correlation: spearman failed (rc=`_rc')"
            exit `_rc'
        }
        matrix `Cmat' = r(Rho)
    }
    else {
        * kendall
        capture noisily quietly ktau `varlist' if `_allobs', stats(taub)
        if _rc {
            display as error "nora_result_correlation: ktau failed (rc=`_rc')"
            exit `_rc'
        }
        matrix `Cmat' = r(Tau_b)
    }

    local k = colsof(`Cmat')
    if `k' == 0 {
        display as error "nora_result_correlation: empty correlation matrix"
        exit 198
    }

    * Variable order: keep the user-supplied varlist order so the
    * emitted JSON's row/column ordering matches what they typed.
    * The matrix produced by Stata has its own ordering (typically
    * varlist order, but ktau's r(Tau_b) has been observed to
    * differ); look up by name from the matrix's colnames so we're
    * robust to that.
    local _cnames : colnames `Cmat'

    tempname fh
    file open `fh' using `"`path'"', write text append

    file write `fh' `"{"type":"correlation_matrix""'
    file write `fh' `","_token":"`_nora_token'""'
    if `"`label'"' != "" {
        file write `fh' `","label":"`label'""'
    }
    file write `fh' `","n":`_ncomplete'"'
    file write `fh' `","method":"`method'""'

    * variables: the canonical user-supplied varlist order.
    file write `fh' `","variables":["'
    local _first = 1
    foreach v of varlist `varlist' {
        if !`_first' file write `fh' ","
        file write `fh' `""`v'""'
        local _first = 0
    }
    file write `fh' "]"

    * correlations: nested dict {row: {col: value}}. Skip cells whose
    * value is missing — the sanitizer will reject the whole payload
    * if the matrix is degenerate, but pairwise NaNs (constant
    * columns) shouldn't fail the JSON write.
    file write `fh' `","correlations":{"'
    local _row_first = 1
    foreach rv of varlist `varlist' {
        * Find rv's index in the matrix's colnames.
        local _ri = 0
        local _idx = 0
        foreach _cn of local _cnames {
            local _idx = `_idx' + 1
            if "`_cn'" == "`rv'" {
                local _ri = `_idx'
            }
        }
        if `_ri' == 0 continue
        if !`_row_first' file write `fh' ","
        file write `fh' `""`rv'":{"'
        local _col_first = 1
        foreach cv of varlist `varlist' {
            local _ci = 0
            local _idx = 0
            foreach _cn of local _cnames {
                local _idx = `_idx' + 1
                if "`_cn'" == "`cv'" {
                    local _ci = `_idx'
                }
            }
            if `_ci' == 0 continue
            local _val = `Cmat'[`_ri', `_ci']
            if missing(`_val') continue
            if !`_col_first' file write `fh' ","
            local _x = strofreal(`_val', "%21.17e")
            file write `fh' `""`cv'":`_x'"'
            local _col_first = 0
        }
        file write `fh' "}"
        local _row_first = 0
    }
    file write `fh' "}"

    file write `fh' `","missing_count":`_missing'"'
    file write `fh' "}" _newline
    file close `fh'

    display as text "nora_result_correlation: wrote result to " as result "`path'"
end
