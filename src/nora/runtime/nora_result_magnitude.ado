*! version 0.0.1  Nora runtime: emit a magnitude_table payload.
*!
*! Computes sum (or mean) of `value_var` by `group_var`, together with
*! the per-group observation count and the largest-contributor share —
*! the dominance metric Nora's sanitizer consults to apply the
*! (1, k)-dominance rule. `max_share` is required by the payload
*! schema but NEVER forwarded to Claude; the sanitizer strips it
*! after deciding whether to suppress.
*!
*! Usage:
*!   nora_result_magnitude state income, label("Total income by state")
*!   nora_result_magnitude state income, aggregation(mean)
*!
*! Missing values in either `group_var` or `value_var` are dropped
*! before aggregation — the sum-of-cell-n in the output therefore
*! reflects non-missing rows. Nora's row-count-change check will
*! flag if that total is less than the source dataset's N.

program define nora_result_magnitude
    version 13
    syntax varlist(min=2 max=2) [, aggregation(string) label(string) ]

    * JSON-escape `label` (Claude-controllable free text). See
    * nora_result_regress for the full explanation of this pattern.
    local label : subinstr local label "\" "\\", all
    local label : subinstr local label `"""' `"\""', all
    local label : subinstr local label "`=char(10)'" " ", all
    local label : subinstr local label "`=char(13)'" " ", all
    local label : subinstr local label "`=char(9)'" " ", all

    local path : env NORA_RESULT_PATH
    if "`path'" == "" {
        display as error "NORA_RESULT_PATH not set. Run through Nora."
        exit 198
    }

    * Per-run authenticity token. See nora_result_regress.ado for the
    * full explanation.
    local _nora_token : env NORA_RUN_TOKEN
    if "`_nora_token'" == "" {
        display as error "NORA_RUN_TOKEN not set. Run through Nora."
        exit 198
    }

    * Parse positional args: first var is the group, second is the value.
    tokenize `varlist'
    local group_var "`1'"
    local value_var "`2'"

    * Validate that value_var is numeric — sums / means over a string
    * variable are meaningless.
    capture confirm numeric variable `value_var'
    if _rc {
        display as error "nora_result_magnitude: value_var `value_var' must be numeric."
        exit 198
    }

    * Default + validate aggregation.
    if "`aggregation'" == "" {
        local aggregation "sum"
    }
    if !inlist("`aggregation'", "sum", "mean") {
        display as error ///
            "nora_result_magnitude: aggregation must be 'sum' or 'mean'; got `aggregation'."
        exit 198
    }

    preserve
    quietly drop if missing(`group_var') | missing(`value_var')

    * Per-group dominance metric BEFORE collapsing. Using absolute values
    * handles mixed-sign data: max_share = max|x| / sum|x|. Guard against
    * all-zero groups (undefined ratio) with cond().
    tempvar _abs _maxabs _sumabs _mshare
    quietly gen double `_abs' = abs(`value_var')
    quietly bysort `group_var': egen double `_maxabs' = max(`_abs')
    quietly bysort `group_var': egen double `_sumabs' = total(`_abs')
    quietly gen double `_mshare' = cond(`_sumabs' > 0, `_maxabs' / `_sumabs', 0)

    * Collapse to one row per group. `_value' is sum or mean per the
    * aggregation argument; `_n_obs' is the group size; `_mshare' is
    * constant within group so (first) retrieves it.
    if "`aggregation'" == "sum" {
        quietly collapse (sum) _value=`value_var' (count) _n_obs=`value_var' ///
            (first) _mshare=`_mshare', by(`group_var')
    }
    else {
        quietly collapse (mean) _value=`value_var' (count) _n_obs=`value_var' ///
            (first) _mshare=`_mshare', by(`group_var')
    }

    tempname fh
    file open `fh' using `"`path'"', write text append

    file write `fh' `"{"type":"magnitude_table""'
    file write `fh' `","_token":"`_nora_token'""'
    if `"`label'"' != "" {
        file write `fh' `","label":"`label'""'
    }
    file write `fh' `","row_variable":"`group_var'""'
    file write `fh' `","value_variable":"`value_var'""'
    file write `fh' `","aggregation":"`aggregation'""'
    file write `fh' `","cells":{"'

    local first = 1
    forvalues i = 1/`=_N' {
        if !`first' file write `fh' ","
        local grp = `group_var'[`i']
        * JSON-escape group values from the data: backslash first, then
        * double-quote, then collapse CR/LF/TAB to spaces (JSON forbids
        * literal control chars in string values).
        local grp_esc : subinstr local grp `"\"' `"\\"', all
        local grp_esc : subinstr local grp_esc `"""' `"\""', all
        local grp_esc : subinstr local grp_esc "`=char(10)'" " ", all
        local grp_esc : subinstr local grp_esc "`=char(13)'" " ", all
        local grp_esc : subinstr local grp_esc "`=char(9)'" " ", all

        * Use scientific notation (%21.17e) for floats — always valid JSON
        * regardless of magnitude; Stata's default format drops leading
        * zeros on values < 1 which would break json.loads.
        local v_str = strofreal(_value[`i'], "%21.17e")
        local ms_str = strofreal(_mshare[`i'], "%21.17e")
        local n_int = _n_obs[`i']

        file write `fh' `""`grp_esc'":{"value":`v_str',"n":`n_int',"max_share":`ms_str'}"'
        local first = 0
    }
    file write `fh' "}}" _newline
    file close `fh'

    restore
    display as text "nora_result_magnitude: wrote result to " as result "`path'"
end
