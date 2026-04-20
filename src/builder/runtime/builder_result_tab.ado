*! version 0.0.1  Builder runtime: emit frequency_table or crosstab from raw data.
*!
*! Pass one variable → 1-way frequency_table payload.
*! Pass two variables → 2-way crosstab payload.
*!
*! Usage:
*!   builder_result_tab state, label("State frequencies")
*!   builder_result_tab state gender, label("State x Gender")
*!
*! Unlike the other helpers, this one does NOT rely on a prior command
*! having populated r() — it computes everything itself from the raw
*! variable(s), using Stata's `contract` to get distinct levels + counts.
*! The script only needs the variable(s) in memory (i.e. the researcher
*! has already read their data with `use`, `insheet`, `import`, etc.).

program define builder_result_tab
    version 13
    syntax varlist(min=1 max=2) [, label(string) ]

    local path : env BUILDER_RESULT_PATH
    if "`path'" == "" {
        display as error "BUILDER_RESULT_PATH not set. Run through Builder."
        exit 198
    }

    local nvars : word count `varlist'
    if `nvars' == 1 {
        _builder_tab_1way `varlist', label(`"`label'"')
    }
    else {
        _builder_tab_2way `varlist', label(`"`label'"')
    }
end


* ---------------------------------------------------------------------------
* 1-way: frequency_table
* ---------------------------------------------------------------------------

program define _builder_tab_1way
    version 13
    syntax varname [, label(string) ]

    * JSON-escape `label` (Claude-controllable free text). See
    * builder_result_regress for the full explanation of this pattern.
    local label : subinstr local label "\" "\\", all
    local label : subinstr local label `"""' `"\""', all
    local label : subinstr local label "`=char(10)'" " ", all
    local label : subinstr local label "`=char(13)'" " ", all
    local label : subinstr local label "`=char(9)'" " ", all

    local vname "`varlist'"
    local path : env BUILDER_RESULT_PATH

    * Totals and missing count BEFORE contract (which drops missings).
    quietly count
    local total = r(N)
    quietly count if missing(`vname')
    local missing = r(N)

    preserve
    * Filter before contract so Stata's numeric missing code (.)
    * doesn't appear as a distinct level. `missing_count` above already
    * carries the correct NA count for the schema.
    quietly keep if !missing(`vname')
    quietly contract `vname'
    * After contract: one row per distinct non-missing value, `_freq' the count.

    tempname fh
    file open `fh' using `"`path'"', write text replace

    file write `fh' `"{"type":"frequency_table""'
    if `"`label'"' != "" {
        file write `fh' `","label":"`label'""'
    }
    file write `fh' `","variable":"`vname'""'
    file write `fh' `","n":`total'"'
    file write `fh' `","missing_count":`missing'"'
    file write `fh' `","counts":{"'

    local first = 1
    forvalues i = 1/`=_N' {
        if !`first' file write `fh' ","
        local lvl = `vname'[`i']
        * JSON-escape level values from the data: backslash first, then
        * double-quote, then collapse CR/LF/TAB to spaces (JSON forbids
        * literal control chars in string values).
        local lvl : subinstr local lvl `"\"' `"\\"', all
        local lvl : subinstr local lvl `"""' `"\""', all
        local lvl : subinstr local lvl "`=char(10)'" " ", all
        local lvl : subinstr local lvl "`=char(13)'" " ", all
        local lvl : subinstr local lvl "`=char(9)'" " ", all
        local cnt = _freq[`i']
        file write `fh' `""`lvl'":`cnt'"'
        local first = 0
    }
    file write `fh' "}}"
    file close `fh'

    restore
    display as text "builder_result_tab (1-way): wrote result to " as result "`path'"
end


* ---------------------------------------------------------------------------
* 2-way: crosstab
* ---------------------------------------------------------------------------

program define _builder_tab_2way
    version 13
    syntax varlist(min=2 max=2) [, label(string) ]

    * JSON-escape `label` (Claude-controllable free text). See
    * builder_result_regress for the full explanation of this pattern.
    local label : subinstr local label "\" "\\", all
    local label : subinstr local label `"""' `"\""', all
    local label : subinstr local label "`=char(10)'" " ", all
    local label : subinstr local label "`=char(13)'" " ", all
    local label : subinstr local label "`=char(9)'" " ", all

    local path : env BUILDER_RESULT_PATH
    tokenize `varlist'
    local row_var "`1'"
    local col_var "`2'"

    quietly count
    local total = r(N)
    quietly count if missing(`row_var') | missing(`col_var')
    local missing = r(N)

    preserve
    * Filter missings before contract — see 1-way comment.
    quietly keep if !missing(`row_var') & !missing(`col_var')
    quietly contract `row_var' `col_var'
    * After contract: one row per (row_var, col_var) combination with _freq.
    * Sort so rows for the same row_var are contiguous — lets us emit
    * nested dict with one inner object per row_var level.
    sort `row_var' `col_var'

    tempname fh
    file open `fh' using `"`path'"', write text replace

    file write `fh' `"{"type":"crosstab""'
    if `"`label'"' != "" {
        file write `fh' `","label":"`label'""'
    }
    file write `fh' `","row_variable":"`row_var'""'
    file write `fh' `","col_variable":"`col_var'""'
    file write `fh' `","missing_count":`missing'"'
    file write `fh' `","counts":{"'

    local prev_row = ""
    local something_written = 0

    forvalues i = 1/`=_N' {
        local row_val = `row_var'[`i']
        local col_val = `col_var'[`i']
        local cnt = _freq[`i']

        * JSON-escape row and col values from the data: backslash first,
        * then double-quote, then collapse CR/LF/TAB to spaces.
        local row_esc : subinstr local row_val `"\"' `"\\"', all
        local row_esc : subinstr local row_esc `"""' `"\""', all
        local row_esc : subinstr local row_esc "`=char(10)'" " ", all
        local row_esc : subinstr local row_esc "`=char(13)'" " ", all
        local row_esc : subinstr local row_esc "`=char(9)'" " ", all
        local col_esc : subinstr local col_val `"\"' `"\\"', all
        local col_esc : subinstr local col_esc `"""' `"\""', all
        local col_esc : subinstr local col_esc "`=char(10)'" " ", all
        local col_esc : subinstr local col_esc "`=char(13)'" " ", all
        local col_esc : subinstr local col_esc "`=char(9)'" " ", all

        if `"`row_esc'"' != `"`prev_row'"' {
            * New row_level. Close the previous inner dict (if any) and
            * add a separator before opening a new one.
            if `something_written' {
                file write `fh' "},"
            }
            file write `fh' `""`row_esc'":{"`col_esc'":`cnt'"'
            local prev_row `"`row_esc'"'
            local something_written = 1
        }
        else {
            * Same row_level as previous iteration — append another cell.
            file write `fh' `","`col_esc'":`cnt'"'
        }
    }
    if `something_written' {
        * Close the final inner dict.
        file write `fh' "}"
    }
    * Close counts object and outer object.
    file write `fh' "}}"
    file close `fh'

    restore
    display as text "builder_result_tab (2-way): wrote result to " as result "`path'"
end
