*! version 0.0.2  Builder runtime: emit a descriptive payload from r().
*!
*! Call after Stata's `summarize` command. Reads r(N), r(mean), r(sd) and
*! computes `missing_count` as the count of missing values for the
*! variable across the full dataset — NOT as `_N - r(N)`, which would
*! mask `summarize if ...` filtering from the row-count audit (a
*! filtered sample would pass `n + missing_count == schema.n` even
*! though the script silently dropped rows).
*!
*! The Builder sanitizer's `descriptive` schema does NOT accept min /
*! max / median / quartiles — those are individual observations and
*! get dropped even if emitted. This helper doesn't emit them.
*!
*! Usage:
*!   summarize income
*!   builder_result_sum income, label("Income descriptives")
*!
*! Note: Stata's r() after `summarize` doesn't carry the variable name,
*! so the researcher passes it as a positional argument. Callers can
*! still override `missing_count` with the `missing(<int>)` option.

program define builder_result_sum
    version 13
    syntax varname [, label(string) missing(integer -1) ]

    * JSON-escape `label` (Claude-controllable free text). See
    * builder_result_regress for the full explanation of this pattern.
    local label : subinstr local label "\" "\\", all
    local label : subinstr local label `"""' `"\""', all
    local label : subinstr local label "`=char(10)'" " ", all
    local label : subinstr local label "`=char(13)'" " ", all
    local label : subinstr local label "`=char(9)'" " ", all

    if "`r(N)'" == "" {
        display as error "builder_result_sum: no summarize results in memory. Run `summarize' first."
        exit 198
    }

    local path : env BUILDER_RESULT_PATH
    if "`path'" == "" {
        display as error "BUILDER_RESULT_PATH not set. Run through Builder."
        exit 198
    }

    * Per-run authenticity token. See builder_result_regress.ado for the
    * full explanation.
    local _builder_token : env BUILDER_RUN_TOKEN
    if "`_builder_token'" == "" {
        display as error "BUILDER_RUN_TOKEN not set. Run through Builder."
        exit 198
    }

    * Variable name from the positional argument. `syntax varname` is a
    * real-variable-in-data reference; the name lands in the `varlist`
    * macro (Stata's naming quirk).
    local vname "`varlist'"

    * Capture summarize's r() values into locals BEFORE running any
    * other r-class command (like `count` below, which clobbers r()).
    local rN "`r(N)'"
    local rmean "`r(mean)'"
    local rsd "`r(sd)'"

    * Missing count. If the caller supplied `missing(...)`, honor it.
    * Otherwise compute it as the count of missing values for `vname`
    * across the full dataset. Using `_N - r(N)` here would be wrong
    * when summarize ran under an `if` condition: `_N` stays at the
    * full dataset size while r(N) drops, so the computed "missing"
    * would include every filtered-out row. That silently satisfies
    * the Builder row-count audit's invariant (`n + missing_count ==
    * schema.n`) even when the script dropped rows — exactly the
    * drift the audit is supposed to catch.
    if `missing' < 0 {
        quietly count if missing(`vname')
        local missing = r(N)
    }

    tempname fh
    file open `fh' using `"`path'"', write text replace

    file write `fh' `"{"type":"descriptive""'
    file write `fh' `","_token":"`_builder_token'""'
    if `"`label'"' != "" {
        file write `fh' `","label":"`label'""'
    }
    file write `fh' `","variable":"`vname'""'
    file write `fh' `","n":`rN'"'

    if "`rmean'" != "" {
        local _x = strofreal(`rmean', "%21.17e")
        file write `fh' `","mean":`_x'"'
    }
    if "`rsd'" != "" {
        local _x = strofreal(`rsd', "%21.17e")
        file write `fh' `","sd":`_x'"'
    }

    file write `fh' `","missing_count":`missing'"'

    file write `fh' "}"
    file close `fh'

    display as text "builder_result_sum: wrote result to " as result "`path'"
end
