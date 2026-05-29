*! version 0.1.0  Nora runtime: emit a descriptive payload from r().
*!
*! Self-contained: the helper runs ``summarize`` internally on the
*! named variable (with the optional [if] you pass), captures the
*! r() scalars before any other r-class operation, and emits the
*! sanitizer-shaped payload. The summarize output is also printed
*! to stdout so the researcher sees the conventional table.
*!
*! Old pattern (still works, helper just re-summarizes):
*!     summarize income
*!     nora_result_sum income, label("Income descriptives")
*!
*! Recommended pattern (one line, no foot-gun):
*!     nora_result_sum income, label("Income descriptives")
*!     nora_result_sum income if region == 1, label("Income, region 1")
*!
*! Why this changed: previously the helper read r() from whatever
*! summarize the script ran most recently. That had two failure
*! modes — (a) any intervening r-class command (save, count, tab,
*! a second summarize) wiped the scalars and the helper either
*! errored or, worse, picked up the wrong variable's r(mean), so
*! a payload labeled "age" silently carried income's mean; (b) the
*! defensive r(mean)-empty guard caught (a)'s loud version but not
*! (a)'s silent one. By summarizing the named variable itself, the
*! helper is correct by construction regardless of what the script
*! ran before it.
*!
*! The Nora sanitizer's `descriptive` schema does NOT accept min /
*! max / median / quartiles — those are individual observations
*! and get dropped even if emitted. This helper doesn't emit them.
*!
*! Optional `distinct` flag: emit the exact count of unique non-missing
*! values of the variable (over the same [if] sample). Unlike mean / sd
*! (which the sanitizer rounds to an N-appropriate number of significant
*! figures), `distinct_count` is an allowed INTEGER field and passes
*! through unrounded — so this is the supported way to release an exact
*! unique/cardinality count. The helper computes it itself (egen group),
*! so it's correct by construction; the whole-payload n>=10 minimum still
*! applies. Example:
*!     nora_result_sum ein, distinct label("Unique EINs")

program define nora_result_sum
    version 13
    syntax varname [if] [, label(string) missing(integer -1) DISTINCT ]

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

    * Variable name from the positional argument. `syntax varname` is
    * a real-variable-in-data reference; the name lands in the
    * `varlist` macro (Stata's naming quirk).
    local vname "`varlist'"

    * Run summarize ourselves so r() definitely holds THIS variable's
    * scalars regardless of what came before. Noisy (no `quietly`)
    * so the researcher still sees the conventional table — that's
    * the value of `summarize` for them, not just for our payload.
    summarize `vname' `if'

    * Capture into locals BEFORE the missing-count `count if`, which
    * is r-class and would clobber r(N)/r(mean)/r(sd).
    local rN "`r(N)'"
    local rmean "`r(mean)'"
    local rsd "`r(sd)'"

    * Defensive: if even our own summarize didn't populate r(mean)
    * (vname has no observations under the if-clause, the variable
    * has no non-missing values, etc.), surface it loudly rather
    * than write a half-empty payload the sanitizer would reject.
    if "`rmean'" == "" {
        display as error "nora_result_sum: summarize on `vname' produced no mean — variable may be all-missing or empty under the if-clause."
        exit 459
    }

    * Missing count. If the caller supplied `missing(...)`, honor it.
    * Otherwise compute as the count of missing values for `vname`
    * across the full dataset (NOT `_N - rN`, which would mask
    * summarize-with-if filtering from the row-count audit's invariant
    * `n + missing_count == schema.n`).
    if `missing' < 0 {
        quietly count if missing(`vname')
        local missing = r(N)
    }

    * Exact distinct count, opt-in via the `distinct` flag. `egen group`
    * assigns each distinct non-missing value of `vname` a sequential id
    * (missing -> missing, NOT a group), so r(max) is the number of unique
    * non-missing values — consistent with `n` (non-missing) and tracked
    * separately from `missing_count`. Respects the same [if] as summarize.
    * Guarded by the all-missing exit above, so r(max) is >= 1 here.
    if "`distinct'" != "" {
        tempvar _nd
        quietly egen `_nd' = group(`vname') `if'
        quietly summarize `_nd', meanonly
        * %18.0f + trim: plain integer literal, no scientific notation and
        * no leading pad, for any cardinality up to the dataset size.
        local distinct_count = trim(strofreal(r(max), "%18.0f"))
    }

    tempname fh
    file open `fh' using `"`path'"', write text append

    file write `fh' `"{"type":"descriptive""'
    file write `fh' `","_token":"`_nora_token'""'
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

    * Only emit when the flag was set and a valid (non-missing) count was
    * produced — a "." would be invalid JSON and the sanitizer would reject
    * the whole payload.
    if "`distinct'" != "" & "`distinct_count'" != "" & "`distinct_count'" != "." {
        file write `fh' `","distinct_count":`distinct_count'"'
    }

    file write `fh' "}" _newline
    file close `fh'

    display as text "nora_result_sum: wrote result to " as result "`path'"
end
