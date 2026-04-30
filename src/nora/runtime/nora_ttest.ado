*! version 0.1.0  Nora runtime: self-contained t-test helper.
*!
*! Single helper that runs the appropriate ``ttest`` command itself,
*! captures r() into locals before any other r-class operation, and
*! emits a sanitizer-shaped t_test payload. Removes the foot-gun
*! where an intervening r-class command (save / count / a second
*! ttest / tabulate / summarize) silently clobbered the scalars
*! ``nora_result_ttest`` was about to read.
*!
*! Mutually-exclusive shape options pick the variant:
*!   nora_ttest income, against(0)            // one-sample
*!   nora_ttest pre, paired(post)             // paired
*!   nora_ttest income, by(treated)           // two-sample, equal var
*!   nora_ttest income, by(treated) unequal   // Welch
*!
*! Optional [if] applies to whichever ttest gets run:
*!   nora_ttest income if region == 1, by(treated)
*!
*! No shape option ⇒ defaults to ``against(0)``: the one-sample
*! "is the mean different from zero" form, which is the most
*! common implicit ask. The four numerics ``r(t)``, ``r(p)``,
*! ``r(df_t)``, ``r(mu_1)`` etc. are always read from the
*! freshly-run ttest, not whatever was sitting in r() beforehand.

program define nora_ttest
    version 13
    syntax varname [if] [, ///
        against(string) ///
        paired(string) ///
        by(string) ///
        unequal ///
        label(string) ]

    * JSON-escape `label` (Claude-controllable free text). Same
    * pattern as nora_result_regress / _sum.
    local label : subinstr local label "\" "\\", all
    local label : subinstr local label `"""' `"\""', all
    local label : subinstr local label "`=char(10)'" " ", all
    local label : subinstr local label "`=char(13)'" " ", all
    local label : subinstr local label "`=char(9)'" " ", all

    * Validate: only one of against / paired / by may be specified.
    * Stata's syntax doesn't enforce mutually-exclusive options
    * declaratively, so we count and reject ambiguity loudly.
    local n_modes = 0
    if "`against'" != "" local n_modes = `n_modes' + 1
    if "`paired'" != "" local n_modes = `n_modes' + 1
    if "`by'" != "" local n_modes = `n_modes' + 1

    if `n_modes' > 1 {
        display as error "nora_ttest: only one of against(...), paired(...), or by(...) may be specified."
        exit 198
    }

    * Default shape: one-sample against 0. The most common implicit
    * ask ("is income's mean different from zero"); explicit
    * against(<other>) overrides.
    if `n_modes' == 0 {
        local against "0"
    }

    * Validate the `against` value parses as a real number when
    * given. confirm-number is Stata's idiomatic guard.
    if "`against'" != "" {
        capture confirm number `against'
        if _rc != 0 {
            display as error "nora_ttest: against(...) must be a real number, got `against'"
            exit 198
        }
    }

    * unequal only meaningful with by(); reject combinations the
    * underlying ttest would silently ignore.
    if "`unequal'" != "" & "`by'" == "" {
        display as error "nora_ttest: unequal requires by(...). Use by(group) unequal for Welch's two-sample."
        exit 198
    }

    * Validate paired-mate variable exists in the dataset.
    if "`paired'" != "" {
        capture confirm variable `paired'
        if _rc != 0 {
            display as error "nora_ttest: paired(`paired') is not a variable in the dataset."
            exit 459
        }
    }

    * Validate by-grouping variable exists.
    if "`by'" != "" {
        capture confirm variable `by'
        if _rc != 0 {
            display as error "nora_ttest: by(`by') is not a variable in the dataset."
            exit 459
        }
    }

    local path : env NORA_RESULT_PATH
    if "`path'" == "" {
        display as error "NORA_RESULT_PATH not set. Run through Nora."
        exit 198
    }

    local _nora_token : env NORA_RUN_TOKEN
    if "`_nora_token'" == "" {
        display as error "NORA_RUN_TOKEN not set. Run through Nora."
        exit 198
    }

    local vname "`varlist'"

    * Run the ttest variant matching the option(s) given. Visible
    * (not quietly) so the researcher sees the conventional table
    * in their raw log alongside any other Stata output.
    if "`paired'" != "" {
        ttest `vname' == `paired' `if'
    }
    else if "`by'" != "" {
        if "`unequal'" != "" {
            ttest `vname' `if', by(`by') unequal
        }
        else {
            ttest `vname' `if', by(`by')
        }
    }
    else {
        ttest `vname' `if' == `against'
    }

    * Capture r() into locals BEFORE any other r-class operation
    * (defensive — there are none below today, but adding one
    * later wouldn't silently break the helper).
    local rN_1 "`r(N_1)'"
    local rN_2 "`r(N_2)'"
    local rN "`r(N)'"
    local rmu_1 "`r(mu_1)'"
    local rmu_2 "`r(mu_2)'"
    local rt "`r(t)'"
    local rdf "`r(df_t)'"
    local rp "`r(p)'"
    local rwelch "`r(welch)'"

    if "`rt'" == "" {
        display as error "nora_ttest: ttest produced no r(t). Check the variable's values and the if-clause."
        exit 459
    }

    * Map the r() shape to Nora's test_type taxonomy. Same logic
    * as nora_result_ttest.
    local subtype "one_sample"
    if "`rN_1'" != "" & "`rN_2'" != "" {
        local subtype "two_sample"
        if "`rwelch'" != "" {
            if `rwelch' == 1 local subtype "welch"
        }
    }
    else if "`rmu_2'" != "" {
        local subtype "paired"
    }

    tempname fh
    file open `fh' using `"`path'"', write text append

    file write `fh' `"{"type":"t_test""'
    file write `fh' `","_token":"`_nora_token'""'
    if `"`label'"' != "" {
        file write `fh' `","label":"`label'""'
    }

    file write `fh' `","test_type":"`subtype'""'

    if "`subtype'" == "two_sample" | "`subtype'" == "welch" {
        file write `fh' `","n1":`rN_1'"'
        file write `fh' `","n2":`rN_2'"'
    }
    else if "`subtype'" == "paired" {
        file write `fh' `","n1":`rN'"'
    }
    else {
        file write `fh' `","n1":`rN'"'
    }

    if "`rmu_1'" != "" {
        local _x = strofreal(`rmu_1', "%21.17e")
        file write `fh' `","mean1":`_x'"'
    }
    if "`rmu_2'" != "" {
        local _x = strofreal(`rmu_2', "%21.17e")
        file write `fh' `","mean2":`_x'"'
    }

    local _x = strofreal(`rt', "%21.17e")
    file write `fh' `","t_statistic":`_x'"'

    if "`rdf'" != "" {
        local _x = strofreal(`rdf', "%21.17e")
        file write `fh' `","degrees_of_freedom":`_x'"'
    }

    if "`rp'" != "" {
        local _x = strofreal(`rp', "%21.17e")
        file write `fh' `","p_value":`_x'"'
    }

    file write `fh' "}" _newline
    file close `fh'

    display as text "nora_ttest: wrote result to " as result "`path'"
end
