*! version 0.0.1  Builder runtime: emit a t_test payload from r().
*!
*! Call after Stata's `ttest` command. Supports the three shapes Stata's
*! ttest produces:
*!
*!   one-sample:   ttest y == 0              (null = constant)
*!   two-sample:   ttest y, by(group)         (equal variance)
*!   unequal:      ttest y, by(group) unequal (Welch)
*!   paired:       ttest y == x
*!
*! Usage:
*!   ttest income, by(treated)
*!   builder_result_ttest, label("income by treatment")

program define builder_result_ttest
    version 13
    syntax [, label(string) ]

    * JSON-escape `label` (Claude-controllable free text). See
    * builder_result_regress for the full explanation of this pattern.
    local label : subinstr local label "\" "\\", all
    local label : subinstr local label `"""' `"\""', all
    local label : subinstr local label "`=char(10)'" " ", all
    local label : subinstr local label "`=char(13)'" " ", all
    local label : subinstr local label "`=char(9)'" " ", all

    if "`r(t)'" == "" {
        display as error "builder_result_ttest: no t-test results in memory. Run `ttest` first."
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

    * Stata's ttest sets different r() scalars for one-sample vs two-sample
    * vs paired. We detect which variant populated r() and map to
    * Builder's test_type taxonomy.
    *
    * Shape markers (from `help ttest`):
    *   r(N_1), r(N_2)     → two-sample (either equal or Welch)
    *   r(mu_2) exists     → either two-sample or paired
    *   r(N), r(mu_1) only → one-sample or paired
    *
    * Simplest heuristic: if r(N_1) and r(N_2) exist, treat as two-sample.
    * Welch is a two-sample variant; r(welch) is 1 on Welch runs.

    local subtype "one_sample"
    if "`r(N_1)'" != "" & "`r(N_2)'" != "" {
        local subtype "two_sample"
        if "`r(welch)'" != "" {
            if `r(welch)' == 1 local subtype "welch"
        }
    }
    else if "`r(mu_2)'" != "" {
        * Paired test: single effective sample size, mu_1 and mu_2 are
        * the two means being differenced.
        local subtype "paired"
    }

    tempname fh
    file open `fh' using `"`path'"', write text replace

    file write `fh' `"{"type":"t_test""'
    file write `fh' `","_token":"`_builder_token'""'
    if `"`label'"' != "" {
        file write `fh' `","label":"`label'""'
    }

    file write `fh' `","test_type":"`subtype'""'

    * Sample sizes. n1 is always required; n2 only for two-sample / welch.
    if "`subtype'" == "two_sample" | "`subtype'" == "welch" {
        file write `fh' `","n1":`=r(N_1)'"'
        file write `fh' `","n2":`=r(N_2)'"'
    }
    else if "`subtype'" == "paired" {
        * r(N) is the number of pairs.
        file write `fh' `","n1":`=r(N)'"'
    }
    else {
        file write `fh' `","n1":`=r(N)'"'
    }

    * Means. Stata: r(mu_1) is always the first group / first variable;
    * r(mu_2) is the second group / constant / paired comparator.
    if "`r(mu_1)'" != "" {
        local _x = strofreal(`=r(mu_1)', "%21.17e")
        file write `fh' `","mean1":`_x'"'
    }
    if "`r(mu_2)'" != "" {
        local _x = strofreal(`=r(mu_2)', "%21.17e")
        file write `fh' `","mean2":`_x'"'
    }

    * Test statistic, df, p-value (two-sided).
    local _x = strofreal(`=r(t)', "%21.17e")
    file write `fh' `","t_statistic":`_x'"'

    if "`r(df_t)'" != "" {
        local _x = strofreal(`=r(df_t)', "%21.17e")
        file write `fh' `","degrees_of_freedom":`_x'"'
    }

    * Stata publishes three p-values: r(p_l), r(p), r(p_u) for one-sided
    * lower, two-sided, one-sided upper. Use two-sided by default.
    if "`r(p)'" != "" {
        local _x = strofreal(`=r(p)', "%21.17e")
        file write `fh' `","p_value":`_x'"'
    }

    file write `fh' "}"
    file close `fh'

    display as text "builder_result_ttest: wrote result to " as result "`path'"
end
