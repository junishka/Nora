*! version 0.0.1  Nora runtime: forest plot comparing one coefficient
*! across multiple stored estimation results.
*!
*! Use case: "Female gap, before vs after controls" — run two
*! regressions, store each, then plot the female coefficient from
*! both side-by-side with their CIs. Without this helper the model
*! has been hand-rolling the comparison plot in R / Python with
*! manually-copied estimates, taking 3+ attempts to land one image.
*!
*! Usage:
*!
*!   regress log_salary female if total_revenue > 0
*!   estimates store m_unadj
*!   nora_result_regress, label("Unadjusted female gap")
*!
*!   regress log_salary female log_assets log_age tenure i.year i.sector ///
*!       if total_revenue > 0
*!   estimates store m_adj
*!   nora_result_regress, label("Adjusted female gap")
*!
*!   nora_plot_estimate_comparison m_unadj m_adj, coef(female) ///
*!       labels("Unadjusted" "Adjusted (controls)") ///
*!       label("Female gap: before vs after controls")
*!
*! ``coef`` is the coefficient name to extract from each stored
*! estimate. ``labels`` is space-separated, optionally quoted; if
*! omitted, the stored-estimate names are used as y-axis labels.

program define nora_plot_estimate_comparison
    version 13
    syntax namelist , coef(string) [ labels(string asis) label(string) ]

    local k : word count `namelist'
    if `k' < 2 {
        display as error "nora_plot_estimate_comparison: need at least 2 stored estimate names; got `k'"
        exit 198
    }

    * Resolve run dir from NORA_RESULT_PATH (Stata's subprocess cwd
    * is the SESSION cwd via the executor preamble cd).
    local resultpath : env NORA_RESULT_PATH
    if "`resultpath'" == "" {
        display as error "nora_plot_estimate_comparison: NORA_RESULT_PATH not set"
        exit 198
    }
    local rundir : subinstr local resultpath "/result.json" ""

    * Snapshot whichever estimation results are currently active so we
    * can restore them at the end. ``estimates restore`` only works on
    * a name; the unnamed "current" results aren't directly recoverable
    * unless the caller already stored them.
    capture estimates store _nora_pec_save

    local _step "init"
    capture noisily {
        local _step "extract estimates"
        * Pull the named coefficient from each stored estimate.
        tempname ests ses
        matrix `ests' = J(`k', 1, .)
        matrix `ses'  = J(`k', 1, .)
        local i = 0
        foreach m of local namelist {
            local i = `i' + 1
            quietly estimates restore `m'
            * Verify the named coefficient exists in this model. If
            * it doesn't, fail with a clear message rather than letting
            * Stata's ``_b[]`` syntax produce missing.
            local cnames : colnames e(b)
            local found = 0
            foreach cn of local cnames {
                if "`cn'" == "`coef'" local found = 1
            }
            if !`found' {
                display as error "nora_plot_estimate_comparison: coefficient `coef' not found in `m'"
                exit 198
            }
            matrix `ests'[`i', 1] = _b[`coef']
            matrix `ses'[`i', 1]  = _se[`coef']
        }

        * Build a temp dataset for the twoway plot.
        preserve
        clear
        set obs `k'
        gen str40 mname = ""
        gen double est = .
        gen double lo = .
        gen double hi = .
        gen pos = _n
        local i = 0
        foreach m of local namelist {
            local i = `i' + 1
            replace mname = "`m'" in `i'
            replace est = `ests'[`i', 1] in `i'
            local _est = `ests'[`i', 1]
            local _se  = `ses'[`i', 1]
            replace lo = `_est' - 1.96 * `_se' in `i'
            replace hi = `_est' + 1.96 * `_se' in `i'
        }

        * Resolve y-axis labels. ``labels`` is parsed asis so the
        * caller can pass quoted strings with spaces. Tokenize and
        * map to positions; fall back to model names if the count
        * doesn't match.
        local nlabels : word count `labels'
        capture label drop _nora_pec_labels
        forvalues i = 1/`k' {
            if `nlabels' == `k' {
                local lbl : word `i' of `labels'
            }
            else {
                local lbl = mname[`i']
            }
            * Stata label-define text needs quoting through the
            * compound-quote variant for safety.
            label define _nora_pec_labels `i' `"`lbl'"', modify
        }
        label values pos _nora_pec_labels

        * Forest plot — same visual conventions as nora_plot_coefficients.
        twoway (rcap lo hi pos, horizontal lcolor(navy) lwidth(medium)) ///
               (scatter pos est, msymbol(O) msize(medium) mcolor(navy)) ///
               , ytitle("") xtitle("`coef' (95% CI)") ///
                 ylabel(1(1)`k', valuelabel angle(0) labsize(small)) ///
                 yscale(reverse) ///
                 legend(off) ///
                 xline(0, lpattern(dash) lcolor(gs8)) ///
                 title("Estimate comparison: `coef'") ///
                 graphregion(color(white)) ///
                 plotregion(margin(medium))

        local _step "export"
        _nora_export_plot using "`rundir'/_nora_plots", ///
            basename("estimate_comparison") width(1600)
        local _file = "`r(file)'"
        local _fmt  = "`r(format)'"
        if "`_file'" == "" {
            display as error "nora_plot_estimate_comparison: every export format failed (last _rc=`r(last_rc)')"
            restore
            exit `r(last_rc)'
        }

        local _step "manifest"
        local lab `"`label'"'
        if "`lab'" == "" local lab "Estimate comparison: `coef'"
        local lab : subinstr local lab "\" "\\", all
        local lab : subinstr local lab `"""' `"\""', all
        local lab : subinstr local lab "`=char(10)'" " ", all
        local lab : subinstr local lab "`=char(13)'" " ", all
        local lab : subinstr local lab "`=char(9)'" " ", all

        local manifestpath "`rundir'/_nora_plots/manifest.jsonl"
        local jsonline `"{"file":"`_file'","kind":"coefficients","label":"`lab'","format":"`_fmt'"}"'
        tempname mh
        file open `mh' using "`manifestpath'", write append text
        file write `mh' `"`jsonline'"' _n
        file close `mh'

        display as text "nora_plot_estimate_comparison: wrote `_file'"
        restore
    }

    if _rc {
        local _orig_rc = _rc
        capture mkdir "`rundir'/_nora_plots"
        local errpath "`rundir'/_nora_plots/helper_errors.jsonl"
        local errline `"{"helper":"plot_estimate_comparison","step":"`_step'","error":"Stata _rc=`_orig_rc'","message":"plot_estimate_comparison failed at step `_step' with _rc=`_orig_rc'; check stderr.log for the underlying error"}"'
        tempname eh
        capture file open `eh' using "`errpath'", write append text
        if !_rc {
            file write `eh' `"`errline'"' _n
            file close `eh'
        }
    }

    * Best-effort: restore whatever was active before the helper ran
    * so the surrounding script doesn't see ``e()`` from the last
    * model in the comparison list.
    capture estimates restore _nora_pec_save
    capture estimates drop _nora_pec_save
end
