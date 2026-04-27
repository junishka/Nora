*! version 0.0.1  Nora runtime: predicted-response curve from e().
*!
*! After a regression, plot the predicted response across the
*! observed range of one variable, holding the other predictors at
*! their means. Uses Stata's native ``margins`` + ``marginsplot``.
*!
*! Usage:
*!   regress log_salary female log_assets log_age tenure
*!   nora_plot_interaction log_assets, label("Salary by log assets")
*!
*!   * Optional ``xlabel`` / ``ylabel`` / ``title`` to override defaults:
*!   nora_plot_interaction log_assets, ///
*!       label("Salary by log assets") ///
*!       xlabel("Log assets (USD)") ylabel("Log CEO salary") ///
*!       title("Predicted CEO salary by firm size")
*!
*! Why this exists: without a Stata helper, the model would extract
*! coefficients, switch to R or Python, and hand-roll a predicted-
*! response curve there — exactly the language-switching loop the
*! researcher complained about. Same posture as
*! ``nora_plot_coefficients``: native to Stata, manifest-allowlisted.

program define nora_plot_interaction
    version 13
    syntax varname , [ label(string) xlabel(string) ylabel(string) title(string) ]

    if "`e(cmd)'" == "" {
        display as error "nora_plot_interaction: no estimation results in memory. Run a regression first."
        exit 198
    }

    local var "`varlist'"

    * Resolve run dir from NORA_RESULT_PATH (Stata's subprocess cwd
    * is the SESSION cwd via the executor preamble cd, NOT the
    * run dir).
    local resultpath : env NORA_RESULT_PATH
    if "`resultpath'" == "" {
        display as error "nora_plot_interaction: NORA_RESULT_PATH not set"
        exit 198
    }
    local rundir : subinstr local resultpath "/result.json" ""

    local _step "init"
    capture noisily {
        local _step "summarize"
        * Compute the variable's observed range. ``summarize`` after
        * the regression — if the var was used in the model, it's
        * still in memory.
        quietly summarize `var'
        if r(N) == 0 {
            display as error "nora_plot_interaction: no observations for `var'"
            exit 198
        }
        local _min = r(min)
        local _max = r(max)
        if `_min' == `_max' {
            display as error "nora_plot_interaction: `var' has no variation"
            exit 198
        }

        * Build a 25-point grid across the observed range. 25 is
        * smooth enough for ``marginsplot`` 's recast(line) without
        * burning compute on huge datasets.
        local step = (`_max' - `_min') / 24
        margins, at(`var' = (`_min'(`step')`_max')) atmeans

        * Pick defaults for axis labels / title that are honest if
        * the caller doesn't override them. Match the R / Python
        * helpers' phrasing so plots from the three languages share
        * a vocabulary.
        local _xt `"`xlabel'"'
        if "`_xt'" == "" local _xt "`var'"
        local _yt `"`ylabel'"'
        if "`_yt'" == "" local _yt "Predicted response"
        local _tt `"`title'"'
        if "`_tt'" == "" local _tt "Predicted response by `var'"

        * recast(line) for the prediction line, recastci(rarea) for
        * a filled CI band — much more readable than the default
        * dashed-line whiskers.
        marginsplot, ///
            recast(line) recastci(rarea) ///
            plotopts(lcolor(navy) lwidth(medthick)) ///
            ciopts(color(navy%18)) ///
            title(`"`_tt'"', size(medium)) ///
            xtitle(`"`_xt'"') ytitle(`"`_yt'"') ///
            xlabel(, labsize(small)) ylabel(, angle(0) labsize(small)) ///
            graphregion(color(white)) plotregion(margin(medsmall))

        local _step "export"
        * Sanitize the variable name into a filename-safe token.
        local safe : subinstr local var ":" "_", all
        local safe : subinstr local safe "." "_", all
        _nora_export_plot using "`rundir'/_nora_plots", ///
            basename("interaction_`safe'") width(1600)
        local _file = "`r(file)'"
        local _fmt  = "`r(format)'"
        if "`_file'" == "" {
            display as error "nora_plot_interaction: every export format failed (last _rc=`r(last_rc)')"
            exit `r(last_rc)'
        }

        local _step "manifest"
        local lab `"`label'"'
        if "`lab'" == "" local lab "Predicted response by `var'"
        local lab : subinstr local lab "\" "\\", all
        local lab : subinstr local lab `"""' `"\""', all
        local lab : subinstr local lab "`=char(10)'" " ", all
        local lab : subinstr local lab "`=char(13)'" " ", all
        local lab : subinstr local lab "`=char(9)'" " ", all

        local manifestpath "`rundir'/_nora_plots/manifest.jsonl"
        local jsonline `"{"file":"`_file'","kind":"interaction","label":"`lab'","format":"`_fmt'"}"'
        tempname mh
        file open `mh' using "`manifestpath'", write append text
        file write `mh' `"`jsonline'"' _n
        file close `mh'

        display as text "nora_plot_interaction: wrote `_file'"
    }

    if _rc {
        local _orig_rc = _rc
        capture mkdir "`rundir'/_nora_plots"
        local errpath "`rundir'/_nora_plots/helper_errors.jsonl"
        local errline `"{"helper":"plot_interaction","step":"`_step'","error":"Stata _rc=`_orig_rc'","message":"plot_interaction failed at step `_step' with _rc=`_orig_rc'; check stderr.log for the underlying error"}"'
        tempname eh
        capture file open `eh' using "`errpath'", write append text
        if !_rc {
            file write `eh' `"`errline'"' _n
            file close `eh'
        }
    }
end
