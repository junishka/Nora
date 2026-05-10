*! version 0.0.1  Nora runtime: forest plot of coefficient estimates.
*!
*! Call after a regression command (regress, areg, logit, probit, etc.
*! — anything that populates e(b) and e(V)). Builds a forest plot of
*! the coefficient point estimates with 95% CIs, drops the intercept
*! by default, writes the plot to ``<run_dir>/_nora_plots/`` and
*! appends a manifest entry so the runner can surface it to the
*! model.
*!
*! Usage:
*!   regress y x1 x2 x3
*!   nora_plot_coefficients, label("Salary regression: female + controls")
*!
*! Why this exists: a coefficient plot is the most common
*! "interpret the regression visually" output, and Stata
*! researchers shouldn't have to leave Stata (in particular,
*! shouldn't have to teach the model to retry in R or Python and
*! fight missing packages) just to produce one. This helper does
*! the postfile + twoway dance the model otherwise builds by hand.

program define nora_plot_coefficients
    version 13
    syntax [, label(string) ]

    if "`e(cmd)'" == "" {
        display as error "nora_plot_coefficients: no estimation results in memory. Run a regression command first."
        exit 198
    }

    * Resolve run dir from NORA_RESULT_PATH (Stata's subprocess
    * cwd is the SESSION cwd via the executor preamble cd, NOT
    * the run dir). Without this, ``_nora_plots/`` would land
    * under session_cwd where the runner never looks.
    local resultpath : env NORA_RESULT_PATH
    if "`resultpath'" == "" {
        display as error "nora_plot_coefficients: NORA_RESULT_PATH not set"
        exit 198
    }
    local rundir : subinstr local resultpath "/result.json" ""

    * Per-run authenticity token. Stamped into every manifest line so
    * the executor can drop manifest entries a hand-crafted file write
    * could otherwise have appended.
    local _nora_token : env NORA_RUN_TOKEN
    if "`_nora_token'" == "" {
        display as error "nora_plot_coefficients: NORA_RUN_TOKEN not set"
        exit 198
    }

    local _step "init"
    capture noisily {
        local _step "extract e()"
        * Pull coefficient estimates and SEs straight from e(b) /
        * e(V). This is canonical post-estimation Stata — no
        * dependency on coefplot or any community package.
        tempname b V
        matrix `b' = e(b)
        matrix `V' = e(V)
        local k = colsof(`b')
        local cnames : colnames `b'

        preserve
        clear
        set obs `k'
        gen str40 cname = ""
        gen double est = .
        gen double lo = .
        gen double hi = .
        gen pos = .
        forvalues i = 1/`k' {
            local cn : word `i' of `cnames'
            replace cname = "`cn'" in `i'
            scalar _est = `b'[1, `i']
            scalar _se = sqrt(`V'[`i', `i'])
            replace est = _est in `i'
            replace lo  = _est - 1.96 * _se in `i'
            replace hi  = _est + 1.96 * _se in `i'
        }

        * Drop _cons. Researchers almost never want the intercept
        * on the same axis as predictors — its scale dwarfs theirs
        * and the resulting plot reads as "intercept big, everything
        * else looks like zero." If only _cons exists, exit cleanly.
        drop if cname == "_cons"
        if _N == 0 {
            display as text "nora_plot_coefficients: nothing to plot after dropping intercept"
            restore
            exit 0
        }
        replace pos = _n

        * Build a value label so the y-axis shows coefficient names
        * rather than 1, 2, 3, .... Each pos integer maps to its
        * cname; we then ylabel the plot using the value label.
        capture label drop _nora_coefnames
        label define _nora_coefnames 0 "filler"
        forvalues i = 1/`=_N' {
            local nm = cname[`i']
            * Stata label values must be valid identifiers; if a
            * cname has a colon (factor levels like 1.x), substitute.
            local safe : subinstr local nm ":" "_", all
            local safe : subinstr local safe "." "_", all
            label define _nora_coefnames `i' `"`safe'"', modify
        }
        label values pos _nora_coefnames

        * Forest plot: rcap for the CI whiskers, scatter for the
        * point estimate. yscale(reverse) so coefficient #1 sits
        * at the top — matches the convention used by R's coef
        * helper and Python's matplotlib version.
        twoway (rcap lo hi pos, horizontal lcolor(navy)) ///
               (scatter pos est, msymbol(O) msize(medium) mcolor(navy)) ///
               , ytitle("") xtitle("Coefficient (95% CI)") ///
                 ylabel(1(1)`=_N', valuelabel angle(0) labsize(small)) ///
                 yscale(reverse) ///
                 legend(off) ///
                 xline(0, lpattern(dash) lcolor(gs8)) ///
                 title("Coefficients") ///
                 graphregion(color(white)) ///
                 plotregion(margin(medium))

        local _step "export"
        _nora_export_plot using "`rundir'/_nora_plots", ///
            basename("coefficients") width(1600)
        local _file = "`r(file)'"
        local _fmt  = "`r(format)'"
        if "`_file'" == "" {
            display as error "nora_plot_coefficients: every export format failed (last _rc=`r(last_rc)')"
            restore
            exit `r(last_rc)'
        }

        local _step "manifest"
        local lab `"`label'"'
        if "`lab'" == "" local lab "Coefficient estimates with 95% CIs"
        local lab : subinstr local lab "\" "\\", all
        local lab : subinstr local lab `"""' `"\""', all
        local lab : subinstr local lab "`=char(10)'" " ", all
        local lab : subinstr local lab "`=char(13)'" " ", all
        local lab : subinstr local lab "`=char(9)'" " ", all

        local manifestpath "`rundir'/_nora_plots/manifest.jsonl"
        local jsonline `"{"file":"`_file'","kind":"coefficients","label":"`lab'","format":"`_fmt'","_token":"`_nora_token'"}"'
        tempname mh
        file open `mh' using "`manifestpath'", write append text
        file write `mh' `"`jsonline'"' _n
        file close `mh'

        display as text "nora_plot_coefficients: wrote `_file'"
        restore
    }

    * Failure path: capture the _rc value IMMEDIATELY before any
    * other Stata operation (mkdir / file open / etc) resets it.
    * The previous version interpolated `_rc' AFTER mkdir, which
    * always read 0 — error logs were uninformative.
    if _rc {
        local _orig_rc = _rc
        capture mkdir "`rundir'/_nora_plots"
        local errpath "`rundir'/_nora_plots/helper_errors.jsonl"
        local errline `"{"helper":"plot_coefficients","step":"`_step'","error":"Stata _rc=`_orig_rc'","message":"plot_coefficients failed at step `_step' with _rc=`_orig_rc'; check stderr.log for the underlying error"}"'
        tempname eh
        capture file open `eh' using "`errpath'", write append text
        if !_rc {
            file write `eh' `"`errline'"' _n
            file close `eh'
        }
    }
end
