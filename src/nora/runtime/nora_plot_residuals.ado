*! version 0.0.1  Nora runtime: residual diagnostic plot from e().
*!
*! Call after a regression that supports rvfplot (regress, areg,
*! reg with weights, etc.). Writes residuals.png into
*! ``_nora_plots/`` (a subdir of the script's run dir, since the
*! Nora executor sets the Stata subprocess cwd to the run dir) and
*! appends an entry to ``_nora_plots/manifest.jsonl`` so the bridge
*! knows the plot is sanctioned to cross to the model on the next
*! turn. Plots NOT registered through this helper (raw ``graph
*! export`` calls) stay researcher-visible only — the manifest
*! allowlist is the privacy gate.
*!
*! Usage:
*!   regress y x1 x2
*!   nora_plot_residuals, label("Residual diagnostics for income~edu+age")
*!
*! Notes:
*! - Errors inside the plotting / file-write are caught with capture
*!   so a broken plot helper never breaks the analysis script
*!   around it. The researcher's ``stderr.log`` will still show the
*!   underlying message.
*! - JSON for the manifest line is built by hand because Stata has
*!   no native JSON serializer and a one-line append is simple
*!   enough that adding a dependency would be silly.

program define nora_plot_residuals
    version 13
    syntax [, label(string) ]

    if "`e(cmd)'" == "" {
        display as error "nora_plot_residuals: no estimation results in memory. Run a regression command first."
        exit 198
    }

    * Resolve the run dir from NORA_RESULT_PATH (set by the executor;
    * always points at ``<run_dir>/result.json``). The Stata
    * subprocess cwd is the SESSION cwd (the executor preamble cd's
    * there so ``use "data.dta"`` resolves), NOT the run dir. Without
    * this resolution, ``_nora_plots/`` would land in the session
    * dir where the bridge's _capture_plots never looks — model
    * vision would silently miss every Stata plot. Bug fixed in P2.
    local resultpath : env NORA_RESULT_PATH
    if "`resultpath'" == "" {
        display as error "nora_plot_residuals: NORA_RESULT_PATH not set"
        exit 198
    }
    local rundir : subinstr local resultpath "/result.json" ""

    local _step "init"
    capture noisily {
        local _step "rvfplot"
        rvfplot, ytitle("Residuals") xtitle("Fitted values") ///
            title("Residuals vs Fitted")

        local _step "export"
        _nora_export_plot using "`rundir'/_nora_plots", ///
            basename("residuals") width(1600)
        local _file = "`r(file)'"
        local _fmt  = "`r(format)'"
        if "`_file'" == "" {
            display as error "nora_plot_residuals: every export format failed (last _rc=`r(last_rc)')"
            exit `r(last_rc)'
        }

        local _step "manifest"
        * JSON-escape the researcher-supplied label.
        local lab `"`label'"'
        if "`lab'" == "" local lab "Residual diagnostics"
        local lab : subinstr local lab "\" "\\", all
        local lab : subinstr local lab `"""' `"\""', all
        local lab : subinstr local lab "`=char(10)'" " ", all
        local lab : subinstr local lab "`=char(13)'" " ", all
        local lab : subinstr local lab "`=char(9)'" " ", all

        local manifestpath "`rundir'/_nora_plots/manifest.jsonl"
        local jsonline `"{"file":"`_file'","kind":"residuals","label":"`lab'","format":"`_fmt'"}"'
        tempname mh
        file open `mh' using "`manifestpath'", write append text
        file write `mh' `"`jsonline'"' _n
        file close `mh'

        display as text "nora_plot_residuals: wrote `_file'"
    }
    if _rc {
        local _orig_rc = _rc
        capture mkdir "`rundir'/_nora_plots"
        local errpath "`rundir'/_nora_plots/helper_errors.jsonl"
        local errline `"{"helper":"plot_residuals","step":"`_step'","error":"Stata _rc=`_orig_rc'","message":"plot_residuals failed at step `_step' with _rc=`_orig_rc'; check stderr.log for the underlying error"}"'
        tempname eh
        capture file open `eh' using "`errpath'", write append text
        if !_rc {
            file write `eh' `"`errline'"' _n
            file close `eh'
        }
    }
end
