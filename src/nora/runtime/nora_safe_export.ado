*! version 0.0.1  Nora runtime: format-fallback wrapper around graph export.
*!
*! Bare ``graph export "x.png"`` in a Stata script ABORTS the entire
*! do-file when ``Graph2png`` is missing — the model's plot fails AND
*! ``nora_result_*`` never runs, so neither a structured payload nor a
*! plot reaches Nora. This is the residual hand-rolled-export gap that
*! the four ``nora_plot_*`` helpers couldn't cover (they're for the
*! built-in plot kinds; community packages like ``coefplot`` produce
*! the plot themselves and need a separate export step).
*!
*! ``nora_safe_export`` wraps the export with the same priority chain
*! as the helpers:
*!   1. The requested format (extension-derived).
*!   2. PDF (Stata 17+ native; older via Graph2pdf).
*!   3. EPS (Stata's native PostScript path; almost always works).
*!   4. .gph (always saves; researcher opens in Stata).
*!
*! Researcher-visible only — does NOT register the plot in the
*! manifest. The manifest is gated to plots produced by the kind-
*! specific helpers (``plot_residuals``, ``plot_coefficients``,
*! ``plot_interaction``, ``plot_estimate_comparison``) so the model
*! can't smuggle a raw histogram through this safe wrapper.
*!
*! Usage:
*!   coefplot, drop(_cons) base
*!   nora_safe_export, file("coef_plot.png")

program define nora_safe_export
    version 13
    syntax , file(string) [ width(integer 1200) ]

    * Identify the requested format from the extension. Default to
    * pdf if the caller didn't include one.
    local _dotpos = strrpos("`file'", ".")
    if `_dotpos' > 0 {
        local _ext = lower(substr("`file'", `_dotpos' + 1, .))
        local _base = substr("`file'", 1, `_dotpos' - 1)
    }
    else {
        local _ext = "pdf"
        local _base = "`file'"
    }

    local _written ""

    * 1. Try the requested format.
    capture noisily graph export "`_base'.`_ext'", as(`_ext') replace ///
        width(`width')
    if !_rc {
        local _written "`_base'.`_ext'"
        display as text "nora_safe_export: wrote `_written'"
        exit 0
    }
    local _last_rc = _rc

    * 2. Fall back to PDF.
    if "`_ext'" != "pdf" {
        capture noisily graph export "`_base'.pdf", as(pdf) replace
        if !_rc {
            local _written "`_base'.pdf"
            display as text "nora_safe_export: requested `_ext' failed (translator?); wrote `_written' instead"
            exit 0
        }
        local _last_rc = _rc
    }

    * 3. Fall back to EPS — Stata's native PostScript engine, so
    * this works on installs missing every translator.
    if "`_ext'" != "eps" {
        capture noisily graph export "`_base'.eps", as(eps) replace
        if !_rc {
            local _written "`_base'.eps"
            display as text "nora_safe_export: PDF and `_ext' both failed; wrote `_written' instead"
            exit 0
        }
        local _last_rc = _rc
    }

    * 4. Last resort: save the .gph for opening in Stata.
    capture graph save "`_base'.gph", replace
    if !_rc {
        display as text "nora_safe_export: every raster/vector export failed; saved `_base'.gph (open in Stata)"
        exit 0
    }

    display as error "nora_safe_export: every export attempt failed (last _rc=`_last_rc')"
end
