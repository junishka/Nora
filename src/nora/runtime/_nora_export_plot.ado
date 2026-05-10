*! version 0.0.1  Nora runtime: plot-export utility used by every
*! Stata plot helper.
*!
*! Stata's PNG export depends on the ``Graph2png`` translator, which
*! is NOT installed by default on many macOS systems. Hardcoding
*! ``as(png)`` made every helper fail on those machines:
*! ``translator Graph2png not found``. The fix is to try formats in
*! priority order (most-supported first) and stop at the first
*! success:
*!
*!   1. PDF  — Stata 17+ native; Stata 15-16 via Graph2pdf
*!             translator. PNG sidecar produced by the bridge via
*!             ``sips`` on macOS so the model still gets vision.
*!   2. PNG  — fast path when Graph2png is available.
*!   3. GPH  — Stata's native binary format; always saves. The
*!             researcher can open it in Stata; the model can't see
*!             it. Set as last resort so the run never produces
*!             nothing at all.
*!
*! Returns the exported filename in ``r(file)`` and the format in
*! ``r(format)``. Returns empty ``r(file)`` if every attempt failed.
*! Failures are logged via ``r(last_rc)`` so the caller can write a
*! useful helper_errors.jsonl entry instead of "_rc=0".
*!
*! Usage:
*!   _nora_export_plot using "`rundir'/_nora_plots", basename("residuals") width(1600)
*!   if "`r(file)'" != "" {
*!       * succeeded — write manifest entry with `r(file)' and `r(format)'
*!   }

program define _nora_export_plot, rclass
    version 13
    syntax using/, basename(string) [ width(integer 1600) ]

    * mkdir is non-fatal — a successful directory creation OR
    * "directory exists" both leave us in a state where graph export
    * can write into the dir.
    capture mkdir "`using'"

    * Disambiguate basename if a prior helper call in this same run
    * already produced ``basename.{pdf,png,eps,gph}``. Without this,
    * a script that calls e.g. ``nora_plot_coefficients`` twice would
    * overwrite the first plot AND append a second manifest row
    * pointing at the same on-disk file — the model would see two
    * "different" plots that are in fact both the second fit. We
    * check all four output extensions, not just the one we're
    * about to write, so the disambiguation is stable regardless of
    * which translator path won the previous call.
    local _basename "`basename'"
    local _i = 2
    forvalues _try = 1/100 {
        local _exists 0
        foreach _ext in pdf png eps gph {
            capture confirm file "`using'/`_basename'.`_ext'"
            if !_rc {
                local _exists 1
            }
        }
        if !`_exists' continue, break
        local _basename "`basename'_`_i'"
        local _i = `_i' + 1
    }
    local basename "`_basename'"

    local exported ""
    local fmt ""
    local last_rc 0

    * 1. PDF — most reliable on modern Stata. Stata 17+ writes PDF
    * natively; older versions go through Graph2pdf which sometimes
    * exists even when Graph2png doesn't.
    capture graph export "`using'/`basename'.pdf", replace as(pdf)
    if !_rc {
        local exported "`basename'.pdf"
        local fmt "pdf"
    }
    else {
        local last_rc = _rc
    }

    * 2. PNG — works when Graph2png is present.
    if "`exported'" == "" {
        capture graph export "`using'/`basename'.png", ///
            replace as(png) width(`width')
        if !_rc {
            local exported "`basename'.png"
            local fmt "png"
        }
        else {
            local last_rc = _rc
        }
    }

    * 3. EPS — Stata's native PostScript path. Doesn't depend on
    * the per-format translators that PDF/PNG go through, so this
    * survives installs where Graph2png AND Graph2pdf are both
    * missing. The bridge converts EPS → PNG via macOS sips for
    * researcher thumbnails AND model vision, so EPS landing here
    * is ALMOST as good as PDF — modulo the Mac-only conversion.
    if "`exported'" == "" {
        capture graph export "`using'/`basename'.eps", ///
            replace as(eps)
        if !_rc {
            local exported "`basename'.eps"
            local fmt "eps"
        }
        else {
            local last_rc = _rc
        }
    }

    * 4. GPH — Stata's native binary format. Always saves (no
    * translator). Researcher-only: the bridge can't rasterize
    * this without re-running Stata, so the model won't see it on
    * the next turn. Better than producing nothing.
    if "`exported'" == "" {
        capture graph save "`using'/`basename'.gph", replace
        if !_rc {
            local exported "`basename'.gph"
            local fmt "gph"
        }
        else {
            local last_rc = _rc
        }
    }

    return local file = "`exported'"
    return local format = "`fmt'"
    return scalar last_rc = `last_rc'
end
