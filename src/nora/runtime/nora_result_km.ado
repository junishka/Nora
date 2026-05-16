*! version 0.0.1  Nora runtime: emit a kaplan_meier payload (safe form).
*!
*! Call after stset. Computes median survival with CI plus S(t) at
*! preset canonical horizons (1y / 3y / 5y / 10y) — each gated by
*! the sanitizer's per-horizon ``n_at_risk_h`` threshold. The full
*! step function is researcher-only by construction (not in the
*! sanitizer's allowlist for this shape).
*!
*! Time-unit translation is the caller's responsibility. The
*! ``horizons`` option is a space-separated list of ``label:time``
*! pairs where label is one of {1y, 3y, 5y, 10y} and time is the
*! numeric horizon in the same units stset used:
*!
*!   * Data measured in years:
*!       stset t_obs, failure(cens)
*!       nora_result_km, horizons("1y:1 3y:3 5y:5") ///
*!                       time(t_obs) event(cens) label("...")
*!
*!   * Data measured in months:
*!       nora_result_km, horizons("1y:12 3y:36 5y:60") ///
*!                       time(follow_up) event(dead)
*!
*! For grouped / log-rank inference, pass ``group(var)`` and the
*! helper will run ``sts test`` internally.

program define nora_result_km
    version 13
    syntax , horizons(string) [time(string) event(string) ///
        group(string) label(string)]

    * Refuse to run when stset hasn't been declared. The helper
    * reads `_t` / `_d` which are populated by stset.
    capture confirm variable _t
    if _rc {
        display as error "nora_result_km: stset must be run before calling"
        exit 198
    }

    * JSON-escape the user-supplied label.
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
    local _nora_token : env NORA_RUN_TOKEN
    if "`_nora_token'" == "" {
        display as error "NORA_RUN_TOKEN not set. Run through Nora."
        exit 198
    }

    * Median + CI via stci. stci posts r(p50)/r(lb)/r(ub)/r(N_sub).
    tempname med ml mh n_sub n_fail
    scalar `med' = .
    scalar `ml' = .
    scalar `mh' = .
    capture quietly stci
    if !_rc {
        if "`r(p50)'" != "" & !missing(`=r(p50)') scalar `med' = r(p50)
        if "`r(lb)'"  != "" & !missing(`=r(lb)')  scalar `ml'  = r(lb)
        if "`r(ub)'"  != "" & !missing(`=r(ub)')  scalar `mh'  = r(ub)
        if "`r(N_sub)'" != "" & !missing(`=r(N_sub)') scalar `n_sub' = r(N_sub)
    }
    if missing(`n_sub') {
        * Fall back to a direct count of subjects in the stset sample.
        capture quietly stsum
        if !_rc & !missing(`=r(N_sub)') scalar `n_sub' = r(N_sub)
    }
    * Number of failures from the stset _d indicator (1 = event).
    quietly count if _d == 1 & _st == 1
    scalar `n_fail' = r(N)

    * Generate the KM survival curve into a temp variable so we can
    * look up S(t) at each horizon. ``sts generate`` writes the
    * step-function value at every observation's _t.
    tempvar nora_km_surv
    quietly sts generate `nora_km_surv' = s

    tempname fh
    file open `fh' using `"`path'"', write text append
    file write `fh' `"{"type":"kaplan_meier""'
    file write `fh' `","_token":"`_nora_token'""'
    if "`label'" != "" {
        file write `fh' `","label":"`label'""'
    }
    if "`time'" != "" {
        file write `fh' `","time_variable":"`time'""'
    }
    if "`event'" != "" {
        file write `fh' `","event_variable":"`event'""'
    }
    if "`group'" != "" {
        file write `fh' `","group_variable":"`group'""'
    }

    if !missing(`n_sub') {
        file write `fh' `","n_subjects":`=`n_sub''"'
    }
    file write `fh' `","n_failures":`=`n_fail''"'

    if !missing(`med') {
        local _x = strofreal(`med', "%21.17e")
        file write `fh' `","median_survival_time":`_x'"'
    }
    if !missing(`ml') {
        local _x = strofreal(`ml', "%21.17e")
        file write `fh' `","median_survival_ci_lower":`_x'"'
    }
    if !missing(`mh') {
        local _x = strofreal(`mh', "%21.17e")
        file write `fh' `","median_survival_ci_upper":`_x'"'
    }

    * Per-horizon look-up. ``horizons`` is "label1:time1 label2:time2 ...".
    * Use ``tokenize`` to walk the pairs.
    local _h_remaining = "`horizons'"
    while "`_h_remaining'" != "" {
        gettoken _pair _h_remaining : _h_remaining
        if "`_pair'" == "" continue
        * Split on ":" into label and time.
        local _hlabel = ""
        local _htime = ""
        local _colon = strpos("`_pair'", ":")
        if `_colon' > 0 {
            local _hlabel = substr("`_pair'", 1, `_colon' - 1)
            local _htime = substr("`_pair'", `_colon' + 1, .)
        }
        if "`_hlabel'" == "" | "`_htime'" == "" {
            display as error "nora_result_km: malformed horizon entry '`_pair'' (expected label:time)"
            continue
        }
        * Only the four canonical labels survive the sanitizer; emit
        * any label the caller supplies but trust the sanitizer to
        * drop non-canonical ones.
        local _h_num = real("`_htime'")
        if missing(`_h_num') {
            display as error "nora_result_km: horizon time '`_htime'' is not numeric; skipping"
            continue
        }
        * S(h): largest event time ≤ h gives the survival value at h.
        * ``summarize ... if _t <= h`` returns r(min) = the rightmost
        * (smallest) survival value among rows with _t <= h, which by
        * monotonicity is S(h).
        capture quietly summarize `nora_km_surv' if _t <= `_h_num' & _st == 1
        if !_rc & r(N) > 0 & !missing(`=r(min)') {
            local _x = strofreal(`=r(min)', "%21.17e")
            file write `fh' `","survival_at_`_hlabel'":`_x'"'
        }
        * n_at_risk(h) = count of subjects with observed time ≥ h
        * (within the stset sample).
        quietly count if _t >= `_h_num' & _st == 1
        file write `fh' `","n_at_risk_`_hlabel'":`r(N)'"'
    }

    * Log-rank χ² across groups. ``sts test <g>`` posts r(chi2) and
    * r(df); r(p) is empirically not populated, compute p from
    * chi2tail(df, chi2).
    if "`group'" != "" {
        capture quietly sts test `group'
        if !_rc & "`r(chi2)'" != "" & !missing(`=r(chi2)') {
            local _chi = `=r(chi2)'
            local _df  = `=r(df)'
            local _x = strofreal(`_chi', "%21.17e")
            file write `fh' `","logrank_chi_squared":`_x'"'
            if !missing(`_df') & `_df' > 0 {
                local _p = chi2tail(`_df', `_chi')
                local _x = strofreal(`_p', "%21.17e")
                file write `fh' `","logrank_p_value":`_x'"'
            }
            * n_groups: distinct values of `group' in the stset sample.
            quietly levelsof `group' if _st == 1, local(_lvls)
            local _ng : word count `_lvls'
            file write `fh' `","n_groups":`_ng'"'
        }
    }

    file write `fh' "}" _newline
    file close `fh'

    display as text "nora_result_km: wrote result to " as result "`path'"
end
