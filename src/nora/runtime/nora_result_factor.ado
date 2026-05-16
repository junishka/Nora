*! version 0.0.1  Nora runtime: emit a factor_decomposition payload.
*!
*! Call after Stata's `pca` or `factor` command. The helper reads e()
*! for loadings, eigenvalues, explained variance, communalities, and
*! uniqueness, then writes the structured payload to NORA_RESULT_PATH.
*!
*! Usage (PCA):
*!   pca x1 x2 x3 x4 x5
*!   nora_result_factor, method("pca") label("PCA on five features")
*!
*!   pca x1 x2 x3 x4 x5
*!   rotate, varimax
*!   nora_result_factor, method("pca") rotation("varimax") label("...")
*!
*! Usage (factor analysis — principal-factor / ML / iterated PF):
*!   factor x1 x2 x3 x4 x5, pf
*!   nora_result_factor, method("principal_factor") label("PF factor analysis")
*!
*!   factor x1 x2 x3 x4 x5, ml
*!   nora_result_factor, method("maximum_likelihood") label("ML factor analysis")
*!
*! Notes:
*! - Component labels are synthetic: PC1 / PC2 / ... for method=="pca",
*!   factor1 / factor2 / ... for the factor-analysis methods. The
*!   sanitizer's cross-field check uses these labels to validate keys
*!   in eigenvalues / explained_variance / loadings / etc.
*! - Per-observation factor scores (Stata's `predict, score`) are
*!   structurally absent — no field on this shape's allowlist accepts
*!   them.
*! - The helper uses e() macros from `pca` or `factor`. Calling it
*!   after a different e()-leaving command produces an invalid
*!   payload that the sanitizer rejects with a clear reason.

program define nora_result_factor
    version 13
    syntax , method(string) [label(string) rotation(string)]

    * JSON-escape label.
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

    local _method = lower("`method'")
    local _rotation = lower("`rotation'")

    * Validate method against the sanitizer's allowlist.
    local _valid_methods "pca factor_analysis principal_factor maximum_likelihood minimum_residual"
    local _ok = 0
    foreach m of local _valid_methods {
        if "`_method'" == "`m'" local _ok = 1
    }
    if !`_ok' {
        display as error "nora_result_factor: method '`method'' not recognised. Use one of: `_valid_methods'"
        exit 198
    }

    * Verify a pca / factor result is in e(). e(cmd) is "pca" for the
    * `pca` command and "factor" for `factor`. Bare guards mean a
    * researcher who runs `regress` and then this helper gets a clear
    * error rather than a malformed payload.
    if "`e(cmd)'" != "pca" & "`e(cmd)'" != "factor" {
        display as error "nora_result_factor: no pca/factor result in memory (e(cmd) is '`e(cmd)''). Run pca or factor first."
        exit 198
    }

    * Component-label prefix: PC for PCA, factor for FA.
    local _cprefix = "PC"
    if "`_method'" != "pca" {
        local _cprefix = "factor"
    }

    * Loadings matrix. After `pca`, e(L) is the unrotated loadings
    * (variables × components). After `rotate, ...`, e(r_L) carries
    * the rotated loadings — prefer that when present.
    * After `factor`, e(L) holds the factor loadings; e(r_L) the
    * rotated version. Same convention.
    tempname Lmat
    if "`e(r_L)'" != "" {
        capture matrix `Lmat' = e(r_L)
    }
    if "`e(r_L)'" == "" | _rc {
        capture matrix `Lmat' = e(L)
    }
    if _rc {
        display as error "nora_result_factor: e(L) loadings matrix unreachable"
        exit 198
    }

    local _vnames : rownames `Lmat'
    local _cnames : colnames `Lmat'
    local _n_vars : word count `_vnames'
    local _n_comps : word count `_cnames'

    if `_n_vars' == 0 | `_n_comps' == 0 {
        display as error "nora_result_factor: empty loadings matrix"
        exit 198
    }

    * n_observations. e(N) for both pca and factor.
    if "`e(N)'" == "" | missing(`=e(N)') {
        display as error "nora_result_factor: e(N) is missing"
        exit 198
    }
    local _n_obs = `=e(N)'

    * Begin JSON line.
    tempname fh
    file open `fh' using `"`path'"', write text append
    file write `fh' `"{"type":"factor_decomposition""'
    file write `fh' `","_token":"`_nora_token'""'
    if "`label'" != "" {
        file write `fh' `","label":"`label'""'
    }

    file write `fh' `","method":"`_method'""'

    * Rotation: explicit override > e(r_class) detection > "none".
    * After `rotate` post-pca, e(r_criterion) and e(r_class) are
    * populated; e(r_class) is "varimax" / "promax" / etc.
    local _rot_out = "`_rotation'"
    if "`_rot_out'" == "" {
        if "`e(r_class)'" != "" {
            local _rot_out = lower("`e(r_class)'")
        }
    }
    if "`_rot_out'" == "" {
        local _rot_out = "none"
    }
    file write `fh' `","rotation":"`_rot_out'""'

    file write `fh' `","n_observations":`_n_obs'"'
    file write `fh' `","n_variables":`_n_vars'"'
    file write `fh' `","n_components":`_n_comps'"'

    * variables list — already taken from the loadings rownames.
    file write `fh' `","variables":["'
    local _first = 1
    forvalues i = 1/`_n_vars' {
        local v : word `i' of `_vnames'
        if !`_first' file write `fh' ","
        file write `fh' `""`v'""'
        local _first = 0
    }
    file write `fh' "]"

    * Synthetic component labels (PC1, PC2, ... or factor1, factor2, ...).
    * The sanitizer validates against these names for every per-component
    * dict and the loadings inner keys.
    file write `fh' `","components":["'
    forvalues j = 1/`_n_comps' {
        if `j' > 1 file write `fh' ","
        file write `fh' `""`_cprefix'`j'""'
    }
    file write `fh' "]"

    * loadings — nested {variable: {component: value}}.
    file write `fh' `","loadings":{"'
    forvalues i = 1/`_n_vars' {
        local v : word `i' of `_vnames'
        if `i' > 1 file write `fh' ","
        file write `fh' `""`v'":{"'
        forvalues j = 1/`_n_comps' {
            if `j' > 1 file write `fh' ","
            if missing(`Lmat'[`i', `j']) {
                file write `fh' `""`_cprefix'`j'":null"'
            }
            else {
                local _x = strofreal(`Lmat'[`i', `j'], "%21.17e")
                file write `fh' `""`_cprefix'`j'":`_x'"'
            }
        }
        file write `fh' "}"
    }
    file write `fh' "}"

    * Eigenvalues. After `pca`, e(Ev) is a 1 × n_components row matrix
    * of eigenvalues (in decreasing order). After `factor`, e(Ev) is
    * the matrix of factor variances (analogous quantity); some methods
    * also expose e(eigvals).
    tempname Evmat
    capture matrix `Evmat' = e(Ev)
    if !_rc & colsof(`Evmat') >= `_n_comps' {
        file write `fh' `","eigenvalues":{"'
        forvalues j = 1/`_n_comps' {
            if `j' > 1 file write `fh' ","
            if missing(`Evmat'[1, `j']) {
                file write `fh' `""`_cprefix'`j'":null"'
            }
            else {
                local _x = strofreal(`Evmat'[1, `j'], "%21.17e")
                file write `fh' `""`_cprefix'`j'":`_x'"'
            }
        }
        file write `fh' "}"
    }

    * Explained variance ratio + cumulative variance.
    * After pca: e(rho) is the cumulative-variance scalar of the
    * RETAINED components (last value of the per-component cumulative
    * series). The per-component series is in the e(L) matrix's
    * scaling, or computed from Ev / sum(Ev).
    * Better: compute from Ev directly so the math is consistent
    * across pca and factor.
    if !_rc & colsof(`Evmat') >= `_n_comps' {
        local _total_var = 0
        forvalues j = 1/`_n_comps' {
            if !missing(`Evmat'[1, `j']) {
                local _total_var = `_total_var' + `Evmat'[1, `j']
            }
        }
        if `_total_var' > 0 {
            file write `fh' `","explained_variance_ratio":{"'
            local _cumul = 0
            forvalues j = 1/`_n_comps' {
                if `j' > 1 file write `fh' ","
                local _evj = `Evmat'[1, `j']
                if missing(`_evj') {
                    file write `fh' `""`_cprefix'`j'":null"'
                }
                else {
                    local _r = `_evj' / `_total_var'
                    local _x = strofreal(`_r', "%21.17e")
                    file write `fh' `""`_cprefix'`j'":`_x'"'
                }
            }
            file write `fh' "}"

            file write `fh' `","cumulative_variance":{"'
            local _cumul = 0
            forvalues j = 1/`_n_comps' {
                if `j' > 1 file write `fh' ","
                local _evj = `Evmat'[1, `j']
                if !missing(`_evj') {
                    local _cumul = `_cumul' + `_evj' / `_total_var'
                }
                local _x = strofreal(`_cumul', "%21.17e")
                file write `fh' `""`_cprefix'`j'":`_x'"'
            }
            file write `fh' "}"
        }
    }

    * Communalities / uniqueness — factor-analysis-specific. pca's
    * "communalities" are just sum(loadings²) per variable; Stata
    * doesn't post them in e() for pca. For factor, e(Psi) is the
    * uniqueness vector (variables in rownames order).
    if "`e(cmd)'" == "factor" {
        tempname Psimat
        capture matrix `Psimat' = e(Psi)
        if !_rc & colsof(`Psimat') >= `_n_vars' {
            file write `fh' `","uniqueness":{"'
            forvalues i = 1/`_n_vars' {
                local v : word `i' of `_vnames'
                if `i' > 1 file write `fh' ","
                if missing(`Psimat'[1, `i']) {
                    file write `fh' `""`v'":null"'
                }
                else {
                    local _x = strofreal(`Psimat'[1, `i'], "%21.17e")
                    file write `fh' `""`v'":`_x'"'
                }
            }
            file write `fh' "}"

            * Communality = 1 - uniqueness (for standardized variables;
            * holds when e(Psi) is on the same scale as the loadings,
            * which Stata's factor uses by default).
            file write `fh' `","communalities":{"'
            forvalues i = 1/`_n_vars' {
                local v : word `i' of `_vnames'
                if `i' > 1 file write `fh' ","
                if missing(`Psimat'[1, `i']) {
                    file write `fh' `""`v'":null"'
                }
                else {
                    local _c = 1 - `Psimat'[1, `i']
                    local _x = strofreal(`_c', "%21.17e")
                    file write `fh' `""`v'":`_x'"'
                }
            }
            file write `fh' "}"
        }
    }

    * Goodness-of-fit (ML factor analysis): e(chi2_i) is the chi-
    * squared independence test; e(chi2_ms) is the model-vs-saturated
    * chi-squared. The latter is the "factor analysis goodness-of-fit"
    * test most researchers want. e(p_ms) is its p-value.
    if "`e(method)'" == "ml" {
        if "`e(chi2_ms)'" != "" & !missing(`=e(chi2_ms)') {
            local _x = strofreal(`=e(chi2_ms)', "%21.17e")
            file write `fh' `","chi_squared":`_x'"'
        }
        if "`e(p_ms)'" != "" & !missing(`=e(p_ms)') {
            local _x = strofreal(`=e(p_ms)', "%21.17e")
            file write `fh' `","chi_squared_p_value":`_x'"'
        }
        if "`e(df_ms)'" != "" & !missing(`=e(df_ms)') {
            file write `fh' `","degrees_of_freedom":`=e(df_ms)'"'
        }
        if "`e(ll)'" != "" & !missing(`=e(ll)') {
            local _x = strofreal(`=e(ll)', "%21.17e")
            file write `fh' `","log_likelihood":`_x'"'
        }
    }

    file write `fh' "}" _newline
    file close `fh'

    display as text "nora_result_factor: wrote result to " as result "`path'"
end
