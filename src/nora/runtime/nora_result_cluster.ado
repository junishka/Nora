*! version 0.0.1  Nora runtime: emit a cluster_analysis payload.
*!
*! Call after a Stata clustering command (cluster kmeans /
*! wardslinkage / completelinkage / averagelinkage / singlelinkage /
*! centroidlinkage / medianlinkage) and pass the cluster-membership
*! variable. The helper computes cluster sizes, per-cluster centroids,
*! within-cluster sum-of-squares, and the SS decomposition, then writes
*! the structured payload to NORA_RESULT_PATH for the Nora executor.
*!
*! Usage (k-means):
*!   cluster kmeans x1 x2 x3, k(4) name(myclus)
*!   nora_result_cluster x1 x2 x3, clusvar(myclus) method("kmeans") ///
*!       label("k=4 on three features")
*!
*! Usage (hierarchical):
*!   cluster wardslinkage x1 x2 x3
*!   cluster generate wardclus = groups(4)
*!   nora_result_cluster x1 x2 x3, clusvar(wardclus) ///
*!       method("hierarchical") linkage("ward") label("Ward k=4")
*!
*! Notes:
*! - Synthetic labels (cluster_1, cluster_2, ...) are generated from the
*!   sorted distinct values in clusvar. The raw cluster-variable values
*!   never cross to the model; the sanitizer enforces the same
*!   constraint and would reject a payload that tried.
*! - Per-observation cluster assignment is structurally absent from the
*!   payload (no field on the cluster_analysis allowlist accepts it).
*! - SDC behaviors fire downstream in the sanitizer: clusters below the
*!   suppression threshold drop WHOLE (size + centroid row +
*!   within_cluster_ss together); centroid precision is clamped per
*!   cluster's own N.

program define nora_result_cluster
    version 13
    syntax varlist(min=1 numeric), clusvar(varname numeric) ///
        method(string) [label(string) linkage(string) distance(string)]

    * JSON-escape `label` (only free-text Claude-controllable field).
    local label : subinstr local label "\" "\\", all
    local label : subinstr local label `"""' `"\""', all
    local label : subinstr local label "`=char(10)'" " ", all
    local label : subinstr local label "`=char(13)'" " ", all
    local label : subinstr local label "`=char(9)'" " ", all

    * Env-var gates — same contract as the other helpers. The
    * NORA_RUN_TOKEN check rejects naive "write hand-crafted JSON"
    * bypasses; sophisticated bypasses still need to extract the
    * token from the running interpreter, which appears in the
    * executed script the researcher reviews.
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
    local _linkage = lower("`linkage'")
    local _distance = lower("`distance'")

    * Validate method against the sanitizer's allowlist. Helper-side
    * refusal is friendlier than letting the sanitizer reject downstream
    * (the researcher sees the helper error in the raw log; the model
    * sees only the sanitizer rejection reason which carries less
    * context).
    local _valid_methods "kmeans hierarchical agglomerative pam kmedoids dbscan hdbscan gaussian_mixture spectral"
    local _ok = 0
    foreach m of local _valid_methods {
        if "`_method'" == "`m'" local _ok = 1
    }
    if !`_ok' {
        display as error "nora_result_cluster: method '`method'' not recognised. Use one of: `_valid_methods'"
        exit 198
    }

    * Distinct cluster IDs (sorted ascending). levelsof skips missings
    * automatically. n_clusters = count of distinct non-missing values.
    quietly levelsof `clusvar', local(_clust_levels)
    local _n_clusters : word count `_clust_levels'
    if `_n_clusters' == 0 {
        display as error "nora_result_cluster: clusvar `clusvar' has no non-missing values"
        exit 198
    }

    * n_observations: count of rows with a non-missing cluster
    * assignment. Rows with missing clusvar (outside the fitted
    * sample, NaN features in some clusterers) drop here so totals
    * and per-cluster sums reconcile.
    quietly count if !missing(`clusvar')
    local _n_obs = r(N)

    * n_features and the variable list.
    local _n_features : word count `varlist'

    * Per-row squared distance to the row's cluster centroid.
    * Build via egen mean by cluster, sum of squared deviations
    * across features. Used downstream for within_cluster_ss
    * (sum by cluster) and total_within_ss (overall sum).
    tempvar _wss_row
    quietly gen double `_wss_row' = 0 if !missing(`clusvar')
    * Per-row squared deviation from the GRAND mean across features
    * — drives total_ss. Same shape as _wss_row but per-cluster
    * means replaced by single grand-mean per variable.
    tempvar _tss_row
    quietly gen double `_tss_row' = 0 if !missing(`clusvar')

    local _vi = 0
    foreach v of varlist `varlist' {
        local _vi = `_vi' + 1
        tempvar _mn`_vi' _gmn`_vi'
        quietly egen double `_mn`_vi'' = mean(`v') if !missing(`clusvar'), by(`clusvar')
        quietly summarize `v' if !missing(`clusvar'), meanonly
        scalar _gmean_`_vi' = r(mean)
        quietly replace `_wss_row' = `_wss_row' + (`v' - `_mn`_vi'')^2 if !missing(`clusvar')
        quietly replace `_tss_row' = `_tss_row' + (`v' - _gmean_`_vi')^2 if !missing(`clusvar')
    }

    * Aggregate SS scalars.
    quietly summarize `_wss_row' if !missing(`clusvar'), meanonly
    local _total_wss = r(sum)
    quietly summarize `_tss_row' if !missing(`clusvar'), meanonly
    local _total_ss = r(sum)
    local _between_ss = `_total_ss' - `_total_wss'
    local _ss_ratio = .
    if `_total_ss' > 0 & !missing(`_total_ss') {
        local _ss_ratio = `_between_ss' / `_total_ss'
    }

    * Begin JSON line.
    tempname fh
    file open `fh' using `"`path'"', write text append
    file write `fh' `"{"type":"cluster_analysis""'
    file write `fh' `","_token":"`_nora_token'""'
    if "`label'" != "" {
        file write `fh' `","label":"`label'""'
    }

    file write `fh' `","method":"`_method'""'
    if "`_linkage'" != "" {
        file write `fh' `","linkage":"`_linkage'""'
    }
    if "`_distance'" != "" {
        file write `fh' `","distance_metric":"`_distance'""'
    }

    file write `fh' `","n_observations":`_n_obs'"'
    file write `fh' `","n_clusters":`_n_clusters'"'
    file write `fh' `","n_features":`_n_features'"'

    * variables list — already validated as numeric varlist. Names
    * are Stata identifiers, no JSON escaping required.
    file write `fh' `","variables":["'
    local _first = 1
    foreach v of varlist `varlist' {
        if !`_first' file write `fh' ","
        file write `fh' `""`v'""'
        local _first = 0
    }
    file write `fh' "]"

    * Synthetic cluster_labels: cluster_1, cluster_2, ... in sorted
    * order of the raw clusvar values. Raw cluster IDs never cross —
    * they could leak group-membership information in some study
    * designs (e.g., cluster-id == hospital-id). The sanitizer also
    * enforces "labels are synthetic" but we generate them clean here.
    file write `fh' `","cluster_labels":["'
    local _ci = 0
    foreach _raw of local _clust_levels {
        local _ci = `_ci' + 1
        if `_ci' > 1 file write `fh' ","
        file write `fh' `""cluster_`_ci'""'
    }
    file write `fh' "]"

    * cluster_sizes — {cluster_X: count}. Below-threshold clusters
    * drop whole in the sanitizer; we emit every cluster's count
    * unconditionally and let SDC do the suppression.
    file write `fh' `","cluster_sizes":{"'
    local _ci = 0
    foreach _raw of local _clust_levels {
        local _ci = `_ci' + 1
        if `_ci' > 1 file write `fh' ","
        quietly count if `clusvar' == `_raw'
        file write `fh' `""cluster_`_ci'":`r(N)'"'
    }
    file write `fh' "}"

    * centroids — {cluster_X: {var_name: mean}}. Required for
    * kmeans/hierarchical (the sanitizer enforces this when method
    * is not dbscan/hdbscan). Per-cluster precision clamping fires
    * downstream; emit raw means here.
    if "`_method'" != "dbscan" & "`_method'" != "hdbscan" {
        file write `fh' `","centroids":{"'
        local _ci = 0
        foreach _raw of local _clust_levels {
            local _ci = `_ci' + 1
            if `_ci' > 1 file write `fh' ","
            file write `fh' `""cluster_`_ci'":{"'
            local _first = 1
            foreach v of varlist `varlist' {
                quietly summarize `v' if `clusvar' == `_raw', meanonly
                if !`_first' file write `fh' ","
                if missing(`=r(mean)') {
                    file write `fh' `""`v'":null"'
                }
                else {
                    local _x = strofreal(`=r(mean)', "%21.17e")
                    file write `fh' `""`v'":`_x'"'
                }
                local _first = 0
            }
            file write `fh' "}"
        }
        file write `fh' "}"
    }

    * Per-cluster within_cluster_ss. Same disclosure profile as
    * cluster_sizes — paired in the suppression rule (cluster drops
    * whole means size + centroid row + within_cluster_ss entry all
    * suppress together).
    file write `fh' `","within_cluster_ss":{"'
    local _ci = 0
    foreach _raw of local _clust_levels {
        local _ci = `_ci' + 1
        if `_ci' > 1 file write `fh' ","
        quietly summarize `_wss_row' if `clusvar' == `_raw', meanonly
        if missing(`=r(sum)') {
            file write `fh' `""cluster_`_ci'":null"'
        }
        else {
            local _x = strofreal(`=r(sum)', "%21.17e")
            file write `fh' `""cluster_`_ci'":`_x'"'
        }
    }
    file write `fh' "}"

    * Aggregate SS scalars. Each gated on a finite computed value so
    * a degenerate fit (all observations identical, n_clusters == 1)
    * doesn't write Stata-missing "." into the JSON.
    if !missing(`_total_wss') {
        local _x = strofreal(`_total_wss', "%21.17e")
        file write `fh' `","total_within_ss":`_x'"'
    }
    if !missing(`_total_ss') {
        local _x = strofreal(`_total_ss', "%21.17e")
        file write `fh' `","total_ss":`_x'"'
    }
    if !missing(`_between_ss') {
        local _x = strofreal(`_between_ss', "%21.17e")
        file write `fh' `","between_cluster_ss":`_x'"'
    }
    if !missing(`_ss_ratio') {
        local _x = strofreal(`_ss_ratio', "%21.17e")
        file write `fh' `","ss_ratio":`_x'"'
    }

    file write `fh' "}" _newline
    file close `fh'

    display as text "nora_result_cluster: wrote result to " as result "`path'"
end
