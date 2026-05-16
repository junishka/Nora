# Changelog

Notable changes per release. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow semver, pre-1.0.

## [0.10.0] — 2026-05-16

Late-beta release. Expands the sanitized analysis surface from
seven shapes to twelve, renames the regression bucket to a name
that reads honestly when applied to GLM / Cox PH / fixest / IV,
and adds reusable per-subgroup SDC infrastructure.

### Renamed (load-bearing, read first)

- **`linear_regression` → `coefficient_table_with_fit_stats`.**
  The sanitizer dispatch table now maps the canonical name to the
  same handler the legacy name dispatches to, so every existing
  stored payload, recall, and `expand_result` lookup keeps working
  unchanged. New emissions carry the canonical name; reads of
  older payloads carry whichever name they were written with. The
  legacy name is documented in `sanitizer.py` as a permanent
  alias, not a deprecation — there is no plan to remove it.

  - **Persisted-session impact:** none for end users. Sessions
    saved under 0.9.x resolve transparently on first reopen
    under 0.10.0.
  - **Code-reading impact:** the canonical name is what the
    sanitizer dispatches to in fresh code paths, what the
    documentation refers to, and what every helper emits.
    `_REGRESSION_TYPE_LEGACY` and `_REGRESSION_TYPE_ALIASES` in
    `sanitizer.py` are the single source of truth for the
    aliasing logic.

### Added — analysis shapes

- **`did_event_study`** — modern heterogeneous-treatment DiD.
  Required fields: `groups`, `event_times`, `att` (nested
  `{group: {event_time: value}}`), `n_treated_per_group`. SDC:
  cohort-N gate drops cohorts below threshold whole (group + ATT
  row + SE/p/CI together) so partial-cell publication can't leak
  cohort size. Helpers: R `nora$from_callaway_santanna`,
  `nora$from_sun_abraham`, `nora$from_twfe_event_study`; Python
  `nora.from_callaway_santanna`. Other estimators (Python
  Sun-Abraham / TWFE-ES, de Chaisemartin, Stata `csdid`) take the
  script-side path via `nora$result(type="did_event_study",
  estimator=...)`.
- **`rdd`** — regression discontinuity (CCT 2014 three-flavor
  convention: conventional / bias-corrected / robust, each with
  τ / SE / p / CI). Required: `running_variable`, `cutoff`,
  `tau_robust`, `se_robust`, `effective_n_left`,
  `effective_n_right`. McCrary density and binscatter are
  structurally refused — they're visual diagnostics for the
  researcher only. Helpers: R / Python `from_rdd` wrapping
  `rdrobust::rdrobust`. Stata `rdrobust` port has known
  maintenance lag and is deferred.
- **`kaplan_meier`** — survival in safe form. The model sees
  median survival + S(t) at canonical horizons (`1y` / `3y` /
  `5y` / `10y`), each gated by a per-horizon `n_at_risk_h`. The
  KM step function itself is researcher-only; the log-rank χ²
  across groups crosses. Helpers: R / Python / Stata
  `from_kaplan_meier` / `nora_result_km`.
- **`cluster_analysis`** — kmeans / k-medoids / hierarchical /
  HDBSCAN / Gaussian mixture / spectral / agglomerative.
  Per-cluster precision clamping via `_clamp_dict_by_per_key_n`;
  clusters below the size threshold drop whole. Per-observation
  `labels_` / `cluster_membership` are structurally absent from
  the allowlist. Helpers: R `from_cluster` (kmeans +
  hierarchical), Python `from_cluster` (KMeans +
  AgglomerativeClustering). DBSCAN supported via
  `nora.result(type="cluster_analysis", method="dbscan", ...)`.
- **`factor_decomposition`** — PCA + factor analysis as one
  shape. Per-component eigenvalue / variance clamps; per-variable
  loading clamps; per-observation factor scores
  (`fit.transform(X)` / `fit$x`) structurally absent from the
  allowlist. Helpers: R `from_pca`, Python `from_pca`. Factor
  analysis variants (`factanal`, `factor_analyzer`,
  `statsmodels.multivariate.factor`) emit via `nora.result(...)`.

### Added — regression bucket sub-features

- **Mixed-effects diagnostics.** R `lmer` / `glmer` and Python
  `statsmodels.mixedlm` fits emit `random_effects_variance`
  (`{varname: variance}` plus a `residual` entry), `n_groups_per_level`
  (same disclosure profile as `fixed_effects`), `fit_method`
  (`"REML"` / `"ML"`), and `icc` for the intercept-only single-grouping
  case. Python `from_lm` accepts `group_variable="..."` so the
  helper knows which dataset column the grouping factor came from
  (R extracts it from the formula automatically). **Stata `mixed` /
  `meglm`** are not yet routed through `nora_result_regress`; queued
  for next release.
- **Cluster-robust SE auto-emission.** R fixest `cluster=~var`,
  Python `cov_type="cluster"`, and Stata `vce(cluster id)` all
  auto-emit `cluster_variables`, `n_clusters: {varname: cluster_count}`,
  and `robust_se_type: "cluster"`.
- **Fixed-effects cardinality.** fixest fits emit `fixed_effects:
  {varname: level_count}` — the cardinality crosses, the level
  identities do not.

### Added — SDC infrastructure

- **`_clamp_dict_by_per_key_n`** (`sanitizer/sdc.py`). New
  reusable primitive: given a flat `{subgroup: scalar}` dict and a
  parallel `{subgroup: N}` dict, clamps each scalar by that
  subgroup's own N rather than by an aggregate N. Used today on
  `within_cluster_ss` and `silhouette_per_cluster` (parallel to
  the per-cluster centroid clamp); extensible to any future shape
  that wants per-subgroup precision. The pattern is: a 12-person
  subgroup's stat carries fewer sigfigs than a 12,000-person
  subgroup's, and a single aggregate clamp would over-publish the
  small subgroup.

### Added — tools

- **`list_results_global(query?)`** — cross-session result
  lookup, newest-first, across all sessions under
  `~/.nora-sessions/`. Env-gated by
  `NORA_ALLOW_CROSS_SESSION_RECALL=1` (default off); path-confined
  so prompt-injected lookups can't direct the loader at arbitrary
  on-disk paths. Researcher-side project separation, not a
  privacy property — payloads were sanitized at write time.
- **`expand_result(result_id, session_path?)`** — accepts a
  `session_path` argument for cross-session payload expansion,
  using the same env gate and confinement rule.
- **`search_schema(dataset, query)`** — case-insensitive substring
  filter against variable names, labels, and value-label content
  for wide datasets. Limit defaults to 50, hard max 200; response
  carries `total_matches` and `truncated`.
- **`submit_script_file(filename)`** — runs a `.do` / `.R` /
  `.Rmd` / `.py` from cwd by basename, skipping the round-trip
  cost of re-emitting attached script bytes as inline tool input.
- **`install_packages(language, packages, action?)`** — install /
  remove / reinstall packages on the researcher's machine via an
  Approve / Deny modal. Out-of-band from script execution (which
  is sandboxed and network-denied). The modal is the only
  confirmation step; the prompt instructs the model to call the
  tool directly rather than ask in chat first.

The MCP tool surface is now fourteen tools (was nine).

### Added — request_data types

- **`quartiles`** — 25th / 75th + IQR for a single variable.
  Median deliberately omitted (forbidden at row level by SDC).
- **`correlation_pair`** — Pearson r between two variables,
  complete-case N. Drove the new optional `variable2` field on
  the tool schema.

### Added — request_data type, per-variable opt-in

- **`DatasetPolicy.non_disclosive_variables`** (`policy.json`).
  Variables on the list (typical: `age`, `year_of_birth`,
  `education_years`) get `min_value` / `max_value` through
  descriptive payloads. Default empty — every variable's extremes
  stay suppressed unless explicitly opted in.

### Added — plot helpers

- **R / Python / Stata: `plot_residuals`, `plot_interaction`,
  `plot_coefficients`, `plot_estimate_comparison`.** Each takes a
  fitted-model object and produces a canonical visualization from
  model outputs (residuals, predictions, coefficients) — never
  from raw rows. Manifest-allowlist gated; no
  `register_plot(file, kind)` escape hatch (tried and removed —
  self-attested kind labels are unverifiable). `plot_residuals`
  is researcher-only; the other three cross to the model.
- **Stata export fallback chain.** `_nora_export_plot` tries
  `as(pdf)` → `as(png)` → `as(eps)` → `graph save .gph` so a
  missing `Graph2png` translator doesn't kill the do-file before
  `nora_result_*` runs. `nora_safe_export` provides the same
  fallback for ad-hoc (non-helper) exports.

### Added — diagnostics

- **`vif`, `condition_number`, `vcov`** emitted by `from_lm` (R +
  Python) when the design matrix is reachable.
- **GLM diagnostics** — `pseudo_r_squared`, `log_likelihood`,
  `aic`, `bic`, deviance-based `chi_squared`.
- **Cox PH diagnostics** — `n_subjects`, `n_failures`,
  `concordance`, `log_likelihood`.
- **IV / 2SLS via `from_iv`** — `instrument_variables`,
  `endogenous_variables`, `first_stage_f`, Hansen J, Wu-Hausman.

### Changed

- **Multi-result `submit_script` wire format.** JSONL append-mode
  emit; executor parses N payloads per script with per-line token
  validation; response carries a `results` list with a shared
  `script_run_id`. Sanitizer is stateless per payload so SDC stays
  exact across N. Fixed the loss of 23-of-24 results on
  event-study batches.
- **Partial-success on script abort** — payloads emitted before
  the abort surface as `status: "execution_failed_partial"`
  alongside `debug_excerpt`. All-rejected-then-aborted falls to
  `execution_failed`.
- **Cluster-rebinding on session switch** — store cache rebinds
  to the focused session's cwd; the per-task ContextVar
  (`nora.config.use_cwd`) makes concurrent runners sandbox-safe.
- **Per-session model memory** — `active_model` persists in
  `.nora/session_state.json` and restores on session open.
- **Stata residual-plot scaling** — `nora_plot_residuals` samples
  to 5000 points before PDF export; 200k-row datasets now render
  in ~750ms instead of ~8s.
- **`debug_excerpt` redaction posture** — exception bodies
  redacted wholesale; only parser-anchored framing crosses.
- **Schema fast paths share a CSV/TSV header peek** — `load_data`
  and the names_only payload agree on headerless files.
- **Malformed per-dataset policy entries clamp to the strictest
  tier** — open-fail removed.

### Fixed

- **Concurrent-session race** — switching sessions during an
  in-flight turn no longer trampled the other runner's cwd.
  Tool handlers read cwd via the per-task `ContextVar`, not a
  process-global.
- **Loop-shutdown race** — a runner could be left permanently
  busy after a specific shutdown ordering; fixed.
- **`_materialize_cache_busted_index`** — routes its bust file
  to a temp directory in packaged builds so the write no longer
  modifies the codesigned bundle. Clean-install Gatekeeper now
  passes.
- **Session focus switch** — no longer wipes captured plot
  images.
- **`delete_credential`** — notices external Keychain deletions
  instead of false "delete failed".
- **JSONL `names_only` schema** — unions keys across the file
  instead of reading only the first record.
- **OLS / formula-fit intercept dropping** — `Intercept` and
  `const` were silently dropped from statsmodels formula-fit
  payloads (only `(Intercept)` / `_cons` / `intercept` were in
  the alias list). Fixed.
- **NaN correlation on constant columns** — rejected with named
  culprit instead of producing a NaN cell.

### Deferred (named, post-release roadmap)

- **Stata coverage for `did_event_study` / `rdd` /
  `factor_decomposition` / `cluster_analysis`.** Stata-only
  researchers open R or Python inside the same session — same
  sandbox + sanitizer — until concrete pilot demand surfaces a
  Stata helper.
- **Stata `mixed` / `meglm` mixed-effects route through
  `nora_result_regress`.** Queued for next release; the gap is
  more likely to be hit than the R+Py-only shapes.
- **Sub-estimator helpers in Python** for Sun-Abraham
  (`pyfixest`) and TWFE-ES (`linearmodels`); de Chaisemartin
  `DIDmultiplegt` and Stata `csdid` in any language.
- **Composition-attack hardening / release ledger.** Inherent to
  every interactive analysis system; see `docs/direction.md`
  §"Cumulative-inference / cross-query composition" for the DP /
  τ-ARGUS / release-ledger options. Not urgent at the current
  single-researcher single-machine deployment shape.

## [0.9.1] — 2026-05-15

- Audit-fixes batch: install-packages consent modal-only,
  `debug_excerpt` redaction posture, schema fast-path CSV/TSV
  header peek, `_names_only_payload` row_count_unknown vs
  empty-dataset distinction, JSONL key union, strictest-tier
  clamp on malformed policy entries, `chat_history` reader
  concurrent-delete safety, cancelled `submit_script` no
  longer model-visible, `delete_credential` external Keychain
  awareness, plot images preserved across session focus
  switch, loop-shutdown race fix, pywebview `FileDialog`
  migration, signed-bundle cache-bust write fix.

## [0.9.0] — 2026-05-12

Public-repo cleanup: README license section, SECURITY.md,
package metadata. Release-readiness / documentation alignment
pass: multi-provider framing, unified install docs, signed +
notarized `.dmg`, canonical `junishka/Nora` URLs.

[0.10.0]: https://github.com/junishka/Nora/releases/tag/v0.10.0
[0.9.1]: https://github.com/junishka/Nora/releases/tag/v0.9.1
[0.9.0]: https://github.com/junishka/Nora/releases/tag/v0.9.0
