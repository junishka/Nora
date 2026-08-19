# Changelog

Notable changes per release. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow semver, pre-1.0.

## [0.11.1] - 2026-08-18

Two interface fixes on top of `0.11.0`, both about the window moving
when it shouldn't. The model picker opened past the right edge of the
screen, and finishing a turn yanked the transcript away from wherever
the researcher was reading.

- **Model picker fits on screen.** The popup took its position from
  the shared permission-popup rule, which anchors to the left edge —
  correct for the permission chip at the left end of the composer row,
  wrong for the model chip at the right end, where a left-anchored
  320px popup grew away from the screen. At the default 960px window
  it ran 172px past the viewport. It now opens leftward from the chip
  and clamps its
  width to the viewport, so it stays fully visible at any window size
  (checked from 480px to 1440px). The permission popup is unchanged.
- **The transcript stays where you put it.** Each incoming reply
  scrolled the view to pin that reply under the question that
  prompted it. Within a single turn every reply re-anchored on the
  same originating question, so a multi-part answer repeatedly threw
  the view back up the transcript, and the end of a turn did it once
  more — losing your place mid-read. Now the transcript follows new
  content only while you are already at the bottom. Scroll up to read
  something and it stays put: replies land silently below, the
  "scroll to latest" button appears, and you return on your own terms.
  Sending a message still jumps to your own message, since that
  follows directly from your own action.

Researcher-visible: opening the model picker no longer pushes the
window into a horizontal scroll, and reading back through a
conversation is no longer interrupted every time an answer lands.

Both predate `0.11.0` — the popup's positioning rule and the
scroll-anchoring behaviour are older than the effort bar — but both
got easier to hit once the picker carried an effort control worth
opening and answers had more parts to them.

**Updated in 0.11.1**: the model picker popup opens leftward from its
chip and clamps to the viewport instead of running off the right edge
of the window, and the transcript no longer scrolls itself when a
reply arrives or a turn ends unless you are already at the bottom.

## [0.11.0] - 2026-08-18

Minor bump on top of `0.10.3`. Two threads, one small fix. The model
picker moves to the current frontier families on both providers, and
reasoning effort — previously pinned to `xhigh` behind the scenes —
becomes a researcher-facing dial next to the model.

- **Model picker: Claude 5 and GPT-5.6.** Anthropic now lists Sonnet
  5, Opus 5, and Fable 5; OpenAI lists GPT-5.6 Terra (the cost tier,
  \$2/\$12 per MTok) and GPT-5.6 Sol (the flagship, \$5/\$30). Both
  defaults move to the direct successor of the previous default —
  Sonnet 5 for Anthropic, Sol for OpenAI — so an upgrade doesn't move
  anyone's per-token rate: Sol sits at the same \$5/\$30 as the
  GPT-5.5 it replaces. The heavier tiers (Opus 5, Fable 5) and the
  cheaper one (Terra) stay explicit per-session opt-ins. Anthropic
  ids keep the `[1m]` suffix so the provider follows one convention;
  on the Claude 5 family a 1M window is already both the default and
  the maximum.
- **Per-session reasoning effort.** The model popup gains an Effort
  bar under the model list, and the composer chip reads
  `Sonnet 5 · xhigh`. The default is `xhigh`, which is exactly what
  both providers were hard-pinned to before, so nothing changes
  until a researcher moves the dial. Effort is per-session like the
  model, is refused while a turn is in flight, and persists to
  `.nora/session_state.json`, so a session that ran at the ceiling
  reopens at the ceiling.
- **The effort bar is provider-specific.** Anthropic offers `low`,
  `medium`, `high`, `xhigh`, `max`; OpenAI offers `low`, `medium`,
  `high`, `xhigh`, `pro`. The bar is rebuilt from the selected
  model's provider, so a level that provider can't take is never
  offered. `pro` is not an effort value — it is OpenAI's separate
  `reasoning.mode`, which buys more model work per turn — but for
  "how hard should this try" it is the rung above `xhigh`, so that
  is where it sits; the provider unpacks it into `mode="pro"` plus
  the highest expressible effort, so the top rung never reasons less
  than the one below it. Switching providers maps by rank, which
  ties the two ceilings: a session on Anthropic `max` that moves to
  OpenAI lands on `pro` rather than dropping a rung, and the
  confirmation toast says so when a level moves.
- **Effort applies differently per provider, and the UI says so.**
  OpenAI carries effort in each request, so a change lands on the
  next message with the conversation intact. The Claude Agent SDK
  accepts effort only when the CLI process launches, so an Anthropic
  session is closed and re-warmed on the next message — the same
  warm-start path a cross-provider model switch already takes, with
  the conversation carried across by the context prefix. The
  confirmation toast states which of the two happened.
- **Sandbox: venv base prefixes.** The Python executor's sandbox
  profile granted a venv's `sys.prefix` but not its `sys.base_prefix`
  — and for a venv the standard library lives under the base. With a
  uv-managed CPython (`~/.local/share/uv/python/`) the interpreter
  launched inside the sandbox and died with `Failed to import
  encodings module` before running anything. Homebrew and python.org
  installs were unaffected, since their base and prefix are the same
  path. Both base prefixes are now granted.

Researcher-visible: the picker lists Sonnet 5 / Opus 5 / Fable 5 and
GPT-5.6 Terra / Sol, with an Effort control beneath it and the level
shown on the composer chip; Python analysis now runs on machines
whose `python3` is a uv-managed virtualenv.

Sessions saved under `0.10.x` resolve unchanged, with one expected
consequence of the catalog move: per-session model memory restores
only models still in the picker, so a session last used on Sonnet
4.6, Opus 4.8, or GPT-5.5 reopens on the current default rather than
a model no longer listed. Recorded effort is restored either way.

**Updated in 0.11.0**: model picker moved to the Claude 5 family
(Sonnet 5, Opus 5, Fable 5) and the GPT-5.6 family (Terra, Sol) with
same-price successors as the defaults, a per-session reasoning-effort
dial in the model popup that persists across reopens and survives
provider switches, and a sandbox-profile fix that lets uv-managed
virtualenv interpreters load their own standard library.

## [0.10.3] - 2026-05-29

Focused patch on top of `0.10.2`. Two threads. The first closes a
real gap: researchers can now release an exact unique-value count
through the descriptive payload, where before every obvious path
returned a significance-rounded approximation. The second moves the
Anthropic model picker from Opus 4.7 to Opus 4.8.

- **Exact unique-value counts via `distinct_count`.** A unique count
  computed and pushed through `from_magnitude_table` (sum) or a
  scalar `mean` was rounded to 3-to-5 significant figures by the SDC
  layer, so 165,813 distinct EINs came back as 166,000 or 165,800.
  `distinct_count` is now a first-class optional field on the
  descriptive payload, surfaced by `nora.from_summarize(...,
  distinct_count=...)` (Python), `nora$from_summarize(...,
  distinct_count=...)` (R), and `nora_result_sum varname, distinct`
  (Stata, which computes the exact count itself over the same `if`
  sample via `egen group`). It is an allowed integer field, so the
  sanitizer passes it through unrounded, the descriptive table
  renders it in a new `Distinct` column, and the carried one-line
  summary includes it so the figure survives context trimming.
- **Small unique counts coarsen to `<10`.** A `distinct_count`
  between 1 and 9 partitions a sample (already gated to at least 10
  rows) into a handful of groups, the same disclosure surface as a
  small frequency cell, so the sanitizer holds it to the
  cell-suppression floor and emits `<10` (mirroring the existing
  `missing_count` rule). Counts at or above the threshold pass
  through exact.
- **Anthropic model picker: Opus 4.7 to Opus 4.8.** The picker now
  offers Opus 4.8 in place of Opus 4.7. Opus 4.8 is a drop-in for
  4.7 (same 1M context window, 128k max output, adaptive thinking,
  no breaking API changes), so the `[1m]` context suffix and every
  call site are unchanged. Per-session model-memory and set-model
  rollback tests now pin the 4.8 id.

Researcher-visible: a unique or cardinality count now comes back as
the exact integer (for example 165,813) in a `Distinct` column
instead of a rounded magnitude, except when fewer than 10 distinct
values would themselves be disclosive (shown as `<10`); the
Anthropic picker lists Opus 4.8.

No data shape changes beyond the additive `distinct_count` field; no
API breaks. Sessions saved under `0.10.0` through `0.10.2` resolve
unchanged.

**Updated in 0.10.3**: exact `distinct_count` on the descriptive
payload across the Python, R, and Stata helpers (passed through the
sanitizer unrounded, rendered in a `Distinct` column, carried in the
result summary), small unique counts coarsened to `<10` like
`missing_count`, and the Anthropic model picker moved from Opus 4.7
to Opus 4.8.

## [0.10.2] — 2026-05-23

Focused patch on top of `0.10.1`. Two threads. The first is
presentation. The model now defaults to the table form for tool
results that carry a `markdown` field, and multi-result composites
render with proper bold group headers even when the script baked
hypothesis tags into helper labels. The second is error-readability
and test-debt. Python tracebacks no longer lead with Nora's wrapper
and runpy frames before reaching `script.py`, and the prior
allowlist-style Stata tests catch up to the post-0.10.1 denylist
contract so the suite reflects the actual SDC posture.

- **Few-shot demonstration biases replies toward tables.** OpenAI
  sessions get a structural 4-item exchange (user, function_call,
  function_call_output, assistant) prepended to round 1 of every
  session. The model sees `submit_script` return a payload whose
  `markdown` field is a pipe table, then an assistant reply that
  pastes that table verbatim plus one short sentence. Costs ~350
  input tokens once per session, rides for free on subsequent turns
  via `previous_response_id`, and re-injects after chain-expiry
  resets. The Claude Agent SDK does not expose a seam to seed prior
  assistant or tool_result turns, so the same demonstration ships as
  rule 7 of `_STYLE_RIDER` on the Anthropic path. Opt-out:
  `NORA_DISABLE_FEWSHOT=1` (OpenAI), `NORA_DISABLE_STYLE_RIDER=1`
  (Anthropic, covers the whole rider).
- **`compose_layout` consolidates baked-in hypothesis prefixes.**
  When the script labels each helper call as `nora_result_*,
  label("H1 :: outcome_a")` and the compose spec passes
  bare result_ids without `group.label`, the renderer used to
  produce a flat ungrouped table with the hypothesis tag pasted into
  every row's first cell. It now detects a `<TAG> :: ` prefix shared
  by every row label in a group and hoists `TAG` to a bold header
  row, or strips it from rows when `group.label` already matches.
  Conservative on edge cases: partial-prefix groups, mixed tags, and
  explicit `group.label` that disagrees with the common prefix all
  leave the input unchanged (the model's explicit decision wins
  over the heuristic). Companion change in the `compose_results`
  description names the anti-pattern explicitly.
- **Python tracebacks reference `script.py` again.** `_extract_python`
  drops `_nora_wrapper.py` and `<frozen runpy>` frames before
  composing the user-visible excerpt. The wrapper's documented
  intent at `executor.py:1768` was that tracebacks reference
  `script.py` and the wrapper's `_nora_*` names never leak into user
  scope, but neither path matched `LIB_PAT`'s site-packages
  fragment, so the excerpt was leading with one wrapper frame plus
  three or four runpy frames before reaching the researcher's code.
  The model now sees the script frame at the top, which is what
  it's actually supposed to act on.
- **System prompt brevity rules tightened.** "Shorter is better.
  When in doubt, cut." prepended to the Voice section so the
  brevity prior is set early, where Opus weights it most. The
  post-run reading rule narrows to "extremely concise and direct
  interpretation on the aspect relevant to the current discussion",
  replacing wording that had been licensing multi-frame
  theorization.
- **Stata redaction tests aligned to the post-0.10.1 denylist
  contract.** Commit `41903e2` (shipped in 0.10.1) moved Stata
  user-code excerpts from allowlist (full redaction) to denylist
  (forward through `_forward_short_body` with cap, data-shape
  detect, downstream scrubs). The tests in
  `tests/test_stata_phase_aware.py` and one case in
  `tests/test_audit_fixes_9.py` were missed in that commit and still
  encoded the prior allowlist contract. Rewritten to pin the new
  contract (forwarded command and body, documented short-scalar
  leak, data-shape detect bound). New paired tests cover the
  data-shape mitigation. Source unchanged.
- **`openai_lockdown` covers the few-shot prepend.** Existing
  chain-pointer test sets `NORA_DISABLE_FEWSHOT=1` to isolate its
  scope. New `test_first_turn_prepends_fewshot_demonstration_then_
  user_message` pins the 5-item round-1 shape;
  `test_disable_fewshot_env_var_skips_prepend` verifies the opt-out.

Researcher-visible: tool results that ship markdown reliably appear
as tables in the reply instead of being re-narrated; multi-result
composites render with proper bold group headers even when the
script baked hypothesis tags into helper labels; Python failure
excerpts start at `script.py` instead of Nora internals; replies are
shorter, more direct, and stay on the relevant frame.

No data shape changes; no API breaks. Sessions saved under `0.10.0`
or `0.10.1` resolve unchanged.

**Updated in 0.10.2**: structural table-preference few-shot (OpenAI
+ Anthropic), `compose_layout` auto-consolidates `<TAG> :: `
prefixes shared by every row in a group, Python excerpts drop
`_nora_wrapper.py` and `<frozen runpy>` frames, system prompt
brevity rules tightened, post-0.10.1 Stata denylist test coverage
caught up, plus few-shot lockdown coverage.

## [0.10.1] — 2026-05-18

Focused patch on top of `0.10.0`. Closes a class of "model can't
tell what's wrong" incidents in the audit and error-readability
paths, plus several presentation papercuts surfaced during a 0.10.0
review pass.

- **Audit coverage caught up to the 0.10 shapes.** `_effective_n`
  recognises `coefficient_table_with_fit_stats` alongside the legacy
  `linear_regression` alias, so the row-count audit fires for current
  R / Python / Stata regression payloads (it was silently skipping
  every modern emission). Extends to `correlation_matrix`,
  `marginal_effects`, `kaplan_meier`, `factor_decomposition`,
  `cluster_analysis`. RDD and DiD deliberately skipped with
  documented reasoning (bandwidth-restricted N, units-vs-rows
  mismatch).
- **Kaplan-Meier rare-event coarsening.** KM `n_failures` now
  coarsens via `_coarsen_small_cox_counts` so a 1500-subject study
  with 3 events shows `n_failures: "<10"` instead of leaking the
  exact event count. Closes a Cox-vs-KM asymmetry where identically
  shaped data got different SDC treatment by estimator.
- **DBSCAN / HDBSCAN cluster payloads render.** The sanitizer accepts
  density-based methods without centroids, but the renderer returned
  `None`, so the model embedded nothing in the chat reply. Now falls
  back to a `Cluster | Size` table plus a caption carrying
  `n_noise_points`, `silhouette_score`, and the standard fit-quality
  scalars.
- **SDC policy lookup honours the basename contract.**
  `_resolve_sdc_and_source_n` normalises `source_dataset` to its
  basename before policy lookup, matching `get_schema` and the
  contract documented in `policy.py:134-137`. Closes a silent gap
  where `./data.csv` or `sub/data.csv` missed a policy entry keyed
  `data.csv` and the researcher's `non_disclosive_variables` opt-in
  was ignored.
- **Error excerpt readable for the common Stata / R failures.**
  Stata and R user-code error bodies forward through
  `_forward_short_body` (length cap + data-shape detect) instead of
  `[message body redacted]`. The model can now read `"highest_
  forprofit_title_pre_ceo_rank invalid varname"` and shorten the
  identifier without re-probing. Data-shape detector refined so
  pure-identifier varlists / formula args forward (a Stata varlist
  or an R `pmin(a, b, c, d, e, f, g)` is legitimate context); mixed
  shape row dumps still get caught. Credential scrubs, URL-userinfo
  collapse, and path-to-basename normalisation still run on the
  final excerpt. Python unchanged: the source-line preview already
  exposes the identifier in the common `KeyError` case.
- **`compose_results` ergonomics.** Rows accept bare result_id
  strings; row labels auto-resolve from the store's helper-call
  label, with explicit per-row labels still overriding. Cuts spec
  verbosity for big multi-result batches so the model takes the
  comparison-table path more readily.
- **System prompt: multi-result presentation as guidance.** Reframed
  as descriptive ("a comparison table reads more cleanly than prose
  for patterns across stored results") with the grouping decision
  named as the model's call after seeing results. `compose_results`
  positioned as the natural surface, `_inline_markdown_omitted` note
  redirected toward it.
- **UI papercuts.** Landing oversize-file message simplified to one
  line, themed via `var(--hot-pink)` with `opacity: 0.9`, auto
  dismisses after 4s. `.tool-output` scrollbar now shares the dark
  panel chrome with `.tool-code` (was painting a near-white track
  against burgundy in light mode).
- **Inline budget bump.** `_INLINE_MARKDOWN_BUDGET` 30k → 45k so a
  typical 12-15 regression batch stays inline before stage-2 stub
  replacement. `_REPLAY_TEXT_ENVELOPE_CAP` bumped 64k → 96k to keep
  the linked-constraint margin comfortable.

Researcher-visible: regression-batch row-count audits stop silently
passing; KM rare-event counts match Cox suppression; DBSCAN results
actually render in chat; Stata / R errors name the offending
identifier so the model can fix without re-probing; comparison
tables stay inline for typical batches instead of getting stubbed.

No data shape changes; no API breaks. Sessions saved under `0.10.0`
resolve unchanged.

**Updated in 0.10.1**: audit-coverage gaps closed across the 0.10
analysis shapes (row-count alias, KM `n_failures`, DBSCAN render,
SDC policy basename); Stata / R error bodies forward under a
denylist posture so the diagnostic reaches the model; plus
`compose_results` bare-string rows, inline markdown budget bump
(30k → 45k), and themed landing oversize message.

## [0.10.0] — 2026-05-16

Late-beta release. Expands the sanitized analysis surface from
seven shapes to thirteen, **ships Stata parity for the four
high-usage shapes that were previously R+Python-only**
(mixed-effects, cluster, factor, KM), renames the regression
bucket to a name that reads honestly when applied to GLM / Cox PH
/ fixest / IV, and adds reusable per-subgroup SDC infrastructure.
Two remaining shapes stay Stata-deferred for substantive reasons
(DiD: SSC install flow documented; RDD: numerics not yet verified
against the CCT 2014 reference).

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
  the allowlist. Helpers across all three languages: R
  `from_cluster` (kmeans + hierarchical), Python `from_cluster`
  (KMeans + AgglomerativeClustering), and **Stata
  `nora_result_cluster`** (kmeans + hierarchical with linkage;
  centroids + within-SS computed from the dataset directly since
  Stata's cluster commands don't store them natively). DBSCAN
  supported via `nora.result(type="cluster_analysis",
  method="dbscan", ...)`.
- **`factor_decomposition`** — PCA + factor analysis as one
  shape. Per-component eigenvalue / variance clamps; per-variable
  loading clamps; per-observation factor scores
  (`fit.transform(X)` / `fit$x`) structurally absent from the
  allowlist. Helpers across all three languages: R `from_pca`,
  Python `from_pca`, and **Stata `nora_result_factor`** (PCA +
  factor with pcf / pf / ml / ipf extraction; reads loadings,
  eigenvalues, explained-variance ratios from `e()`; ML-FA
  goodness-of-fit fields from `e(chi2_ms)` / `e(p_ms)` /
  `e(df_ms)` / `e(ll)`). R / Python factor-analysis variants
  beyond PCA (`factanal`, `factor_analyzer`,
  `statsmodels.multivariate.factor`) still emit via
  `nora.result(...)` until dedicated helpers ship.

### Added — regression bucket sub-features

- **Mixed-effects diagnostics across all three languages.** R `lmer` /
  `glmer`, Python `statsmodels.mixedlm`, **and Stata `mixed` /
  `meglm`** all emit `random_effects_variance` (`{varname: variance}`
  plus a `residual` entry for Gaussian families), `n_groups_per_level`
  (same disclosure profile as `fixed_effects`), `fit_method` (`"REML"`
  / `"ML"`), and `icc` for the intercept-only single-grouping case.
  Python `from_lm` accepts `group_variable="..."` so the helper knows
  which dataset column the grouping factor came from (R extracts it
  from the formula; Stata reads `e(ivars)`). Stata's path runs
  `estat recovariance` for natural-scale RE covariance matrices and
  `estat icc` for the ICC, and restricts the coefficient submatrix to
  the fixed-effects equation (`e(k_f)` leading columns) so transformed
  variance components don't appear as "coefficients" in the payload.
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

### Fixed (silent infrastructure bug)

- **Stata `nora_result_km` was unreachable under 0.9.x.** The
  `.ado` file existed in `src/nora/runtime/` but was never added
  to the executor's `_stage_runtime_library` `stata_ados` tuple,
  so the helper never landed on the runtime adopath at script
  time. Researchers calling `nora_result_km` got "command
  unknown" with no obvious cause. Fixed in `executor.py` (both
  the staging list and the `capture program drop` shadowing
  defense) and pinned by `test_executor_profile.py`.

### Deferred (named, post-release roadmap)

- **Stata `did_event_study` via `csdid` (SSC).** Nora's
  `install_packages` tool does not reach SSC; the system prompt
  now instructs the model to direct the researcher to
  `ssc install csdid` in their own Stata window first. Once
  installed, a Stata script can fit csdid and emit via
  `nora.result(...)`; no opinionated helper is provided. R or
  Python in the same session remains the recommended path.
- **Stata `rdd` via SSC `rdrobust` — targeted for 0.10.1.** The
  Stata SSC `rdrobust` port has maintenance lag and the output
  has not been verified against the CCT 2014 reference that R and
  Python `rdrobust` already reproduce (we have those two pinned
  cross-language in `tests/test_from_rdd_real_fits.py`). Shipping a
  Stata helper whose numerics are not trusted would be worse than
  no helper. **Concrete go/no-go for 0.10.1:** on a Stata-equipped
  machine, fit `rdrobust y x, c(0)` in Stata and the equivalent
  call in R / Python on the same simulated DGP (e.g., `y = 0.5 *
  (x > 0) + 0.3*x + ε`, `n = 2000`, `seed = 42`); compare
  conventional / bias-corrected / robust τ, SE, and the
  CCT-optimal bandwidths within 0.5% relative tolerance. If they
  agree, write `nora_result_rdd.ado` following the same pattern as
  `nora_result_factor.ado` (read `e()` macros, JSONL append-mode
  emit, `%21.17e` floats); ship in 0.10.1. If they disagree,
  document the divergence in `direction.md` and keep the deferral.
  This is roughly a one-hour task for anyone with Stata installed,
  not an open-ended research question.
- **Sub-estimator helpers in Python** for Sun-Abraham
  (`pyfixest`) and TWFE-ES (`linearmodels`); de Chaisemartin
  `DIDmultiplegt` in any language.
- **Composition-attack hardening / release ledger.** Inherent to
  every interactive analysis system; see `docs/direction.md`
  §"Cumulative-inference / cross-query composition" for the DP /
  τ-ARGUS / release-ledger options. Not urgent at the current
  single-researcher single-machine deployment shape.

### Patch update — 2026-05-17

Same `0.10.0` tag, new `.dmg`. Closes a class of "script silently
fails at startup; model speculates" incidents triggered when the
researcher's only `python3` was Apple's `/usr/bin/python3` (an
xcselect stub that dlopens `libxcrun` from a path the Nora sandbox
doesn't allow). Every script died before user code ran, and the
redaction layer treated the failure as user-code-suspicious, so the
model never saw the actual error.

- **Sandbox-health probe in interpreter discovery.** Rejects Apple's
  xcselect stub at detection time by running `python3 -c "print(1)"`
  under a real Nora profile (sharing the executor's profile builder
  so probe and run can't drift). A separate baseline check
  distinguishes "sandbox itself is broken" from "interpreter
  rejected by a working sandbox", so a researcher inside a
  nested-sandbox harness doesn't get pointed at Homebrew.
- **Broader interpreter PATH coverage.** Launcher now picks up
  pyenv shims, uv-managed Pythons (per-version subdirs under
  `~/.local/share/uv/python/cpython-*`), conda (`~/miniconda3` and
  five other common roots), and python.org framework versions
  (`/Library/Frameworks/Python.framework/Versions/*/bin`).
- **`nora --doctor` CLI + bridge method.** Per-runtime report
  (`ok` / `warning` / `blocked`) with concrete fix advice. Catches
  the Apple-stub case at first-run before any script submission;
  non-zero exit so a shell-init wrapper can gate `.app` launch. The
  bridge method exposes the same data for a future UI banner.
- **Structured `_environment` block on every `submit_script` response.**
  Carries interpreter binary, version, `sys.prefix`, installed vs
  missing required/optional packages, sandbox-exec presence, and
  any sandbox-probe-rejected `python3` candidates. Phase-safe by
  construction. The model can self-diagnose environment-shaped
  failures instead of guessing from absence.
- **Phase-aware redaction in `error_summary`** via a kernel-level
  buffer-split stderr. The Python preamble `dup2`s fd 2 onto
  `stderr.phase_a` at startup and `stderr.phase_b` just before user
  code, so the SDC boundary is enforced at the file-descriptor
  level. Pre-user-code failures (libxcrun, sandbox-deny, preamble
  syntax) reach the model unredacted because no user code could
  have touched data yet. User-code failures stay redacted as
  before, and now correctly: the prior "no traceback means safe"
  classifier would have leaked stderr from a segfault that wrote
  to it before crashing. Stata gets equivalent treatment via a
  preamble marker line; failing commands logged before the marker
  forward verbatim, commands after stay redacted.
- **Clean on-disk `script.do` / `script.py` + `mixed`/`meglm`
  helper parity.** The researcher's `<run_dir>/script.{do,py}` now
  contains only the model's code; the executor's preamble (capture
  drops, adopath, cd, fd-level stderr split) lives in a sibling
  `_nora_wrapper.{do,py}` that the runner invokes. Opening the
  script from Finder or the result card's "Open in Stata / Python"
  button shows the bytes the model wrote. Also fixes
  `nora_result_regress.ado` for `mixed` and `meglm` on Stata 18+
  where the previous `r(Cov_<N>)` walk silently dropped variance
  components after Stata's rename to `r(Cov<N>)`: variances are
  now read from `e(b)` (`lns<L>_<E>_<T>` / `lnsig_e` /
  `var(_cons[group])`), the meglm path is detected via
  `e(cmd2) == "meglm"` (gsem-backed in newer Stata), and ICC
  prefers `r(icc2)`.

Researcher-visible: scripts that used to silently exit on Apple's
xcselect stub now produce a clean message naming the actual cause,
with a concrete fix. The on-disk run logs in `.nora/runs/` keep
the un-scrubbed stderr regardless of phase; only the model-visible
`debug_excerpt` channel is affected by the redaction change.

No data shape changes; no API breaks. Sessions saved under the
original `0.10.0` `.dmg` resolve unchanged.

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
