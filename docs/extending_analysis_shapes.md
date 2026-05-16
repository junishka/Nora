# Extending analysis coverage

How to extend the sanitizer's coverage in the two ways it gets
extended: **adding a field to an existing shape**, and **adding a
new shape**. Worked from the patterns the existing 12 shapes
already establish; reference, not tutorial.

The standard for "extension is done" is a two-bar test, applied
the same way for every shape, every language:

1. **Sanitizer-valid** — a real fit's emitted payload passes
   `nora.sanitizer.sanitize()` with `ok=True`.
2. **Inference-adequate** — the payload carries enough fit-metric
   or design-detail fields that the model can interpret the result
   without round-tripping for missing scalars. *Helper produces a
   payload that doesn't crash* is below the bar.

This bar exists because the original `from_lm` audit (March 2026)
found Cox PH hard-failing through the R helper and every GLM
shipping a coefficient table with zero fit metrics — both modes
passed mocked-fit tests and went unnoticed. Mocked tests verify
"if a fit has these attributes, the helper works." Real-fit tests
verify "a real fit has these attributes." The difference is
non-trivial.

---

## What's already in

The sanitizer accepts 12 shapes ([src/nora/sanitizer.py:_HANDLERS](../src/nora/sanitizer.py)).
Roughly grouped:

| Shape | Covers | Helpers |
|---|---|---|
| `coefficient_table_with_fit_stats` (legacy alias `linear_regression`) | OLS, logit, probit, Poisson, NegBin, Cox PH, fixest with absorbed FE, IV/2SLS, mixed-effects (lmer / glmer / mixedlm / mixed / meglm) | R `from_lm` + `from_iv`; Python `from_lm` + `from_iv`; Stata `nora_result_regress` (covers `regress`/`logit`/`probit`/`poisson`/`stcox`/`xtreg fe`/`areg`/`ivregress`/`mixed`/`meglm`) |
| `t_test` | one-sample / two-sample / Welch / paired | `from_t_test` (R, Python), `nora_ttest` (Stata) |
| `descriptive` | `from_summarize` univariate | each language |
| `frequency_table` | 1-D counts | each language |
| `crosstab` | 2-D counts | each language |
| `magnitude_table` | sum/mean by group with dominance | each language |
| `correlation_matrix` | Pearson / Spearman / Kendall | each language |
| `cluster_analysis` | kmeans + hierarchical (Ward / complete / average / single linkage) | R `from_cluster`; Python `from_cluster`; Stata `nora_result_cluster` (after `cluster kmeans` / `cluster wardslinkage` / etc.) |
| `factor_decomposition` | PCA + factor analysis (pcf / pf / ml / ipf) | R `from_pca`; Python `from_pca`; Stata `nora_result_factor` (after `pca` or `factor`) |
| `did_event_study` | Callaway-Sant'Anna (helpers); Sun-Abraham + TWFE-ES (R helpers); de Chaisemartin + Python sub-estimators via generic `result()` | R `from_callaway_santanna` + `from_sun_abraham` + `from_twfe_event_study`; Python `from_callaway_santanna`. Stata deferred — no realistic workflow yet (see [handoff Stata coverage matrix](handoff.md#stata-coverage-matrix-release-status)) |
| `rdd` | sharp + fuzzy local-polynomial (via rdrobust) | R + Python `from_rdd`. Stata helper targeted 0.10.1 pending cross-language numerics verification (see [CHANGELOG.md](../CHANGELOG.md) deferred section for the protocol) |
| `kaplan_meier` | safe-form survival (median + horizon scalars) | R + Python + Stata `from_kaplan_meier` / `nora_result_km` |

All have real-fit pins under `tests/test_from_*_real_fits.py` and
`tests/test_nora_result_*_real_fits.py`. The 16+16+16 property
tests in `tests/test_{did_event_study,rdd_and_kaplan_meier}.py`
verify the sanitizer's behavior on hand-crafted payloads.

---

## Layer 1 — extending an existing shape

When a researcher's use case lands inside an existing shape but
needs a field the allowlist doesn't accept, the move is to widen
the allowlist. The cost is small, the risk is small, the
disclosure surface grows by exactly the field you add.

### When this is the right move

- The new estimator emits the same payload **shape** as an
  existing one. Mixed-effects (lmer / glmer) is a coefficient
  table with fit stats + a variance-components block — fits the
  regression bucket modulo the variance block. Logit / probit /
  Poisson / NegBin / Cox PH all already fit because they're
  coefficient tables with different fit metrics.
- The new diagnostic is **a bounded scalar or a small dict of
  counts**. Cluster-robust SE metadata is bounded
  (cluster_var_name + cluster_count); first-stage F is a scalar.
  Both went into the regression bucket as allowlist additions in
  one diff each.
- The disclosure profile is **already understood**. Coefficient
  names, cluster identifiers, FE-variable names are dataset
  column names the model has already seen in the schema. Their
  cardinalities are aggregates. Adding `n_clusters` next to
  `fixed_effects` is one more entry in the same dict-int category.

### When to NOT extend an existing shape

- The new estimator's output is **shaped differently** — nested
  dict, indexed-by-time, multiple linked tables. ATT(g, t) from
  Callaway-Sant'Anna is a `{cohort: {event_time: value}}` panel
  with per-cohort N gates; it doesn't fit a flat coefficient
  table. Build a new shape.
- The new field is a **distribution / curve / per-observation
  series**. Kaplan-Meier step function, McCrary density curve,
  binscatter cells near a cutoff. These are researcher-only
  diagnostics; they go on the **structural-exclusion** side of
  the privacy carve-out, not the allowlist side.
- The new diagnostic needs a **new SDC primitive**. If the
  suppression rule is "drop the entire cohort when its N is
  small" or "drop horizon h when n_at_risk_h < threshold", that's
  not a generic allowlist add — it goes into a sanitizer-module
  function specific to the shape.

### The five frozen sets per shape

Every shape's sanitizer module declares the same kind of frozen
sets. Working example: [`_sanitize_linear_regression` block in
src/nora/sanitizer.py](../src/nora/sanitizer.py).

```python
_OLS_REQUIRED:           frozenset[str]   # fields that MUST be present
_OLS_ALLOWED_NUMERIC_FIELDS:  frozenset[str]   # scalar floats
_OLS_ALLOWED_INT_FIELDS:      frozenset[str]   # scalar ints
_OLS_ALLOWED_STRING_FIELDS:   frozenset[str]   # short identifiers
_OLS_ALLOWED_DICT_NUMERIC:    frozenset[str]   # dict[str, float]
_OLS_ALLOWED_LIST_STRING:     frozenset[str]   # list[str]
```

Most additions are one-line entries in one of these sets. The
sanitizer's `_collect_allowed` ([src/nora/sanitizer.py:554](../src/nora/sanitizer.py))
walks the raw payload, drops unknown top-level fields with a
count-only transformation note (the names are deliberately
withheld — they're caller-controlled and could carry raw data
bytes), and validates types per slot.

### Cross-field validation

Some `dict_numeric` fields have keys that should equal coefficient
names (`coefficients`, `standard_errors`, `t_statistics`,
`p_values`, `vif`). The regression sanitizer enforces this
cross-field check in a loop near
[src/nora/sanitizer.py:1243](../src/nora/sanitizer.py). Adding a
new dict-numeric field whose keys are coefficient names = just
add to `_OLS_ALLOWED_DICT_NUMERIC`. Adding one whose keys are
*different* (FE-variable names, clustering-variable names) means
adding to **both**:

- `_OLS_ALLOWED_DICT_NUMERIC` so the type filter accepts it
- `_OLS_DICT_FIELDS_SKIP_COEF_KEY_CHECK` so the cross-field loop
  skips it

`fixed_effects` and `n_clusters` are both in the skip set today.

### Cardinality dicts vs measurement dicts

For dicts that carry **integer counts** (cluster cardinalities,
FE level counts) rather than data-precision measurements, the
sigfigs clamp distorts the value at small N. The fix is the
`_OLS_DICT_FIELDS_INT_COUNTS` set
([src/nora/sanitizer.py](../src/nora/sanitizer.py)) — fields in
this set get coerced to int rather than precision-clamped. Same
disclosure profile (positive integer per dataset variable name).

### Enum validation

For string fields with a small fixed valid set
(`kernel = "triangular" | "uniform" | "epanechnikov"`,
`bandwidth_selector` in the RDD shape, `aggregation_method` in
DiD), define a `_<SHAPE>_VALID_<FIELD>: frozenset[str]` and check
inside the sanitize function. Pattern from
`_sanitize_rdd` ([src/nora/sanitizer.py](../src/nora/sanitizer.py)):

```python
bwsel = out.get("bandwidth_selector")
if bwsel is not None and bwsel not in _RDD_VALID_BANDWIDTH_SELECTOR:
    transformations.append(
        "dropped 'bandwidth_selector' value (not in valid set)"
    )
    del out["bandwidth_selector"]
```

Drop-with-transformation-note is the right behavior; reject the
whole payload only when the field is required.

### Worked example — adding `cluster_variables` + `n_clusters`

This is the cluster-robust SE modifier, landed during the
audit arc. End-to-end:

1. Allowlist additions in [src/nora/sanitizer.py](../src/nora/sanitizer.py):
   ```python
   _OLS_ALLOWED_LIST_STRING += "cluster_variables"
   _OLS_ALLOWED_DICT_NUMERIC += "n_clusters"
   _OLS_DICT_FIELDS_SKIP_COEF_KEY_CHECK += "n_clusters"
   _OLS_DICT_FIELDS_INT_COUNTS += "n_clusters"
   ```
2. Helper extraction:
   - Python: detect `cov_type == "cluster"` on the result, pull
     cluster variable name(s) from `cov_kwds["groups"]`, compute
     cardinality per dim. ([src/nora/runtime/nora.py:_extract_cluster_metadata](../src/nora/runtime/nora.py))
   - R: `m$call$cluster` formula → variable names via
     `all.vars`; `attr(summary(m)$cov.scaled, "G")` for counts.
   - Stata: `e(vce) == "cluster"`, `e(clustvar)`, `e(N_clust)`.
3. Real-fit pins ([tests/test_from_lm_python_real_fits.py:test_python_cluster_robust_emits_cardinality](../tests/test_from_lm_python_real_fits.py),
   parallel R + Stata pins): fit OLS with cluster-robust SE,
   assert the emitted payload carries `cluster_variables == ["firm_id"]`
   and `n_clusters == {"firm_id": 40}`.
4. System prompt: one line under "Regression diagnostics"
   describing the auto-emission and what the model can expect to
   see ([src/nora/system_prompt.py](../src/nora/system_prompt.py)).

Cost: ~half a day of focused work including the cross-language
helper updates and tests. No renderer change needed — the
regression renderer doesn't display cluster metadata in the
inline card today (acceptable; it's in the payload for the model
and surfaces on `expand_result(view="full")`).

---

## Layer 2 — adding a new shape

When the output doesn't fit any existing shape — the data is
nested, indexed by time, multi-dimensional, or governed by a new
SDC primitive — build a new shape. The cost is real (one full PR
per shape) but the structure is the same every time.

The minimum unit per shape is **five things**:

1. **Sanitizer module** — required fields, allowlist sets, the
   `_sanitize_<shape>` function, dispatch entry in `_HANDLERS`.
2. **Renderer entry** — `_render_<shape>` in
   [src/nora/result_render.py](../src/nora/result_render.py)
   that produces a table + caption from the sanitized payload,
   plus dispatch entry in `_HANDLERS` there.
3. **Helper(s)** — `from_<shape>` in each supported language,
   wrapping the canonical library for that estimator. Use the
   community-standard package; don't roll your own.
4. **Real-fit pins** — `tests/test_from_<shape>_real_fits.py`,
   one parametrize per supported language, plus cross-language
   equivalence when more than one language has a helper.
5. **System prompt entry** — one paragraph in the
   "Analysis shapes" block of
   [src/nora/system_prompt.py](../src/nora/system_prompt.py)
   covering required fields, the helper signature, privacy
   framing, and the deferral note for any language without a
   helper yet.

If any of these is missing the work isn't done.

### Worked example — `did_event_study` from scratch

The most complex new shape so far. Walks the full arc.

**Step 1 — sanitizer module.** Required fields gate the SDC
primitive's inputs:

```python
_DID_EVENT_REQUIRED = frozenset((
    "type", "groups", "event_times", "att",
    "n_treated_per_group",      # drives the cohort-N gate
))
```

Allowlist sets per type slot (numeric / int / string / list /
nested-dict-of-dict). New for this shape: a `_DID_EVENT_NESTED_DICT_FIELDS`
frozenset listing fields like `att` / `standard_errors` /
`p_values` / `ci_lower` / `ci_upper` — each is a `{cohort: {event_time: value}}`
nested structure. The standard `_collect_allowed` path doesn't
handle nested-dict-of-dict, so the sanitizer function processes
these fields separately after the top-level filter.

The new SDC primitive: **min-N gated by metadata, not by cell
count**. A cohort of 4 firms with 8 quarterly periods has 32
"cells" in the ATT panel, but the disclosure unit is the 4 firms
whose outcome trajectories are summarized by the cohort's ATT
series. The gate fires on `n_treated_per_group[g]`, not on the
cell count of `att[g]`. Cohorts below threshold are dropped
**whole** — partial-cell publication would leak the cohort size
through *which* cells survived (and the cohort label is
data-derived, so its identity is withheld in the transformations
log).

Structural caps on dimensionality: `groups` capped at 50
cohorts, `event_times` at 30. Payloads exceeding these reject
outright (probable adversarial; a real CS analysis ships a
handful of cohorts over ±5 event-times).

Cross-field validation walks the nested dicts: every outer key
must be a surviving cohort, every inner key must be in
`event_times`, otherwise drop.

**Step 2 — renderer.** `_render_did_event_study` in
[src/nora/result_render.py](../src/nora/result_render.py)
produces a cohort × event-time matrix as a wide markdown table
with the aggregate ATT and method in the caption. The model gets
a drop-in pipe-table for its reply; without this entry, the
inline result card would render as raw JSON and degrade
follow-up reasoning.

**Step 3 — helpers.** Library choice matters: use the canonical
package per language. For CS DiD that's R's `did` package and
Python's `differences` package (chosen over `csdid` for the
cleaner Python-native API).

Both helpers do the same three things:
- Pull cohort labels and event-time grid from the fit
- Pivot ATT(cohort, calendar_t) → ATT(cohort, event_time) (where
  `event_time = t - cohort`)
- Read per-cohort treated counts from the fit's internal data
  surface (not from a passed panel argument — keep the helper's
  data input minimal)

Aggregate ATT comes from the package's aggregate-method call
(`did::aggte` in R, `result.aggregate(type_of_aggregation="simple")`
in Python). Compute the two-sided p-value via `2 * normal(-|z|)`
on `att/se` if the package doesn't expose it directly.

Pass through config from the fit: `comparison_group` (R's
`mp$DIDparams$control_group`, Python's
`attgt.estimation_details()["control_group"]`),
`anticipation_periods`, `base_period`. These don't change the
analysis — they tell the model the identification assumptions
the estimator ran under, which the model needs to write an
honest interpretation.

**Step 4 — real-fit pins.** Three tests in
[tests/test_from_callaway_santanna_real_fits.py](../tests/test_from_callaway_santanna_real_fits.py):

- R end-to-end on a seeded staggered-adoption panel
- Python end-to-end on the same DGP shape
- Cross-language equivalence: assert that R and Python aggregate
  ATTs agree within `3 · pooled_SE` on the shared DGP

The cross-language pin is the **unification check** — if R's
`did` and Python's `differences` started disagreeing on the same
DGP it would mean one had drifted from the CS reference, and we'd
want to know immediately. Strict bit-equivalence isn't achievable
(different RNGs), but 3·SE is generous-but-not-trivial — methodo-
logical divergence shows up as much larger gaps.

**Step 5 — system prompt.** A paragraph in the "Analysis shapes"
section of [src/nora/system_prompt.py](../src/nora/system_prompt.py)
listing required fields, both helper signatures with their
canonical `aggregation_method` defaults per language ("dynamic"
for R `did`, "event" for Python `differences`), the
`estimator: "callaway_santanna"` scope note, and an honest framing
of the deferred Stata path: the only Stata route today is
hand-authoring JSON to `NORA_RESULT_PATH` against the field schema,
which is a contributor escape hatch (not an end-user workflow), so
the model is told to recommend opening R or Python in the same
session and loading the `.dta` via `haven` / `pyreadstat`.

### Privacy carve-out patterns

Two patterns establish the same property from different angles.

**Structural absence** — a field has no slot in the allowlist, so
even hand-crafted payloads through `nora.result(type=X, ...)`
silently drop it. Pinned by sanitizer tests like
`test_rdd_mccrary_curve_structurally_excluded` and
`test_km_curve_data_structurally_excluded`. This is the primary
carve-out for diagnostics like the McCrary density curve,
binscatter near a cutoff, the full KM step function — anything
that's structurally a distribution / curve / per-observation
series over a sensitive variable.

**Helper refusal** — the helper signature raises if a known-bad
kwarg is passed, before any payload is written. Pinned by tests
like `test_python_from_rdd_refuses_density_and_binscatter_kwargs`
and the parallel R `test_r_from_rdd_refuses_mccrary_kwarg`. This
is the **secondary** defense — it catches a script that tries to
slip a curve through `**extra` kwargs to `from_rdd`, before the
sanitizer would have to drop it anyway.

Both layers belong on every new shape that has a privacy
carve-out, not just one. The structural exclusion is the
load-bearing defense; the helper refusal makes intent visible at
the call site.

The right framing in the system prompt is **positive**: *"the
analytical results that cross are τ, bandwidth, effective N"*,
not *"you can't access McCrary density"*. The latter invites
probing; the former describes the shape of the analytical
surface.

### Cross-language helper choice

For shapes with multiple supported languages, use the same
canonical package across all of them when one exists. RDD's
`rdrobust` is published in R, Python, and Stata by the same
authors and shares output structure across languages — same
helper structure works for each. KM's `survival::survfit` (R) /
`statsmodels.SurvfuncRight` (Python) / `sts list` (Stata) are
separate packages but each is the language-standard, and the
helpers agree on what payload they emit.

Don't write inference logic in the helper. The helper's job is to
extract from a fit object into the payload shape. Inference
(log-rank, first-stage F, p-values for missing test outputs) goes
in the caller's script. The helper accepts those as kwargs and
packages them. Two reasons:

- The helper stays small (one function, ~150-200 lines).
- The researcher sees what was computed in their own script,
  rather than having a helper that silently re-runs an analysis
  with different defaults than the package would.

### When a language doesn't have the canonical package

Two patterns surface in the current shapes:

- **Defer with a workaround** — Stata RDD and Stata CS DiD don't
  have helpers shipped today. The two deferrals have different
  reasons:
  - **RDD: numerics-unverified, targeted 0.10.1.** SSC `rdrobust`
    has maintenance-lag risk; cross-language numerics check
    (Stata vs R vs Python `rdrobust` on the same DGP at 0.5%
    relative tolerance) is the go/no-go gate. Protocol in
    [CHANGELOG.md](../CHANGELOG.md) deferred section.
  - **DiD: no realistic Stata workflow.** `csdid` is
    SSC-distributed and `install_packages` does not reach SSC, so
    Nora cannot install it. Even if a researcher runs
    `ssc install csdid` themselves, the only path to emit a
    `did_event_study` payload from Stata is hand-authoring JSON
    against the field schema — that's a contributor escape hatch,
    not an end-user workflow. System prompt directs the model to
    recommend opening R or Python in the same session
    (the `.dta` opens via `haven` / `pyreadstat`).
- **Implement a minimal version** — the runtime helper itself
  doesn't need a package; if the canonical package is unavailable
  but the math is tractable, compute it. KM survival probabilities
  via step-function look-up against the raw durations is a few
  lines of numpy — works without lifelines.

The deferral pattern is preferred when the canonical package
exists; the minimal-version pattern is for shapes where the math
is simple enough that a wrapper would add complexity without
value.

---

## Anti-patterns

- **Don't roll your own when a canonical package exists.** For
  CS DiD, three packages (`did`, `differences`, `csdid`) all
  emit ATT(g, t) in compatible shapes — wrap them, don't
  reimplement. Reimplementing means tracking the inference
  literature; wrapping means tracking one library version.
- **Don't widen the disclosure surface to make a helper
  ergonomic.** If a field needs to be in the payload, it needs
  to be in the allowlist with documented SDC treatment. There is
  no "helper-only field" path.
- **Don't add a `label` field to a shape's allowlist** just
  because the transformations log says "dropped 'label'" on
  every payload. The script-side label is informational; the
  `submit_script` MCP tool stores its own label separately. The
  noise is harmless.
- **Don't reach around `_collect_allowed` to accept arbitrary
  fields.** Every field through the sanitizer goes through one
  of the typed slots. If a new field doesn't fit any slot, add a
  new slot kind, don't carve a back-door.
- **Don't mock fits for helper tests.** The pre-rewrite tests
  for `from_lm` used `_FakeFit` with OLS-shaped attributes, and
  they passed cleanly while real Cox PH and fixest fits aborted
  the helper. Real fits or no test.
- **Don't normalize the input `type` field.** The sanitizer's
  `analysis_type` mirrors what came in. Old payloads with
  `linear_regression` round-trip as `linear_regression`; new
  ones with `coefficient_table_with_fit_stats` round-trip as
  themselves. This keeps existing SQLite stores readable across
  the rename.

---

## File map

The recipe across the codebase:

| Layer | File | What lives there |
|---|---|---|
| Sanitizer | [src/nora/sanitizer.py](../src/nora/sanitizer.py) | Frozen-set allowlists, `_sanitize_<shape>` functions, `_HANDLERS` dispatch, SDC primitives |
| Renderer | [src/nora/result_render.py](../src/nora/result_render.py) | Per-shape `_render_<shape>`, `_HANDLERS` dispatch, table primitives |
| R helper | [src/nora/runtime/nora.R](../src/nora/runtime/nora.R) | `nora$from_<shape>` definitions, per-class dispatch via `inherits()` |
| Python helper | [src/nora/runtime/nora.py](../src/nora/runtime/nora.py) | `nora.from_<shape>` definitions, duck-typed extraction via `_safe_attr` |
| Stata helper | `src/nora/runtime/nora_result_<shape>.ado` | One `.ado` per shape (or per family). Pattern in `nora_result_regress.ado` and `nora_result_km.ado` |
| Real-fit tests | `tests/test_from_<shape>_real_fits.py` | Parametrized over supported languages; cross-language equivalence pin where applicable |
| Property tests | `tests/test_<shape>.py` | Sanitizer behavior against hand-crafted payloads (structural caps, suppression, cross-field validation, privacy carve-outs) |
| System prompt | [src/nora/system_prompt.py](../src/nora/system_prompt.py) | "Analysis shapes" section + per-shape paragraph |
| Dev deps | [pyproject.toml](../pyproject.toml) `[dependency-groups] dev` | Language libraries needed for real-fit tests (statsmodels, rdrobust, differences, …) |

---

## The principle the recipe is for

Every emitted payload has a name, a known output shape, and a
documented SDC story. The model can't request "run something
tabular and show me the cells"; it calls `from_callaway_santanna(mp, ...)`
and gets back a payload whose every field has been thought
about. The recipe in this doc is what keeps that property as
new shapes get added — each new entry in the dispatch table
goes through the same five-thing minimum unit. Generic-tabular
or schema-on-write would sacrifice the property for breadth you
can get more safely by walking this recipe.
