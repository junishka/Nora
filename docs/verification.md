# Nora — manual verification recipes

Short recipes for developer-level sanity checks that CI can't run —
either because a commercial dependency (Stata) isn't available on
the runner, or because the machine-level behavior being verified
(full sandbox enforcement end-to-end) requires a fresh un-nested
environment.

Run these before tagging a release, or when touching anything in
`executor.py` or the runtime libraries.

## Prerequisites

- macOS with `/usr/bin/sandbox-exec` present (default).
- `Rscript` on `PATH` (from an R install).
- Stata installed at `/Applications/Stata` OR on `PATH` as
  `stata-mp` / `stata-se` / `stata`.
- `uv` tooling.
- A shell that is NOT itself inside a sandbox (sandbox-apply must
  actually work — test harness sandboxes like some CI runners block
  nested `sandbox_apply`, and every integration test will skip).

## Quick end-to-end smoke

From the repo root:

```bash
uv run python -c "
from pathlib import Path
from nora.executor import run_script

cwd = Path('/tmp/nora_verify')
cwd.mkdir(exist_ok=True)

print('=== R ===')
r = run_script('R', 'nora\$from_lm(lm(mpg ~ wt, data = mtcars), label = \"smoke\")', cwd)
print(f'ok={r.ok}, n={r.result_payload[\"n\"] if r.result_payload else None}')

print('=== Stata ===')
r = run_script('Stata', 'sysuse auto, clear\nregress price mpg\nnora_result_regress, label(\"smoke\")', cwd)
print(f'ok={r.ok}, n={r.result_payload[\"n\"] if r.result_payload else None}')
"
```

Expect `ok=True` for both. Any failure means the sandbox profile, the
runtime library, or the authenticity-token plumbing has regressed.

## Full pytest run (including gated integration tests)

```bash
uv run pytest -q
```

On a machine with both R and Stata installed and sandbox-apply
working, every integration test should run (none should skip) and
the full suite should pass. On a machine missing Stata, the
`test_executor_sandbox_stata.py` tests skip cleanly and the rest
still passes.

## Exfil-attempt probe (R side)

Confirms the narrow subpath allowlist actually stops reads outside
cwd and runtime-dep paths:

```bash
uv run python -c "
from pathlib import Path
from nora.executor import run_script
cwd = Path('/tmp/nora_verify')
cwd.mkdir(exist_ok=True)

for target in ['/etc/passwd', '/Library/Keychains/System.keychain',
               Path.home() / '.zshrc']:
    r = run_script('R', f'''
probe <- tryCatch(readLines(\"{target}\", n=1),
                  error = function(e) paste(\"DENIED:\", conditionMessage(e)),
                  warning = function(w) paste(\"DENIED:\", conditionMessage(w)))
df <- data.frame(x=1:12, y=(1:12)*2)
nora\$from_lm(lm(y ~ x, df), label = paste0(\"probe=\", substr(probe, 1, 60)))
''', cwd)
    label = r.result_payload['label'] if r.ok else '(executor-error)'
    print(f'{str(target):50s} => {label}')
"
```

Every line should print `probe=DENIED: ...`. Any line that shows the
actual file contents is a critical sandbox regression.

## Runtime-library bypass probe

Confirms the per-run authenticity token rejects hand-crafted
payloads:

```bash
uv run python -c "
from pathlib import Path
from nora.executor import run_script
cwd = Path('/tmp/nora_verify')
cwd.mkdir(exist_ok=True)
r = run_script('R', '''
result_path <- Sys.getenv(\"NORA_RESULT_PATH\")
con <- file(result_path, open = \"w\", encoding = \"UTF-8\")
writeLines(\"{\\\"type\\\":\\\"linear_regression\\\",\\\"n\\\":1000,\\\"response_variable\\\":\\\"y\\\",\\\"predictor_variables\\\":[\\\"x\\\"],\\\"coefficients\\\":{\\\"x\\\":1.0},\\\"standard_errors\\\":{\\\"x\\\":0.1},\\\"r_squared\\\":0.5}\", con)
close(con)
''', cwd)
print(f'ok={r.ok}, error={r.error}')
"
```

Expect `ok=False` and `error` mentioning `_token` or "authenticity".
A script that manages to bypass the library now has to first recover
the token from the interpreter's loaded environment — which shows up
in the executed script visible to the researcher — rather than just
writing a file.

## Clean-install smoke (signed `.dmg` only)

The 0.10.0 release ships five new analysis shapes
(`did_event_study`, `rdd`, `cluster_analysis`, `factor_decomposition`,
`kaplan_meier`) plus mixed-effects diagnostics. Every helper
through that pipe was tested on a developer machine with the full
analysis stack already installed. The first thing a clean-install
user is likely to hit is a helper-side missing-package failure
(`matplotlib`, `rdrobust`, `differences`, `did`, `survival`,
`fixest`, `lme4`) — the failure mode the regular CI suite cannot
exercise. **Run this before signing the release `.dmg`** on a
machine that has not run Nora before. Either a clean macOS user
account, a fresh VM, or a colleague's machine works.

Setup (one-time on the test machine):

```bash
# Install nothing beyond what the .dmg needs. Specifically do NOT
# install the optional Python / R packages — the point is to
# verify the helper-failure paths surface correctly.
```

Then:

1. **Open `Nora.dmg`, drag to `/Applications`, double-click**.
   First launch must hit the auth screen with no Gatekeeper
   error. (Gatekeeper passing is the codesign + notarization
   check; if it fails, do not ship.)
2. **Auth screen — enter an API key** for one provider. Should
   land at the landing-screen drop zone.
3. **Drop a small CSV** (any 100-row dataset). The session should
   open into chat with the dataset listed.
4. **Run one fit per analysis shape**, in any language available
   on the test machine:
   - `coefficient_table_with_fit_stats` (OLS regression). Should
     succeed end-to-end. Verifies the baseline path.
   - `descriptive` / `frequency_table` / `crosstab` /
     `magnitude_table` / `correlation_matrix`. Quick batch.
   - `t_test`.
   - `kaplan_meier`. Verifies `survival` package detection (R) /
     `lifelines` or statsmodels (Python).
   - `did_event_study`. Verifies `did` (R) / `differences` (Py).
   - `rdd`. Verifies `rdrobust` (R or Py — both are missing on a
     clean install).
   - `cluster_analysis`. Verifies `scikit-learn` (Py).
   - `factor_decomposition`. Verifies `scikit-learn` (Py PCA).
5. **For each missing-package failure**, confirm:
   - The result envelope reports `status: "execution_failed"`
     with a `debug_excerpt` carrying the language's native
     error idiom.
   - The model surfaces it to the researcher with a clear
     install hint (e.g. "`install.packages('did')` is required
     for Callaway-Sant'Anna").
   - Calling `install_packages` from chat pops the Approve /
     Deny modal listing the packages.
   - After approval, the package installs and the next fit
     succeeds.
6. **Run a plot helper** (`plot_coefficients` after a regression).
   Verifies `matplotlib` (Py) / `ggplot2` (R) fallback and the
   manifest-allowlisted capture path.
7. **`plot_residuals`** — confirm the model only sees the
   `researcher_only: true` marker, while the researcher sees the
   thumbnail in the Files panel.
8. **Switch sessions / drop a second dataset** — confirm
   concurrent-session isolation. The original session's in-flight
   work (if any) continues.

A clean-install run that surfaces every missing-package failure
gracefully (no silent helper crash, no model hallucinating a
result) is the green-light for signing. Failures here are
shippable as known issues only if accompanied by a documented
install hint the model surfaces consistently.
