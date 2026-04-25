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
