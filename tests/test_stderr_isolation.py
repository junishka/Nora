"""Regression test: raw stderr / stdout never reaches the model on a
SUCCESSFUL run.

The architecture separates two sinks for script output:
- The researcher's TUI, which sees everything (``raw_stdout``,
  ``raw_stderr``, the scratch dir on disk).
- The model, which sees only sanitized structured payloads on success
  and a tightly-bounded ``debug_excerpt`` on failure.

This test codifies the split for the SUCCESS path. A failure path now
forwards a 500-1000 char ``debug_excerpt`` of the language's own error
output. That channel has its own SDC boundary tests in
``test_error_summary_no_leak.py``. Here we just pin that on a clean
run, nothing extra crosses: stdout / stderr printed by the script
(including data prints, debug `cat()`, etc.) must not appear in the
tool response.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from nora.config import set_cwd
from nora.env_detect import detect_environment
from nora.tools import submit_script


# A token nothing should reasonably produce organically, so its presence
# in Claude-visible output would be a hard failure.
_INJECTION_CANARY = "CANARY_LEAK_MARKER_b2a7f3e8"


@pytest.mark.skipif(
    detect_environment().r is None,
    reason="R not installed; executor can't be exercised",
)
def test_stderr_never_leaks_to_tool_response(tmp_path: Path):
    set_cwd(tmp_path)
    # Script prints the canary to stdout AND stderr, then emits a valid
    # regression result so the executor path runs to completion.
    code = f"""
cat("{_INJECTION_CANARY} STDOUT\\n")
message("{_INJECTION_CANARY} STDERR")
set.seed(1)
x <- rnorm(50); y <- rnorm(50)
m <- lm(y ~ x)
nora$from_lm(m)
"""
    # @tool-decorated functions are wrapped in SdkMcpTool; the underlying
    # async function lives on `.handler`.
    response = asyncio.run(
        submit_script.handler({"language": "R", "code": code, "label": "canary test"})
    )
    # submit_script returns an MCP-content envelope; serialize the whole
    # thing and check the canary is not anywhere in Claude's view.
    blob = json.dumps(response)
    assert _INJECTION_CANARY not in blob, (
        "raw stdout/stderr leaked into the tool response — the token "
        "that was only printed to the subprocess stdout/stderr is "
        "visible in what Claude receives. This breaks the data "
        "boundary."
    )
