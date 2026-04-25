"""Regression test: raw stderr / stdout never reaches Claude.

The architecture intentionally separates two sinks for script output:
- The researcher's TUI, which sees everything (``raw_stdout``,
  ``raw_stderr``, the scratch dir on disk).
- Claude, which sees only sanitized structured payloads.

This test codifies that split at the ``submit_script`` tool layer — if
a future refactor accidentally threads raw subprocess output into the
MCP tool response, the resulting Claude-visible field becomes an
injection channel (R / Stata errors can echo data content, e.g.
``"variable contains invalid UTF-8 near <malicious>"``).

The test runs a script that deliberately prints a recognizable token
to stdout AND stderr, then asserts that token does not appear anywhere
in the string representation of the tool's response.
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
