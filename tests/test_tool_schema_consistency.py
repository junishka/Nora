"""Guard against drift between the canonical tool schemas and the
SDK-side @tool decorations in ``nora.tools``.

Both providers depend on the canonical specs in
``nora.provider.tool_schemas`` for tool name + description + input
shape. The Anthropic path also feeds the @tool decorator literals to
the Claude Agent SDK. If those two diverge, the model sees different
guidance depending on which provider is active — a subtle correctness
bug.

This test enforces:
  1. Same set of tool names in both sources.
  2. Identical descriptions (so model behaviour is provider-agnostic).
  3. Same set of parameter names per tool.
  4. ``HANDLERS`` covers every tool exactly once and points to the
     same callable as the SDK-registered ``handler`` attribute.
"""

from __future__ import annotations

from nora.provider.tool_schemas import build_tool_specs
from nora.tools import HANDLERS, REGISTERED_TOOLS


def test_tool_names_match():
    sdk_names = {t.name for t in REGISTERED_TOOLS}
    spec_names = {s.name for s in build_tool_specs()}
    assert sdk_names == spec_names, (
        f"SDK / spec tool-name drift: SDK has {sdk_names}, "
        f"specs have {spec_names}"
    )


def test_tool_descriptions_match():
    sdk_by_name = {t.name: t for t in REGISTERED_TOOLS}
    for spec in build_tool_specs():
        sdk = sdk_by_name[spec.name]
        assert sdk.description == spec.description, (
            f"description drift for {spec.name!r}:\n"
            f"  SDK: {sdk.description!r}\n  spec: {spec.description!r}"
        )


def test_tool_parameter_names_match():
    """SDK stores ``input_schema`` as ``{param_name: python_type}``;
    the canonical spec is full JSON-Schema. Compare property names
    only (types are checked separately by ``ToolSpec.as_sdk_args``)."""
    sdk_by_name = {t.name: t for t in REGISTERED_TOOLS}
    for spec in build_tool_specs():
        sdk = sdk_by_name[spec.name]
        sdk_props = set((sdk.input_schema or {}).keys())
        spec_props = set(spec.input_schema.get("properties", {}).keys())
        assert sdk_props == spec_props, (
            f"parameter-name drift for {spec.name!r}: "
            f"SDK has {sdk_props}, spec has {spec_props}"
        )


def test_tool_parameter_types_match():
    """Each spec's ``as_sdk_args()`` should reproduce the SDK's
    ``{name: type}`` mapping exactly."""
    sdk_by_name = {t.name: t for t in REGISTERED_TOOLS}
    for spec in build_tool_specs():
        sdk = sdk_by_name[spec.name]
        sdk_args = sdk.input_schema or {}
        spec_args = spec.as_sdk_args()
        assert sdk_args == spec_args, (
            f"parameter-type drift for {spec.name!r}: "
            f"SDK has {sdk_args}, spec produces {spec_args}"
        )


def test_handlers_cover_every_tool():
    spec_names = {s.name for s in build_tool_specs()}
    handler_names = set(HANDLERS.keys())
    assert handler_names == spec_names, (
        f"HANDLERS / spec mismatch: handlers={handler_names}, "
        f"specs={spec_names}"
    )


def test_handlers_point_at_sdk_handlers():
    sdk_by_name = {t.name: t for t in REGISTERED_TOOLS}
    for name, fn in HANDLERS.items():
        assert fn is sdk_by_name[name].handler, (
            f"HANDLERS[{name!r}] is not the same callable as "
            f"REGISTERED_TOOLS[{name!r}].handler — dispatch will diverge"
        )
