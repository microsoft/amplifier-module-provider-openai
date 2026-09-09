"""Regression coverage for Responses function-tool strictness conversion."""

from __future__ import annotations

from copy import deepcopy
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from amplifier_core.message_models import ToolSpec

from amplifier_module_provider_openai import OpenAIProvider

OPTIONAL_ENUM_SCHEMA = {
    "type": "object",
    "properties": {
        "operation": {
            "type": "string",
            "enum": ["review", "accept", "skip"],
        },
        "comment": {"type": "string"},
    },
    "required": ["operation"],
    "additionalProperties": False,
}


def _make_provider(**config_overrides: Any) -> OpenAIProvider:
    coordinator = MagicMock()
    coordinator.get_capability = MagicMock(return_value=None)
    coordinator.hooks.emit = AsyncMock()
    return OpenAIProvider(
        api_key="[REDACTED:SECRET]",
        config={"max_retries": 0, **config_overrides},
        coordinator=coordinator,
    )


def _tool_spec(**extra: Any) -> ToolSpec:
    return ToolSpec(
        name="review_memory",
        description="Review a memory",
        parameters=deepcopy(OPTIONAL_ENUM_SCHEMA),
        **extra,
    )


def test_toolspec_without_strict_defaults_to_false_without_schema_mutation() -> None:
    provider = _make_provider()
    tool = _tool_spec()
    original_schema = deepcopy(tool.parameters)

    converted = provider._convert_tools_from_request([tool])

    assert converted[0]["strict"] is False
    assert converted[0]["parameters"] == original_schema
    assert tool.parameters == original_schema


@pytest.mark.parametrize("strict", [True, False])
def test_toolspec_explicit_boolean_strict_is_preserved(strict: bool) -> None:
    provider = _make_provider()
    tool = _tool_spec(strict=strict)

    converted = provider._convert_tools_from_request([tool])

    assert converted[0]["strict"] is strict


@pytest.mark.parametrize(
    ("provided_strict", "expected_strict"),
    [(True, True), (False, False), (None, False)],
)
def test_function_tool_dict_strictness_matches_toolspec(
    provided_strict: bool | None, expected_strict: bool
) -> None:
    provider = _make_provider()
    tool = {
        "type": "function",
        "name": "review_memory",
        "description": "Review a memory",
        "parameters": deepcopy(OPTIONAL_ENUM_SCHEMA),
    }
    if provided_strict is not None:
        tool["strict"] = provided_strict
    original_tool = deepcopy(tool)

    converted = provider._convert_tools_from_request([tool])

    assert converted[0]["strict"] is expected_strict
    assert converted[0]["parameters"] == OPTIONAL_ENUM_SCHEMA
    assert tool == original_tool


def test_namespaced_function_tool_preserves_explicit_strict() -> None:
    provider = _make_provider(tool_search={"mode": "namespaced"})
    tool = ToolSpec(
        name="read_file",
        description="Read a file",
        parameters=deepcopy(OPTIONAL_ENUM_SCHEMA),
        strict=True,
    )

    converted = provider._convert_tools_from_request([tool])

    files = next(item for item in converted if item.get("name") == "files")
    assert files["tools"][0]["strict"] is True


def test_native_tool_dict_is_untouched() -> None:
    provider = _make_provider()
    tool = {"type": "web_search", "search_context_size": "low"}
    original_tool = deepcopy(tool)

    converted = provider._convert_tools_from_request([tool])

    assert converted == [original_tool]
    assert converted[0] is tool
    assert tool == original_tool