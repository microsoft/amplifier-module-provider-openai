"""Tests for apply_patch native tool support in the OpenAI provider.

Tests the three localized changes:
1. _convert_tools_from_request: sends {"type": "apply_patch"} for native engine
2. _convert_to_chat_response: parses apply_patch_call blocks into ToolCalls
3. _convert_messages: emits apply_patch_call_output for native tool results
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from amplifier_module_provider_openai import OpenAIProvider
from amplifier_module_provider_openai._constants import NATIVE_TOOL_TYPES


# --- Fixtures ---


def _make_provider(**overrides: Any) -> OpenAIProvider:
    """Create a minimal provider instance for unit testing."""
    coordinator = MagicMock()
    coordinator.get_capability = MagicMock(return_value=None)
    coordinator.hooks = MagicMock()
    config = {**overrides}
    provider = OpenAIProvider(api_key="test-key", config=config, coordinator=coordinator)
    return provider


def _make_apply_patch_tool_spec() -> MagicMock:
    """Create a mock ToolSpec for apply_patch."""
    spec = MagicMock()
    spec.name = "apply_patch"
    spec.description = "Apply V4A patches"
    spec.parameters = {
        "type": "object",
        "properties": {"patch": {"type": "string"}},
        "required": ["patch"],
    }
    return spec


# --- Test NATIVE_TOOL_TYPES includes apply_patch ---


class TestNativeToolTypes:
    def test_apply_patch_in_native_tool_types(self) -> None:
        assert "apply_patch" in NATIVE_TOOL_TYPES


# --- Test _convert_tools_from_request ---


class TestConvertToolsFromRequest:
    def test_native_apply_patch_sends_type_only(self) -> None:
        """When apply_patch.engine is native, send {"type": "apply_patch"}."""
        provider = _make_provider()
        provider._apply_patch_native = True

        tool_spec = _make_apply_patch_tool_spec()
        result = provider._convert_tools_from_request([tool_spec])

        # Should produce {"type": "apply_patch"} — not a function tool
        native_tools = [t for t in result if t.get("type") == "apply_patch"]
        assert len(native_tools) == 1
        assert native_tools[0] == {"type": "apply_patch"}

    def test_function_apply_patch_sends_function_tool(self) -> None:
        """When apply_patch.engine is function, send as normal function tool."""
        provider = _make_provider()
        provider._apply_patch_native = False

        tool_spec = _make_apply_patch_tool_spec()
        result = provider._convert_tools_from_request([tool_spec])

        # Should produce a function tool, not native
        func_tools = [t for t in result if t.get("type") == "function"]
        assert len(func_tools) == 1
        assert func_tools[0]["name"] == "apply_patch"

    def test_other_tools_unaffected(self) -> None:
        """Non-apply_patch tools are converted normally."""
        provider = _make_provider()
        provider._apply_patch_native = True

        other_tool = MagicMock()
        other_tool.name = "read_file"
        other_tool.description = "Read files"
        other_tool.parameters = {"type": "object", "properties": {}}

        result = provider._convert_tools_from_request([other_tool])
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["name"] == "read_file"


# --- Test _convert_to_chat_response (apply_patch_call parsing) ---


class TestConvertResponseApplyPatchCall:
    def test_parses_apply_patch_call_block(self) -> None:
        """apply_patch_call blocks should become ToolCall(name='apply_patch')."""
        provider = _make_provider()
        provider._native_call_ids = set()

        # Mock an API response with an apply_patch_call block
        mock_block = MagicMock()
        mock_block.type = "apply_patch_call"
        mock_block.call_id = "call_abc123"
        mock_operation = MagicMock()
        mock_operation.type = "update_file"
        mock_operation.path = "src/main.py"
        mock_operation.diff = "@@ def main():\n-old\n+new"
        mock_block.operation = mock_operation

        mock_response = MagicMock()
        mock_response.output = [mock_block]
        mock_response.usage = MagicMock()
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50
        mock_response.usage.total_tokens = 150
        mock_response.usage.output_tokens_details = None
        mock_response.usage.input_tokens_details = None
        mock_response.id = "resp_123"
        mock_response.status = "completed"
        mock_response.incomplete_details = None

        chat_response = provider._convert_to_chat_response(mock_response)

        assert chat_response.tool_calls is not None
        assert len(chat_response.tool_calls) == 1
        tc = chat_response.tool_calls[0]
        assert tc.id == "call_abc123"
        assert tc.name == "apply_patch"
        assert tc.arguments["type"] == "update_file"
        assert tc.arguments["path"] == "src/main.py"
        assert tc.arguments["diff"] == "@@ def main():\n-old\n+new"

    def test_tracks_native_call_ids(self) -> None:
        """apply_patch_call call_ids should be tracked for round-trip output."""
        provider = _make_provider()
        provider._native_call_ids = set()

        mock_block = MagicMock()
        mock_block.type = "apply_patch_call"
        mock_block.call_id = "call_xyz789"
        mock_operation = MagicMock()
        mock_operation.type = "create_file"
        mock_operation.path = "new.py"
        mock_operation.diff = "+content"
        mock_block.operation = mock_operation

        mock_response = MagicMock()
        mock_response.output = [mock_block]
        mock_response.usage = MagicMock()
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5
        mock_response.usage.total_tokens = 15
        mock_response.usage.output_tokens_details = None
        mock_response.usage.input_tokens_details = None
        mock_response.id = "resp_456"
        mock_response.status = "completed"
        mock_response.incomplete_details = None

        provider._convert_to_chat_response(mock_response)
        assert "call_xyz789" in provider._native_call_ids


# --- Test _convert_messages (apply_patch_call_output) ---


class TestConvertMessagesApplyPatchOutput:
    def test_native_tool_result_uses_apply_patch_call_output(self) -> None:
        """Tool results for native call_ids should use apply_patch_call_output type."""
        provider = _make_provider()
        provider._native_call_ids = {"call_abc123"}

        messages = [
            {
                "role": "tool",
                "tool_call_id": "call_abc123",
                "content": "M src/main.py",
                "tool_name": "apply_patch",
            }
        ]

        result = provider._convert_messages(messages)

        # Find the apply_patch_call_output item
        patch_outputs = [
            m
            for m in result
            if isinstance(m, dict) and m.get("type") == "apply_patch_call_output"
        ]
        assert len(patch_outputs) == 1
        assert patch_outputs[0]["call_id"] == "call_abc123"
        assert patch_outputs[0]["output"] == "M src/main.py"

    def test_regular_tool_result_uses_function_call_output(self) -> None:
        """Non-native tool results still use function_call_output type."""
        provider = _make_provider()
        provider._native_call_ids = {"call_abc123"}  # only this is native

        messages = [
            {
                "role": "tool",
                "tool_call_id": "call_other",
                "content": "file content here",
                "tool_name": "read_file",
            }
        ]

        result = provider._convert_messages(messages)

        func_outputs = [
            m
            for m in result
            if isinstance(m, dict) and m.get("type") == "function_call_output"
        ]
        assert len(func_outputs) == 1
        assert func_outputs[0]["call_id"] == "call_other"
