"""Tests for apply_patch native tool support in the OpenAI provider.

Tests the three localized changes:
1. _convert_tools_from_request: sends {"type": "apply_patch"} for native engine
2. _convert_to_chat_response: parses apply_patch_call blocks into ToolCalls
3. _convert_messages: emits apply_patch_call_output for native tool results
"""

# pyright: reportAttributeAccessIssue=false

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
    provider = OpenAIProvider(
        api_key="test-key", config=config, coordinator=coordinator
    )
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


class TestConvertToolsFromRequestModelGating:
    """Model-aware gating for native apply_patch.

    Regression coverage for the bug where the native ``{"type": "apply_patch"}``
    tool shape was sent to gpt-5-mini (which does not support it), causing:
    "Tool 'apply_patch' is not supported with gpt-5-mini".

    The gate must check BOTH the session-wide apply_patch.engine capability
    AND the specific model's ModelCapabilities.supports_native_apply_patch.
    """

    def test_native_apply_patch_falls_back_for_unsupported_model(self) -> None:
        """gpt-5-mini does not support native apply_patch — must fall back
        to the function-tool shape instead of erroring against the API."""
        provider = _make_provider(default_model="gpt-5-mini")
        provider._apply_patch_native = True

        tool_spec = _make_apply_patch_tool_spec()
        result = provider._convert_tools_from_request([tool_spec], "gpt-5-mini")

        native_tools = [t for t in result if t.get("type") == "apply_patch"]
        assert len(native_tools) == 0
        func_tools = [t for t in result if t.get("type") == "function"]
        assert len(func_tools) == 1
        assert func_tools[0]["name"] == "apply_patch"

    def test_native_apply_patch_sent_for_supporting_model(self) -> None:
        """gpt-5.1 is confirmed to support native apply_patch — the native
        shape must still be sent for models known to support it."""
        provider = _make_provider(default_model="gpt-5.1")
        provider._apply_patch_native = True

        tool_spec = _make_apply_patch_tool_spec()
        result = provider._convert_tools_from_request([tool_spec], "gpt-5.1")

        native_tools = [t for t in result if t.get("type") == "apply_patch"]
        assert len(native_tools) == 1
        assert native_tools[0] == {"type": "apply_patch"}

    def test_gpt_5_4_still_gets_native_shape(self) -> None:
        """gpt-5.4 (confirmed supported) is unaffected by the model gate."""
        provider = _make_provider(default_model="gpt-5.4")
        provider._apply_patch_native = True

        tool_spec = _make_apply_patch_tool_spec()
        result = provider._convert_tools_from_request([tool_spec], "gpt-5.4")

        native_tools = [t for t in result if t.get("type") == "apply_patch"]
        assert len(native_tools) == 1

    def test_falls_back_to_default_model_when_model_name_omitted(self) -> None:
        """When callers don't pass model_name explicitly, the provider's
        configured default_model is used to resolve capabilities."""
        provider = _make_provider(default_model="gpt-5-mini")
        provider._apply_patch_native = True

        tool_spec = _make_apply_patch_tool_spec()
        result = provider._convert_tools_from_request([tool_spec])

        native_tools = [t for t in result if t.get("type") == "apply_patch"]
        assert len(native_tools) == 0
        func_tools = [t for t in result if t.get("type") == "function"]
        assert len(func_tools) == 1

    # --- Default-True-with-named-exclusions regression coverage ---
    #
    # Empirical basis: live calls to the Responses API disproved a naive
    # "exclude all -mini/-nano" rule. The dividing line is VERSION, not SIZE:
    # every gpt-5.1+ variant supports native apply_patch (mini/nano included),
    # while the original gpt-5.0 generation (bare gpt-5, gpt-5-mini, gpt-5-nano,
    # gpt-5-codex, gpt-5-pro, gpt-5-chat-latest) does not. Per explicit product
    # direction, the philosophy is now DEFAULT TRUE / exclude only proven
    # failures: "-chat-latest" aliases are NOT excluded as a blanket class
    # anymore — only the single alias proven to fail (gpt-5.1-chat-latest) is
    # a named exception. gpt-5.3-chat-latest, previously a known false
    # negative, now correctly resolves True.

    def test_gpt_5_4_mini_supports_native_apply_patch(self) -> None:
        """gpt-5.4-mini is confirmed supported despite being a 'mini' size —
        the version-gated rule (minor >= 1) overrides model size."""
        provider = _make_provider(default_model="gpt-5.4-mini")
        provider._apply_patch_native = True

        tool_spec = _make_apply_patch_tool_spec()
        result = provider._convert_tools_from_request([tool_spec], "gpt-5.4-mini")

        native_tools = [t for t in result if t.get("type") == "apply_patch"]
        assert len(native_tools) == 1

    def test_gpt_5_4_nano_supports_native_apply_patch(self) -> None:
        """gpt-5.4-nano is confirmed supported — 'nano' size is also not an
        exclusion signal once minor >= 1."""
        provider = _make_provider(default_model="gpt-5.4-nano")
        provider._apply_patch_native = True

        tool_spec = _make_apply_patch_tool_spec()
        result = provider._convert_tools_from_request([tool_spec], "gpt-5.4-nano")

        native_tools = [t for t in result if t.get("type") == "apply_patch"]
        assert len(native_tools) == 1

    def test_gpt_5_nano_bare_does_not_support_native_apply_patch(self) -> None:
        """gpt-5-nano (gpt-5.0 generation, minor==0) is confirmed NOT supported."""
        provider = _make_provider(default_model="gpt-5-nano")
        provider._apply_patch_native = True

        tool_spec = _make_apply_patch_tool_spec()
        result = provider._convert_tools_from_request([tool_spec], "gpt-5-nano")

        native_tools = [t for t in result if t.get("type") == "apply_patch"]
        assert len(native_tools) == 0
        func_tools = [t for t in result if t.get("type") == "function"]
        assert len(func_tools) == 1

    def test_gpt_5_codex_bare_does_not_support_native_apply_patch(self) -> None:
        """gpt-5-codex (gpt-5.0 generation) is confirmed NOT supported."""
        provider = _make_provider(default_model="gpt-5-codex")
        provider._apply_patch_native = True

        tool_spec = _make_apply_patch_tool_spec()
        result = provider._convert_tools_from_request([tool_spec], "gpt-5-codex")

        native_tools = [t for t in result if t.get("type") == "apply_patch"]
        assert len(native_tools) == 0
        func_tools = [t for t in result if t.get("type") == "function"]
        assert len(func_tools) == 1

    def test_gpt_5_1_codex_max_supports_native_apply_patch(self) -> None:
        """gpt-5.1-codex-max is confirmed supported (minor==1)."""
        provider = _make_provider(default_model="gpt-5.1-codex-max")
        provider._apply_patch_native = True

        tool_spec = _make_apply_patch_tool_spec()
        result = provider._convert_tools_from_request([tool_spec], "gpt-5.1-codex-max")

        native_tools = [t for t in result if t.get("type") == "apply_patch"]
        assert len(native_tools) == 1

    def test_gpt_5_1_chat_latest_does_not_support_native_apply_patch(self) -> None:
        """gpt-5.1-chat-latest is confirmed NOT supported despite minor==1 —
        the blanket "-chat-latest" exclusion overrides the version rule."""
        provider = _make_provider(default_model="gpt-5.1-chat-latest")
        provider._apply_patch_native = True

        tool_spec = _make_apply_patch_tool_spec()
        result = provider._convert_tools_from_request(
            [tool_spec], "gpt-5.1-chat-latest"
        )

        native_tools = [t for t in result if t.get("type") == "apply_patch"]
        assert len(native_tools) == 0
        func_tools = [t for t in result if t.get("type") == "function"]
        assert len(func_tools) == 1

    def test_o3_mini_does_not_support_native_apply_patch(self) -> None:
        """o3-mini (o-series) is confirmed NOT supported."""
        provider = _make_provider(default_model="o3-mini")
        provider._apply_patch_native = True

        tool_spec = _make_apply_patch_tool_spec()
        result = provider._convert_tools_from_request([tool_spec], "o3-mini")

        native_tools = [t for t in result if t.get("type") == "apply_patch"]
        assert len(native_tools) == 0
        func_tools = [t for t in result if t.get("type") == "function"]
        assert len(func_tools) == 1


# --- Direct unit tests on get_capabilities() version-gated apply_patch rule ---


class TestGetCapabilitiesApplyPatchDefaultTrueRule:
    """Direct tests on `_capabilities.get_capabilities()` — not routed through
    the provider's tool-conversion wrapper. Covers the DEFAULT-TRUE-WITH-
    NAMED-EXCLUSIONS rule (superseding the earlier version-gated-only rule),
    empirically derived from live Responses API calls (2026-07).

    Philosophy: default to True (assume support); only exclude models/
    families we have direct empirical proof reject the native tool. This
    means the gpt-5.0 generation, o-series, deep-research, gpt-5.1-chat-latest
    (named exception), and gpt-4.x (named prefix carve-out) are the ONLY
    exclusions — everything else, including future/unrecognized model ids,
    defaults True.
    """

    def test_gpt_5_4_mini_supported(self) -> None:
        from amplifier_module_provider_openai._capabilities import get_capabilities

        assert get_capabilities("gpt-5.4-mini").supports_native_apply_patch is True

    def test_gpt_5_4_nano_supported(self) -> None:
        from amplifier_module_provider_openai._capabilities import get_capabilities

        assert get_capabilities("gpt-5.4-nano").supports_native_apply_patch is True

    def test_gpt_5_mini_bare_not_supported(self) -> None:
        from amplifier_module_provider_openai._capabilities import get_capabilities

        assert get_capabilities("gpt-5-mini").supports_native_apply_patch is False

    def test_gpt_5_nano_bare_not_supported(self) -> None:
        from amplifier_module_provider_openai._capabilities import get_capabilities

        assert get_capabilities("gpt-5-nano").supports_native_apply_patch is False

    def test_gpt_5_codex_bare_not_supported(self) -> None:
        from amplifier_module_provider_openai._capabilities import get_capabilities

        assert get_capabilities("gpt-5-codex").supports_native_apply_patch is False

    def test_gpt_5_pro_bare_not_supported(self) -> None:
        from amplifier_module_provider_openai._capabilities import get_capabilities

        assert get_capabilities("gpt-5-pro").supports_native_apply_patch is False

    def test_gpt_5_bare_not_supported(self) -> None:
        from amplifier_module_provider_openai._capabilities import get_capabilities

        assert get_capabilities("gpt-5").supports_native_apply_patch is False

    def test_gpt_5_1_codex_max_supported(self) -> None:
        from amplifier_module_provider_openai._capabilities import get_capabilities

        assert get_capabilities("gpt-5.1-codex-max").supports_native_apply_patch is True

    def test_gpt_5_1_codex_mini_supported(self) -> None:
        from amplifier_module_provider_openai._capabilities import get_capabilities

        assert (
            get_capabilities("gpt-5.1-codex-mini").supports_native_apply_patch is True
        )

    def test_gpt_5_1_chat_latest_not_supported(self) -> None:
        from amplifier_module_provider_openai._capabilities import get_capabilities

        assert (
            get_capabilities("gpt-5.1-chat-latest").supports_native_apply_patch is False
        )

    def test_gpt_5_3_chat_latest_supported_no_longer_excluded(self) -> None:
        """Accuracy improvement over the old rule: gpt-5.3-chat-latest is
        empirically SUPPORTED. The old blanket "-chat-latest" exclusion
        produced a false negative here; the new default-True-with-named-
        exclusions rule correctly resolves this to True since it is not the
        one named exception (gpt-5.1-chat-latest)."""
        from amplifier_module_provider_openai._capabilities import get_capabilities

        assert (
            get_capabilities("gpt-5.3-chat-latest").supports_native_apply_patch is True
        )

    def test_gpt_5_chat_latest_bare_not_supported(self) -> None:
        """gpt-5-chat-latest (gpt-5.0 generation, minor==0) is confirmed
        NOT supported — covered by the minor==0 rule, not the named
        gpt-5.1-chat-latest exception."""
        from amplifier_module_provider_openai._capabilities import get_capabilities

        assert (
            get_capabilities("gpt-5-chat-latest").supports_native_apply_patch is False
        )

    def test_gpt_5_5_supported(self) -> None:
        from amplifier_module_provider_openai._capabilities import get_capabilities

        assert get_capabilities("gpt-5.5").supports_native_apply_patch is True

    def test_o3_mini_not_supported(self) -> None:
        from amplifier_module_provider_openai._capabilities import get_capabilities

        assert get_capabilities("o3-mini").supports_native_apply_patch is False

    def test_o4_mini_not_supported(self) -> None:
        from amplifier_module_provider_openai._capabilities import get_capabilities

        assert get_capabilities("o4-mini").supports_native_apply_patch is False

    def test_deep_research_not_supported(self) -> None:
        from amplifier_module_provider_openai._capabilities import get_capabilities

        assert get_capabilities("o3-deep-research").supports_native_apply_patch is False

    # --- gpt-4.x carve-out: confirmed NOT supported, must not silently ---
    # --- flip True when the "unknown" bucket's default changes to True. ---

    def test_gpt_4_1_not_supported(self) -> None:
        """gpt-4.1 is confirmed NOT supported — the gpt-4.x carve-out must
        override the new "unknown family defaults True" behavior."""
        from amplifier_module_provider_openai._capabilities import get_capabilities

        assert get_capabilities("gpt-4.1").supports_native_apply_patch is False

    def test_gpt_4o_not_supported(self) -> None:
        """gpt-4o is confirmed NOT supported — same carve-out as gpt-4.1."""
        from amplifier_module_provider_openai._capabilities import get_capabilities

        assert get_capabilities("gpt-4o").supports_native_apply_patch is False

    def test_gpt_4_1_mini_not_supported(self) -> None:
        """gpt-4.1-mini is confirmed NOT supported."""
        from amplifier_module_provider_openai._capabilities import get_capabilities

        assert get_capabilities("gpt-4.1-mini").supports_native_apply_patch is False

    def test_gpt_4o_mini_not_supported(self) -> None:
        """gpt-4o-mini is confirmed NOT supported."""
        from amplifier_module_provider_openai._capabilities import get_capabilities

        assert get_capabilities("gpt-4o-mini").supports_native_apply_patch is False

    def test_gpt_4_x_family_stays_unknown(self) -> None:
        """The gpt-4.x carve-out must be a narrow model_id override, NOT a
        new family branch — family stays "unknown" so every other capability
        default (context_window, reasoning, etc.) is untouched."""
        from amplifier_module_provider_openai._capabilities import get_capabilities

        caps = get_capabilities("gpt-4.1")
        assert caps.family == "unknown"
        assert caps.supports_native_apply_patch is False
        # Other fields must match the plain "unknown" bucket defaults.
        assert caps.context_window == 200_000
        assert caps.max_output_tokens == 128_000
        assert caps.supports_reasoning is False

    # --- "unknown" bucket now defaults True (loud-failure-accepted trade-off) ---

    def test_novel_unrecognized_model_defaults_true(self) -> None:
        """A genuinely novel/untested model id (not matching any known
        family or the gpt-4.x prefix) now defaults True — the accepted
        trade-off per explicit product direction: default-optimistic, find
        out via a loud API rejection if a future model doesn't support it."""
        from amplifier_module_provider_openai._capabilities import get_capabilities

        caps = get_capabilities("gpt-6-hypothetical")
        assert caps.family == "unknown"
        assert caps.supports_native_apply_patch is True

    def test_default_dataclass_supports_native_apply_patch_is_true(self) -> None:
        """The bare dataclass default (no family-specific override at all)
        is True — the new baseline philosophy."""
        from amplifier_module_provider_openai._capabilities import ModelCapabilities

        assert ModelCapabilities(family="test").supports_native_apply_patch is True


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
        mock_response.model = "gpt-5.5"
        mock_response.output = [mock_block]
        mock_response.usage = MagicMock()
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50
        mock_response.usage.total_tokens = 150
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_response.usage.output_tokens_details = None
        mock_response.usage.input_tokens_details = None
        mock_response.id = "resp_123"
        mock_response.status = "completed"
        mock_response.incomplete_details = None
        mock_response.finish_reason = None
        mock_response.output_text = None

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
        mock_response.model = "gpt-5.5"
        mock_response.output = [mock_block]
        mock_response.usage = MagicMock()
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5
        mock_response.usage.total_tokens = 15
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.output_tokens_details = None
        mock_response.usage.input_tokens_details = None
        mock_response.id = "resp_456"
        mock_response.status = "completed"
        mock_response.incomplete_details = None
        mock_response.finish_reason = None
        mock_response.output_text = None

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

    def test_apply_patch_call_output_includes_status_completed(self) -> None:
        """apply_patch_call_output must include status field for OpenAI API."""
        provider = _make_provider()
        provider._native_call_ids = {"call_abc123"}

        messages = [
            {
                "role": "tool",
                "tool_call_id": "call_abc123",
                "content": "A new-file.txt",
                "tool_name": "apply_patch",
            },
        ]

        result = provider._convert_messages(messages)

        output_items = [
            m
            for m in result
            if isinstance(m, dict) and m.get("type") == "apply_patch_call_output"
        ]
        assert len(output_items) == 1
        assert output_items[0]["status"] == "completed"

    def test_apply_patch_call_output_status_failed_on_error(self) -> None:
        """apply_patch_call_output should have status 'failed' when tool errored."""
        provider = _make_provider()
        provider._native_call_ids = {"call_err456"}

        messages = [
            {
                "role": "tool",
                "tool_call_id": "call_err456",
                "content": '{"success": false, "error": {"message": "File not found"}}',
                "tool_name": "apply_patch",
            },
        ]

        result = provider._convert_messages(messages)

        output_items = [
            m
            for m in result
            if isinstance(m, dict) and m.get("type") == "apply_patch_call_output"
        ]
        assert len(output_items) == 1
        assert output_items[0]["status"] == "failed"

    def test_apply_patch_call_output_status_failed_on_plain_string_error(self) -> None:
        """Plain-string errors (non-JSON) should also get status 'failed'.

        The apply_patch tool returns plain strings like 'File already exists: ...'
        or 'Context mismatch in ...'. These are not JSON-parseable, but they are
        errors and must produce status 'failed' — not 'completed'.
        """
        provider = _make_provider()
        provider._native_call_ids = {"call_plain_err"}

        messages = [
            {
                "role": "tool",
                "tool_call_id": "call_plain_err",
                "content": (
                    "File already exists: test.txt. Use update_file to modify "
                    "existing files.\n\nCurrent content of test.txt (1 lines):\n"
                    "1\tHello world"
                ),
                "tool_name": "apply_patch",
            },
        ]

        result = provider._convert_messages(messages)

        output_items = [
            m
            for m in result
            if isinstance(m, dict) and m.get("type") == "apply_patch_call_output"
        ]
        assert len(output_items) == 1
        assert output_items[0]["status"] == "failed"

    def test_apply_patch_call_output_status_failed_on_context_mismatch_string(
        self,
    ) -> None:
        """Context mismatch plain-string errors should get status 'failed'."""
        provider = _make_provider()
        provider._native_call_ids = {"call_ctx_err"}

        messages = [
            {
                "role": "tool",
                "tool_call_id": "call_ctx_err",
                "content": (
                    "Context mismatch in main.py: Invalid Context 0:\n"
                    "old line — your diff does not match the current file.\n\n"
                    "Current content of main.py (3 lines):\n"
                    "1\tline one\n2\tline two\n3\tline three\n\n"
                    "Construct a new diff based on the content above."
                ),
                "tool_name": "apply_patch",
            },
        ]

        result = provider._convert_messages(messages)

        output_items = [
            m
            for m in result
            if isinstance(m, dict) and m.get("type") == "apply_patch_call_output"
        ]
        assert len(output_items) == 1
        assert output_items[0]["status"] == "failed"

    def test_apply_patch_call_output_status_completed_on_success_string(self) -> None:
        """Git-style success strings should still get status 'completed'."""
        provider = _make_provider()
        provider._native_call_ids = {"call_ok1", "call_ok2", "call_ok3"}

        for call_id, content in [
            ("call_ok1", "M src/main.py"),
            ("call_ok2", "A new-file.txt"),
            ("call_ok3", "D old-file.py"),
        ]:
            messages = [
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": content,
                    "tool_name": "apply_patch",
                },
            ]
            result = provider._convert_messages(messages)
            output_items = [
                m
                for m in result
                if isinstance(m, dict) and m.get("type") == "apply_patch_call_output"
            ]
            assert len(output_items) == 1, f"Failed for {content}"
            assert output_items[0]["status"] == "completed", (
                f"Expected 'completed' for success output '{content}', "
                f"got '{output_items[0]['status']}'"
            )

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


# --- Test the wire-path pairing backstop counts native result envelopes ---
#
# _convert_messages ends with a last-resort backstop: any function_call item
# replayed into the input with no paired output gets a synthesized "[error]
# result missing" output, so the request cannot 400 on an orphan. That check
# must recognize the SAME set of result envelopes as
# _enforce_chain_output_pairing (_PAIRED_OUTPUT_ITEM_TYPES). Counting only
# function_call_output declares a real, successful native result missing and
# ships a fabricated error beside it -- two contradictory results for one
# call_id, inviting the model to re-run an already-applied patch.


class TestWirePathPairingCountsNativeOutputs:
    def test_native_output_satisfies_wire_path_pairing(self) -> None:
        """A real apply_patch_call_output pairs its call; nothing is synthesized.

        Uses the shape where the assistant turn replays as a function_call
        (the stored operation type is not one the history-replay shape
        detector recognizes) while the result still replays as a native
        apply_patch_call_output, because the call_id is in _native_call_ids.
        That divergence is what exposes the backstop's blind spot.
        """
        provider = _make_provider()
        provider._native_call_ids = {"call_np1"}

        messages = [
            {"role": "user", "content": "patch it"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_call",
                        "id": "call_np1",
                        "name": "apply_patch",
                        "input": {"type": "move_file", "path": "a.py", "diff": ""},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_np1",
                "content": "M a.py",
                "tool_name": "apply_patch",
            },
        ]

        result = provider._convert_messages(messages)

        patch_outputs = [
            m
            for m in result
            if isinstance(m, dict) and m.get("type") == "apply_patch_call_output"
        ]
        assert len(patch_outputs) == 1, "the real native result must survive"
        assert patch_outputs[0]["output"] == "M a.py"

        synthesized = [
            m
            for m in result
            if isinstance(m, dict)
            and m.get("type") == "function_call_output"
            and "[error]" in str(m.get("output", ""))
        ]
        assert not synthesized, (
            "synthesized a 'result missing' error for a call that already had a "
            "real, successful native result -- the model would see two "
            "contradictory results for one call_id"
        )

    def test_orphaned_function_call_still_repaired(self) -> None:
        """The backstop still fires when an output is genuinely absent."""
        provider = _make_provider()

        messages = [
            {"role": "user", "content": "read it"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_call",
                        "id": "call_orphan",
                        "name": "read_file",
                        "input": {"path": "a.py"},
                    }
                ],
            },
        ]

        result = provider._convert_messages(messages)

        synthesized = [
            m
            for m in result
            if isinstance(m, dict)
            and m.get("type") == "function_call_output"
            and "[error]" in str(m.get("output", ""))
        ]
        assert len(synthesized) == 1
        assert synthesized[0]["call_id"] == "call_orphan"


# --- Test capability-based activation ---


class TestCapabilityDetection:
    """Provider should auto-detect native apply_patch engine via coordinator capability."""

    def test_detects_native_engine_from_capability(self) -> None:
        """When apply_patch.engine capability is 'native', provider activates native mode."""
        provider = _make_provider()
        assert provider._apply_patch_native is False  # starts False

        # Configure coordinator to return "native" for the capability query
        provider.coordinator.get_capability = MagicMock(
            side_effect=lambda key: "native" if key == "apply_patch.engine" else None
        )

        tool_spec = _make_apply_patch_tool_spec()
        result = provider._convert_tools_from_request([tool_spec])

        # Should have auto-detected and activated native mode
        assert provider._apply_patch_native is True
        # Should send native tool type, not function tool
        native_tools = [t for t in result if t.get("type") == "apply_patch"]
        assert len(native_tools) == 1
        assert native_tools[0] == {"type": "apply_patch"}

    def test_no_capability_stays_function_mode(self) -> None:
        """When apply_patch.engine capability is absent, provider stays in function mode."""
        provider = _make_provider()
        provider.coordinator.get_capability = MagicMock(return_value=None)

        tool_spec = _make_apply_patch_tool_spec()
        result = provider._convert_tools_from_request([tool_spec])

        assert provider._apply_patch_native is False
        func_tools = [t for t in result if t.get("type") == "function"]
        assert len(func_tools) == 1
        assert func_tools[0]["name"] == "apply_patch"

    def test_function_engine_capability_stays_function_mode(self) -> None:
        """When apply_patch.engine capability is 'function', provider stays in function mode."""
        provider = _make_provider()
        provider.coordinator.get_capability = MagicMock(
            side_effect=lambda key: "function" if key == "apply_patch.engine" else None
        )

        tool_spec = _make_apply_patch_tool_spec()
        result = provider._convert_tools_from_request([tool_spec])

        assert provider._apply_patch_native is False
        func_tools = [t for t in result if t.get("type") == "function"]
        assert len(func_tools) == 1

    def test_detection_is_lazy_and_cached(self) -> None:
        """Once native mode is detected, subsequent calls don't re-query the capability."""
        provider = _make_provider()
        call_count = 0

        def mock_get_capability(key: str) -> str | None:
            nonlocal call_count
            if key == "apply_patch.engine":
                call_count += 1
                return "native"
            return None

        provider.coordinator.get_capability = MagicMock(side_effect=mock_get_capability)

        tool_spec = _make_apply_patch_tool_spec()

        # First call — should query capability
        provider._convert_tools_from_request([tool_spec])
        assert call_count == 1
        assert provider._apply_patch_native is True

        # Second call — flag is already True, should NOT query again
        provider._convert_tools_from_request([tool_spec])
        assert call_count == 1  # No additional query


# --- Test history replay of apply_patch_call and Bug A (call_id priority) ---


class TestApplyPatchCallHistoryReplay:
    def test_apply_patch_call_replayed_as_apply_patch_call_not_function_call(
        self,
    ) -> None:
        """Historical native apply_patch_call must replay as apply_patch_call, not function_call.

        When _convert_messages encounters a stored assistant 'tool_call' block whose
        name is 'apply_patch' and whose input has a 'type' key matching a native
        operation type (e.g. 'update_file'), it must emit an apply_patch_call item
        rather than a function_call item.
        """
        provider = _make_provider()
        provider._native_call_ids = set()

        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_call",
                        "id": "call_native_123",
                        "name": "apply_patch",
                        "input": {
                            "type": "update_file",
                            "path": "src/main.py",
                            "diff": "@@ -1 +1 @@\n-old\n+new",
                        },
                    }
                ],
            }
        ]

        result = provider._convert_messages(messages)

        # Must NOT produce a function_call item for this native op
        function_call_items = [
            m
            for m in result
            if isinstance(m, dict) and m.get("type") == "function_call"
        ]
        assert len(function_call_items) == 0, (
            "Historical native apply_patch_call should not be replayed as function_call"
        )

        # Must produce an apply_patch_call item with the right shape
        patch_call_items = [
            m
            for m in result
            if isinstance(m, dict) and m.get("type") == "apply_patch_call"
        ]
        assert len(patch_call_items) == 1
        item = patch_call_items[0]
        assert item["call_id"] == "call_native_123"
        assert isinstance(item.get("operation"), dict)
        assert item["operation"]["type"] == "update_file"
        assert item["operation"]["path"] == "src/main.py"

    def test_apply_patch_call_output_type_correct_for_historical_native_call(
        self,
    ) -> None:
        """Tool result for a historical native apply_patch_call must use apply_patch_call_output.

        After _convert_messages processes the assistant message containing a native
        apply_patch_call, the call_id is registered in _native_call_ids.  The
        subsequent tool-result message for that call_id must therefore be emitted as
        apply_patch_call_output (not function_call_output).
        """
        provider = _make_provider()
        provider._native_call_ids = set()

        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_call",
                        "id": "call_native_456",
                        "name": "apply_patch",
                        "input": {
                            "type": "create_file",
                            "path": "new_module.py",
                            "diff": "+print('hello')",
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_native_456",
                "content": "A new_module.py",
                "tool_name": "apply_patch",
            },
        ]

        result = provider._convert_messages(messages)

        # The tool result must be apply_patch_call_output, not function_call_output
        func_outputs = [
            m
            for m in result
            if isinstance(m, dict) and m.get("type") == "function_call_output"
        ]
        assert len(func_outputs) == 0, (
            "Historical native apply_patch_call output should not use function_call_output"
        )

        patch_outputs = [
            m
            for m in result
            if isinstance(m, dict) and m.get("type") == "apply_patch_call_output"
        ]
        assert len(patch_outputs) == 1
        assert patch_outputs[0]["call_id"] == "call_native_456"
        assert patch_outputs[0]["output"] == "A new_module.py"

    def test_function_call_id_uses_call_id_not_item_id(self) -> None:
        """Bug A: function_call response items should use call_id, not the server item id.

        When _convert_to_chat_response parses a function_call (or tool_call) response
        item that carries both 'id' (server-assigned response-item ID, e.g. 'fc_abc...')
        and 'call_id' (the tool correlation ID, e.g. 'call_abc...'), the resulting
        ToolCallBlock and ToolCall must use the call_id value, not the item id.
        """
        provider = _make_provider()
        provider._native_call_ids = set()

        # Simulate a response block with both id and call_id set
        mock_block = MagicMock()
        mock_block.type = "function_call"
        mock_block.id = "fc_abc123"
        mock_block.call_id = "call_abc123"
        mock_block.name = "read_file"
        mock_block.input = {"path": "src/main.py"}
        mock_block.arguments = '{"path": "src/main.py"}'

        mock_response = MagicMock()
        mock_response.model = "gpt-5.5"
        mock_response.output = [mock_block]
        mock_response.usage = MagicMock()
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5
        mock_response.usage.total_tokens = 15
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.output_tokens_details = None
        mock_response.usage.input_tokens_details = None
        mock_response.id = "resp_bugA"
        mock_response.status = "completed"
        mock_response.incomplete_details = None
        mock_response.finish_reason = None
        mock_response.output_text = None

        chat_response = provider._convert_to_chat_response(mock_response)

        assert chat_response.tool_calls is not None
        assert len(chat_response.tool_calls) == 1
        tc = chat_response.tool_calls[0]
        # Must be call_id ("call_abc123"), NOT item id ("fc_abc123")
        assert tc.id == "call_abc123", (
            f"Expected call_id 'call_abc123' but got '{tc.id}' — "
            "function_call items must use call_id, not the server item id"
        )

    def test_function_call_apply_patch_without_operation_type_stays_function_call(
        self,
    ) -> None:
        """apply_patch in function-mode (no native op type) must stay as function_call.

        A 'tool_call' block with name='apply_patch' whose input uses the function-mode
        argument shape (e.g. {'operations': [...]}) does NOT have a 'type' key matching
        a native operation type.  It must be replayed as a regular function_call, not
        promoted to apply_patch_call.
        """
        provider = _make_provider()
        provider._native_call_ids = set()

        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_call",
                        "id": "call_func_ap",
                        "name": "apply_patch",
                        # Function-mode shape: no top-level 'type' key matching op types
                        "input": {
                            "operations": [
                                {
                                    "operation": "replace",
                                    "path": "src/foo.py",
                                    "content": "new content",
                                }
                            ]
                        },
                    }
                ],
            }
        ]

        result = provider._convert_messages(messages)

        # Must NOT produce an apply_patch_call item
        patch_call_items = [
            m
            for m in result
            if isinstance(m, dict) and m.get("type") == "apply_patch_call"
        ]
        assert len(patch_call_items) == 0, (
            "Function-mode apply_patch (no native op type) must not be replayed as apply_patch_call"
        )

        # Must produce a regular function_call item
        func_call_items = [
            m
            for m in result
            if isinstance(m, dict) and m.get("type") == "function_call"
        ]
        assert len(func_call_items) == 1
        assert func_call_items[0]["call_id"] == "call_func_ap"
        assert func_call_items[0]["name"] == "apply_patch"

        # Must NOT have added call_id to native_call_ids
        assert "call_func_ap" not in provider._native_call_ids
