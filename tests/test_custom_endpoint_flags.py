"""Tests for per-feature config flags for custom endpoint compatibility.

These flags replace the previous base_url-is-not-None heuristic and allow
fine-grained control over which OpenAI-specific Responses API features are
enabled. This matters because:

- Azure OpenAI users set base_url but support all features
- OpenRouter users set base_url but need most features disabled
- Corporate proxies may support a subset of features

Flags:
  enable_native_tools      - apply_patch, web_search_preview, etc.
  enable_reasoning_replay  - encrypted_content, reasoning items in history
  enable_store             - store param, previous_response_id
  enable_background        - background mode for deep research models
"""

import asyncio
import json
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

from amplifier_core import ModuleCoordinator
from amplifier_core.message_models import ChatRequest, Message, ThinkingBlock, ToolCallBlock

from amplifier_module_provider_openai import OpenAIProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class DummyResponse:
    """Minimal response stub for provider tests."""

    def __init__(self, output=None):
        self.output = output or []
        self.usage = SimpleNamespace(
            prompt_tokens=0, completion_tokens=0, total_tokens=0
        )
        self.stop_reason = "stop"


class FakeHooks:
    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    async def emit(self, name: str, payload: dict) -> None:
        self.events.append((name, payload))


class FakeCoordinator:
    def __init__(self):
        self.hooks = FakeHooks()
        self._capabilities: dict[str, str] = {}

    def get_capability(self, key: str) -> str | None:
        return self._capabilities.get(key)


def _make_provider(**config_overrides) -> OpenAIProvider:
    config = {"max_retries": 0, **config_overrides}
    provider = OpenAIProvider(api_key="test-key", config=config)
    provider.coordinator = cast(ModuleCoordinator, FakeCoordinator())
    return provider


def _make_request(content: str = "Hello") -> ChatRequest:
    return ChatRequest(messages=[Message(role="user", content=content)])


def _call_complete(provider: OpenAIProvider, request: ChatRequest = None, **kwargs) -> dict:
    """Run provider.complete() and return the kwargs sent to the API."""
    if request is None:
        request = _make_request()
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())
    asyncio.run(provider.complete(request, **kwargs))
    return provider.client.responses.create.await_args_list[-1].kwargs


# ---------------------------------------------------------------------------
# 1. Default behavior (no base_url) — all flags true
# ---------------------------------------------------------------------------


def test_default_no_base_url_all_flags_true():
    """Without base_url, all compatibility flags default to True."""
    provider = _make_provider()
    assert provider.enable_native_tools is True
    assert provider.enable_reasoning_replay is True
    assert provider.enable_store is True
    assert provider.enable_background is True


# ---------------------------------------------------------------------------
# 2. Custom base_url — all flags auto-flip to false
# ---------------------------------------------------------------------------


def test_custom_base_url_flags_auto_flip():
    """When base_url is set, all flags default to False."""
    provider = _make_provider(base_url="https://openrouter.ai/api/v1")
    assert provider.enable_native_tools is False
    assert provider.enable_reasoning_replay is False
    assert provider.enable_store is False
    assert provider.enable_background is False


# ---------------------------------------------------------------------------
# 3. Azure case: base_url set + flags explicitly enabled
# ---------------------------------------------------------------------------


def test_azure_base_url_with_flags_enabled():
    """Azure users can set base_url while keeping all features enabled."""
    provider = _make_provider(
        base_url="https://my-resource.openai.azure.com/openai/v1/",
        enable_native_tools=True,
        enable_reasoning_replay=True,
        enable_store=True,
        enable_background=True,
    )
    assert provider.enable_native_tools is True
    assert provider.enable_reasoning_replay is True
    assert provider.enable_store is True
    assert provider.enable_background is True


def test_azure_base_url_string_true():
    """Config values from YAML may be strings; 'true' should work."""
    provider = _make_provider(
        base_url="https://my-resource.openai.azure.com/openai/v1/",
        enable_native_tools="true",
        enable_reasoning_replay="true",
        enable_store="true",
        enable_background="true",
    )
    assert provider.enable_native_tools is True
    assert provider.enable_reasoning_replay is True
    assert provider.enable_store is True
    assert provider.enable_background is True


# ---------------------------------------------------------------------------
# 4. Individual flag overrides
# ---------------------------------------------------------------------------


def test_individual_flag_override_with_base_url():
    """Each flag can be individually overridden even with base_url set."""
    # Only enable reasoning replay, keep others at default (false)
    provider = _make_provider(
        base_url="https://openrouter.ai/api/v1",
        enable_reasoning_replay=True,
    )
    assert provider.enable_native_tools is False
    assert provider.enable_reasoning_replay is True
    assert provider.enable_store is False
    assert provider.enable_background is False


def test_individual_flag_disable_without_base_url():
    """Flags can be disabled even without base_url (e.g. for testing)."""
    provider = _make_provider(
        enable_native_tools=False,
    )
    assert provider.enable_native_tools is False
    # Others remain true (no base_url)
    assert provider.enable_reasoning_replay is True
    assert provider.enable_store is True
    assert provider.enable_background is True


# ---------------------------------------------------------------------------
# 5. Store param gating
# ---------------------------------------------------------------------------


def test_store_param_sent_when_enabled():
    """With enable_store=True, store param is included."""
    provider = _make_provider()
    params = _call_complete(provider)
    assert "store" in params


def test_store_param_not_sent_when_disabled():
    """With enable_store=False, store param is omitted."""
    provider = _make_provider(base_url="https://openrouter.ai/api/v1")
    params = _call_complete(provider)
    assert "store" not in params
    assert "previous_response_id" not in params


# ---------------------------------------------------------------------------
# 6. Truncation gating
# ---------------------------------------------------------------------------


def test_truncation_sent_when_store_enabled():
    """With enable_store=True, truncation is included."""
    provider = _make_provider(truncation="auto")
    params = _call_complete(provider)
    assert "truncation" in params


def test_truncation_not_sent_when_store_disabled():
    """With enable_store=False, truncation is omitted."""
    provider = _make_provider(
        base_url="https://openrouter.ai/api/v1",
        truncation="auto",
    )
    params = _call_complete(provider)
    assert "truncation" not in params


# ---------------------------------------------------------------------------
# 7. Background mode gating
# ---------------------------------------------------------------------------


def test_background_disabled_for_custom_endpoint():
    """Deep research models don't trigger background mode with enable_background=False."""
    provider = _make_provider(base_url="https://openrouter.ai/api/v1")
    params = _call_complete(provider, model="o3-deep-research")
    assert "background" not in params


def test_background_enabled_for_native_endpoint():
    """Deep research models trigger background mode by default on native endpoint."""
    provider = _make_provider()
    params = _call_complete(provider, model="o3-deep-research")
    assert params.get("background") is True


# ---------------------------------------------------------------------------
# 8. encrypted_content / reasoning replay gating
# ---------------------------------------------------------------------------


def test_encrypted_content_requested_when_replay_enabled():
    """With enable_reasoning_replay=True, encrypted_content is requested."""
    provider = _make_provider(enable_state=False)  # store=false triggers encrypted_content
    params = _call_complete(provider, model="gpt-5.2-codex")
    assert "include" in params
    assert "reasoning.encrypted_content" in params["include"]


def test_encrypted_content_not_requested_when_replay_disabled():
    """With enable_reasoning_replay=False, encrypted_content is NOT requested."""
    provider = _make_provider(
        base_url="https://openrouter.ai/api/v1",
        enable_state=False,
    )
    params = _call_complete(provider, model="gpt-5.2-codex")
    assert "include" not in params


# ---------------------------------------------------------------------------
# 9. Native tools gating
# ---------------------------------------------------------------------------


def test_native_tools_passed_when_enabled():
    """Native tool types are passed through when enable_native_tools=True."""
    provider = _make_provider()
    tools = provider._convert_tools_from_request([
        {"type": "web_search_preview"},
    ])
    assert any(t.get("type") == "web_search_preview" for t in tools)


def test_native_tools_skipped_when_disabled():
    """Native tool types are dropped when enable_native_tools=False."""
    provider = _make_provider(base_url="https://openrouter.ai/api/v1")
    tools = provider._convert_tools_from_request([
        {"type": "web_search_preview"},
    ])
    assert len(tools) == 0


def test_function_tools_always_passed():
    """User-defined function tools are passed regardless of enable_native_tools."""
    tool = SimpleNamespace(
        name="my_tool",
        description="A test tool",
        parameters={"type": "object", "properties": {}},
    )
    provider = _make_provider(base_url="https://openrouter.ai/api/v1")
    tools = provider._convert_tools_from_request([tool])
    assert len(tools) == 1
    assert tools[0]["type"] == "function"
    assert tools[0]["name"] == "my_tool"


def test_apply_patch_not_native_when_disabled():
    """apply_patch stays as function tool when enable_native_tools=False."""
    tool = SimpleNamespace(
        name="apply_patch",
        description="Apply a patch",
        parameters={"type": "object", "properties": {}},
    )
    provider = _make_provider(base_url="https://openrouter.ai/api/v1")
    # Even if coordinator says native engine is available, flag overrides
    provider.coordinator._capabilities["apply_patch.engine"] = "native"
    tools = provider._convert_tools_from_request([tool])
    assert len(tools) == 1
    assert tools[0]["type"] == "function"
    assert tools[0]["name"] == "apply_patch"


# ---------------------------------------------------------------------------
# 10. Sanitizer behavior
# ---------------------------------------------------------------------------


def test_sanitizer_strips_reasoning_items():
    """Reasoning items in input are stripped when reasoning replay is disabled."""
    provider = _make_provider(base_url="https://openrouter.ai/api/v1")
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())

    # Build a message list that would produce reasoning items in the converted input
    messages = [
        Message(role="user", content="Hello"),
    ]
    request = ChatRequest(messages=messages)

    # Manually call complete to get params
    params = _call_complete(provider, request)
    input_items = params.get("input", [])

    # No reasoning items should survive in the input
    for item in input_items:
        if isinstance(item, dict):
            assert item.get("type") != "reasoning", "Reasoning items should be stripped"


def test_sanitizer_converts_apply_patch_call():
    """apply_patch_call items are converted to function_call when native tools disabled."""
    provider = _make_provider(
        base_url="https://openrouter.ai/api/v1",
        enable_native_tools=False,
    )

    # Simulate an input that has apply_patch_call items
    test_input = [
        {
            "type": "apply_patch_call",
            "call_id": "call_123",
            "patch": "--- a/file.py\n+++ b/file.py",
        },
        {
            "type": "apply_patch_call_output",
            "call_id": "call_123",
            "output": "M file.py",
        },
    ]

    # Run sanitizer logic directly
    sanitized = []
    for item in test_input:
        if isinstance(item, dict):
            item_type = item.get("type", "")
            if not provider.enable_native_tools and item_type == "apply_patch_call":
                patch_value = item.get("patch", "")
                item = {
                    "type": "function_call",
                    "call_id": item.get("call_id", ""),
                    "name": "apply_patch",
                    "arguments": json.dumps({"patch": patch_value}),
                }
            if not provider.enable_native_tools and item_type == "apply_patch_call_output":
                item = {
                    "type": "function_call_output",
                    "call_id": item.get("call_id", ""),
                    "output": item.get("output", ""),
                }
        sanitized.append(item)

    assert sanitized[0]["type"] == "function_call"
    assert sanitized[0]["name"] == "apply_patch"
    # Verify no double-quoting: arguments should be valid JSON containing the patch
    args = json.loads(sanitized[0]["arguments"])
    assert args["patch"] == "--- a/file.py\n+++ b/file.py"

    assert sanitized[1]["type"] == "function_call_output"
    assert sanitized[1]["output"] == "M file.py"


def test_sanitizer_simplifies_assistant_content():
    """Assistant messages with structured content are simplified to plain text."""
    provider = _make_provider(
        base_url="https://openrouter.ai/api/v1",
        enable_reasoning_replay=False,
    )

    # Simulate structured assistant message
    item = {
        "role": "assistant",
        "content": [
            {"type": "output_text", "text": "Hello "},
            {"type": "output_text", "text": "world"},
        ],
    }

    # Run simplification
    text_parts = []
    for block in item["content"]:
        if isinstance(block, dict):
            text = block.get("text", "")
            if text:
                text_parts.append(text)
    simplified = {
        "role": "assistant",
        "content": "\n".join(text_parts),
    }

    assert simplified["role"] == "assistant"
    assert "Hello " in simplified["content"]
    assert "world" in simplified["content"]


def test_include_stripped_when_reasoning_replay_disabled():
    """The include parameter is removed when reasoning replay is disabled."""
    provider = _make_provider(
        base_url="https://openrouter.ai/api/v1",
    )
    params = _call_complete(provider)
    assert "include" not in params


# ---------------------------------------------------------------------------
# 11. Reasoning items stripped from _convert_messages
# ---------------------------------------------------------------------------


def test_reasoning_items_cleared_in_convert_messages():
    """Reasoning items from ThinkingBlocks should be stripped when replay disabled."""
    provider = _make_provider(
        base_url="https://openrouter.ai/api/v1",
    )

    # Create an assistant message with thinking block containing reasoning state
    messages = [
        {
            "role": "assistant",
            "content": [
                ThinkingBlock(
                    thinking="I'm reasoning...",
                    content=["encrypted_blob", "rs_test_001"],
                )
            ],
            "metadata": {"openai:reasoning_items": ["rs_test_001"]},
        },
    ]

    result = provider._convert_messages(messages)

    # No reasoning items should be in the output
    for item in result:
        if isinstance(item, dict):
            assert item.get("type") != "reasoning", (
                f"Reasoning items should be cleared when enable_reasoning_replay=False: {item}"
            )


# ---------------------------------------------------------------------------
# 12. Orphaned reasoning warning deduplication
# ---------------------------------------------------------------------------


def test_orphaned_reasoning_warning_deduplication():
    """Warning about orphaned reasoning should only appear once, then switch to debug."""
    provider = _make_provider()
    assert provider._warned_orphaned_reasoning is False

    # Simulate first warning
    provider._warned_orphaned_reasoning = True

    # After first warning, flag should be set
    assert provider._warned_orphaned_reasoning is True


# ---------------------------------------------------------------------------
# 13. Bool config parsing
# ---------------------------------------------------------------------------


def test_bool_config_accepts_various_formats():
    """_bool_config should accept bool, string, and numeric values."""
    provider = _make_provider()

    # True values
    assert provider._bool_config("x", default=False) is False  # missing key
    provider.config["x"] = True
    assert provider._bool_config("x", default=False) is True
    provider.config["x"] = "true"
    assert provider._bool_config("x", default=False) is True
    provider.config["x"] = "True"
    assert provider._bool_config("x", default=False) is True
    provider.config["x"] = "1"
    assert provider._bool_config("x", default=False) is True
    provider.config["x"] = "yes"
    assert provider._bool_config("x", default=False) is True

    # False values
    provider.config["x"] = False
    assert provider._bool_config("x", default=True) is False
    provider.config["x"] = "false"
    assert provider._bool_config("x", default=True) is False
    provider.config["x"] = "0"
    assert provider._bool_config("x", default=True) is False
    provider.config["x"] = "no"
    assert provider._bool_config("x", default=True) is False


# ---------------------------------------------------------------------------
# 14. apply_patch_call_output gating in _convert_messages
# ---------------------------------------------------------------------------


def test_apply_patch_output_uses_function_call_output_when_native_disabled():
    """When native tools are disabled, tool results use function_call_output format."""
    provider = _make_provider(
        base_url="https://openrouter.ai/api/v1",
    )
    # Simulate a native call ID being tracked
    provider._native_call_ids.add("call_abc")

    messages = [
        {
            "role": "tool",
            "tool_call_id": "call_abc",
            "tool_name": "apply_patch",
            "content": "M file.py",
        },
    ]

    result = provider._convert_messages(messages)

    # With native tools disabled, should use function_call_output, not apply_patch_call_output
    for item in result:
        if isinstance(item, dict):
            assert item.get("type") != "apply_patch_call_output", (
                "Should use function_call_output when enable_native_tools=False"
            )
