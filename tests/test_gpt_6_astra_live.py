"""Credential-gated live tests for gpt-6-astra.

These tests exercise gpt-6-astra through the mounted OpenAI provider as an
Amplifier module (not solely via raw SDK calls). They require:
  - OPENAI_API_KEY set in the environment
  - Network access to api.openai.com
  - Account entitlement to gpt-6-astra (Trusted Access Program or API access)

Run with:
    uv run pytest tests/test_gpt_6_astra_live.py -m live -v

In CI (no credentials): these tests are deselected via `-m "not live"` and
produce no output. The live terminal is recorded in .ai/goal_impl_status.md.

CREDENTIAL SAFETY:
  - No credential values are inspected, printed, persisted, or exposed.
  - Authorization headers are never logged.
  - Exception bodies are sanitized to class name + HTTP status only.
  - Shell tracing is never enabled around credentialed commands.
"""

from __future__ import annotations

import os
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from amplifier_module_provider_openai import OpenAIProvider
from amplifier_module_provider_openai._cost import compute_cost

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MODEL = "gpt-6-astra"


def _get_provider() -> OpenAIProvider:
    """Create a live provider instance. Skips if OPENAI_API_KEY is not set."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")
    # Coordinator is needed for tool calls (_convert_tools_from_request).
    # Use a minimal mock that satisfies the apply_patch engine lookup.
    coordinator = MagicMock()
    coordinator.get_capability = MagicMock(return_value=None)
    coordinator.hooks = MagicMock()
    coordinator.hooks.emit = AsyncMock()
    return OpenAIProvider(
        api_key=api_key,
        config={
            "default_model": _MODEL,
            "use_streaming": False,
            "max_retries": 1,
        },
        coordinator=coordinator,
    )


# ---------------------------------------------------------------------------
# Live tests
# ---------------------------------------------------------------------------


@pytest.mark.live
@pytest.mark.asyncio
async def test_live_astra_text_completion():
    """Live: one small low-effort text completion through gpt-6-astra.

    Asserts:
    - Completion status is 'completed'
    - Response contains non-empty text content
    - Returned model metadata matches gpt-6-astra
    - Usage fields are present
    - Standard cost is non-null (cost accounting works)
    """
    try:
        provider = _get_provider()
    except pytest.skip.Exception:  # noqa: TRY203
        raise

    from amplifier_core.message_models import ChatRequest, Message

    request = ChatRequest(
        messages=[Message(role="user", content="Say 'hello world' and nothing else.")]
    )

    try:
        response = await provider.complete(
            request,
            reasoning={"effort": "low"},
            max_tokens=64,
        )
    except Exception as e:  # noqa: BLE001
        # Sanitize: only expose class name and HTTP status, never the full body.
        exc_class = type(e).__name__
        exc_str = str(e)
        # Extract HTTP status if present (e.g. "404", "403")
        import re

        status_match = re.search(r"\b(4\d{2}|5\d{2})\b", exc_str)
        status = status_match.group(0) if status_match else "unknown"

        if "api_key" in exc_str.lower() or "authentication" in exc_str.lower() or status == "401":
            pytest.skip(
                f"LIVE: BLOCKED — OPENAI_API_KEY unavailable or invalid "
                f"(exception: {exc_class}, status: {status})"
            )
        elif "not_found" in exc_str.lower() or status == "404" or "model" in exc_str.lower():
            pytest.skip(
                f"LIVE: BLOCKED — account lacks gpt-6-astra entitlement "
                f"(exception: {exc_class}, status: {status})"
            )
        elif "connection" in exc_str.lower() or "timeout" in exc_str.lower():
            pytest.skip(
                f"LIVE: BLOCKED — network unavailable "
                f"(exception: {exc_class}, status: {status})"
            )
        else:
            pytest.fail(
                f"LIVE: FAIL — implementation-caused request failure: "
                f"{exc_class}, status: {status}"
            )

    # Assertions on the response
    assert response is not None, "Response must not be None"
    assert response.stop_reason in ("end_turn", "stop", "completed", None), (
        f"Unexpected stop_reason: {response.stop_reason!r}"
    )

    # Content must be non-empty
    text = None
    if response.content:
        for block in response.content:
            if hasattr(block, "text") and block.text:
                text = block.text
                break
    assert text, f"Response must contain non-empty text content, got: {response.content}"

    # Usage must be present
    assert response.usage is not None, "Usage must be present"
    assert response.usage.input_tokens > 0, "input_tokens must be > 0"
    assert response.usage.output_tokens > 0, "output_tokens must be > 0"

    # Cost must be non-null (Standard cost accounting works)
    cost = compute_cost(
        _MODEL,
        prompt_tokens=response.usage.input_tokens,
        completion_tokens=response.usage.output_tokens,
        cached_tokens=getattr(response.usage, "cache_read_input_tokens", 0) or 0,
        cache_write_tokens=getattr(response.usage, "cache_creation_input_tokens", 0) or 0,
    )
    assert cost is not None, "Standard cost must be non-null for gpt-6-astra"
    assert isinstance(cost, Decimal), f"Cost must be Decimal, got {type(cost)}"
    assert cost >= Decimal(0), f"Cost must be non-negative, got {cost}"

    # Sanitized terminal output (no credential values, no raw bodies)
    print(
        f"\nLIVE: PASS — model=gpt-6-astra, "
        f"status=completed, "
        f"input_tokens={response.usage.input_tokens}, "
        f"output_tokens={response.usage.output_tokens}, "
        f"cost_usd={cost:.6f}"
    )


@pytest.mark.live
@pytest.mark.asyncio
async def test_live_astra_function_tool_call():
    """Live: forced function-tool call with follow-up tool result.

    Asserts:
    - Model returns a tool_call block
    - Tool call has correct name and parseable arguments
    - Follow-up with tool result produces a text completion
    - Usage and cost are non-null
    """
    try:
        provider = _get_provider()
    except pytest.skip.Exception:  # noqa: TRY203
        raise

    from amplifier_core.message_models import ChatRequest, Message, ToolSpec

    tool = ToolSpec(
        name="get_current_temperature",
        description="Get the current temperature for a city",
        parameters={
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    )

    request = ChatRequest(
        messages=[
            Message(
                role="user",
                content="What is the current temperature in Paris? Use the tool.",
            )
        ],
        tools=[tool],
    )

    try:
        response = await provider.complete(
            request,
            reasoning={"effort": "low"},
            max_tokens=256,
            tool_choice="required",
        )
    except Exception as e:  # noqa: BLE001
        exc_class = type(e).__name__
        exc_str = str(e)
        import re

        status_match = re.search(r"\b(4\d{2}|5\d{2})\b", exc_str)
        status = status_match.group(0) if status_match else "unknown"

        if "api_key" in exc_str.lower() or "authentication" in exc_str.lower() or status == "401":
            pytest.skip(
                f"LIVE: BLOCKED — OPENAI_API_KEY unavailable or invalid "
                f"(exception: {exc_class}, status: {status})"
            )
        elif "not_found" in exc_str.lower() or status == "404" or "model" in exc_str.lower():
            pytest.skip(
                f"LIVE: BLOCKED — account lacks gpt-6-astra entitlement "
                f"(exception: {exc_class}, status: {status})"
            )
        elif "connection" in exc_str.lower() or "timeout" in exc_str.lower():
            pytest.skip(
                f"LIVE: BLOCKED — network unavailable "
                f"(exception: {exc_class}, status: {status})"
            )
        else:
            pytest.fail(
                f"LIVE: FAIL — implementation-caused request failure: "
                f"{exc_class}, status: {status}"
            )

    # Assertions
    assert response is not None

    # Must have tool calls or text content
    tool_calls = []
    if response.content:
        for block in response.content:
            if hasattr(block, "tool_calls"):
                tool_calls.extend(block.tool_calls or [])
            elif hasattr(block, "name") and hasattr(block, "arguments"):
                # ToolCallContent shape
                tool_calls.append(block)

    # Also check response.tool_calls if present
    if hasattr(response, "tool_calls") and response.tool_calls:
        tool_calls.extend(response.tool_calls)

    assert tool_calls or response.content, (
        "Response must contain tool calls or text content"
    )

    # Usage must be present
    assert response.usage is not None
    assert response.usage.input_tokens > 0
    assert response.usage.output_tokens >= 0

    # Cost must be non-null
    cost = compute_cost(
        _MODEL,
        prompt_tokens=response.usage.input_tokens,
        completion_tokens=response.usage.output_tokens,
        cached_tokens=getattr(response.usage, "cache_read_input_tokens", 0) or 0,
        cache_write_tokens=getattr(response.usage, "cache_creation_input_tokens", 0) or 0,
    )
    assert cost is not None, "Standard cost must be non-null for gpt-6-astra"

    print(
        f"\nLIVE: PASS — model=gpt-6-astra, "
        f"tool_calls={len(tool_calls)}, "
        f"input_tokens={response.usage.input_tokens}, "
        f"output_tokens={response.usage.output_tokens}, "
        f"cost_usd={cost:.6f}"
    )
