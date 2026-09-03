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
from collections.abc import AsyncGenerator
from decimal import Decimal

import pytest
import pytest_asyncio
from amplifier_core.testing import MockCoordinator

from amplifier_module_provider_openai import OpenAIProvider, mount

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MODEL = "gpt-6-astra"


@pytest_asyncio.fixture
async def astra_provider() -> AsyncGenerator[OpenAIProvider, None]:
    """Mount the local provider module and return its OpenAI provider."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")
    coordinator = MockCoordinator()
    cleanup = await mount(
        coordinator,
        {
            "api_key": api_key,
            "default_model": _MODEL,
            "use_streaming": False,
            "max_retries": 1,
        },
    )
    provider = coordinator.mount_points["providers"]["openai"]
    try:
        yield provider
    finally:
        if cleanup is not None:
            await cleanup()


def _handle_live_error(exc: Exception) -> None:
    """Convert only confirmed access failures to sanitized BLOCKED terminals."""
    candidates: list[BaseException] = [exc]
    if exc.__cause__ is not None:
        candidates.append(exc.__cause__)
    if exc.__context__ is not None:
        candidates.append(exc.__context__)

    status = next(
        (
            value
            for candidate in candidates
            if (value := getattr(candidate, "status_code", None)) is not None
        ),
        None,
    )
    class_names = {type(candidate).__name__ for candidate in candidates}
    exc_class = type(exc).__name__

    if status in {401, 403} or class_names & {
        "AuthenticationError",
        "PermissionDeniedError",
    }:
        pytest.skip(
            "LIVE: BLOCKED — OPENAI_API_KEY unavailable or invalid "
            f"(exception: {exc_class}, status: {status or 'unknown'})"
        )
    if status == 404 or "NotFoundError" in class_names:
        pytest.skip(
            "LIVE: BLOCKED — account lacks gpt-6-astra entitlement "
            f"(exception: {exc_class}, status: {status or 'unknown'})"
        )
    if class_names & {
        "APIConnectionError",
        "APITimeoutError",
        "ConnectionError",
        "TimeoutError",
    }:
        pytest.skip(
            "LIVE: BLOCKED — network unavailable "
            f"(exception: {exc_class}, status: {status or 'unknown'})"
        )
    pytest.fail(
        "LIVE: FAIL — implementation-caused request failure: "
        f"{exc_class}, status: {status or 'unknown'}"
    )


# ---------------------------------------------------------------------------
# Live tests
# ---------------------------------------------------------------------------


@pytest.mark.live
@pytest.mark.asyncio
async def test_live_astra_text_completion(astra_provider: OpenAIProvider):
    """Live: one small low-effort text completion through gpt-6-astra.

    Asserts:
    - Completion status is 'completed'
    - Response contains non-empty text content
    - Returned model metadata matches gpt-6-astra
    - Usage fields are present
    - Standard cost is non-null (cost accounting works)
    """
    from amplifier_core.message_models import ChatRequest, Message

    request = ChatRequest(
        messages=[Message(role="user", content="Say 'hello world' and nothing else.")]
    )

    try:
        response = await astra_provider.complete(
            request,
            reasoning={"effort": "low"},
            max_tokens=64,
        )
    except Exception as e:  # noqa: BLE001
        _handle_live_error(e)

    # Assertions on the response
    assert response is not None, "Response must not be None"
    assert response.metadata
    assert response.metadata.get("openai:status") == "completed"
    assert response.metadata.get("model") == _MODEL
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
    assert text, (
        f"Response must contain non-empty text content, got: {response.content}"
    )

    # Usage must be present
    assert response.usage is not None, "Usage must be present"
    assert response.usage.input_tokens > 0, "input_tokens must be > 0"
    assert response.usage.output_tokens > 0, "output_tokens must be > 0"

    cost = response.usage.cost_usd
    assert cost is not None, "Provider must stamp Standard cost for gpt-6-astra"
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
async def test_live_astra_function_tool_call(astra_provider: OpenAIProvider):
    """Live: forced function-tool call with follow-up tool result.

    Asserts:
    - Model returns a tool_call block
    - Tool call has correct name and parseable arguments
    - Follow-up with tool result produces a text completion
    - Usage and cost are non-null
    """
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
        response = await astra_provider.complete(
            request,
            reasoning={"effort": "low"},
            max_tokens=256,
            tool_choice="required",
        )
    except Exception as e:  # noqa: BLE001
        _handle_live_error(e)

    assert response is not None
    assert response.tool_calls, "A required tool choice must return a tool call"
    tool_call = next(
        (call for call in response.tool_calls if call.name == tool.name),
        None,
    )
    assert tool_call is not None, f"Expected tool call {tool.name!r}"
    assert isinstance(tool_call.arguments, dict)
    assert tool_call.arguments.get("city") == "Paris"
    assert response.usage is not None
    assert response.usage.cost_usd is not None

    follow_up = ChatRequest(
        messages=[
            *request.messages,
            Message(
                role="assistant",
                content=response.content,
            ),
            Message(
                role="tool",
                name=tool_call.name,
                tool_call_id=tool_call.id,
                content='{"temperature": 20, "unit": "celsius"}',
            ),
        ],
        tools=[tool],
    )
    try:
        final_response = await astra_provider.complete(
            follow_up,
            reasoning={"effort": "low"},
            max_tokens=128,
        )
    except Exception as e:  # noqa: BLE001
        _handle_live_error(e)

    assert final_response.metadata
    assert final_response.metadata.get("openai:status") == "completed"
    assert final_response.metadata.get("model") == _MODEL
    assert final_response.usage is not None
    assert final_response.usage.cost_usd is not None
    assert any(
        getattr(block, "text", "").strip() for block in final_response.content
    ), "Follow-up tool result must produce final text"

    print(
        f"\nLIVE: PASS — model=gpt-6-astra, "
        f"tool_calls=1, "
        f"input_tokens={final_response.usage.input_tokens}, "
        f"output_tokens={final_response.usage.output_tokens}, "
        f"cost_usd={final_response.usage.cost_usd:.6f}"
    )
