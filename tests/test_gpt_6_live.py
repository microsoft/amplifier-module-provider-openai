"""Opt-in live GPT-6 Sol/Luna contract tests against the real OpenAI API.

Marked `@pytest.mark.live` (deselected in CI via `-m "not live"`, per
pyproject.toml). Beyond that marker, every test here ALSO requires an
explicit opt-in environment variable -- `AMPLIFIER_GPT6_LIVE=1` -- in
addition to a genuine `OPENAI_API_KEY`. CI sets a placeholder API key so
other (non-live) tests can mount the provider, which defeats a plain
"if not api_key: skip" check; the extra opt-in variable is the real gate
here so these never fire by accident from a placeholder key alone.

Covers Sol and Luna only. Astra is intentionally excluded by default --
it is the more expensive/likely-slower tier and isn't needed to validate
the same request/response contract Sol and Luna already exercise.

Kept deliberately cheap: `max_retries=0`, tiny `max_output_tokens`, and no
retry-inducing behavior. Nothing here persists a raw request/response
payload -- assertions are on parsed, sanitized shape only (types, small
length checks), never printing or writing full model output.
"""

from __future__ import annotations

import os

import pytest
from amplifier_core import llm_errors as kernel_errors
from amplifier_core.message_models import (
    ChatRequest,
    Message,
    TextBlock,
    ToolCallBlock,
    ToolSpec,
)

from amplifier_module_provider_openai import OpenAIProvider


def _live_opt_in() -> str | None:
    """Return a real-looking API key only when BOTH gates are satisfied.

    Returns None (meaning "skip") unless AMPLIFIER_GPT6_LIVE=1 is set AND
    OPENAI_API_KEY looks like a real key, not CI's placeholder.
    """
    if os.environ.get("AMPLIFIER_GPT6_LIVE") != "1":
        return None
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key or api_key.startswith("[REDACTED"):
        return None
    return api_key


def _require_live() -> str:
    api_key = _live_opt_in()
    if not api_key:
        pytest.skip("Live GPT-6 test requires OPENAI_API_KEY and AMPLIFIER_GPT6_LIVE=1")
    return api_key


@pytest.mark.live
@pytest.mark.asyncio
async def test_live_sol_minimal_streaming_text() -> None:
    """Sol, streaming on, a tiny prompt: a minimal text response streams
    back and the final ChatResponse carries non-empty text."""
    api_key = _require_live()
    provider = OpenAIProvider(
        api_key=api_key,
        config={
            "default_model": "gpt-6-sol",
            "use_streaming": True,
            "max_retries": 0,
        },
    )
    try:
        request = ChatRequest(
            messages=[
                Message(role="user", content="Reply with exactly the word: pong")
            ],
            max_output_tokens=16,
        )
        response = await provider.complete(request)

        text_blocks = [b for b in response.content if isinstance(b, TextBlock)]
        assert len(text_blocks) >= 1
        assert len(text_blocks[0].text.strip()) > 0
        assert response.usage.output_tokens is not None
        assert response.usage.output_tokens > 0
    finally:
        await provider.close()


@pytest.mark.live
@pytest.mark.asyncio
async def test_live_luna_strict_function_tool_call() -> None:
    """Luna, a single strict tool with an enum-constrained parameter: the
    model must return a well-formed, JSON-parsed tool call."""
    api_key = _require_live()
    provider = OpenAIProvider(
        api_key=api_key,
        config={
            "default_model": "gpt-6-luna",
            "use_streaming": False,
            "max_retries": 0,
        },
    )
    try:
        tool = ToolSpec(
            name="get_weather",
            description="Get the current weather for a city",
            parameters={
                "type": "object",
                "properties": {
                    "city": {"type": "string", "enum": ["Paris", "Tokyo"]},
                },
                "required": ["city"],
                "additionalProperties": False,
            },
            strict=True,
        )
        request = ChatRequest(
            messages=[
                Message(
                    role="user",
                    content="Call get_weather for Paris. Use the tool, do not answer in text.",
                )
            ],
            tools=[tool],
            max_output_tokens=64,
        )
        response = await provider.complete(request)

        tool_blocks = [b for b in response.content if isinstance(b, ToolCallBlock)]
        assert len(tool_blocks) >= 1
        call = tool_blocks[0]
        assert call.name == "get_weather"
        assert isinstance(call.input, dict)
        assert call.input.get("city") in {"Paris", "Tokyo"}
    finally:
        await provider.close()


@pytest.mark.live
@pytest.mark.asyncio
async def test_live_unavailable_invented_gpt6_model_raises_not_found() -> None:
    """An invented GPT-6 model ID that does not exist classifies as
    NotFoundError, not a generic/opaque LLMError."""
    api_key = _require_live()
    provider = OpenAIProvider(
        api_key=api_key,
        config={"use_streaming": False, "max_retries": 0},
    )
    try:
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            max_output_tokens=16,
        )
        with pytest.raises(kernel_errors.NotFoundError):
            await provider.complete(request, model="gpt-6-vega-invented")
    finally:
        await provider.close()


@pytest.mark.live
@pytest.mark.asyncio
async def test_live_invalid_key_raises_authentication_error() -> None:
    """A syntactically-plausible but invalid API key classifies as
    AuthenticationError against the real API."""
    # Only requires the opt-in gate, not a real key -- this test supplies
    # its own (deliberately invalid) one.
    if os.environ.get("AMPLIFIER_GPT6_LIVE") != "1":
        pytest.skip("Live GPT-6 test requires AMPLIFIER_GPT6_LIVE=1")

    provider = OpenAIProvider(
        api_key="sk-invalid-not-a-real-key-0000000000000000000000",
        config={
            "default_model": "gpt-6-sol",
            "use_streaming": False,
            "max_retries": 0,
        },
    )
    try:
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            max_output_tokens=16,
        )
        with pytest.raises(kernel_errors.AuthenticationError):
            await provider.complete(request)
    finally:
        await provider.close()
