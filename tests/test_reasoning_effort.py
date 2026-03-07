"""Phase 2: reasoning_effort support tests.

Verifies the precedence chain:
  kwargs["reasoning"] > request.reasoning_effort > config default > None

And that reasoning_effort maps to the correct OpenAI reasoning param format.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from amplifier_core.message_models import ChatRequest, Message

from amplifier_module_provider_openai import OpenAIProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class DummyResponse:
    """Minimal response stub."""

    def __init__(self):
        self.output = [
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text="Hi")],
            )
        ]
        self.usage = SimpleNamespace(input_tokens=10, output_tokens=5)
        self.status = "completed"
        self.id = "resp_test"


def _make_provider(**config_overrides) -> OpenAIProvider:
    config = {"max_retries": 0, **config_overrides}
    provider = OpenAIProvider(api_key="test-key", config=config)
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())
    return provider


def _request_with_effort(effort: str | None) -> ChatRequest:
    return ChatRequest(
        messages=[Message(role="user", content="Hello")],
        reasoning_effort=effort,
    )


def _get_call_kwargs(provider: OpenAIProvider) -> dict:
    """Extract the kwargs from the last client.responses.create call."""
    return provider.client.responses.create.await_args_list[-1].kwargs


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_reasoning_effort_high():
    """reasoning_effort='high' -> reasoning={'effort': 'high', 'summary': ...}."""
    provider = _make_provider()
    asyncio.run(provider.complete(_request_with_effort("high")))

    kwargs = _get_call_kwargs(provider)
    assert "reasoning" in kwargs
    assert kwargs["reasoning"]["effort"] == "high"
    assert "summary" in kwargs["reasoning"]


def test_reasoning_effort_low():
    """reasoning_effort='low' -> reasoning={'effort': 'low', 'summary': ...}."""
    provider = _make_provider()
    asyncio.run(provider.complete(_request_with_effort("low")))

    kwargs = _get_call_kwargs(provider)
    assert "reasoning" in kwargs
    assert kwargs["reasoning"]["effort"] == "low"
    assert "summary" in kwargs["reasoning"]


def test_reasoning_effort_medium():
    """reasoning_effort='medium' -> reasoning={'effort': 'medium', 'summary': ...}."""
    provider = _make_provider()
    asyncio.run(provider.complete(_request_with_effort("medium")))

    kwargs = _get_call_kwargs(provider)
    assert "reasoning" in kwargs
    assert kwargs["reasoning"]["effort"] == "medium"
    assert "summary" in kwargs["reasoning"]


def test_reasoning_effort_none_no_reasoning_param():
    """reasoning_effort=None and no config -> no reasoning param sent.
    Uses a non-reasoning model since reasoning-capable models (gpt-5.*, o-series,
    codex) now auto-set reasoning={summary: 'auto'} for observability."""
    provider = _make_provider(default_model="gpt-4.1-mini")  # non-reasoning model
    asyncio.run(provider.complete(_request_with_effort(None)))

    kwargs = _get_call_kwargs(provider)
    assert "reasoning" not in kwargs


def test_kwargs_reasoning_overrides_request_effort():
    """kwargs['reasoning'] takes precedence over request.reasoning_effort."""
    provider = _make_provider()
    request = _request_with_effort("medium")

    # Pass kwargs reasoning that should override
    asyncio.run(
        provider.complete(request, reasoning={"effort": "high", "summary": "concise"})
    )

    kwargs = _get_call_kwargs(provider)
    assert kwargs["reasoning"]["effort"] == "high"
    assert kwargs["reasoning"]["summary"] == "concise"


def test_config_default_used_when_no_request_effort():
    """Config reasoning default is used when request.reasoning_effort is None."""
    provider = _make_provider(reasoning="low")
    asyncio.run(provider.complete(_request_with_effort(None)))

    kwargs = _get_call_kwargs(provider)
    assert kwargs["reasoning"]["effort"] == "low"


def test_request_effort_overrides_config_default():
    """request.reasoning_effort overrides config default."""
    provider = _make_provider(reasoning="low")
    asyncio.run(provider.complete(_request_with_effort("high")))

    kwargs = _get_call_kwargs(provider)
    assert kwargs["reasoning"]["effort"] == "high"


def test_reasoning_summary_from_config():
    """reasoning_summary config is used in the built reasoning param."""
    provider = _make_provider(reasoning_summary="concise")
    asyncio.run(provider.complete(_request_with_effort("high")))

    kwargs = _get_call_kwargs(provider)
    assert kwargs["reasoning"]["summary"] == "concise"


def test_reasoning_effort_with_extended_thinking():
    """reasoning_effort sets reasoning, extended_thinking doesn't override it."""
    provider = _make_provider()
    request = _request_with_effort("low")

    # extended_thinking=True but reasoning already set by reasoning_effort
    asyncio.run(provider.complete(request, extended_thinking=True))

    kwargs = _get_call_kwargs(provider)
    # reasoning_effort="low" already set reasoning, so it should be "low"
    assert kwargs["reasoning"]["effort"] == "low"


def test_reasoning_effort_xhigh():
    """reasoning_effort='xhigh' -> reasoning={'effort': 'xhigh', 'summary': ...}."""
    provider = _make_provider()
    asyncio.run(provider.complete(_request_with_effort("xhigh")))
    kwargs = _get_call_kwargs(provider)
    assert "reasoning" in kwargs
    assert kwargs["reasoning"]["effort"] == "xhigh"
    assert "summary" in kwargs["reasoning"]


def test_reasoning_effort_none_explicit():
    """reasoning_effort='none' explicitly set -> reasoning={'effort': 'none', 'summary': ...}.
    This is different from reasoning_effort=None (Python None = not set).
    GPT-5.4 uses 'none' as a string value meaning 'no reasoning'."""
    provider = _make_provider()
    asyncio.run(provider.complete(_request_with_effort("none")))
    kwargs = _get_call_kwargs(provider)
    assert "reasoning" in kwargs
    assert kwargs["reasoning"]["effort"] == "none"
    assert "summary" in kwargs["reasoning"]


def test_gpt54_without_effort_still_includes_encrypted_content():
    """GPT-5.4 with no explicit reasoning_effort -> include=[reasoning.encrypted_content] IS sent.

    GPT-5.4 is a reasoning-capable model (supports_reasoning=True) that CAN produce
    reasoning tokens even without explicit effort. Without include=[reasoning.encrypted_content],
    reasoning token content is lost when store=false (Amplifier's default), causing
    orphaned reasoning references (70 occurrences observed on test device).

    Regression test: the include-guard incorrectly gated on default_reasoning_effort
    (None for GPT-5.4), but reasoning-capable models CAN produce tokens even without
    explicit effort. The guard must use supports_reasoning (the capability flag) instead,
    matching the Anthropic provider's pattern of always preserving thinking content for
    capable models.
    """
    provider = _make_provider(default_model="gpt-5.4")
    asyncio.run(provider.complete(_request_with_effort(None)))
    kwargs = _get_call_kwargs(provider)
    assert "include" in kwargs, (
        "GPT-5.4 is reasoning-capable; include=[reasoning.encrypted_content] must be sent "
        "even without explicit reasoning_effort, to prevent silent reasoning token loss."
    )
    assert kwargs["include"] == ["reasoning.encrypted_content"]
