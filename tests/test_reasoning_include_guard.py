"""Regression tests for reasoning.encrypted_content include guard.

When store=false, the provider should only send
  include: ["reasoning.encrypted_content"]
if reasoning is actually active in the request. Non-reasoning models
(gpt-4.1-mini, gpt-4o, etc.) must NOT receive this parameter.

See: upstream-fix-2-provider-openai-reasoning-include.md
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


def _get_call_kwargs(provider: OpenAIProvider) -> dict:
    """Extract the kwargs from the last client.responses.create call."""
    return provider.client.responses.create.await_args_list[-1].kwargs


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_no_include_for_non_reasoning_model():
    """When reasoning is NOT active on a non-reasoning model, include parameter
    should NOT be sent, even when store=false (the default).
    Note: must use a non-reasoning model — reasoning-capable models (gpt-5.*,
    o-series, codex) now get include automatically via model capability detection."""
    provider = _make_provider(default_model="gpt-4.1-mini")  # non-reasoning model
    request = ChatRequest(
        messages=[Message(role="user", content="Hello")],
        reasoning_effort=None,  # No reasoning requested
    )
    asyncio.run(provider.complete(request))

    kwargs = _get_call_kwargs(provider)
    assert "include" not in kwargs, (
        "Non-reasoning request should NOT have 'include' parameter, "
        f"but got include={kwargs.get('include')}"
    )


def test_include_sent_when_reasoning_active_store_false():
    """When reasoning IS active and store=false, include parameter SHOULD be sent."""
    provider = _make_provider()  # store defaults to false
    request = ChatRequest(
        messages=[Message(role="user", content="Hello")],
        reasoning_effort="high",  # Reasoning IS active
    )
    asyncio.run(provider.complete(request))

    kwargs = _get_call_kwargs(provider)
    assert "include" in kwargs, (
        "Reasoning request with store=false should have 'include' parameter"
    )
    assert kwargs["include"] == ["reasoning.encrypted_content"]


def test_no_include_when_store_true():
    """When store=true, include parameter should NOT be sent (not needed)."""
    provider = _make_provider(enable_state=True)
    request = ChatRequest(
        messages=[Message(role="user", content="Hello")],
        reasoning_effort="high",  # Reasoning active, but store=true
    )
    asyncio.run(provider.complete(request))

    kwargs = _get_call_kwargs(provider)
    assert "include" not in kwargs, (
        "When store=true, 'include' should NOT be sent (stored responses "
        "don't need encrypted_content)"
    )


def test_no_include_when_explicit_none_effort_on_default_reasoning_model():
    """Regression: explicit reasoning_effort='none' must override default_reasoning_effort.

    gpt-5.2-codex has default_reasoning_effort='medium', so the model *would*
    reason by default.  But when the caller explicitly sets effort='none', the
    include-guard MUST NOT send include=[reasoning.encrypted_content].

    Bug: the old second-clause check
        (self._model_may_reason(model) and caps.default_reasoning_effort is not None)
    evaluated True for gpt-5.2-codex regardless of the explicit 'none' effort,
    causing an incorrect 'include' parameter to be sent.
    """
    provider = _make_provider(default_model="gpt-5.2-codex")
    request = ChatRequest(
        messages=[Message(role="user", content="Hello")],
        reasoning_effort="none",  # Explicitly disable reasoning
    )
    asyncio.run(provider.complete(request))

    kwargs = _get_call_kwargs(provider)
    assert "include" not in kwargs, (
        "Explicit reasoning_effort='none' on a reason-by-default model should NOT "
        f"have 'include' parameter, but got include={kwargs.get('include')}"
    )


def test_reasoning_capable_model_always_gets_include_param():
    """Regression test for reasoning token loss bug — GPT-5.4 sub-sessions lost
    reasoning tokens because include param was not sent.

    GPT-5.4 is reasoning-capable (supports_reasoning=True) but has
    default_reasoning_effort=None. The old include-guard gated on
    ``default_reasoning_effort is not None``, which evaluated False for GPT-5.4,
    so include=[reasoning.encrypted_content] was never sent.  This caused 70
    orphaned reasoning references ("Reasoning IDs in metadata but
    encrypted_content unavailable") across 4 GPT-5.4 sub-sessions on a real
    test device.

    The fix: gate on caps.supports_reasoning (the capability flag) rather than
    default_reasoning_effort, matching the Anthropic provider's pattern of
    always preserving thinking content for capable models.
    """
    provider = _make_provider(default_model="gpt-5.4")  # enable_state defaults to false
    request = ChatRequest(
        messages=[Message(role="user", content="Hello")],
        # No reasoning_effort set — mimics a delegated sub-session with default config
    )
    asyncio.run(provider.complete(request))

    kwargs = _get_call_kwargs(provider)
    assert "include" in kwargs, (
        "GPT-5.4 reasoning-capable model with no explicit effort must have 'include' parameter "
        "to prevent silent reasoning token loss (70 occurrences observed on test device)"
    )
    assert kwargs["include"] == ["reasoning.encrypted_content"]


def test_include_with_reasoning_effort_low_store_false():
    """reasoning_effort='low' with store=false should include encrypted_content."""
    provider = _make_provider()
    request = ChatRequest(
        messages=[Message(role="user", content="Hello")],
        reasoning_effort="low",
    )
    asyncio.run(provider.complete(request))

    kwargs = _get_call_kwargs(provider)
    assert "include" in kwargs, (
        "reasoning_effort='low' with store=false should have 'include'"
    )
    assert kwargs["include"] == ["reasoning.encrypted_content"]
