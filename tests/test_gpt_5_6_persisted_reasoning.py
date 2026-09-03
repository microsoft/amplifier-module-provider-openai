"""Tests for reasoning.context (GPT-5.6 "persisted reasoning") passthrough.

Shape verified against the GPT-5.6 model guide (developers.openai.com,
fetched 2026-07-14): reasoning.context in {"auto", "current_turn", "all_turns"}.
"current_turn" trims rendered reasoning context on long agent loops -- the
documented mitigation for context_length_exceeded.

Forwarding is UNGATED: whenever the caller supplies `reasoning.context` in an
explicit `reasoning` dict, it is forwarded as-is (same stance as `mode` --
an explicit reasoning dict is a deliberate provider-specific override, and
the caller owns the consequences). The historical `enable_reasoning_context`
flag gate and the chain/store gate (both tied to the now-removed
`previous_response_id` chaining path) are gone -- the provider is
stateless-only. `_validate_reasoning_context` still rejects bad values.
"""

import asyncio
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from amplifier_core import llm_errors as kernel_errors
from amplifier_core.message_models import ChatRequest, Message

from amplifier_module_provider_openai import (
    OpenAIProvider,
    _validate_reasoning_context,
)


def _make_provider(**config_overrides) -> OpenAIProvider:
    config = {"max_retries": 0, "use_streaming": False, **config_overrides}
    return OpenAIProvider(api_key="test-key", config=config)


def _simple_request() -> ChatRequest:
    return ChatRequest(messages=[Message(role="user", content="Hello")])


class DummyResponse:
    def __init__(self):
        self.output = [
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text="Hi")],
            )
        ]
        self.usage = SimpleNamespace(input_tokens=1, output_tokens=1)
        self.status = "completed"
        self.id = "resp_test"


def _captured_params(provider: OpenAIProvider) -> Any:
    mock = cast(AsyncMock, provider.client.responses.create)
    return mock.call_args.kwargs


# ---------------------------------------------------------------------------
# reasoning.context validator (value-shape only; unaffected by the flag/chain
# gates, which apply only at the forwarding site).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ok",
    [
        None,
        "high",
        {},
        {"context": "auto"},
        {"context": "current_turn"},
        {"context": "all_turns"},
    ],
)
def test_validate_reasoning_context_accepts(ok):
    _validate_reasoning_context(ok)  # must not raise


@pytest.mark.parametrize(
    "bad", [{"context": "later"}, {"context": ""}, {"context": "turn"}]
)
def test_validate_reasoning_context_rejects(bad):
    with pytest.raises(kernel_errors.InvalidRequestError):
        _validate_reasoning_context(bad)


# ---------------------------------------------------------------------------
# reasoning.context passthrough into the API call -- flag + chain/store gated
# ---------------------------------------------------------------------------


def test_reasoning_context_forwarded_ungated():
    """An explicit reasoning.context value is forwarded unconditionally --
    no flag, no chain/store gate. The provider is stateless-only; an
    explicit `reasoning` dict is a deliberate override the caller owns."""
    provider = _make_provider(default_model="gpt-5.6-sol")
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())
    asyncio.run(
        provider.complete(
            _simple_request(),
            reasoning={"effort": "high", "context": "current_turn"},
        )
    )
    reasoning = _captured_params(provider)["reasoning"]
    assert reasoning["context"] == "current_turn"
    assert reasoning["effort"] == "high"


def test_reasoning_context_absent_when_not_set():
    provider = _make_provider(default_model="gpt-5.6-sol")
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())
    asyncio.run(provider.complete(_simple_request(), reasoning={"effort": "medium"}))
    assert "context" not in _captured_params(provider)["reasoning"]


def test_reasoning_context_forwarded_even_with_leftover_legacy_config():
    """Stale `enable_reasoning_context` config (recognized-but-inert) must
    not suppress forwarding -- the flag gate no longer exists."""
    provider = _make_provider(
        default_model="gpt-5.6-sol", enable_reasoning_context=False
    )
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())
    asyncio.run(
        provider.complete(
            _simple_request(),
            reasoning={"effort": "high", "context": "current_turn"},
        )
    )
    reasoning = _captured_params(provider)["reasoning"]
    assert reasoning["context"] == "current_turn"


def test_reasoning_context_forwarded_on_continuation():
    """reasoning.context must survive an incomplete->continuation sequence.

    It lives inside the `reasoning` dict, already forwarded via the existing
    `if "reasoning" in params` continuation-forward line.
    """
    provider = _make_provider(default_model="gpt-5.6-sol")
    incomplete_resp = SimpleNamespace(
        status="incomplete", id="resp_incomplete", output=[], incomplete_details=None
    )
    provider.client.responses.create = AsyncMock(
        side_effect=[incomplete_resp, DummyResponse()]
    )
    asyncio.run(
        provider.complete(
            _simple_request(),
            reasoning={"effort": "high", "context": "current_turn"},
        )
    )

    calls = provider.client.responses.create.call_args_list
    assert len(calls) == 2
    for call in calls:
        assert call.kwargs["reasoning"]["context"] == "current_turn"
