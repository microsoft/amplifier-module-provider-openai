"""Tests for context_management (GPT-5.6 server-side compaction) passthrough.

Shape verified against the GPT-5.6 model guide (developers.openai.com,
fetched 2026-07-14):
    context_management = [{"type": "compaction", "compact_threshold": N}]

This is a direct, opt-in mitigation for context_length_exceeded (Error 2):
the server compacts context instead of hard-erroring.
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
    _validate_context_management,
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
# context_management validator
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [{"type": "compaction"}, "x", 123])
def test_validate_context_management_rejects_non_list(bad):
    with pytest.raises(kernel_errors.InvalidRequestError):
        _validate_context_management(bad)


@pytest.mark.parametrize(
    "ok",
    [
        [],
        [{"type": "compaction", "compact_threshold": 200_000}],
    ],
)
def test_validate_context_management_accepts_list(ok):
    _validate_context_management(ok)  # must not raise


# ---------------------------------------------------------------------------
# context_management passthrough into the API call
# ---------------------------------------------------------------------------


def test_context_management_from_config():
    directive = [{"type": "compaction", "compact_threshold": 200_000}]
    provider = _make_provider(default_model="gpt-5.6-sol", context_management=directive)
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())
    asyncio.run(provider.complete(_simple_request()))
    assert _captured_params(provider)["context_management"] == directive


def test_context_management_kwarg_overrides_config():
    config_directive = [{"type": "compaction", "compact_threshold": 100_000}]
    kwarg_directive = [{"type": "compaction", "compact_threshold": 300_000}]
    provider = _make_provider(
        default_model="gpt-5.6-sol", context_management=config_directive
    )
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())
    asyncio.run(
        provider.complete(_simple_request(), context_management=kwarg_directive)
    )
    assert _captured_params(provider)["context_management"] == kwarg_directive


def test_context_management_omitted_when_none():
    provider = _make_provider(default_model="gpt-5.6-sol")
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())
    asyncio.run(provider.complete(_simple_request()))
    assert "context_management" not in _captured_params(provider)


def test_context_management_forwarded_on_continuation():
    directive = [{"type": "compaction", "compact_threshold": 200_000}]
    provider = _make_provider(default_model="gpt-5.6-sol", context_management=directive)
    incomplete_resp = SimpleNamespace(
        status="incomplete", id="resp_incomplete", output=[], incomplete_details=None
    )
    provider.client.responses.create = AsyncMock(
        side_effect=[incomplete_resp, DummyResponse()]
    )
    asyncio.run(provider.complete(_simple_request()))

    calls = provider.client.responses.create.call_args_list
    assert len(calls) == 2
    for call in calls:
        assert call.kwargs.get("context_management") == directive
