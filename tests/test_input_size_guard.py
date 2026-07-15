"""Tests for the advisory pre-flight input-size guard (Gap I).

The guard is a cheap chars/4 heuristic that logs a WARNING when the estimated
input size exceeds a model's context window. It NEVER blocks or raises --
this is diagnosability for the opaque `context_length_exceeded` API error
(Error 2), not a hard limit.
"""

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

from amplifier_core.message_models import ChatRequest, Message

from amplifier_module_provider_openai import (
    OpenAIProvider,
    _estimate_input_tokens,
)


def _make_provider(**config_overrides) -> OpenAIProvider:
    config = {"max_retries": 0, "use_streaming": False, **config_overrides}
    return OpenAIProvider(api_key="test-key", config=config)


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


# ---------------------------------------------------------------------------
# _estimate_input_tokens sanity
# ---------------------------------------------------------------------------


def test_estimate_input_tokens_monotonic():
    small = [{"role": "user", "content": "hi"}]
    large = [{"role": "user", "content": "hi" * 100_000}]
    assert _estimate_input_tokens(large) > _estimate_input_tokens(small)


# ---------------------------------------------------------------------------
# Warning behavior -- advisory only, never blocks
# ---------------------------------------------------------------------------


def test_warn_when_input_exceeds_window(caplog):
    # gpt-5-mini has a 128K context window (_capabilities.py). Build an input
    # whose serialized length comfortably exceeds 4 * 128_000 chars.
    huge_text = "x" * (4 * 128_000 * 2)
    provider = _make_provider(default_model="gpt-5-mini")
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())

    request = ChatRequest(messages=[Message(role="user", content=huge_text)])

    with caplog.at_level(logging.WARNING):
        result = asyncio.run(provider.complete(request))

    # Never blocks: complete() still succeeds.
    assert result is not None

    warning_messages = [
        rec.message for rec in caplog.records if rec.levelno == logging.WARNING
    ]
    assert any("context_length_exceeded" in msg for msg in warning_messages), (
        f"Expected a context_length_exceeded advisory warning; got: {warning_messages}"
    )


def test_no_warn_under_window(caplog):
    provider = _make_provider(default_model="gpt-5-mini")
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())

    request = ChatRequest(messages=[Message(role="user", content="hello")])

    with caplog.at_level(logging.WARNING):
        asyncio.run(provider.complete(request))

    warning_messages = [
        rec.message for rec in caplog.records if rec.levelno == logging.WARNING
    ]
    assert not any("context_length_exceeded" in msg for msg in warning_messages)
