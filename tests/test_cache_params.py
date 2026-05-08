"""Tests for prompt-caching hint parameters.

Verifies that prompt_cache_key, prompt_cache_retention, and safety_identifier
are emitted (or suppressed) correctly by _complete_chat_request() and that
continuation calls inherit the same cache params.
"""

import asyncio
import logging
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

from amplifier_core.message_models import ChatRequest, Message

from amplifier_module_provider_openai import OpenAIProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_provider(**config_overrides) -> OpenAIProvider:
    config = {"max_retries": 0, "use_streaming": False, **config_overrides}
    return OpenAIProvider(api_key="test-key", config=config)


def _simple_request() -> ChatRequest:
    return ChatRequest(messages=[Message(role="user", content="Hello")])


class DummyResponse:
    """Minimal response stub (copied from test_usage_fields.py pattern)."""

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
    """Return the kwargs dict passed to the mocked create() call."""
    mock = cast(AsyncMock, provider.client.responses.create)
    return mock.call_args.kwargs


# ---------------------------------------------------------------------------
# Tests — prompt_cache_key
# ---------------------------------------------------------------------------


def test_prompt_cache_key_sent_when_configured():
    """prompt_cache_key is forwarded to the API when set in config."""
    provider = _make_provider(prompt_cache_key="conv-123")
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())
    asyncio.run(provider.complete(_simple_request()))
    assert _captured_params(provider)["prompt_cache_key"] == "conv-123"


def test_prompt_cache_key_omitted_when_none():
    """prompt_cache_key is absent from the API call when not configured."""
    provider = _make_provider()
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())
    asyncio.run(provider.complete(_simple_request()))
    assert "prompt_cache_key" not in _captured_params(provider)


def test_prompt_cache_key_kwarg_overrides_config():
    """Per-call kwarg overrides the config-level prompt_cache_key."""
    provider = _make_provider(prompt_cache_key="default-key")
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())
    asyncio.run(provider.complete(_simple_request(), prompt_cache_key="override"))
    assert _captured_params(provider)["prompt_cache_key"] == "override"


def test_prompt_cache_key_empty_string_treated_as_none():
    """An empty string in config is coerced to None (field not sent)."""
    provider = _make_provider(prompt_cache_key="")
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())
    asyncio.run(provider.complete(_simple_request()))
    assert "prompt_cache_key" not in _captured_params(provider)


# ---------------------------------------------------------------------------
# Tests — safety_identifier
# ---------------------------------------------------------------------------


def test_safety_identifier_sent_when_configured():
    """safety_identifier is forwarded to the API when set in config."""
    provider = _make_provider(safety_identifier="user_42")
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())
    asyncio.run(provider.complete(_simple_request()))
    assert _captured_params(provider)["safety_identifier"] == "user_42"


def test_safety_identifier_omitted_when_none():
    """safety_identifier is absent from the API call when not configured."""
    provider = _make_provider()
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())
    asyncio.run(provider.complete(_simple_request()))
    assert "safety_identifier" not in _captured_params(provider)


def test_safety_identifier_kwarg_overrides_config():
    """Per-call kwarg overrides the config-level safety_identifier."""
    provider = _make_provider(safety_identifier="config-user")
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())
    asyncio.run(provider.complete(_simple_request(), safety_identifier="kwarg-user"))
    assert _captured_params(provider)["safety_identifier"] == "kwarg-user"


# ---------------------------------------------------------------------------
# Tests — prompt_cache_retention
# ---------------------------------------------------------------------------


def test_prompt_cache_retention_24h_sent_for_gpt_5_4():
    """prompt_cache_retention='24h' passes through for gpt-5.4."""
    provider = _make_provider(prompt_cache_retention="24h", default_model="gpt-5.4")
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())
    asyncio.run(provider.complete(_simple_request()))
    assert _captured_params(provider)["prompt_cache_retention"] == "24h"


def test_prompt_cache_retention_in_memory_sent_for_gpt_5_4():
    """prompt_cache_retention='in_memory' passes through for gpt-5.4 (supported)."""
    provider = _make_provider(
        prompt_cache_retention="in_memory", default_model="gpt-5.4"
    )
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())
    asyncio.run(provider.complete(_simple_request()))
    assert _captured_params(provider)["prompt_cache_retention"] == "in_memory"


def test_prompt_cache_retention_in_memory_dropped_for_gpt_5_5(caplog):
    """prompt_cache_retention='in_memory' is dropped with a warning for gpt-5.5."""
    caplog.set_level(logging.WARNING, logger="amplifier_module_provider_openai")
    provider = _make_provider(
        prompt_cache_retention="in_memory", default_model="gpt-5.5"
    )
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())
    asyncio.run(provider.complete(_simple_request()))
    params = _captured_params(provider)
    assert "prompt_cache_retention" not in params
    assert any(
        "in_memory" in r.message and "gpt-5.5" in r.message
        for r in caplog.records
        if r.levelno == logging.WARNING
    )


def test_prompt_cache_retention_24h_passes_through_for_gpt_4o():
    """prompt_cache_retention='24h' passes through for gpt-4o (unknown family)."""
    provider = _make_provider(prompt_cache_retention="24h", default_model="gpt-4o")
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())
    asyncio.run(provider.complete(_simple_request()))
    assert _captured_params(provider)["prompt_cache_retention"] == "24h"


def test_prompt_cache_retention_kwarg_overrides_config():
    """Per-call kwarg overrides the config-level prompt_cache_retention."""
    provider = _make_provider(
        prompt_cache_retention="in_memory", default_model="gpt-5.4"
    )
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())
    asyncio.run(provider.complete(_simple_request(), prompt_cache_retention="24h"))
    assert _captured_params(provider)["prompt_cache_retention"] == "24h"


# ---------------------------------------------------------------------------
# Tests — combined / edge cases
# ---------------------------------------------------------------------------


def test_all_three_omitted_by_default():
    """None of the three cache params appear when not configured."""
    provider = _make_provider()
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())
    asyncio.run(provider.complete(_simple_request()))
    params = _captured_params(provider)
    assert "prompt_cache_key" not in params
    assert "prompt_cache_retention" not in params
    assert "safety_identifier" not in params


def test_continuation_inherits_cache_params():
    """All three cache params are forwarded to continuation calls.

    Prime an incomplete→completed two-call sequence and assert that both
    create() calls received all three cache hint parameters. Covers each
    of the three `if "X" in params:` lines in the continuation block —
    if any of those lines is accidentally removed, this test catches it.
    """
    provider = _make_provider(
        prompt_cache_key="session-key",
        prompt_cache_retention="24h",
        safety_identifier="user-42",
        default_model="gpt-5.4",  # supports both retention values
    )

    incomplete_resp = SimpleNamespace(
        status="incomplete",
        id="resp_incomplete",
        output=[],
        incomplete_details=None,
    )
    completed_resp = DummyResponse()

    provider.client.responses.create = AsyncMock(
        side_effect=[incomplete_resp, completed_resp]
    )
    asyncio.run(provider.complete(_simple_request()))

    calls = provider.client.responses.create.call_args_list
    assert len(calls) == 2
    # Both the initial call and the continuation call must carry all three
    # cache hint parameters so the continuation lands on the same cache shard.
    for call in calls:
        assert call.kwargs.get("prompt_cache_key") == "session-key"
        assert call.kwargs.get("prompt_cache_retention") == "24h"
        assert call.kwargs.get("safety_identifier") == "user-42"
