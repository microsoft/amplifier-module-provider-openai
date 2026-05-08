"""Tests for cache-friendly default values (PR-A: cache-friendly defaults).

Verifies that:
  - prompt_cache_retention defaults to "24h" (DEFAULT_PROMPT_CACHE_RETENTION)
  - truncation defaults to None (omitted from the API call)
  - per-call kwarg overrides and explicit-None opt-outs work correctly
  - the _drop_unsupported_24h_retention() helper gates the new default
  - continuation calls inherit the default "24h" retention
"""

import asyncio
import logging
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, patch

from amplifier_core.message_models import ChatRequest, Message

from amplifier_module_provider_openai import OpenAIProvider
from amplifier_module_provider_openai._capabilities import ModelCapabilities


# ---------------------------------------------------------------------------
# Helpers (mirrors test_cache_params.py pattern)
# ---------------------------------------------------------------------------


def _make_provider(**config_overrides) -> OpenAIProvider:
    config = {"max_retries": 0, "use_streaming": False, **config_overrides}
    return OpenAIProvider(api_key="test-key", config=config)


def _simple_request() -> ChatRequest:
    return ChatRequest(messages=[Message(role="user", content="Hello")])


class DummyResponse:
    """Minimal response stub."""

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
# Tests — prompt_cache_retention new default
# ---------------------------------------------------------------------------


def test_default_retention_is_24h():
    """No config → prompt_cache_retention defaults to '24h'."""
    provider = _make_provider()
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())
    asyncio.run(provider.complete(_simple_request()))
    assert _captured_params(provider)["prompt_cache_retention"] == "24h"


def test_explicit_none_retention_omits_field():
    """config={"prompt_cache_retention": None} → field absent from captured params."""
    provider = _make_provider(prompt_cache_retention=None)
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())
    asyncio.run(provider.complete(_simple_request()))
    assert "prompt_cache_retention" not in _captured_params(provider)


def test_explicit_empty_string_retention_omits_field():
    """config={"prompt_cache_retention": ""} → field absent (empty-string coercion)."""
    provider = _make_provider(prompt_cache_retention="")
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())
    asyncio.run(provider.complete(_simple_request()))
    assert "prompt_cache_retention" not in _captured_params(provider)


def test_explicit_in_memory_retention_passes_through_for_gpt_5_4():
    """config in_memory + gpt-5.4 → 'in_memory' passes through.

    Regression guard: the default-24h change must not break the existing
    in_memory drop path. gpt-5.4 supports both retention modes; an explicit
    'in_memory' config must still reach the API unchanged.
    """
    provider = _make_provider(
        prompt_cache_retention="in_memory", default_model="gpt-5.4"
    )
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())
    asyncio.run(provider.complete(_simple_request()))
    assert _captured_params(provider)["prompt_cache_retention"] == "in_memory"


def test_default_24h_dropped_for_24h_rejecting_model(caplog):
    """Model with supports_24h_retention=False → default '24h' is dropped with WARNING."""
    fake_caps = ModelCapabilities(
        family="hypothetical",
        supports_in_memory_retention=True,
        supports_24h_retention=False,
    )
    caplog.set_level(logging.WARNING, logger="amplifier_module_provider_openai")

    with patch(
        "amplifier_module_provider_openai.get_capabilities",
        return_value=fake_caps,
    ):
        provider = _make_provider(default_model="hypothetical-model")
        provider.client.responses.create = AsyncMock(return_value=DummyResponse())
        asyncio.run(provider.complete(_simple_request()))

    params = _captured_params(provider)
    assert "prompt_cache_retention" not in params
    assert any(
        "24h" in r.message and "hypothetical-model" in r.message
        for r in caplog.records
        if r.levelno == logging.WARNING
    )


# ---------------------------------------------------------------------------
# Tests — truncation new default
# ---------------------------------------------------------------------------


def test_default_truncation_omitted():
    """No config → 'truncation' key absent from captured params."""
    provider = _make_provider()
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())
    asyncio.run(provider.complete(_simple_request()))
    assert "truncation" not in _captured_params(provider)


def test_truncation_auto_opt_in_passes_through():
    """config={"truncation": "auto"} → captured params has truncation == 'auto'."""
    provider = _make_provider(truncation="auto")
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())
    asyncio.run(provider.complete(_simple_request()))
    assert _captured_params(provider)["truncation"] == "auto"


def test_truncation_kwarg_overrides_default_none():
    """Default config (truncation=None) + call kwarg 'auto' → truncation sent.

    Pins the §2.5 restructure: the old `if self.truncation:` short-circuit
    would never reach the kwargs.get() path when self.truncation is None,
    silently ignoring the per-call override.
    """
    provider = _make_provider()  # truncation defaults to None
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())
    asyncio.run(provider.complete(_simple_request(), truncation="auto"))
    assert _captured_params(provider)["truncation"] == "auto"


def test_truncation_kwarg_none_overrides_config_auto():
    """config truncation='auto', call kwarg truncation=None → field omitted."""
    provider = _make_provider(truncation="auto")
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())
    asyncio.run(provider.complete(_simple_request(), truncation=None))
    assert "truncation" not in _captured_params(provider)


# ---------------------------------------------------------------------------
# Tests — continuation inherits default "24h"
# ---------------------------------------------------------------------------


def test_continuation_inherits_default_24h():
    """Continuation calls carry prompt_cache_retention='24h' from the new default.

    No explicit retention config — relies entirely on DEFAULT_PROMPT_CACHE_RETENTION.
    Both the initial call and the continuation call must include the field so the
    continuation lands on the same cache shard.
    """
    provider = _make_provider(default_model="gpt-5.4")  # no explicit retention

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
    for call in calls:
        assert call.kwargs.get("prompt_cache_retention") == "24h"
