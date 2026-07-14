"""Phase-2 tests: gpt-5.6 reasoning.mode ("pro") and prompt_cache_options.

Shapes verified live against gpt-5.6-sol on 2026-07-14:
- reasoning.mode in {"standard", "pro"} ("pro" = extended internal reasoning).
- prompt_cache_options {"mode": "implicit"|"explicit", "ttl": "30m"}, which
  COEXISTS with prompt_cache_retention (both are echoed together -- it is NOT a
  replacement/deprecation of prompt_cache_retention).
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
    _validate_prompt_cache_options,
    _validate_reasoning_mode,
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
# reasoning.mode validator
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ok", [None, "high", {}, {"mode": "standard"}, {"mode": "pro"}]
)
def test_validate_reasoning_mode_accepts(ok):
    _validate_reasoning_mode(ok)  # must not raise


@pytest.mark.parametrize("bad", [{"mode": "turbo"}, {"mode": "ultra"}, {"mode": ""}])
def test_validate_reasoning_mode_rejects(bad):
    with pytest.raises(kernel_errors.InvalidRequestError):
        _validate_reasoning_mode(bad)


# ---------------------------------------------------------------------------
# prompt_cache_options validator
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ok",
    [
        {"mode": "implicit"},
        {"mode": "explicit"},
        {"ttl": "30m"},
        {"mode": "explicit", "ttl": "30m"},
        {},
    ],
)
def test_validate_prompt_cache_options_accepts(ok):
    _validate_prompt_cache_options(ok)  # must not raise


@pytest.mark.parametrize("bad", [{"mode": "auto"}, {"mode": "zzz"}, "explicit", 123])
def test_validate_prompt_cache_options_rejects(bad):
    with pytest.raises(kernel_errors.InvalidRequestError):
        _validate_prompt_cache_options(bad)


# ---------------------------------------------------------------------------
# reasoning.mode passthrough into the API call
# ---------------------------------------------------------------------------


def test_reasoning_mode_pro_forwarded():
    provider = _make_provider(default_model="gpt-5.6-sol")
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())
    asyncio.run(
        provider.complete(
            _simple_request(), reasoning={"effort": "high", "mode": "pro"}
        )
    )
    reasoning = _captured_params(provider)["reasoning"]
    assert reasoning["mode"] == "pro"
    assert reasoning["effort"] == "high"


def test_reasoning_mode_absent_when_not_set():
    provider = _make_provider(default_model="gpt-5.6-sol")
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())
    asyncio.run(provider.complete(_simple_request(), reasoning={"effort": "medium"}))
    assert "mode" not in _captured_params(provider)["reasoning"]


# ---------------------------------------------------------------------------
# prompt_cache_options passthrough + coexistence with retention
# ---------------------------------------------------------------------------


def test_prompt_cache_options_forwarded_from_config():
    provider = _make_provider(
        default_model="gpt-5.6-sol", prompt_cache_options={"mode": "explicit"}
    )
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())
    asyncio.run(provider.complete(_simple_request()))
    params = _captured_params(provider)
    assert params["prompt_cache_options"] == {"mode": "explicit"}
    # Coexistence: the default "24h" retention is still sent alongside it.
    assert params["prompt_cache_retention"] == "24h"


def test_prompt_cache_options_omitted_when_none():
    provider = _make_provider(default_model="gpt-5.6-sol")
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())
    asyncio.run(provider.complete(_simple_request()))
    assert "prompt_cache_options" not in _captured_params(provider)


def test_prompt_cache_options_kwarg_overrides_config():
    provider = _make_provider(
        default_model="gpt-5.6-sol", prompt_cache_options={"mode": "implicit"}
    )
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())
    asyncio.run(
        provider.complete(_simple_request(), prompt_cache_options={"mode": "explicit"})
    )
    assert _captured_params(provider)["prompt_cache_options"] == {"mode": "explicit"}
