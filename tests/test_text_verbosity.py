"""Tests for text.verbosity (GPT-5.6) passthrough.

Shape verified against the GPT-5.6 model guide (developers.openai.com,
fetched 2026-07-14): text.verbosity in {"low", "medium", "high"}.
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
    _validate_text_verbosity,
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
# text.verbosity validator
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ok", [None, "low", "medium", "high"])
def test_validate_text_verbosity_accepts(ok):
    _validate_text_verbosity(ok)  # must not raise


@pytest.mark.parametrize("bad", ["lowest", "", "verbose", 3])
def test_validate_text_verbosity_rejects(bad):
    with pytest.raises(kernel_errors.InvalidRequestError):
        _validate_text_verbosity(bad)


# ---------------------------------------------------------------------------
# text.verbosity passthrough into the API call
# ---------------------------------------------------------------------------


def test_text_verbosity_from_config():
    provider = _make_provider(default_model="gpt-5.6-sol", text_verbosity="low")
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())
    asyncio.run(provider.complete(_simple_request()))
    assert _captured_params(provider)["text"] == {"verbosity": "low"}


def test_text_verbosity_kwarg_overrides_config():
    provider = _make_provider(default_model="gpt-5.6-sol", text_verbosity="low")
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())
    asyncio.run(provider.complete(_simple_request(), text_verbosity="high"))
    assert _captured_params(provider)["text"] == {"verbosity": "high"}


def test_text_verbosity_omitted_when_none():
    provider = _make_provider(default_model="gpt-5.6-sol")
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())
    asyncio.run(provider.complete(_simple_request()))
    assert "text" not in _captured_params(provider)


def test_text_verbosity_forwarded_on_continuation():
    provider = _make_provider(default_model="gpt-5.6-sol", text_verbosity="medium")
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
        assert call.kwargs.get("text") == {"verbosity": "medium"}


# ---------------------------------------------------------------------------
# ConfigField presence (mirrors test_long_context_reporting.py pattern)
# ---------------------------------------------------------------------------


class TestConfigFieldPresent:
    def test_text_verbosity_is_settings_only_not_a_config_field(self):
        """Config-surface V2 reduced the wizard to 4 fields; text_verbosity
        is settings-only now (the config key still works exactly as
        before -- see test_text_verbosity_forwarded_when_set below)."""
        provider = _make_provider()
        info = provider.get_info()
        field = next((f for f in info.config_fields if f.id == "text_verbosity"), None)
        assert field is None
