"""Tests for response chaining (previous_response_id) for reasoning models.

PR-B: Wire `store=True` + `previous_response_id` chaining so multi-turn reasoning
workloads get 40-80% better cache utilization via the Responses API.

Test design mirrors test_cache_params.py structure:
- _make_provider / DummyResponse / _captured_params helpers
- AsyncMock + asyncio.run() — no live API calls
"""

import asyncio
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import openai
import pytest
from amplifier_core import ModuleCoordinator, llm_errors as kernel_errors
from amplifier_core.message_models import (
    ChatRequest,
    Message,
    TextBlock,
    ThinkingBlock,
)
from httpx import Request as HttpxRequest
from httpx import Response as HttpxResponse

from amplifier_module_provider_openai import OpenAIProvider
from amplifier_module_provider_openai._constants import (
    METADATA_RESPONSE_ID,
    RESPONSE_CHAIN_INVALIDATED,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_provider(**config_overrides) -> OpenAIProvider:
    config = {"max_retries": 0, "use_streaming": False, **config_overrides}
    return OpenAIProvider(api_key="test-key", config=config)


class DummyResponse:
    """Minimal response stub — matches the shape _convert_to_chat_response() needs."""

    def __init__(self, response_id: str = "resp_test"):
        self.output = [
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text="Hi")],
            )
        ]
        self.usage = SimpleNamespace(input_tokens=1, output_tokens=1)
        self.status = "completed"
        self.id = response_id


def _captured_params(provider: OpenAIProvider) -> Any:
    """Return the kwargs dict passed to the mocked create() call."""
    mock = cast(AsyncMock, provider.client.responses.create)
    return mock.call_args.kwargs


def _all_calls(provider: OpenAIProvider) -> list[Any]:
    """Return all kwargs dicts from all calls to the mocked create()."""
    mock = cast(AsyncMock, provider.client.responses.create)
    return [call.kwargs for call in mock.call_args_list]


def _make_not_found_error(code: str = "response_not_found") -> openai.NotFoundError:
    """Create an openai.NotFoundError with the given error code body."""
    req = HttpxRequest("GET", "https://api.openai.com/v1/responses")
    resp = HttpxResponse(404, request=req)
    return openai.NotFoundError(
        message=f"response not found: {code}",
        response=resp,
        body={"error": {"code": code, "message": "Response not found"}},
    )


def _simple_request() -> ChatRequest:
    return ChatRequest(messages=[Message(role="user", content="Hello")])


def _request_with_prior_response_id(
    response_id: str = "resp_abc",
    prior_thinking: bool = False,
) -> ChatRequest:
    """Build a 2-turn ChatRequest where the assistant's metadata carries response_id.

    When prior_thinking=True, also attaches a ThinkingBlock with encrypted_content
    to test reinsertion-suppression logic.
    """
    if prior_thinking:
        # Assistant turn has both a ThinkingBlock (with encrypted state) and a text
        content: Any = [
            ThinkingBlock(
                thinking="some reasoning",
                content=["encrypted_blob_xyz", "rs_abc"],
            ),
            TextBlock(type="text", text="Hello!"),
        ]
    else:
        content = "Hello!"

    msgs = [
        Message(role="user", content="Hi"),
        Message(
            role="assistant",
            content=content,
            metadata={METADATA_RESPONSE_ID: response_id},
        ),
        Message(role="user", content="Follow-up"),
    ]
    return ChatRequest(messages=msgs)


# ---------------------------------------------------------------------------
# Streaming helpers (copied from test_streaming.py pattern)
# ---------------------------------------------------------------------------


class MockStreamContext:
    """Async context manager that yields a mock stream object."""

    def __init__(self, stream):
        self._stream = stream

    async def __aenter__(self):
        return self._stream

    async def __aexit__(self, *args):
        pass


class SuccessStream:
    """A stream that immediately returns the given response."""

    def __init__(self, response):
        self._response = SimpleNamespace(headers={})
        self._api_response = response

    async def get_final_response(self):
        return self._api_response


# ---------------------------------------------------------------------------
# FakeCoordinator for observability tests
# ---------------------------------------------------------------------------


class FakeHooks:
    """Records every (event_name, payload) pair emitted via hooks."""

    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    async def emit(self, name: str, payload: dict) -> None:
        self.events.append((name, payload))


class FakeCoordinator:
    """Minimal coordinator stub with hooks recording."""

    def __init__(self):
        self.hooks = FakeHooks()


# ---------------------------------------------------------------------------
# Tests — auto (default) behavior
# ---------------------------------------------------------------------------


def test_chain_auto_on_for_reasoning_model_first_turn():
    """Reasoning model + first turn → store=True, NO previous_response_id."""
    provider = _make_provider(default_model="gpt-5.5")
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())

    request = _simple_request()
    asyncio.run(provider.complete(request))

    params = _captured_params(provider)
    assert params["store"] is True, (
        f"gpt-5.5 first turn should have store=True, got store={params.get('store')}"
    )
    assert "previous_response_id" not in params, (
        f"First turn should have no previous_response_id, got {params.get('previous_response_id')}"
    )
    # No encrypted_content include on chaining path
    include = params.get("include", [])
    assert "reasoning.encrypted_content" not in include, (
        f"Chaining path must not request encrypted_content; got include={include}"
    )


def test_chain_auto_on_for_reasoning_model_second_turn():
    """Reasoning model + second turn → store=True, previous_response_id set."""
    provider = _make_provider(default_model="gpt-5.5")
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())

    request = _request_with_prior_response_id("resp_abc")
    asyncio.run(provider.complete(request))

    params = _captured_params(provider)
    assert params["store"] is True
    assert params.get("previous_response_id") == "resp_abc", (
        f"Second turn should forward previous_response_id, got {params.get('previous_response_id')}"
    )
    include = params.get("include", [])
    assert "reasoning.encrypted_content" not in include, (
        f"Chaining path must not request encrypted_content; got include={include}"
    )


def test_chain_auto_off_for_non_reasoning_model():
    """Non-reasoning model → no chaining, behavior unchanged from today."""
    provider = _make_provider(default_model="gpt-5-mini")
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())

    request = _simple_request()
    asyncio.run(provider.complete(request))

    params = _captured_params(provider)
    # Default enable_state=False, chain_active=False for gpt-5-mini
    assert params["store"] is False, (
        f"gpt-5-mini default should have store=False, got {params.get('store')}"
    )
    assert "previous_response_id" not in params, (
        "Non-reasoning model should not set previous_response_id"
    )


# ---------------------------------------------------------------------------
# Tests — explicit overrides
# ---------------------------------------------------------------------------


def test_chain_explicit_true_forces_on_for_non_reasoning_model():
    """enable_response_chaining=True overrides capability check."""
    provider = _make_provider(default_model="gpt-5-mini", enable_response_chaining=True)
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())

    # Include a prior response_id to verify it's forwarded
    request = _request_with_prior_response_id("resp_forced")
    asyncio.run(provider.complete(request))

    params = _captured_params(provider)
    assert params["store"] is True, (
        "Explicit enable_response_chaining=True should force store=True"
    )
    assert params.get("previous_response_id") == "resp_forced", (
        "Explicit chaining on non-reasoning model should forward previous_response_id"
    )


def test_chain_explicit_false_disables_for_reasoning_model():
    """enable_response_chaining=False (ZDR opt-out) → no chaining even on gpt-5.5."""
    provider = _make_provider(default_model="gpt-5.5", enable_response_chaining=False)
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())

    request = _request_with_prior_response_id("resp_should_not_appear")
    asyncio.run(provider.complete(request))

    params = _captured_params(provider)
    assert params["store"] is False, (
        f"ZDR opt-out should have store=False, got {params.get('store')}"
    )
    assert "previous_response_id" not in params, (
        "ZDR opt-out must not send previous_response_id"
    )
    # Stateless reasoning path: encrypted_content IS requested
    include = params.get("include", [])
    assert "reasoning.encrypted_content" in include, (
        f"Stateless path should request encrypted_content, got include={include}"
    )


def test_chain_kwarg_overrides_config():
    """Per-call kwarg wins over provider config."""
    # Provider has chaining disabled
    provider = _make_provider(default_model="gpt-5.5", enable_response_chaining=False)
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())

    request = _request_with_prior_response_id("resp_kwarg_override")
    # Per-call kwarg enables chaining
    asyncio.run(provider.complete(request, enable_response_chaining=True))

    params = _captured_params(provider)
    assert params["store"] is True, (
        "Per-call kwarg should override config chaining=False"
    )
    assert params.get("previous_response_id") == "resp_kwarg_override", (
        "Per-call kwarg override should forward previous_response_id"
    )


def test_chain_auto_string_normalized():
    """'auto' string in config is preserved; empty string normalizes to auto."""
    # Empty string → auto (resolved to True for gpt-5.5)
    provider_empty = _make_provider(
        default_model="gpt-5.5", enable_response_chaining=""
    )
    assert provider_empty.enable_response_chaining == "auto", (
        f"Empty string should normalize to 'auto', got {provider_empty.enable_response_chaining!r}"
    )

    # "auto" string → auto
    provider_auto = _make_provider(
        default_model="gpt-5.5", enable_response_chaining="auto"
    )
    assert provider_auto.enable_response_chaining == "auto", (
        f"'auto' string should remain 'auto', got {provider_auto.enable_response_chaining!r}"
    )

    # Both should enable chaining for gpt-5.5 (reasoning model)
    for label, prov in [("empty", provider_empty), ("auto", provider_auto)]:
        prov.client.responses.create = AsyncMock(return_value=DummyResponse())
        asyncio.run(prov.complete(_simple_request()))
        params = prov.client.responses.create.call_args.kwargs
        assert params["store"] is True, (
            f"Provider with enable_response_chaining={label!r} should enable chaining for gpt-5.5"
        )


# ---------------------------------------------------------------------------
# Tests — encrypted-content interaction
# ---------------------------------------------------------------------------


def test_encrypted_content_not_requested_when_chaining_active():
    """Q4: don't send include=reasoning.encrypted_content with chaining."""
    provider = _make_provider(default_model="gpt-5.5")  # auto → chain_active=True
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())

    asyncio.run(provider.complete(_simple_request()))

    params = _captured_params(provider)
    include = params.get("include", [])
    assert "reasoning.encrypted_content" not in include, (
        f"Chaining active: must NOT send include=reasoning.encrypted_content; got {include}"
    )


def test_encrypted_content_still_requested_when_chaining_off():
    """Backward compat: stateless reasoning path still requests it."""
    provider = _make_provider(default_model="gpt-5.5", enable_response_chaining=False)
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())

    asyncio.run(provider.complete(_simple_request()))

    params = _captured_params(provider)
    include = params.get("include", [])
    assert "reasoning.encrypted_content" in include, (
        f"Stateless gpt-5.5 path should request encrypted_content; got include={include}"
    )


def test_thinkingblock_not_reinserted_when_chaining_active():
    """ThinkingBlock with encrypted_content is NOT re-emitted into input array."""
    provider = _make_provider(default_model="gpt-5.5")  # auto → chain_active=True
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())

    request = _request_with_prior_response_id("resp_prev", prior_thinking=True)
    asyncio.run(provider.complete(request))

    params = _captured_params(provider)
    input_items = params.get("input", [])
    reasoning_items = [
        item
        for item in input_items
        if isinstance(item, dict) and item.get("type") == "reasoning"
    ]
    assert len(reasoning_items) == 0, (
        f"Chaining active: reasoning items must NOT be re-inserted into input; "
        f"found {len(reasoning_items)}: {reasoning_items}"
    )
    # Chain should still be set
    assert params.get("previous_response_id") == "resp_prev"


def test_thinkingblock_still_reinserted_when_chaining_off():
    """Regression: stateless path keeps the existing reinsertion behavior."""
    provider = _make_provider(default_model="gpt-5.5", enable_response_chaining=False)
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())

    request = _request_with_prior_response_id("resp_prev", prior_thinking=True)
    asyncio.run(provider.complete(request))

    params = _captured_params(provider)
    input_items = params.get("input", [])
    reasoning_items = [
        item
        for item in input_items
        if isinstance(item, dict) and item.get("type") == "reasoning"
    ]
    assert len(reasoning_items) >= 1, (
        f"Stateless path must re-insert reasoning item into input; got input={input_items}"
    )


# ---------------------------------------------------------------------------
# Tests — invalidation handling
# ---------------------------------------------------------------------------


def test_response_not_found_triggers_retry_without_chain():
    """404 + code=response_not_found → retry once without previous_response_id."""
    provider = _make_provider(default_model="gpt-5.5")
    provider.client.responses.create = AsyncMock(
        side_effect=[_make_not_found_error("response_not_found"), DummyResponse()]
    )

    request = _request_with_prior_response_id("resp_abc")
    # Must complete without raising
    result = asyncio.run(provider.complete(request))

    calls = _all_calls(provider)
    assert len(calls) == 2, (
        f"Expected exactly 2 API calls (invalidation retry), got {len(calls)}"
    )
    # First call had previous_response_id
    assert calls[0].get("previous_response_id") == "resp_abc", (
        f"First call should have previous_response_id=resp_abc, got {calls[0].get('previous_response_id')}"
    )
    # Second call (retry) did NOT have previous_response_id
    assert "previous_response_id" not in calls[1], (
        f"Retry call must NOT have previous_response_id; got {calls[1].get('previous_response_id')}"
    )
    # No exception bubbled
    assert result is not None


def test_response_chain_invalidated_event_emitted():
    """The retry path emits provider:response_chain_invalidated."""
    coordinator = FakeCoordinator()
    provider = _make_provider(default_model="gpt-5.5")
    provider.coordinator = cast(ModuleCoordinator, coordinator)
    provider.client.responses.create = AsyncMock(
        side_effect=[_make_not_found_error("response_not_found"), DummyResponse()]
    )

    request = _request_with_prior_response_id("resp_abc")
    asyncio.run(provider.complete(request))

    event_names = [name for name, _ in coordinator.hooks.events]
    assert RESPONSE_CHAIN_INVALIDATED in event_names, (
        f"Expected {RESPONSE_CHAIN_INVALIDATED!r} event; got events: {event_names}"
    )

    # Find the invalidation event payload
    payloads = [
        payload
        for name, payload in coordinator.hooks.events
        if name == RESPONSE_CHAIN_INVALIDATED
    ]
    assert len(payloads) == 1
    payload = payloads[0]
    assert payload["invalidated_id"] == "resp_abc", (
        f"Event payload should carry invalidated_id=resp_abc, got {payload}"
    )
    assert payload["error_code"] == "response_not_found", (
        f"Event payload should carry error_code=response_not_found, got {payload}"
    )


def test_response_not_found_does_not_recurse_indefinitely():
    """If the retry also fails, the second NotFoundError bubbles out.

    Termination is guaranteed because params.pop('previous_response_id') makes
    the second call unable to enter the chain-invalidation branch again.
    """
    provider = _make_provider(default_model="gpt-5.5")
    provider.client.responses.create = AsyncMock(
        side_effect=[
            _make_not_found_error("response_not_found"),
            _make_not_found_error("response_not_found"),
        ]
    )

    request = _request_with_prior_response_id("resp_abc")
    with pytest.raises(kernel_errors.NotFoundError):
        asyncio.run(provider.complete(request))

    # Exactly 2 API calls: initial attempt + one retry
    assert provider.client.responses.create.call_count == 2, (
        f"Expected 2 attempts (initial + one retry), got {provider.client.responses.create.call_count}"
    )


def test_unrelated_404_not_swallowed():
    """A 404 with a different error code is NOT treated as chain invalidation."""
    provider = _make_provider(default_model="gpt-5.5")
    provider.client.responses.create = AsyncMock(
        side_effect=[_make_not_found_error("model_not_found")]
    )

    request = _request_with_prior_response_id("resp_abc")
    with pytest.raises(kernel_errors.NotFoundError):
        asyncio.run(provider.complete(request))

    # Only 1 call — no retry was attempted
    assert provider.client.responses.create.call_count == 1, (
        f"Unrelated 404 must NOT trigger retry; got {provider.client.responses.create.call_count} calls"
    )


# ---------------------------------------------------------------------------
# Tests — streaming + chaining
# ---------------------------------------------------------------------------


def test_chaining_works_in_streaming_path():
    """params reach client.responses.stream() with previous_response_id set."""
    provider = _make_provider(default_model="gpt-5.5", use_streaming=True)

    dummy = DummyResponse(response_id="resp_stream_result")
    stream = SuccessStream(dummy)
    mock_stream_call = MagicMock(return_value=MockStreamContext(stream))
    provider.client.responses.stream = mock_stream_call

    request = _request_with_prior_response_id("resp_abc")
    asyncio.run(provider.complete(request))

    assert mock_stream_call.called, "responses.stream() was never called"
    stream_kwargs = mock_stream_call.call_args.kwargs
    assert stream_kwargs.get("previous_response_id") == "resp_abc", (
        f"Streaming call must include previous_response_id=resp_abc; "
        f"got {stream_kwargs.get('previous_response_id')}"
    )
    assert stream_kwargs.get("store") is True, (
        f"Streaming call must have store=True for chaining; got {stream_kwargs.get('store')}"
    )


def test_streaming_response_id_propagates_to_metadata():
    """ChatResponse.metadata['openai:response_id'] is set in streaming path.

    This verifies turn N+1 can chain from turn N via the standard metadata-on-message
    mechanism — the same path that _complete_chat_request() scans for previous_response_id.
    """
    provider = _make_provider(default_model="gpt-5.5", use_streaming=True)

    dummy = DummyResponse(response_id="resp_new_turn")
    stream = SuccessStream(dummy)
    mock_stream_call = MagicMock(return_value=MockStreamContext(stream))
    provider.client.responses.stream = mock_stream_call

    request = _simple_request()
    chat_response = asyncio.run(provider.complete(request))

    assert chat_response.metadata.get(METADATA_RESPONSE_ID) == "resp_new_turn", (
        f"Streaming ChatResponse must carry METADATA_RESPONSE_ID='resp_new_turn'; "
        f"got metadata={chat_response.metadata}"
    )
