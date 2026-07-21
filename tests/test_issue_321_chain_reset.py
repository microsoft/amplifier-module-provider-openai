"""Issue #321: the OpenAI provider must stop rebuilding the pre-compaction
server-side context via previous_response_id after the local context is
compacted (or on a context_length_exceeded overflow).

Two behaviours are verified here:

R1 -- On a compaction event the provider drops previous_response_id on the next
       request, so OpenAI rebuilds a fresh prefix from the compacted transcript
       instead of chaining from the (large) pre-compaction response.

R2 -- On a context_length_exceeded error while a chain is active, the provider
       breaks the chain once and retries with the full compacted transcript.
       This self-heals the resume path, where a fresh process re-lifts a stale
       on-disk response_id before any compaction event fires.

Harness mirrors test_response_chaining.py (AsyncMock client, asyncio.run(),
no live API calls).
"""

import asyncio
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import openai
import pytest
from amplifier_core import llm_errors as kernel_errors
from amplifier_core.message_models import ChatRequest, Message
from httpx import Request as HttpxRequest
from httpx import Response as HttpxResponse

from amplifier_module_provider_openai import OpenAIProvider, mount
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
    """Minimal response stub matching what _convert_to_chat_response() needs."""

    def __init__(self, response_id: str = "resp_new"):
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
    mock = cast(AsyncMock, provider.client.responses.create)
    return mock.call_args.kwargs


def _all_calls(provider: OpenAIProvider) -> list[Any]:
    mock = cast(AsyncMock, provider.client.responses.create)
    return [call.kwargs for call in mock.call_args_list]


def _make_context_length_error() -> openai.BadRequestError:
    """A 400 whose message trips the provider's context-length detection."""
    req = HttpxRequest("POST", "https://api.openai.com/v1/responses")
    resp = HttpxResponse(400, request=req)
    return openai.BadRequestError(
        message="This model's maximum context length is 400000 tokens",
        response=resp,
        body={
            "error": {
                "code": "context_length_exceeded",
                "message": "This model's maximum context length is 400000 tokens",
            }
        },
    )


def _make_generic_bad_request() -> openai.BadRequestError:
    """A 400 that is NOT a context-length error (must not be swallowed)."""
    req = HttpxRequest("POST", "https://api.openai.com/v1/responses")
    resp = HttpxResponse(400, request=req)
    return openai.BadRequestError(
        message="Invalid 'temperature': must be <= 2",
        response=resp,
        body={"error": {"code": "invalid_request_error", "message": "bad temp"}},
    )


def _request_with_prior_response_id(
    response_id: str = "resp_precompaction",
) -> ChatRequest:
    """2-turn request where the assistant's metadata carries a response_id
    (this is what the provider reverse-scans to build previous_response_id)."""
    msgs = [
        Message(role="user", content="Hi"),
        Message(
            role="assistant",
            content="Hello!",
            metadata={METADATA_RESPONSE_ID: response_id},
        ),
        Message(role="user", content="Follow-up"),
    ]
    return ChatRequest(messages=msgs)


def _simple_request() -> ChatRequest:
    return ChatRequest(messages=[Message(role="user", content="Hello")])


# ---------------------------------------------------------------------------
# Recording coordinator for the mount()/hook-wiring tests
# ---------------------------------------------------------------------------


class RecordingHooks:
    def __init__(self):
        self.subscriptions: list[tuple[str, Any]] = []
        self.events: list[tuple[str, dict]] = []

    def on(self, event: str, handler: Any, *args, **kwargs) -> None:
        self.subscriptions.append((event, handler))

    async def emit(self, name: str, payload: dict) -> None:
        self.events.append((name, payload))


class RecordingCoordinator:
    def __init__(self):
        self.hooks = RecordingHooks()
        self.mounted: dict[tuple[str, str | None], Any] = {}
        self.contributors: list[Any] = []

    async def mount(self, kind: str, obj: Any, name: str | None = None) -> None:
        self.mounted[(kind, name)] = obj

    def register_contributor(self, *args, **kwargs) -> None:
        self.contributors.append((args, kwargs))


def _mount_provider() -> RecordingCoordinator:
    coord = RecordingCoordinator()
    asyncio.run(
        mount(coord, {"api_key": "test-key", "use_streaming": False, "max_retries": 0})
    )
    return coord


# ---------------------------------------------------------------------------
# R1 wiring -- mount() subscribes and the handler flips the flag
# ---------------------------------------------------------------------------


def test_mount_subscribes_to_all_three_compaction_events():
    """The fix must survive a swapped context module, so it subscribes to the
    default module's literal name AND the kernel's pre/post constants."""
    coord = _mount_provider()
    subscribed = {event for event, _ in coord.hooks.subscriptions}
    assert "context:compaction" in subscribed
    assert "context:pre_compact" in subscribed
    assert "context:post_compact" in subscribed


def test_compaction_handler_sets_reset_flag_and_returns_continue():
    coord = _mount_provider()
    provider = coord.mounted[("providers", "openai")]
    assert provider._reset_chain_on_next_request is False

    handler = dict(coord.hooks.subscriptions)["context:compaction"]
    result = asyncio.run(handler("context:compaction", {"reason": "overflow"}))

    assert provider._reset_chain_on_next_request is True
    assert result.action == "continue"


# ---------------------------------------------------------------------------
# R1 behaviour -- flag drops previous_response_id on the next request only
# ---------------------------------------------------------------------------


def test_reset_flag_breaks_chain_on_next_request():
    provider = _make_provider(default_model="gpt-5.5")
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())

    # Simulate a compaction having fired since the last request.
    provider._reset_chain_on_next_request = True

    asyncio.run(provider.complete(_request_with_prior_response_id()))

    params = _captured_params(provider)
    assert "previous_response_id" not in params, (
        "After compaction the chain must be broken -- the pre-compaction "
        f"response id must not be sent. Got {params.get('previous_response_id')}"
    )
    # One-shot: the flag is consumed.
    assert provider._reset_chain_on_next_request is False


def test_chain_resumes_after_single_reset():
    """The reset is one-shot: the request AFTER the post-compaction request
    chains normally again."""
    provider = _make_provider(default_model="gpt-5.5")
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())

    provider._reset_chain_on_next_request = True
    asyncio.run(provider.complete(_request_with_prior_response_id()))  # chain broken
    asyncio.run(provider.complete(_request_with_prior_response_id()))  # chain resumes

    second_call = _all_calls(provider)[1]
    assert second_call.get("previous_response_id") == "resp_precompaction", (
        "Second request should chain normally again once the one-shot reset "
        "has been consumed."
    )


def test_reset_flag_cleared_even_when_no_prior_id():
    """If the post-compaction request happens to carry no prior id, the flag
    must still be consumed so it does not leak into a later request."""
    provider = _make_provider(default_model="gpt-5.5")
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())

    provider._reset_chain_on_next_request = True
    asyncio.run(provider.complete(_simple_request()))

    assert provider._reset_chain_on_next_request is False


# ---------------------------------------------------------------------------
# R2 behaviour -- context_length_exceeded self-heals by breaking the chain
# ---------------------------------------------------------------------------


def test_context_length_with_active_chain_retries_without_chain():
    provider = _make_provider(default_model="gpt-5.5")
    provider.client.responses.create = AsyncMock(
        side_effect=[_make_context_length_error(), DummyResponse()]
    )

    asyncio.run(provider.complete(_request_with_prior_response_id()))

    calls = _all_calls(provider)
    assert len(calls) == 2, "Provider should retry exactly once after the overflow"
    # First attempt chained from the pre-compaction id.
    assert calls[0].get("previous_response_id") == "resp_precompaction"
    # Retry dropped the chain and sent the full converted transcript.
    assert "previous_response_id" not in calls[1]
    assert calls[1].get("input"), "Retry must send the full input, not an empty delta"


def test_context_length_retry_emits_chain_invalidated_event():
    class FakeHooks:
        def __init__(self):
            self.events: list[tuple[str, dict]] = []

        async def emit(self, name: str, payload: dict) -> None:
            self.events.append((name, payload))

    class FakeCoordinator:
        def __init__(self):
            self.hooks = FakeHooks()

    coord = FakeCoordinator()
    provider = _make_provider(default_model="gpt-5.5")
    provider.coordinator = coord
    provider.client.responses.create = AsyncMock(
        side_effect=[_make_context_length_error(), DummyResponse()]
    )

    asyncio.run(provider.complete(_request_with_prior_response_id()))

    names = [n for n, _ in coord.hooks.events]
    assert RESPONSE_CHAIN_INVALIDATED in names
    payload = dict(coord.hooks.events)[RESPONSE_CHAIN_INVALIDATED]
    assert payload["error_code"] == "context_length_exceeded"
    assert payload["invalidated_id"] == "resp_precompaction"


def test_context_length_without_chain_raises_context_length_error():
    """No chain to break -> the provider must surface ContextLengthError so the
    context manager compacts. It must NOT silently retry forever."""
    provider = _make_provider(default_model="gpt-5.5")
    provider.client.responses.create = AsyncMock(
        side_effect=_make_context_length_error()
    )

    with pytest.raises(kernel_errors.ContextLengthError):
        asyncio.run(provider.complete(_simple_request()))

    assert len(_all_calls(provider)) == 1, "Without a chain there must be no retry"


def test_second_overflow_after_chain_break_raises():
    """If the request still overflows after the chain is broken, the provider
    must not loop -- it raises ContextLengthError for the context manager."""
    provider = _make_provider(default_model="gpt-5.5")
    provider.client.responses.create = AsyncMock(
        side_effect=[_make_context_length_error(), _make_context_length_error()]
    )

    with pytest.raises(kernel_errors.ContextLengthError):
        asyncio.run(provider.complete(_request_with_prior_response_id()))

    assert len(_all_calls(provider)) == 2, "Exactly one retry, then raise"


def test_generic_bad_request_not_treated_as_overflow():
    """A non-context-length 400 must not trigger the chain-break retry."""
    provider = _make_provider(default_model="gpt-5.5")
    provider.client.responses.create = AsyncMock(
        side_effect=_make_generic_bad_request()
    )

    # A generic 400 is translated to InvalidRequestError (not ContextLengthError)
    # and, crucially, must NOT trigger the Issue #321 chain-break retry.
    with pytest.raises(kernel_errors.InvalidRequestError):
        asyncio.run(provider.complete(_request_with_prior_response_id()))

    assert len(_all_calls(provider)) == 1, "Generic 400 must not retry"
