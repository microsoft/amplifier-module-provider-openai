"""Actual SDK serialization and HTTP attempts, using only owned fake transports."""

import asyncio
import copy
import json

import httpx
import pytest
from amplifier_core.message_models import ChatRequest, Message
from openai import AsyncOpenAI

import amplifier_module_provider_openai as module
from amplifier_module_provider_openai import OpenAIProvider
from amplifier_module_provider_openai import _single_attempt as bounded


def request(**changes):
    values = {
        "messages": [Message(role="user", content="Reply with OK.")],
        "model": "gpt-5.6-terra",
        "reasoning_effort": "high",
        "max_output_tokens": 1024,
        "timeout": 45,
        "stream": False,
    }
    values.update(changes)
    return ChatRequest(**values)


def response(**changes):
    value = {
        "id": "resp_offline",
        "object": "response",
        "created_at": 0,
        "status": "completed",
        "error": None,
        "incomplete_details": None,
        "model": "gpt-5.6-terra",
        "instructions": None,
        "output": [
            {
                "type": "message",
                "id": "msg_offline",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "OK", "annotations": []}],
            }
        ],
        "parallel_tool_calls": False,
        "tools": [],
        "tool_choice": "auto",
        "usage": {
            "input_tokens": 12,
            "output_tokens": 8,
            "total_tokens": 20,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 6},
        },
    }
    value.update(changes)
    return value


@pytest.fixture
def wire(monkeypatch):
    class Wire:
        def __init__(self):
            self.calls = []
            self.clients = []
            self.count = {"input_tokens": 12}
            self.generation = response()

        closed = 0
        count_status = 200
        generation_status = 200
        close_failure = None
        hold = None
        redirect = None

        async def send(self, req):
            payload = json.loads(req.content)
            self.calls.append((req.url.path, payload))
            if self.redirect and req.url.path == self.redirect:
                return httpx.Response(
                    307, headers={"location": "https://api.openai.com/v1/elsewhere"}
                )
            if self.hold:
                await self.hold.wait()
            if req.url.path == "/v1/responses/input_tokens":
                return httpx.Response(self.count_status, json=self.count)
            assert req.url.path == "/v1/responses"
            return httpx.Response(self.generation_status, json=self.generation)

        def http_client(self, timeout):
            owner = self

            class Transport(httpx.MockTransport):
                async def aclose(self):
                    owner.closed += 1
                    if owner.close_failure == "hang":
                        await asyncio.Event().wait()
                    if owner.close_failure:
                        raise RuntimeError("private close diagnostic")
                    await super().aclose()

            return httpx.AsyncClient(
                timeout=timeout,
                transport=Transport(self.send),
                trust_env=False,
                follow_redirects=False,
            )

        def factory(self, **kwargs):
            assert kwargs["max_retries"] == 0
            if "http_client" not in kwargs:
                kwargs["http_client"] = self.http_client(45)
            client = AsyncOpenAI(**kwargs)
            self.clients.append(client)
            return client

    value = Wire()
    monkeypatch.setattr(module, "AsyncOpenAI", value.factory)
    monkeypatch.setattr(bounded, "_http_client", value.http_client)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    return value


async def run(provider=None, **changes):
    provider = provider or OpenAIProvider(api_key="offline-dummy")
    return await provider.complete(
        request(**changes), request_options={"single_attempt": True}
    )


@pytest.mark.asyncio
async def test_actual_sdk_exact_count_create_and_confirmed_close(wire):
    provider = OpenAIProvider(
        api_key="offline-dummy",
        config={
            "reasoning_effort": "low",
            "max_retries": 5,
            "use_streaming": True,
            "extra_request_params": {"max_output_tokens": 4000},
        },
    )
    original = copy.deepcopy(provider.config)
    admitted = request()
    saved = admitted.model_dump()
    assert bounded.CAPABILITY in provider.get_info().capabilities
    result = await provider.complete(admitted, request_options={"single_attempt": True})
    assert [path for path, _ in wire.calls] == [
        "/v1/responses/input_tokens",
        "/v1/responses",
    ]
    count, generation = [payload for _, payload in wire.calls]
    assert count == {
        key: value for key, value in generation.items() if key in provider._COUNT_FIELDS
    }
    assert generation["input"] == [
        {"role": "user", "content": [{"type": "input_text", "text": "Reply with OK."}]}
    ]
    assert generation["model"] == "gpt-5.6-terra"
    assert generation["reasoning"]["effort"] == "high"
    assert generation["max_output_tokens"] == 1024
    assert not generation.get("stream") and not generation.get("tools")
    assert len(wire.clients) == wire.closed == 1 and wire.clients[0].is_closed()
    receipt = result.metadata[bounded.RECEIPT_KEY]
    assert receipt == {
        "version": 1,
        "model": "gpt-5.6-terra",
        "reasoning_effort": "high",
        "max_output_tokens": 1024,
        "timeout_seconds": 45,
        "native_count_requests": 1,
        "generation_requests": 1,
        "native_input_tokens": 12,
        "retries": 0,
        "continuations": 0,
        "closed": True,
        "input_sha256": bounded._digest(generation["input"]),
        "request_sha256": bounded._digest(generation),
    }
    assert result.usage.output_tokens == 8 and result.usage.reasoning_tokens == 6
    assert provider._client is None and provider.config == original
    assert provider.use_streaming is True and provider._retry_config.max_retries == 5
    assert admitted.model_dump() == saved


@pytest.mark.parametrize(
    "extra",
    [
        {"model": "gpt-5.6-sol"},
        {"reasoning": {"effort": "low"}},
        {"input": "different"},
        {"instructions": "different"},
        {"tools": [{"type": "image_generation"}]},
        {"background": True},
        {"stream": True},
        {"store": True},
        {"conversation": "old"},
        {"previous_response_id": "old"},
        {"truncation": "auto"},
        {"extra_body": {"model": "different"}},
        {"extra_body": {"input": "different"}},
        {"extra_body": {"max_output_tokens": 100000}},
        {"extra_headers": {"x-test": "override"}},
        {"extra_query": {"override": "true"}},
        {"timeout": 10000},
    ],
)
@pytest.mark.asyncio
async def test_final_wire_override_fails_before_any_client_or_send(wire, extra):
    provider = OpenAIProvider(
        api_key="offline-dummy", config={"extra_request_params": extra}
    )
    with pytest.raises(bounded.SingleAttemptError):
        await run(provider)
    assert wire.calls == wire.clients == []


@pytest.mark.parametrize(
    "changes",
    [
        {"reasoning_effort": None},
        {"model": None},
        {"max_output_tokens": None},
        {"max_output_tokens": 0},
        {"timeout": None},
        {"timeout": float("inf")},
        {"stream": True},
        {"conversation_id": "old"},
        {"tool_choice": "auto"},
        {"messages": [Message(role="user", content=" ")]},
        {"messages": [Message(role="assistant", content="OK")]},
        {
            "messages": [
                Message(role="user", content="hi"),
                Message(role="user", content="again"),
            ]
        },
    ],
)
@pytest.mark.asyncio
async def test_unadmitted_request_fails_before_dispatch(wire, changes):
    with pytest.raises(bounded.SingleAttemptError):
        await run(**changes)
    assert wire.calls == wire.clients == []


@pytest.mark.parametrize(
    "url",
    [
        "http://api.openai.com/v1",
        "https://elsewhere.test/v1",
        "https://api.openai.com/v1?other=1",
        "https://user:secret@api.openai.com/v1",
        "https://api.openai.com:444/v1",
        "https://api.openai.com/v1#fragment",
    ],
)
@pytest.mark.asyncio
async def test_unsupported_endpoint_has_no_capability_or_dispatch(wire, url):
    provider = OpenAIProvider(api_key="offline-dummy", config={"base_url": url})
    assert bounded.CAPABILITY not in provider.get_info().capabilities
    with pytest.raises(bounded.SingleAttemptError, match="unsupported_endpoint"):
        await run(provider)
    assert wire.calls == wire.clients == []


@pytest.mark.asyncio
async def test_injected_client_is_not_replaced_or_closed(wire):
    existing = AsyncOpenAI(
        api_key="offline-dummy",
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(lambda req: pytest.fail("unexpected HTTP"))
        ),
    )
    provider = OpenAIProvider(api_key="offline-dummy", client=existing)
    assert bounded.CAPABILITY not in provider.get_info().capabilities
    with pytest.raises(bounded.SingleAttemptError):
        await run(provider)
    assert provider._client is existing and not existing.is_closed()
    assert wire.calls == wire.clients == []
    await existing.close()


@pytest.mark.parametrize(
    "count,status",
    [
        ({}, 200),
        ({"input_tokens": -1}, 200),
        ({"input_tokens": True}, 200),
        ({"input_tokens": 12}, 500),
        ({"input_tokens": 12}, 429),
        ({"input_tokens": 10**9}, 200),
    ],
)
@pytest.mark.asyncio
async def test_count_failure_never_falls_back_or_retries(wire, count, status):
    wire.count, wire.count_status = count, status
    with pytest.raises(bounded.SingleAttemptError):
        await run()
    assert len(wire.calls) == wire.closed == 1
    assert wire.calls[0][0].endswith("/input_tokens")


@pytest.mark.parametrize(
    "change",
    [
        {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}},
        {"status": "failed", "error": {"code": "server_error", "message": "private"}},
        {"output": []},
        {"model": "gpt-5.6-sol"},
        {"object": "wrong"},
        {
            "output": [
                {
                    "type": "message",
                    "id": "m",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "refusal", "refusal": "No"}],
                }
            ]
        },
        {
            "output": [
                {
                    "type": "function_call",
                    "call_id": "c",
                    "id": "f",
                    "name": "patch",
                    "arguments": "{",
                    "status": "incomplete",
                }
            ]
        },
        {"usage": {"input_tokens": 12, "output_tokens": 1025, "total_tokens": 1037}},
    ],
)
@pytest.mark.asyncio
async def test_incomplete_refused_or_invalid_response_never_continues(wire, change):
    wire.generation = response(**change)
    with pytest.raises(bounded.SingleAttemptError):
        await run()
    assert len(wire.calls) == 2 and wire.closed == 1


@pytest.mark.asyncio
async def test_generation_http_failure_never_retries(wire):
    wire.generation_status = 500
    with pytest.raises(bounded.SingleAttemptError) as error:
        await run()
    assert error.value.retryable is False
    assert len(wire.calls) == 2 and wire.closed == 1


@pytest.mark.parametrize(
    "path,attempts", [("/v1/responses/input_tokens", 1), ("/v1/responses", 2)]
)
@pytest.mark.asyncio
async def test_redirect_does_not_add_a_transmission(wire, path, attempts):
    wire.redirect = path
    with pytest.raises(bounded.SingleAttemptError):
        await run()
    assert len(wire.calls) == attempts and wire.closed == 1


def test_owned_http_client_disables_redirects():
    client = bounded._http_client(45)
    assert client.follow_redirects is False
    asyncio.run(client.aclose())


@pytest.mark.parametrize("failure", ["error", "hang"])
@pytest.mark.asyncio
async def test_close_failure_cannot_return_success(wire, monkeypatch, failure):
    wire.close_failure = failure
    monkeypatch.setattr(bounded, "CLOSE_TIMEOUT", 0.01)
    with pytest.raises(bounded.SingleAttemptError, match="close_failed"):
        await run()
    assert len(wire.calls) == 2 and wire.closed == 1


@pytest.mark.asyncio
async def test_timeout_or_cancellation_closes_without_generation(wire):
    wire.hold = asyncio.Event()
    with pytest.raises(bounded.SingleAttemptError, match="timeout"):
        await run(timeout=0.01)
    assert len(wire.calls) == wire.closed == 1
    task = asyncio.create_task(run())
    while len(wire.calls) < 2:
        await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(wire.calls) == wire.closed == 2


@pytest.mark.asyncio
async def test_normal_path_defaults_and_continuation_remain(wire):
    provider = OpenAIProvider(api_key="offline-dummy", config={"use_streaming": False})
    assert provider._retry_config.max_retries == 5
    first = response(
        status="incomplete", incomplete_details={"reason": "max_output_tokens"}
    )
    original = wire.send

    async def send(req):
        if req.url.path == "/v1/responses":
            wire.generation = (
                first
                if not any(path == req.url.path for path, _ in wire.calls)
                else response()
            )
        return await original(req)

    wire.send = send
    # The normal count client does not set max_retries (historical behavior),
    # so initialize its normal generation client once via the existing property.
    _ = provider.client
    result = await provider.complete(request())
    assert sum(path == "/v1/responses" for path, _ in wire.calls) == 2
    assert bounded.RECEIPT_KEY not in result.metadata
    await provider.close()


@pytest.mark.parametrize("option", ["true", 1, None])
@pytest.mark.asyncio
async def test_malformed_mode_is_not_silently_ignored(wire, option):
    provider = OpenAIProvider(api_key="offline-dummy")
    with pytest.raises(bounded.SingleAttemptError, match="invalid_options"):
        await provider.complete(request(), request_options={"single_attempt": option})
    assert wire.calls == wire.clients == []
