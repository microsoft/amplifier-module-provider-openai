"""Model work waits for completion; only explicit deadlines interrupt it."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import openai
import pytest
from amplifier_core.llm_errors import LLMError, LLMTimeoutError
from amplifier_core.message_models import ChatRequest, Message

from amplifier_module_provider_openai import OpenAIProvider


def request(**kwargs):
    return ChatRequest(messages=[Message(role="user", content="Fixture")], **kwargs)


def completed():
    return {
        "id": "resp_fixture", "object": "response", "created_at": 1,
        "model": "gpt-6-astra", "status": "completed", "output": [],
        "usage": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["create", "raw", "stream", "compact"])
@pytest.mark.parametrize("configured", [None, 7, openai.Timeout(3, connect=1)])
async def test_actual_sdk_model_transports_apply_wait_policy(mode, configured):
    model_calls = []
    async def handle(req):
        if req.url.path.endswith("input_tokens"):
            return httpx.Response(200, json={"input_tokens": 100})
        model_calls.append(req)
        if mode == "compact":
            return httpx.Response(200, json={
                "id": "cmp_fixture", "object": "response.compaction", "created_at": 1,
                "output": [{"type": "compaction", "encrypted_content": "fixture"}],
                "usage": {"input_tokens": 100, "output_tokens": 10, "total_tokens": 110},
            })
        if mode == "stream":
            body = "".join("data: " + json.dumps({"type": kind, "sequence_number": seq,
                "response": completed()}) + "\n\n" for seq, kind in enumerate(
                    ["response.created", "response.completed"]))
            return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json=completed())

    # Even an injected SDK client with a short timeout must not silently impose
    # that limit on model work when the caller did not request a deadline.
    sdk = openai.AsyncOpenAI(api_key="fixture", timeout=0.001,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)))
    provider = OpenAIProvider(client=sdk, config={"default_model": "gpt-6-astra",
        "use_streaming": mode == "stream", "enable_long_context": True, "max_retries": 0,
        **({"extra_request_params": {"timeout": configured}} if configured is not None else {})})
    try:
        assert "timeout" not in provider._budget_params(request())
        if mode == "compact":
            await provider.compact_context(request())
        elif mode == "raw":
            await provider._create_response({"model": "gpt-6-astra", "input": [],
                "tools": [{"type": "computer"}], "timeout": configured})
        else:
            await provider.complete(request())
        assert len(model_calls) == 1
        expected = configured if isinstance(configured, openai.Timeout) else openai.Timeout(
            configured, connect=5.0, pool=5.0)
        assert model_calls[0].extensions["timeout"] == expected.as_dict()
    finally:
        await provider.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [True, False])
async def test_slow_completion_is_not_cut_off_by_default(monkeypatch, streaming):
    # Accelerate any finite default to expose regressions without waiting ten
    # minutes. The model returns after the accelerated deadline would expire.
    actual_timeout = asyncio.timeout
    actual_wait_for = asyncio.wait_for
    deadlines = []
    def accelerated_timeout(delay):
        deadlines.append(delay)
        return actual_timeout(None if delay is None else 0.001)
    async def accelerated_wait_for(awaitable, timeout):
        deadlines.append(timeout)
        return await actual_wait_for(awaitable, None if timeout is None else 0.001)
    monkeypatch.setattr(asyncio, "timeout", accelerated_timeout)
    monkeypatch.setattr(asyncio, "wait_for", accelerated_wait_for)
    async def finish(**kwargs):
        await asyncio.sleep(0.01)
        return SimpleNamespace(**completed())
    class Stream:
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def get_final_response(self): return await finish()
    client = SimpleNamespace(responses=SimpleNamespace(create=AsyncMock(side_effect=finish),
        stream=lambda **kwargs: Stream()))
    provider = OpenAIProvider(client=client, config={"use_streaming": streaming, "max_retries": 0})
    result = await provider.complete(request())
    assert result is not None
    assert deadlines and all(value is None for value in deadlines)


@pytest.mark.asyncio
async def test_background_polling_waits_past_old_default(monkeypatch):
    import amplifier_module_provider_openai as module
    now = [0.0]
    monkeypatch.setattr(module.time, "time", lambda: now[0])
    queued = SimpleNamespace(**{**completed(), "status": "in_progress"})
    async def poll(*args, **kwargs):
        now[0] += 86400
        return queued if now[0] == 86400 else SimpleNamespace(**completed())
    responses = SimpleNamespace(create=AsyncMock(return_value=queued), retrieve=AsyncMock(side_effect=poll))
    provider = OpenAIProvider(client=SimpleNamespace(responses=responses),
        config={"use_streaming": False, "background": True, "max_retries": 0})
    await provider.complete(request(), background=True, poll_interval=0)
    assert responses.retrieve.await_count == 2
    assert responses.retrieve.call_args.kwargs["timeout"].read is None


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["deadline", "connection", "cancelled"])
async def test_background_failure_never_returns_unfinished_success_or_replays(failure):
    queued = SimpleNamespace(**{**completed(), "status": "in_progress"})
    responses = SimpleNamespace(create=AsyncMock(return_value=queued),
        retrieve=AsyncMock(side_effect=ConnectionError("lost connection")))
    if failure == "cancelled":
        responses.retrieve = AsyncMock(return_value=SimpleNamespace(
            **{**completed(), "status": "cancelled"}))
    provider = OpenAIProvider(client=SimpleNamespace(responses=responses),
        config={"use_streaming": False, "max_retries": 0})
    req = request(timeout=0.001) if failure == "deadline" else request()
    with pytest.raises(LLMTimeoutError if failure == "deadline" else LLMError):
        await provider.complete(req, background=True, poll_interval=0.02 if failure == "deadline" else 0)
    assert responses.create.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [True, False])
async def test_cancellation_stops_wait_without_retry(streaming):
    entered, closed = asyncio.Event(), asyncio.Event()
    async def wait(**kwargs):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            closed.set()
    class Stream:
        async def __aenter__(self): return self
        async def __aexit__(self, *args): closed.set()
        async def get_final_response(self): return await wait()
    responses = SimpleNamespace(create=AsyncMock(side_effect=wait), stream=lambda **kwargs: Stream())
    provider = OpenAIProvider(client=SimpleNamespace(responses=responses), config={"use_streaming": streaming})
    task = asyncio.create_task(provider.complete(request()))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert closed.is_set()
    assert responses.create.await_count == (0 if streaming else 1)


def test_explicit_request_deadline_overrides_configuration():
    provider = OpenAIProvider(api_key="fixture", config={"timeout": 30, "background_timeout": 60})
    assert provider._request_timeout(request()) == 30
    assert provider._request_timeout(request(), background=True) == 60
    assert provider._request_timeout(request(timeout=3)) == 3
    assert provider._request_timeout(request(timeout=None)) is None
    assert provider._request_timeout(request(timeout=None), background=True) is None
