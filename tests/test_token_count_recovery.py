"""Counting retries use the real SDK with an in-memory HTTP transport."""
import asyncio
import json

import httpx
import openai
import pytest
from amplifier_core import ChatRequest, Message

from amplifier_module_provider_openai import OpenAIProvider
from amplifier_module_provider_openai import _token_count as counting


@pytest.fixture
def delays(monkeypatch):
    recorded = []

    async def sleep(delay):
        recorded.append(delay)

    monkeypatch.setattr(counting.asyncio, 'sleep', sleep)
    return recorded


def provider(handler):
    sdk = openai.AsyncOpenAI(api_key='fixture', max_retries=7,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    return OpenAIProvider(client=sdk, config={'default_model': 'gpt-5-mini'}), sdk


async def measure(model):
    return await model.request_budget(ChatRequest(messages=[Message(role='user', content='private prompt')]), context_estimate=5)


@pytest.mark.asyncio
@pytest.mark.parametrize('status', [408, 429, 500, 502, 503, 504])
async def test_transient_error_recovers_with_identical_count_only_requests(status, delays):
    wires = []

    def handler(request):
        assert request.url.path == '/v1/responses/input_tokens'
        wires.append(request.content)
        return httpx.Response(status if len(wires) < 3 else 200,
            json={'error': {'message': 'private payload'}} if len(wires) < 3 else {'input_tokens': 12})

    model, sdk = provider(handler)
    try:
        result = await measure(model)
        assert result['measurement']['input_tokens'] == 12
        assert len(wires) == 3 and len(set(wires)) == 1
        assert delays == [1, 2]
        assert sdk.max_retries == 7 and not sdk.is_closed()
    finally:
        await sdk.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('status,category', [(400, 'invalid_request'), (401, 'authentication'), (403, 'permission'), (422, 'invalid_request'), (429, 'rate_limit'), (503, 'service')])
async def test_failure_is_bounded_sanitized_and_later_measurement_can_succeed(status, category, delays):
    calls = []
    recovered = False

    def handler(request):
        calls.append(request.url.path)
        return httpx.Response(200 if recovered else status, headers={'x-request-id': 'req-safe_123'},
            json={'input_tokens': 14} if recovered else {'error': {'message': 'private prompt SECRET', 'code': 'fixture'}})

    model, sdk = provider(handler)
    try:
        with pytest.raises(counting.TokenCountError) as result:
            await measure(model)
        retryable = status in (429, 503)
        assert result.value.count_failure == {'category': category, 'retryable': retryable,
            'httpStatus': status, 'requestId': 'req-safe_123', 'attempts': 3 if retryable else 1}
        assert len(calls) == (3 if retryable else 1)
        assert set(calls) == {'/v1/responses/input_tokens'}
        assert result.value.retryable is False  # no outer generation replay
        assert result.value.__context__ is None
        assert 'SECRET' not in str(result.value) + json.dumps(result.value.count_failure)
        recovered = True
        assert (await measure(model))['measurement']['input_tokens'] == 14
        assert len(calls) == (4 if retryable else 2)
    finally:
        await sdk.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('header,expected_calls', [('5', 3), ('120', 1), ('Wed, 21 Oct 2026 07:28:00 GMT', 1)])
async def test_server_retry_delay_is_respected_or_left_for_user(header, expected_calls, delays):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(429, headers={'retry-after': header}, json={'error': {'message': 'busy'}})

    model, sdk = provider(handler)
    try:
        with pytest.raises(counting.TokenCountError):
            await measure(model)
        assert len(calls) == expected_calls
        assert delays == ([5, 5] if expected_calls == 3 else [])
    finally:
        await sdk.close()


@pytest.mark.asyncio
async def test_quota_exhaustion_does_not_retry(delays):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(429, json={'error': {'message': 'private', 'code': 'insufficient_quota'}})

    model, sdk = provider(handler)
    try:
        with pytest.raises(counting.TokenCountError) as result:
            await measure(model)
        assert result.value.count_failure['category'] == 'quota'
        assert len(calls) == 1 and delays == []
    finally:
        await sdk.close()


@pytest.mark.asyncio
async def test_timeout_has_a_bounded_wire_count(monkeypatch, delays):
    monkeypatch.setattr(counting, 'ATTEMPT_TIMEOUT', .01)
    calls = []

    async def handler(request):
        calls.append(request)
        await asyncio.Event().wait()

    model, sdk = provider(handler)
    try:
        with pytest.raises(counting.TokenCountError) as result:
            await measure(model)
        assert result.value.count_failure['category'] == 'timeout'
        assert len(calls) == 3
    finally:
        await sdk.close()


@pytest.mark.asyncio
async def test_cancel_during_backoff_does_not_retry_or_dispatch(monkeypatch):
    waiting = asyncio.Event()
    calls = []

    async def sleep(delay):
        waiting.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(counting.asyncio, 'sleep', sleep)

    def handler(request):
        calls.append(request)
        return httpx.Response(503, json={'error': {'message': 'busy'}})

    model, sdk = provider(handler)
    task = asyncio.create_task(measure(model))
    try:
        await asyncio.wait_for(waiting.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert len(calls) == 1
    finally:
        task.cancel()
        await sdk.close()
