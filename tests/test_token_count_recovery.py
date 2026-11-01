"""Counting retries use the real SDK with an in-memory HTTP transport."""
import asyncio
import json
import sys
import traceback
from types import SimpleNamespace

import httpx
import openai
import pytest
from amplifier_core import ChatRequest, Message

from amplifier_module_provider_openai import OpenAIProvider, mount
from amplifier_module_provider_openai import _token_count as counting


class RecordingHooks:
    def __init__(self):
        self.events = []

    async def emit(self, event, data):
        self.events.append((event, data))


def observe(model, hooks=None):
    hooks = hooks if hooks is not None else RecordingHooks()
    model.coordinator = SimpleNamespace(hooks=hooks)
    return hooks


def failure_event(diagnostic, events):
    attribution = next(data['attribution'] for event, data in events
                       if event == 'provider:token_count_failed')
    assert attribution['version'] == 1
    assert len(attribution['scope_id']) == len(attribution['count_id']) == 32
    assert len(attribution['request']['payload_tag']) == 64
    assert 'private prompt' not in json.dumps(attribution)
    return ('provider:token_count_failed', {
        'provider': 'openai',
        'operation': 'responses.input_tokens.count',
        'count_failure': diagnostic,
        'outer_retryable': False,
        'attribution': attribution,
    })


@pytest.fixture
def delays(monkeypatch):
    recorded = []

    async def sleep(delay):
        recorded.append(delay)

    monkeypatch.setattr(counting.asyncio, 'sleep', sleep)
    return recorded


def provider(handler, *, sdk_retries=7, **config):
    sdk = openai.AsyncOpenAI(api_key='fixture', max_retries=sdk_retries,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    return OpenAIProvider(client=sdk, config={
        'default_model': 'gpt-5-mini', 'use_streaming': False, **config,
    }), sdk


def request():
    return ChatRequest(messages=[Message(role='user', content='private prompt')])


async def measure(model):
    return await model.request_budget(request(), context_estimate=5)


async def invoke(model, operation):
    if operation == 'preflight':
        return await measure(model)
    assert operation == 'complete'
    return await model.complete(request())


def completed_response():
    return httpx.Response(200, json={
        'id': 'resp-fixture', 'object': 'response', 'created_at': 1,
        'model': 'gpt-5-mini', 'status': 'completed',
        'output': [{'type': 'message', 'id': 'msg-fixture', 'role': 'assistant',
                    'status': 'completed', 'content': [
                        {'type': 'output_text', 'text': 'OK', 'annotations': []},
                    ]}],
        'usage': {'input_tokens': 12, 'output_tokens': 1, 'total_tokens': 13},
    })


@pytest.mark.asyncio
async def test_mount_contributes_token_count_failure_event_without_client():
    class Coordinator:
        def __init__(self):
            self.providers = {}
            self.contributors = {}

        async def mount(self, point, instance, *, name):
            assert point == 'providers'
            self.providers[name] = instance

        def register_contributor(self, channel, name, callback):
            self.contributors[channel, name] = callback

    coordinator = Coordinator()
    cleanup = await mount(coordinator, {'api_key': 'fixture'})
    try:
        assert coordinator.providers['openai']._client is None
        contributed = coordinator.contributors[
            'observability.events', 'provider-openai'
        ]()
        assert contributed == ['provider:token_count_failed', 'provider:request_attribution']
        assert ('session.cost', 'provider-openai') in coordinator.contributors
    finally:
        await cleanup()


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
    hooks = observe(model)
    try:
        result = await measure(model)
        assert result['measurement']['input_tokens'] == 12
        assert len(wires) == 3 and len(set(wires)) == 1
        assert delays == [1, 2]
        assert sdk.max_retries == 7 and not sdk.is_closed()
        assert hooks.events == []
    finally:
        await sdk.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('operation', ['preflight', 'complete'])
@pytest.mark.parametrize('status,category', [(400, 'invalid_request'), (401, 'authentication'), (403, 'permission'), (422, 'invalid_request'), (429, 'rate_limit'), (503, 'service')])
async def test_failure_is_bounded_sanitized_and_later_measurement_can_succeed(operation, status, category, delays):
    calls = []
    recovered = False

    def handler(request):
        calls.append(request.url.path)
        return httpx.Response(200 if recovered else status, headers={'x-request-id': 'req-safe_123'},
            json={'input_tokens': 14} if recovered else {'error': {'message': 'private prompt SECRET', 'code': 'fixture'}})

    model, sdk = provider(handler)
    hooks = observe(model)
    try:
        with pytest.raises(counting.TokenCountError) as result:
            await invoke(model, operation)
        retryable = status in (429, 503)
        assert result.value.count_failure == {'category': category, 'retryable': retryable,
            'httpStatus': status, 'requestId': 'req-safe_123', 'attempts': 3 if retryable else 1}
        assert len(calls) == (3 if retryable else 1)
        assert set(calls) == {'/v1/responses/input_tokens'}
        assert result.value.retryable is False  # no outer generation replay
        assert result.value.__context__ is None
        assert result.value.__cause__ is None
        assert 'SECRET' not in str(result.value) + json.dumps(result.value.count_failure)
        assert hooks.events == [failure_event(result.value.count_failure, hooks.events)]
        assert hooks.events[0][1]['count_failure'] is not result.value.count_failure
        assert delays == ([1, 2] if retryable else [])
        recovered = True
        assert (await measure(model))['measurement']['input_tokens'] == 14
        assert len(calls) == (4 if retryable else 2)
        assert len(hooks.events) == 1
    finally:
        await sdk.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('header,expected_calls', [('5', 3), ('120', 1), ('malformed', 1), ('Wed, 21 Oct 2026 07:28:00 GMT', 1)])
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
@pytest.mark.parametrize('operation', ['preflight', 'complete'])
@pytest.mark.parametrize('code', ['insufficient_quota', 'billing_hard_limit_reached'])
async def test_quota_exhaustion_does_not_retry(operation, code, delays):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(429, json={'error': {'message': 'private', 'code': code}})

    model, sdk = provider(handler)
    hooks = observe(model)
    try:
        with pytest.raises(counting.TokenCountError) as result:
            await invoke(model, operation)
        assert result.value.count_failure == {
            'category': 'quota', 'retryable': False, 'httpStatus': 429, 'attempts': 1,
        }
        assert result.value.retryable is False
        assert hooks.events == [failure_event(result.value.count_failure, hooks.events)]
        assert all(call.url.path == '/v1/responses/input_tokens' for call in calls)
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
    hooks = observe(model)
    try:
        with pytest.raises(counting.TokenCountError) as result:
            await measure(model)
        assert result.value.count_failure == {
            'category': 'timeout', 'retryable': True, 'attempts': 3,
        }
        assert len(calls) == 3
        assert delays == [1, 2]
        assert hooks.events == [failure_event(result.value.count_failure, hooks.events)]
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
    hooks = observe(model)
    task = asyncio.create_task(measure(model))
    try:
        await asyncio.wait_for(waiting.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert len(calls) == 1
        assert hooks.events == []
    finally:
        task.cancel()
        await sdk.close()


@pytest.mark.asyncio
async def test_complete_waits_for_recovered_count_before_request_event_and_generation(delays):
    timeline = []
    counts = []

    class Hooks(RecordingHooks):
        async def emit(self, event, data):
            timeline.append(event)
            await super().emit(event, data)

    def handler(request):
        timeline.append(request.url.path)
        if request.url.path == '/v1/responses/input_tokens':
            assert 'llm:request' not in timeline
            counts.append(request.content)
            return httpx.Response(503 if len(counts) < 3 else 200, json=(
                {'error': {'message': 'private count body'}} if len(counts) < 3
                else {'input_tokens': 12}
            ))
        assert request.url.path == '/v1/responses'
        return completed_response()

    model, sdk = provider(handler)
    hooks = observe(model, Hooks())
    try:
        result = await model.complete(request())
        assert result.usage.input_tokens == 12
        assert len(counts) == 3 and len(set(counts)) == 1
        assert delays == [1, 2]
        assert [entry for entry in timeline if entry in {
            '/v1/responses/input_tokens', 'llm:request', '/v1/responses',
        }] == ['/v1/responses/input_tokens'] * 3 + ['llm:request', '/v1/responses']
        assert not any(event == 'provider:token_count_failed' for event, _ in hooks.events)
    finally:
        await sdk.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('category', ['timeout', 'connection'])
@pytest.mark.parametrize('recover', [False, True])
async def test_transport_failures_recover_or_emit_once(category, recover, delays):
    calls = []

    def handler(request):
        assert request.url.path == '/v1/responses/input_tokens'
        assert set(request.extensions['timeout'].values()) == {30}
        calls.append(request.content)
        if len(calls) < 3 or not recover:
            error_type = httpx.ReadTimeout if category == 'timeout' else httpx.ConnectError
            raise error_type('private transport detail', request=request)
        return httpx.Response(200, json={'input_tokens': 12})

    model, sdk = provider(handler)
    hooks = observe(model)
    try:
        if recover:
            assert (await measure(model))['measurement']['input_tokens'] == 12
            assert hooks.events == []
        else:
            with pytest.raises(counting.TokenCountError) as result:
                await measure(model)
            assert result.value.count_failure == {
                'category': category, 'retryable': True, 'attempts': 3,
            }
            assert result.value.retryable is False
            assert result.value.__context__ is None
            assert result.value.__cause__ is None
            assert hooks.events == [failure_event(result.value.count_failure, hooks.events)]
        assert len(calls) == 3 and len(set(calls)) == 1
        assert delays == [1, 2]
        assert sdk.max_retries == 7 and not sdk.is_closed()
    finally:
        await sdk.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('operation', ['preflight', 'complete'])
async def test_invalid_native_response_is_failure_not_unavailable(operation, delays):
    calls = []

    def handler(request):
        calls.append(request.url.path)
        return httpx.Response(200, json={'input_tokens': -1})

    model, sdk = provider(handler)
    hooks = observe(model)
    try:
        with pytest.raises(counting.TokenCountError) as result:
            await invoke(model, operation)
        diagnostic = {'category': 'invalid_response', 'retryable': False, 'attempts': 1}
        assert result.value.count_failure == diagnostic
        assert hooks.events == [failure_event(diagnostic, hooks.events)]
        assert calls == ['/v1/responses/input_tokens'] and delays == []
    finally:
        await sdk.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('operation', ['preflight', 'complete'])
async def test_hostile_request_id_and_sdk_body_stay_out_of_terminal_diagnostic(operation, delays):
    calls = []
    hostile_id = 'https://private.invalid/PRIVATE_ID?body=hidden'
    private_body = 'PRIVATE_SDK_BODY'

    def handler(request):
        calls.append(request.url.path)
        return httpx.Response(503, headers={'x-request-id': hostile_id},
            json={'error': {'message': private_body, 'code': 'fixture'}})

    model, sdk = provider(handler)
    hooks = observe(model)
    try:
        with pytest.raises(counting.TokenCountError) as result:
            await invoke(model, operation)
        diagnostic = {'category': 'service', 'retryable': True, 'httpStatus': 503, 'attempts': 3}
        assert result.value.count_failure == diagnostic
        assert hooks.events == [failure_event(diagnostic, hooks.events)]
        assert result.value.__context__ is None and result.value.__cause__ is None
        published = json.dumps(hooks.events) + ''.join(traceback.format_exception(result.value))
        assert hostile_id not in published and private_body not in published
        assert 'private prompt' not in published
        assert calls == ['/v1/responses/input_tokens'] * 3
    finally:
        await sdk.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('operation', ['preflight', 'complete'])
@pytest.mark.parametrize('coordinator', [None, SimpleNamespace(), SimpleNamespace(hooks=None)])
async def test_missing_observer_still_raises_count_failure(operation, coordinator, delays):
    calls = []

    def handler(request):
        calls.append(request.url.path)
        return httpx.Response(503, json={'error': {'message': 'private'}})

    model, sdk = provider(handler)
    model.coordinator = coordinator
    try:
        with pytest.raises(counting.TokenCountError) as result:
            await invoke(model, operation)
        assert result.value.retryable is False
        assert result.value.count_failure['attempts'] == 3
        assert calls == ['/v1/responses/input_tokens'] * 3
    finally:
        await sdk.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('temporary_client', [False, True])
async def test_observer_failure_preserves_original_error_without_private_chain(
    temporary_client, monkeypatch, caplog, delays,
):
    calls = []
    private_body = 'PRIVATE_SDK_BODY'
    private_observer = 'PRIVATE_OBSERVER_BODY'
    original_errors = []

    class Hooks(RecordingHooks):
        async def emit(self, event, data):
            await super().emit(event, data)
            original_errors.append(sys.exception())
            raise RuntimeError(private_observer)

    def handler(request):
        calls.append(request.url.path)
        return httpx.Response(503, json={'error': {'message': private_body}})

    model, sdk = provider(handler)
    if temporary_client:
        # Keep a real SDK/MockTransport while exercising owned-client cleanup.
        monkeypatch.setattr('amplifier_module_provider_openai.AsyncOpenAI', lambda **kwargs: sdk)
        model._client = None
    hooks = observe(model, Hooks())
    try:
        with pytest.raises(counting.TokenCountError) as result:
            await measure(model)
        assert original_errors == [result.value]
        assert result.value.__context__ is None and result.value.__cause__ is None
        assert hooks.events == [failure_event(result.value.count_failure, hooks.events)]
        assert hooks.events[0][1]['count_failure'] is not result.value.count_failure
        published = (json.dumps(hooks.events) + caplog.text
                     + ''.join(traceback.format_exception(result.value)))
        assert private_body not in published and private_observer not in published
        records = [record for record in caplog.records
                   if record.name == 'amplifier_module_provider_openai']
        assert len(records) == 1
        assert records[0].getMessage() == 'Could not emit token count failure event.'
        assert records[0].exc_info is None and records[0].args == ()
        assert calls == ['/v1/responses/input_tokens'] * 3
        assert sdk.is_closed() is temporary_client
    finally:
        await sdk.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('temporary_client', [False, True])
async def test_cancellation_through_observer_propagates_and_closes_owned_client(
    temporary_client, monkeypatch, delays,
):
    entered = asyncio.Event()
    calls = []

    class Hooks(RecordingHooks):
        async def emit(self, event, data):
            await super().emit(event, data)
            entered.set()
            await asyncio.Event().wait()

    def handler(request):
        calls.append(request.url.path)
        return httpx.Response(503, json={'error': {'message': 'private'}})

    model, sdk = provider(handler)
    if temporary_client:
        monkeypatch.setattr('amplifier_module_provider_openai.AsyncOpenAI', lambda **kwargs: sdk)
        model._client = None
    hooks = observe(model, Hooks())
    task = asyncio.create_task(measure(model))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert len(hooks.events) == 1
        assert hooks.events[0][0] == 'provider:token_count_failed'
        assert calls == ['/v1/responses/input_tokens'] * 3
        assert sdk.is_closed() is temporary_client
    finally:
        task.cancel()
        await sdk.close()


@pytest.mark.asyncio
async def test_failed_refresh_count_stops_outer_retry_without_replaying_generation(delays):
    calls = []

    def handler(request):
        calls.append((request.url.path, request.content))
        if len(calls) == 1:
            assert request.url.path == '/v1/responses/input_tokens'
            return httpx.Response(200, json={'input_tokens': 12})
        if len(calls) == 2:
            assert request.url.path == '/v1/responses'
            return httpx.Response(503, json={'error': {'message': 'generation unavailable'}})
        return httpx.Response(503, headers={'x-request-id': 'req-count'},
            json={'error': {'message': 'private count body'}})

    # Use ordinary provider retries but no SDK generation retries. Counting
    # independently disables SDK retries even for an injected retrying client.
    model, sdk = provider(handler, sdk_retries=0, max_retries=5,
        min_retry_delay=1, retry_jitter=False)
    hooks = observe(model)
    try:
        with pytest.raises(counting.TokenCountError) as result:
            await model.complete(request())
        assert [path for path, _ in calls] == [
            '/v1/responses/input_tokens', '/v1/responses',
            '/v1/responses/input_tokens', '/v1/responses/input_tokens',
            '/v1/responses/input_tokens',
        ]
        assert len({body for path, body in calls if path.endswith('/input_tokens')}) == 1
        assert result.value.retryable is False
        assert result.value.count_failure == {
            'category': 'service', 'retryable': True, 'httpStatus': 503,
            'requestId': 'req-count', 'attempts': 3,
        }
        assert result.value.__context__ is None and result.value.__cause__ is None
        assert [entry for entry in hooks.events if entry[0] == 'provider:token_count_failed'] == [
            failure_event(result.value.count_failure, hooks.events),
        ]
        assert sum(event == 'llm:request' for event, _ in hooks.events) == 1
        assert sum(event == 'provider:retry' for event, _ in hooks.events) == 1
        responses = [data for event, data in hooks.events if event == 'llm:response']
        assert len(responses) == 1 and responses[0]['status'] == 'error'
        assert responses[0]['error'] == str(result.value)
        assert not any(event == 'provider:error' for event, _ in hooks.events)
        assert delays == [1, 1, 2]  # one generation backoff, then two count backoffs
    finally:
        await sdk.close()
