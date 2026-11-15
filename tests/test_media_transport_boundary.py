"""Prepared Responses requests must not bypass the last borrowed transport fence."""
import importlib
import inspect

import httpx
import pytest
from amplifier_core.llm_errors import InvalidRequestError
from openai import AsyncOpenAI, DefaultAsyncHttpxClient

from tests.test_media_projection import provider, request, response


async def fit(p, req):
    value = p.request_budget(req, context_estimate=100)
    if inspect.isawaitable(value):
        value = await value
    assert value is not None


def reply(p, wire):
    if wire.url.path.endswith("/input_tokens"):
        return {"object": "response.input_tokens", "input_tokens": 30}
    return response(p.default_model)


@pytest.mark.asyncio
@pytest.mark.parametrize("mounted", [False, True])
@pytest.mark.parametrize("cut", ["before_fit", "after_fit", "request_hook"])
async def test_prepared_transport_instance_wrapper_never_sees_private_wire(monkeypatch, mounted, cut):
    req = request()
    p, _, calls, _ = provider(req)
    physical, observer_calls = [], []
    http = p.client._client
    async def bottom(_transport, wire):
        physical.append(wire.url.path)
        return httpx.Response(200, json=reply(p, wire))
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", bottom)
    transport = httpx.AsyncHTTPTransport()
    if mounted:
        from httpx._utils import URLPattern
        http._mounts[URLPattern("https://")] = transport
    else:
        http._transport = transport
    async def observer(wire):
        observer_calls.append(True)
        wire.headers["authorization"] = "Bearer other-account"
        return await bottom(transport, wire)
    emit = p.coordinator.hooks.emit
    async def hook(name, payload):
        if name == "llm:request":
            transport.handle_async_request = observer
        return await emit(name, payload)
    try:
        if cut == "before_fit":
            transport.handle_async_request = observer
        else:
            await fit(p, req)
            if cut == "after_fit":
                transport.handle_async_request = observer
            else:
                p.coordinator.hooks.emit = hook
        before = len(physical)
        with pytest.raises(InvalidRequestError):
            await p.complete(req)
        assert observer_calls == [] and calls == []
        if cut == "request_hook":
            # complete may refresh its native count before llm:request. Those
            # earlier authorized counts are not the subsequently refused send.
            assert all(path.endswith("/input_tokens") for path in physical)
        else:
            assert len(physical) == before
    finally:
        await p.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("fitted", [False, True])
async def test_prepared_closed_original_http_client_is_refused(fitted):
    req = request()
    p, _, calls, _ = provider(req)
    try:
        if fitted:
            await fit(p, req)
        before = len(calls)
        await p.client.close()
        with pytest.raises(InvalidRequestError):
            await p.complete(req)
        assert len(calls) == before
    finally:
        await p.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("fitted", [False, True])
async def test_prepared_inner_http_pool_retries_are_not_silent_attempts(monkeypatch, fitted):
    req = request()
    p, _, calls, _ = provider(req)
    physical = []
    async def bottom(_transport, wire):
        physical.append(True)
        return httpx.Response(200, json=reply(p, wire))
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", bottom)
    transport = httpx.AsyncHTTPTransport(retries=0 if fitted else 2)
    p.client._client._transport = transport
    try:
        if fitted:
            await fit(p, req)
            transport._pool._retries = 2
        before = len(physical)
        with pytest.raises(InvalidRequestError):
            await p.complete(req)
        assert calls == [] and len(physical) == before
    finally:
        await p.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mounted", [False, True])
@pytest.mark.parametrize("sdk_default", [False, True])
async def test_standard_transport_text_open_then_media_preserves_pool_owner(
    monkeypatch, mounted, sdk_default,
):
    req = request()
    p, _, _, _ = provider(req)
    physical, closed = [], []
    module = httpx
    if sdk_default:
        base = next(cls for cls in DefaultAsyncHttpxClient.__mro__
                    if cls.__name__ == "AsyncClient")
        module = importlib.import_module(base.__module__.split(".")[0])

    async def bottom(_transport, wire):
        physical.append(wire.url.path)
        return module.Response(200, json=reply(p, wire))
    original_close = module.AsyncHTTPTransport.aclose
    async def close(transport):
        closed.append(True)
        await original_close(transport)
    monkeypatch.setattr(module.AsyncHTTPTransport, "handle_async_request", bottom)
    monkeypatch.setattr(module.AsyncHTTPTransport, "aclose", close)
    transport = module.AsyncHTTPTransport()
    http = (DefaultAsyncHttpxClient if sdk_default else module.AsyncClient)(
        trust_env=False, follow_redirects=False,
        **({"mounts": {"https://": transport}} if mounted else {"transport": transport}),
    )
    await p.client.close()
    p._client = AsyncOpenAI(api_key="offline", max_retries=0, http_client=http)
    try:
        await p.client.responses.create(model=p.default_model, input="ordinary text")
        assert not http.is_closed
        await fit(p, req)
        await p.complete(req)
        assert physical[0].endswith("/responses")
        assert physical[-1].endswith("/responses")
        assert any(path.endswith("/input_tokens") for path in physical)
        assert closed == [] and not http.is_closed
    finally:
        await p.close()
    assert closed  # Only the owning provider closes its transports.


@pytest.mark.asyncio
@pytest.mark.parametrize("mounted", [False, True])
@pytest.mark.parametrize("fitted", [False, True])
async def test_production_pool_execution_wrapper_is_refused(mounted, fitted):
    req = request()
    p, _, _, _ = provider(req)
    core = importlib.import_module("httpcore")
    observers, writes = [], []
    count_data = b'{"object":"response.input_tokens","input_tokens":30}'
    import json
    send_data = json.dumps(response(p.default_model)).encode()
    class Stream(core.AsyncMockStream):
        async def write(self, buffer, timeout=None):
            writes.append(buffer)
    class Backend(core.AsyncMockBackend):
        async def connect_tcp(self, *args, **kwargs):
            # This fixture's only legitimate pre-mutation request is count.
            data = count_data if fitted else send_data
            return Stream([b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                b"Connection: close\r\nContent-Length: " + str(len(data)).encode()
                + b"\r\n\r\n" + data])
    transport = httpx.AsyncHTTPTransport()
    pool = transport._pool
    pool._network_backend = Backend([])
    if mounted:
        from httpx._utils import URLPattern
        p.client._client._mounts[URLPattern("https://")] = transport
    else:
        p.client._client._transport = transport
    original = pool.handle_async_request
    async def observer(wire):
        observers.append(True)
        return await original(wire)
    try:
        if fitted:
            await fit(p, req)
        pool.handle_async_request = observer
        before = len(writes)
        with pytest.raises(InvalidRequestError):
            await p.complete(req)
        assert observers == [] and len(writes) == before
    finally:
        await p.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("sdk_default", [False, True])
async def test_real_vendor_pool_with_controlled_network_backend_remains_usable(sdk_default):
    import json
    req = request()
    p, _, _, _ = provider(req)
    module = httpx
    if sdk_default:
        base = next(cls for cls in DefaultAsyncHttpxClient.__mro__
                    if cls.__name__ == "AsyncClient")
        module = importlib.import_module(base.__module__.split(".")[0])
    core = importlib.import_module(
        type(module.AsyncHTTPTransport()._pool).__module__.split(".")[0]
    )
    writes, packets = [], []
    class Stream(core.AsyncMockStream):
        async def write(self, buffer, timeout=None):
            writes.append(buffer)
    class Backend(core.AsyncMockBackend):
        async def connect_tcp(self, *args, **kwargs):
            # Explicit fit plus complete's own native-count refresh precede send.
            data = (b'{"object":"response.input_tokens","input_tokens":30}'
                    if len(packets) < 2 else json.dumps(response(p.default_model)).encode())
            packets.append(True)
            return Stream([b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                b"Connection: close\r\nContent-Length: " + str(len(data)).encode()
                + b"\r\n\r\n" + data])
    transport = module.AsyncHTTPTransport()
    pool = transport._pool
    pool._network_backend = Backend([])
    await p.client.close()
    http = (DefaultAsyncHttpxClient if sdk_default else module.AsyncClient)(
        transport=transport, trust_env=False, follow_redirects=False,
    )
    p._client = AsyncOpenAI(api_key="offline", max_retries=0, http_client=http)
    try:
        assert type(pool) is core.AsyncConnectionPool
        await fit(p, req)
        await p.complete(req)
        assert any(b"/v1/responses/input_tokens" in w for w in writes)
        assert any(b"POST /v1/responses " in w for w in writes)
        assert not http.is_closed and http._transport._pool is pool
    finally:
        await p.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["transport_method", "pool_method", "closed"])
async def test_capability_validation_cannot_reopen_transport_after_check(monkeypatch, change):
    req = request()
    p, cap, _, _ = provider(req)
    observed = []
    async def forbidden(*args, **kwargs):
        observed.append(True)
        raise AssertionError("Private transport observer ran")
    transport = httpx.AsyncHTTPTransport()
    p.client._client._transport = transport
    validate = cap.validate
    def mutate(value, **kwargs):
        result = validate(value, **kwargs)
        if change == "transport_method":
            transport.handle_async_request = forbidden
        elif change == "pool_method":
            transport._pool.handle_async_request = forbidden
        else:
            # Source-independent synchronous simulation of an owner's close
            # state at this no-await callback boundary.
            from httpx._client import ClientState
            p.client._client._state = ClientState.CLOSED
        return result
    cap.validate = mutate
    try:
        with pytest.raises(InvalidRequestError):
            await p.complete(req)
        assert observed == []
    finally:
        await p.close()
        await transport.aclose()  # Also settle the simulated CLOSED-state case.


@pytest.mark.asyncio
@pytest.mark.parametrize("target", [
    "transport", "pool", "key", "base", "http_auth", "http_hook", "closed",
])
@pytest.mark.parametrize("cut", ["count", "send"])
async def test_real_sdk_first_platform_await_cannot_change_transport(monkeypatch, target, cut):
    import asyncio

    import openai._base_client as sdk_base

    req = request()
    p, _, _, _ = provider(req)
    observed = []
    physical = []

    async def bottom(_transport, wire):
        physical.append(wire.url.path)
        return httpx.Response(200, json=reply(p, wire))

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", bottom)
    transport = httpx.AsyncHTTPTransport()
    p.client._client._transport = transport

    async def observer(*args, **kwargs):
        observed.append(True)
        raise AssertionError("Private observer received first count")

    asyncify = sdk_base.asyncify

    def platform_wrapper(function):
        real = asyncify(function)

        async def wait_and_mutate(*args, **kwargs):
            result = await real(*args, **kwargs)
            await asyncio.sleep(0)
            if target == "transport":
                transport.handle_async_request = observer
            elif target == "pool":
                transport._pool.handle_async_request = observer
            elif target == "key":
                p.client.api_key = "other-account"
            elif target == "base":
                p.client.base_url = "https://other.invalid/v1/"
            elif target == "http_auth":
                p.client._client.auth = httpx.BasicAuth("other", "account")
            elif target == "http_hook":
                p.client._client.event_hooks["request"].append(observer)
            else:
                from httpx._client import ClientState
                p.client._client._state = ClientState.CLOSED
            return result

        return wait_and_mutate

    try:
        if cut == "send":
            await fit(p, req)
            emit = p.coordinator.hooks.emit

            async def arm(name, payload):
                if name == "llm:request":
                    monkeypatch.setattr(sdk_base, "asyncify", platform_wrapper)
                return await emit(name, payload)

            p.coordinator.hooks.emit = arm
        else:
            monkeypatch.setattr(sdk_base, "asyncify", platform_wrapper)
        with pytest.raises(InvalidRequestError):
            if cut == "count":
                await fit(p, req)
            else:
                await p.complete(req)
        assert observed == []
        assert all(path.endswith("/input_tokens") for path in physical)
    finally:
        await p.close()
        await transport.aclose()
