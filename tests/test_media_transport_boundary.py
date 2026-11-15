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
    "transport", "pool", "key", "base", "http_auth", "http_hook", "closed", "http_send",
])
@pytest.mark.parametrize("cut", ["count", "send"])
async def test_real_sdk_first_platform_await_cannot_change_transport(monkeypatch, target, cut):
    import asyncio

    import openai._base_client as sdk_base

    req = request()
    p, _, _, _ = provider(req)
    observed = []
    physical = []
    armed_at = []

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
            elif target == "http_send":
                p.client._client.send = observer
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
                    armed_at.append(len(physical))
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
        if cut == "count":
            assert physical == []
        else:
            assert len(armed_at) == 1
            assert len(physical) == armed_at[0]
            assert all(path.endswith("/input_tokens") for path in physical)
    finally:
        await p.close()
        await transport.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("malformed", [False, True])
async def test_private_sdk_copy_preserves_selected_response_validation(strict, malformed):
    from openai import APIResponseValidationError

    from amplifier_module_provider_openai import _media

    req = request()
    def handler(wire, cap, calls):
        data = response("gpt-6.1-sol")
        data.update(parallel_tool_calls=True, tool_choice="auto", tools=[])
        data["usage"].update(
            input_tokens_details={"cached_tokens": 0, "cache_write_tokens": 0},
            output_tokens_details={"reasoning_tokens": 0},
        )
        if malformed:
            data["output"] = 123
        return httpx.Response(200, json=data)
    p, _, calls, _ = provider(req, handler=handler)
    p.client._strict_response_validation = strict
    scope = _media.Scope(p, req, {})
    token = _media.CURRENT.set(scope)
    try:
        client = p._media_client()
        if strict and malformed:
            with pytest.raises(APIResponseValidationError):
                await client.responses.create(model=p.default_model, input="selected validation")
        else:
            result = await client.responses.create(model=p.default_model, input="selected validation")
            assert result.id == "resp_fixture"
        assert len(calls) == 1
        assert client._strict_response_validation is strict
        assert p.client._strict_response_validation is strict
    finally:
        _media.CURRENT.reset(token)
        await p.close()


@pytest.mark.asyncio
async def test_strict_response_policy_drift_after_count_refuses_without_send():
    req = request()
    p, _, calls, _ = provider(req)
    p.client._strict_response_validation = True
    try:
        await fit(p, req)
        before = len(calls)
        p.client._strict_response_validation = False
        with pytest.raises(InvalidRequestError):
            await p.complete(req)
        assert len(calls) == before
    finally:
        await p.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("cookie_at", ["count", "send"])
@pytest.mark.parametrize("route", ["normal", "raw", "stream"])
@pytest.mark.parametrize("sdk_default", [False, True])
async def test_response_cookies_do_not_change_selected_media_account(cookie_at, route, sdk_default):
    import json

    physical = []
    module = httpx
    if sdk_default:
        base = next(cls for cls in DefaultAsyncHttpxClient.__mro__
                    if cls.__name__ == "AsyncClient")
        module = importlib.import_module(base.__module__.split(".")[0])
    def handler(wire, cap, calls):
        physical.append((wire.url.path, dict(wire.headers)))
        data = ({"object": "response.input_tokens", "input_tokens": 30}
                if wire.url.path.endswith("/input_tokens") else response("gpt-6.1-sol"))
        cookie = "route=server-cookie; Path=/"
        headers = ({"set-cookie": cookie}
                   if (cookie_at == "count") == wire.url.path.endswith("/input_tokens") else {})
        if route == "stream" and not wire.url.path.endswith("/input_tokens"):
            headers["content-type"] = "text/event-stream"
            return module.Response(200, headers=headers, text="".join(
                "data: " + json.dumps(event) + "\n\n" for event in (
                    {"type": "response.created", "response": {**data, "output": []}},
                    {"type": "response.completed", "response": data},
                )
            ))
        return module.Response(200, json=data, headers=headers)
    req = request()
    p, cap, _, _ = provider(req, handler=handler, stream=route == "stream")
    if sdk_default:
        await p.client.close()
        p._client = AsyncOpenAI(
            api_key="offline", max_retries=0,
            http_client=DefaultAsyncHttpxClient(
                trust_env=False, follow_redirects=False,
                transport=module.MockTransport(lambda wire: handler(wire, cap, [])),
            ),
        )
    if route == "raw":
        p.extra_request_params["tools"] = [{"type": "computer"}]
    try:
        p.client._client.cookies.set("selected", "account-A")
        await fit(p, req)
        await p.complete(req)
        assert sum(not path.endswith("/input_tokens") for path, _ in physical) == 1
        assert p.client._client.cookies.get("route") is None
        next_request = request()
        cap.request = next_request
        await fit(p, next_request)
        await p.complete(next_request)
        assert sum(not path.endswith("/input_tokens") for path, _ in physical) == 2
        assert all(headers.get("cookie") == "selected=account-A" for _, headers in physical)
        assert not p.client._client.is_closed
    finally:
        await p.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("change_at", ["before_count", "before_send"])
async def test_explicit_cookie_account_drift_still_refuses(change_at):
    req = request()
    p, _, calls, _ = provider(req)
    try:
        p.client._client.cookies.set("selected", "account-A")
        if change_at == "before_send":
            await fit(p, req)
        else:
            # Explicit mutation after the original scope snapshot.
            from amplifier_module_provider_openai import _media
            scope = _media.Scope(p, req, {})
            p.client._client.cookies.set("selected", "account-B")
            with pytest.raises(InvalidRequestError):
                scope.validate("count")
            return
        before = len(calls)
        p.client._client.cookies.set("selected", "account-B")
        with pytest.raises(InvalidRequestError):
            await p.complete(req)
        assert len(calls) == before
    finally:
        await p.close()


@pytest.mark.asyncio
async def test_no_cap_canonical_image_count_copy_pins_selected_connection(monkeypatch):
    captured = []
    def handler(wire, cap, calls):
        captured.append((str(wire.url), dict(wire.headers)))
    req = request()
    p, _, calls, _ = provider(req, handler=handler)
    p.coordinator.get_capability = lambda name: None
    client = p.client
    client.organization = None
    client.project = None
    client._custom_headers = {"x-selected": "original"}
    client._custom_query = {"selected": "original"}
    try:
        monkeypatch.setenv("OPENAI_ORG_ID", "ambient-other-org")
        monkeypatch.setenv("OPENAI_PROJECT_ID", "ambient-other-project")
        monkeypatch.setenv("OPENAI_CUSTOM_HEADERS", "Host: other.invalid\nx-selected: ambient")
        await fit(p, req)
        await p.complete(req)
        assert len(calls) >= 2
        for url, headers in captured:
            assert "selected=original" in url and "other.invalid" not in url
            assert headers["host"] == "api.openai.com"
            assert headers["x-selected"] == "original"
            assert "openai-organization" not in headers and "openai-project" not in headers
            assert headers["authorization"] == "Bearer offline"
    finally:
        await p.close()
