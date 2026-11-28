"""Bind positive trace evidence to SDK error.request, not auxiliary traffic."""
import importlib.util
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from amplifier_core import llm_errors as errors

spec = importlib.util.spec_from_file_location("request_boundary", Path(__file__).with_name("test_network_boundary.py"))
boundary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(boundary)
matrix, network, httpx, NAME = boundary.matrix, boundary.network, boundary.httpx, boundary.NAME
local_only_sockets = matrix.local_only_sockets
VARIANTS = ["aux_failed_replaced", "aux_success", "response_aux", "auth_aux", "other_client_replaced",
            "wrapped", "before", "late_hook", "aux_only"]


@asynccontextmanager
async def variant_provider(variant, streaming):
    async with AsyncExitStack() as stack:
        refused = await stack.enter_async_context(matrix.endpoint("refused"))
        success, _ = await stack.enter_async_context(boundary.local_response(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nOK"))
        partial = variant in {"aux_failed_replaced", "response_aux", "auth_aux", "other_client_replaced", "late_hook"}
        if partial:
            url, counts = await stack.enter_async_context(boundary.local_response())
        else:
            url, counts = refused, {"sends": 0}
        def opaque(req):
            raise httpx.ConnectError("generation has no trace report", request=req)
        http = httpx.AsyncClient(transport=httpx.MockTransport(opaque)) if variant == "aux_only" else httpx.AsyncClient()
        other = httpx.AsyncClient()
        stack.push_async_callback(other.aclose)
        seen = []
        counts.update(aux=0, partial=0, generation_hooks=0)
        async def auxiliary(client, good=False):
            counts["aux"] += 1
            try:
                await client.get((success if good else refused) + "/auxiliary")
            except httpx.HTTPError:
                pass
        async def before(req):
            if req.url.path.endswith("auxiliary"):
                return
            counts["generation_hooks"] += 1
            seen.append(req)
            if variant == "before":
                async def caller(event, info):
                    counts["caller_events"] = counts.get("caller_events", 0) + 1
                req.extensions["trace"] = caller
        http.event_hooks["request"].append(before)
        p = await stack.enter_async_context(boundary.supplied_client(http, streaming, url))
        network.watch_response(http)
        network.watch_response(other)
        # The returned observations expose only request identity booleans/counts.
        async def after(req):
            if req.url.path.endswith("auxiliary"):
                return
            if variant in {"aux_failed_replaced", "late_hook", "other_client_replaced", "aux_success", "aux_only"}:
                await auxiliary(other if variant in {"other_client_replaced", "aux_only"} else http, variant == "aux_success")
            if variant in {"aux_failed_replaced", "late_hook", "other_client_replaced"}:
                async def silent(event, info):
                    pass
                req.extensions["trace"] = silent
            if variant == "wrapped":
                trace = req.extensions["trace"]
                async def wrapped(event, info):
                    await trace(event, info)
                req.extensions["trace"] = wrapped
        http.event_hooks["request"].append(after)
        async def response_hook(response):
            if response.request.url.path.endswith("auxiliary"):
                return
            if variant == "response_aux":
                await auxiliary(http)
            if partial and variant != "auth_aux":
                async for part in response.aiter_raw():
                    counts["partial"] += 1
                    break
                raise httpx.ReadError("generation partial bytes", request=response.request)
        http.event_hooks["response"].append(response_hook)
        if variant == "auth_aux":
            class Auth(httpx.Auth):
                async def async_auth_flow(self, req):
                    if req.url.path.endswith("auxiliary"):
                        yield req
                        return
                    response = yield req
                    await auxiliary(other)
                    async for part in response.aiter_raw():
                        counts["partial"] += 1
                        break
                    raise httpx.ReadError("auth partial response", request=req)
            http.auth = Auth()
        yield p, counts, seen


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("variant", VARIANTS)
async def test_auxiliary_variants(variant, streaming):
    async with variant_provider(variant, streaming) as (p, counts, requests):
        with pytest.raises(errors.LLMError) as caught:
            await p.complete(matrix.request())
        assert type(caught.value) is (errors.NetworkError if variant == "before" else errors.LLMError)
        assert counts["generation_hooks"] == 1
        if variant in {"aux_failed_replaced", "response_aux", "auth_aux", "other_client_replaced", "late_hook"}:
            assert counts["sends"] == counts["partial"] == 1
        assert caught.value.__cause__.request is requests[-1]


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("sdk_retries", [0, 1])
async def test_error_request_is_last_transport_request(streaming, sdk_retries, monkeypatch):
    hooks, transport = [], []
    async with matrix.failing_provider("dns", streaming) as p:
        p._client = p.client.with_options(max_retries=sdk_retries)
        async def hook(req):
            hooks.append(req)
            previous = req.extensions.get("trace")
            # Capture through a caller trace installed BEFORE evidence.
            async def caller(event, info):
                if event == "connection.connect_tcp.started":
                    transport.append(req)
                if previous is not None:
                    await previous(event, info)
            req.extensions["trace"] = caller
        p.client._client.event_hooks["request"].append(hook)
        monkeypatch.setattr("asyncio.sleep", AsyncMock())
        with pytest.raises(errors.NetworkError) as caught:
            await p.complete(matrix.request())
        assert len(hooks) == len(transport) == sdk_retries + 1
        assert caught.value.__cause__.request is hooks[-1] is transport[-1]
        assert len({id(req) for req in hooks}) == sdk_retries + 1


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_redirect_error_request_is_conservative(streaming, monkeypatch):
    observed, requests = [], []
    original = network.can_promote
    def inspect(error):
        state = network._response_state.get()
        observed.append((error.request is requests[0], error.request is requests[-1],
                         len(requests), state["received"], original(error)))
        return original(error)
    monkeypatch.setattr(network, "can_promote", inspect)
    async with boundary.regression_provider("redirect", streaming) as (p, _counts):
        async def remember(req):
            requests.append(req)
        p.client._client.event_hooks["request"].append(remember)
        with pytest.raises(errors.LLMError) as caught:
            await p.complete(matrix.request())
        assert type(caught.value) is errors.LLMError
        assert observed == [(True, False, 2, True, False)]
        assert requests[0].extensions["trace"] is not requests[1].extensions["trace"]


@pytest.mark.asyncio
async def test_absent_request_and_foreign_trace_are_unknown():
    with network.observe_response():
        req = httpx.Request("GET", "http://127.0.0.1:9")
        await network.trace_request(req)
        trace = req.extensions["trace"]
        await trace("connection.connect_tcp.started", {})
        error = httpx.ConnectError("test", request=req)
        assert network.can_promote(error)
        assert not network.can_promote(RuntimeError("no request"))
        async def wrapped(event, info):
            await trace(event, info)
        req.extensions["trace"] = wrapped
        assert not network.can_promote(error)


@asynccontextmanager
async def no_request_provider(streaming):
    async with matrix.failing_provider("dns", streaming) as p:
        counts = {"invocations": 0}
        error = matrix.openai.APIConnectionError(request=None)
        error.__cause__ = httpx.ConnectError("opaque SDK failure")
        def fail(**kwargs):
            counts["invocations"] += 1
            raise error
        if streaming:
            p.client.responses.stream = fail
        else:
            p.client.responses.create = AsyncMock(side_effect=fail)
        yield p, counts


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_sdk_error_with_no_request_keeps_old_class(streaming):
    async with no_request_provider(streaming) as (p, counts):
        with pytest.raises(errors.LLMError) as caught:
            await p.complete(matrix.request())
        assert type(caught.value) is errors.LLMError
        assert caught.value.__cause__.request is None
        assert counts["invocations"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("response", [b"HTT", b"HTTP/1.1 100 Continue\r\n\r\n", b"HTTP/1.1 102 Processing\r\n\r\n"])
@pytest.mark.parametrize("streaming", [False, True])
async def test_incomplete_or_interim_headers_are_pre_answer(response, streaming):
    async with boundary.local_response(response) as (url, counts):
        p = matrix.provider(url, streaming)
        try:
            with pytest.raises(errors.NetworkError):
                await p.complete(matrix.request())
            assert counts["sends"] == 1
        finally:
            await p.close()
