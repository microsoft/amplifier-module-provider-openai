"""Positive trace evidence replaces all client/transport type inspection."""
import importlib.util
from pathlib import Path

import pytest
from amplifier_core import llm_errors as errors

spec = importlib.util.spec_from_file_location("trace_boundary", Path(__file__).with_name("test_network_boundary.py"))
boundary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(boundary)
matrix, network, httpx, NAME = boundary.matrix, boundary.network, boundary.httpx, boundary.NAME
local_only_sockets = matrix.local_only_sockets


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("case", ["refused", "closed", "dns", "sdk_timeout", "certificate", "unsupported_protocol", "invalid_url"])
async def test_sdk_failures_require_actual_trace_events(streaming, case):
    async with matrix.failing_provider(case, streaming) as p:
        with pytest.raises(errors.LLMError) as caught:
            await p.complete(matrix.request())
        assert type(caught.value) is matrix.CASES[case]


@pytest.mark.asyncio
@pytest.mark.parametrize("event", ["http11.receive_response_headers.complete", "http11.receive_response_body.started",
                                       "http11.receive_response_body.failed", "http2.receive_response_headers.complete"])
async def test_response_evidence_blocks_promotion(event):
    with network.observe_response():
        req = httpx.Request("POST", "http://127.0.0.1:9")
        await network.trace_request(req)
        trace = req.extensions["trace"]
        report = RuntimeError("reported transport failure")
        error = httpx.ReadError("test", request=req)
        error.__cause__ = report
        await trace("connection.connect_tcp.failed", {"exception": report})
        assert network.can_promote(error)
        await trace(event, {})
        # A later allowed failure must not erase the attempt's response fact.
        await trace("connection.connect_tcp.failed", {"exception": report})
        assert not network.can_promote(error)


@pytest.mark.asyncio
async def test_no_evidence_and_http2_are_unknown():
    with network.observe_response():
        assert not network.can_promote(RuntimeError("none"))
        req = httpx.Request("POST", "http://127.0.0.1:9")
        await network.trace_request(req)
        await req.extensions["trace"]("http2.send_request_headers.started", {})
        assert not network.can_promote(RuntimeError("unqualified HTTP2"))


@pytest.mark.asyncio
async def test_caller_trace_chained_once_for_each_event_and_redirect_hook():
    called = []
    async def caller(event, info):
        called.append(event)
    with network.observe_response():
        req = httpx.Request("POST", "http://127.0.0.1:9", extensions={"trace": caller})
        await network.trace_request(req)
        await network.trace_request(req)  # Public redirect/repeated hook invocation.
        for event in ("connection.connect_tcp.started", "http11.receive_response_headers.complete"):
            await req.extensions["trace"](event, {})
        assert called == ["connection.connect_tcp.started", "http11.receive_response_headers.complete"]
        assert not network.can_promote(RuntimeError("response"))


@pytest.mark.asyncio
async def test_caller_trace_exception_passes_unchanged():
    original = httpx.ReadError("caller trace")
    async def caller(event, info):
        raise original
    with network.observe_response():
        req = httpx.Request("POST", "http://127.0.0.1:9", extensions={"trace": caller})
        await network.trace_request(req)
        with pytest.raises(httpx.ReadError) as caught:
            await req.extensions["trace"]("http11.receive_response_headers.complete", {})
        assert caught.value is original
        assert not network.can_promote(original)


@pytest.mark.asyncio
async def test_installation_only_touches_public_request_hooks():
    class PublicOnly:
        def __init__(self):
            self.event_hooks = {"request": [], "response": []}
        def __getattr__(self, name):
            raise AssertionError("private access: " + name)
    http = PublicOnly()
    network.watch_response(http)
    network.watch_response(http)
    assert http.event_hooks == {"request": [network.trace_request], "response": []}
    network.watch_response(object())  # Missing public interface is harmless.


@pytest.mark.asyncio
async def test_evidence_error_is_unknown():
    with network.observe_response():
        req = httpx.Request("POST", "http://127.0.0.1:9")
        await network.trace_request(req)
        await req.extensions["trace"](None, {})
        assert not network.can_promote(RuntimeError("invalid event"))


@pytest.mark.asyncio
async def test_hook_not_installed_or_trace_replaced_does_not_promote():
    async with matrix.failing_provider("dns") as p:
        http = p.client._client
        # Install then replace after our hook, not before it.
        network.watch_response(http)
        # Async callback required by httpcore: replacement itself works but
        # reports no events to the module.
        async def silent(event, info):
            pass
        async def replace(req):
            req.extensions["trace"] = silent
        http.event_hooks["request"].append(replace)
        with pytest.raises(errors.LLMError) as caught:
            await p.complete(matrix.request())
        assert type(caught.value) is errors.LLMError
