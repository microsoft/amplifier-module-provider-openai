"""Real trace integration: callers, redirects, legacy HTTPX, proxies, media."""
import importlib.util
from pathlib import Path

import pytest
from amplifier_core import llm_errors as errors
from amplifier_core.message_models import ChatRequest, ImageBlock, Message

spec = importlib.util.spec_from_file_location("trace_integration_boundary", Path(__file__).with_name("test_network_boundary.py"))
boundary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(boundary)
matrix, network, httpx, NAME = boundary.matrix, boundary.network, boundary.httpx, boundary.NAME
local_only_sockets = matrix.local_only_sockets


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_real_trace_chains_to_caller_once_before_hooks(streaming):
    events = []
    decisions = []
    async def caller(event, info):
        events.append(event)
        if event.endswith("receive_response_headers.complete"):
            decisions.append(network.can_promote(RuntimeError("response")))
    async def request_hook(req):
        req.extensions["trace"] = caller
    async with boundary.local_response() as (url, counts):
        http = httpx.AsyncClient(event_hooks={"request": [request_hook]})
        async with boundary.supplied_client(http, streaming, url) as p:
            with pytest.raises(errors.LLMError) as caught:
                await p.complete(matrix.request())
            assert type(caught.value) is errors.LLMError
            assert counts["sends"] == 1
    assert "connection.connect_tcp.started" in events
    assert "http11.receive_response_headers.complete" in events
    assert len(events) == len(set(events))
    assert decisions == [False]  # Our response fact is set before caller code.


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("case", ["unsupported_protocol", "invalid_url"])
async def test_config_fault_has_no_transport_report(streaming, case, monkeypatch):
    seen = []
    original = network.can_promote
    def inspect(error):
        state = network._response_state.get()
        seen.append(dict(state))
        return original(error)
    monkeypatch.setattr(network, "can_promote", inspect)
    async with matrix.failing_provider(case, streaming) as p:
        with pytest.raises(errors.LLMError) as caught:
            await p.complete(matrix.request())
        assert type(caught.value) is errors.LLMError
    # InvalidURL can bypass the SDK's APIError branch; inspect only if called.
    assert all(not any(r["seen"] for r in state["requests"].values()) for state in seen)


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_legacy_httpx_default_transport_is_reported(streaming):
    import httpx as legacy
    async with matrix.endpoint("refused") as url:  # noqa: SIM117
        async with boundary.supplied_client(legacy.AsyncClient(), streaming, url) as p:
            with pytest.raises(errors.NetworkError):
                await p.complete(matrix.request())


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("mode", ["proxy_refused", "target_refused", "closed_tunnel"])
async def test_local_connect_proxy(mode, streaming):
    events = []
    async def caller(event, info):
        events.append(event)
    async def install(req):
        req.extensions["trace"] = caller
    if mode == "proxy_refused":
        async with matrix.endpoint("refused") as proxy:
            http = httpx.AsyncClient(proxy=proxy, verify=False, event_hooks={"request": [install]})
            async with boundary.supplied_client(http, streaming, "https://127.0.0.1:9") as p:
                with pytest.raises(errors.NetworkError):
                    await p.complete(matrix.request())
        assert "connection.connect_tcp.failed" in events
        assert not any(e.endswith("receive_response_headers.complete") for e in events)
        return
    response = (b"HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\n\r\n" if mode == "target_refused" else
                b"HTTP/1.1 200 Connection Established\r\n\r\n")
    async with boundary.local_response(response) as (proxy, counts):
        http = httpx.AsyncClient(proxy=proxy, verify=False, event_hooks={"request": [install]})
        async with boundary.supplied_client(http, streaming, "https://127.0.0.1:9") as p:
            with pytest.raises(errors.LLMError) as caught:
                await p.complete(matrix.request())
            assert type(caught.value) is errors.LLMError
            assert counts["sends"] == 1
    assert "http11.receive_response_headers.complete" in events
    if mode == "closed_tunnel":
        assert any(e.endswith("start_tls.failed") for e in events)


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_image_bearing_request_is_not_promoted(streaming):
    image = ChatRequest(messages=[Message(role="user", content=[ImageBlock(source={
        "type": "base64", "media_type": "image/png", "data": "bG9jYWwtaW1hZ2U=",
    })])])
    async with matrix.failing_provider("dns", streaming) as p:
        with pytest.raises(errors.LLMError) as caught:
            await p._complete_chat_request(image)
        assert type(caught.value) is errors.LLMError
        assert caught.value.retryable is True
