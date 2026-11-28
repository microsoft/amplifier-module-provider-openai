"""A text call must not invalidate an in-flight media request binding."""
import asyncio
import importlib.util
import json
from pathlib import Path

import httpx
import pytest
from amplifier_core import llm_errors as errors
from amplifier_core.message_models import ChatRequest, Message
from amplifier_core.utils.retry import RetryConfig

from amplifier_module_provider_openai import _transport as network

spec = importlib.util.spec_from_file_location("media_fixtures5", Path(__file__).with_name("test_media_request_seams.py"))
media = importlib.util.module_from_spec(spec)
spec.loader.exec_module(media)


@pytest.mark.asyncio
@pytest.mark.parametrize("http_mode", ["legacy", "sdk-injected", "sdk-default"])
@pytest.mark.parametrize("scenario", ["between_count_send", "after_503", "waiting_retry"])
async def test_text_does_not_break_media(http_mode, scenario, monkeypatch):
    req = media.request()
    replies = None if scenario == "between_count_send" else [
        httpx.Response(200, json={"input_tokens": 12}),
        httpx.Response(503, json={"error": {"message": "local overload", "type": "server_error"}}),
    ]
    p, cap, wires, _events = media.fixture(req, http_mode=http_mode, replies=replies)
    text = ChatRequest(messages=[Message(role="user", content="local text naming request")])
    original_emit = p.coordinator.hooks.emit
    ran = []
    async def run_text():
        if ran:
            return
        ran.append(True)
        await p.complete(text)
    async def emit(name, data):
        await original_emit(name, data)
        if scenario == "between_count_send" and name == "llm:request":
            await run_text()
    p.coordinator.hooks.emit = emit
    try:
        if scenario == "after_503":
            with pytest.raises(errors.ProviderUnavailableError):
                await p.complete(req)
            await run_text()
            result = await p.complete(req)
        elif scenario == "waiting_retry":
            p._retry_config = RetryConfig(max_retries=1, initial_delay=0.01, jitter=False)
            waiting, resume = asyncio.Event(), asyncio.Event()
            async def sleep(delay):
                waiting.set()
                await resume.wait()
            monkeypatch.setattr("amplifier_core.utils.retry.asyncio.sleep", sleep)
            task = asyncio.create_task(p.complete(req))
            await asyncio.wait_for(waiting.wait(), 5)
            await run_text()
            resume.set()
            result = await asyncio.wait_for(task, 5)
        else:
            result = await p.complete(req)
        assert ran == [True]
        assert result.content[0].text == "OK"
        generation = [body for path, body, _ in wires if path.endswith("/responses")]
        assert len(generation) == (2 if scenario == "between_count_send" else 3)
        assert sum(media.PIXELS in json.dumps(body) for body in generation) == (1 if scenario == "between_count_send" else 2)
        assert all(same for same, _, _ in cap.calls)
        assert p.client._client.event_hooks["request"].count(network.trace_request) == 1
    finally:
        await p.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("http_mode", ["legacy", "sdk-injected", "sdk-default"])
@pytest.mark.parametrize("name", ["trace_request", "caller_hook"])
async def test_media_binding_never_ignores_caller_hook_by_name(http_mode, name):
    req = media.request()
    p, _cap, wires, _events = media.fixture(req, http_mode=http_mode)
    original_emit = p.coordinator.hooks.emit
    async def caller(request):
        pass
    caller.__name__ = name
    assert caller is not network.trace_request
    async def emit(event, data):
        await original_emit(event, data)
        if event == "llm:request":
            p.client._client.event_hooks["request"].append(caller)
    p.coordinator.hooks.emit = emit
    try:
        with pytest.raises(errors.InvalidRequestError):
            await p.complete(req)
        assert [path for path, _body, _headers in wires] == ["/v1/responses/input_tokens"]
    finally:
        await p.close()


@pytest.mark.asyncio
async def test_keepalive_then_evidence_identity_and_options():
    import httpx2
    called, options = [], []
    class Socket:
        def setsockopt(self, *args):
            options.append(args)
    class Stream:
        def get_extra_info(self, name):
            return Socket()
    async def caller(event, info):
        called.append(event)
    with network.observe_response():
        req = httpx2.Request("POST", "http://127.0.0.1:9", extensions={"trace": caller})
        # This order must match the real client's public request hook order.
        await network.configure_request(req)
        await network.trace_request(req)
        for _ in range(2):
            await req.extensions["trace"]("connection.connect_tcp.complete", {"return_value": Stream()})
        error = httpx2.ConnectError("local loss", request=req)
        assert not network.can_promote(error)  # Connect completion is not a failure.
        reported = RuntimeError("transport connect failure")
        error.__cause__ = reported
        await req.extensions["trace"]("connection.connect_tcp.failed", {"exception": reported})
        assert network.can_promote(error)
        assert called == ["connection.connect_tcp.complete"] * 2 + ["connection.connect_tcp.failed"]
        assert options == network.keepalive_options() * 2
        hop = httpx2.Request("POST", "http://127.0.0.1:9/redirect", extensions=dict(req.extensions))
        await network.configure_request(hop)
        await network.trace_request(hop)
        await hop.extensions["trace"]("connection.connect_tcp.complete", {"return_value": Stream()})
        assert called == ["connection.connect_tcp.complete"] * 2 + ["connection.connect_tcp.failed", "connection.connect_tcp.complete"]
        assert options == network.keepalive_options() * 3
        hop_error = httpx2.ConnectError("hop", request=hop)
        hop_error.__cause__ = reported
        assert not network.can_promote(hop_error)  # Hop has not reported failure yet.
        await hop.extensions["trace"]("connection.connect_tcp.failed", {"exception": reported})
        assert network.can_promote(hop_error)
    # Test construction order as well: swapping the two module hooks makes
    # the outermost callable foreign to evidence and fails this assertion.
    from amplifier_module_provider_openai import OpenAIProvider
    p = OpenAIProvider(api_key="not-a-real-key", config={"base_url": "http://127.0.0.1:9"})
    try:
        network.watch_response(p.client._client)
        hooks = p.client._client.event_hooks["request"]
        assert hooks.index(network.configure_request) < hooks.index(network.trace_request)
    finally:
        await p.close()
