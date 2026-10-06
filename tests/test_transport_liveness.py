"""Real SDK + loopback sockets. No credentials, proxies, or hosted endpoints.

The accepted marker models possible provider work, not actual hosted effects.
OPENAI_SILENCE_SOAK_SECONDS=2100 selects the optional long soak.
"""

import asyncio
import json
import os
import socket
import time
from types import SimpleNamespace

import httpcore
import httpx
import openai
import pytest
from amplifier_core import llm_errors
from amplifier_core.message_models import ChatRequest, Message
from httpx._transports.default import map_httpcore_exceptions

from amplifier_module_provider_openai import OpenAIProvider
from amplifier_module_provider_openai._generation_errors import OUTCOME_UNKNOWN_MESSAGE


def response(status="completed"):
    return {
        "id": "resp_loopback", "object": "response", "created_at": 1,
        "model": "gpt-6-astra", "status": status, "error": None,
        "output": ([] if status != "completed" else [{
            "type": "message", "id": "msg_loopback", "role": "assistant",
            "status": "completed", "content": [{
                "type": "output_text", "text": "fixture answer", "annotations": [],
            }],
        }]),
        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15,
                  "input_tokens_details": {"cached_tokens": 0},
                  "output_tokens_details": {"reasoning_tokens": 0}},
        "parallel_tool_calls": True, "tools": [], "tool_choice": "auto",
    }


def sse(kind, seq, **data):
    return ("data: " + json.dumps({
        "type": kind, "sequence_number": seq, **data,
    }) + "\n\n").encode()


class Hooks:
    def __init__(self):
        self.events = []
        self.retry = asyncio.Event()

    async def emit(self, name, payload):
        self.events.append((name, payload))
        if name == "provider:retry":
            self.retry.set()


class Loopback:
    def __init__(self, mode="silent", *, streaming=False):
        self.mode, self.streaming = mode, streaming
        self.posts = self.accepted = self.counts = self.retrieves = 0
        self.accepted_event, self.activity_sent = asyncio.Event(), asyncio.Event()
        self.release = asyncio.Event()
        self.tasks, self.writers, self.errors = set(), set(), []
        self.payloads = []

    async def __aenter__(self):
        self.server = await asyncio.start_server(self.handle, "127.0.0.1", 0)
        self.port = self.server.sockets[0].getsockname()[1]
        return self

    async def __aexit__(self, *_):
        self.server.close()
        self.release.set()
        for writer in list(self.writers):
            writer.close()
        for task in list(self.tasks):
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        await self.server.wait_closed()
        assert not self.tasks
        assert not self.errors, self.errors
        print("loopback_evidence=" + json.dumps({
            "mode": self.mode, "streaming": self.streaming,
            "generation_posts": self.posts, "accepted": self.accepted,
            "count_posts": self.counts, "same_id_retrieves": self.retrieves,
            "fixture_tasks_remaining": len(self.tasks),
        }))

    async def send(self, writer, body, *, status=200, sse_body=False, truncated=False):
        writer.write(
            f"HTTP/1.1 {status} Fixture\r\n"
            f"Content-Type: {'text/event-stream' if sse_body else 'application/json'}\r\n"
            f"Content-Length: {len(body) + (100 if truncated else 0)}\r\n"
            "Connection: close\r\n\r\n".encode() + body
        )
        await writer.drain()

    async def handle(self, reader, writer):
        task = asyncio.current_task()
        self.tasks.add(task)
        self.writers.add(writer)
        try:
            header = await reader.readuntil(b"\r\n\r\n")
            first, *lines = header.decode().split("\r\n")
            method, path, _ = first.split()
            headers = dict(line.split(": ", 1) for line in lines if ": " in line)
            body = await reader.readexactly(int(headers.get("Content-Length", "0")))
            if path == "/v1/responses/input_tokens":
                self.counts += 1
                assert method == "POST"
                count_payload = json.loads(body)
                assert count_payload["model"] == "gpt-6-astra"
                assert "input" in count_payload and "stream" not in count_payload
                await self.send(writer, b'{"input_tokens":10}')
                return
            if method == "GET":
                assert path == "/v1/responses/resp_loopback"
                self.retrieves += 1
                if self.mode == "background_failure":
                    writer.transport.abort()
                else:
                    await self.send(writer, json.dumps(response()).encode())
                return
            if path == "/v1/refused":
                await self.send(writer, b'{"error":{"code":"rate_limit_exceeded"}}', status=429)
                return
            assert (method, path) == ("POST", "/v1/responses")
            self.posts += 1
            payload = json.loads(body)
            self.payloads.append(payload)
            assert "timeout" not in payload
            if self.mode == "refusal" and self.posts == 1:
                await self.send(writer, json.dumps({"error": {
                    "type": "rate_limit_error", "code": "rate_limit_exceeded",
                    "message": "private-rate-sentinel",
                }}).encode(), status=429)
                return
            self.accepted += 1
            self.accepted_event.set()
            if self.mode.startswith("redirect"):
                target = "/v1/refused" if self.mode == "redirect_refusal" else "/v1/unreachable"
                writer.write(
                    f"HTTP/1.1 307 Fixture\r\nLocation: https://api.openai.com{target}\r\n"
                    "Content-Length: 0\r\nConnection: close\r\n\r\n".encode()
                )
                await writer.drain()
                return
            if self.mode in {"reset", "eof"}:
                if self.mode == "reset":
                    writer.transport.abort()
                return
            if self.mode == "500":
                await self.send(writer, b'{"error":{"message":"private-error-sentinel"}}', status=500)
                return
            if self.mode in {"background", "background_failure"}:
                await self.send(writer, json.dumps(response("in_progress")).encode())
                return
            if self.mode == "malformed" and not self.streaming:
                await self.send(writer, b'{"output":BROKEN-private-error-sentinel')
                return
            if self.streaming:
                writer.write(
                    b"HTTP/1.1 200 Fixture\r\nContent-Type: text/event-stream\r\n"
                    b"Connection: close\r\n\r\n"
                )
                await writer.drain()
                if self.mode in {"activity", "reset_after_event", "truncated", "malformed", "post_event_error"}:
                    writer.write(sse("response.created", 0, response=response("in_progress")))
                    await writer.drain()
                    self.activity_sent.set()
                    if self.mode == "reset_after_event":
                        # Release lets the test prove the SDK consumed the event.
                        await self.release.wait()
                        writer.transport.abort()
                        return
                    if self.mode == "truncated":
                        return
                    if self.mode == "malformed":
                        writer.write(b"data: {BROKEN-private-error-sentinel\n\n")
                        await writer.drain()
                        return
                    if self.mode == "post_event_error":
                        writer.write(sse("error", 1, error={
                            "code": "rate_limit_exceeded", "type": "rate_limit_error",
                            "message": "private-error-sentinel",
                        }))
                        await writer.drain()
                        return
                if self.mode in {"silent", "activity"}:
                    await self.release.wait()
                if self.mode not in {"activity", "reset_after_event", "truncated", "malformed", "post_event_error"}:
                    writer.write(sse("response.created", 0, response=response("in_progress")))
                writer.write(sse("response.completed", 2, response=response()))
                await writer.drain()
            else:
                if self.mode == "silent":
                    await self.release.wait()
                await self.send(writer, json.dumps(response()).encode(),
                                truncated=self.mode == "truncated")
        except (ConnectionError, asyncio.IncompleteReadError, asyncio.CancelledError):
            pass
        except Exception as error:
            self.errors.append(type(error).__name__ + ": " + str(error))
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except ConnectionError:
                pass
            self.writers.discard(writer)
            self.tasks.discard(task)


class LoopbackTransport(httpx.AsyncBaseTransport):
    """Keep the native-count route eligible, but route ALL bytes to loopback."""

    def __init__(self, fixture, fault=None):
        self.fixture, self.fault = fixture, fault
        self.http = httpx.AsyncHTTPTransport(retries=0)
        self.limits = []
        self.dispatches = 0

    async def handle_async_request(self, request):
        assert request.url.host == "api.openai.com"
        self.limits.append(request.extensions["timeout"])
        generation = request.url.path == "/v1/responses"
        port = self.fixture.port
        if request.url.path == "/v1/unreachable":
            with socket.socket() as closed:
                closed.bind(("127.0.0.1", 0))
                port = closed.getsockname()[1]
        if generation:
            self.dispatches += 1
            if self.dispatches == 1 and self.fault:
                if self.fault == "connect_refused":
                    # Real HTTP socket failure before headers, not a mock error.
                    with socket.socket() as closed:
                        closed.bind(("127.0.0.1", 0))
                        port = closed.getsockname()[1]
                else:
                    # Deterministic phase-mapping proof, not a DNS/OS soak.
                    with map_httpcore_exceptions():
                        raise self.fault("private-phase-sentinel")
        url = request.url.copy_with(scheme="http", host="127.0.0.1", port=port)
        local = httpx.Request(request.method, url, headers=request.headers,
                             content=await request.aread(), extensions=request.extensions)
        return await self.http.handle_async_request(local)

    async def aclose(self):
        await self.http.aclose()


def provider_for(fixture, *, fault=None, **config):
    transport = LoopbackTransport(fixture, fault)
    sdk = openai.AsyncOpenAI(
        api_key="loopback-placeholder", max_retries=0,
        http_client=httpx.AsyncClient(transport=transport, trust_env=False,
                                     follow_redirects=fixture.mode.startswith("redirect")),
    )
    provider = OpenAIProvider(client=sdk, config={
        "default_model": "gpt-6-astra", "use_streaming": fixture.streaming,
        "max_retries": 2, "min_retry_delay": 0.01, "retry_jitter": False,
        "max_concurrent_requests": 0, **config,
    })
    hooks = Hooks()
    provider.coordinator = SimpleNamespace(hooks=hooks)
    return provider, transport, hooks


def request(**kwargs):
    return ChatRequest(messages=[Message(role="user", content="private-prompt-sentinel")], **kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_real_sdk_default_silent_wait_then_success(streaming):
    async with Loopback(streaming=streaming) as fixture:
        provider, transport, _ = provider_for(fixture)
        try:
            task = asyncio.create_task(provider.complete(request()))
            await asyncio.wait_for(fixture.accepted_event.wait(), 2)
            await asyncio.sleep(0.05)
            assert not task.done()
            assert fixture.posts == fixture.accepted == fixture.counts == 1
            fixture.release.set()
            result = await asyncio.wait_for(task, 2)
            assert result.usage.input_tokens == 10 and result.usage.output_tokens == 5
            assert result.content[0].text == "fixture answer"
            assert transport.limits[-1] == {"connect": 5.0, "pool": 5.0, "read": None, "write": None}
        finally:
            await provider.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("mode", ["silent", "reset", "eof", "500", "truncated", "malformed"])
async def test_real_sdk_ambiguous_acceptance_never_replayed(streaming, mode):
    async with Loopback(mode, streaming=streaming) as fixture:
        provider, _, _ = provider_for(fixture, timeout=0.05 if mode == "silent" else None)
        try:
            with pytest.raises(llm_errors.LLMError) as caught:
                await asyncio.wait_for(provider.complete(request()), 2)
            error = caught.value
            assert error.retryable is False
            assert error.request_outcome == "unknown" and error.effects == "may_have_occurred"
            assert str(error) == OUTCOME_UNKNOWN_MESSAGE
            assert error.__cause__ is not None
            assert fixture.posts == fixture.accepted == fixture.counts == 1
            if mode == "silent":
                assert isinstance(error, llm_errors.LLMTimeoutError)
        finally:
            await provider.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_real_sdk_qualifying_refusal_retries_without_prior_acceptance(streaming):
    async with Loopback("refusal", streaming=streaming) as fixture:
        provider, _, hooks = provider_for(fixture)
        try:
            result = await asyncio.wait_for(provider.complete(request()), 2)
            assert result.usage.output_tokens == 5
            assert fixture.posts == fixture.counts == 2 and fixture.accepted == 1
            retries = [data for name, data in hooks.events if name == "provider:retry"]
            assert len(retries) == 1 and retries[0]["attempt"] == 1
            assert "private-rate-sentinel" not in json.dumps(retries)
        finally:
            await provider.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["connect_refused", httpcore.ConnectTimeout, httpcore.PoolTimeout])
async def test_exact_pre_send_sdk_chain_allows_bounded_retry(fault):
    async with Loopback("success") as fixture:
        provider, transport, hooks = provider_for(fixture, fault=fault)
        try:
            await asyncio.wait_for(provider.complete(request()), 2)
            assert transport.dispatches == 2
            assert fixture.posts == fixture.accepted == 1 and fixture.counts == 2
            assert len([name for name, _ in hooks.events if name == "provider:retry"]) == 1
        finally:
            await provider.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_real_sdk_cancel_before_event_no_replacement_post(streaming):
    async with Loopback(streaming=streaming) as fixture:
        provider, _, _ = provider_for(fixture)
        try:
            task = asyncio.create_task(provider.complete(request()))
            await asyncio.wait_for(fixture.accepted_event.wait(), 2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert fixture.posts == fixture.accepted == 1
        finally:
            await provider.close()


@pytest.mark.asyncio
async def test_real_sdk_cancel_in_refusal_backoff_no_new_post():
    async with Loopback("refusal") as fixture:
        provider, _, hooks = provider_for(fixture, min_retry_delay=5)
        try:
            task = asyncio.create_task(provider.complete(request()))
            await asyncio.wait_for(hooks.retry.wait(), 2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert fixture.posts == fixture.counts == 1 and fixture.accepted == 0
        finally:
            await provider.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_optional_long_silence_soak(streaming):
    seconds = float(os.environ.get("OPENAI_SILENCE_SOAK_SECONDS", "0"))
    if seconds <= 1800:
        pytest.skip("Opt-in >30-minute wall-clock soak, not an ordinary CI test")
    async with Loopback(streaming=streaming) as fixture:
        provider, _, _ = provider_for(fixture)
        try:
            task = asyncio.create_task(provider.complete(request()))
            await asyncio.wait_for(fixture.accepted_event.wait(), 2)
            start = time.monotonic()
            await asyncio.sleep(seconds)
            assert not task.done() and fixture.posts == 1
            fixture.release.set()
            await asyncio.wait_for(task, 10)
            print(f"soak_elapsed_seconds={time.monotonic() - start}")
        finally:
            await provider.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("mode", ["redirect_connect", "redirect_refusal"])
async def test_redirect_hop_cannot_prove_original_generation_unaccepted(streaming, mode):
    async with Loopback(mode, streaming=streaming) as fixture:
        provider, _, hooks = provider_for(fixture)
        try:
            with pytest.raises(llm_errors.LLMError) as caught:
                await asyncio.wait_for(provider.complete(request()), 2)
            assert caught.value.request_outcome == "unknown"
            assert caught.value.retryable is False
            assert fixture.posts == fixture.accepted == 1
            assert not [name for name, _ in hooks.events if name == "provider:retry"]
        finally:
            await provider.close()