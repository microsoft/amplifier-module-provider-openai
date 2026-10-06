"""Real SDK + loopback sockets. No credentials, proxies, or hosted endpoints.

The accepted marker models possible provider work, not actual hosted effects.
OPENAI_SILENCE_SOAK_SECONDS=2100 selects the optional long soak.
"""

import asyncio
import json
import os
import socket
import time
from contextvars import ContextVar
from types import SimpleNamespace

import httpcore
import httpx
import openai
import pytest
from amplifier_core import llm_errors
from amplifier_core.message_models import ChatRequest, Message
from httpx._transports.default import map_httpcore_exceptions

from amplifier_module_provider_openai import OpenAIProvider
from amplifier_module_provider_openai import _wait_observation as observation_module
from amplifier_module_provider_openai._generation_errors import (
    OUTCOME_UNKNOWN_MESSAGE,
    InjectedClientConfigurationError,
    RequestOutcomeUnknownError,
)
from amplifier_module_provider_openai._wait_observation import _WaitObserver


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
        self.activity = asyncio.Event()

    async def emit(self, name, payload):
        self.events.append((name, payload))
        if name == "provider:retry":
            self.retry.set()
        if name == "llm:progress" and payload["observation"] == "response_activity":
            self.activity.set()


class Loopback:
    def __init__(self, mode="silent", *, streaming=False, failed_response=None, initial_response=None):
        self.mode, self.streaming = mode, streaming
        self.failed_response = failed_response
        self.initial_response = initial_response
        self.posts = self.accepted = self.counts = self.retrieves = 0
        self.accepted_event, self.activity_sent = asyncio.Event(), asyncio.Event()
        self.release = asyncio.Event()
        self.pending_continuation = asyncio.Event()
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
                await self.send(writer, b'{"input_tokens":99999999}' if self.mode == "count_overflow"
                                else b'{"input_tokens":10}')
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
            if path == "/v1/quota":
                await self.send(writer, b'{"error":{"code":"insufficient_quota"}}', status=429)
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
            if self.mode in {"quota", "unknown_429", "html_429"}:
                body = (b"<html>fixture challenge</html>" if self.mode == "html_429"
                        else json.dumps({"error": {
                            "code": ("insufficient_quota" if self.mode == "quota" else "unknown"),
                            "message": "private-quota-sentinel",
                        }}).encode())
                await self.send(writer, body, status=429)
                return
            self.accepted += 1
            self.accepted_event.set()
            if self.mode.startswith("redirect"):
                target = ("/v1/refused" if self.mode == "redirect_refusal" else
                          "/v1/quota" if self.mode == "redirect_quota" else "/v1/unreachable")
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
            if self.mode in {"continuation", "continuation_pending", "tool_truncation"} and self.posts == 1:
                incomplete = self.initial_response or response("incomplete")
                incomplete["incomplete_details"] = {"reason": "max_output_tokens"}
                if self.mode == "tool_truncation":
                    incomplete["output"] = [{
                        "type": "function_call", "id": "fc_fixture",
                        "call_id": "call_fixture", "name": "fixture_tool",
                        "arguments": '{"private-tool-sentinel":', "status": "incomplete",
                    }]
                else:
                    incomplete["output"] = response()["output"]
                await self.send(writer, json.dumps(incomplete).encode())
                return
            if self.mode == "continuation_pending":
                self.pending_continuation.set()
                await self.release.wait()
            if self.mode == "malformed" and not self.streaming:
                await self.send(writer, b'{"output":BROKEN-private-error-sentinel')
                return
            if self.mode == "private_payload":
                private_response = response()
                private_response["output"].extend([
                    {"type": "reasoning", "id": "rs_fixture",
                     "summary": [{"type": "summary_text", "text": "private-thinking-sentinel"}],
                     "encrypted_content": "private-encrypted-sentinel"},
                    {"type": "function_call", "id": "fc_fixture", "call_id": "call_fixture",
                     "name": "fixture_tool", "arguments": '{"secret":"private-tool-sentinel"}',
                     "status": "completed"},
                ])
                await self.send(writer, json.dumps(private_response).encode())
                return
            if self.streaming:
                writer.write(
                    b"HTTP/1.1 200 Fixture\r\nContent-Type: text/event-stream\r\n"
                    b"Connection: close\r\n\r\n"
                )
                await writer.drain()
                if self.mode == "failed_usage":
                    writer.write(sse("response.created", 0, response=response("in_progress")))
                    writer.write(sse("response.failed", 1, response=self.failed_response))
                    await writer.drain()
                    return
                if self.mode == "post_event_quota":
                    writer.write(sse("response.created", 0, response=response("in_progress")))
                    writer.write(sse("error", 1, error={
                        "code": "insufficient_quota", "message": "private-quota-sentinel",
                    }))
                    await writer.drain()
                    return
                if self.mode in {"partial_eof", "partial_cancel"}:
                    writer.write(sse("response.created", 0, response=response("in_progress")))
                    writer.write(sse("response.output_item.added", 1, output_index=0, item={
                        "type": "message", "id": "msg_partial", "role": "assistant",
                        "status": "in_progress", "content": [],
                    }))
                    writer.write(sse("response.content_part.added", 2, output_index=0,
                                     content_index=0, item_id="msg_partial", part={
                                         "type": "output_text", "text": "", "annotations": [],
                                     }))
                    writer.write(sse("response.output_text.delta", 3, output_index=0,
                                     content_index=0, item_id="msg_partial",
                                     delta="private-partial-text", logprobs=[]))
                    await writer.drain()
                    if self.mode == "partial_cancel":
                        await self.release.wait()
                    return
                if self.mode == "comment_only":
                    writer.write(b": fixture heartbeat comment, not a parsed response\n\n")
                    await writer.drain()
                if self.mode in {"observation_success", "observation_error", "observation_cancel"}:
                    writer.write(sse("response.created", 0, response=response("in_progress")))
                    writer.write(sse("response.in_progress", 1, response=response("in_progress")))
                    await writer.drain()
                    await self.release.wait()
                    if self.mode == "observation_error":
                        writer.transport.abort()
                        return
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
                if self.mode in {"silent", "activity", "comment_only"}:
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
        except Exception as error:  # noqa: BLE001 - retain fixture failures for teardown
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
        self.generation_limits = []
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
            self.generation_limits.append(request.extensions["timeout"])
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


def provider_for(fixture, *, fault=None, sdk_retries=0, **config):
    transport = LoopbackTransport(fixture, fault)
    sdk = openai.AsyncOpenAI(
        api_key="loopback-placeholder", max_retries=sdk_retries,
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


def assert_no_provider_delivery_tasks():
    assert not [task for task in asyncio.all_tasks()
                if task.get_coro().__qualname__ == "_settle_optional.<locals>.delivery"]


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_real_sdk_quota_refusal_is_not_an_unknown_billable_attempt(streaming):
    async with Loopback("quota", streaming=streaming) as fixture:
        provider, transport, hooks = provider_for(fixture)
        costs = []
        provider._add_cost = costs.append
        try:
            with pytest.raises(llm_errors.RateLimitError) as caught:
                await provider.complete(request())
            error = caught.value
            assert error.retryable is False
            assert error.request_outcome == "not_accepted" and error.effects == "none"
            assert error.vendor_code == "insufficient_quota"
            assert getattr(error, "cost_usd", None) is None
            assert getattr(error, "usage", None) is None and costs == []
            assert fixture.posts == fixture.counts == transport.dispatches == 1
            assert fixture.accepted == 0
            assert not any(name == "provider:retry" for name, _ in hooks.events)
            public = [(name, data) for name, data in hooks.events if name == "llm:response"]
            assert len(public) == 1 and public[0][1]["status"] == "error"
            assert "usage" not in public[0][1]
            assert "private-quota-sentinel" not in str(error) + json.dumps(public)
        finally:
            await provider.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["unknown_429", "html_429", "redirect_quota", "post_event_quota"])
async def test_real_sdk_quota_counterexamples_remain_unknown(mode):
    async with Loopback(mode, streaming=mode == "post_event_quota") as fixture:
        provider, transport, hooks = provider_for(fixture)
        try:
            with pytest.raises(RequestOutcomeUnknownError) as caught:
                await provider.complete(request())
            assert caught.value.retryable is False
            assert caught.value.request_outcome == "unknown"
            assert caught.value.usage["attempts"][0]["input_tokens"] is None
            assert fixture.posts == transport.dispatches == 1
            assert fixture.accepted == (1 if mode in {"redirect_quota", "post_event_quota"} else 0)
            assert not any(name == "provider:retry" for name, _ in hooks.events)
            assert "private-quota-sentinel" not in str(caught.value)
        finally:
            await provider.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["500", "silent"])
@pytest.mark.parametrize("streaming", [False, True])
async def test_real_sdk_injected_retries_do_not_replay_accepted_failure(mode, streaming):
    async with Loopback(mode, streaming=streaming) as fixture:
        transport = LoopbackTransport(fixture)
        sdk = openai.AsyncOpenAI(
            api_key="fixture",  # Deliberately retain the actual SDK default retries.
            timeout=openai.Timeout(None, connect=5, pool=5, read=0.02, write=None),
            default_headers={"x-fixture-option": "preserved"},
            organization="fixture-org", project="fixture-project",
            http_client=httpx.AsyncClient(transport=transport, trust_env=False),
        )
        provider = None
        try:
            assert sdk.max_retries == 2
            provider = OpenAIProvider(client=sdk, config={
                "default_model": "gpt-6-astra", "use_streaming": streaming,
                "max_retries": 2, "max_concurrent_requests": 0,
                "extra_request_params": {"timeout": sdk.timeout},
            })
            hooks = Hooks()
            provider.coordinator = SimpleNamespace(hooks=hooks)
            assert provider.client is not sdk and provider.client.max_retries == 0
            assert sdk.max_retries == 2 and not sdk.is_closed()
            assert provider.client.timeout == sdk.timeout
            assert provider.client.base_url == sdk.base_url
            assert provider.client.api_key == sdk.api_key
            assert provider.client.organization == sdk.organization
            assert provider.client.project == sdk.project
            assert provider.client.default_headers["x-fixture-option"] == "preserved"
            with pytest.raises(llm_errors.LLMError) as caught:
                await provider.complete(request())
            assert caught.value.retryable is False
            assert caught.value.request_outcome == "unknown"
            assert fixture.posts == fixture.accepted == transport.dispatches == 1
            assert not any(name == "provider:retry" for name, _ in hooks.events)
            assert transport.generation_limits[0] == {
                "connect": 5, "pool": 5, "read": 0.02, "write": None,
            }
        finally:
            if provider is not None:
                await provider.close()
                assert sdk.is_closed()  # Existing injected-transport close ownership.
            await sdk.close()


@pytest.mark.asyncio
async def test_injected_sdk_subclass_with_retries_fails_locally_before_post():
    class CustomSDK(openai.AsyncOpenAI):
        pass

    async with Loopback("500") as fixture:
        transport = LoopbackTransport(fixture)
        async with CustomSDK(
            api_key="fixture", max_retries=2,
            http_client=httpx.AsyncClient(transport=transport, trust_env=False),
        ) as sdk:
            with pytest.raises(InjectedClientConfigurationError) as caught:
                OpenAIProvider(client=sdk)
            assert caught.value.retryable is False
            assert caught.value.request_outcome == "not_sent" and caught.value.effects == "none"
            assert fixture.posts == fixture.accepted == fixture.counts == transport.dispatches == 0
            assert sdk.max_retries == 2 and not sdk.is_closed()


@pytest.mark.asyncio
@pytest.mark.parametrize("displayed", [False, True])
@pytest.mark.parametrize("behavior", ["ok", "raise", "internal_cancel", "block"])
async def test_real_sdk_cancel_preserves_primary_through_optional_cleanup(displayed, behavior):
    mode = "partial_cancel" if displayed else "observation_cancel"
    async with Loopback(mode, streaming=True) as fixture:
        provider, transport, hooks = provider_for(fixture)
        reached = asyncio.Event()
        hook_drained = asyncio.Event()
        original_emit = hooks.emit
        activities = 0

        async def emit(name, data):
            nonlocal activities
            await original_emit(name, data)
            if name == "llm:progress" and data["observation"] == "response_activity":
                activities += 1
            if name == "llm:stream_block_delta":
                reached.set()
            if not displayed and activities == 1:
                reached.set()
            cleanup = (name == "llm:stream_aborted" if displayed else (
                name == "llm:progress" and activities == 2))
            if cleanup:
                try:
                    if behavior == "raise":
                        raise ValueError("private-cleanup")
                    if behavior == "internal_cancel":
                        raise asyncio.CancelledError("hook-not-caller")
                    if behavior == "block":
                        await asyncio.Event().wait()
                finally:
                    hook_drained.set()

        hooks.emit = emit
        task = asyncio.create_task(provider.complete(request()))
        try:
            await asyncio.wait_for(reached.wait(), 2)
            start = time.monotonic()
            task.cancel("caller-stop")
            with pytest.raises(asyncio.CancelledError, match="caller-stop") as caught:
                await asyncio.wait_for(task, 0.5)
            assert time.monotonic() - start < 0.25
            assert task.cancelling() == 1
            assert caught.value.usage["input_tokens"] is None
            assert fixture.posts == fixture.accepted == fixture.counts == transport.dispatches == 1
            assert hook_drained.is_set()
            aborts = [data for name, data in hooks.events if name == "llm:stream_aborted"]
            deltas = [data for name, data in hooks.events if name == "llm:stream_block_delta"]
            assert len(aborts) == (1 if displayed else 0)
            if displayed:
                assert aborts[0]["request_id"] == deltas[0]["request_id"]
                assert aborts[0]["error"]["type"] == "CancelledError"
                assert "private-" not in json.dumps(aborts)
            assert not any(name == "provider:retry" for name, _ in hooks.events)
            assert not any(name == "llm:response" and data["status"] == "ok"
                           for name, data in hooks.events)
            assert_no_provider_delivery_tasks()
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            await provider.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("measurement", ["known", "absent", "zero", "partial"])
async def test_real_sdk_cancelled_continuation_usage_is_known_subtotal_only(measurement):
    initial = response("incomplete")
    initial["service_tier"] = "default"
    if measurement == "absent":
        initial["usage"] = None
    elif measurement == "zero":
        initial["usage"].update(input_tokens=0, output_tokens=0, total_tokens=0)
    elif measurement == "partial":
        initial["usage"]["input_tokens"] = None
    async with Loopback("continuation_pending", initial_response=initial) as fixture:
        provider, transport, hooks = provider_for(fixture)
        costs = []
        provider._add_cost = costs.append
        task = asyncio.create_task(provider.complete(request()))
        try:
            await asyncio.wait_for(fixture.pending_continuation.wait(), 2)
            task.cancel("caller-stop")
            with pytest.raises(asyncio.CancelledError, match="caller-stop") as caught:
                await task
            usage = caught.value.usage
            assert fixture.posts == fixture.accepted == fixture.counts == transport.dispatches == 2
            assert len(usage["attempts"]) == 2
            assert usage["attempts"][1]["input_tokens"] is None
            assert usage["input_tokens"] is usage["cost_usd"] is None
            assert usage["cost_complete"] is False and usage["cost_scope"] == "known_attempts"
            if measurement in {"known", "zero"}:
                assert usage["attempts"][0]["input_tokens"] == (10 if measurement == "known" else 0)
                assert len(costs) == 1
                assert usage["cost_known_subtotal_usd"] == str(costs[0])
            else:
                assert usage["cost_known_subtotal_usd"] is None and costs == []
            if measurement == "partial":
                assert usage["attempts"][0]["output_tokens"] == 5
            terminal = [data for name, data in hooks.events if name == "llm:response"]
            assert len(terminal) == 1 and terminal[0]["status"] == "cancelled"
            assert terminal[0]["usage"] == usage
            assert not any(name == "provider:retry" for name, _ in hooks.events)
            assert_no_provider_delivery_tasks()
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            await provider.close()


@pytest.mark.asyncio
async def test_real_sdk_second_caller_cancel_during_abort_is_not_swallowed():
    async with Loopback("partial_cancel", streaming=True) as fixture:
        provider, transport, hooks = provider_for(fixture)
        displayed, abort_entered, drained = asyncio.Event(), asyncio.Event(), asyncio.Event()
        original_emit = hooks.emit

        async def emit(name, data):
            await original_emit(name, data)
            if name == "llm:stream_block_delta":
                displayed.set()
            if name == "llm:stream_aborted":
                abort_entered.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    drained.set()

        hooks.emit = emit
        task = asyncio.create_task(provider.complete(request()))
        try:
            await asyncio.wait_for(displayed.wait(), 2)
            task.cancel("first-stop")
            await abort_entered.wait()
            task.cancel("second-stop")
            with pytest.raises(asyncio.CancelledError, match="second-stop") as caught:
                await task
            assert task.cancelling() == 2 and drained.is_set()
            assert caught.value.usage["input_tokens"] is None
            assert fixture.posts == fixture.accepted == fixture.counts == transport.dispatches == 1
            assert len([name for name, _ in hooks.events if name == "llm:stream_aborted"]) == 1
            assert not any(name == "provider:retry" for name, _ in hooks.events)
            assert_no_provider_delivery_tasks()
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            await provider.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("secondary", ["calculator", "callback", "terminal_hook"])
async def test_real_sdk_secondary_failure_never_masks_dispatched_unknown(secondary):
    failed = response("failed")
    failed["service_tier"] = "default"
    failed["error"] = {"code": "server_error", "message": "private-primary"}
    async with Loopback("failed_usage", streaming=True, failed_response=failed) as fixture:
        provider, transport, hooks = provider_for(fixture)
        costs = []

        def callback(cost):
            costs.append(cost)
            if secondary == "callback":
                raise ValueError("private-accounting")

        def calculator(_):
            raise ValueError("private-pricing")

        provider._add_cost = callback
        if secondary == "calculator":
            provider._compute_attempt_cost = calculator
        original_emit = hooks.emit

        async def emit(name, data):
            await original_emit(name, data)
            if secondary == "terminal_hook" and name == "llm:response":
                raise asyncio.CancelledError("hook-not-caller")

        hooks.emit = emit
        try:
            with pytest.raises(RequestOutcomeUnknownError) as caught:
                await provider.complete(request())
            assert caught.value.request_outcome == "unknown"
            assert caught.value.retryable is False
            assert fixture.posts == fixture.accepted == fixture.counts == transport.dispatches == 1
            usage = caught.value.usage
            assert usage["input_tokens"] == 10 and usage["output_tokens"] == 5
            assert len(costs) == (0 if secondary == "calculator" else 1)
            if secondary == "calculator":
                assert usage["cost_usd"] is None
            elif secondary == "callback":
                assert usage["cost_commit"] == "unknown"
            terminal = [data for name, data in hooks.events if name == "llm:response"]
            assert len(terminal) == 1 and terminal[0]["status"] == "error"
            assert terminal[0]["usage"] == usage
            assert "private-" not in json.dumps(terminal)
            assert_no_provider_delivery_tasks()
        finally:
            await provider.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("measurement", ["priority", "tier_missing", "absent", "zero"])
async def test_real_sdk_failed_usage_settles_once_without_replay(measurement):
    failed = response("failed")
    failed["service_tier"] = None if measurement == "tier_missing" else "default"
    failed["error"] = {"code": "server_error", "message": "private-failure-detail"}
    if measurement == "priority":
        failed["service_tier"] = "priority"
        failed["usage"].update(input_tokens=272_001, output_tokens=1, total_tokens=272_002)
    elif measurement == "tier_missing":
        failed["usage"].update(input_tokens=100, output_tokens=10, total_tokens=110)
    elif measurement == "absent":
        failed["usage"] = None
    else:
        failed["usage"].update(input_tokens=0, output_tokens=0, total_tokens=0)
    async with Loopback("failed_usage", streaming=True, failed_response=failed) as fixture:
        provider, transport, hooks = provider_for(fixture, max_retries=1)
        costs = []
        provider._add_cost = costs.append
        try:
            with pytest.raises(RequestOutcomeUnknownError) as caught:
                await asyncio.wait_for(provider.complete(request()), 3)
            error = caught.value
            assert str(error) == OUTCOME_UNKNOWN_MESSAGE and error.retryable is False
            assert error.effects == "may_have_occurred"
            assert fixture.posts == fixture.accepted == fixture.counts == transport.dispatches == 1
            usage = error.usage
            events = [data for name, data in hooks.events if name == "llm:response"]
            assert len(events) == 1 and events[0]["status"] == "error"
            assert events[0]["usage"] == usage
            assert len(usage["attempts"]) == 1
            if measurement == "priority":
                assert usage["input_tokens"] == 272_001 and usage["output_tokens"] == 1
                assert usage["cost_usd"] == "10.880190"
                assert [str(cost) for cost in costs] == ["10.880190"]
            elif measurement == "tier_missing":
                assert usage["input_tokens"] == 100 and usage["output_tokens"] == 10
                assert usage["cost_usd"] is None and costs == []
            elif measurement == "absent":
                assert usage["input_tokens"] is usage["output_tokens"] is None
                assert usage["cost_usd"] is None and costs == []
            else:
                assert usage["input_tokens"] == usage["output_tokens"] == 0
                assert len(costs) == 1 and costs[0] == 0
            assert "private-" not in json.dumps(events)
            assert "resp_loopback" not in json.dumps(events)
            assert not any(name == "provider:retry" for name, _ in hooks.events)
        finally:
            await provider.close()


@pytest.mark.asyncio
async def test_real_sdk_partial_eof_aborts_once_without_canonical_result():
    async with Loopback("partial_eof", streaming=True) as fixture:
        provider, transport, hooks = provider_for(fixture, max_retries=1)
        costs = []
        provider._add_cost = costs.append
        try:
            with pytest.raises(RequestOutcomeUnknownError) as caught:
                await asyncio.wait_for(provider.complete(request()), 3)
            assert str(caught.value) == OUTCOME_UNKNOWN_MESSAGE
            assert caught.value.retryable is False
            assert fixture.posts == fixture.accepted == fixture.counts == transport.dispatches == 1
            stream_events = [(name, data) for name, data in hooks.events if name.startswith("llm:stream_")]
            assert [name for name, _ in stream_events] == [
                "llm:stream_block_start", "llm:stream_block_delta", "llm:stream_aborted",
            ]
            assert len({data["request_id"] for _, data in stream_events}) == 1
            assert stream_events[1][1]["sequence"] == 0
            assert stream_events[2][1]["error"] == {
                "type": "RequestOutcomeUnknown", "msg": OUTCOME_UNKNOWN_MESSAGE,
            }
            results = [data for name, data in hooks.events if name == "llm:response"]
            assert len(results) == 1 and results[0]["status"] == "error"
            assert results[0]["usage"] == caught.value.usage
            assert caught.value.usage["input_tokens"] is caught.value.usage["output_tokens"] is None
            assert costs == [] and "private-partial-text" not in json.dumps(results)
            assert hooks.events.index(stream_events[2]) < next(
                i for i, (name, _) in enumerate(hooks.events) if name == "llm:response"
            )
        finally:
            await provider.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_real_sdk_default_silent_wait_then_success(streaming):
    async with Loopback(streaming=streaming) as fixture:
        provider, transport, hooks = provider_for(fixture)
        try:
            task = asyncio.create_task(provider.complete(request()))
            await asyncio.wait_for(fixture.accepted_event.wait(), 2)
            await asyncio.sleep(0.05)
            assert not task.done()
            assert fixture.posts == fixture.accepted == fixture.counts == 1
            assert [e["observation"] for e in progress(hooks)] == ["attempt_started"]
            fixture.release.set()
            result = await asyncio.wait_for(task, 2)
            assert result.usage.input_tokens == 10 and result.usage.output_tokens == 5
            assert result.content[0].text == "fixture answer"
            assert transport.limits[-1] == {"connect": 5.0, "pool": 5.0, "read": None, "write": None}
        finally:
            await provider.close()


def progress(hooks):
    return [data for name, data in hooks.events if name == "llm:progress"]


def assert_payload(data):
    assert set(data) == {"version", "observation", "attempt", "limits"}
    assert data["version"] == 1
    assert data["observation"] in {"attempt_started", "response_activity"}
    assert type(data["attempt"]) is int and data["attempt"] > 0
    assert set(data["limits"]) == {
        "mode", "elapsed_seconds", "connect_seconds", "pool_seconds",
        "read_seconds", "write_seconds",
    }
    assert data["limits"]["mode"] in {"none", "elapsed", "phase"}
    for key, value in data["limits"].items():
        if key != "mode" and value is not None:
            assert type(value) is float and 0 <= value < float("inf")


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("policy", ["default", "scalar", "explicit_none", "extra_none", "phase", "request_scalar"])
async def test_real_sdk_exact_effective_limit_metadata(streaming, policy):
    phase = openai.Timeout(None, connect=1, pool=2, read=3, write=4)
    config, req_kwargs = {}, {}
    effective = None
    if policy == "scalar":
        config, effective = {"timeout": 7}, 7
    elif policy == "explicit_none":
        config = {"timeout": 9, "extra_request_params": {"timeout": 8}}
        req_kwargs = {"timeout": None}
    elif policy == "extra_none":
        config = {"timeout": 9, "extra_request_params": {"timeout": None}}
    elif policy == "phase":
        config, effective = {"extra_request_params": {"timeout": phase}}, phase
    elif policy == "request_scalar":
        config = {"extra_request_params": {"timeout": phase}}
        req_kwargs, effective = {"timeout": 6}, 6
    async with Loopback("success", streaming=streaming) as fixture:
        provider, transport, hooks = provider_for(fixture, **config)
        try:
            result = await provider.complete(request(**req_kwargs))
            assert result.usage.output_tokens == 5
            events = progress(hooks)
            assert events[0]["observation"] == "attempt_started"
            assert events[-1]["observation"] == "response_activity"
            for data in events:
                assert_payload(data)
                limits = data["limits"]
                assert limits["mode"] == ("phase" if policy == "phase" else (
                    "elapsed" if effective is not None else "none"))
                assert limits["elapsed_seconds"] == (effective if isinstance(effective, int) else None)
                assert {key.removesuffix("_seconds"): value for key, value in limits.items()
                        if key not in {"mode", "elapsed_seconds"}} == transport.generation_limits[0]
            assert fixture.posts == fixture.counts == 1
            assert hooks.events[-2][0] == "llm:progress"
            assert hooks.events[-1][0] == "llm:response"
            assert "private-prompt-sentinel" not in json.dumps(events)
        finally:
            await provider.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("settlement", ["success", "error", "cancel"])
async def test_real_sdk_pending_actual_flush_before_success_error_cancel(monkeypatch, settlement):
    now = [0.0]
    monkeypatch.setattr(observation_module, "time", SimpleNamespace(monotonic=lambda: now[0]))
    original = _WaitObserver.response_activity
    parsed, seen = asyncio.Event(), [0]

    async def observe(self):
        seen[0] += 1
        if seen[0] == 2:
            now[0] = 0.10
        await original(self)
        if seen[0] == 2:
            parsed.set()

    monkeypatch.setattr(_WaitObserver, "response_activity", observe)
    async with Loopback("observation_" + settlement, streaming=True) as fixture:
        provider, _, hooks = provider_for(fixture)
        try:
            task = asyncio.create_task(provider.complete(request()))
            await asyncio.wait_for(parsed.wait(), 2)
            assert len(progress(hooks)) == 2  # attempt + A, B still pending
            now[0] = 0.11
            if settlement == "cancel":
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
                # A host wrapper settles only after invocation propagates cancel.
                await hooks.emit("wrapper:cancelled", {})
            else:
                fixture.release.set()
                if settlement == "error":
                    with pytest.raises(llm_errors.LLMError):
                        await task
                else:
                    await task
            assert progress(hooks)[-1]["observation"] == "response_activity"
            assert len(progress(hooks)) == 3
            if settlement == "cancel":
                assert [name for name, _ in hooks.events[-3:]] == [
                    "llm:progress", "llm:response", "wrapper:cancelled",
                ]
                assert hooks.events[-2][1]["status"] == "cancelled"
            else:
                assert hooks.events[-2][0] == "llm:progress"
            assert hooks.events[-1][0] == ("wrapper:cancelled" if settlement == "cancel" else "llm:response")
            assert fixture.posts == fixture.accepted == fixture.counts == 1
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            await provider.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["refusal", "continuation", "tool_truncation"])
async def test_real_sdk_logical_throttle_across_physical_generations(monkeypatch, mode):
    monkeypatch.setattr(observation_module, "time", SimpleNamespace(monotonic=lambda: 0.0))
    async with Loopback(mode) as fixture:
        provider, _, hooks = provider_for(fixture, max_output_tokens=1024)
        try:
            result = await provider.complete(request())
            events = progress(hooks)
            assert [(e["observation"], e["attempt"]) for e in events] == (
                [("attempt_started", 1), ("attempt_started", 2), ("response_activity", 2)]
                if mode == "refusal" else [
                    ("attempt_started", 1), ("response_activity", 1),
                    ("attempt_started", 2), ("response_activity", 2),
                ])
            assert fixture.posts == fixture.counts == 2
            assert fixture.accepted == (1 if mode == "refusal" else 2)
            assert result.usage.output_tokens == (5 if mode == "refusal" else 10)
            assert "private-tool-sentinel" not in json.dumps(events)
            assert hooks.events[-2][0] == "llm:progress"
        finally:
            await provider.close()


@pytest.mark.asyncio
async def test_real_sdk_background_same_id_is_activity_not_new_attempt(monkeypatch):
    monkeypatch.setattr(observation_module, "time", SimpleNamespace(monotonic=lambda: 0.0))
    async with Loopback("background") as fixture:
        provider, _, hooks = provider_for(fixture)
        try:
            await provider.complete(request(), background=True, poll_interval=0)
            assert fixture.posts == fixture.counts == fixture.retrieves == 1
            assert [(e["observation"], e["attempt"]) for e in progress(hooks)] == [
                ("attempt_started", 1), ("response_activity", 1), ("response_activity", 1),
            ]
            assert "resp_loopback" not in json.dumps(progress(hooks))
        finally:
            await provider.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_real_sdk_no_hook_and_failing_progress_hook_preserve_success(streaming):
    async with Loopback("success", streaming=streaming) as fixture:
        provider, _, hooks = provider_for(fixture)
        try:
            provider.coordinator = None
            first = await provider.complete(request())
            assert first.usage.output_tokens == 5
            assert not hooks.events
            ordinary_emit = hooks.emit

            async def emit(name, data):
                if name == "llm:progress":
                    raise RuntimeError("private-hook-error")
                await ordinary_emit(name, data)

            hooks.emit = emit
            provider.coordinator = SimpleNamespace(hooks=hooks)
            second = await provider.complete(request())
            assert second.usage == first.usage
            assert fixture.posts == fixture.counts == fixture.accepted == 2
            assert hooks.events[-1][0] == "llm:response"
        finally:
            await provider.close()


@pytest.mark.asyncio
async def test_real_sdk_concurrent_call_context_not_arrival_order():
    # The provider must preserve caller task context across its wait_for task.
    # The host's actual CURRENT_CALL ingestion remains a separate integration gate.
    current_call = ContextVar("fixture_current_call")
    attributed = []
    hooks = Hooks()
    ordinary_emit = hooks.emit

    async def emit(name, data):
        if name == "llm:progress":
            attributed.append((current_call.get(), data))
        await ordinary_emit(name, data)

    hooks.emit = emit
    async with Loopback() as slow, Loopback("success") as fast:
        provider_slow, _, _ = provider_for(slow)
        provider_fast, _, _ = provider_for(fast, timeout=7)
        provider_slow.coordinator = provider_fast.coordinator = SimpleNamespace(hooks=hooks)

        async def call(provider, identity):
            token = current_call.set(identity)
            try:
                return await provider.complete(request())
            finally:
                current_call.reset(token)

        slow_task = asyncio.create_task(call(provider_slow, "root"))
        try:
            await asyncio.wait_for(slow.accepted_event.wait(), 2)
            await call(provider_fast, "worker")
            slow.release.set()
            await slow_task
            assert [identity for identity, data in attributed
                    if data["observation"] == "response_activity"] == ["worker", "root"]
            for identity, data in attributed:
                assert_payload(data)
                assert data["limits"]["elapsed_seconds"] == (7 if identity == "worker" else None)
                assert data["attempt"] == 1
            assert slow.posts == fast.posts == 1
        finally:
            if not slow_task.done():
                slow_task.cancel()
                await asyncio.gather(slow_task, return_exceptions=True)
            await provider_slow.close()
            await provider_fast.close()


@pytest.mark.asyncio
async def test_real_sdk_response_secrets_tools_and_thinking_absent_from_metadata():
    async with Loopback("private_payload") as fixture:
        provider, _, hooks = provider_for(fixture)
        try:
            result = await provider.complete(request())
            # Actual sensitive fixture content survives normal response conversion;
            # absence from progress cannot be explained by absence from the wire.
            converted = result.model_dump_json()
            assert "private-thinking-sentinel" in converted
            assert "private-encrypted-sentinel" in converted
            assert "private-tool-sentinel" in converted
            assert "private-prompt-sentinel" in json.dumps(fixture.payloads)
            encoded = json.dumps(progress(hooks))
            for forbidden in ("private-", "resp_loopback", "api.openai.com", "fixture_tool",
                              "Authorization", "encrypted_content"):
                assert forbidden not in encoded
            for data in progress(hooks):
                assert_payload(data)
        finally:
            await provider.close()


@pytest.mark.asyncio
async def test_real_sdk_raw_create_optional_observer_after_guards():
    async with Loopback("success") as fixture:
        provider, transport, hooks = provider_for(fixture)
        observer = _WaitObserver(hooks)
        try:
            await provider._create_response({
                "model": "gpt-6-astra", "input": [],
                "tools": [{"type": "computer"}],
            }, timeout=None, observer=observer)
            await observer.flush()
            assert [e["observation"] for e in progress(hooks)] == [
                "attempt_started", "response_activity",
            ]
            assert fixture.posts == fixture.counts == 1
            assert transport.generation_limits == [
                {"connect": 5.0, "pool": 5.0, "read": None, "write": None},
            ]
        finally:
            await provider.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["reset_after_event", "post_event_error", "background_failure"])
async def test_real_sdk_activity_before_failure_still_never_replays(mode):
    async with Loopback(mode, streaming=mode != "background_failure") as fixture:
        provider, _, hooks = provider_for(fixture)
        task = asyncio.create_task(provider.complete(
            request(), background=mode == "background_failure", poll_interval=0))
        try:
            await asyncio.wait_for(hooks.activity.wait(), 2)
            fixture.release.set()
            with pytest.raises(llm_errors.LLMError) as caught:
                await asyncio.wait_for(task, 2)
            assert caught.value.retryable is False
            assert caught.value.request_outcome == "unknown"
            assert fixture.posts == fixture.accepted == fixture.counts == 1
            assert fixture.retrieves == (1 if mode == "background_failure" else 0)
            assert not [name for name, _ in hooks.events if name == "provider:retry"]
            assert progress(hooks)[-1]["observation"] == "response_activity"
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            await provider.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_real_sdk_count_guard_precedes_attempt_observation(streaming):
    async with Loopback("count_overflow", streaming=streaming) as fixture:
        provider, _, hooks = provider_for(fixture)
        try:
            with pytest.raises(llm_errors.ContextLengthError):
                await provider.complete(request())
            assert fixture.posts == 0 and fixture.counts == 1
            assert not progress(hooks)
        finally:
            await provider.close()


@pytest.mark.asyncio
async def test_real_sdk_sse_comments_do_not_invent_activity():
    async with Loopback("comment_only", streaming=True) as fixture:
        provider, _, hooks = provider_for(fixture)
        task = asyncio.create_task(provider.complete(request()))
        try:
            await asyncio.wait_for(fixture.accepted_event.wait(), 2)
            await asyncio.sleep(0.05)
            assert not task.done()
            assert [e["observation"] for e in progress(hooks)] == ["attempt_started"]
            fixture.release.set()
            await task
            assert progress(hooks)[-1]["observation"] == "response_activity"
            assert fixture.posts == 1
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
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