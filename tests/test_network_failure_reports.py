"""The transport's own terminal failure report, not final callable identity."""
import importlib.util
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from amplifier_core import llm_errors as errors

spec = importlib.util.spec_from_file_location("terminal_boundary", Path(__file__).with_name("test_network_boundary.py"))
boundary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(boundary)
matrix, httpx, network, NAME = boundary.matrix, boundary.httpx, boundary.network, boundary.NAME
local_only_sockets = matrix.local_only_sockets
CASES = ["replace_restore", "hide_restore", "copy_aux_restore", "aux_exception",
         "sdk_untraced_first", "sdk_503_first", "both_connect", "wrapper_all"]


@asynccontextmanager
async def failure_report_provider(case, streaming):
    async with AsyncExitStack() as stack:
        refused = await stack.enter_async_context(matrix.endpoint("refused"))
        full = b"HTTP/1.1 200 OK\r\nContent-Length: 7\r\n\r\npartial"
        response = (b'HTTP/1.1 503 Unavailable\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\n{}'
                    if case == "sdk_503_first" else full)
        url, server_counts = await stack.enter_async_context(boundary.local_response(response))
        if case in {"both_connect", "wrapper_all"}:
            url = refused
        http, auxiliary = httpx.AsyncClient(), httpx.AsyncClient()
        stack.push_async_callback(auxiliary.aclose)
        p = await stack.enter_async_context(boundary.supplied_client(http, streaming, url))
        if case.startswith("sdk_") or case == "both_connect":
            p._client = p.client.with_options(max_retries=1)
        counts = {"generation_attempts": 0, "accepted": server_counts, "partial": 0, "reports": []}
        traces, auxiliary_error = {}, []
        network.watch_response(http)
        # An auxiliary request with its own record, for the manufactured-chain
        # case. Copied trace uses a different client with no module hook.
        tracked_auxiliary = httpx.AsyncClient()
        stack.push_async_callback(tracked_auxiliary.aclose)
        network.watch_response(tracked_auxiliary)
        async def request_hook(req):
            counts["generation_attempts"] += 1
            n = counts["generation_attempts"]
            if n > 1:
                req.url = httpx.URL(refused + "/generation")
            saved = req.extensions["trace"]
            traces[id(req)] = saved
            async def wrapper(event, info):
                if event.endswith(".failed"):
                    counts["reports"].append((event, info.get("exception")))
                if case in {"hide_restore", "aux_exception"} and "receive_response" in event:
                    return
                await saved(event, info)
            async def silent(event, info):
                pass
            if case == "copy_aux_restore":
                try:
                    await auxiliary.get(refused + "/aux", extensions={"trace": saved})
                except httpx.HTTPError as exc:
                    auxiliary_error.append(exc)
                req.extensions["trace"] = silent
            elif case == "replace_restore" or (case == "sdk_untraced_first" and n == 1):
                req.extensions["trace"] = silent
            elif case in {"hide_restore", "aux_exception", "wrapper_all"}:
                req.extensions["trace"] = wrapper
        http.event_hooks["request"].append(request_hook)
        async def response_hook(response):
            if response.status_code == 503:
                return
            data = await response.aread()
            assert data == b"partial"
            counts["partial"] += 1
            if case in {"replace_restore", "hide_restore", "copy_aux_restore", "aux_exception"}:
                response.request.extensions["trace"] = traces[id(response.request)]
            if case == "aux_exception":
                try:
                    await tracked_auxiliary.get(refused + "/aux")
                except httpx.HTTPError as exc:
                    raise httpx.ReadError("caller response error", request=response.request) from exc
            raise httpx.ReadError("caller response error", request=response.request)
        http.event_hooks["response"].append(response_hook)
        yield p, counts


def rules(error):
    state = network._response_state.get()
    records = list(state["requests"].values())
    chain = network.exception_chain(error)
    return {
        "rule1": any(r["failure"] is e for r in records for e in chain),
        "rule2": bool(records) and all(r["last"] in network.PRE_RESPONSE_FAILURES and isinstance(r["failure"], BaseException) for r in records),
        "no_response": not state["received"],
        "last": [r["last"] for r in records],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("case", CASES)
async def test_report_variants(case, streaming, monkeypatch):
    decisions = []
    original = network.can_promote
    def inspect(error):
        decisions.append(rules(error))
        return original(error)
    monkeypatch.setattr(network, "can_promote", inspect)
    monkeypatch.setattr("asyncio.sleep", AsyncMock())
    async with failure_report_provider(case, streaming) as (p, counts):
        with pytest.raises(errors.LLMError) as caught:
            await p.complete(matrix.request())
        assert type(caught.value) is (errors.NetworkError if case in {"both_connect", "wrapper_all"} else errors.LLMError)
        assert counts["partial"] == (0 if case in {"sdk_503_first", "both_connect", "wrapper_all"} else 1)
        assert counts["generation_attempts"] == (2 if case.startswith("sdk_") or case == "both_connect" else 1)
        decision = decisions[-1]
        if case in {"hide_restore", "replace_restore", "sdk_untraced_first", "aux_exception"}:
            assert not decision["rule2"]
        if case == "copy_aux_restore":
            assert not decision["rule1"]
        if case == "sdk_503_first":
            assert not decision["no_response"]


@pytest.mark.asyncio
async def test_unrelated_report_cannot_authorize_error_and_last_event_is_required():
    with network.observe_response():
        request = httpx.Request("GET", "http://127.0.0.1:9")
        await network.trace_request(request)
        report = RuntimeError("transport failure")
        trace = request.extensions["trace"]
        await trace("connection.connect_tcp.failed", {"exception": report})
        error = httpx.ConnectError("mapped")
        assert not network.can_promote(error)
        error.__cause__ = report
        assert network.can_promote(error)
        await trace("http11.response_closed.complete", {})
        assert not network.can_promote(error)


@pytest.mark.asyncio
async def test_body_event_is_never_a_pre_response_failure():
    with network.observe_response():
        request = httpx.Request("GET", "http://127.0.0.1:9")
        await network.trace_request(request)
        report = RuntimeError("body failure")
        await request.extensions["trace"]("http11.receive_response_body.failed", {"exception": report})
        assert "http11.receive_response_body.failed" not in network.PRE_RESPONSE_FAILURES
        error = httpx.ReadError("body")
        error.__cause__ = report
        assert not network.can_promote(error)


@pytest.mark.asyncio
async def test_chain_is_bounded_and_cycle_safe():
    e = RuntimeError("cycle"); e.__cause__ = e
    assert network.exception_chain(e) == [e]
    for _ in range(65):
        outer = RuntimeError("outer"); outer.__cause__ = e; e = outer
    with pytest.raises(ValueError):
        network.exception_chain(e)
    assert not network.can_promote(e)

@asynccontextmanager
async def module_retry_provider(streaming, shown=False):
    async with AsyncExitStack() as stack:
        refused = await stack.enter_async_context(matrix.endpoint("refused"))
        if not streaming:
            body = b'{"id":"partial"'
        elif NAME == "openai":
            body = (b'event: response.created\ndata: {"type":"response.created","response":{"id":"r","object":"response","status":"in_progress","model":"gpt-6-astra","output":[]}}\n\n'
                    b'event: response.output_item.added\ndata: {"type":"response.output_item.added","output_index":0,"item":{"id":"item","type":"message","role":"assistant","status":"in_progress","content":[]}}\n\n'
                    b'event: response.content_part.added\ndata: {"type":"response.content_part.added","item_id":"item","output_index":0,"content_index":0,"part":{"type":"output_text","text":"","annotations":[]}}\n\n')
            if shown:
                body += b'event: response.output_text.delta\ndata: {"type":"response.output_text.delta","item_id":"item","output_index":0,"content_index":0,"delta":"shown"}\n\n'
        else:
            body = (b'event: message_start\ndata: {"type":"message_start","message":{"id":"m","type":"message","role":"assistant","content":[],"model":"claude-sonnet-5-5","usage":{"input_tokens":1,"output_tokens":0}}}\n\n'
                    b'event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n')
            if shown:
                body += b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"shown"}}\n\n'
        response = b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: 100000\r\n\r\n" + body
        url, counts = await stack.enter_async_context(boundary.local_response(response))
        p = matrix.provider(url, streaming, max_retries=1, min_retry_delay=0.001, retry_jitter=False)
        stack.push_async_callback(p.close)
        counts["generation_attempts"] = 0
        async def hook(req):
            counts["generation_attempts"] += 1
            if counts["generation_attempts"] > 1:
                req.url = httpx.URL(refused + "/generation")
        p.client._client.event_hooks["request"].append(hook)
        yield p, counts


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_module_try_scope_resets_after_broken_answer(streaming, monkeypatch):
    monkeypatch.setattr("asyncio.sleep", AsyncMock())
    async with module_retry_provider(streaming) as (p, counts):
        with pytest.raises(errors.NetworkError):
            await p.complete(matrix.request())
        assert counts == {"sends": 1, "generation_attempts": 2}

@pytest.mark.asyncio
async def test_nested_tasks_keep_failure_reports_attempt_local():
    import asyncio
    ready, release = asyncio.Event(), asyncio.Event()
    async def one():
        with network.observe_response():
            request = httpx.Request("GET", "http://127.0.0.1:9")
            await network.trace_request(request)
            failure = RuntimeError("task one report")
            await asyncio.wait_for(request.extensions["trace"]("connection.connect_tcp.failed", {"exception": failure}), 2)
            error = httpx.ConnectError("mapped"); error.__cause__ = failure
            assert network.can_promote(error)
            ready.set()
            await release.wait()
            assert network.can_promote(error)
    async def two():
        await ready.wait()
        with network.observe_response():
            assert not network.can_promote(RuntimeError("no report in second scope"))
            request = httpx.Request("GET", "http://127.0.0.1:9")
            await network.trace_request(request)
            failure = RuntimeError("task two report")
            await request.extensions["trace"]("connection.connect_tcp.failed", {"exception": failure})
            await request.extensions["trace"]("http11.receive_response_headers.complete", {})
            error = httpx.ConnectError("mapped"); error.__cause__ = failure
            assert not network.can_promote(error)
            release.set()
    await asyncio.gather(one(), two())
