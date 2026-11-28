"""Measure transport failure-object identity independently of recovery policy."""
import importlib
import importlib.util
import json
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location("failure_boundary", Path(__file__).with_name("test_network_boundary.py"))
boundary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(boundary)
matrix, NAME = boundary.matrix, boundary.NAME
local_only_sockets = matrix.local_only_sockets


def chain(error):
    pending, seen, result = [error], set(), []
    while pending and len(seen) < 64:
        error = pending.pop()
        if error is None or id(error) in seen:
            continue
        seen.add(id(error)); result.append(error)
        pending.extend((error.__cause__, error.__context__))
    return result


KINDS = ["refused", "dns", "connect_timeout", "tls_cut", "closed", "header_cut", "head_timeout"]


@asynccontextmanager
async def failure_provider(kind, streaming, library="httpcore2"):
    http = importlib.import_module("httpx" if library == "httpcore" else "httpx2")
    core = importlib.import_module(library)
    async with AsyncExitStack() as stack:
        if kind in {"closed", "tls_cut", "header_cut"}:
            response = b"HTTP/1.1 200 O" if kind == "header_cut" else b"x" if kind == "tls_cut" else b""
            url, _counts = await stack.enter_async_context(boundary.local_response(response))
            # local_response reads a full HTTP head; a TLS client speaks first
            # instead. Use the simple close-on-first-read server for TLS cut.
            if kind in {"closed", "tls_cut"}:
                url = await stack.enter_async_context(matrix.endpoint("closed"))
            if kind == "tls_cut":
                url = url.replace("http:", "https:")
        else:
            url = await stack.enter_async_context(matrix.endpoint("no_answer" if kind == "head_timeout" else "refused"))
        transport = http.AsyncHTTPTransport(verify=False)
        if kind in {"dns", "connect_timeout"}:
            async def connect_tcp(*args, **kwargs):
                if kind == "dns":
                    import socket
                    raise core.ConnectError("local fake DNS") from socket.gaierror(-2,"local fake")
                raise core.ConnectTimeout("local fake connect timeout")
            transport._pool._network_backend = SimpleNamespace(connect_tcp=connect_tcp)
        client = http.AsyncClient(transport=transport)
        p = await stack.enter_async_context(boundary.supplied_client(client, streaming, url))
        if kind == "head_timeout":
            if NAME == "openai":
                p._transport_timeout = lambda value: http.Timeout(0.1)
            else:
                p._sdk_timeout = matrix.anthropic.Timeout(0.1)
        events = []
        async def caller(event, info):
            events.append((event, info.get("exception")))
        async def install(req):
            req.extensions["trace"] = caller
        client.event_hooks["request"].append(install)
        yield p, events


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("library", ["httpcore2", "httpcore"] if NAME == "openai" else ["httpcore2"])
async def test_transport_failure_object_in_sdk_chain(kind, streaming, library):
    async with failure_provider(kind, streaming, library) as (p, events):
        with pytest.raises(Exception) as caught:
            await p.complete(matrix.request())
        objects = chain(caught.value)
        reports = [(name, error) for name,error in events if name.endswith(".failed") and error is not None]
        assert reports
        assert any(any(error is obj for obj in objects) for _,error in reports)
        report = next((name for name,error in reports if any(error is obj for obj in objects)))
        print("IDENTITY " + json.dumps({"provider":NAME,"library":library,"kind":kind,"streaming":streaming,
            "identity":True,"failed_event":report,"last_event":events[-1][0],
            "module_class":type(caught.value).__name__}), flush=True)
