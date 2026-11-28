"""Real-SDK regression cases; response evidence comes from transport traces."""
import asyncio
import importlib
import importlib.util
from contextlib import asynccontextmanager
from http.cookiejar import CookieJar
from pathlib import Path

import pytest
from amplifier_core import llm_errors as errors

NAME = "openai" if "openai" in str(Path(__file__).resolve()) else "anthropic"
spec = importlib.util.spec_from_file_location("matrix_" + NAME, Path(__file__).with_name("test_network_errors.py"))
matrix = importlib.util.module_from_spec(spec)
spec.loader.exec_module(matrix)
httpx = matrix.httpx
network = importlib.import_module("amplifier_module_provider_" + NAME + ("._transport" if NAME == "openai" else "._network_errors"))
local_only_sockets = matrix.local_only_sockets


@asynccontextmanager
async def supplied_client(http, streaming, url):
    p = matrix.provider(url, streaming)
    sdk = matrix.openai if NAME == "openai" else matrix.anthropic
    cls = sdk.AsyncOpenAI if NAME == "openai" else sdk.AsyncAnthropic
    p._client = cls(api_key="not-a-real-key", base_url=url, max_retries=0, http_client=http)
    try:
        yield p
    finally:
        await p.close()
        await http.aclose()


@asynccontextmanager
async def local_response(response=None):
    counts, tasks = {"sends": 0}, set()
    async def handle(reader, writer):
        task = asyncio.current_task()
        tasks.add(task)
        try:
            head = await reader.readuntil(b"\r\n\r\n")
            length = next((int(line.split(b":", 1)[1]) for line in head.split(b"\r\n") if line.lower().startswith(b"content-length:")), 0)
            await reader.readexactly(length)
            counts["sends"] += 1
            data = response or b"HTTP/1.1 200 OK\r\nContent-Length: 10000\r\nContent-Type: text/event-stream\r\nSet-Cookie: local=yes\r\n\r\npartial"
            writer.write(data)
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()
            tasks.discard(task)
    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    try:
        yield f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}", counts
    finally:
        server.close()
        pending = list(tasks)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        await server.wait_closed()


class HiddenClient:
    def __init__(self, wrapped):
        self.wrapped = wrapped
    def __getattr__(self, name):
        if name == "_client":
            raise AttributeError(name)
        return getattr(self.wrapped, name)


REGRESSIONS = ["hook_ReadError", "hook_ReadTimeout", "self_remove_ReadError", "self_remove_ReadTimeout",
               "cookie", "auth", "request_hook", "hidden", "renamed", "trace_replaced",
               "route_subclass", "send_subclass", "custom_transport", "mounted_transport", "redirect"]


@asynccontextmanager
async def regression_provider(case, streaming):
    if case in {"hidden", "renamed", "trace_replaced"}:
        async with matrix.failing_provider("dns", streaming) as p:
            if case in {"hidden", "renamed"}:
                p._client = HiddenClient(p.client)
            else:
                # Put replacement after installation, so it really replaces
                # rather than gets chained by the module hook.
                network.watch_response(p.client._client)
                async def replace(request):
                    request.extensions.pop("trace", None)
                p.client._client.event_hooks["request"].append(replace)
            yield p, {"sends": 1}  # Fake backend: one attempted transport operation.
        return
    if case in {"route_subclass", "send_subclass", "custom_transport", "mounted_transport"}:
        counts = {"sends": 0}
        class Transport(httpx.AsyncHTTPTransport):
            async def handle_async_request(self, request):
                counts["sends"] += 1
                # Opaque wrapper consumed a response, without any trace report.
                raise httpx.ReadError("custom transport consumed bytes", request=request)
        transport = Transport()
        class Client(httpx.AsyncClient):
            def _transport_for_url(self, url):
                return transport
        class SendClient(httpx.AsyncClient):
            async def send(self, request, **kwargs):
                return await transport.handle_async_request(request)
        http = (Client() if case == "route_subclass" else SendClient() if case == "send_subclass" else
                httpx.AsyncClient(mounts={"http://127.0.0.1:9": transport}) if case == "mounted_transport" else
                httpx.AsyncClient(transport=transport))
        try:
            async with supplied_client(http, streaming, "http://127.0.0.1:9") as p:
                yield p, counts
        finally:
            await transport.aclose()
        return
    if case == "redirect":
        async with matrix.endpoint("refused") as target:
            body = b"redirect body"
            response = b"HTTP/1.1 307 Temporary Redirect\r\nLocation: " + target.encode() + b"/next\r\nContent-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body
            async with local_response(response) as (url, counts):  # noqa: SIM117
                async with supplied_client(httpx.AsyncClient(follow_redirects=True), streaming, url) as p:
                    yield p, counts
        return
    async with local_response() as (url, counts):
        class Jar(CookieJar):
            def extract_cookies(self, response, request):
                counts["cookies"] = counts.get("cookies", 0) + 1
                raise httpx.ReadError("cookie store saw headers")
        http = httpx.AsyncClient(cookies=Jar() if case == "cookie" else None)
        if case.startswith(("hook_", "self_remove_")):
            async def caller(response):
                async for part in response.aiter_raw():
                    counts["partial"] = counts.get("partial", 0) + 1
                    break
                if case.startswith("self_remove"):
                    http.event_hooks["response"].remove(caller)
                raise getattr(httpx, case.rsplit("_", 1)[1])("caller saw body")
            http.event_hooks["response"].append(caller)
        elif case == "auth":
            class Auth(httpx.Auth):
                async def async_auth_flow(self, request):
                    response = yield request
                    async for part in response.aiter_raw():
                        counts["partial"] = counts.get("partial", 0) + 1
                        break
                    raise httpx.ReadError("auth saw response")
            http.auth = Auth()
        elif case == "request_hook":
            async def caller(request):
                counts["sends"] += 1  # Caller invocation before network dispatch.
                raise httpx.ReadError("request hook failed")
            http.event_hooks["request"].append(caller)
        async with supplied_client(http, streaming, url) as p:
            yield p, counts


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("case", REGRESSIONS)
async def test_reviewer_regressions(case, streaming):
    async with regression_provider(case, streaming) as (p, counts):
        with pytest.raises(errors.LLMError) as caught:
            await p.complete(matrix.request())
        assert type(caught.value) is errors.LLMError
        assert counts["sends"] == 1
        if case.startswith(("hook_", "self_remove_")) or case == "auth":
            assert counts["partial"] == 1
        if case == "cookie":
            assert counts["cookies"] == 1


@asynccontextmanager
async def race_provider(streaming):
    entered, release = asyncio.Event(), asyncio.Event()
    async with local_response() as (url, counts):
        http = httpx.AsyncClient()
        async def caller(response):
            counts["hooks"] = counts.get("hooks", 0) + 1
            if counts["hooks"] == 1:
                entered.set()
                await asyncio.wait_for(release.wait(), 5)
        async def request_hook(request):
            if entered.is_set():
                release.set()
        network.watch_response(http)
        http.event_hooks["response"].insert(0, caller)
        http.event_hooks["request"].append(request_hook)
        async with supplied_client(http, streaming, url) as p:
            yield p, entered, counts


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_shared_client_race(streaming):
    async with race_provider(streaming) as (p, entered, counts):
        a = asyncio.create_task(p.complete(matrix.request()))
        await asyncio.wait_for(entered.wait(), 5)
        b = asyncio.create_task(p.complete(matrix.request()))
        results = await asyncio.wait_for(asyncio.gather(a, b, return_exceptions=True), 10)
        assert all(type(error) is errors.LLMError for error in results)
        assert counts == {"sends": 2, "hooks": 2}
