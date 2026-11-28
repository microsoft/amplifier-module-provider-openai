"""Last-round extension-boundary and shared-client race regressions."""
import asyncio
import importlib.util
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from amplifier_core import llm_errors as errors

spec = importlib.util.spec_from_file_location("boundary_round3", Path(__file__).with_name("test_network_boundary.py"))
boundary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(boundary)
matrix, httpx, network, NAME = boundary.matrix, boundary.httpx, boundary.network, boundary.NAME
local_only_sockets = matrix.local_only_sockets


@asynccontextmanager
async def supplied_client(http, streaming, url="http://127.0.0.1:9/v1"):
    if NAME == "gemini":
        from amplifier_module_provider_gemini import GeminiProvider
        from google import genai
        from google.genai import types
        p = GeminiProvider(api_key="not-a-real-key", config={"use_streaming": streaming, "max_retries": 0, "max_concurrent_requests": 0})
        p._client = genai.Client(api_key="not-a-real-key", vertexai=False, http_options=types.HttpOptions(
            base_url=url, httpx_async_client=http, retry_options=types.HttpRetryOptions(attempts=1)))
    else:
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
async def subclass_provider(streaming, kind="route"):
    delivered, attempts = [], []

    class Transport(httpx.AsyncHTTPTransport):
        async def handle_async_request(self, request):
            attempts.append(request)
            delivered.append(b"partial-in-custom-transport")
            raise httpx.ReadError("custom route consumed bytes", request=request)

    transport = Transport()

    class Client(httpx.AsyncClient):
        def _transport_for_url(self, url):
            return transport

    class SendClient(httpx.AsyncClient):
        async def send(self, request, **kwargs):
            return await transport.handle_async_request(request)

    if kind == "route":
        http = Client()
    elif kind == "send":
        http = SendClient()
    elif kind == "mount":
        http = httpx.AsyncClient(mounts={"http://127.0.0.1:9": transport})
    else:
        http = httpx.AsyncClient(transport=transport)
    try:
        async with supplied_client(http, streaming) as p:
            yield p, delivered, attempts
    finally:
        await transport.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("kind", ["route", "send", "transport", "mount"])
async def test_extension_types_keep_old_class(streaming, kind):
    async with subclass_provider(streaming, kind) as (p, delivered, attempts):
        with pytest.raises(Exception) as caught:
            await p.complete(matrix.request())
        assert type(caught.value) is (httpx.ReadError if NAME == "gemini" and streaming else errors.LLMError)
        assert len(delivered) == len(attempts) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", ["_transport", "_mounts"])
async def test_private_attribute_drift_is_unknown(missing):
    async with matrix.failing_provider("dns") as p:
        http = boundary.http_client(p)
        with network.observe_response():
            network.watch_response(p.client if NAME == "gemini" else http)
            saved = getattr(http, missing)
            delattr(http, missing)
            try:
                assert network.observation_confirmed() is False
            finally:
                setattr(http, missing, saved)


@pytest.mark.asyncio
async def test_observer_never_reinserted_or_moved():
    async with matrix.failing_provider("dns") as p:
        http = boundary.http_client(p)
        watch = p.client if NAME == "gemini" else http
        with network.observe_response():
            network.watch_response(watch)
            async def first(response):
                pass
            hooks = http.event_hooks["response"]
            hooks.insert(0, first)
            before = list(hooks)
            network.watch_response(watch)
            assert hooks == before
            assert not network.observation_confirmed()
            hooks.remove(network._received_headers)
            network.watch_response(watch)
            assert network._received_headers not in hooks
            assert not network.observation_confirmed()


@pytest.mark.asyncio
@pytest.mark.parametrize("extension", ["auth", "request_hook"])
async def test_public_pre_response_extensions_are_unknown(extension):
    async with matrix.failing_provider("dns") as p:
        http = boundary.http_client(p)
        with network.observe_response():
            network.watch_response(p.client if NAME == "gemini" else http)
            assert network.observation_confirmed()
            if extension == "auth":
                http.auth = httpx.BasicAuth("dummy", "dummy")
            else:
                async def caller(request):
                    pass
                http.event_hooks["request"].append(caller)
            assert not network.observation_confirmed()


@asynccontextmanager
async def race_provider(streaming):
    """A waits in a displaced hook; B starts while A's hook list is active."""
    entered, second_started = asyncio.Event(), asyncio.Event()
    counts = {"sends": 0, "hook_a": 0}
    tasks = set()

    async def handle(reader, writer):
        task = asyncio.current_task()
        tasks.add(task)
        try:
            headers = await reader.readuntil(b"\r\n\r\n")
            length = next((int(line.split(b":", 1)[1]) for line in headers.split(b"\r\n") if line.lower().startswith(b"content-length:")), 0)
            await reader.readexactly(length)
            counts["sends"] += 1
            identity = b"A" if counts["sends"] == 1 else b"B"
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nX-Case: " + identity + b"\r\nContent-Length: 10000\r\n\r\npartial")
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()
            tasks.discard(task)

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    url = f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}"
    http = httpx.AsyncClient()
    try:
        async with supplied_client(http, streaming, url) as p:
            with network.observe_response():
                network.watch_response(p.client if NAME == "gemini" else http)
                assert network.observation_confirmed()

            async def displaced(response):
                if response.headers.get("x-case") == "A":
                    counts["hook_a"] += 1
                    entered.set()
                    await asyncio.wait_for(second_started.wait(), 5)

            async def request_hook(request):
                if entered.is_set():
                    second_started.set()

            http.event_hooks["response"].insert(0, displaced)
            http.event_hooks["request"].append(request_hook)
            yield p, entered, counts
    finally:
        await http.aclose()
        server.close()
        pending = list(tasks)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        await server.wait_closed()


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_shared_client_race_never_moves_observer(streaming):
    async with race_provider(streaming) as (p, entered, counts):
        a = asyncio.create_task(p.complete(matrix.request()))
        await asyncio.wait_for(entered.wait(), 5)
        b = asyncio.create_task(p.complete(matrix.request()))
        results = await asyncio.wait_for(asyncio.gather(a, b, return_exceptions=True), 10)
        expected = httpx.RemoteProtocolError if NAME == "gemini" and streaming else errors.LLMError
        assert all(type(error) is expected for error in results)
        assert counts == {"sends": 2, "hook_a": 1}


@asynccontextmanager
async def default_provider(streaming, url):
    if NAME == "gemini":
        from amplifier_module_provider_gemini import GeminiProvider
        from google import genai
        from google.genai import types
        p = GeminiProvider(api_key="not-a-real-key", config={"use_streaming": streaming, "max_retries": 0})
        # Force HTTPX without supplying a client: the SDK constructs its own
        # default class. Normal Unified selects aiohttp and is excluded.
        p._client = genai.Client(api_key="not-a-real-key", vertexai=False, http_options=types.HttpOptions(
            base_url=url, async_client_args={"transport": httpx.AsyncHTTPTransport()}))
    else:
        p = matrix.provider(url, streaming)
    try:
        yield p
    finally:
        await p.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("failure", ["refused", "closed", "connect_timeout"])
async def test_sdk_default_is_confirmed_and_promoted(streaming, failure):
    async with matrix.endpoint("closed" if failure == "closed" else "refused") as url:  # noqa: SIM117
        async with default_provider(streaming, url) as p:
            http = boundary.http_client(p)
            with network.observe_response():
                network.watch_response(p.client if NAME == "gemini" else http)
                assert type(http) in network._CLIENT_TYPES
                assert network.observation_confirmed()
            if failure == "connect_timeout":
                async def connect_tcp(*args, **kwargs):
                    raise matrix.httpcore.ConnectTimeout("fake local connect timeout")
                from types import SimpleNamespace
                http._transport._pool._network_backend = SimpleNamespace(connect_tcp=connect_tcp)
            with pytest.raises(errors.LLMError) as caught:
                await p.complete(matrix.request())
            assert type(caught.value) is (errors.LLMTimeoutError if failure == "connect_timeout" else errors.NetworkError)
