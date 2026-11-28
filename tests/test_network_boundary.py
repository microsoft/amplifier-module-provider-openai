"""Adversarial response-boundary tests through real SDKs, local transports only."""
import importlib
import importlib.util
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from amplifier_core import llm_errors as errors

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = next(ROOT.glob("amplifier_module_provider_*" )).name
NAME = PACKAGE.removeprefix("amplifier_module_provider_")
spec = importlib.util.spec_from_file_location("matrix_" + NAME, Path(__file__).with_name("test_network_errors.py"))
matrix = importlib.util.module_from_spec(spec)
spec.loader.exec_module(matrix)
httpx = matrix.httpx
network = importlib.import_module(PACKAGE + ("._transport" if NAME == "openai" else "._network_errors"))
local_only_sockets = matrix.local_only_sockets


@asynccontextmanager
async def sdk_provider(streaming, transport=None):
    if NAME == "gemini":
        async with matrix.provider("http://127.0.0.1:9", streaming, transport=transport) as p:
            yield p
    else:
        p = matrix.provider("http://127.0.0.1:9/v1" if NAME == "openai" else "http://127.0.0.1:9", streaming)
        sdk = matrix.openai if NAME == "openai" else matrix.anthropic
        client_class = sdk.AsyncOpenAI if NAME == "openai" else sdk.AsyncAnthropic
        p._client = client_class(api_key="not-a-real-key", base_url="http://127.0.0.1:9/v1", max_retries=0,
                                http_client=httpx.AsyncClient(transport=transport))
        try:
            yield p
        finally:
            await p.close()


def http_client(p):
    if NAME == "gemini":
        return p.client._api_client._async_httpx_client
    return p.client._client


class HiddenClient:
    def __init__(self, client):
        self.wrapped = client

    def __getattr__(self, name):
        if name in {"_api_client", "_client"}:
            raise AttributeError(name)
        return getattr(self.wrapped, name)


@asynccontextmanager
async def partial_hook_provider(streaming, error_name):
    """The caller sees partial bytes in its first existing response hook."""
    delivered, attempts, ordering = [], [], []
    original = getattr(httpx, error_name)("caller hook body failed")

    class Body(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"partial-private-answer"
            raise original

    def send(req):
        attempts.append(req)
        return httpx.Response(200, stream=Body(), request=req)

    async def first(response):
        ordering.append(("first", network.response_received()))
        async for part in response.aiter_raw():
            delivered.append(part)

    async def second(response):
        ordering.append(("second", network.response_received()))

    async with sdk_provider(streaming, httpx.MockTransport(send)) as p:
        http_client(p).event_hooks["response"] = [first, second]
        yield p, delivered, attempts, ordering


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("error_name", ["ReadError", "ReadTimeout"])
async def test_existing_hook_reads_partial_body_before_failing(streaming, error_name):
    async with partial_hook_provider(streaming, error_name) as (p, delivered, attempts, ordering):
        with pytest.raises(Exception) as caught:
            await p.complete(matrix.request())
        expected = getattr(httpx, error_name) if NAME == "gemini" and streaming else errors.LLMError
        assert type(caught.value) is expected
        assert len(attempts) == 1
        assert delivered == [b"partial-private-answer"]
        assert ordering == [("first", True)]  # Observer MUST run before caller.


@pytest.mark.asyncio
async def test_observer_deduplicated_and_caller_order_preserved():
    called = []

    async def first(response):
        called.append(("first", network.response_received()))

    async def second(response):
        called.append(("second", network.response_received()))

    async with sdk_provider(False, httpx.MockTransport(lambda req: httpx.Response(200, json={}, request=req))) as p:
        http = http_client(p)
        http.event_hooks["response"] = [first, second]
        with network.observe_response():
            for _ in range(3):
                network.watch_response(p.client if NAME == "gemini" else http)
            assert http.event_hooks["response"] == [network._received_headers, first, second]
            await http.get("http://127.0.0.1:9/local")
        assert called == [("first", True), ("second", True)]


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_hidden_sdk_private_client_keeps_old_class(streaming):
    async with matrix.failing_provider("dns", streaming) as p:
        p._client = HiddenClient(p.client)
        with pytest.raises(Exception) as caught:
            await p.complete(matrix.request())
        assert type(caught.value) is (httpx.ConnectError if NAME == "gemini" and streaming else errors.LLMError)


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_hook_inserted_after_installation_is_unknown(streaming):
    async with matrix.failing_provider("dns", streaming) as p:
        http = http_client(p)

        async def late(response):
            pass

        async def mutate(request):
            http.event_hooks["response"].insert(0, late)

        http.event_hooks["request"].append(mutate)
        with pytest.raises(Exception) as caught:
            await p.complete(matrix.request())
        assert type(caught.value) is (httpx.ConnectError if NAME == "gemini" and streaming else errors.LLMError)


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_custom_transport_cannot_claim_no_response(streaming):
    # A transport can read internally before it returns to HTTPX. There is no
    # trustworthy negative response observation for arbitrary wrappers/fakes.
    def send(req):
        raise httpx.ReadError("could have consumed bytes inside transport", request=req)

    async with sdk_provider(streaming, httpx.MockTransport(send)) as p:
        with pytest.raises(Exception) as caught:
            await p.complete(matrix.request())
        assert type(caught.value) is (httpx.ReadError if NAME == "gemini" and streaming else errors.LLMError)


@pytest.mark.asyncio
async def test_real_httpx_observation_is_confirmed_not_assumed():
    async with matrix.failing_provider("dns") as p:
        http = http_client(p)
        with network.observe_response():
            network.watch_response(p.client if NAME == "gemini" else http)
            assert http.event_hooks["response"][0] is network._received_headers
            assert network.observation_confirmed()
            http._transport._pool._http2 = True
            assert not network.observation_confirmed()
            http._transport._pool._http2 = False


@pytest.mark.asyncio
@pytest.mark.parametrize("flow", ["redirect", "auth"])
async def test_redirect_and_auth_response_hooks_precede_body_reads(flow):
    # Even intermediate responses must be observed before caller auth consumes.
    seen = []
    original = httpx.ReadError("intermediate body failed")

    class Body(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"intermediate"
            raise original

    def send(req):
        return httpx.Response(302 if flow == "redirect" else 401,
                              headers={"location": "http://127.0.0.1:9/next"}, stream=Body(), request=req)

    class Auth(httpx.Auth):
        async def async_auth_flow(self, request):
            response = yield request
            seen.append(network.response_received())
            await response.aread()

    async with sdk_provider(False, httpx.MockTransport(send)) as p:
        http = http_client(p)
        with network.observe_response():
            network.watch_response(p.client if NAME == "gemini" else http)
            with pytest.raises(httpx.ReadError):
                await http.get("http://127.0.0.1:9/start", follow_redirects=True, auth=Auth() if flow == "auth" else None)
            assert network.response_received()
        if flow == "auth":
            assert seen == [True]
