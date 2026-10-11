"""Exercise provider-owned HTTP sockets, not only mocked SDK calls."""
import asyncio
import json

import pytest

from amplifier_module_provider_openai import OpenAIProvider
from amplifier_module_provider_openai import _transport


@pytest.mark.asyncio
@pytest.mark.parametrize("via_proxy", [False, True])
async def test_owned_sdk_connection_enables_keepalive_and_allows_quiet_response(monkeypatch, via_proxy):
    inspected = []
    original = _transport.configure_request

    async def configure(request):
        await original(request)
        configured = request.extensions['trace']
        async def trace(event, info):
            await configured(event, info)
            if event.endswith('connect_tcp.complete'):
                sock = info['return_value'].get_extra_info('socket')
                inspected.append([sock.getsockopt(level, option)
                                  for level, option, _ in _transport.keepalive_options()])
        request.extensions['trace'] = trace
    monkeypatch.setattr(_transport, 'configure_request', configure)
    finished = asyncio.Event()
    async def serve(reader, writer):
        try:
            headers = await reader.readuntil(b'\r\n\r\n')
            if via_proxy:
                assert headers.startswith(b'GET http://fixture.invalid/v1/models ')
            await asyncio.sleep(.05)
            body = json.dumps({'object': 'list', 'data': []}).encode()
            writer.write(b'HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: '
                         + str(len(body)).encode() + b'\r\nConnection: close\r\n\r\n' + body)
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()
            finished.set()
    server = await asyncio.start_server(serve, '127.0.0.1', 0)
    port = server.sockets[0].getsockname()[1]
    if via_proxy:
        monkeypatch.setenv('HTTP_PROXY', f'http://127.0.0.1:{port}')
        monkeypatch.setenv('NO_PROXY', '')
    base_url = 'http://fixture.invalid/v1' if via_proxy else f'http://127.0.0.1:{port}/v1'
    provider = OpenAIProvider(api_key='fixture', config={'base_url': base_url})
    try:
        async with server:
            result = await provider.client.models.list(timeout=None)
        assert result.data == []
        assert inspected == [[value for _, _, value in _transport.keepalive_options()]]
    finally:
        await provider.close()
        server.close()
        await server.wait_closed()
        await asyncio.wait_for(finished.wait(), 2)


@pytest.mark.asyncio
async def test_proxy_trace_and_existing_callback_are_preserved():
    from types import SimpleNamespace
    calls, events = [], []
    async def previous(event, info):
        events.append(event)
    request = SimpleNamespace(extensions={'trace': previous})
    await _transport.configure_request(request)
    connection = SimpleNamespace(setsockopt=lambda *args: calls.append(args))
    stream = SimpleNamespace(get_extra_info=lambda key: connection if key == 'socket' else None)
    await request.extensions['trace']('connection.connect_tcp.complete', {'return_value': stream})
    assert calls == _transport.keepalive_options()
    assert events == ['connection.connect_tcp.complete']


@pytest.mark.asyncio
async def test_missing_or_unsupported_socket_option_does_not_fail_request():
    from types import SimpleNamespace
    request = SimpleNamespace(extensions={})
    await _transport.configure_request(request)
    await request.extensions['trace']('connection.connect_tcp.complete', {'return_value': None})
    def unsupported(*args):
        raise OSError('Not supported')
    stream = SimpleNamespace(get_extra_info=lambda key: SimpleNamespace(setsockopt=unsupported))
    await request.extensions['trace']('connection.connect_tcp.complete', {'return_value': stream})
