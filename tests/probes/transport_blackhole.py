"""Manual Linux/DTU-only synthetic transport probe; requires root/iptables.

Owns loopback ports 18980/18981 and one temporary INPUT rule on source port
18980. Always removes that exact rule. Never contacts a real provider.
"""
import asyncio
import json
import subprocess
import time

import openai

from amplifier_module_provider_openai import OpenAIProvider

RULE = ['INPUT', '-p', 'tcp', '--sport', '18980', '-m', 'comment',
        '--comment', 'readiness100-tcp', '-j', 'DROP']

async def main():
    dropped = False
    handlers = set()
    errors = []
    async def serve(reader, writer, blackhole):
        nonlocal dropped
        task = asyncio.current_task()
        handlers.add(task)
        try:
            await reader.readuntil(b'\r\n\r\n')
            if blackhole:
                # Intentional blocking: no await between insertion and ownership.
                subprocess.run(['iptables', '-I', *RULE], check=True)  # noqa: ASYNC221
                dropped = True
                await asyncio.Event().wait()
            else:
                await asyncio.sleep(65)
                body = b'{"object":"list","data":[]}'
                writer.write(b'HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: '
                    + str(len(body)).encode() + b'\r\nConnection: close\r\n\r\n' + body)
                await writer.drain()
        except asyncio.CancelledError:
            raise
        # Collect unexpected failures so the final diagnostic assertion fails.
        except Exception as exc:  # noqa: BLE001
            errors.append(type(exc).__name__)
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except ConnectionResetError:
                pass  # Expected after the client declares the blackholed peer dead.
            handlers.discard(task)
    servers, clients = [], []
    async def probe(port, blackhole):
        provider = OpenAIProvider(api_key='synthetic-only', config={'base_url': f'http://127.0.0.1:{port}/v1'})
        clients.append(provider)
        started = time.monotonic()
        try:
            result = await asyncio.wait_for(provider.client.models.list(timeout=None), 80)
            assert not blackhole and result.data == []
            outcome = 'healthy-quiet-response'
        except openai.APIConnectionError:
            assert blackhole
            outcome = 'dead-peer-detected'
        return {'case': outcome, 'elapsedSeconds': round(time.monotonic()-started, 2)}
    try:
        for port, blackhole in [(18980, True), (18981, False)]:
            async def handler(reader, writer, lost=blackhole):
                await serve(reader, writer, lost)
            servers.append(await asyncio.start_server(handler, '127.0.0.1', port))
        results = await asyncio.gather(probe(18980, True), probe(18981, False))
        assert not errors
        print(json.dumps({'results': results, 'modelRequests': 0, 'realCredentials': False}), flush=True)
    finally:
        if dropped:
            # Intentional blocking: exact-rule deletion precedes resource cleanup.
            subprocess.run(['iptables', '-D', *RULE], check=True)  # noqa: ASYNC221
        for server in servers:
            server.close()
            await server.wait_closed()
        for task in list(handlers):
            task.cancel()
        await asyncio.gather(*list(handlers), return_exceptions=True)
        for client in clients:
            await client.close()

asyncio.run(main())
