"""Detect lost TCP peers without putting a deadline on model reasoning.

The HTTPX trace extension exposes each newly connected socket, including proxy
connections. Keep the SDK's client defaults, TLS and environment proxy routing.
"""

import logging
import socket
import ssl
from contextlib import contextmanager
from contextvars import ContextVar
from weakref import WeakSet

import httpx
import httpx2

logger = logging.getLogger(__name__)
_response_state = ContextVar("openai_response_state", default=None)
_installed = WeakSet()
_CLIENT_TYPES = (httpx.AsyncClient, httpx2.AsyncClient)
try:
    from openai._base_client import AsyncHttpxClientWrapper, DefaultAsyncHttpxClient

    _CLIENT_TYPES += (AsyncHttpxClientWrapper, DefaultAsyncHttpxClient)
except (ImportError, AttributeError):
    pass  # Unknown SDK defaults do not qualify for recovery.


def _client_module(client):
    if type(client) not in _CLIENT_TYPES:
        return None
    return httpx2 if isinstance(client, httpx2.AsyncClient) else httpx


async def _received_headers(response):
    state = _response_state.get()
    if state is not None:
        state["received"] = True


def watch_response(http_client):
    try:
        if _client_module(http_client) is None:
            return
        hooks = http_client.event_hooks.setdefault("response", [])
        if http_client not in _installed:
            # Install once. Never reorder a list that another request may be
            # iterating. Displacement/removal permanently means unknown.
            _installed.add(http_client)
            hooks.insert(0, _received_headers)
        state = _response_state.get()
        if state is not None:
            state["client"] = http_client
    except Exception:  # noqa: BLE001 - any observation failure means unknown
        return  # Optional observation must never replace the provider error.


def observation_confirmed():
    try:
        state = _response_state.get()
        client = state.get("client") if state is not None else None
        module = _client_module(client)
        if module is None or client not in _installed:
            return False
        hooks = client.event_hooks.get("response", [])
        if not hooks or hooks[0] is not _received_headers:
            return False
        # Caller auth/request hooks can perform their own body reads before
        # HTTPX's response boundary. Only our socket-only request hook is known.
        if client.auth is not None or any(
            hook is not configure_request for hook in client.event_hooks.get("request", [])
        ):
            return False
        transports = [client._transport, *client._mounts.values()]
        return all(t is None or (
            type(t) is module.AsyncHTTPTransport and not t._pool._http2
        ) for t in transports)
    except Exception:  # noqa: BLE001 - library-private drift must fail closed
        return False


def can_promote(error):
    return observation_confirmed() and not response_received() and not certificate_failure(error)


def certificate_failure(error):
    seen = set()
    pending = [error]
    while pending:
        error = pending.pop()
        if error is None or id(error) in seen:
            continue
        seen.add(id(error))
        if isinstance(error, ssl.SSLCertVerificationError):
            return True
        pending.extend((error.__cause__, error.__context__))
    return False


@contextmanager
def observe_response():
    """Observe headers before the SDK reads/wraps a failing response body.

    The hook is context-local, not provider/client state: a shared HTTP client
    and concurrent requests cannot transfer an observation to another attempt.
    A mutable cell also carries observations out of asyncio.wait_for's task.
    """
    token = _response_state.set({"received": False, "client": None})
    try:
        yield
    finally:
        _response_state.reset(token)


def response_received():
    state = _response_state.get()
    return state is not None and state["received"]


def reset_response_observation():
    # Native input counting may precede generation in the same attempt.
    state = _response_state.get()
    if state is not None:
        state["received"] = False
        state["client"] = None


def is_connection_failure(error):
    """Require transport evidence, not merely the SDK's catch-all wrapper.

    The SDK also wraps configuration errors (UnsupportedProtocol, InvalidURL)
    in APIConnectionError. Neither is a lost connection. Include legacy HTTPX
    because the SDK still accepts caller-supplied HTTPX clients.
    """
    return isinstance(error.__cause__, (
        httpx.NetworkError, httpx.RemoteProtocolError,
        httpx2.NetworkError, httpx2.RemoteProtocolError,
    ))


def keepalive_options():
    options = [(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)]
    # Darwin calls the idle interval TCP_KEEPALIVE; Linux/Windows use KEEP IDLE.
    idle = getattr(socket, "TCP_KEEPIDLE", getattr(socket, "TCP_KEEPALIVE", None))
    for option, value in (
        (idle, 30),
        (getattr(socket, "TCP_KEEPINTVL", None), 10),
        (getattr(socket, "TCP_KEEPCNT", None), 3),
    ):
        if option is not None:
            options.append((socket.IPPROTO_TCP, option, value))
    return options


async def configure_request(request):
    previous = request.extensions.get("trace")

    async def trace(event, info):
        if event.endswith("connect_tcp.complete"):
            stream = info.get("return_value")
            connection = stream.get_extra_info("socket") if stream is not None else None
            if connection is not None:
                for level, option, value in keepalive_options():
                    try:
                        connection.setsockopt(level, option, value)
                    except OSError:
                        # A platform may expose a constant unsupported by its
                        # running kernel. Retain its other supported probes.
                        logger.warning("TCP keepalive option %s is unavailable", option)
        if previous is not None:
            await previous(event, info)

    request.extensions["trace"] = trace
