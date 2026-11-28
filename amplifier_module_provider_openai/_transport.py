"""Detect lost TCP peers without putting a deadline on model reasoning.

The HTTPX trace extension exposes each newly connected socket, including proxy
connections. Keep the SDK's client defaults, TLS and environment proxy routing.
"""

import logging
import socket
from contextlib import contextmanager
from contextvars import ContextVar

import httpx
import httpx2

logger = logging.getLogger(__name__)
_response_state = ContextVar("openai_response_state", default=None)


async def _received_headers(response):
    state = _response_state.get()
    if state is not None:
        state["received"] = True


def watch_response(http_client):
    if isinstance(http_client, (httpx.AsyncClient, httpx2.AsyncClient)):
        hooks = http_client.event_hooks.setdefault("response", [])
        if _received_headers not in hooks:
            hooks.append(_received_headers)


@contextmanager
def observe_response():
    """Observe headers before the SDK reads/wraps a failing response body.

    The hook is context-local, not provider/client state: a shared HTTP client
    and concurrent requests cannot transfer an observation to another attempt.
    A mutable cell also carries observations out of asyncio.wait_for's task.
    """
    token = _response_state.set({"received": False})
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
