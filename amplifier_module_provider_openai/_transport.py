"""TCP keepalive and positive evidence from the public HTTPX trace extension."""
import logging
import socket
import ssl
from contextlib import contextmanager
from contextvars import ContextVar

import httpx
import httpx2

logger = logging.getLogger(__name__)
_response_state = ContextVar("openai_transport_evidence", default=None)


@contextmanager
def observe_response():
    token = _response_state.set({"requests": {}, "received": False, "unknown": False})
    try:
        yield
    finally:
        _response_state.reset(token)


def reset_response_observation():
    # Counting can precede generation inside this same attempt scope.
    _response_state.set({"requests": {}, "received": False, "unknown": False})


async def trace_request(request):
    state = _response_state.get()
    if state is None:
        return
    try:
        previous = request.extensions.get("trace")
        record = state["requests"].get(request)
        if record is not None and record["trace"] is previous:
            return
        if hasattr(previous, "_openai_previous"):
            previous = previous._openai_previous
        record = {"seen": False}

        async def trace(event, info):
            try:
                record["seen"] = True
                if event.endswith("receive_response_headers.complete") or "receive_response_body" in event:
                    state["received"] = True
                if event.startswith("http2."):
                    state["unknown"] = True  # No local HTTP/2 qualification.
            except Exception:  # noqa: BLE001 - evidence errors mean unknown
                if state is not None:
                    state["unknown"] = True
            if previous is not None:
                await previous(event, info)

        trace._openai_previous = previous
        record["trace"] = trace
        state["requests"][request] = record
        request.extensions["trace"] = trace
    except Exception:  # noqa: BLE001 - optional evidence must not break requests
        if state is not None:
            state["unknown"] = True


def watch_response(http_client):
    try:
        hooks = http_client.event_hooks.setdefault("request", [])
        if trace_request not in hooks:
            hooks.append(trace_request)
    except Exception:  # noqa: BLE001 - no trace report means unknown
        return


def can_promote(error):
    try:
        state = _response_state.get()
        request = error.request
        trace = request.extensions.get("trace")
        record = state["requests"].get(request) if state else None
        return bool(record and trace is record["trace"] and record["seen"] and not state["received"]
                    and not state["unknown"] and not certificate_failure(error))
    except Exception:  # noqa: BLE001 - always preserve the original SDK failure
        return False


def certificate_failure(error):
    seen, pending = set(), [error]
    while pending:
        error = pending.pop()
        if error is None or id(error) in seen:
            continue
        seen.add(id(error))
        if isinstance(error, ssl.SSLCertVerificationError):
            return True
        pending.extend((error.__cause__, error.__context__))
    return False


def is_connection_failure(error):
    return isinstance(error.__cause__, (
        httpx.NetworkError, httpx.RemoteProtocolError,
        httpx2.NetworkError, httpx2.RemoteProtocolError,
    ))


def keepalive_options():
    options = [(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)]
    idle = getattr(socket, "TCP_KEEPIDLE", getattr(socket, "TCP_KEEPALIVE", None))
    for option, value in (
        (idle, 30), (getattr(socket, "TCP_KEEPINTVL", None), 10),
        (getattr(socket, "TCP_KEEPCNT", None), 3),
    ):
        if option is not None:
            options.append((socket.IPPROTO_TCP, option, value))
    return options


async def configure_request(request):
    previous = request.extensions.get("trace")
    # Redirects copy request extensions. Replace only our direct wrappers so
    # a new request gets one keepalive callback and one evidence callback.
    if hasattr(previous, "_openai_previous"):
        previous = previous._openai_previous
    if hasattr(previous, "_openai_keepalive_previous"):
        previous = previous._openai_keepalive_previous

    async def trace(event, info):
        if event.endswith("connect_tcp.complete"):
            stream = info.get("return_value")
            connection = stream.get_extra_info("socket") if stream is not None else None
            if connection is not None:
                for level, option, value in keepalive_options():
                    try:
                        connection.setsockopt(level, option, value)
                    except OSError:
                        logger.warning("TCP keepalive option %s is unavailable", option)
        if previous is not None:
            await previous(event, info)

    trace._openai_keepalive_previous = previous
    request.extensions["trace"] = trace
