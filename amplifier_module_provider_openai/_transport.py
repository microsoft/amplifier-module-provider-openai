"""Detect lost TCP peers without putting a deadline on model reasoning.

The HTTPX trace extension exposes each newly connected socket, including proxy
connections. Keep the SDK's client defaults, TLS and environment proxy routing.
"""

import logging
import socket

logger = logging.getLogger(__name__)


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
