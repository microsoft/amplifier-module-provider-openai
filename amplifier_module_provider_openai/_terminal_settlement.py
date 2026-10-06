"""Best-effort local terminal notifications, never generation or retry work."""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

logger = logging.getLogger(__name__)

# Three terminal deliveries (abort, pending progress, outcome) take at most
# 150ms plus cooperative cancellation/drain. Hooks must yield and honor cancel;
# asyncio cannot forcibly stop synchronous or cancellation-suppressing code.
_DELIVERY_SECONDS = 0.05


def _caller_cancellations() -> int:
    """Synchronous accounting also runs outside an event loop in callers/tests."""
    try:
        task = asyncio.current_task()
    except RuntimeError:
        return 0
    return task.cancelling() if task is not None else 0


async def _settle_optional(deliver: Callable[[], Awaitable[Any]]) -> None:
    """Own and drain delivery; only a new cancellation of our caller escapes."""
    async def delivery() -> None:
        try:
            await deliver()
        except (Exception, asyncio.CancelledError):  # noqa: BLE001 - arbitrary optional hook
            # This is the owned child, not the caller. A hook's own cancellation
            # must not masquerade as Stop or replace an existing provider error.
            logger.debug("Optional terminal delivery unavailable")

    task = asyncio.create_task(delivery())
    cancellation = None
    try:
        await asyncio.wait({task}, timeout=_DELIVERY_SECONDS)
    except asyncio.CancelledError as exc:
        cancellation = exc
    finally:
        if not task.done():
            task.cancel()
        while not task.done():
            cancellations = _caller_cancellations()
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError as exc:
                # Retain the newest real caller cancellation, including its
                # message/count. Never uncancel the caller or detach the child.
                # shield may also report a child cancelled before it started.
                if _caller_cancellations() > cancellations:
                    cancellation = exc
        try:
            task.result()
        except (Exception, asyncio.CancelledError):  # noqa: BLE001 - consume owned child only
            # Cancellation can arrive before the delivery coroutine even starts.
            logger.debug("Optional terminal task unavailable")
    if cancellation is not None:
        raise cancellation