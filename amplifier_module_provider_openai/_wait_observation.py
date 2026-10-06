"""Optional, payload-free observations owned by one logical generation call."""

import asyncio
import logging
import math
import time
from typing import Any

import openai

logger = logging.getLogger(__name__)


def _seconds(value: Any) -> float | None:
    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    ):
        return float(value)
    return None


class _WaitObserver:
    """No timers: publish only actual activity, retaining one pending snapshot."""

    def __init__(self, hooks: Any) -> None:
        self._hooks = hooks if callable(getattr(hooks, "emit", None)) else None
        self._attempt = 0
        self._limits: dict[str, Any] = {}
        self._last_activity: float | None = None
        self._pending: dict[str, Any] | None = None

    async def _emit(self, payload: dict[str, Any]) -> None:
        if self._hooks is not None:
            cancellations = asyncio.current_task().cancelling()
            try:
                await self._hooks.emit("llm:progress", payload)
            except Exception:  # noqa: BLE001 - optional hook has no exception taxonomy
                # Observation failures must not fail or replay generation.
                logger.debug("Optional wait observation hook failed")
            except asyncio.CancelledError:
                # In-wait hooks run in the caller. Only an actual task cancel
                # (including an explicit asyncio deadline) propagates.
                if asyncio.current_task().cancelling() > cancellations:
                    raise
                logger.debug("Optional wait observation hook cancelled itself")

    async def attempt_started(self, timeout: Any, transport_timeout: Any) -> None:
        self._attempt += 1
        elapsed = None if isinstance(timeout, openai.Timeout) else _seconds(timeout)
        self._limits = {
            "mode": "phase" if isinstance(timeout, openai.Timeout) else (
                "elapsed" if elapsed is not None and elapsed > 0 else "none"
            ),
            "elapsed_seconds": elapsed if elapsed is not None and elapsed > 0 else None,
            **{
                f"{phase}_seconds": _seconds(getattr(transport_timeout, phase, None))
                for phase in ("connect", "pool", "read", "write")
            },
        }
        await self._emit(self._payload("attempt_started"))

    def _payload(self, observation: str) -> dict[str, Any]:
        return {
            "version": 1,
            "observation": observation,
            "attempt": self._attempt,
            "limits": dict(self._limits),
        }

    async def response_activity(self) -> None:
        if self._hooks is None:
            return
        now = time.monotonic()
        self._pending = self._payload("response_activity")
        if self._last_activity is None or now - self._last_activity >= 1.0:
            await self.flush()
            # Slow awaited hooks must not bunch later in-wait publications.
            self._last_activity = time.monotonic()

    async def flush(self) -> None:
        """Immediately publish one pending actual observation before settlement."""
        if self._pending is not None:
            payload, self._pending = self._pending, None
            await self._emit(payload)