"""Exact metadata and monotonic pacing, independently of SDK wire fixtures."""

import asyncio
from types import SimpleNamespace

import openai
import pytest

from amplifier_module_provider_openai import _wait_observation as module
from amplifier_module_provider_openai._wait_observation import _WaitObserver


class Hooks:
    def __init__(self):
        self.events = []

    async def emit(self, name, payload):
        self.events.append((name, payload))


@pytest.mark.asyncio
async def test_logical_throttle_and_terminal_spacing_exception(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: now[0]))
    hooks = Hooks()
    observer = _WaitObserver(hooks)
    timeout = openai.Timeout(None, connect=5, pool=5)
    await observer.attempt_started(None, timeout)
    await observer.response_activity()  # A, t=0
    now[0] = 0.10
    await observer.response_activity()  # B, suppressed while waiting
    assert len(hooks.events) == 2
    now[0] = 0.11
    await observer.flush()  # No delay: B retained before terminal settlement.
    assert [data["observation"] for _, data in hooks.events] == [
        "attempt_started", "response_activity", "response_activity",
    ]
    await observer.flush()
    assert len(hooks.events) == 3

    # Neither a physical retry nor continuation resets the logical clock.
    await observer.attempt_started(7, openai.Timeout(7, connect=5, pool=5))
    now[0] = 0.20
    await observer.response_activity()
    assert len(hooks.events) == 4
    await observer.attempt_started(None, timeout)
    now[0] = 1.0
    await observer.response_activity()
    assert len(hooks.events) == 6
    assert hooks.events[-1][1]["attempt"] == 3
    assert hooks.events[-1][1]["limits"]["elapsed_seconds"] is None


@pytest.mark.asyncio
async def test_pending_snapshot_retains_actual_attempt_and_limits(monkeypatch):
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: 0))
    hooks = Hooks()
    observer = _WaitObserver(hooks)
    await observer.attempt_started(7, openai.Timeout(7, connect=5, pool=5))
    await observer.response_activity()
    await observer.response_activity()
    await observer.attempt_started(None, openai.Timeout(None, connect=5, pool=5))
    await observer.flush()
    assert hooks.events[-1][1]["attempt"] == 1
    assert hooks.events[-1][1]["limits"]["elapsed_seconds"] == 7


@pytest.mark.asyncio
async def test_slow_hook_does_not_bunch_in_wait_publications(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: now[0]))

    class Slow(Hooks):
        async def emit(self, name, payload):
            await super().emit(name, payload)
            if payload["observation"] == "response_activity":
                now[0] += 2

    hooks = Slow()
    observer = _WaitObserver(hooks)
    await observer.attempt_started(None, openai.Timeout(None))
    await observer.response_activity()
    assert now[0] == 2
    now[0] = 2.1
    await observer.response_activity()
    assert len(hooks.events) == 2
    now[0] = 3.0
    await observer.response_activity()
    assert len(hooks.events) == 3


@pytest.mark.asyncio
async def test_exact_payload_and_no_silent_wait_timer(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Observation must not sleep, spawn, or schedule")

    monkeypatch.setattr(asyncio, "sleep", forbidden)
    monkeypatch.setattr(asyncio, "create_task", forbidden)
    monkeypatch.setattr(asyncio.get_running_loop(), "call_later", forbidden)
    hooks = Hooks()
    observer = _WaitObserver(hooks)
    phase = openai.Timeout(None, connect=0, pool=2, read=None, write=4)
    await observer.attempt_started(phase, phase)
    await observer.flush()
    assert len(hooks.events) == 1
    name, data = hooks.events[0]
    assert name == "llm:progress"
    assert data == {
        "version": 1, "observation": "attempt_started", "attempt": 1,
        "limits": {"mode": "phase", "elapsed_seconds": None,
                   "connect_seconds": 0.0, "pool_seconds": 2.0,
                   "read_seconds": None, "write_seconds": 4.0},
    }
    await observer.response_activity()
    await observer.response_activity()
    await observer.flush()
    assert len(hooks.events) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", [float("inf"), float("nan"), -1, True, "private-secret"])
async def test_nonfinite_or_untyped_limits_not_retained(bad):
    hooks = Hooks()
    observer = _WaitObserver(hooks)
    await observer.attempt_started(None, SimpleNamespace(
        connect=bad, pool=bad, read=bad, write=bad))
    assert hooks.events[0][1]["limits"] == {
        "mode": "none", "elapsed_seconds": None, "connect_seconds": None,
        "pool_seconds": None, "read_seconds": None, "write_seconds": None,
    }


@pytest.mark.asyncio
async def test_optional_hooks_and_hook_exceptions_do_not_fail_observations():
    class Broken:
        async def emit(self, *_):
            raise RuntimeError("private-hook-error")

    for hooks in (None, object(), SimpleNamespace(emit=None), Broken()):
        observer = _WaitObserver(hooks)
        await observer.attempt_started(None, openai.Timeout(None, connect=5, pool=5))
        await observer.response_activity()
        await observer.response_activity()
        await observer.flush()


@pytest.mark.asyncio
async def test_hook_cancellation_propagates():
    class Cancel:
        async def emit(self, *_):
            raise asyncio.CancelledError

    observer = _WaitObserver(Cancel())
    with pytest.raises(asyncio.CancelledError):
        await observer.attempt_started(None, openai.Timeout(None))