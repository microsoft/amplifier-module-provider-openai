"""Tests for process-wide concurrency semaphore and diagnostic logging.

Two features under test:

  Feature 1 — Semaphore
    A configurable ``max_concurrent_requests`` semaphore (default 5) limits how
    many API calls a single process can have in-flight simultaneously.  The
    semaphore is process-wide (shared across all OpenAIProvider instances) so
    that parent + delegated child sessions in the same process share the gate.
    Setting ``max_concurrent_requests=0`` disables the semaphore entirely.

  Feature 2 — Diagnostic logging
    Structured ``provider:concurrency`` events are emitted before each API call
    attempt, carrying the current active/waiting request counts, the configured
    limit, and os.getpid(), so that post-mortem analysis of events.jsonl can
    prove (or disprove) that concurrent request volume was responsible for a
    given issue.
"""

import asyncio
import os
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest
from amplifier_core import ModuleCoordinator
from amplifier_core.message_models import ChatRequest, Message
from openai import AsyncOpenAI

import amplifier_module_provider_openai as _mod
from amplifier_module_provider_openai import OpenAIProvider

# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------


class DummyResponse:
    """Minimal response stub that satisfies _convert_to_chat_response."""

    def __init__(self):
        self.output = []
        self.usage = SimpleNamespace(input_tokens=10, output_tokens=5)
        self.status = "completed"
        self.id = "resp_test"


class FakeHooks:
    """Records every (event_name, payload) pair emitted via hooks."""

    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    async def emit(self, name: str, payload: dict) -> None:
        self.events.append((name, payload))


class FakeCoordinator:
    def __init__(self):
        self.hooks = FakeHooks()


# ---------------------------------------------------------------------------
# Fixture: isolate module-level globals between tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_concurrency_globals():
    """Reset the process-wide semaphore and request counters before every test.

    asyncio.Semaphore objects are tied to the event loop that was running when
    they were created.  Each ``asyncio.run()`` call spins a new loop, so we
    must invalidate the cached semaphore to prevent "Future attached to a
    different loop" errors when tests run back-to-back.
    """
    _mod._process_semaphore = None
    _mod._process_semaphore_loop = None
    _mod._process_semaphore_max = 0
    _mod._active_requests = 0
    _mod._waiting_requests = 0
    yield
    # Clean up after the test too (defensive, avoids leaking into next fixture)
    _mod._process_semaphore = None
    _mod._process_semaphore_loop = None
    _mod._process_semaphore_max = 0
    _mod._active_requests = 0
    _mod._waiting_requests = 0


# ---------------------------------------------------------------------------
# Provider factory
# ---------------------------------------------------------------------------


def _make_provider(
    max_concurrent: int = 5, **extra_config
) -> tuple[OpenAIProvider, FakeCoordinator]:
    config = {
        "use_streaming": False,  # Use blocking path so tests can mock create()
        "max_retries": 0,
        "max_concurrent_requests": max_concurrent,
        **extra_config,
    }
    # Both Responses operations must be inert. Mocking only create() leaves the
    # final-input token counter using real transport before the semaphore gate.
    client = SimpleNamespace(
        base_url="https://api.openai.com/v1",
        responses=SimpleNamespace(
            create=AsyncMock(return_value=DummyResponse()),
            input_tokens=SimpleNamespace(
                count=AsyncMock(return_value=SimpleNamespace(input_tokens=10))
            ),
        ),
    )
    provider = OpenAIProvider(config=config, client=cast(AsyncOpenAI, client))
    coordinator = FakeCoordinator()
    provider.coordinator = cast(ModuleCoordinator, coordinator)
    return provider, coordinator


def _simple_request() -> ChatRequest:
    return ChatRequest(messages=[Message(role="user", content="Hello")])


async def _assert_concurrency(providers, expected_peak, monkeypatch):
    """Hold API calls until every request has reached the real semaphore gate."""
    at_gate = 0
    all_at_gate = asyncio.Event()
    release = asyncio.Event()
    in_flight = 0
    peak = 0
    calls = 0
    get_semaphore = _mod._get_process_semaphore

    async def observe_gate(limit):
        nonlocal at_gate
        semaphore = await get_semaphore(limit)
        at_gate += 1
        if at_gate == len(providers):
            all_at_gate.set()
        return semaphore

    async def held_api(**kwargs):
        nonlocal in_flight, peak, calls
        calls += 1
        in_flight += 1
        peak = max(peak, in_flight)
        try:
            await release.wait()
            return DummyResponse()
        finally:
            in_flight -= 1

    monkeypatch.setattr(_mod, "_get_process_semaphore", observe_gate)
    for provider in providers:
        provider.client.responses.create = held_api
    tasks = [
        asyncio.create_task(provider.complete(_simple_request()))
        for provider in providers
    ]
    try:
        # The timeout only bounds a broken test. No sleep duration determines
        # overlap: all requests reach the gate before any API call is released.
        async with asyncio.timeout(5):
            await all_at_gate.wait()
            assert in_flight == expected_peak
            release.set()
            results = await asyncio.gather(*tasks)
        assert peak == expected_peak
        assert calls == len(providers)
        assert all(result is not None for result in results)
        assert in_flight == 0
        assert _mod._active_requests == _mod._waiting_requests == 0
        for provider in {id(p): p for p in providers}.values():
            assert provider.client.responses.input_tokens.count.await_count == sum(
                candidate is provider for candidate in providers
            )
    finally:
        release.set()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


# ============================================================================
# Feature 1: Semaphore — configuration
# ============================================================================


class TestSemaphoreConfig:
    """Unit tests for max_concurrent_requests config parsing."""

    def test_default_is_5(self):
        """Default max_concurrent_requests should be 5."""
        provider = OpenAIProvider(api_key="test-key")
        assert provider._max_concurrent_requests == 5

    def test_config_overrides_default(self):
        provider = OpenAIProvider(
            api_key="test-key", config={"max_concurrent_requests": 3}
        )
        assert provider._max_concurrent_requests == 3

    def test_zero_disables_semaphore(self):
        provider = OpenAIProvider(
            api_key="test-key", config={"max_concurrent_requests": 0}
        )
        assert provider._max_concurrent_requests == 0

    def test_get_process_semaphore_returns_none_for_zero(self):
        async def _run():
            sem = await _mod._get_process_semaphore(0)
            assert sem is None

        asyncio.run(_run())

    def test_get_process_semaphore_returns_semaphore_for_positive(self):
        async def _run():
            sem = await _mod._get_process_semaphore(3)
            assert sem is not None
            assert isinstance(sem, asyncio.Semaphore)

        asyncio.run(_run())

    def test_get_process_semaphore_is_idempotent_within_same_loop(self):
        """Same semaphore instance should be reused within one event loop."""

        async def _run():
            sem1 = await _mod._get_process_semaphore(5)
            sem2 = await _mod._get_process_semaphore(5)
            assert sem1 is sem2

        asyncio.run(_run())

    def test_get_process_semaphore_refreshes_across_loops(self):
        """Semaphore created in loop A must not be reused in loop B."""
        sem_from_loop_a: asyncio.Semaphore | None = None

        async def _loop_a():
            nonlocal sem_from_loop_a
            sem_from_loop_a = await _mod._get_process_semaphore(5)

        asyncio.run(_loop_a())
        # Reset only the loop reference to simulate a new run (globals fixture
        # already ensures a clean slate, but we want a specific mid-test reset)
        _mod._process_semaphore_loop = None  # trigger recreation

        async def _loop_b():
            sem = await _mod._get_process_semaphore(5)
            # Different object — must have been recreated for this loop
            assert sem is not sem_from_loop_a

        asyncio.run(_loop_b())


# ============================================================================
# Feature 1: Semaphore — concurrency enforcement
# ============================================================================


class TestSemaphoreLimitsConcurrency:
    """Verify that at most max_concurrent API calls are in-flight at once."""

    def test_semaphore_limits_concurrent_calls(self, monkeypatch):
        """With limit=2 and 5 concurrent tasks, peak in-flight must equal 2."""
        provider, _ = _make_provider(max_concurrent=2)
        asyncio.run(_assert_concurrency([provider] * 5, 2, monkeypatch))

    def test_semaphore_limit_of_1_serializes_calls(self, monkeypatch):
        """Limit of 1 must fully serialize all API calls."""
        provider, _ = _make_provider(max_concurrent=1)

        asyncio.run(_assert_concurrency([provider] * 4, 1, monkeypatch))

    def test_disabled_semaphore_allows_full_concurrency(self, monkeypatch):
        """With max_concurrent=0, all calls run without a gate."""
        provider, _ = _make_provider(max_concurrent=0)

        asyncio.run(_assert_concurrency([provider] * 5, 5, monkeypatch))

    def test_all_requests_complete_with_semaphore(self, monkeypatch):
        """Semaphore must not prevent any request from completing."""
        provider, _ = _make_provider(max_concurrent=2)

        asyncio.run(_assert_concurrency([provider] * 6, 2, monkeypatch))

    def test_semaphore_is_shared_across_provider_instances(self, monkeypatch):
        first, _ = _make_provider(max_concurrent=2)
        second, _ = _make_provider(max_concurrent=2)
        asyncio.run(_assert_concurrency([first, second] * 3, 2, monkeypatch))


# ============================================================================
# Feature 2: provider:concurrency event emission
# ============================================================================


class TestConcurrencyEventEmission:
    """Verify provider:concurrency events are emitted with correct payload."""

    def test_event_emitted_on_success(self):
        """A provider:concurrency event should be emitted for each API call."""
        provider, coordinator = _make_provider(max_concurrent=5)
        provider.client.responses.create = AsyncMock(return_value=DummyResponse())

        request = _simple_request()
        asyncio.run(provider.complete(request))

        concurrency_events = [
            e for e in coordinator.hooks.events if e[0] == "provider:concurrency"
        ]
        assert len(concurrency_events) >= 1

    def test_event_has_all_required_fields(self):
        """provider:concurrency payload must contain every documented field."""
        provider, coordinator = _make_provider(max_concurrent=3)
        provider.client.responses.create = AsyncMock(return_value=DummyResponse())

        request = _simple_request()
        asyncio.run(provider.complete(request))

        concurrency_events = [
            e for e in coordinator.hooks.events if e[0] == "provider:concurrency"
        ]
        assert len(concurrency_events) >= 1
        payload = concurrency_events[0][1]

        for field in (
            "provider",
            "model",
            "active_requests",
            "waiting_requests",
            "max_concurrent",
            "process_id",
        ):
            assert field in payload, (
                f"Missing field '{field}' in provider:concurrency event"
            )

    def test_event_provider_name_is_openai(self):
        """provider field in event must be 'openai'."""
        provider, coordinator = _make_provider(max_concurrent=5)
        provider.client.responses.create = AsyncMock(return_value=DummyResponse())

        request = _simple_request()
        asyncio.run(provider.complete(request))

        concurrency_events = [
            e for e in coordinator.hooks.events if e[0] == "provider:concurrency"
        ]
        assert concurrency_events[0][1]["provider"] == "openai"

    def test_event_max_concurrent_matches_config(self):
        """max_concurrent in event must equal the configured value."""
        provider, coordinator = _make_provider(max_concurrent=7)
        provider.client.responses.create = AsyncMock(return_value=DummyResponse())

        request = _simple_request()
        asyncio.run(provider.complete(request))

        concurrency_events = [
            e for e in coordinator.hooks.events if e[0] == "provider:concurrency"
        ]
        assert concurrency_events[0][1]["max_concurrent"] == 7

    def test_event_process_id_matches_current_process(self):
        """process_id in event must equal os.getpid()."""
        provider, coordinator = _make_provider(max_concurrent=5)
        provider.client.responses.create = AsyncMock(return_value=DummyResponse())

        request = _simple_request()
        asyncio.run(provider.complete(request))

        concurrency_events = [
            e for e in coordinator.hooks.events if e[0] == "provider:concurrency"
        ]
        assert concurrency_events[0][1]["process_id"] == os.getpid()

    def test_event_emitted_when_semaphore_disabled(self):
        """provider:concurrency is emitted even when max_concurrent=0."""
        provider, coordinator = _make_provider(max_concurrent=0)
        provider.client.responses.create = AsyncMock(return_value=DummyResponse())

        request = _simple_request()
        asyncio.run(provider.complete(request))

        concurrency_events = [
            e for e in coordinator.hooks.events if e[0] == "provider:concurrency"
        ]
        assert len(concurrency_events) >= 1
        # max_concurrent should be 0 (disabled) in the payload
        assert concurrency_events[0][1]["max_concurrent"] == 0

    def test_active_requests_at_least_1_during_call(self):
        """active_requests in the event should be ≥ 1 (the call itself)."""
        provider, coordinator = _make_provider(max_concurrent=5)
        provider.client.responses.create = AsyncMock(return_value=DummyResponse())

        request = _simple_request()
        asyncio.run(provider.complete(request))

        concurrency_events = [
            e for e in coordinator.hooks.events if e[0] == "provider:concurrency"
        ]
        assert concurrency_events[0][1]["active_requests"] >= 1

    def test_no_event_emitted_without_coordinator(self):
        """When no coordinator is attached, provider:concurrency is silently skipped."""
        provider, _ = _make_provider(max_concurrent=5)
        provider.coordinator = None
        provider.client.responses.create = AsyncMock(return_value=DummyResponse())

        request = _simple_request()
        # Should not raise even without a coordinator
        result = asyncio.run(provider.complete(request))
        assert result is not None
