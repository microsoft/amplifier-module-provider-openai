"""Tests for _accumulate hook and register_contributor in mount().

Verifies that mount() registers:
  - an `llm:response` hook (_accumulate) that sums cost_usd into a closure-captured dict
  - a lazy contributor callback on session.cost channel under name 'provider-openai'
"""

from decimal import Decimal

import pytest

from amplifier_module_provider_openai import mount


# ---------------------------------------------------------------------------
# Mock coordinator fixture
# ---------------------------------------------------------------------------


class _MockHooks:
    def __init__(self):
        self._handlers: dict = {}

    def register(self, event: str, handler) -> None:
        self._handlers[event] = handler

    async def emit(self, event: str, data: dict) -> None:
        if event in self._handlers:
            await self._handlers[event](event, data)


class _MockCoordinator:
    def __init__(self):
        self.hooks = _MockHooks()
        self.registered_hooks = self.hooks._handlers  # shared reference
        self.registered_contributors: dict = {}

    async def mount(self, *args, **kwargs) -> None:
        pass

    def register_contributor(self, channel: str, name: str, callback) -> None:
        self.registered_contributors[(channel, name)] = callback

    def get_capability(self, *args, **kwargs):
        return None


@pytest.fixture
def mock_coordinator():
    return _MockCoordinator()


# ---------------------------------------------------------------------------
# test_contributor_registered_at_mount
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_contributor_registered_at_mount(mock_coordinator):
    """mount() must register a contributor on ('session.cost', 'provider-openai')."""
    await mount(mock_coordinator, config={})
    assert (
        "session.cost",
        "provider-openai",
    ) in mock_coordinator.registered_contributors


# ---------------------------------------------------------------------------
# test_contributor_returns_none_before_any_calls
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_contributor_returns_none_before_any_calls(mock_coordinator):
    """Contributor callback returns None when no llm:response events have fired."""
    await mount(mock_coordinator, config={})
    callback = mock_coordinator.registered_contributors[
        ("session.cost", "provider-openai")
    ]
    assert callback() is None


# ---------------------------------------------------------------------------
# test_contributor_accumulates_after_llm_response
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_contributor_accumulates_after_llm_response(mock_coordinator):
    """_accumulate sums cost_usd over multiple events; callback returns Decimal total."""
    await mount(mock_coordinator, config={})

    accumulate = mock_coordinator.registered_hooks["llm:response"]
    callback = mock_coordinator.registered_contributors[
        ("session.cost", "provider-openai")
    ]

    await accumulate("llm:response", {"provider": "openai", "usage": {"cost_usd": "0.05"}})
    await accumulate("llm:response", {"provider": "openai", "usage": {"cost_usd": "0.03"}})

    result = callback()
    assert result is not None, "Callback should return a dict after cost events"
    assert "cost_usd" in result
    assert result["cost_usd"] == Decimal("0.08"), (
        f"Expected Decimal('0.08'), got {result['cost_usd']!r}"
    )
    assert isinstance(result["cost_usd"], Decimal), (
        f"cost_usd must be Decimal, got {type(result['cost_usd'])}"
    )


# ---------------------------------------------------------------------------
# test_contributor_ignores_none_cost
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_contributor_ignores_none_cost(mock_coordinator):
    """_accumulate ignores events where cost_usd is None; has_data stays False."""
    await mount(mock_coordinator, config={})

    accumulate = mock_coordinator.registered_hooks["llm:response"]
    callback = mock_coordinator.registered_contributors[
        ("session.cost", "provider-openai")
    ]

    await accumulate("llm:response", {"provider": "openai", "usage": {"cost_usd": None}})

    assert callback() is None, (
        "Callback should still return None after a None-cost event"
    )
