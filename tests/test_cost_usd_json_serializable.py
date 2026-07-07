"""Regression tests: cost_usd JSON-serialization boundary fix.

Documents Brian's correct fix pattern (ref: amplifier-module-provider-anthropic#54):
  - cost_usd stays as Decimal INTERNALLY (in Usage model and _totals accumulator)
  - str() conversion happens only at EMISSION BOUNDARIES:
      1. event_usage["cost_usd"] in the llm:response emit block
      2. register_contributor("session.cost", ...) lambda

A raw Decimal on an emitted payload would crash the Rust hook registry which
calls json.dumps() on every payload before dispatch.

References:
  - microsoft-amplifier/amplifier-support#225
  - microsoft/amplifier-module-provider-anthropic#54
"""

import asyncio
import json
from decimal import Decimal
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest
from amplifier_core import ModuleCoordinator
from amplifier_core.message_models import ChatRequest, Message

from amplifier_module_provider_openai import OpenAIProvider


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class CapturingHooks:
    """Records every (name, payload) pair emitted via coordinator.hooks.emit()."""

    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    async def emit(self, name: str, payload: dict) -> None:
        self.events.append((name, payload))


class FakeCoordinator:
    def __init__(self):
        self.hooks = CapturingHooks()


class DummyResponseWithModel:
    """Minimal response stub backed by a priced model.

    model="gpt-5.4" is a known-priced model; 1M tokens each side yields a
    non-zero Decimal from compute_cost(), exercising the serialization path.
    """

    def __init__(self):
        self.output = [
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text="Hello")],
            )
        ]
        self.usage = SimpleNamespace(
            input_tokens=1_000_000,
            output_tokens=1_000_000,
        )
        self.status = "completed"
        self.id = "resp_cost_boundary_test"
        self.model = "gpt-5.4"  # priced model — non-zero Decimal cost

    def model_dump(self):
        return {"id": self.id, "status": self.status}


class DummyResponseWithUnknownModel:
    """Response stub with an unrecognised model so cost_usd stays None."""

    def __init__(self):
        self.output = [
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text="Hi")],
            )
        ]
        self.usage = SimpleNamespace(
            input_tokens=1_000_000,
            output_tokens=1_000_000,
        )
        self.status = "completed"
        self.id = "resp_unknown_model_test"
        self.model = "gpt-unknown-9999"  # not in pricing table → cost_usd = None

    def model_dump(self):
        return {"id": self.id, "status": self.status}


def _make_provider(**config_overrides) -> OpenAIProvider:
    config = {
        "use_streaming": False,  # non-streaming path so we can mock create()
        **config_overrides,
    }
    return OpenAIProvider(api_key="test-key", config=config)


def _simple_request() -> ChatRequest:
    return ChatRequest(messages=[Message(role="user", content="Hello")])


def _get_llm_response_payload(events: list[tuple[str, dict]]) -> dict:
    return next(payload for name, payload in events if name == "llm:response")


# ---------------------------------------------------------------------------
# Test 1 — llm:response event must be JSON-serialisable for a priced model
# ---------------------------------------------------------------------------


def test_llm_response_event_is_json_serializable_known_model():
    """json.dumps(llm:response payload) must not raise for a known-priced model.

    The Rust hook registry calls json.dumps() on every emitted payload.  A raw
    Decimal stored at event_usage["cost_usd"] would crash that path with a
    TypeError.  The fix converts to str() at the emission boundary only.
    """
    provider = _make_provider()
    fake_coordinator = FakeCoordinator()
    provider.coordinator = cast(ModuleCoordinator, fake_coordinator)
    provider.client.responses.create = AsyncMock(return_value=DummyResponseWithModel())

    asyncio.run(provider.complete(_simple_request()))

    payload = _get_llm_response_payload(fake_coordinator.hooks.events)

    try:
        json.dumps(payload)
    except TypeError as exc:
        cost_usd = payload.get("usage", {}).get("cost_usd")
        pytest.fail(
            f"json.dumps(llm:response payload) raised TypeError: {exc}\n"
            f"cost_usd={cost_usd!r} (type={type(cost_usd).__name__})"
        )


# ---------------------------------------------------------------------------
# Test 2 — cost_usd in the emitted payload must be a str for a priced model
# ---------------------------------------------------------------------------


def test_llm_response_event_cost_usd_is_str_for_known_model():
    """cost_usd in llm:response event usage must be str, not Decimal.

    Decimal is not JSON-native; the emission boundary converts it to str.
    The numeric value must be positive (gpt-5.4 with 1M tokens each side).
    """
    provider = _make_provider()
    fake_coordinator = FakeCoordinator()
    provider.coordinator = cast(ModuleCoordinator, fake_coordinator)
    provider.client.responses.create = AsyncMock(return_value=DummyResponseWithModel())

    asyncio.run(provider.complete(_simple_request()))

    payload = _get_llm_response_payload(fake_coordinator.hooks.events)
    cost_usd = payload.get("usage", {}).get("cost_usd")

    assert cost_usd is not None, (
        "cost_usd should be set for a priced model with non-zero tokens"
    )
    assert isinstance(cost_usd, str), (
        f"cost_usd in emitted event must be str (JSON-safe), "
        f"got {type(cost_usd).__name__!r}: {cost_usd!r}"
    )
    assert not isinstance(cost_usd, Decimal), (
        f"cost_usd must not be Decimal in emitted payload; got {cost_usd!r}"
    )
    # 1M input @ $2.50/M + 1M output @ $15.00/M = $17.50
    assert Decimal(cost_usd) > 0, f"Parsed cost must be positive, got {cost_usd!r}"


# ---------------------------------------------------------------------------
# Test 3 — unknown model must leave cost_usd as None in the emitted payload
# ---------------------------------------------------------------------------


def test_llm_response_event_cost_usd_is_none_for_unknown_model():
    """An unrecognised model must emit cost_usd=None, not raise or produce 0.

    None distinguishes "cost unknown" from "cost was zero"; it must be
    preserved as-is through the str() boundary guard.
    """
    provider = _make_provider()
    fake_coordinator = FakeCoordinator()
    provider.coordinator = cast(ModuleCoordinator, fake_coordinator)
    provider.client.responses.create = AsyncMock(
        return_value=DummyResponseWithUnknownModel()
    )

    asyncio.run(provider.complete(_simple_request()))

    payload = _get_llm_response_payload(fake_coordinator.hooks.events)
    cost_usd = payload.get("usage", {}).get("cost_usd")

    assert cost_usd is None, (
        f"Unknown model must emit cost_usd=None, got {cost_usd!r} "
        f"(type={type(cost_usd).__name__})"
    )


# ---------------------------------------------------------------------------
# Test 4 — cost_usd must survive a json.dumps/loads round-trip unchanged
# ---------------------------------------------------------------------------


def test_llm_response_event_cost_usd_round_trips_through_json():
    """cost_usd value must be identical before and after a json.dumps/loads cycle.

    Confirms that the str representation is lossless for the values produced
    by compute_cost() using Decimal arithmetic.
    """
    provider = _make_provider()
    fake_coordinator = FakeCoordinator()
    provider.coordinator = cast(ModuleCoordinator, fake_coordinator)
    provider.client.responses.create = AsyncMock(return_value=DummyResponseWithModel())

    asyncio.run(provider.complete(_simple_request()))

    payload = _get_llm_response_payload(fake_coordinator.hooks.events)
    cost_before = payload.get("usage", {}).get("cost_usd")

    assert cost_before is not None, "Precondition: cost_usd must be set"

    serialised = json.dumps(payload)
    restored_payload = json.loads(serialised)
    cost_after = restored_payload.get("usage", {}).get("cost_usd")

    assert cost_before == cost_after, (
        f"cost_usd changed across json round-trip: {cost_before!r} → {cost_after!r}"
    )


# ---------------------------------------------------------------------------
# Test 5 — Usage model stores Decimal internally (fix is at boundary only)
# ---------------------------------------------------------------------------


def test_usage_model_direct_access_is_decimal_but_model_dump_is_str():
    """result.usage.cost_usd is Decimal via direct attribute access, but
    model_dump() always stringifies it — even in plain (non-JSON) mode.

    Brian's fix keeps Decimal *inside* the system for arithmetic precision;
    direct attribute access on the Usage instance bypasses serialization and
    returns the raw Decimal untouched.

    However, `amplifier_core.message_models.Usage.cost_usd` carries:

        @field_serializer("cost_usd", when_used="always")
        def serialize_cost_usd(self, v): return str(v) if v is not None else None

    `when_used="always"` means the serializer fires on *every* model_dump()
    call, not just `mode="json"`. This is deliberate (amplifier-core commit
    91fa469, "fix: Decimal JSON serialization for Usage.cost_usd — required
    for M2 cost stamping (#73)") so that Decimal can never leak out through a
    plain `model_dump()` anywhere in the system, not only at explicit JSON
    boundaries. amplifier-core's own test suite asserts this exact contract
    (`tests/test_cost_usd_serialization.py::test_model_dump_returns_string`).

    So: direct attribute access → Decimal (arithmetic-safe).
        model_dump() (plain or JSON mode) → str (JSON-safe), always.

    NOTE: the identical stale assertion (expecting model_dump() to preserve
    Decimal) was also found in microsoft/amplifier-module-provider-anthropic's
    test suite and fails there for the same reason. Out of scope for this fix,
    but flagged for awareness — that repo's test needs the same correction.
    """
    provider = _make_provider()
    fake_coordinator = FakeCoordinator()
    provider.coordinator = cast(ModuleCoordinator, fake_coordinator)
    provider.client.responses.create = AsyncMock(return_value=DummyResponseWithModel())

    result = asyncio.run(provider.complete(_simple_request()))

    assert result.usage is not None, "Precondition: usage must be populated"
    internal_cost = result.usage.cost_usd

    assert internal_cost is not None, (
        "cost_usd must be set internally for a priced model with non-zero tokens"
    )
    assert isinstance(internal_cost, Decimal), (
        f"cost_usd inside Usage must remain Decimal for arithmetic precision, "
        f"got {type(internal_cost).__name__!r}: {internal_cost!r}"
    )
    assert not isinstance(internal_cost, str), (
        f"cost_usd must NOT be str inside Usage; str conversion belongs at the "
        f"emission boundary only. Got {internal_cost!r}"
    )

    # model_dump() always stringifies cost_usd via Usage's field_serializer
    # (when_used="always"), even in plain non-JSON mode — this is intentional
    # so a plain model_dump() can never leak a JSON-unsafe Decimal.
    dumped_cost = result.usage.model_dump().get("cost_usd")
    assert isinstance(dumped_cost, str), (
        f"model_dump()['cost_usd'] must be str (the field_serializer runs with "
        f"when_used='always'), got {type(dumped_cost).__name__!r}: {dumped_cost!r}"
    )
