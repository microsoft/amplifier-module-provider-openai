"""Regression tests: OpenAI cache-write tokens must not double-count input.

Bug: OpenAI's Responses API usage.input_tokens is the RAW GROSS total --
fresh + cache_read + cache_write ALL COMBINED (cache_write is a SUBSET of
it; see amplifier_module_provider_openai._cost's
`fresh_input = prompt_tokens - cached_tokens - cache_write_tokens`
derivation). This is the OPPOSITE of Anthropic, where cache_write
(cache_creation) is a genuinely DISJOINT bucket reported on top of
input_tokens.

The kernel Usage contract (amplifier_core CONTRACTS.md: "input_tokens ...
gross total (fresh + cache_read combined)") -- and every consumer built on
it (e.g. amplifier-module-hooks-streaming-ui's
`total_input = input_tokens + cache_write_tokens`) -- assumes cache_write is
ALWAYS additive on top of input_tokens, matching Anthropic's semantics.

Before this fix, this provider emitted OpenAI's raw input_tokens verbatim,
which ALREADY contained cache_write. The consumer then added cache_write a
second time, inflating the displayed "Input" figure and (as a side effect)
suppressing the displayed cache-hit percentage (cache_read / inflated-total).

Measured on a real turn (gpt-5.6-sol, short context):
    raw usage.input_tokens        = 9,028   (fresh=3, cache_read=0, cache_write=9,025)
    usage.input_tokens_details.cached_tokens        = 0
    usage.input_tokens_details.cache_write_tokens   = 9,025
    usage.output_tokens                             = 7
    measured cost_usd                               = $0.05663125

Before the fix: consumer displayed Input = 9,028 (raw) + 9,025 (cache_write) = 18,053.
After the fix:  provider emits input_tokens = 9,028 - 9,025 = 3 (fresh + cache_read);
                consumer displays Input = 3 + 9,025 = 9,028 (the true gross).
"""

import asyncio
from decimal import Decimal
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

from amplifier_core import ModuleCoordinator
from amplifier_core.message_models import ChatRequest, Message
from amplifier_module_provider_openai import OpenAIProvider
from amplifier_module_provider_openai._response_handling import (
    convert_response_with_accumulated_output,
)

# ---------------------------------------------------------------------------
# Measured numbers (see module docstring)
# ---------------------------------------------------------------------------
_RAW_INPUT_TOKENS = 9_028  # usage.input_tokens as reported by the live API
_CACHE_WRITE_TOKENS = 9_025
_CACHE_READ_TOKENS = 0
_OUTPUT_TOKENS = 7
_MODEL = "gpt-5.6-sol"
_EXPECTED_COST = Decimal("0.05663125")


def _make_provider(**config_overrides) -> OpenAIProvider:
    config = {
        "max_retries": 0,
        "use_streaming": False,
        "default_model": _MODEL,
        **config_overrides,
    }
    return OpenAIProvider(api_key="test-key", config=config)


def _simple_request() -> ChatRequest:
    return ChatRequest(messages=[Message(role="user", content="Hello")])


class _MeasuredUsageResponse:
    """Response stub carrying the exact measured usage numbers, plus model."""

    def __init__(self, model: str = _MODEL):
        self.output = [
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text="Hi there")],
            )
        ]
        self.usage = SimpleNamespace(
            input_tokens=_RAW_INPUT_TOKENS,
            output_tokens=_OUTPUT_TOKENS,
            input_tokens_details=SimpleNamespace(
                cached_tokens=_CACHE_READ_TOKENS,
                cache_write_tokens=_CACHE_WRITE_TOKENS,
            ),
        )
        self.status = "completed"
        self.id = "resp_test"
        self.model = model


# ---------------------------------------------------------------------------
# 1. Non-continuation path (_convert_to_chat_response via provider.complete)
# ---------------------------------------------------------------------------


def test_normal_path_normalizes_input_tokens_excluding_cache_write():
    """Usage.input_tokens must exclude cache_write (fresh + cache_read only)."""
    provider = _make_provider()
    provider.client.responses.create = AsyncMock(return_value=_MeasuredUsageResponse())

    result = asyncio.run(provider.complete(_simple_request()))

    assert result.usage is not None
    # fresh + cache_read = 9028 - 9025 = 3 (NOT the raw 9,028).
    assert result.usage.input_tokens == 3
    assert result.usage.output_tokens == _OUTPUT_TOKENS
    assert result.usage.cache_write_tokens == _CACHE_WRITE_TOKENS
    assert result.usage.cache_read_tokens == _CACHE_READ_TOKENS
    # total_tokens mirrors the normalized input, not the raw vendor total.
    assert result.usage.total_tokens == 3 + _OUTPUT_TOKENS


def test_normal_path_consumer_formula_reconstructs_true_gross():
    """The documented consumer formula (input + cache_write) must equal the
    true measured gross input (9,028), not the inflated 18,053."""
    provider = _make_provider()
    provider.client.responses.create = AsyncMock(return_value=_MeasuredUsageResponse())

    result = asyncio.run(provider.complete(_simple_request()))
    assert result.usage is not None

    total_input = result.usage.input_tokens + (result.usage.cache_write_tokens or 0)
    assert total_input == _RAW_INPUT_TOKENS  # 9,028 -- not 18,053
    # Also equals the raw vendor field directly: proof the normalization is
    # an exact, lossless re-partition (fresh+read vs. write), not a fudge.
    assert total_input == _RAW_INPUT_TOKENS


def test_normal_path_cost_uses_raw_gross_not_normalized_input():
    """compute_cost must be fed the RAW vendor total (9,028), not the
    normalized Usage.input_tokens (3) -- otherwise cost would silently break
    as a second, independent consequence of the same field-meaning change."""
    provider = _make_provider()
    provider.client.responses.create = AsyncMock(return_value=_MeasuredUsageResponse())

    result = asyncio.run(provider.complete(_simple_request()))

    assert result.usage is not None
    assert result.usage.cost_usd == _EXPECTED_COST


# ---------------------------------------------------------------------------
# 2. Continuation / accumulated-output path
# ---------------------------------------------------------------------------


class _AccumulatedFinalResponse:
    """Final response stub for the continuation/accumulated-output path."""

    def __init__(self):
        self.usage = SimpleNamespace(
            input_tokens=_RAW_INPUT_TOKENS,
            output_tokens=_OUTPUT_TOKENS,
            input_tokens_details=SimpleNamespace(
                cached_tokens=_CACHE_READ_TOKENS,
                cache_write_tokens=_CACHE_WRITE_TOKENS,
            ),
        )
        self.id = "resp_continued"
        self.status = "completed"


def test_accumulated_output_path_normalizes_input_tokens():
    """convert_response_with_accumulated_output must apply the same
    fresh+cache_read normalization as the non-continuation path."""
    from amplifier_module_provider_openai import OpenAIChatResponse

    accumulated_output = [
        SimpleNamespace(
            type="message",
            content=[SimpleNamespace(type="output_text", text="Hi")],
        )
    ]
    chat_response = convert_response_with_accumulated_output(
        _AccumulatedFinalResponse(),
        accumulated_output,
        continuation_count=1,
        chat_response_class=OpenAIChatResponse,
    )

    assert chat_response.usage is not None
    assert chat_response.usage.input_tokens == 3
    assert chat_response.usage.output_tokens == _OUTPUT_TOKENS
    assert chat_response.usage.cache_write_tokens == _CACHE_WRITE_TOKENS
    assert chat_response.usage.cache_read_tokens == _CACHE_READ_TOKENS
    assert chat_response.usage.total_tokens == 3 + _OUTPUT_TOKENS

    total_input = chat_response.usage.input_tokens + (
        chat_response.usage.cache_write_tokens or 0
    )
    assert total_input == _RAW_INPUT_TOKENS


# ---------------------------------------------------------------------------
# 3. Cache-hit percentage side effect: no longer suppressed/halved
# ---------------------------------------------------------------------------


def test_cache_hit_percentage_no_longer_suppressed():
    """A turn with genuine cache reuse must report its true cache-hit ratio,
    not one diluted by an inflated (double-counted) denominator.

    Scenario: raw usage.input_tokens=20,000 = fresh(4,000) + cache_read(8,000)
    + cache_write(8,000).

    Before the fix: displayed total_input = 20,000 (raw, already write-inclusive)
    + 8,000 (cache_write added again) = 28,000 -> cache_pct = 8,000/28,000 ~= 28.6%
    (suppressed from the true 40%).

    After the fix: input_tokens = 20,000 - 8,000 = 12,000 (fresh+cache_read);
    consumer total_input = 12,000 + 8,000 = 20,000 (true gross) ->
    cache_pct = 8,000 / 20,000 = 40% (correct).
    """
    provider = _make_provider()

    class _Resp:
        def __init__(self):
            self.output = [
                SimpleNamespace(
                    type="message",
                    content=[SimpleNamespace(type="output_text", text="Hi")],
                )
            ]
            self.usage = SimpleNamespace(
                input_tokens=20_000,
                output_tokens=1,
                input_tokens_details=SimpleNamespace(
                    cached_tokens=8_000,
                    cache_write_tokens=8_000,
                ),
            )
            self.status = "completed"
            self.id = "resp_cache_pct"
            self.model = _MODEL

    provider.client.responses.create = AsyncMock(return_value=_Resp())
    result = asyncio.run(provider.complete(_simple_request()))
    assert result.usage is not None

    # Consumer-side formula (mirrors amplifier-module-hooks-streaming-ui's
    # _compute_total_input): input_tokens + cache_write_tokens.
    total_input = result.usage.input_tokens + (result.usage.cache_write_tokens or 0)
    cache_read = result.usage.cache_read_tokens or 0

    assert total_input == 20_000  # true gross, not 28,000
    cache_pct = cache_read / total_input
    assert cache_pct == 0.4  # 40%, not the suppressed ~28.6%


# ---------------------------------------------------------------------------
# 4. COLD path at the EVENT-PAYLOAD level
# ---------------------------------------------------------------------------
# The tests above assert on ChatResponse.usage (the object). That is what the
# `content_block:end` event carries, via `response.usage.model_dump()` in
# amplifier-module-loop-streaming -- a FULL dump, so cache_write_tokens rides
# along and gross input is reconstructible.
#
# The `llm:response` event is DIFFERENT: the provider hand-builds its `usage`
# sub-dict field by field. That builder listed only input/output/cache_read/
# cost_usd -- it never had a cache_write_tokens line. On a COLD turn (fresh
# session, cache_read=0, huge cache_write) that dropped a real 45,320-token
# cache write on the floor, leaving `input_tokens: 3` with nothing to add it
# back to: gross input was UNRECOVERABLE from that payload.
#
# Measured cold turn (gpt-5.6-sol), reconstructed exactly from its reported
# cost of $0.283445:
#     3 @ $5.00/M (fresh) + X @ $6.25/M (write) + 6 @ $30.00/M (out) = 0.283445
#   solves to X = 45,320 exactly, so raw usage.input_tokens = 45,323.
#
# These tests assert on the ACTUAL EMITTED EVENT, not the Usage object, because
# the object-level tests above passed while this shipped.

_COLD_RAW_INPUT = 45_323
_COLD_CACHE_WRITE = 45_320
_COLD_OUTPUT = 6
_COLD_COST = Decimal("0.283445")


class _RecordingHooks:
    """Hooks stub that records emitted event payloads."""

    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    async def emit(self, name: str, payload: dict) -> None:
        self.events.append((name, payload))

    def payload_for(self, event_name: str) -> dict | None:
        for name, payload in self.events:
            if name == event_name:
                return payload
        return None


class _FakeCoordinator:
    def __init__(self):
        self.hooks = _RecordingHooks()


def _consumer_total_input(usage: dict) -> int:
    """Byte-for-byte mirror of streaming-ui's _compute_total_input()."""
    input_tokens = usage.get("input_tokens") or 0
    cache_create = (
        usage.get("cache_write_tokens") or usage.get("cache_creation_input_tokens") or 0
    )
    return input_tokens + cache_create


def _cold_response():
    return SimpleNamespace(
        output=[
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text="Hi")],
            )
        ],
        usage=SimpleNamespace(
            input_tokens=_COLD_RAW_INPUT,
            output_tokens=_COLD_OUTPUT,
            input_tokens_details=SimpleNamespace(
                cached_tokens=0,  # COLD: nothing served from cache
                cache_write_tokens=_COLD_CACHE_WRITE,  # large real write
            ),
        ),
        status="completed",
        id="resp_cold",
        model=_MODEL,
    )


def _run_and_get_llm_response_usage(response) -> dict:
    provider = _make_provider()
    provider.coordinator = cast(ModuleCoordinator, _FakeCoordinator())
    provider.client.responses.create = AsyncMock(return_value=response)
    asyncio.run(provider.complete(_simple_request()))
    hooks = cast(_FakeCoordinator, provider.coordinator).hooks
    payload = hooks.payload_for("llm:response")
    assert payload is not None, "llm:response was never emitted"
    return payload["usage"]


def test_cold_turn_llm_response_event_carries_cache_write():
    """A large real cache write must NOT serialize away to nothing.

    This is the specific regression: the emitted llm:response usage dict
    previously had exactly four keys and no cache_write_tokens.
    """
    usage = _run_and_get_llm_response_usage(_cold_response())

    assert "cache_write_tokens" in usage, (
        f"cache_write_tokens missing from llm:response usage payload: "
        f"{sorted(usage)} -- a real {_COLD_CACHE_WRITE:,}-token cache write "
        f"serialized away to nothing"
    )
    assert usage["cache_write_tokens"] == _COLD_CACHE_WRITE


def test_cold_turn_gross_input_reconstructible_from_event():
    """Consumer formula on the EVENT payload must yield true gross (45,323),
    not the bare fresh remainder (3)."""
    usage = _run_and_get_llm_response_usage(_cold_response())

    assert _consumer_total_input(usage) == _COLD_RAW_INPUT
    # And the fix must NOT have been made by re-inflating input_tokens --
    # that would reintroduce the original double-count on warm turns.
    assert usage["input_tokens"] == 3


def test_cold_turn_event_cost_and_cache_read_unchanged():
    """cost_usd and cache_read_tokens keep their existing values/semantics."""
    usage = _run_and_get_llm_response_usage(_cold_response())

    assert usage["cost_usd"] == str(_COLD_COST)
    assert usage["cache_read_tokens"] == 0  # measured zero, not absent


def test_warm_turn_gross_input_reconstructible_from_event():
    """The warm counterpart: high cache_read, no new write. Guards against a
    'fix' that only works cold."""
    warm = SimpleNamespace(
        output=[
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text="Hi")],
            )
        ],
        usage=SimpleNamespace(
            input_tokens=45_324,
            output_tokens=7,
            input_tokens_details=SimpleNamespace(
                cached_tokens=44_420,
                cache_write_tokens=0,  # nothing new written
            ),
        ),
        status="completed",
        id="resp_warm",
        model=_MODEL,
    )
    usage = _run_and_get_llm_response_usage(warm)

    # cache_write=0 -> input_tokens is already the full gross; adding 0 is a no-op.
    assert usage["input_tokens"] == 45_324
    assert usage["cache_write_tokens"] == 0
    assert _consumer_total_input(usage) == 45_324
    pct = int((usage["cache_read_tokens"] / _consumer_total_input(usage)) * 100)
    assert pct == 98  # reproduces the observed "(98% cached)" footer exactly


def test_pre_5_6_model_without_cache_write_field_still_reconstructs():
    """Models that never report cache_write (pre-5.6): the key is absent, and
    input_tokens is ALREADY the full gross -- formula still yields the truth."""
    legacy = SimpleNamespace(
        output=[
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text="Hi")],
            )
        ],
        usage=SimpleNamespace(
            input_tokens=1_000,
            output_tokens=10,
            input_tokens_details=SimpleNamespace(cached_tokens=400),
        ),
        status="completed",
        id="resp_legacy",
        model="gpt-5.5",
    )
    usage = _run_and_get_llm_response_usage(legacy)

    # No cache_write measured -> key omitted (mirrors cache_read's None guard).
    assert "cache_write_tokens" not in usage
    assert usage["input_tokens"] == 1_000
    assert _consumer_total_input(usage) == 1_000
