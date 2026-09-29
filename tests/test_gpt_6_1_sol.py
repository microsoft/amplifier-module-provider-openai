"""Offline regression coverage for the exact ``gpt-6.1-sol`` integration."""

from __future__ import annotations

import asyncio
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from amplifier_core import llm_errors as kernel_errors
from amplifier_core.message_models import ChatRequest, Message

from amplifier_module_provider_openai import OpenAIProvider
from amplifier_module_provider_openai._capabilities import get_capabilities
from amplifier_module_provider_openai._cost import compute_cost

MODEL = "gpt-6.1-sol"


def _provider(**config: object) -> OpenAIProvider:
    return OpenAIProvider(
        api_key="[REDACTED:SECRET]",
        config={
            "default_model": MODEL,
            # Disable native token-count HTTP calls before mocked generation.
            "base_url": "https://offline.invalid/v1",
            "max_retries": 0,
            "use_streaming": False,
            "reasoning_summary": "auto",
            **config,
        },
    )


def test_offline_fixture_disables_native_count_network_probe() -> None:
    assert _provider()._provider_count_available() is False


def _request(*, reasoning_effort: str | None = None) -> ChatRequest:
    return ChatRequest(
        messages=[Message(role="user", content="Hello")],
        reasoning_effort=reasoning_effort,
    )


def _response(
    *,
    model: str = MODEL,
    service_tier: str | None = "default",
    input_tokens: int = 100,
    output_tokens: int = 10,
    status: str = "completed",
) -> SimpleNamespace:
    return SimpleNamespace(
        id="resp_gpt_6_1_sol",
        model=model,
        status=status,
        service_tier=service_tier,
        output=[
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text="Hi")],
            )
        ],
        incomplete_details=None,
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_tokens_details=SimpleNamespace(
                cached_tokens=0,
                cache_write_tokens=0,
            ),
        ),
    )


def test_exact_gpt_6_1_sol_capabilities_and_exact_id_boundary() -> None:
    caps = get_capabilities(MODEL)

    assert caps.family == MODEL
    assert caps.context_window == 922_000
    assert caps.max_input_tokens == 922_000
    assert caps.max_output_tokens == 128_000
    assert caps.long_context_pricing_threshold == 272_000
    assert caps.supports_reasoning is True
    assert caps.supports_vision is True
    assert caps.supports_streaming is True
    assert caps.supports_native_apply_patch is True
    assert caps.supports_native_computer_use is True
    for unrecognized in (
        "gpt-6.1",
        "gpt-6.1-terra",
        "gpt-6.1-luna",
        "gpt-6.1-sol-2099-01-01",
    ):
        assert get_capabilities(unrecognized).family != MODEL


@pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max"])
def test_gpt_6_1_sol_passes_every_valid_reasoning_effort(effort: str) -> None:
    provider = _provider()
    provider.client.responses.create = AsyncMock(return_value=_response())

    asyncio.run(provider.complete(_request(reasoning_effort=effort)))

    assert provider.client.responses.create.call_args.kwargs["reasoning"] == {
        "effort": effort,
        "summary": "auto",
    }


@pytest.mark.parametrize("reasoning_effort", [None, "none"])
def test_gpt_6_1_sol_default_and_config_none_omit_reasoning(
    reasoning_effort: str | None,
) -> None:
    config = {} if reasoning_effort is None else {"reasoning_effort": reasoning_effort}
    provider = _provider(**config)
    provider.client.responses.create = AsyncMock(return_value=_response())

    asyncio.run(provider.complete(_request()))

    params = provider.client.responses.create.call_args.kwargs
    assert "reasoning" not in params
    assert params["include"] == ["reasoning.encrypted_content"]


@pytest.mark.parametrize("effort", ["none", "minimal"])
@pytest.mark.parametrize("source", ["request", "nested_config", "extras"])
def test_gpt_6_1_sol_rejects_literal_none_and_minimal_after_final_merge(
    effort: str, source: str
) -> None:
    config: dict[str, object] = {}
    request = _request()
    if source == "request":
        request = _request(reasoning_effort=effort)
    elif source == "nested_config":
        config["reasoning"] = {"effort": effort}
    else:
        config["extra_request_params"] = {"reasoning": {"effort": effort}}
    provider = _provider(**config)
    provider.client.responses.create = AsyncMock(return_value=_response())

    with pytest.raises(kernel_errors.InvalidRequestError, match=effort):
        asyncio.run(provider.complete(request))

    provider.client.responses.create.assert_not_awaited()


@pytest.mark.parametrize(
    ("request_effort", "extra"),
    [
        (None, {"temperature": 0}),
        ("low", {"top_p": 0.5}),
        ("high", {"include": ["message.output_text.logprobs"]}),
    ],
)
def test_gpt_6_1_sol_rejects_sampling_under_default_or_active_reasoning(
    request_effort: str | None, extra: dict[str, object]
) -> None:
    provider = _provider(extra_request_params=extra)
    provider.client.responses.create = AsyncMock(return_value=_response())

    with pytest.raises(kernel_errors.InvalidRequestError):
        asyncio.run(provider.complete(_request(reasoning_effort=request_effort)))

    provider.client.responses.create.assert_not_awaited()


@pytest.mark.parametrize("model", ["gpt-6-sol", "gpt-6-luna"])
def test_existing_gpt_6_sol_luna_literal_none_remains_legal(model: str) -> None:
    provider = _provider(default_model=model)
    provider.client.responses.create = AsyncMock(return_value=_response(model=model))

    asyncio.run(provider.complete(_request(reasoning_effort="none")))

    assert provider.client.responses.create.call_args.kwargs["reasoning"] == {
        "effort": "none",
        "summary": "auto",
    }


def test_gpt_6_1_sol_caching_uses_only_30m_and_omits_legacy_retention() -> None:
    provider = _provider(
        prompt_cache_retention="24h",
        prompt_cache_options={"ttl": "30m"},
    )
    provider.client.responses.create = AsyncMock(return_value=_response())

    asyncio.run(provider.complete(_request()))

    params = provider.client.responses.create.call_args.kwargs
    assert "prompt_cache_retention" not in params
    assert params["prompt_cache_options"] == {"ttl": "30m"}

    invalid = _provider(prompt_cache_options={"ttl": "24h"})
    invalid.client.responses.create = AsyncMock(return_value=_response())
    with pytest.raises(kernel_errors.InvalidRequestError, match="30m"):
        asyncio.run(invalid.complete(_request()))
    invalid.client.responses.create.assert_not_awaited()


def test_gpt_6_1_sol_discovery_has_display_name_and_opt_in_long_context() -> None:
    default = _provider(hide_dated_models=False)
    default._client = AsyncMock()
    default._client.models.list = AsyncMock(
        return_value=SimpleNamespace(
            data=[
                SimpleNamespace(id=MODEL),
                SimpleNamespace(id="gpt-6.1-sol-2099-01-01"),
                SimpleNamespace(id="gpt-6.1-luna"),
            ]
        )
    )

    models = asyncio.run(default.list_models())

    assert [
        (model.id, model.display_name, model.context_window) for model in models
    ] == [(MODEL, "GPT 6.1 Sol", 272_000)]
    assert default.get_info().defaults["context_window"] == 272_000
    assert (
        _provider(enable_long_context=True).get_info().defaults["context_window"]
        == 922_000
    )


def test_gpt_6_1_sol_cost_uses_independent_short_and_long_cache_oracles() -> None:
    # Fresh/cached/cache-write/output rates are $2/$0.10/$2.50/$10 short and
    # $4/$0.20/$5/$15 long per million tokens.
    assert compute_cost(
        MODEL,
        prompt_tokens=272_000,
        cached_tokens=100_000,
        cache_write_tokens=50_000,
        completion_tokens=20_000,
    ) == Decimal("0.579")
    assert compute_cost(
        MODEL,
        prompt_tokens=272_001,
        cached_tokens=100_000,
        cache_write_tokens=50_000,
        completion_tokens=20_000,
    ) == Decimal("1.058004")


@pytest.mark.parametrize(
    ("tier", "expected"),
    [
        ("default", Decimal(4)),
        ("flex", Decimal(2)),
        ("fast", Decimal(8)),
        ("priority", Decimal(8)),
        ("unknown", None),
        (None, None),
    ],
)
def test_gpt_6_1_sol_cost_uses_response_service_tier(
    tier: str | None, expected: Decimal | None
) -> None:
    assert compute_cost(MODEL, prompt_tokens=1_000_000, service_tier=tier) == expected
    assert compute_cost("gpt-6.1-sol-2099-01-01", prompt_tokens=1) is None


def test_gpt_6_1_sol_complete_uses_actual_attempt_cost() -> None:
    provider = _provider()
    provider.client.responses.create = AsyncMock(
        return_value=_response(
            service_tier="flex",
            input_tokens=272_001,
            output_tokens=10,
        )
    )

    response = asyncio.run(provider.complete(_request()))

    # ((272001 * $4) + (10 * $15)) / 1M, then the Flex 0.5 multiplier.
    assert response.usage.cost_usd == Decimal("0.544077")


def test_gpt_6_1_sol_continuation_accumulates_per_attempt_tiers() -> None:
    reported_costs: list[Decimal] = []
    provider = _provider()
    provider._add_cost = reported_costs.append
    provider.client.responses.create = AsyncMock(
        side_effect=[
            _response(status="incomplete", service_tier="flex", input_tokens=272_001),
            _response(service_tier="default", input_tokens=100),
        ]
    )

    result = asyncio.run(provider.complete(_request()))

    # Long Flex $0.544077, then short Standard (100 * $2 + 10 * $10) / 1M.
    assert result.usage.cost_usd == Decimal("0.544377")
    assert reported_costs == [Decimal("0.544377")]
    assert provider.client.responses.create.await_count == 2
    assert all(
        call.kwargs["model"] == MODEL
        for call in provider.client.responses.create.await_args_list
    )
