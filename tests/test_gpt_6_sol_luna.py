"""Offline regression coverage for the exact GPT-6 Sol/Luna integration."""

from __future__ import annotations

import asyncio
import logging
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from amplifier_core import llm_errors as kernel_errors
from amplifier_core.message_models import ChatRequest, Message

from amplifier_module_provider_openai import OpenAIProvider
from amplifier_module_provider_openai._capabilities import get_capabilities
from amplifier_module_provider_openai._cost import compute_cost


def _provider(**config: object) -> OpenAIProvider:
    return OpenAIProvider(
        api_key="[REDACTED:SECRET]",
        config={
            "default_model": "gpt-6-sol",
            "max_retries": 0,
            "use_streaming": False,
            "reasoning_summary": "auto",
            **config,
        },
    )


def _request() -> ChatRequest:
    return ChatRequest(messages=[Message(role="user", content="Hello")])


def _response(
    *,
    model: str = "gpt-6-sol",
    status: str = "completed",
    service_tier: str | None = "default",
    input_tokens: int = 100,
    output_tokens: int = 10,
) -> SimpleNamespace:
    return SimpleNamespace(
        id="resp_sol_luna",
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
            input_tokens_details=SimpleNamespace(cached_tokens=0, cache_write_tokens=0),
        ),
    )


@pytest.mark.parametrize("model", ["gpt-6-sol", "gpt-6-luna"])
def test_sol_luna_have_exact_documented_capabilities(model: str) -> None:
    caps = get_capabilities(model)

    assert caps.family == model
    assert caps.context_window == 922_000
    assert caps.max_input_tokens == 922_000
    assert caps.max_output_tokens == 128_000
    assert caps.default_reasoning_effort is None
    assert caps.long_context_pricing_threshold == 272_000
    assert caps.supports_reasoning is True
    assert caps.supports_native_apply_patch is True
    assert caps.supports_native_computer_use is True


@pytest.mark.parametrize(
    "model",
    [
        "gpt-6",
        "gpt-6-terra",
        "gpt-6-sol-2099-01-01",
        "gpt-6-luna-2099-01-01",
    ],
)
def test_only_documented_exact_gpt_6_ids_receive_gpt_6_capabilities(model: str) -> None:
    assert get_capabilities(model).family not in {
        "gpt-6-astra",
        "gpt-6-sol",
        "gpt-6-luna",
    }


@pytest.mark.parametrize("model", ["gpt-6-sol", "gpt-6-luna"])
def test_config_none_is_an_omission_sentinel_but_nested_none_is_literal(model: str) -> None:
    sentinel = _provider(default_model=model, reasoning_effort="none")
    nested = _provider(default_model=model, reasoning={"effort": "none"})
    sentinel.client.responses.create = AsyncMock(return_value=_response(model=model))
    nested.client.responses.create = AsyncMock(return_value=_response(model=model))

    asyncio.run(sentinel.complete(_request()))
    asyncio.run(nested.complete(_request()))

    sentinel_params = sentinel.client.responses.create.call_args.kwargs
    nested_params = nested.client.responses.create.call_args.kwargs
    assert "reasoning" not in sentinel_params
    assert sentinel_params["include"] == ["reasoning.encrypted_content"]
    assert nested_params["reasoning"] == {"effort": "none", "summary": "auto"}
    assert "include" not in nested_params


@pytest.mark.parametrize("model", ["gpt-6-sol", "gpt-6-luna"])
@pytest.mark.parametrize("effort", ["none", "low", "medium", "high", "xhigh", "max"])
def test_sol_luna_accept_all_documented_request_reasoning_efforts(
    model: str, effort: str
) -> None:
    provider = _provider(default_model=model)
    provider.client.responses.create = AsyncMock(return_value=_response(model=model))
    request = ChatRequest(
        messages=[Message(role="user", content="Hello")], reasoning_effort=effort
    )

    asyncio.run(provider.complete(request))

    assert provider.client.responses.create.call_args.kwargs["reasoning"] == {
        "effort": effort,
        "summary": "auto",
    }


@pytest.mark.parametrize("model", ["gpt-6-sol", "gpt-6-luna"])
def test_literal_none_allows_documented_responses_sampling_and_removes_reasoning_include(
    model: str,
) -> None:
    provider = _provider(
        default_model=model,
        extra_request_params={
            "top_p": 0.5,
            "top_logprobs": 3,
            "include": [
                "reasoning.encrypted_content",
                "message.output_text.logprobs",
            ],
        },
    )
    provider.client.responses.create = AsyncMock(return_value=_response(model=model))

    asyncio.run(
        provider.complete(
            _request(),
            reasoning={"effort": "none"},
            temperature=0.2,
        )
    )

    params = provider.client.responses.create.call_args.kwargs
    assert params["reasoning"] == {"effort": "none", "summary": "auto"}
    assert params["temperature"] == 0.2
    assert params["top_p"] == 0.5
    assert params["top_logprobs"] == 3
    assert params["include"] == ["message.output_text.logprobs"]


@pytest.mark.parametrize("model", ["gpt-6-sol", "gpt-6-luna"])
def test_literal_none_still_rejects_top_level_responses_logprobs(model: str) -> None:
    provider = _provider(default_model=model, extra_request_params={"logprobs": True})
    provider.client.responses.create = AsyncMock(return_value=_response(model=model))

    with pytest.raises(kernel_errors.InvalidRequestError, match="logprobs"):
        asyncio.run(provider.complete(_request(), reasoning={"effort": "none"}))

    provider.client.responses.create.assert_not_awaited()


@pytest.mark.parametrize("model", ["gpt-6-sol", "gpt-6-luna"])
@pytest.mark.parametrize("source", ["config", "request", "extras"])
def test_sol_luna_reject_minimal_from_every_final_parameter_source(
    model: str, source: str
) -> None:
    config: dict[str, object] = {"default_model": model}
    request = _request()
    if source == "config":
        config["reasoning_effort"] = "minimal"
    elif source == "request":
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")], reasoning_effort="minimal"
        )
    else:
        config["extra_request_params"] = {"reasoning": {"effort": "minimal"}}
    provider = _provider(**config)
    provider.client.responses.create = AsyncMock(return_value=_response(model=model))

    with pytest.raises(kernel_errors.InvalidRequestError, match="minimal"):
        asyncio.run(provider.complete(request))

    provider.client.responses.create.assert_not_awaited()


@pytest.mark.parametrize("model", ["gpt-6-sol", "gpt-6-luna"])
@pytest.mark.parametrize(
    ("ttl", "valid"),
    [("30m", True), ("24h", False)],
)
def test_sol_luna_final_extra_cache_ttl_is_validated(model: str, ttl: str, valid: bool) -> None:
    provider = _provider(
        default_model=model,
        extra_request_params={"prompt_cache_options": {"ttl": ttl}},
    )
    provider.client.responses.create = AsyncMock(return_value=_response(model=model))

    if valid:
        asyncio.run(provider.complete(_request()))
        assert provider.client.responses.create.call_args.kwargs["prompt_cache_options"] == {
            "ttl": "30m"
        }
    else:
        with pytest.raises(kernel_errors.InvalidRequestError, match="30m"):
            asyncio.run(provider.complete(_request()))
        provider.client.responses.create.assert_not_awaited()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"extra_request_params": {"temperature": 0}},
        {"extra_request_params": {"top_p": 0}},
        {"extra_request_params": {"top_logprobs": 1}},
        {"extra_request_params": {"include": ["message.output_text.logprobs"]}},
    ],
)
def test_sol_default_reasoning_rejects_incompatible_sampling(kwargs: dict[str, object]) -> None:
    provider = _provider(**kwargs)
    provider.client.responses.create = AsyncMock(return_value=_response())

    with pytest.raises(kernel_errors.InvalidRequestError):
        asyncio.run(provider.complete(_request()))

    provider.client.responses.create.assert_not_awaited()


@pytest.mark.parametrize("model", ["gpt-6-sol", "gpt-6-luna"])
def test_sol_luna_legacy_retention_is_cleaned_for_primary_and_continuation(
    model: str, caplog
) -> None:
    caplog.set_level(logging.WARNING, logger="amplifier_module_provider_openai")
    provider = _provider(default_model=model, prompt_cache_retention="24h")
    provider.client.responses.create = AsyncMock(
        side_effect=[_response(model=model, status="incomplete"), _response(model=model)]
    )

    asyncio.run(provider.complete(_request()))

    assert provider.client.responses.create.await_count == 2
    assert all(
        "prompt_cache_retention" not in call.kwargs
        for call in provider.client.responses.create.call_args_list
    )
    assert len(
        [
            record
            for record in caplog.records
            if "Dropping prompt_cache_retention" in record.message
        ]
    ) == 1


@pytest.mark.parametrize("model", ["gpt-6-sol", "gpt-6-luna"])
def test_sol_luna_implicit_retention_is_omitted_without_warning(model: str, caplog) -> None:
    caplog.set_level(logging.WARNING, logger="amplifier_module_provider_openai")
    provider = _provider(default_model=model)
    provider.client.responses.create = AsyncMock(return_value=_response(model=model))

    asyncio.run(provider.complete(_request()))

    assert "prompt_cache_retention" not in provider.client.responses.create.call_args.kwargs
    assert not [
        record for record in caplog.records if "prompt_cache_retention" in record.message
    ]


def test_sol_luna_discovery_has_exact_models_and_default_long_context_budget() -> None:
    provider = _provider(hide_dated_models=False)
    provider._client = AsyncMock()
    provider._client.models.list = AsyncMock(
        return_value=SimpleNamespace(
            data=[
                SimpleNamespace(id="gpt-6-sol"),
                SimpleNamespace(id="gpt-6-luna"),
                SimpleNamespace(id="gpt-6-terra"),
                SimpleNamespace(id="gpt-6-sol-2099-01-01"),
            ]
        )
    )

    models = asyncio.run(provider.list_models())

    assert [(model.id, model.display_name) for model in models] == [
        ("gpt-6-luna", "GPT 6 Luna"),
        ("gpt-6-sol", "GPT 6 Sol"),
    ]
    assert all(model.context_window == 272_000 for model in models)


@pytest.mark.parametrize("model", ["gpt-6-sol", "gpt-6-luna"])
@pytest.mark.parametrize(
    ("enable_long_context", "expected_context"),
    [(False, 272_000), (True, 922_000)],
)
def test_sol_luna_per_call_model_override_uses_exact_context_budget(
    model: str, enable_long_context: bool, expected_context: int
) -> None:
    provider = _provider(default_model="gpt-5.4", enable_long_context=enable_long_context)
    provider.client.responses.create = AsyncMock(return_value=_response(model=model))

    asyncio.run(provider.complete(_request(), model=model))

    assert provider.client.responses.create.call_args.kwargs["model"] == model
    provider.client.models.list = AsyncMock(
        return_value=SimpleNamespace(data=[SimpleNamespace(id=model)])
    )
    models = asyncio.run(provider.list_models())
    assert models[0].context_window == expected_context


@pytest.mark.parametrize(
    ("model", "tier", "expected"),
    [
        ("gpt-6-sol", "default", Decimal(4)),
        ("gpt-6-sol", "flex", Decimal(2)),
        ("gpt-6-sol", "priority", Decimal(8)),
        ("gpt-6-sol", "fast", Decimal(8)),
        ("gpt-6-luna", "default", Decimal("0.20")),
        ("gpt-6-luna", "flex", Decimal("0.10")),
        ("gpt-6-luna", "priority", Decimal("0.40")),
        ("gpt-6-luna", "fast", Decimal("0.40")),
    ],
)
def test_sol_luna_pricing_uses_returned_service_tier(
    model: str, tier: str, expected: Decimal
) -> None:
    assert compute_cost(model, prompt_tokens=1_000_000, service_tier=tier) == expected


@pytest.mark.parametrize(
    ("model", "short", "long"),
    [
        ("gpt-6-sol", Decimal("0.544"), Decimal("1.088004")),
        ("gpt-6-luna", Decimal("0.0272"), Decimal("0.0544002")),
    ],
)
def test_sol_luna_long_context_boundary_is_strict(
    model: str, short: Decimal, long: Decimal
) -> None:
    assert compute_cost(model, prompt_tokens=272_000) == short
    assert compute_cost(model, prompt_tokens=272_001) == long


def test_sol_mixed_cache_tokens_are_billed_once_and_unknown_costs_stay_unknown() -> None:
    assert compute_cost(
        "gpt-6-sol",
        prompt_tokens=1_000,
        completion_tokens=100,
        cached_tokens=200,
        cache_write_tokens=300,
        service_tier="default",
    ) == Decimal("0.00279")
    assert compute_cost("gpt-6-sol", prompt_tokens=1, service_tier=None) is None
    assert compute_cost("gpt-6-luna", prompt_tokens=1, service_tier="scale") is None
    assert compute_cost("gpt-6-sol-2099-01-01", prompt_tokens=1) is None


def test_compute_attempt_cost_uses_raw_responses_cache_usage() -> None:
    provider = _provider()
    response = _response()
    response.usage = SimpleNamespace(
        input_tokens=1_000,
        output_tokens=100,
        input_tokens_details=SimpleNamespace(cached_tokens=200, cache_write_tokens=300),
    )

    converted = provider._convert_to_chat_response(response)

    assert converted.usage.input_tokens == 700
    assert converted.usage.cache_read_tokens == 200
    assert converted.usage.cache_write_tokens == 300
    assert converted.usage.cost_usd == Decimal("0.00279")


@pytest.mark.parametrize(
    ("tiers", "expected_input_tokens", "expected_cost"),
    [
        # Long first attempt: (272001 * $4 + 10 * $15) / 1M.
        (("default",), 272_001, Decimal("1.088154")),
        # Short Flex continuation adds (100 * $2 + 10 * $10) / 1M * 0.5.
        (("default", "flex"), 272_101, Decimal("1.088304")),
        (("default", None), 272_101, None),
    ],
    ids=["single_attempt_known", "continuation_known", "continuation_unknown"],
)
def test_complete_uses_attempt_cost_for_usage_response_event_and_accumulator(
    tiers: tuple[str | None, ...],
    expected_input_tokens: int,
    expected_cost: Decimal | None,
) -> None:
    accumulated_costs: list[Decimal] = []
    coordinator = SimpleNamespace(
        hooks=SimpleNamespace(emit=AsyncMock()),
        get_capability=lambda _: None,
    )
    provider = _provider()
    provider._add_cost = accumulated_costs.append
    provider.coordinator = coordinator
    responses = [
        _response(
            status="incomplete" if index < len(tiers) - 1 else "completed",
            service_tier=tier,
            input_tokens=272_001 if index == 0 else 100,
        )
        for index, tier in enumerate(tiers)
    ]
    provider.client.responses.create = AsyncMock(
        side_effect=responses
    )

    result = asyncio.run(provider.complete(_request()))

    assert result.usage.input_tokens == expected_input_tokens
    assert result.usage.cost_usd == expected_cost
    assert accumulated_costs == ([expected_cost] if expected_cost is not None else [])
    response_events = [
        call.args[1]
        for call in coordinator.hooks.emit.await_args_list
        if call.args[0] == "llm:response"
    ]
    assert response_events[-1]["usage"]["cost_usd"] == (
        str(expected_cost) if expected_cost is not None else None
    )


class _CustomCostProvider(OpenAIProvider):
    def __init__(self, *args, **kwargs) -> None:
        self.cost_calls = 0
        super().__init__(*args, **kwargs)

    def _compute_attempt_cost(self, response: object) -> Decimal | None:
        self.cost_calls += 1
        return Decimal(7)


def test_subclass_cost_seam_controls_conversion_and_single_aggregation() -> None:
    reported_costs: list[Decimal] = []
    provider = _CustomCostProvider(
        api_key="[REDACTED:SECRET]",
        add_cost=reported_costs.append,
        config={
            "default_model": "gpt-6-sol",
            "max_retries": 0,
            "use_streaming": False,
        },
    )
    assert provider._convert_to_chat_response(_response()).usage.cost_usd == Decimal(7)
    provider.client.responses.create = AsyncMock(
        side_effect=[
            _response(status="incomplete"),
            _response(),
        ]
    )

    result = asyncio.run(provider.complete(_request()))

    assert result.usage.cost_usd == Decimal(14)
    assert reported_costs == [Decimal(14)]
    # Direct base conversion and every billed continuation attempt use the same
    # override; session accumulation occurs only after the attempts complete.
    assert provider.cost_calls == 3
