"""Offline exact-ID policy checks; eligibility is not live qualification."""

from dataclasses import FrozenInstanceError
from decimal import Decimal
from types import SimpleNamespace

import pytest
from amplifier_core import llm_errors as errors
from amplifier_core.message_models import ChatRequest, Message

from amplifier_module_provider_openai import OpenAIProvider
from amplifier_module_provider_openai._capabilities import (
    get_capabilities,
    get_request_policy,
)

# Deliberately independent of the production registry: adding a production key
# must not silently expand the expected matrix.
MODERN_MODELS = (
    "gpt-6-astra",
    "gpt-6-sol",
    "gpt-6-luna",
    "gpt-6.1-sol",
    "gpt-6.1-terra",
    "gpt-5.6",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-5.6-luna",
)
POLICY_CASES = [
    ("gpt-6-astra", {"low", "medium", "high", "xhigh", "max"}, {"30m", "12h"}, True),
    ("gpt-6-sol", {"none", "low", "medium", "high", "xhigh", "max"}, {"30m", "12h"}, True),
    ("gpt-6-luna", {"none", "low", "medium", "high", "xhigh", "max"}, {"30m", "12h"}, True),
    ("gpt-6.1-sol", {"low", "medium", "high", "xhigh", "max"}, {"30m", "12h"}, True),
    ("gpt-6.1-terra", {"none", "low", "medium", "high", "xhigh", "max"}, {"30m", "12h"}, True),
    ("gpt-5.6", {"none", "low", "medium", "high", "xhigh", "max"}, {"30m", "12h"}, False),
    ("gpt-5.6-sol", {"none", "low", "medium", "high", "xhigh", "max"}, {"30m", "12h"}, False),
    ("gpt-5.6-terra", {"none", "low", "medium", "high", "xhigh", "max"}, {"30m", "12h"}, False),
    ("gpt-5.6-luna", {"none", "low", "medium", "high", "xhigh", "max"}, {"30m", "12h"}, False),
    ("gpt-5.5", {"none", "low", "medium", "high", "xhigh"}, set(), False),
    ("gpt-5.5-pro", {"medium", "high", "xhigh"}, set(), False),
    ("gpt-5.4", {"none", "low", "medium", "high", "xhigh"}, set(), False),
    ("gpt-5.4-mini", {"none", "low", "medium", "high", "xhigh"}, set(), False),
    ("gpt-5.2", {"none", "low", "medium", "high", "xhigh"}, set(), False),
    ("gpt-5.2-pro", {"medium", "high", "xhigh"}, set(), False),
]


def _provider(model, **config):
    return OpenAIProvider(
        api_key="offline-fixture",
        config={
            "default_model": model,
            "base_url": "https://offline.invalid/v1",
            "use_streaming": False,
            "max_retries": 0,
            **config,
        },
    )


def _request(**kwargs):
    return ChatRequest(messages=[Message(role="user", content="Hello")], **kwargs)


@pytest.mark.parametrize("model,efforts,ttls,native_features", POLICY_CASES)
def test_exact_documented_policy_matrix(model, efforts, ttls, native_features):
    policy = get_request_policy(model)
    assert policy.allowed_efforts == frozenset(efforts)
    assert policy.cache_ttls == frozenset(ttls)
    assert policy.supports_configuration_update is native_features
    assert policy.supports_steering is native_features
    assert policy.supports_async_tools is native_features


@pytest.mark.parametrize("model", MODERN_MODELS)
def test_request_policy_and_its_sets_are_immutable(model):
    policy = get_request_policy(model)
    with pytest.raises(FrozenInstanceError):
        policy.supports_steering = False
    with pytest.raises(AttributeError):
        policy.allowed_efforts.add("minimal")
    with pytest.raises(AttributeError):
        policy.cache_ttls.add("24h")


@pytest.mark.parametrize("model", [
    "",
    "gpt-6",
    "gpt-6.1",
    "gpt-6-terra",
    "gpt-6.1-luna",
    "gpt-6.1-terra-pro",
    "gpt-6.2-sol",
    "gpt-5.6-sol-preview",
    "gpt-5.7",
    "gpt-5.4-nano",
    "GPT-6-SOL",
    "gpt-6-sol ",
    *[f"{model}-2099-01-01" for model, *_ in POLICY_CASES],
])
def test_near_misses_and_invented_snapshots_have_no_new_request_eligibility(model):
    policy = get_request_policy(model)
    assert policy.allowed_efforts is None
    assert policy.cache_ttls == frozenset()
    assert not policy.supports_configuration_update
    assert not policy.supports_steering
    assert not policy.supports_async_tools


def test_legacy_snapshot_capability_reader_does_not_qualify_request_features():
    snapshot = "gpt-5.6-sol-2099-01-01"
    assert get_capabilities(snapshot).max_input_tokens == 922_000
    assert get_request_policy(snapshot).cache_ttls == frozenset()
    assert get_request_policy(snapshot).allowed_efforts is None


@pytest.mark.parametrize("model,efforts,_ttls,_native", POLICY_CASES)
@pytest.mark.parametrize("effort", [
    "none", "minimal", "low", "medium", "high", "xhigh", "max", "invented",
])
def test_final_request_effort_uses_exact_model_matrix(model, efforts, _ttls, _native, effort):
    provider = _provider(model)
    request = _request(reasoning_effort=effort)
    if effort in efforts:
        assert provider._budget_params(request)["reasoning"]["effort"] == effort
    else:
        with pytest.raises(errors.InvalidRequestError, match=effort):
            provider._budget_params(request)


@pytest.mark.parametrize("model", MODERN_MODELS)
@pytest.mark.parametrize("source", ["config", "kwargs", "extras"])
@pytest.mark.parametrize("ttl", ["30m", "12h"])
def test_modern_cache_ttls_survive_final_assembly_without_legacy_retention(model, source, ttl):
    config, kwargs = {"prompt_cache_retention": "24h"}, {}
    if source == "config":
        config["prompt_cache_options"] = {"ttl": ttl}
    elif source == "kwargs":
        kwargs["prompt_cache_options"] = {"ttl": ttl}
    else:
        config["extra_request_params"] = {"prompt_cache_options": {"ttl": ttl}}
    params = _provider(model, **config)._budget_params(_request(), **kwargs)
    assert params["prompt_cache_options"] == {"ttl": ttl}
    assert "prompt_cache_retention" not in params


@pytest.mark.parametrize("model", MODERN_MODELS)
@pytest.mark.parametrize("ttl", ["24h", "in_memory", "1h", "", None, 30])
def test_modern_cache_ttl_is_rechecked_after_final_extra_merge(model, ttl):
    provider = _provider(
        model,
        prompt_cache_options={"ttl": "30m"},
        extra_request_params={"prompt_cache_options": {"ttl": ttl}},
    )
    with pytest.raises(errors.InvalidRequestError, match="ttl"):
        provider._budget_params(_request())


@pytest.mark.parametrize("configured,final,effort,valid", [
    ("gpt-6.1-terra", "gpt-6.1-sol", "none", False),
    ("gpt-6.1-sol", "gpt-6.1-terra", "none", True),
    ("gpt-6-sol", "gpt-5.2-pro", "low", False),
    ("gpt-5.2-pro", "gpt-6-sol", "low", True),
    ("gpt-5.5-pro", "gpt-6-sol", "low", True),
    ("gpt-6-sol", "gpt-5.5-pro", "low", False),
])
def test_final_extra_model_controls_effort_policy(configured, final, effort, valid):
    provider = _provider(
        configured,
        extra_request_params={"model": final, "reasoning": {"effort": effort}},
    )
    if valid:
        params = provider._budget_params(_request())
        assert params["model"] == final
        assert params["reasoning"]["effort"] == effort
    else:
        with pytest.raises(errors.InvalidRequestError, match=effort):
            provider._budget_params(_request())


@pytest.mark.parametrize("model", ["gpt-5.2", "gpt-5.2-pro"])
def test_gpt_5_2_default_omits_effort_but_explicit_effort_survives(model):
    provider = _provider(model)
    assert get_capabilities(model).context_window == 400_000
    assert get_capabilities(model).default_reasoning_effort is None
    assert "reasoning" not in provider._budget_params(_request())
    assert provider._budget_params(_request(reasoning_effort="high"))["reasoning"]["effort"] == "high"


# Fresh 100 input + 10 output, no reads/writes, Standard tier. These literal
# numerical oracles do not consult or derive from the production price tables.
ZERO_WRITE_COSTS = [
    ("gpt-6-astra", Decimal(".0015")),
    ("gpt-6-sol", Decimal(".0003")),
    ("gpt-6-luna", Decimal(".000015")),
    ("gpt-6.1-sol", Decimal(".0003")),
    ("gpt-6.1-terra", Decimal(".00015")),
    ("gpt-5.6", Decimal(".0006")),
    ("gpt-5.6-sol", Decimal(".0006")),
    ("gpt-5.6-terra", Decimal(".00032")),
    ("gpt-5.6-luna", Decimal(".000032")),
]


def _response(model, details):
    return SimpleNamespace(
        model=model,
        service_tier="default",
        output=[],
        usage=SimpleNamespace(
            input_tokens=100,
            output_tokens=10,
            input_tokens_details=SimpleNamespace(**details),
        ),
    )


@pytest.mark.parametrize("model,expected", ZERO_WRITE_COSTS)
def test_explicit_zero_cache_counts_are_priceable_without_inventing_lifetimes(model, expected):
    response = _response(model, {"cached_tokens": 0, "cache_write_tokens": 0})
    assert _provider(model)._convert_to_chat_response(response).usage.cost_usd == expected


@pytest.mark.parametrize("model", MODERN_MODELS)
@pytest.mark.parametrize("field", ["cached_tokens", "cache_write_tokens"])
@pytest.mark.parametrize("bad", ["missing", None, -1, True, "0"])
def test_missing_or_invalid_cache_counts_are_not_assumed_zero(model, field, bad):
    details = {"cached_tokens": 0, "cache_write_tokens": 0}
    if bad == "missing":
        details.pop(field)
    else:
        details[field] = bad
    response = _response(model, details)
    assert _provider(model)._compute_attempt_cost(response) is None


@pytest.mark.parametrize("model", MODERN_MODELS)
def test_nonzero_writes_require_reported_lifetime_not_requested_ttl(model):
    provider = _provider(model, prompt_cache_options={"ttl": "30m"})
    response = _response(model, {"cached_tokens": 0, "cache_write_tokens": 50})
    assert provider._compute_attempt_cost(response) is None


@pytest.mark.parametrize("model", [
    *MODERN_MODELS, "gpt-5.6-sol-2026-07-09",
])
@pytest.mark.parametrize("field", ["input_tokens", "output_tokens"])
@pytest.mark.parametrize("bad", [None, False, -1, "0"])
def test_missing_raw_totals_never_become_a_zero_cost(model, field, bad):
    response = _response(model, {"cached_tokens": 0, "cache_write_tokens": 0})
    setattr(response.usage, field, bad)
    assert _provider(model)._compute_attempt_cost(response) is None


def test_historical_snapshot_pricing_requires_cache_count_evidence():
    model = "gpt-5.6-sol-2026-07-09"
    assert get_request_policy(model).allowed_efforts is None
    response = _response(model, {"cached_tokens": 0})
    assert _provider(model)._compute_attempt_cost(response) is None