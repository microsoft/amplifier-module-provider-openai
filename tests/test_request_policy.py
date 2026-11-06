"""Exact-ID documentary eligibility, independent from endpoint qualification."""

from dataclasses import FrozenInstanceError

import pytest

from amplifier_module_provider_openai._capabilities import (
    get_capabilities,
    get_request_policy,
)


@pytest.mark.parametrize("model,none_allowed,new_features", [
    ("gpt-6-astra", False, True),
    ("gpt-6-sol", True, True),
    ("gpt-6-luna", True, True),
    ("gpt-6.1-sol", False, True),
    ("gpt-6.1-terra", True, True),
    ("gpt-5.6", True, False),
    ("gpt-5.6-sol", True, False),
    ("gpt-5.6-terra", True, False),
    ("gpt-5.6-luna", True, False),
])
def test_exact_model_policy(model, none_allowed, new_features):
    policy = get_request_policy(model)
    efforts = {"low", "medium", "high", "xhigh", "max"}
    if none_allowed:
        efforts.add("none")
    assert policy.allowed_efforts == efforts
    assert policy.cache_ttls == {"30m", "12h"}
    assert policy.supports_steering is new_features
    assert policy.supports_async_tools is new_features
    assert policy.supports_configuration_update is new_features
    assert get_capabilities(model).context_window == 1_050_000
    assert get_capabilities(model).max_input_tokens == 922_000
    with pytest.raises(FrozenInstanceError):
        policy.supports_steering = False


@pytest.mark.parametrize("model", [
    "gpt-6", "gpt-6-astra-preview", "gpt-6-sol-2099-01-01",
    "gpt-6.1-luna", "gpt-5.6-cyber", "my-deployment", None, [],
])
def test_unknown_ids_are_not_qualified(model):
    policy = get_request_policy(model)
    assert policy.allowed_efforts is None
    assert not policy.cache_ttls
    assert not policy.supports_steering
    assert not policy.supports_async_tools
    assert not policy.supports_configuration_update


@pytest.mark.parametrize("model,context,input_cap,threshold", [
    ("gpt-5.2", 400_000, None, None),
    ("gpt-5.2-pro", 400_000, None, None),
    ("gpt-5.4-mini", 400_000, 272_000, None),
    ("gpt-5.5", 1_050_000, None, 272_000),
    ("gpt-5.5-pro", 1_050_000, None, 272_000),
])
def test_independent_legacy_model_facts(model, context, input_cap, threshold):
    caps = get_capabilities(model)
    assert caps.context_window == context
    assert caps.max_input_tokens == input_cap
    assert caps.long_context_pricing_threshold == threshold
    assert caps.default_reasoning_effort is None