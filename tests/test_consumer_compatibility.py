"""Compatibility with reasoning selectors in released consumers."""

import amplifier_module_provider_openai as provider
from amplifier_module_provider_openai._capabilities import get_capabilities


def test_existing_agent_reasoning_selector_imports():
    # This is the branch order used by amplifier-agent's reasoning.support.
    efforts = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
    expected = {
        "gpt-5": efforts,
        "gpt-5.5-pro": ("medium", "high", "xhigh"),
        "gpt-6-sol": ("none", "low", "medium", "high", "xhigh", "max"),
        "gpt-6-luna": ("none", "low", "medium", "high", "xhigh", "max"),
        "gpt-6-astra": ("low", "medium", "high", "xhigh", "max"),
        "gpt-6.1-sol": ("low", "medium", "high", "xhigh", "max"),
    }
    for model, supported in expected.items():
        assert get_capabilities(model).supports_reasoning
        allowed = efforts
        if model.startswith("gpt-5.5-pro"):
            allowed = provider._GPT_5_5_PRO_ALLOWED_EFFORTS
        elif model in provider.GPT_6_SOL_LUNA_MODELS:
            allowed = provider._GPT_6_SOL_LUNA_ALLOWED_EFFORTS
        elif model in provider.GPT_6_MODELS:
            allowed = provider._GPT_6_ASTRA_ALLOWED_EFFORTS
        assert tuple(effort for effort in efforts if effort in allowed) == supported
