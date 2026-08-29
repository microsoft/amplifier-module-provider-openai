"""Test for the config-surface V2 wizard reduction: exactly 4 ConfigFields,
with the exact prompt strings specified by the task.
"""

from amplifier_module_provider_openai import OpenAIProvider


def _make_provider(**config_overrides) -> OpenAIProvider:
    config = {"max_retries": 0, "use_streaming": False, **config_overrides}
    return OpenAIProvider(api_key="test-key", config=config)


def test_wizard_surface_has_exactly_four_fields_with_exact_prompts():
    provider = _make_provider()
    info = provider.get_info()
    fields_by_id = {f.id: f for f in info.config_fields}

    assert set(fields_by_id) == {
        "api_key",
        "base_url",
        "reasoning_effort",
        "enable_long_context",
    }, f"Expected exactly 4 ConfigFields; got {sorted(fields_by_id)}"

    assert fields_by_id["api_key"].prompt == "Enter your OpenAI API key"
    assert fields_by_id["base_url"].prompt == "API base URL"
    assert (
        fields_by_id["reasoning_effort"].prompt
        == "Reasoning effort — higher is smarter, slower, costlier"
    )
    assert (
        fields_by_id["enable_long_context"].prompt
        == "Allow requests over 272K input tokens (≈2× cost)"
    )
