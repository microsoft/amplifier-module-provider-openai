"""Test for the config-surface V2 wizard reduction: exactly 4 ConfigFields,
with the exact prompt strings specified by the task.
"""

import pytest

from amplifier_module_provider_openai import OpenAIProvider


def _make_provider(**config_overrides) -> OpenAIProvider:
    config = {"max_retries": 0, "use_streaming": False, **config_overrides}
    return OpenAIProvider(api_key="test-key", config=config)


@pytest.mark.parametrize(
    "config",
    [{}, {"auto_continue": False}, {"auto_continue": "false"}],
    ids=["default", "disabled", "legacy-disabled"],
)
def test_wizard_surface_has_exactly_four_fields_with_exact_prompts(config):
    provider = _make_provider(**config)
    assert provider._client is None
    info = provider.get_info()
    assert provider._client is None
    fields_by_id = {f.id: f for f in info.config_fields}

    assert len(info.config_fields) == 4
    assert set(fields_by_id) == {
        "api_key",
        "base_url",
        "reasoning_effort",
        "enable_long_context",
    }, f"Expected exactly 4 ConfigFields; got {sorted(fields_by_id)}"

    assert "auto_continue" not in fields_by_id
    assert all(
        f.display_name != "Continue truncated responses"
        and f.prompt != "Automatically continue responses that reach the output limit"
        for f in info.config_fields
    )

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
