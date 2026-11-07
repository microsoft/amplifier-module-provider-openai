"""Known GPT-5/deep-research metadata remains available with explicit filters."""

from unittest.mock import AsyncMock

import pytest

from amplifier_module_provider_openai import OpenAIProvider


@pytest.mark.asyncio
@pytest.mark.parametrize("model, vision", [
    ("gpt-5.1", True), ("o3-deep-research", False), ("gpt-5.2", True),
])
async def test_explicit_listing_preserves_known_vision_capabilities(model, vision):
    provider = OpenAIProvider(api_key="test-key")
    provider._fetch_model_catalog = AsyncMock(return_value={model: None})
    assert await provider.list_models() == []
    rows = await provider.list_models(filters={})
    assert len(rows) == 1
    assert ("vision" in rows[0].capabilities) is vision
    if vision:
        assert {"tools", "reasoning", "streaming"} <= set(rows[0].capabilities)
