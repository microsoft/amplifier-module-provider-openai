"""Tests for vision capability on GPT-5.x models."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from amplifier_module_provider_openai import OpenAIProvider


class TestVisionCapability:
    """Verify GPT-5.x models include 'vision' in capabilities."""

    @pytest.fixture
    def provider(self):
        return OpenAIProvider(api_key="test-key", config={"filtered": True})

    def _make_mock_model(self, model_id: str):
        m = MagicMock()
        m.id = model_id
        return m

    @pytest.mark.asyncio
    async def test_gpt5_has_vision(self, provider):
        mock_response = MagicMock()
        mock_response.data = [self._make_mock_model("gpt-5.1")]

        provider._client = MagicMock()
        provider._client.models.list = AsyncMock(return_value=mock_response)

        models = await provider.list_models()
        model = models[0]
        assert "vision" in model.capabilities

    @pytest.mark.asyncio
    async def test_deep_research_no_vision(self, provider):
        """Deep research models don't get vision capability."""
        mock_response = MagicMock()
        mock_response.data = [self._make_mock_model("o3-deep-research")]

        provider._client = MagicMock()
        provider._client.models.list = AsyncMock(return_value=mock_response)

        models = await provider.list_models()
        model = models[0]
        assert "vision" not in model.capabilities

    @pytest.mark.asyncio
    async def test_vision_coexists_with_other_caps(self, provider):
        mock_response = MagicMock()
        mock_response.data = [self._make_mock_model("gpt-5.2")]

        provider._client = MagicMock()
        provider._client.models.list = AsyncMock(return_value=mock_response)

        models = await provider.list_models()
        model = models[0]
        assert "tools" in model.capabilities
        assert "reasoning" in model.capabilities
        assert "streaming" in model.capabilities
        assert "vision" in model.capabilities
