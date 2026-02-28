"""Tests for pricing data, cost fields in ModelInfo, and vision capability.

Validates:
1. _OPENAI_PRICING constant and _openai_pricing_for_model helper
2. list_models() populates cost_per_input_token, cost_per_output_token, metadata
3. GPT-5.x models include 'vision' in capabilities
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from amplifier_module_provider_openai import OpenAIProvider


# --- 1. Pricing helper ---


class TestOpenAIPricingHelper:
    """Verify _openai_pricing_for_model returns correct values."""

    def test_mini_model_pricing(self):
        from amplifier_module_provider_openai import _openai_pricing_for_model

        p = _openai_pricing_for_model("gpt-5-mini", is_deep_research=False)
        assert p["input"] == 0.4e-6
        assert p["output"] == 1.6e-6
        assert p["tier"] == "low"

    def test_nano_model_pricing(self):
        from amplifier_module_provider_openai import _openai_pricing_for_model

        p = _openai_pricing_for_model("gpt-5-nano", is_deep_research=False)
        assert p["tier"] == "low"

    def test_pro_model_pricing(self):
        from amplifier_module_provider_openai import _openai_pricing_for_model

        p = _openai_pricing_for_model("gpt-5-pro", is_deep_research=False)
        assert p["input"] == 15.0e-6
        assert p["output"] == 60.0e-6
        assert p["tier"] == "high"

    def test_default_model_pricing(self):
        from amplifier_module_provider_openai import _openai_pricing_for_model

        p = _openai_pricing_for_model("gpt-5.1", is_deep_research=False)
        assert p["input"] == 2.5e-6
        assert p["output"] == 10.0e-6
        assert p["tier"] == "medium"

    def test_deep_research_pricing(self):
        from amplifier_module_provider_openai import _openai_pricing_for_model

        p = _openai_pricing_for_model("o3-deep-research", is_deep_research=True)
        assert p["input"] is None
        assert p["output"] is None
        assert p["tier"] == "extreme"


# --- 2. list_models() cost fields ---


class TestListModelsCostFields:
    """Verify list_models() populates cost and metadata on ModelInfo."""

    @pytest.fixture
    def provider(self):
        return OpenAIProvider(api_key="test-key", config={"filtered": True})

    def _make_mock_model(self, model_id: str):
        m = MagicMock()
        m.id = model_id
        return m

    @pytest.mark.asyncio
    async def test_gpt5_model_has_cost_fields(self, provider):
        mock_response = MagicMock()
        mock_response.data = [self._make_mock_model("gpt-5.1")]

        provider._client = MagicMock()
        provider._client.models.list = AsyncMock(return_value=mock_response)

        models = await provider.list_models()

        assert len(models) >= 1
        model = models[0]
        assert model.cost_per_input_token == 2.5e-6
        assert model.cost_per_output_token == 10.0e-6
        assert model.metadata == {"cost_tier": "medium"}

    @pytest.mark.asyncio
    async def test_mini_model_has_low_cost_tier(self, provider):
        mock_response = MagicMock()
        mock_response.data = [self._make_mock_model("gpt-5-mini")]

        provider._client = MagicMock()
        provider._client.models.list = AsyncMock(return_value=mock_response)

        models = await provider.list_models()

        assert len(models) >= 1
        model = models[0]
        assert model.metadata["cost_tier"] == "low"

    @pytest.mark.asyncio
    async def test_deep_research_model_has_extreme_tier(self, provider):
        mock_response = MagicMock()
        mock_response.data = [self._make_mock_model("o3-deep-research")]

        provider._client = MagicMock()
        provider._client.models.list = AsyncMock(return_value=mock_response)

        models = await provider.list_models()

        assert len(models) >= 1
        model = models[0]
        assert model.metadata["cost_tier"] == "extreme"
        assert model.cost_per_input_token is None
        assert model.cost_per_output_token is None


# --- 3. Vision capability ---


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
