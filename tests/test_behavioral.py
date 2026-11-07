"""Behavioral tests for openai provider.

Inherits authoritative tests from amplifier-core.
"""

from unittest.mock import AsyncMock

import pytest
from amplifier_core.validation.behavioral import ProviderBehaviorTests


class TestOpenaiProviderBehavior(ProviderBehaviorTests):
    """Run standard provider behavioral tests for openai.

    All tests from ProviderBehaviorTests run automatically.
    Add module-specific tests below if needed.
    """

    @pytest.mark.asyncio
    async def test_list_models_returns_list(self, provider_module):
        """Exercise the Core no-argument contract with offline catalog IDs."""
        provider_module._fetch_model_catalog = AsyncMock(return_value={"gpt-6.1-sol": None})
        await super().test_list_models_returns_list(provider_module)
