"""Behavioral tests for openai provider.

Inherits authoritative tests from amplifier-core.
"""

import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from amplifier_core.validation.behavioral import ProviderBehaviorTests


class TestOpenaiProviderBehavior(ProviderBehaviorTests):
    """Run standard provider behavioral tests for openai.

    All tests from ProviderBehaviorTests run automatically.
    Add module-specific tests below if needed.
    """

    @pytest.mark.asyncio
    async def test_list_models_returns_list(self, provider_module):
        """Exercise actual catalog mapping against the SDK boundary offline."""
        provider_module.client.models.list = AsyncMock(return_value=SimpleNamespace(
            data=[SimpleNamespace(id="gpt-6.1-sol")]
        ))
        await super().test_list_models_returns_list(provider_module)
