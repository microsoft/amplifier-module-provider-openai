"""Behavioral tests for openai provider.

Inherits authoritative tests from amplifier-core.
"""

import pytest

from amplifier_core.validation.behavioral import ProviderBehaviorTests


class TestOpenaiProviderBehavior(ProviderBehaviorTests):
    """Run standard provider behavioral tests for openai.

    All tests from ProviderBehaviorTests run automatically.
    Add module-specific tests below if needed.
    """

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_list_models_returns_list(self, provider_module):
        """Override to mark this inherited test 'live'.

        ProviderBehaviorTests.test_list_models_returns_list calls
        provider_module.list_models(), which makes a real call to
        OpenAI's /models endpoint -- it cannot pass in CI without a
        genuine OPENAI_API_KEY. Deselected in CI via `-m "not live"`;
        run locally with real credentials to validate.
        """
        await super().test_list_models_returns_list(provider_module)
