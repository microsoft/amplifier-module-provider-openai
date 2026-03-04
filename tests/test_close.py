"""Tests for OpenAIProvider.close() method and mount() cleanup bug fix."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from amplifier_module_provider_openai import OpenAIProvider, mount


class TestOpenAIProviderClose:
    """Tests for the async close() method on OpenAIProvider."""

    @pytest.mark.asyncio
    async def test_close_calls_client_close_when_initialized(self):
        """close() should call _client.close() and nil the reference."""
        provider = OpenAIProvider(api_key="fake-key")
        mock_client = MagicMock()
        mock_client.close = AsyncMock()
        provider._client = mock_client

        await provider.close()

        mock_client.close.assert_awaited_once()
        assert provider._client is None

    @pytest.mark.asyncio
    async def test_close_is_safe_when_client_is_none(self):
        """close() should be a no-op when _client is None."""
        provider = OpenAIProvider(api_key="fake-key")
        assert provider._client is None

        await provider.close()  # Should not raise

        assert provider._client is None

    @pytest.mark.asyncio
    async def test_close_can_be_called_twice(self):
        """Calling close() twice should only close the client once."""
        provider = OpenAIProvider(api_key="fake-key")
        mock_client = MagicMock()
        mock_client.close = AsyncMock()
        provider._client = mock_client

        await provider.close()
        await provider.close()

        mock_client.close.assert_awaited_once()
        assert provider._client is None


class TestMountCleanupBugFix:
    """Tests that mount() cleanup does not trigger lazy client initialization."""

    @pytest.mark.asyncio
    async def test_mount_cleanup_does_not_trigger_lazy_init(self):
        """mount() cleanup should use _client (backing field), not .client (property).

        The old code accessed provider.client which is a @property that lazily
        creates an AsyncOpenAI client — cleanup was creating a brand new client
        just to immediately close it.
        """

        # Create a fake coordinator that records mount calls
        class FakeCoordinator:
            def __init__(self):
                self.mounted = []

            async def mount(self, category, instance, name=None):
                self.mounted.append((category, instance, name))

        coordinator = FakeCoordinator()

        with patch.dict("os.environ", {"OPENAI_API_KEY": "fake-key-for-test"}):
            cleanup_ref = await mount(coordinator, config={})  # type: ignore[arg-type]

        assert cleanup_ref is not None

        # Grab the provider that was mounted
        assert len(coordinator.mounted) == 1
        provider = coordinator.mounted[0][1]

        # The provider should not have a client yet (lazy init)
        assert provider._client is None

        # Calling cleanup should NOT create one
        await cleanup_ref()

        # After cleanup, _client should still be None — NOT lazily initialized
        assert provider._client is None, (
            "cleanup() triggered lazy client initialization via .client property"
        )
