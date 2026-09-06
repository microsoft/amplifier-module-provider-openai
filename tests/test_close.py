"""Tests for OpenAIProvider.close() method and mount() cleanup bug fix."""

import asyncio
import logging
import time
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

    @pytest.mark.asyncio
    async def test_close_is_bounded_when_client_close_never_returns(self, caplog):
        """A client whose close() never returns must not hang cleanup.

        Regression guard: close() previously awaited ``self._client.close()``
        with no ceiling, and mount()'s cleanup() awaits close() directly --
        so a wedged httpx transport hung session cleanup for the whole
        process.
        """
        provider = OpenAIProvider(api_key="test-key", config={"close_timeout": 0.05})
        assert provider.close_timeout == 0.05

        release = asyncio.Event()

        class _UnclosableClient:
            async def close(self):
                # Never returns until the test explicitly releases it.
                await release.wait()

        provider._client = _UnclosableClient()  # type: ignore[assignment]

        started = time.monotonic()
        with caplog.at_level(logging.WARNING):
            await provider.close()  # must not raise, must not hang
        elapsed = time.monotonic() - started

        assert elapsed < 2.0, f"close() took {elapsed:.2f}s; expected ~0.05s"
        assert "did not complete within" in caplog.text
        assert "abandoning client" in caplog.text
        assert "openai" in caplog.text
        # Client reference dropped so the lazy-init property can rebuild.
        assert provider._client is None

        # Let the abandoned close task finish so the loop shuts down clean.
        release.set()
        await asyncio.sleep(0)

    @pytest.mark.asyncio
    async def test_close_normal_client_logs_no_warning(self, caplog):
        """A well-behaved client closes once, quietly, and is released."""
        provider = OpenAIProvider(api_key="test-key")
        mock_client = MagicMock()
        mock_client.close = AsyncMock()
        provider._client = mock_client

        with caplog.at_level(logging.WARNING):
            await provider.close()

        mock_client.close.assert_awaited_once()
        assert caplog.text == ""
        assert provider._client is None

    def test_close_timeout_defaults_to_five_seconds(self):
        """Unconfigured providers get the 5.0s default ceiling."""
        assert OpenAIProvider(api_key="test-key").close_timeout == 5.0

    def test_close_timeout_accepts_string_config(self):
        """settings.yaml / CLI string values coerce to float."""
        provider = OpenAIProvider(api_key="test-key", config={"close_timeout": "2.5"})
        assert provider.close_timeout == 2.5


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

            def register_contributor(self, *args, **kwargs):  # noqa: ARG002
                pass

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
