"""Pytest configuration for module tests.

Behavioral tests use inheritance from amplifier-core base classes.
See tests/test_behavioral.py for the inherited tests.

The amplifier-core pytest plugin provides fixtures automatically:
- module_path: Detected path to this module
- module_type: Detected type (provider, tool, hook, etc.)
- provider_module, tool_module, etc.: Mounted module instances
"""

import pytest


@pytest.fixture(autouse=True)
def offline_contract_credentials(request, monkeypatch):
    """Mount real providers for offline inherited contracts, without secrets."""
    if request.node.path.name in {"test_behavioral.py", "test_validation.py"}:
        monkeypatch.setenv("OPENAI_API_KEY", "offline-contract-placeholder")


@pytest.fixture(autouse=True)
def legacy_generation_counter(monkeypatch):
    """Legacy generation-only mocks need an offline count as well.

    These tests replace SDK generation methods on a real client. Previously
    their unmocked count endpoint attempted real HTTP, failed, and silently
    fell back. Dedicated count mocks and real SDK MockTransport tests retain
    their own counting behavior.
    """
    from unittest.mock import Mock
    import inspect
    import openai
    from amplifier_module_provider_openai import OpenAIProvider
    original = OpenAIProvider._native_input_token_count

    async def count(self, params):
        client = self._client
        responses = getattr(client, 'responses', None)
        counter = getattr(getattr(responses, 'input_tokens', None), 'count', None)
        if (isinstance(client, openai.AsyncOpenAI) and self._provider_count_available()
                and callable(counter) and not isinstance(counter, Mock)
                and any(isinstance(method, Mock) or inspect.isfunction(method)
                        for method in (getattr(responses, 'create', None),
                                       getattr(responses, 'stream', None)))):
            return 1
        return await original(self, params)

    monkeypatch.setattr(OpenAIProvider, '_native_input_token_count', count)
