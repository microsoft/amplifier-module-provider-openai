"""Lifecycle facts never remove explicit pins or classify deployment aliases."""

import asyncio
from datetime import date
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
from amplifier_core.message_models import ChatRequest, Message
from openai import AsyncOpenAI

from amplifier_module_provider_openai import OpenAIProvider
from amplifier_module_provider_openai._lifecycle import is_shutdown, shutdown_date


def test_exact_shutdown_dates_and_negative_controls():
    assert shutdown_date("o3-deep-research") == "2026-07-23"
    assert shutdown_date("gpt-5.2-chat-latest") == "2026-08-10"
    assert shutdown_date("gpt-5.1") == "2027-04-01"
    assert shutdown_date("gpt-5.1-my-deployment") is None
    assert shutdown_date("gpt-6.1-terra") is None
    assert is_shutdown("gpt-5.1", today=date(2027, 4, 1))
    assert not is_shutdown("gpt-5.1", today=date(2027, 3, 31))
    assert not is_shutdown("unknown", today=date(2099, 1, 1))
    assert shutdown_date("gpt-5.1", "2030-01-01") == "2030-01-01"
    for malformed in ("2028-02-30", "tomorrow", 123, None):
        assert shutdown_date("unknown", malformed) is None


def test_real_sdk_catalog_preserves_shutdown_field_and_drops_only_known_retirements():
    async def run():
        def handler(request):
            assert request.url.path == "/v1/models"
            return httpx.Response(200, json={"object": "list", "data": [
                {"id": model, "object": "model", "created": 0, "owned_by": "openai",
                 **extra}
                for model, extra in (
                    ("gpt-5.2", {}),
                    ("gpt-5.1", {"shutdown_date": "2027-04-01"}),
                    ("gpt-5.2-codex", {}),
                    ("o3-deep-research", {}),
                    ("gpt-5-future", {"shutdown_date": "2020-01-01"}),
                    ("gpt-5-unknown", {"shutdown_date": "invalid"}),
                )
            ]})
        async with AsyncOpenAI(
            api_key="test", base_url="https://api.openai.com/v1",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        ) as client:
            provider = OpenAIProvider(api_key="test", client=client, config={"max_retries": 0})
            models = {model.id: model for model in await provider.list_models()}
            assert set(models) == {"gpt-5.2", "gpt-5.1", "gpt-5-unknown"}
            assert models["gpt-5.1"].defaults["shutdown_date"] == "2027-04-01"
            assert "shutdown_date" not in models["gpt-5.2"].defaults
    asyncio.run(run())


def test_custom_endpoint_and_subclass_catalog_are_not_openai_lifecycle():
    async def run():
        for provider_type in (OpenAIProvider, type("AzureLike", (OpenAIProvider,), {"name": "azure-openai"})):
            client = SimpleNamespace(
                base_url="https://gateway.invalid/v1",
                models=SimpleNamespace(list=AsyncMock(return_value=SimpleNamespace(
                    data=[SimpleNamespace(id="o3-deep-research", shutdown_date="2020-01-01")]
                ))),
            )
            provider = provider_type(api_key="test", client=client, config={"max_retries": 0})
            models = await provider.list_models()
            assert models[0].id == "o3-deep-research"
            assert "shutdown_date" not in models[0].defaults
    asyncio.run(run())


def test_retired_pin_warns_but_no_silent_replacement_or_automatic_background(caplog):
    provider = OpenAIProvider(api_key="test", config={"default_model": "o3-deep-research"})
    assert provider.default_model == "o3-deep-research"
    assert "pin is preserved" in caplog.text
    request = ChatRequest(messages=[Message(role="user", content="test")])
    params = provider._budget_params(request)
    assert params["model"] == "o3-deep-research"
    assert not params["store"]
    assert "background" not in params
    explicit = provider._budget_params(request, background=True)
    assert explicit["background"] is True and explicit["store"] is True


def test_compatible_endpoint_retains_legacy_background_identity():
    provider = OpenAIProvider(api_key="test", config={
        "default_model": "o3-deep-research", "base_url": "https://gateway.invalid/v1",
    })
    params = provider._budget_params(ChatRequest(messages=[Message(role="user", content="test")]))
    assert params["background"] is True and params["store"] is True