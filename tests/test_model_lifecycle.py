"""Lifecycle facts never remove explicit pins or classify deployment aliases."""

import asyncio
from datetime import date
from unittest.mock import patch

import httpx
import pytest
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
                    ("gpt-5.1", {"shutdown_date": "2099-04-01"}),
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
            models = {model.id: model for model in await provider.list_models(
                filters={"hide_shutdown": True},
            )}
            assert set(models) == {"gpt-5.2", "gpt-5.1", "gpt-5-unknown"}
            assert models["gpt-5.1"].defaults["shutdown_date"] == "2099-04-01"
            assert "shutdown_date" not in models["gpt-5.2"].defaults
            # Explicit all is not quietly filtered by upstream lifecycle policy.
            all_models = {model.id: model for model in await provider.list_models(filters={})}
            assert set(all_models) == {
                "gpt-5.2", "gpt-5.1", "gpt-5.2-codex", "o3-deep-research",
                "gpt-5-future", "gpt-5-unknown",
            }
            assert all_models["o3-deep-research"].defaults["shutdown_date"] == "2026-07-23"
    asyncio.run(run())


@pytest.mark.parametrize("model_id", [
    "o3-deep-research", "gpt-image-2", "gpt-image-2-2026-04-21",
])
def test_custom_endpoint_and_subclass_catalog_are_not_openai_lifecycle(model_id, caplog):
    async def run():
        for provider_type in (OpenAIProvider, type("AzureLike", (OpenAIProvider,), {"name": "azure-openai"})):
            for endpoint in ("https://gateway.invalid/v1", "https://api.openai.com/v1"):
                if provider_type is OpenAIProvider and "api.openai.com" in endpoint:
                    continue  # Direct OpenAI is exercised above.
                def handler(request):
                    assert request.method == "GET" and not request.url.query
                    return httpx.Response(200, json={"object": "list", "data": [
                        {"id": model_id, "shutdown_date": "2020-01-01"},
                    ]})
                async with AsyncOpenAI(
                    api_key="test", base_url=endpoint,
                    http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
                ) as client:
                    caplog.clear()
                    provider = provider_type(client=client, config={
                        "max_retries": 0, "default_model": model_id,
                    })
                    assert provider.default_model == model_id
                    assert "OpenAI shutdown_date" not in caplog.text
                    models = await provider.list_models(filters={"hide_shutdown": True})
                    assert [model.id for model in models] == [model_id]
                    assert "shutdown_date" not in models[0].defaults
    asyncio.run(run())


@pytest.mark.parametrize("model_id", ["gpt-image-2", "gpt-image-2-2026-04-21"])
def test_image2_exact_shutdown_dates_and_boundaries(model_id):
    assert shutdown_date(model_id) == "2027-02-10"
    assert not is_shutdown(model_id, today=date(2027, 2, 9))
    assert is_shutdown(model_id, today=date(2027, 2, 10))
    assert shutdown_date(model_id, "2099-01-01") == "2099-01-01"
    assert shutdown_date(model_id, "2027-02-30") == "2027-02-10"
    for unknown in (
        "gpt-image-2-custom", "gpt-image-2-2099-01-01",
        "gpt-image-2.6-sunburst", "gpt-image-2.6-flare",
    ):
        assert shutdown_date(unknown) is None
        assert not is_shutdown(unknown, today=date(2099, 1, 1))


@pytest.mark.asyncio
async def test_image2_catalog_metadata_and_date_filter():
    retiring = {"gpt-image-2", "gpt-image-2-2026-04-21"}
    replacements = {"gpt-image-2.6-sunburst", "gpt-image-2.6-flare"}

    def handler(request):
        assert request.method == "GET" and request.url.path == "/v1/models"
        assert not request.url.query
        return httpx.Response(200, json={"object": "list", "data": [
            {"id": model_id, "object": "model", "created": 0, "owned_by": "openai"}
            for model_id in sorted(retiring | replacements)
        ]})

    real_is_shutdown = is_shutdown
    async with AsyncOpenAI(
        api_key="test", base_url="https://api.openai.com/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    ) as client:
        provider = OpenAIProvider(client=client, config={"max_retries": 0})
        for today, visible in (
            (date(2027, 2, 9), retiring | replacements),
            (date(2027, 2, 10), replacements),
        ):
            with patch(
                "amplifier_module_provider_openai._lifecycle.is_shutdown",
                side_effect=lambda model_id, *, reported=None, today=today: real_is_shutdown(
                    model_id, reported=reported, today=today,
                ),
            ):
                models = {model.id: model for model in await provider.list_models(
                    filters={"hide_shutdown": True, "hide_dated": False},
                )}
                assert set(models) == visible
                for model_id in retiring & visible:
                    assert models[model_id].defaults["shutdown_date"] == "2027-02-10"
                for model_id in replacements:
                    assert "shutdown_date" not in models[model_id].defaults
                all_models = {
                    model.id: model for model in await provider.list_models(filters={})
                }
                assert set(all_models) == retiring | replacements
                for model_id in retiring:
                    assert all_models[model_id].defaults["shutdown_date"] == "2027-02-10"
                assert await provider.list_models() == []
                assert provider.default_model == "gpt-6.2-sol"


@pytest.mark.parametrize("model_id", ["gpt-image-2", "gpt-image-2-2026-04-21"])
def test_image2_explicit_pins_warn_without_replacement(model_id, caplog, monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    provider = OpenAIProvider(api_key="test", config={"default_model": model_id})
    assert provider.default_model == model_id
    assert model_id in caplog.text
    assert "2027-02-10" in caplog.text
    assert "pin is preserved" in caplog.text


@pytest.mark.asyncio
async def test_default_lifecycle_and_date_filters_compose_without_losing_orphan_snapshots():
    payload = {"object": "list", "data": [
        {"id": "gpt-6-sol", "shutdown_date": "2020-01-01"},
        {"id": "gpt-6-sol-2099-01-01"},
        {"id": "gpt-6-new-tier", "shutdown_date": "2099-04-01"},
        {"id": "gpt-6-preview", "shutdown_date": "invalid"},
    ]}
    requests = []

    def handler(request):
        requests.append(request)
        assert request.method == "GET" and not request.url.query
        return httpx.Response(200, json=payload)

    async with AsyncOpenAI(
        api_key="test", base_url="https://api.openai.com/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    ) as client:
        provider = OpenAIProvider(client=client)
        models = {m.id: m for m in await provider.list_models()}
        assert set(models) == {"gpt-6-sol-2099-01-01", "gpt-6-new-tier", "gpt-6-preview"}
        assert models["gpt-6-new-tier"].context_window == 0
        assert models["gpt-6-new-tier"].defaults == {"shutdown_date": "2099-04-01"}
        assert models["gpt-6-preview"].defaults == {}
        assert len(await provider.list_models(filters={})) == 4
        assert len(await provider.list_models(filters={"hide_dated": True})) == 3
        assert len(requests) == 3


@pytest.mark.asyncio
async def test_shutdown_policy_is_frozen_per_config_and_override_not_a_shared_toggle():
    payload = {"object": "list", "data": [
        {"id": "gpt-6-new-tier", "shutdown_date": "2020-01-01"},
    ]}
    async with AsyncOpenAI(
        api_key="test", base_url="https://api.openai.com/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(
            lambda _: httpx.Response(200, json=payload),
        )),
    ) as client:
        config = {"model_filters": {"hide_shutdown": True}}
        provider = OpenAIProvider(client=client, config=config)
        config["model_filters"]["hide_shutdown"] = False
        assert await provider.list_models() == []
        assert len(await provider.list_models(filters={})) == 1
        assert await provider.list_models() == []
        unfiltered = OpenAIProvider(client=client, config={"model_filters": {}})
        assert len(await unfiltered.list_models()) == 1


@pytest.mark.asyncio
async def test_no_missing_model_synthesis_when_direct_catalog_is_empty():
    async with AsyncOpenAI(
        api_key="test",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(
            lambda _: httpx.Response(200, json={"object": "list", "data": []}),
        )),
    ) as client:
        provider = OpenAIProvider(client=client)
        with patch("amplifier_module_provider_openai._lifecycle.shutdown_date") as fact:
            assert await provider.list_models(filters={}) == []
            fact.assert_not_called()


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