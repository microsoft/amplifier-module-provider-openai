"""Canonical listing through the real SDK and offline HTTP MockTransport."""

import asyncio
import inspect
import json
from contextlib import asynccontextmanager
from copy import deepcopy

import httpx
import pytest
from amplifier_core import ModelInfo
from amplifier_core import llm_errors as kernel_errors
from openai import AsyncOpenAI

from amplifier_module_provider_openai import OpenAIProvider
from amplifier_module_provider_openai._capabilities import (
    ModelCapabilities,
    RequestPolicy,
    get_capabilities,
    get_known_context_window,
    get_request_policy,
)
from amplifier_module_provider_openai._cost import compute_cost

ENDPOINT = "https://catalog.example.invalid/account/v1/"


def catalog(*ids, **extra):
    return {"object": "list", "data": [
        {"id": mid, "object": "model", "created": 1, "owned_by": "fixture"}
        for mid in ids
    ], **extra}


def ids(rows):
    assert all(isinstance(row, ModelInfo) for row in rows)
    return {row.id for row in rows}


@asynccontextmanager
async def sdk_provider(receive, **config):
    requests = []

    def transport(request):
        requests.append(request)
        assert request.method == "GET"
        assert str(request.url) == ENDPOINT + "models"
        assert not request.url.query and not request.content
        assert request.headers["authorization"] == "Bearer fixture-discovery-key"
        assert request.headers["openai-organization"] == "fixture-org"
        assert request.headers["openai-project"] == "fixture-project"
        return receive(request)

    http = httpx.AsyncClient(transport=httpx.MockTransport(transport), trust_env=False)
    async with AsyncOpenAI(
        api_key="fixture-discovery-key", base_url=ENDPOINT,
        organization="fixture-org", project="fixture-project",
        http_client=http, max_retries=2,
    ) as client:
        provider = OpenAIProvider(client=client, config={"max_retries": 0, **config})
        yield provider, requests, client
        assert provider.client is client and client.max_retries == 2
        assert not http.is_closed  # Caller owns its transport.
    assert http.is_closed


def test_chat_latest_exact_facts_without_new_request_policy():
    assert get_capabilities("chat-latest") == ModelCapabilities(
        family="chat-latest", context_window=400_000, max_input_tokens=272_000,
        max_output_tokens=128_000, capability_tags=("tools", "streaming", "json_mode", "vision"),
    )
    assert get_known_context_window("chat-latest") == 400_000
    assert get_request_policy("chat-latest") == RequestPolicy()


@pytest.mark.parametrize("model", [
    "chat-latest-preview", "chat-latest-2099-01-01", "Chat-Latest", " chat-latest ",
])
def test_chat_latest_near_misses_remain_unknown(model):
    assert get_capabilities(model) == ModelCapabilities(family="unknown")
    assert get_known_context_window(model) is None
    assert get_request_policy(model) == RequestPolicy()


@pytest.mark.asyncio
@pytest.mark.parametrize("long_context", [False, True])
async def test_chat_latest_catalog_requires_explicit_selector_and_endpoint_presence(long_context):
    returned = ["chat-latest", "chat-latest-preview", "chat-latest-2099-01-01", "Chat-Latest",
                "gpt-6.1-sol", "gpt-5.4"]
    async with sdk_provider(
        lambda _: httpx.Response(200, json=catalog(*returned)),
        enable_long_context=long_context,
    ) as (p, requests, _):
        before = deepcopy(p.config)
        assert ids(await p.list_models()) == {"gpt-6.1-sol"}
        for selector in ({}, {"include": ["chat-latest"]}):
            rows = await p.list_models(filters=selector)
            row = next(row for row in rows if row.id == "chat-latest")
            assert row.context_window == 272_000
            assert row.max_output_tokens == 128_000
            assert row.capabilities == ["tools", "streaming", "json_mode", "vision"]
            # Existing provider omission sentinel, not an API effort qualification.
            assert row.defaults == {"max_tokens": 16384, "reasoning_effort": "none"}
            for unknown in (r for r in rows if r.id in returned[1:4]):
                assert unknown.context_window == unknown.max_output_tokens == 0
                assert unknown.capabilities == [] and unknown.defaults == {}
        assert ids(await p.list_models()) == {"gpt-6.1-sol"}
        assert p.default_model == "gpt-6.1-sol" and p.config == before
        assert len(requests) == 4
    async with sdk_provider(
        lambda _: httpx.Response(200, json=catalog("gpt-5.4")),
        default_model="gpt-5.4",
    ) as (p, _, _):
        assert await p.list_models(filters={"include": ["chat-latest"]}) == []
        assert ids(await p.list_models(filters={})) == {"gpt-5.4"}
        assert p.default_model == "gpt-5.4"


@pytest.mark.parametrize("model", [
    "chat-latest", "chat-latest-preview", "chat-latest-2099-01-01",
])
def test_chat_latest_cost_remains_unknown_in_response_json(model):
    from types import SimpleNamespace

    from openai.types.responses import Response

    assert compute_cost(model, prompt_tokens=100, completion_tokens=10) is None
    response = Response.model_construct(
        id="resp_offline_chat", model=model, status="completed", output=[],
        usage=SimpleNamespace(
            input_tokens=100, output_tokens=10,
            input_tokens_details=SimpleNamespace(cached_tokens=0, cache_write_tokens=0),
        ),
    )
    provider = OpenAIProvider(api_key="offline-fixture", config={"default_model": "chat-latest"})
    converted = provider._convert_to_chat_response(response)
    assert response.model == model  # Never relabel the observed SKU with the requested alias.
    assert converted.usage.cost_usd is None
    assert json.loads(converted.model_dump_json())["usage"]["cost_usd"] is None


@pytest.mark.asyncio
async def test_default_gpt6_pattern_future_previews_and_only_endpoint_ids():
    returned = ["gpt-6-astra", "gpt-6.1-sol", "gpt-6-new-tier", "gpt-6-preview",
                "gpt-6-astra-2026-10-01", "gpt-6-orphan-2026-10-01",
                "gpt-5.6", "o3-mini", "o3-deep-research", "gpt-image-1"]
    async with sdk_provider(lambda _: httpx.Response(200, json=catalog(*returned))) as (p, req, _):
        rows = await p.list_models()
        assert ids(rows) == {"gpt-6-astra", "gpt-6.1-sol", "gpt-6-new-tier",
                             "gpt-6-preview", "gpt-6-orphan-2026-10-01"}
        assert rows == sorted(rows, key=lambda r: r.display_name.lower())
        for row in rows:
            if row.id not in {"gpt-6-astra", "gpt-6.1-sol"}:
                assert row.context_window == row.max_output_tokens == 0
                assert row.capabilities == [] and row.defaults == {}
        assert "preview" in next(r.display_name for r in rows if r.id == "gpt-6-preview")
        assert len(req) == 1


@pytest.mark.asyncio
async def test_explicit_all_bypasses_config_family_date_without_synthesis():
    returned = ["gpt-4o", "gpt-5.6", "gpt-6-sol", "gpt-6-sol-2026-10-01",
                "o3-mini", "gpt-image-1", "text-embedding-3-small", "ft:custom"]
    async with sdk_provider(
        lambda _: httpx.Response(200, json=catalog(*returned)),
        model_filters={"include": [], "hide_dated": True}, default_model="not-in-catalog",
    ) as (p, req, _):
        assert await p.list_models() == []
        assert ids(await p.list_models(filters={})) == set(returned)
        assert ids(await p.list_models(filters={"include": ["*"], "hide_dated": False})) == set(returned)
        assert len(req) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("selector, expected", [
    ({"include": ["gpt-5*", "o3-*"], "exclude": ["*-deep-research"]}, {"gpt-5.6", "o3-mini"}),
    ({"include": []}, set()),
    ({"exclude": ["gpt-*"]}, {"o3-mini", "o3-deep-research"}),
    ({"include": ["GPT-6*"]}, {"gpt-6-solx"}),
    ({"include": ["gpt-6-????"]}, {"gpt-6-solx"}),
])
async def test_custom_include_exclude_replace_defaults(selector, expected):
    async with sdk_provider(lambda _: httpx.Response(200, json=catalog(
        "gpt-5.6", "gpt-6-solx", "o3-mini", "o3-deep-research",
    ))) as (p, _, _):
        assert ids(await p.list_models(filters=selector)) == expected


@pytest.mark.asyncio
async def test_case_insensitive_globs_preserve_exact_endpoint_bytes():
    async with sdk_provider(lambda _: httpx.Response(200, json=catalog(
        "GpT-6-NeW-Tier", "GPT-6-PREVIEW", "Gpt-5.6",
    ))) as (p, _, _):
        assert ids(await p.list_models()) == {"GpT-6-NeW-Tier", "GPT-6-PREVIEW"}
        assert ids(await p.list_models(filters={
            "include": ["gPt-6*"], "exclude": ["*preview"],
        })) == {"GpT-6-NeW-Tier"}


@pytest.mark.asyncio
@pytest.mark.parametrize("selector, expected", [
    ({"hide_dated": True}, {"gpt-6-sol", "gpt-6-only-2026-10-01", "gpt-6-preview"}),
    ({"hide_dated": True, "exclude": ["gpt-6-sol"]},
     {"gpt-6-sol-2026-10-01", "gpt-6-only-2026-10-01", "gpt-6-preview"}),
    ({"include": ["*-2026-10-01"], "hide_dated": True},
     {"gpt-6-sol-2026-10-01", "gpt-6-only-2026-10-01"}),
])
async def test_snapshots_hidden_only_when_corresponding_alias_selected(selector, expected):
    async with sdk_provider(lambda _: httpx.Response(200, json=catalog(
        "gpt-6-sol", "gpt-6-sol-2026-10-01", "gpt-6-only-2026-10-01", "gpt-6-preview",
    ))) as (p, _, _):
        assert ids(await p.list_models(filters=selector)) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("config, hidden", [
    ({}, True), ({"hide_dated_models": False}, False),
    ({"filtered": False}, False), ({"filtered": "false"}, False),
    ({"hide_dated_models": True, "filtered": False}, True),
    ({"hide_dated_models": False, "model_filters": {"include": ["gpt-6*"], "hide_dated": True}}, True),
])
async def test_legacy_date_settings_and_precedence(config, hidden):
    async with sdk_provider(lambda _: httpx.Response(200, json=catalog(
        "gpt-6-sol", "gpt-6-sol-2026-10-01", "gpt-5.6",
    )), **config) as (p, _, _):
        expected = {"gpt-6-sol"} if hidden else {"gpt-6-sol", "gpt-6-sol-2026-10-01"}
        assert ids(await p.list_models()) == expected
        assert "gpt-5.6" in ids(await p.list_models(filters={}))


@pytest.mark.asyncio
async def test_config_and_capability_readback_detached_snapshots():
    filters = {"include": ["gpt-5*"], "exclude": [], "hide_dated": False}
    async with sdk_provider(lambda _: httpx.Response(200, json=catalog("gpt-5.6", "gpt-6-sol")),
                            model_filters=filters) as (p, _, _):
        filters["include"].append("*")
        p.config["model_filters"]["exclude"].append("*")
        p.hide_dated_models = True
        info = p.get_model_list_info()
        info["default_filters"]["include"].clear()
        info["filters_schema"]["properties"].clear()
        assert ids(await p.list_models()) == {"gpt-5.6"}
        assert p.get_model_list_info()["default_filters"]["include"] == ["gpt-5*"]
        assert p.get_model_list_info()["filters_schema"]["properties"]


@pytest.mark.asyncio
async def test_concurrent_overrides_snapshot_before_await_never_mutate_shared_state():
    started, release = asyncio.Event(), asyncio.Event()

    async def transport(request):
        assert request.method == "GET" and not request.url.query
        started.set()
        await release.wait()
        return httpx.Response(200, json=catalog("gpt-5.6", "gpt-6-sol"))

    async with AsyncOpenAI(api_key="fixture", http_client=httpx.AsyncClient(
        transport=httpx.MockTransport(transport),
    )) as client:
        p = OpenAIProvider(client=client)
        before = deepcopy(p.config)
        custom = {"include": ["gpt-5*"]}
        first = asyncio.create_task(p.list_models(filters=custom))
        await started.wait()
        custom["include"][:] = ["*"]
        second = asyncio.create_task(p.list_models())
        release.set()
        assert ids(await first) == {"gpt-5.6"}
        assert ids(await second) == {"gpt-6-sol"}
        assert p.config == before


@pytest.mark.asyncio
@pytest.mark.parametrize("filters", [
    False, "all", [], {"include": "*"}, {"include": [None]}, {"include": [""]},
    {"exclude": [1]}, {"hide_dated": "false"}, {"hide_dated": None}, {"typo": True},
    {"hide_shutdown": "false"}, {"hide_shutdown": None},
])
async def test_invalid_filters_fail_before_io_and_config_fails_at_construction(filters):
    with pytest.raises((ValueError, TypeError)):
        OpenAIProvider(config={"model_filters": filters})
    async with sdk_provider(lambda _: pytest.fail("invalid selector reached transport")) as (p, req, _):
        with pytest.raises((ValueError, TypeError)):
            await p.list_models(filters=filters)
        assert req == []


def test_optional_selector_advertisement_no_parallel_discovery_surface():
    p = OpenAIProvider()
    assert not hasattr(p, "discover_models")
    signature = inspect.signature(p.list_models)
    assert signature.parameters["filters"].kind == inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["filters"].default is None
    info = p.get_model_list_info()
    assert info["version"] == 1 and info["filters_keyword"] == "filters"
    assert info["source"] == "provider-api"
    assert info["unfiltered"] == {} and info["override"] == "replace"
    assert info["default_filters"] == {
        "include": ["gpt-6*"], "exclude": [], "hide_dated": True, "hide_shutdown": True,
    }
    assert p._client is None


@pytest.mark.asyncio
async def test_bound_account_absence_and_headers_survive_environment_changes(monkeypatch):
    for name in ("OPENAI_ORG_ID", "OPENAI_PROJECT_ID", "OPENAI_ADMIN_KEY", "OPENAI_CUSTOM_HEADERS"):
        monkeypatch.delenv(name, raising=False)
    requests = []

    def transport(request):
        requests.append(request)
        assert request.method == "GET" and str(request.url) == ENDPOINT + "models"
        assert p.client is client and client.max_retries == 2
        assert client.organization is None and client.project is None
        assert request.headers["authorization"] == "Bearer fixture-discovery-key"
        assert request.headers["x-discovery-fixture"] == "bound-header"
        assert "openai-organization" not in request.headers
        assert "openai-project" not in request.headers
        return httpx.Response(200, json=catalog("gpt-6-new-tier"))

    http = httpx.AsyncClient(transport=httpx.MockTransport(transport), trust_env=False)
    async with AsyncOpenAI(
        api_key="fixture-discovery-key", base_url=ENDPOINT, organization=None, project=None,
        default_headers={"X-Discovery-Fixture": "bound-header"}, http_client=http, max_retries=2,
    ) as client:
        headers_before = {k: v for k, v in client.default_headers.items() if isinstance(v, str)}
        p = OpenAIProvider(client=client, config={"max_retries": 0})
        monkeypatch.setenv("OPENAI_ORG_ID", "fixture-later-org")
        monkeypatch.setenv("OPENAI_PROJECT_ID", "fixture-later-project")
        assert ids(await p.list_models()) == {"gpt-6-new-tier"}
        assert ids(await p.list_models(filters={})) == {"gpt-6-new-tier"}
        assert client.api_key == "fixture-discovery-key" and str(client.base_url) == ENDPOINT
        assert {k: v for k, v in client.default_headers.items() if isinstance(v, str)} == headers_before
        assert not http.is_closed
    assert http.is_closed and len(requests) == 2


@pytest.mark.asyncio
async def test_admin_only_credentials_rejected_before_transport(monkeypatch):
    for name in ("OPENAI_API_KEY", "OPENAI_ADMIN_KEY", "OPENAI_CUSTOM_HEADERS", "OPENAI_ORG_ID", "OPENAI_PROJECT_ID"):
        monkeypatch.delenv(name, raising=False)
    requests = []

    def transport(request):
        requests.append(request)
        return httpx.Response(200, json=catalog("gpt-6-admin-fallback-must-not-be-used"))

    http = httpx.AsyncClient(transport=httpx.MockTransport(transport), trust_env=False)
    async with AsyncOpenAI(api_key="", admin_api_key="fixture-admin", base_url=ENDPOINT,
                           http_client=http, max_retries=2) as client:
        p = OpenAIProvider(client=client, config={"max_retries": 0})
        with pytest.raises(TypeError, match="Could not resolve authentication method"):
            await client.models.list()
        with pytest.raises(kernel_errors.LLMError) as raised:
            await p.list_models(filters={})
        assert raised.value.provider == "openai" and raised.value.retryable is False
        assert requests == []
        assert p.client is client and client.api_key == "" and client.admin_api_key == "fixture-admin"
        assert client.max_retries == 2 and not http.is_closed
    assert http.is_closed


@pytest.mark.asyncio
async def test_fresh_refresh_and_empty_without_defaults_aliases_or_cache():
    pages = [catalog("gpt-6-new-tier"), catalog(), catalog("gpt-6-other-tier")]
    async with sdk_provider(lambda _: httpx.Response(200, json=pages.pop(0)), default_model="absent") as (p, req, _):
        first = await p.list_models()
        assert ids(first) == {"gpt-6-new-tier"}
        assert await p.list_models() == []
        assert ids(await p.list_models()) == {"gpt-6-other-tier"}
        assert ids(first) == {"gpt-6-new-tier"} and len(req) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("status, error_type, retryable", [
    (400, kernel_errors.LLMError, False), (401, kernel_errors.AuthenticationError, False),
    (403, kernel_errors.AccessDeniedError, False), (404, kernel_errors.NotFoundError, False),
    (429, kernel_errors.RateLimitError, True), (500, kernel_errors.ProviderUnavailableError, True),
    (503, kernel_errors.ProviderUnavailableError, True),
])
async def test_api_errors_typed_sanitized_not_empty(status, error_type, retryable, caplog):
    secret = "fixture-secret-not-for-diagnostics"
    async with sdk_provider(lambda _: httpx.Response(status, json={"error": {"message": secret}})) as (p, req, _):
        with pytest.raises(error_type) as raised:
            await p.list_models()
        assert type(raised.value) is error_type
        assert raised.value.provider == "openai" and raised.value.status_code == status
        assert raised.value.retryable is retryable
        assert secret not in str(raised.value) and secret not in caplog.text
        assert len(req) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", [httpx.ConnectError, httpx.ReadTimeout])
async def test_transport_failures_typed_retryable_sanitized(error_type):
    def receive(request):
        raise error_type("fixture-secret-transport", request=request)
    async with sdk_provider(receive) as (p, req, _):
        with pytest.raises(kernel_errors.LLMError) as raised:
            await p.list_models()
        assert raised.value.retryable is True
        assert "fixture-secret" not in str(raised.value) and len(req) == 1


@pytest.mark.asyncio
async def test_cloudflare_typed_transient():
    async with sdk_provider(lambda _: httpx.Response(403, content="<html>Cloudflare challenge</html>",
                                                    headers={"content-type": "text/html"})) as (p, req, _):
        with pytest.raises(kernel_errors.ProviderUnavailableError) as raised:
            await p.list_models()
        assert raised.value.status_code == 403 and raised.value.retryable is True
        assert len(req) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("body", [
    None, [], {}, {"data": []}, {"object": "model", "data": []}, {"object": "list"},
    {"object": "list", "data": None}, {"object": "list", "data": {}},
    {"object": "list", "data": [None]}, {"object": "list", "data": ["gpt-6-sol"]},
    {"object": "list", "data": [{}]}, catalog(None), catalog(123), catalog(True),
    catalog(""), catalog(" "), catalog(" padded-id "), catalog("gpt-6-sol", has_more="true"),
    catalog("gpt-6-sol", has_more=None), catalog("gpt-6-sol", has_more=True),
])
async def test_malformed_or_partial_catalog_nonretryable_not_partial_success(body):
    async with sdk_provider(lambda _: httpx.Response(200, content=json.dumps(body),
                                                    headers={"content-type": "application/json"})) as (p, req, _):
        with pytest.raises(kernel_errors.LLMError) as raised:
            await p.list_models()
        assert raised.value.retryable is False and raised.value.provider == "openai"
        assert len(req) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("body", [b"not json", b"{", b"<html>login</html>", b"\xff"])
async def test_unparseable_response_typed(body):
    async with sdk_provider(lambda _: httpx.Response(200, content=body)) as (p, req, _):
        with pytest.raises(kernel_errors.LLMError) as raised:
            await p.list_models()
        assert raised.value.retryable is False and len(req) == 1


@pytest.mark.asyncio
async def test_cancellation_propagates_without_retry_or_empty_success():
    def receive(_):
        raise asyncio.CancelledError
    async with sdk_provider(receive) as (p, req, _):
        with pytest.raises(asyncio.CancelledError):
            await p.list_models()
        assert len(req) == 1


@pytest.mark.asyncio
async def test_missing_credentials_typed_without_creating_client():
    p = OpenAIProvider()
    with pytest.raises(kernel_errors.AuthenticationError):
        await p.list_models()
    assert p._client is None
