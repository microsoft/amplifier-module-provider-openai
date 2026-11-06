"""Strict optional discovery through the real SDK and an offline HTTP transport.

At base b3388a679f1d0ef76dcdaedc8b51cfc7d04c7561 OpenAIProvider has no
``discover_models`` method (source/AST baseline). These tests must fail there;
none replaces discovery, SDK model listing, or the production error mapper.
"""

import asyncio
import json
from contextlib import asynccontextmanager
from copy import deepcopy

import httpx
import pytest
from amplifier_core import llm_errors as kernel_errors
from openai import AsyncOpenAI

from amplifier_module_provider_openai import OpenAIProvider

ENDPOINT = "https://catalog.example.invalid/account/v1/"


def catalog(*ids, **pagination):
    return {
        "object": "list",
        "data": [
            {"id": model_id, "object": "model", "created": 1, "owned_by": "fixture"}
            for model_id in ids
        ],
        **pagination,
    }


def discovered(*ids):
    return {
        "version": 1,
        "source": "provider-api",
        "models": [{"id": model_id, "display_name": model_id} for model_id in ids],
    }


@asynccontextmanager
async def sdk_provider(receive, **config):
    """Exercise the real bound client; every test also forbids inference traffic."""
    requests = []

    def transport(request):
        requests.append(request)
        assert request.method == "GET"
        assert str(request.url).split("?")[0] == ENDPOINT + "models"
        assert set(request.url.params) <= {"after"}
        assert not request.content
        assert request.headers["authorization"] == "Bearer fixture-discovery-key"
        assert request.headers["openai-organization"] == "fixture-org"
        assert request.headers["openai-project"] == "fixture-project"
        return receive(request)

    http = httpx.AsyncClient(transport=httpx.MockTransport(transport), trust_env=False)
    async with AsyncOpenAI(
        api_key="fixture-discovery-key",
        base_url=ENDPOINT,
        organization="fixture-org",
        project="fixture-project",
        http_client=http,
        max_retries=2,
    ) as client:
        provider = OpenAIProvider(
            client=client,
            config={"max_retries": 0, "base_url": ENDPOINT, **config},
        )
        yield provider, requests, client
        assert provider.client is client
        assert client.max_retries == 2  # Discovery must not mutate inference settings.
        assert not http.is_closed  # The bound transport still belongs to the caller.
    assert http.is_closed


@pytest.mark.asyncio
async def test_discovery_returns_only_exact_api_ids_without_legacy_filters_or_defaults():
    ids = [
        "gpt-4o",
        "gpt-4.1-2025-04-14",
        "o3-mini",
        "gpt-5.6",  # Do not expand this into a configured alias target.
        "gpt-5.6-sol-2026-07-09",
        "gpt-6.1-sol",
        "ft:gpt-4.1:fixture:custom:123",
        "gpt-image-1",
        "text-embedding-3-small",
        "future-unrecognized-family",
    ]
    async with sdk_provider(
        lambda _: httpx.Response(200, json=catalog(*ids)),
        default_model="selection-only-not-on-api",
        hide_dated_models=True,
    ) as (provider, requests, _):
        assert await provider.discover_models() == discovered(*ids)
        assert len(requests) == 1
        assert not requests[0].url.query


@pytest.mark.asyncio
async def test_discovery_preserves_absent_account_binding_after_environment_changes(monkeypatch):
    # Absence is part of the bound account. SDK client reconstruction must not
    # pick up organization/project values added after the original was created.
    for name in (
        "OPENAI_ORG_ID", "OPENAI_PROJECT_ID",
        "OPENAI_ADMIN_KEY", "OPENAI_CUSTOM_HEADERS",
    ):
        monkeypatch.delenv(name, raising=False)

    requests = []
    pages = [
        catalog("first-page-id", has_more=True),
        catalog("second-page-id", has_more=False),
    ]

    def transport(request):
        requests.append(request)
        # Check during the request too: temporarily mutating shared retry/account
        # settings and restoring them afterwards is not safe for inference.
        assert provider.client is client
        assert client.max_retries == 2
        assert client.organization is None
        assert client.project is None
        return httpx.Response(200, json=pages[len(requests) - 1])

    http = httpx.AsyncClient(transport=httpx.MockTransport(transport), trust_env=False)
    async with AsyncOpenAI(
        api_key="fixture-discovery-key",
        base_url=ENDPOINT,
        organization=None,
        project=None,
        default_headers={"X-Discovery-Fixture": "bound-header"},
        http_client=http,
        max_retries=2,
    ) as client:
        assert client.organization is None
        assert client.project is None
        # Absent fields use fresh Omit sentinel instances on each property read.
        # Compare actual string header values, not sentinel object identities.
        headers_before = {
            name: value for name, value in client.default_headers.items()
            if isinstance(value, str)
        }
        provider = OpenAIProvider(
            client=client, config={"max_retries": 0, "base_url": ENDPOINT},
        )
        monkeypatch.setenv("OPENAI_ORG_ID", "fixture-later-org")
        monkeypatch.setenv("OPENAI_PROJECT_ID", "fixture-later-project")

        assert await provider.discover_models() == discovered("first-page-id", "second-page-id")
        assert [dict(request.url.params) for request in requests] == [
            {}, {"after": "first-page-id"},
        ]
        for request in requests:
            assert request.method == "GET"
            assert str(request.url).split("?")[0] == ENDPOINT + "models"
            assert not request.content
            assert request.headers["authorization"] == "Bearer fixture-discovery-key"
            assert request.headers["x-discovery-fixture"] == "bound-header"
            assert "openai-organization" not in request.headers
            assert "openai-project" not in request.headers

        assert provider.client is client
        assert client.organization is None
        assert client.project is None
        assert client.api_key == "fixture-discovery-key"
        assert str(client.base_url) == ENDPOINT
        assert {
            name: value for name, value in client.default_headers.items()
            if isinstance(value, str)
        } == headers_before
        assert client.max_retries == 2
        assert not http.is_closed
    assert http.is_closed


@pytest.mark.asyncio
async def test_discovery_rejects_admin_only_credentials_before_transport(monkeypatch):
    for name in (
        "OPENAI_API_KEY", "OPENAI_ADMIN_KEY", "OPENAI_CUSTOM_HEADERS",
        "OPENAI_ORG_ID", "OPENAI_PROJECT_ID",
    ):
        monkeypatch.delenv(name, raising=False)

    requests = []

    def transport(request):
        requests.append(request)
        # A valid response makes an unauthorized admin fallback observable as
        # success, rather than a transport exception that discovery could wrap.
        return httpx.Response(200, json=catalog("admin-fallback-must-not-be-used"))

    http = httpx.AsyncClient(transport=httpx.MockTransport(transport), trust_env=False)
    async with AsyncOpenAI(
        api_key="",
        admin_api_key="fixture-admin",
        base_url=ENDPOINT,
        http_client=http,
        max_retries=2,
    ) as client:
        provider = OpenAIProvider(
            client=client, config={"max_retries": 0, "base_url": ENDPOINT},
        )
        # The generated /models operation allows bearer_auth only. Generic
        # client.get defaults also allow admin_api_key_auth in SDK 3.29.
        with pytest.raises(TypeError, match="Could not resolve authentication method"):
            await client.models.list()
        assert requests == []

        with pytest.raises(kernel_errors.LLMError) as raised:
            await provider.discover_models()
        assert raised.value.provider == "openai"
        assert raised.value.retryable is False
        assert requests == []  # Refuse locally, not after sending the admin key.
        assert provider.client is client
        assert client.api_key == ""
        assert client.admin_api_key == "fixture-admin"
        assert client.max_retries == 2
        assert not http.is_closed
    assert http.is_closed


@pytest.mark.asyncio
async def test_successful_empty_catalog_is_not_a_default_or_error():
    async with sdk_provider(
        lambda _: httpx.Response(200, json=catalog()),
        default_model="selection-only-not-on-api",
    ) as (provider, requests, _):
        assert await provider.discover_models() == discovered()
        assert len(requests) == 1


@pytest.mark.asyncio
async def test_each_refresh_reads_every_page_and_observes_changes_without_a_cache():
    pages = [
        catalog("gpt-4o", has_more=True, last_id="cursor-first"),
        catalog("o3-mini", has_more=True),  # SDK AsyncPage ignores this cursor.
        catalog("gpt-5.6", has_more=False),
        catalog("new-api-id", has_more=True, last_id="cursor-refresh"),
        catalog(has_more=False),
        catalog(),
    ]
    queries = []

    def receive(request):
        queries.append(dict(request.url.params))
        return httpx.Response(200, json=pages[len(queries) - 1])

    async with sdk_provider(receive) as (provider, requests, _):
        first = await provider.discover_models()
        assert first == discovered("gpt-4o", "o3-mini", "gpt-5.6")
        assert await provider.discover_models() == discovered("new-api-id")
        assert await provider.discover_models() == discovered()
        assert first == discovered("gpt-4o", "o3-mini", "gpt-5.6")
        assert len(requests) == 6
    assert queries == [
        {}, {"after": "cursor-first"}, {"after": "o3-mini"},
        {}, {"after": "cursor-refresh"}, {},
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "selection",
    [
        {},
        {"model": "not-in-catalog", "default_model": "gpt-4o",
         "reasoning_effort": "low", "priority": 1},
        {"model": "another-selection", "default_model": "gpt-5.6-sol",
         "reasoning_effort": "high", "priority": 99, "hide_dated_models": False},
    ],
)
async def test_selection_configuration_does_not_change_inventory_or_wire_request(selection):
    async with sdk_provider(
        lambda _: httpx.Response(200, json=catalog("exact-api-id", "gpt-4o-2024-08-06")),
        **selection,
    ) as (provider, requests, _):
        before = deepcopy(provider.config)
        default_before = provider.default_model
        assert await provider.discover_models() == discovered(
            "exact-api-id", "gpt-4o-2024-08-06"
        )
        assert provider.config == before
        assert provider.default_model == default_before
        assert len(requests) == 1
        assert not requests[0].url.query


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status, error_type, retryable",
    [
        (400, kernel_errors.LLMError, False),
        (401, kernel_errors.AuthenticationError, False),
        (403, kernel_errors.AccessDeniedError, False),
        (404, kernel_errors.NotFoundError, False),
        (429, kernel_errors.RateLimitError, True),
        (500, kernel_errors.ProviderUnavailableError, True),
        (503, kernel_errors.ProviderUnavailableError, True),
    ],
)
async def test_api_errors_are_typed_not_empty_or_unsupported(status, error_type, retryable):
    async with sdk_provider(
        lambda _: httpx.Response(status, json={"error": {"message": "fixture failure"}}),
        max_retries=4,
    ) as (provider, requests, _):
        with pytest.raises(error_type) as raised:
            await provider.discover_models()
        assert type(raised.value) is error_type
        assert raised.value.provider == "openai"
        assert raised.value.status_code == status
        assert raised.value.retryable is retryable
        assert len(requests) == 1  # Neither SDK nor legacy retries are used.


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", [httpx.ConnectError, httpx.ReadTimeout])
async def test_transport_failures_are_typed_and_retryable(error_type):
    def receive(request):
        raise error_type("fixture transport failure", request=request)

    async with sdk_provider(receive) as (provider, requests, _):
        with pytest.raises(kernel_errors.LLMError) as raised:
            await provider.discover_models()
        assert raised.value.provider == "openai"
        assert raised.value.retryable is True
        assert len(requests) == 1


@pytest.mark.asyncio
async def test_cloudflare_challenge_is_typed_transient_without_hidden_retries():
    async with sdk_provider(
        lambda _: httpx.Response(
            403, content="<html>Cloudflare challenge</html>",
            headers={"content-type": "text/html"},
        )
    ) as (provider, requests, _):
        with pytest.raises(kernel_errors.ProviderUnavailableError) as raised:
            await provider.discover_models()
        assert raised.value.status_code == 403
        assert raised.value.retryable is True
        assert len(requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        None, [], {}, {"data": []}, {"object": "model", "data": []},
        {"object": "list"}, {"object": "list", "data": None},
        {"object": "list", "data": {}},
        {"object": "list", "data": [None]},
        {"object": "list", "data": ["gpt-4o"]},
        {"object": "list", "data": [{}]},
        catalog(None), catalog(123), catalog(True), catalog(""), catalog(" "),
        catalog(" padded-id "), catalog("gpt-4o", has_more="true"),
        catalog("gpt-4o", has_more=None), catalog(has_more=True),
        catalog("gpt-4o", has_more=True, last_id=""),
        catalog("gpt-4o", has_more=True, last_id=123),
    ],
)
async def test_malformed_payload_is_a_nonretryable_kernel_error(body):
    # A null response must be JSON null, not httpx's absent-body sentinel.
    async with sdk_provider(
        lambda _: httpx.Response(
            200, content=json.dumps(body), headers={"content-type": "application/json"}
        )
    ) as (provider, requests, _):
        with pytest.raises(kernel_errors.LLMError) as raised:
            await provider.discover_models()
        assert raised.value.retryable is False
        assert raised.value.provider == "openai"
        assert len(requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("body", [b"not json", b"{", b"<html>login</html>", b"\xff"])
async def test_unparseable_response_is_a_typed_error(body):
    async with sdk_provider(
        lambda _: httpx.Response(200, content=body, headers={"content-type": "text/html"})
    ) as (provider, requests, _):
        with pytest.raises(kernel_errors.LLMError) as raised:
            await provider.discover_models()
        assert raised.value.retryable is False
        assert len(requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["auth", "malformed", "repeated-cursor"])
async def test_later_page_failure_never_returns_partial_success_and_refresh_restarts(failure):
    calls = 0

    def receive(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(200, json=catalog("first-page-id", has_more=True))
        if calls == 2:
            assert request.url.params["after"] == "first-page-id"
            if failure == "auth":
                return httpx.Response(401, json={"error": {"message": "fixture revoked"}})
            if failure == "malformed":
                return httpx.Response(200, json={"object": "list", "data": None})
            return httpx.Response(200, json=catalog(
                "second-page-id", has_more=True, last_id="first-page-id"
            ))
        assert not request.url.query
        return httpx.Response(200, json=catalog("fresh-id"))

    async with sdk_provider(receive) as (provider, requests, _):
        error_type = (kernel_errors.AuthenticationError if failure == "auth"
                      else kernel_errors.LLMError)
        with pytest.raises(error_type):
            await provider.discover_models()
        assert await provider.discover_models() == discovered("fresh-id")
        assert len(requests) == 3


@pytest.mark.asyncio
async def test_cancellation_propagates_instead_of_returning_empty_or_retrying():
    def receive(_):
        raise asyncio.CancelledError

    async with sdk_provider(receive) as (provider, requests, _):
        with pytest.raises(asyncio.CancelledError):
            await provider.discover_models()
        assert len(requests) == 1


@pytest.mark.asyncio
async def test_missing_credentials_are_typed_without_creating_a_client():
    provider = OpenAIProvider()
    with pytest.raises(kernel_errors.AuthenticationError):
        await provider.discover_models()
    assert provider._client is None


@pytest.mark.asyncio
async def test_discovery_does_not_change_legacy_listing_filter_metadata_or_order():
    ids = ["gpt-5.6", "gpt-4o", "gpt-5.1-2025-11-13", "gpt-5.1", "o3-mini"]
    async with sdk_provider(
        lambda _: httpx.Response(200, json=catalog(*ids)), hide_dated_models=True,
    ) as (provider, requests, _):
        before = await provider.list_models()
        assert [row.id for row in before] == ["gpt-5.1", "gpt-5.6"]
        assert all(row.defaults for row in before)
        assert await provider.discover_models() == discovered(*ids)
        after = await provider.list_models()
        assert before == after
        assert len(requests) == 3
