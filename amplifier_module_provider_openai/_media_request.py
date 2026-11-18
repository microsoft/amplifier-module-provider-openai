"""Optional Foundation request facts and request-local physical-send witness.

No Foundation import, hydration, credential discovery or host transport patching.
Unmarked ordinary requests never enter this path.
"""

import copy
import hashlib
import inspect
import json
import os
import weakref
from collections.abc import Mapping
from contextlib import asynccontextmanager

import httpx
import openai
from amplifier_core.llm_errors import InvalidRequestError

from ._chat_images import image_url, known_vision, project_messages

KEY = "experimental.media.v1"
# The API permits non-animated GIF, which a MIME-only fact cannot express.
MIMES = ("image/png", "image/jpeg", "image/webp")


def empty_support():
    return {"v": 2, "positions": [], "mime_types": list(MIMES)}


def refused():
    return InvalidRequestError(
        "Private media request is unavailable or its count/send binding changed; "
        "prepare and fit a fresh request through the current media capability.",
        provider="openai", retryable=False,
    )


def digest(value):
    try:
        return hashlib.sha256(json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False,
            default=lambda item: item.model_dump(mode="json") if hasattr(item, "model_dump") else _invalid_json(),
        ).encode()).hexdigest()
    except (TypeError, ValueError):
        raise refused() from None


def _invalid_json():
    raise TypeError("Unsupported binding value")


def marked(request):
    return any(KEY in (m.metadata or {}) for m in request.messages)


def option_digest(provider, request, options):
    """Match Foundation's redundant model/effort normalization, not SDK fields."""
    normalized = dict(options)
    if normalized.get("model") == provider._resolve_request_model(request):
        normalized.pop("model", None)
    if normalized.get("reasoning_effort") == request.reasoning_effort:
        normalized.pop("reasoning_effort", None)
    return digest(normalized)


def support(provider, request, model, options):
    """Only classify the frozen Core positions the existing converter accepts."""
    empty = empty_support()
    if (
        not provider._uses_standard_openai_endpoint()
        or not isinstance(options, Mapping)
        or not isinstance(model, str)
        or model != provider._resolve_request_model(request, options.get("model"))
        or not known_vision(model)
        or provider.extra_request_params
        or (provider._client is not None and type(provider._client) is not openai.AsyncOpenAI)
    ):
        return empty
    # These routes have separate contracts, not indexed generic image input.
    if any(k in options for k in (
        "extra_body", "extra_headers", "extra_query", "single_attempt",
        "single_attempt_version", "background",
    )):
        return empty
    messages = [m.model_dump() for m in request.messages]
    from .neutral_computer import ordinary_names

    ordinary = tuple(ordinary_names([*(request.tools or []), *(options.get("tools") or [])]))
    for msg in messages:
        if msg["role"] != "assistant":
            continue
        content = msg["content"] if isinstance(msg["content"], list) else []
        for call in [*(msg.get("tool_calls") or []), *content]:
            if not isinstance(call, dict):
                continue
            args, name = call.get("input") or call.get("arguments") or {}, call.get("name")
            if name in ordinary and name != "apply_patch":
                continue
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    return empty
            if isinstance(args, dict) and (
                (name == "computer" and isinstance(args.get("actions"), list))
                or (name == "apply_patch" and args.get("type") in {
                    "create_file", "update_file", "delete_file", "rename_file",
                })
            ):
                return empty
    positions = []
    for index, msg in enumerate(messages):
        metadata = msg.get("metadata") or {}
        canonical = metadata.get("openai:output_items", [])
        if not isinstance(canonical, list) or any(
            not isinstance(item, dict) or item.get("type") not in {
                "function_call", "tool_call", "message", "reasoning",
            } for item in canonical
        ) or any("compaction" in k or "computer" in k for k in metadata):
            return empty
        if msg["role"] not in {"user", "tool"}:
            continue
        if msg.get("tool_call_id") in provider._native_call_ids:
            return empty
        content = msg["content"]
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        if any(not isinstance(b, dict) or b.get("type") not in {"text", "image"} for b in content):
            continue
        if any(
            b.get("type") == "image" and (
                image_url(b) is None
                or (b["source"].get("type") == "base64" and b["source"].get("media_type") not in MIMES)
            ) for b in content
        ):
            return empty
        positions.append({"message_index": index, "role": msg["role"]})
        # Detached synthetic classifier input, never a request or asset. This
        # shares the real converter's strict complete Core call/result checks.
        msg["content"] = [*content, {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": "eA=="},
        }]
    try:
        projected = project_messages(
            messages, qualified=True, model=model, strict_direct=True,
            ordinary_names=ordinary,
        )
        # A generic tool must remain a correlated ordinary result, never native.
        if len(projected) != len(messages):
            return empty
        for pos in positions:
            msg = projected[pos["message_index"]]
            if msg.get("role") != pos["role"] or not any(
                b.get("type") == "image" for b in msg.get("content", [])
            ):
                return empty
    except (TypeError, ValueError, InvalidRequestError):
        return empty
    return {"v": 2, "positions": positions, "mime_types": list(MIMES)}


class _PhysicalValidation:
    """Provider-owned SDK adapter: run AFTER SDK auth and HTTP request hooks.

    The original HTTP client still owns routing, connection pools and transport.
    Nothing is installed on or monkeypatched into the shared client.
    """

    def __init__(self, original, validate):
        self._original = original
        self._validate = validate
        super().__init__(
            event_hooks=original.event_hooks, cookies=original.cookies.jar,
            headers=dict(original.headers), params=dict(original.params),
            follow_redirects=False, trust_env=False,
        )

    async def _send_single_request(self, request):
        # SDK/HTTP hooks have finished. Freeze their actual outgoing body and
        # headers before validation; a retained hook reference cannot mutate the
        # admitted request while the original transport later awaits connection.
        body = b"".join([part async for part in request.stream])
        request = type(request)(
            request.method, request.url, headers=request.headers,
            content=body, extensions=dict(request.extensions),
        )
        self._validate(request)
        return await self._original._send_single_request(request)


class _PhysicalClient(_PhysicalValidation, httpx.AsyncClient):
    """The SDK's supported legacy HTTPX injection path."""


class _SDKPhysicalClient(_PhysicalValidation, openai.DefaultAsyncHttpxClient):
    """The SDK's own default HTTP family (HTTPX2 in SDK3.29)."""


class Binding:
    def __init__(self, provider, request, options):
        self.provider = provider
        self.request_ref = weakref.ref(request)
        self.options = options
        self.model = provider._resolve_request_model(request, options.get("model"))
        self.options_digest = option_digest(provider, request, options)
        self.original_options_digest = digest(options)
        self.request_digest = digest(request.model_dump())
        self.coordinator = provider.coordinator
        self.params_digest = None
        self.connection_digest = None
        self.client = provider._client
        before_connection = self._initial_connection()
        before_injected = self._connection() if self.client is not None else None
        self.capability = self._capability()
        # Validate exact issuer identity BEFORE creating the legitimate cold SDK.
        self.validate_capability()
        if self._initial_connection() != before_connection:
            raise refused()
        if before_injected is not None and self._connection() != before_injected:
            raise refused()
        self.client = provider.client
        if type(self.client) is not openai.AsyncOpenAI or type(self.client._client) not in {
            httpx.AsyncClient, openai.DefaultAsyncHttpxClient,
        }:
            raise refused()
        # Dynamic auth/providers and custom HTTP auth require their own proven
        # after-auth binding. Do not invent it for this first finite route.
        if any(getattr(self.client, k, None) is not None for k in (
            "_api_key_provider", "_provider_runtime", "workload_identity",
        )) or self.client._client.auth is not None:
            raise refused()
        self.connection_digest = self._connection()

    def _capability(self):
        getter = getattr(self.provider.coordinator, "get_capability", None)
        try:
            return getter(KEY) if callable(getter) else None
        except Exception:  # noqa: BLE001 - capability lookup is not diagnostic output
            raise refused() from None

    def _initial_connection(self):
        p = self.provider
        return digest({
            "key": p._api_key, "config": p.config, "base_url": p.base_url,
            "default_model": p.default_model, "client": id(p._client),
            "cold_environment": [os.environ.get(name) for name in (
                "OPENAI_BASE_URL", "OPENAI_ORG_ID", "OPENAI_PROJECT_ID",
            )],
        })

    def validate_capability(self):
        capability = self._capability()
        request = self.request_ref()
        if (
            self.provider.coordinator is not self.coordinator
            or capability is not self.capability
            or getattr(capability, "enabled", None) is not True
            or not callable(getattr(capability, "validate", None))
            or digest(self.options) != self.original_options_digest
            or request is None or digest(request.model_dump()) != self.request_digest
        ):
            raise refused()
        try:
            result = capability.validate(
                request, purpose=self.purpose, request_options=copy.deepcopy(self.options),
            )
        except Exception:  # noqa: BLE001 - issuer errors may contain private data
            raise refused() from None
        if inspect.isawaitable(result):
            if inspect.iscoroutine(result):
                result.close()
            raise refused()
        if not isinstance(result, dict) or result.get("valid") is not True:
            raise refused()
        if (
            self._capability() is not capability or self.provider.coordinator is not self.coordinator
            or getattr(capability, "enabled", None) is not True
            or digest(self.options) != self.original_options_digest
            or digest(request.model_dump()) != self.request_digest
        ):
            raise refused()

    purpose = "provider-count"

    def _connection(self):
        p, c = self.provider, self.client
        if p._client is not c or not p._uses_standard_openai_endpoint():
            raise refused()
        if any(getattr(c, k, None) is not None for k in (
            "_api_key_provider", "_provider_runtime", "workload_identity",
        )):
            raise refused()
        if not isinstance(c.api_key, str) or not c.api_key:
            raise refused()
        # Explicit HTTP defaults may otherwise select another account beneath
        # the SDK. This finite route preserves generic headers, but refuses
        # an independently configured account/auth layer rather than dropping it.
        for name, value in (
            ("authorization", "Bearer " + c.api_key),
            ("openai-organization", c.organization),
            ("openai-project", c.project),
        ):
            if name in c._client.headers and c._client.headers[name] != value:
                raise refused()
        return digest({
            "provider_key": p._api_key, "config": p.config,
            "base_url": p.base_url, "default_model": p.default_model,
            "extras": p.extra_request_params,
            "client": id(c), "http": id(c._client),
            "endpoint": str(c.base_url), "key": c.api_key,
            "organization": c.organization, "project": c.project,
            "headers": dict(c._custom_headers), "query": dict(c._custom_query),
            "http_headers": dict(c._client.headers), "http_query": str(c._client.params),
            "auth": id(c._client.auth),
            "transport": id(c._client._transport_for_url(
                type(c._client.base_url)("https://api.openai.com/v1/responses")
            )),
            "cookies": [(cookie.name, cookie.value, cookie.domain, cookie.path) for cookie in c._client.cookies.jar],
            "request_hooks": [id(h) for h in c._client.event_hooks["request"]],
            "streaming": p.use_streaming, "timeout": str(p.timeout),
            "retry": [p._retry_config.max_retries, p._retry_config.initial_delay,
                      p._retry_config.max_delay],
        })

    def validate(self, params, purpose):
        self.purpose = purpose
        body = {k: v for k, v in params.items() if k != "timeout"}
        if body.get("model") != self.model or body.get("background") or body.get("extra_body"):
            raise refused()
        fingerprint = digest(body)
        if self.params_digest is None:
            self.params_digest = fingerprint
        if self.params_digest != fingerprint or self._connection() != self.connection_digest:
            raise refused()
        request = self.request_ref()
        if request is None:
            raise refused()
        current = self.provider._budget_params(request, **self.options)
        if digest({k: v for k, v in current.items() if k != "timeout"}) != fingerprint:
            raise refused()
        self.validate_capability()
        if (
            self._connection() != self.connection_digest or digest(body) != fingerprint
            or digest(request.model_dump()) != self.request_digest
        ):
            raise refused()

    def sdk_client(self, params, purpose, *, stream=False):
        self.validate(params, purpose)
        expected = self.provider._native_count_params(params) if purpose == "provider-count" else dict(params)
        if expected is None:
            raise refused()
        expected.pop("timeout", None)
        if stream:
            expected["stream"] = True
        expected_digest = digest(expected)

        def physical(request):
            self.validate(params, purpose)
            suffix = "/responses/input_tokens" if purpose == "provider-count" else "/responses"
            try:
                actual_digest = digest(json.loads(request.content))
            except (TypeError, ValueError):
                raise refused() from None
            if (
                request.method != "POST" or str(request.url) != "https://api.openai.com/v1" + suffix
                or actual_digest != expected_digest
                or request.headers.get("authorization") != "Bearer " + self.client.api_key
                or request.headers.get("openai-organization") != self.client.organization
                or request.headers.get("openai-project") != self.client.project
            ):
                raise refused()

        adapter_type = _PhysicalClient if isinstance(self.client._client, httpx.AsyncClient) else _SDKPhysicalClient
        adapter = adapter_type(self.client._client, physical)
        return self.client.with_options(http_client=adapter, max_retries=0)

    @asynccontextmanager
    async def sdk_context(self, params, purpose):
        client = self.sdk_client(params, purpose)
        try:
            yield client
        finally:
            await client.close()

    @asynccontextmanager
    async def stream(self, params, timeout):
        client = self.sdk_client(params, "provider-send", stream=True)
        try:
            async with client.responses.stream(**params, timeout=timeout) as stream:
                yield stream
        finally:
            await client.close()


def binding(provider, request, options):
    records = getattr(provider, "_media_request_bindings", None)
    existing = (records or {}).get(id(request))
    if existing is not None and existing.request_ref() is request:
        if not marked(request) or existing.options_digest != option_digest(provider, request, options):
            raise refused()
        existing.validate_capability()
        return existing
    if not marked(request):
        return None
    model = provider._resolve_request_model(request, options.get("model"))
    if not support(provider, request, model, options)["positions"]:
        raise refused()
    if records is None:
        records = provider._media_request_bindings = {}
    key = id(request)
    result = Binding(provider, request, options)
    # No pixels/request retained by the provider after caller/issuer release.
    result.request_ref = weakref.ref(request, lambda _: records.pop(key, None))
    records[key] = result
    return result