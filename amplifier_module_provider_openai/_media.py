"""Provider-owned media witness and known Responses diagnostic boundaries.

No hydration/store access, public witness metadata, Foundation or peer imports.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import weakref
from contextvars import ContextVar
from copy import deepcopy
from functools import wraps
from typing import Any

import httpx
from amplifier_core.llm_errors import InvalidRequestError
from openai import AsyncAzureOpenAI, AsyncOpenAI, DefaultAsyncHttpxClient
from openai.resources.responses import AsyncResponses
from openai.resources.responses.input_tokens import AsyncInputTokens
from pydantic import BaseModel

NAMESPACE = "experimental.media.v1"
NOT_RETAINED = "[Image transport not retained; no retrievable pixel reference]"
UNAVAILABLE = "[Image unavailable: selected model/route image support is unknown; no pixels supplied]"
# Exact documented input-image model facts. No optimistic unknown-model default.
VISION_MODELS = frozenset(
    {
        "gpt-6-astra",
        "gpt-6-sol",
        "gpt-6-luna",
        "gpt-6.1-sol",
        "gpt-6.1-terra",
        "gpt-5.6",
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-5.6-luna",
        "gpt-5.4",
        "gpt-5.4-pro",
        "gpt-5.4-mini",
        "gpt-5.5",
        "gpt-5.5-pro",
        "gpt-5.2",
        "gpt-5.2-pro",
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4.1-nano",
        "o1",
        "o1-pro",
        "o3",
        "o3-pro",
        "o4-mini",
    }
)
EVIDENCE = "https://developers.openai.com/api/docs/guides/images-vision"
CURRENT: ContextVar = ContextVar("openai_media_attempt", default=None)
# Capture SDK/HTTP dispatch implementations before any per-instance callbacks.
# These are function objects, never tokens, requests or image data.
SDK_DISPATCH = {
    cls: {name: fn for name, fn in inspect.getmembers(cls, inspect.isfunction)}
    for cls in (AsyncOpenAI, AsyncAzureOpenAI)
}
HTTP_DISPATCH = {
    cls: dict(inspect.getmembers(cls, inspect.isfunction))
    for cls in (DefaultAsyncHttpxClient, httpx.AsyncClient)
}
RESOURCE_DISPATCH = {
    cls: dict(inspect.getmembers(cls, inspect.isfunction))
    for cls in (AsyncResponses, AsyncInputTokens)
}


def standard_dispatch(client):
    expected = SDK_DISPATCH.get(type(client))
    if expected is None:
        return False
    http = next(
        (
            methods
            for cls, methods in HTTP_DISPATCH.items()
            if isinstance(client._client, cls)
        ),
        None,
    )
    if http is None:
        return False
    for obj, methods in ((client, expected), (client._client, http)):
        for name, function in methods.items():
            actual = getattr(obj, name, None)
            if getattr(actual, "__func__", actual) is not function:
                return False
    for resource in (client.responses, client.responses.input_tokens):
        methods = RESOURCE_DISPATCH.get(type(resource))
        if methods is None or resource._client is not client:
            return False
        for name, function in methods.items():
            actual = getattr(resource, name, None)
            if getattr(actual, "__func__", actual) is not function:
                return False
        for method in ("get", "post", "patch", "put", "delete"):
            if getattr(resource, "_" + method, None) != getattr(client, method):
                return False
    raw = client.responses.with_raw_response
    if raw._responses is not client.responses:
        return False
    for name in ("create", "compact"):
        actual = inspect.unwrap(getattr(raw, name))
        if (
            getattr(actual, "__self__", None) is not client.responses
            or getattr(actual, "__func__", None)
            is not RESOURCE_DISPATCH[AsyncResponses][name]
        ):
            return False
    return True


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).digest()


def capability(provider):
    getter = getattr(provider.coordinator, "get_capability", None)
    value = getter(NAMESPACE) if callable(getter) else None
    return value if getattr(value, "version", None) == 1 else None


def has_refs(request):
    return any(
        isinstance(getattr(m, "metadata", None), dict) and NAMESPACE in m.metadata
        for m in request.messages
    )


def canonical_images(value):
    if isinstance(value, (list, tuple)):
        return any(canonical_images(v) for v in value)
    if isinstance(value, BaseModel):
        value = value.model_dump(warnings=False)
    elif hasattr(value, "model_dump"):
        value = value.model_dump()
    return isinstance(value, dict) and (
        value.get("type") == "image"
        or (
            value.get("type") == "tool_result"
            and canonical_images(value.get("output", value.get("content")))
        )
    )


def protected(provider, request):
    return (
        capability(provider) is not None
        or has_refs(request)
        or any(canonical_images(m.content) for m in request.messages)
    )


def refuse():
    raise InvalidRequestError(
        "Private media request, selected connection or wire changed/expired/revoked; prepare and fit a new request. Request not sent.",
        provider="openai",
        retryable=False,
    )


class Scope:
    def __init__(self, provider, request, options):
        self.provider, self.request, self.options = provider, request, options
        extra = provider.extra_request_params
        if "input" in extra or (
            isinstance(extra.get("extra_body"), dict)
            and any(key in extra["extra_body"] for key in ("input", "model"))
        ):
            refuse()
        self.request_digest, self.options_digest = (
            digest(request.model_dump()),
            digest(options),
        )
        self.cap = capability(provider)
        if self.cap is None and has_refs(request):
            refuse()
        self.connection = self.binding()
        self.params_digest = None
        self.received = False
        self.entered = False
        self.values = ()
        self.native_request_view = None

    def binding(self):
        p = self.provider
        # Subclasses (notably Azure) own a different client slot. Bind the
        # already initialized actual transport, not the base cache attribute.
        client = p.client
        return (
            id(capability(p)),
            id(p.coordinator),
            p.base_url,
            p.default_model,
            id(client),
            digest(p.config),
            digest(p.extra_request_params),
            str(getattr(client, "base_url", p.base_url)),
            getattr(client, "organization", None),
            getattr(client, "project", None),
            digest(getattr(client, "api_key", None)),
            digest(getattr(client, "_custom_headers", None)),
            digest(getattr(client, "_custom_query", None)),
            id(getattr(client, "_client", None)),
            id(getattr(getattr(client, "_client", None), "_transport", None)),
        )

    def validate(self, purpose, params=None):
        if (
            self.binding() != self.connection
            or digest(self.request.model_dump()) != self.request_digest
            or digest(self.options) != self.options_digest
        ):
            refuse()
        if self.cap is not None:
            try:
                result = self.cap.validate(
                    self.request,
                    purpose=purpose,
                    request_options=deepcopy(self.options),
                )
            except Exception:  # noqa: BLE001 -- host refusal text may contain private data
                refuse()
            if not isinstance(result, dict) or result.get("valid") is not True:
                refuse()
        if params is not None:
            fingerprint = digest(params)
            if self.params_digest is None:
                records = getattr(self.provider, "_media_fitted", {})
                old = records.get(id(self.request))
                if (
                    old is not None
                    and old[0]() is self.request
                    and old[1:] != (fingerprint, self.options_digest, self.connection)
                ):
                    refuse()
                self.params_digest = fingerprint
                self.values = transport_values(params)
            elif self.params_digest != fingerprint:
                refuse()

    def remember_count(self):
        if self.params_digest is None:
            return
        records = getattr(self.provider, "_media_fitted", None)
        if records is None:
            records = self.provider._media_fitted = {}
        identity = id(self.request)
        records[identity] = (
            weakref.ref(self.request, lambda _: records.pop(identity, None)),
            self.params_digest,
            self.options_digest,
            self.connection,
        )


def check(purpose, params=None):
    current = CURRENT.get()
    if current is not None:
        current.validate(purpose, params)
    return current


def operation(method):
    @wraps(method)
    async def wrapped(self, request, *args, **kwargs):
        options = self._merge_request_options(
            kwargs.get("request_options"),
            {k: v for k, v in kwargs.items() if k != "request_options"},
        )
        current = CURRENT.get()
        if current is not None and current.provider is self:
            if (
                request is not current.request
                and request is not current.native_request_view
            ):
                refuse()
            if digest(options) != current.options_digest:
                refuse()
            return await method(self, request, *args, **kwargs)
        if not protected(self, request):
            return await method(self, request, *args, **kwargs)
        # Resolve the actual SDK connection before freezing it, without I/O.
        _ = self.client
        scope = Scope(self, request, options)
        token = CURRENT.set(scope)
        try:
            self._media_client()
            scope.validate("transport")
            return await method(self, request, *args, **kwargs)
        except Exception as error:
            if scope.values:
                error.args = (safe_text(str(error)),)
            if getattr(error, "retryable", False) and scope.entered:
                error.retryable = False
            raise
        finally:
            CURRENT.reset(token)

    return wrapped


def budget(method):
    @wraps(method)
    def wrapped(self, request, *args, **kwargs):
        options = self._merge_request_options(
            kwargs.get("request_options"),
            {
                k: v
                for k, v in kwargs.items()
                if k not in ("request_options", "context_estimate")
            },
        )
        current = CURRENT.get()
        if current is not None and current.provider is self:
            if (
                request is not current.request
                and request is not current.native_request_view
            ):
                refuse()
            if digest(options) != current.options_digest:
                refuse()
            return method(self, request, *args, **kwargs)
        if not protected(self, request):
            return method(self, request, *args, **kwargs)
        _ = self.client
        scope = Scope(self, request, options)
        token = CURRENT.set(scope)
        try:
            self._media_client()
            scope.validate("count")
            result = method(self, request, *args, **kwargs)
            if not inspect.isawaitable(result) and result is not None:
                scope.validate("count")
                scope.remember_count()
        finally:
            CURRENT.reset(token)
        if not inspect.isawaitable(result):
            return result

        async def settle():
            token = CURRENT.set(scope)
            try:
                result_value = await result
                scope.validate("count")
                if result_value is not None:
                    scope.remember_count()
                return result_value
            finally:
                CURRENT.reset(token)

        return settle()

    return wrapped


def diagnostic_copy(value: Any) -> Any:
    if isinstance(value, BaseModel):
        value = value.model_dump(warnings=False)
    elif hasattr(value, "model_dump"):
        value = value.model_dump()
    if isinstance(value, (list, tuple)):
        return [diagnostic_copy(v) for v in value]
    if not isinstance(value, dict):
        return deepcopy(value)
    kind = value.get("type")
    if isinstance(kind, str) and kind in {
        "input_image",
        "image",
        "image_url",
        "computer_screenshot",
    }:
        return {"type": "input_text", "text": NOT_RETAINED}
    copied = deepcopy(value)
    for key in (
        "input",
        "output",
        "content",
        "messages",
        "events",
        "response",
        "item",
        "extra_body",
    ):
        if isinstance(copied.get(key), (dict, list, tuple)):
            copied[key] = diagnostic_copy(copied[key])
    return copied


def transport_values(payload):
    found = []

    def visit(value):
        if isinstance(value, (list, tuple)):
            for part in value:
                visit(part)
        elif isinstance(value, dict):
            if value.get("type") in ("input_image", "computer_screenshot"):
                url = value.get("image_url")
                if isinstance(url, str) and url:
                    found.append(url)
                    if ";base64," in url:
                        found.append(url.split(";base64,", 1)[1])
            for key in ("input", "output", "content", "messages", "extra_body"):
                if key in value:
                    visit(value[key])

    visit(payload)
    return tuple(sorted(set(found), key=len, reverse=True))


def safe_text(text, params=None):
    scope = CURRENT.get()
    values = (
        transport_values(params)
        if params is not None
        else scope.values
        if scope
        else ()
    )
    for value in values:
        text = text.replace(value, NOT_RETAINED)
    return text


def diagnostic_error_copy(event):
    """Omit known structured transports and exact submitted error echoes only."""
    copied = diagnostic_copy(event)
    scope = CURRENT.get()
    if not scope or not isinstance(copied, dict):
        return copied

    def error_fields(value):
        if not isinstance(value, dict):
            return
        safe = diagnostic_copy(value)
        value.clear()
        value.update(safe)
        for key in ("message", "msg", "detail"):
            if isinstance(value.get(key), str):
                value[key] = safe_text(value[key])
        if isinstance(value.get("error"), dict):
            error_fields(value["error"])
        if isinstance(value.get("response"), dict):
            error_fields(value["response"])

    if copied.get("type") in ("error", "response.failed", "response.incomplete"):
        error_fields(copied)
    return copied
