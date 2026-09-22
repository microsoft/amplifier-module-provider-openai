"""Opt-in, one-count/one-generation completion for bounded readiness checks.

This is a provider request boundary, not a retry policy for ordinary chats. The
caller still owns durable admission/idempotency across processes and calls.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import math
import os
import re
from urllib.parse import urlparse

import httpx
from amplifier_core import llm_errors

CAPABILITY = "completion:single_attempt:v1"
RECEIPT_KEY = "openai:single_attempt"
CLOSE_TIMEOUT = 3.0


class SingleAttemptError(llm_errors.LLMError):
    """Sanitized failure with no permission to replay the request."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(f"single_attempt.{reason}", provider="openai", retryable=False)


def endpoint(provider):
    """Only a fresh base provider can own this mode's complete HTTP lifetime."""
    from . import OpenAIProvider, _installed_openai_supports_native_input_token_count

    if (
        type(provider) is not OpenAIProvider
        or provider._client is not None
        or not _installed_openai_supports_native_input_token_count()
    ):
        return None
    value = (
        provider.base_url
        or os.environ.get("OPENAI_BASE_URL")
        or "https://api.openai.com/v1"
    )
    try:
        parsed = urlparse(str(value))
        if (
            parsed.scheme != "https"
            or parsed.hostname != "api.openai.com"
            or parsed.port not in (None, 443)
            or parsed.path.rstrip("/") != "/v1"
            or parsed.query
            or parsed.fragment
            or parsed.username
            or parsed.password
        ):
            return None
    except ValueError:
        return None
    return str(value)


def _digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _integer(value):
    return type(value) is int and value >= 0


def _plan(provider, request, options):
    # Snapshot admission before any await; assembly is already a pure provider
    # transaction, and never commits chat history or conversion state here.
    request = copy.deepcopy(request)
    if (
        not isinstance(request.model, str)
        or not request.model
        or not isinstance(request.reasoning_effort, str)
        or not request.reasoning_effort
        or not _integer(request.max_output_tokens)
        or request.max_output_tokens < 1
        or type(request.timeout) not in (int, float)
        or not math.isfinite(request.timeout)
        or request.timeout <= 0
        or len(request.messages) != 1
        or request.messages[0].role != "user"
        or not isinstance(request.messages[0].content, str)
        or not request.messages[0].content.strip()
        or request.tools
        or request.tool_choice is not None
        or request.conversation_id is not None
        or request.stream is not False
        or request.response_format is not None
    ):
        raise SingleAttemptError("invalid_request")
    # Only the legacy equivalents of explicit portable selection fields are
    # allowed. They must not override the admitted values.
    if set(options) - {"model", "reasoning_effort"} or any(
        value != getattr(request, key) for key, value in options.items()
    ):
        raise SingleAttemptError("invalid_options")
    if request.messages[0].model_dump(exclude_none=True) != {
        "role": "user",
        "content": request.messages[0].content,
    }:
        raise SingleAttemptError("invalid_request")
    params, _, _ = provider._assemble_initial_responses_params(request, **options)
    expected_input = [
        {
            "role": "user",
            "content": [{"type": "input_text", "text": request.messages[0].content}],
        }
    ]
    reasoning = params.get("reasoning")
    if (
        params.get("model") != request.model
        or params.get("input") != expected_input
        or type(params.get("max_output_tokens")) is not int
        or params["max_output_tokens"] != request.max_output_tokens
        or not isinstance(reasoning, dict)
        or reasoning.get("effort") != request.reasoning_effort
        or any(
            params.get(key)
            for key in (
                "instructions",
                "tools",
                "tool_choice",
                "background",
                "stream",
                "store",
                "previous_response_id",
                "conversation",
            )
        )
        or params.get("truncation") not in (None, "disabled")
    ):
        raise SingleAttemptError("wire_mismatch")
    count_params = provider._native_count_params(params)
    if count_params is None:
        raise SingleAttemptError("unsupported_count")
    return request, params, count_params


def _validate_response(response, request):
    response_model = getattr(response, "model", "")
    if (
        getattr(response, "object", None) != "response"
        or not isinstance(getattr(response, "id", None), str)
        or not response.id
        or not isinstance(response_model, str)
        or not (
            response_model == request.model
            or re.fullmatch(
                re.escape(request.model) + r"-\d{4}-\d{2}-\d{2}", response_model
            )
        )
        or getattr(response, "status", None) != "completed"
        or getattr(response, "error", None) is not None
    ):
        raise SingleAttemptError("invalid_response")
    text = []
    for item in getattr(response, "output", []) or []:
        if item.type == "reasoning":
            continue
        if (
            item.type != "message"
            or item.role != "assistant"
            or item.status != "completed"
        ):
            raise SingleAttemptError("invalid_response")
        for block in item.content:
            if block.type != "output_text":
                raise SingleAttemptError("invalid_response")
            text.append(block.text)
    usage = getattr(response, "usage", None)
    if (
        not any(value.strip() for value in text)
        or usage is None
        or not _integer(usage.input_tokens)
        or not _integer(usage.output_tokens)
        or usage.output_tokens > request.max_output_tokens
    ):
        raise SingleAttemptError("invalid_response")


def _http_client(timeout):
    # The SDK's default client follows redirects, which could silently add a
    # transmission. This transport is owned exclusively by the bounded call.
    return httpx.AsyncClient(timeout=timeout, follow_redirects=False)


async def complete(provider, request, options):
    """Return only after one bounded SDK exchange and confirmed client close."""
    from . import AsyncOpenAI

    base_url = endpoint(provider)
    if base_url is None:
        raise SingleAttemptError("unsupported_endpoint_or_client")
    try:
        request, params, count_params = _plan(provider, request, options)
    except SingleAttemptError:
        raise
    except Exception:  # noqa: BLE001 - sanitize provider assembly diagnostics
        raise SingleAttemptError("invalid_request") from None
    request_hash = _digest(params)
    client = None
    transport = None
    try:
        # One owned client for BOTH operations: no temporary default-retry
        # count client, injected/shared transport, local fallback, or polling.
        transport = _http_client(request.timeout)
        client = AsyncOpenAI(
            api_key=provider._api_key,
            base_url=base_url,
            max_retries=0,
            timeout=request.timeout,
            http_client=transport,
        )
        async with asyncio.timeout(request.timeout):
            count = await client.responses.input_tokens.count(**count_params)
            tokens = getattr(count, "input_tokens", None)
            if not _integer(tokens):
                raise SingleAttemptError("count_failed")
            provider._guard_assembled_params(params, native_input_tokens=tokens)
            if _digest(params) != request_hash:
                raise SingleAttemptError("wire_mismatch")
            response = await client.responses.create(**params)
            _validate_response(response, request)
            result = provider._convert_to_chat_response(response)
            if result.tool_calls or not any(
                block.type == "text" and block.text.strip() for block in result.content
            ):
                raise SingleAttemptError("invalid_response")
    except (SingleAttemptError, asyncio.CancelledError):
        raise
    except TimeoutError:
        raise SingleAttemptError("timeout") from None
    except Exception:  # noqa: BLE001 - unknown SDK outcomes must not permit retry
        raise SingleAttemptError("failed") from None
    finally:
        if transport is not None:
            try:
                # Do not use the ordinary close helpers: their best-effort
                # cleanup deliberately swallows exceptions/timeouts.
                close = client.close if client is not None else transport.aclose
                await asyncio.wait_for(close(), CLOSE_TIMEOUT)
                if not transport.is_closed or (
                    client is not None and not client.is_closed()
                ):
                    raise SingleAttemptError("close_failed")
            except Exception:  # noqa: BLE001 - all close failures prohibit success
                raise SingleAttemptError("close_failed") from None
    result.metadata = {
        **(result.metadata or {}),
        RECEIPT_KEY: {
            "version": 1,
            "model": request.model,
            "reasoning_effort": request.reasoning_effort,
            "max_output_tokens": request.max_output_tokens,
            "timeout_seconds": request.timeout,
            "native_count_requests": 1,
            "generation_requests": 1,
            "native_input_tokens": tokens,
            "retries": 0,
            "continuations": 0,
            "closed": True,
            "input_sha256": _digest(params["input"]),
            "request_sha256": request_hash,
        },
    }
    return result
