"""Bounded retries for counting only; never retry model generation here."""

import asyncio
import hashlib
import hmac
import json
import math
import re
import secrets
import uuid
from functools import lru_cache
from pathlib import Path

import openai
from amplifier_core.llm_errors import LLMError

MAX_ATTEMPTS = 3
ATTEMPT_TIMEOUT = 30
MAX_RETRY_DELAY = 10


class CountedInput(int):
    """Request-local count receipt; remains numeric for existing guard callers."""

    def __new__(cls, value, attribution=None):
        result = super().__new__(cls, value)
        result.attribution = attribution
        return result


@lru_cache(maxsize=1)
def source_provenance(provider_file):
    """Observe source files once per worker, not a claim about imported bytes."""
    result = {"imported_bytes_verified": False}
    for label, path in (("provider_source_sha256", provider_file),
                        ("count_source_sha256", __file__)):
        try:
            result[label] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        except OSError:
            result[label] = None
    return result


class RequestAttribution:
    """Content-free JSON identity within one provider instance, not across runs.

    A raw/publicly salted digest permits guessing low-entropy prompts. This key
    stays in memory and is never exported. Tags compare canonical JSON values,
    not HTTP byte order, vendor tokenization, or semantic equivalence.
    """

    TYPES = frozenset({
        "message", "function_call", "function_call_output", "compaction",
        "reasoning", "computer_call", "computer_call_output",
        "apply_patch_call", "apply_patch_call_output", "item_reference",
    })

    def __init__(self):
        self._key = secrets.token_bytes(32)
        self.scope_id = uuid.uuid4().hex

    @staticmethod
    def _encoded(value):
        return json.dumps(value, sort_keys=True, ensure_ascii=False,
                          separators=(",", ":"), allow_nan=False).encode("utf-8")

    def tag(self, params):
        if not isinstance(params, dict):
            return None
        try:
            return hmac.new(self._key, b"responses-input-v1\0" + self._encoded(params),
                            hashlib.sha256).hexdigest()
        except (ValueError, TypeError, UnicodeError, OverflowError, RecursionError):
            return None

    def describe(self, params, fields=None):
        """Bounded metadata only. Unsupported JSON yields unknown attribution."""
        if not isinstance(params, dict):
            return None
        try:
            # Serialize each component once and compose canonical JSON, so the
            # full envelope and compact subset reuse the large input encoding.
            fields = fields if fields is not None else {k: self._encoded(v) for k, v in params.items()}
            encoded = b"{" + b",".join(self._encoded(k) + b":" + fields[k] for k in sorted(params)) + b"}"
            items = params.get("input", [])
            tools = params.get("tools", [])
            types = {}
            for item in items if isinstance(items, list) else []:
                kind = item.get("type", "message") if isinstance(item, dict) else "other"
                kind = kind if isinstance(kind, str) and kind in self.TYPES else "other"
                types[kind] = types.get(kind, 0) + 1
            return {
                "payload_tag": hmac.new(self._key, b"responses-input-v1\0" + encoded,
                                        hashlib.sha256).hexdigest(),
                "serialized_bytes": len(encoded),
                "input_items": len(items) if isinstance(items, list) else 1,
                "input_types": types,
                "input_bytes": len(fields["input"]) if "input" in params else 2,
                "instructions_bytes": len(fields["instructions"]) if "instructions" in params else 4,
                "tools": len(tools) if isinstance(tools, list) else 0,
                "tools_bytes": len(fields["tools"]) if "tools" in params else 2,
            }
        except (ValueError, TypeError, UnicodeError, OverflowError, RecursionError):
            return None

    def receipt(self, params):
        try:
            fields = {k: self._encoded(v) for k, v in params.items()}
        except (ValueError, TypeError, UnicodeError, OverflowError, RecursionError):
            return None
        request = self.describe(params, fields)
        if request is None:
            return None
        return {
            "version": 1, "scope_id": self.scope_id, "count_id": uuid.uuid4().hex,
            "request": request,
            # Compact dispatch omits tools/options that the full preflight counts.
            "compact_input": self.describe({k: params[k] for k in
                                           ("model", "input", "instructions") if k in params}, fields),
        }


class TokenCountError(LLMError):
    def __init__(self, diagnostic):
        self.count_failure = diagnostic
        # The count operation has already exhausted its own retry policy.
        # An outer generation retry must not multiply or replay that work.
        super().__init__("Could not check conversation size.", provider="openai", retryable=False)


def failure(error):
    status = getattr(error, "status_code", None)
    status = status if type(status) is int and 400 <= status <= 599 else None
    category, retryable = "invalid_response", False
    if isinstance(error, (TimeoutError, openai.APITimeoutError)) or status == 408:
        category, retryable = "timeout", True
    elif isinstance(error, openai.APIConnectionError):
        category, retryable = "connection", True
    elif status == 429:
        category, retryable = "rate_limit", True
        if getattr(error, "code", None) in {"insufficient_quota", "billing_hard_limit_reached"}:
            category, retryable = "quota", False
    elif status is not None and status >= 500:
        category, retryable = "service", True
    elif status in (401, 403):
        category = "authentication" if status == 401 else "permission"
    elif status is not None:
        category = "invalid_request"
    result = {"category": category, "retryable": retryable}
    if status is not None:
        result["httpStatus"] = status
    request_id = getattr(error, "request_id", None)
    if isinstance(request_id, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,100}", request_id):
        result["requestId"] = request_id
    return result


def retry_delay(error, attempt):
    headers = getattr(getattr(error, "response", None), "headers", {})
    value = headers.get("retry-after-ms")
    seconds = headers.get("retry-after")
    if value is None and seconds is None:
        return float(2 ** (attempt - 1))
    try:
        delay = float(value) / 1000 if value is not None else float(seconds)
    except (ValueError, TypeError):
        # Unknown/date-form server delays must not be retried prematurely.
        return None
    return max(float(2 ** (attempt - 1)), delay) if math.isfinite(delay) and 0 <= delay <= MAX_RETRY_DELAY else None


async def count_tokens(client, params):
    # Disable SDK retries even for injected clients; our attempt budget is the
    # complete wire budget. The copy shares transport and is not ours to close.
    if isinstance(client, openai.AsyncOpenAI):
        client = client.with_options(max_retries=0, timeout=ATTEMPT_TIMEOUT)
    counter = client.responses.input_tokens.count
    for attempt in range(1, MAX_ATTEMPTS + 1):
        delay = None
        try:
            async with asyncio.timeout(ATTEMPT_TIMEOUT):
                result = await counter(**params)
            tokens = getattr(result, "input_tokens", None)
            if type(tokens) is not int or tokens < 0:
                raise ValueError("Invalid count response")
            return tokens
        except asyncio.CancelledError:
            raise
        except (openai.APIError, TimeoutError, RuntimeError, TypeError, ValueError) as error:
            diagnostic = {**failure(error), "attempts": attempt}
            if diagnostic["retryable"] and attempt < MAX_ATTEMPTS:
                delay = retry_delay(error, attempt)
        # Exit the exception handler before sleeping/raising so the public
        # exception does not retain an SDK body through implicit chaining.
        if delay is None:
            raise TokenCountError(diagnostic) from None
        await asyncio.sleep(delay)
