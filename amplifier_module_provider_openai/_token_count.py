"""Bounded retries for counting only; never retry model generation here."""

import asyncio
import math
import re

import openai
from amplifier_core.llm_errors import LLMError

MAX_ATTEMPTS = 3
ATTEMPT_TIMEOUT = 30
MAX_RETRY_DELAY = 10


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
