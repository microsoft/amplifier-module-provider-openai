"""Generation safety policy; read-only calls keep their existing retry policy."""

import httpcore
import httpx
import openai
from amplifier_core import llm_errors

OUTCOME_UNKNOWN_MESSAGE = (
    "Provider wait ended without a confirmed result. The request may have been "
    "accepted; no automatic replacement request was sent."
)


class RequestOutcomeUnknownError(llm_errors.LLMError):
    """Local settlement is not evidence of provider-side rollback."""

    request_outcome = "unknown"
    effects = "may_have_occurred"

    def __init__(self, *, provider, status_code=None):
        super().__init__(
            OUTCOME_UNKNOWN_MESSAGE, provider=provider,
            status_code=status_code, retryable=False,
        )


class InjectedClientConfigurationError(llm_errors.InvalidRequestError):
    """An unsafe injected SDK client is refused locally before dispatch."""

    request_outcome = "not_dispatched"
    effects = "none"

    def __init__(self, *, provider):
        super().__init__(
            "Injected OpenAI SDK client must support disabling SDK retries.",
            provider=provider, retryable=False,
        )


class LocalRequestError(llm_errors.LLMError):
    """A known local phase failure never authorizes generation replay."""

    def __init__(self, *, provider, received=False):
        self.request_outcome = "received" if received else "not_dispatched"
        self.effects = "occurred" if received else "none"
        super().__init__(
            "Provider response was received; local processing failed."
            if received else "Local request preparation failed before generation dispatch.",
            provider=provider, retryable=False,
        )


def quota_refusal(*, provider):
    error = llm_errors.RateLimitError(
        "Provider refused this request because billing quota is unavailable.",
        provider=provider, status_code=429, retryable=False,
    )
    error.request_outcome = "not_accepted"
    error.effects = "none"
    error.vendor_code = "insufficient_quota"
    return error


def unknown_timeout(*, provider):
    error = llm_errors.LLMTimeoutError(
        OUTCOME_UNKNOWN_MESSAGE, provider=provider, retryable=False,
    )
    error.request_outcome = "unknown"
    error.effects = "may_have_occurred"
    return error


def proven_before_send(error):
    """Only the inspected SDK -> httpx -> httpcore setup-phase mapping.

    httpcore connects/acquires a pool connection before sending HTTP headers.
    httpx maps these exact types with ``raise ... from exc``; the SDK preserves
    that cause when wrapping. A generic wrapper, subclass, message, or missing
    chain proves nothing. Callers must also exclude prior response activity.
    """
    if type(error) not in (openai.APIConnectionError, openai.APITimeoutError):
        return False
    phase = error.__cause__
    pairs = {
        httpx.ConnectError: httpcore.ConnectError,
        httpx.ConnectTimeout: httpcore.ConnectTimeout,
        httpx.PoolTimeout: httpcore.PoolTimeout,
    }
    expected = pairs.get(type(phase))
    if expected is None:
        return False
    try:
        same_request = phase.request is error.request
    except RuntimeError:
        return False
    # The SDK wrapper retains the original request; httpx attaches the failing
    # hop to its phase error. A different request means a redirect could have
    # submitted a previous POST. Missing identity is also insufficient proof.
    return (
        type(phase.__cause__) is expected and same_request
    )


def proven_rate_refusal(error):
    """Documented pre-response rate admission refusal, not any arbitrary 429.

    https://developers.openai.com/api/docs/guides/error-codes
    Rate admission errors permit bounded backoff; billing/quota errors and
    unstructured proxy responses do not establish this contract.
    """
    if (type(error) is not openai.RateLimitError or error.status_code != 429
            or error.response.history):
        return False
    body = error.body
    if not isinstance(body, dict):
        return False
    body = body.get("error", body)
    return isinstance(body, dict) and (
        body.get("code") == "rate_limit_exceeded"
        or (body.get("type") == "rate_limit_error" and body.get("code") == "slow_down")
    )


def proven_quota_refusal(error):
    """Legacy structured billing refusal, separate from rate-backoff proof.

    Only the exact HTTP 429 SDK error and documented insufficient_quota code
    qualify, never message text or error.type alone. SDK 2.9+ preserves the
    structured body (unwrapping its error envelope). The caller must exclude
    prior parsed activity; redirected refusals cannot prove nonacceptance.
    """
    if (type(error) is not openai.RateLimitError or error.status_code != 429
            or error.response.history):
        return False
    body = error.body
    if not isinstance(body, dict):
        return False
    body = body.get("error", body)
    return isinstance(body, dict) and body.get("code") == "insufficient_quota"