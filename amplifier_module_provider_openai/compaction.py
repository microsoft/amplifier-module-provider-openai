"""Lossless transport for the canonical window returned by Responses compact.

The output may contain retained user/tool items as well as encrypted state.
Never decode, summarize, normalize, filter, or prune that returned window.
"""

import copy
import hmac

import openai
from amplifier_core.llm_errors import LLMError

from ._chat_images import media_values

KEY = "openai:compaction"
QUALIFIED_SWITCH = frozenset({"gpt-6.2-astra", "gpt-6.2-sol"})
SOL_UPGRADE = ("gpt-6.1-sol", "gpt-6.2-sol")


def operation_params(assembled, messages):
    """Project the exact standalone or explicit-trigger request for both count and send."""
    from .configuration_updates import validate_request

    updates = validate_request(assembled, [message.model_dump() for message in messages])
    if not updates:
        return {key: assembled[key] for key in ("model", "input", "instructions") if key in assembled}, []
    body = {**assembled, **(assembled.get("extra_body") or {})}
    params = {key: body[key] for key in ("model", "instructions", "reasoning") if key in body}
    params["input"] = [*body["input"], {"type": "compaction_trigger"}]
    params["store"] = False
    return params, updates


async def submit_compaction(operation, params, *, timeout):
    """Keep the SDK's image-bearing HTTP objects out of public compaction errors."""
    try:
        return await operation(**params, timeout=timeout)
    except openai.APIError as error:
        if not media_values(params):
            raise
        public_error = LLMError(
            f"OpenAI native compaction failed ({type(error).__name__}).",
            provider="openai", model=params["model"],
            status_code=getattr(error, "status_code", None), retryable=False,
        )
    # The submitted request may already have been processed. Drop the SDK
    # exception chain and leave any further attempt to the caller's admission.
    raise public_error


def canonical_usage(usage):
    """Map native gross usage to Core's disjoint cache-write convention.

    No cost is invented when the compact response does not provide one.
    """
    if not isinstance(usage, dict):
        return {}
    result = {
        key: value
        for key, value in usage.items()
        if key in {"input_tokens", "output_tokens", "total_tokens"}
        and type(value) is int
        and value >= 0
    }
    details = usage.get("input_tokens_details") or {}
    for vendor, public in (
        ("cached_tokens", "cache_read_tokens"),
        ("cache_write_tokens", "cache_write_tokens"),
    ):
        value = details.get(vendor)
        if type(value) is int and value >= 0:
            result[public] = value
    if "input_tokens" in result:
        result["input_tokens"] = max(
            0, result["input_tokens"] - result.get("cache_write_tokens", 0)
        )
    if "total_tokens" in result:
        result["total_tokens"] = max(
            0, result["total_tokens"] - result.get("cache_write_tokens", 0)
        )
    return result


def compacted_message(model, output, *, scope=None):
    if (
        not isinstance(output, list)
        or not output
        or not all(isinstance(item, dict) for item in output)
    ):
        raise ValueError("Native compaction did not return a nonempty canonical window")
    if not any(
        item.get("type") == "compaction" and item.get("encrypted_content")
        for item in output
    ):
        raise ValueError(
            "Native compaction did not return an encrypted compaction item"
        )
    return {
        "role": "user",
        "content": "Native compacted conversation (reference state; originals remain in the transcript).",
        "metadata": {
            "ephemeral": True,
            "persisted": True,
            "source": "context-managed",
            KEY: {"version": 1, "model": model, "output": copy.deepcopy(output),
                  **({"scope": scope} if scope else {})},
        },
    }


def canonical_window(message, model):
    state = (message.get("metadata") or {}).get(KEY)
    if state is None:
        return None
    if (
        not isinstance(state, dict)
        or state.get("version") != 1
        or state.get("model") != model
    ):
        raise ValueError(
            "Native compacted context belongs to a different model or unsupported format"
        )
    # Reuse the structural validation without interpreting the opaque payload.
    return compacted_message(model, state.get("output"))["metadata"][KEY]["output"]


def matches_scope(state, scope):
    stored = state.get("scope")
    return bool(scope and isinstance(stored, str) and hmac.compare_digest(stored, scope))


def rebind_window(message, source_model, model, scope):
    if not scope or ({source_model, model} != QUALIFIED_SWITCH and (source_model, model) != SOL_UPGRADE):
        return None
    state = (message.get("metadata") or {}).get(KEY)
    if not isinstance(state, dict) or not matches_scope(state, scope):
        return None
    window = canonical_window(message, source_model)
    return compacted_message(model, window, scope=scope)
