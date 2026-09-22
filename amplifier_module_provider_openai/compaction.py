"""Lossless transport for the canonical window returned by Responses compact.

The output may contain retained user/tool items as well as encrypted state.
Never decode, summarize, normalize, filter, or prune that returned window.
"""

import copy

KEY = "openai:compaction"


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


def compacted_message(model, output):
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
            KEY: {"version": 1, "model": model, "output": copy.deepcopy(output)},
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
