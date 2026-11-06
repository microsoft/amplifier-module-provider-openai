"""Explicit, persisted Responses configuration changes, never an effort policy.

Version 1 deliberately accepts only reasoning effort. Hosts append the carrier
between responses, before the next user message, and persist it as a Core Message.
Request-level effort stays unchanged. API response.reasoning.effort still reports
that request setting, not this conversation setting.
"""

from __future__ import annotations

import copy
from typing import Any

KEY = "openai:configuration_update"
RESTATE_KEY = "openai:configuration_update_after_compaction"
REQUEST_KEY = "openai:request_configuration"


def request_configuration(params: dict[str, Any]) -> dict[str, Any]:
    """Record a narrow final-wire baseline, never the response's effort echo."""
    body = {**params, **(params.get("extra_body") or {})}
    reasoning = body.get("reasoning") or {}
    effort = reasoning.get("effort") if isinstance(reasoning, dict) else None
    return {"version": 1, "model": body.get("model"), "request_effort": effort}


def configuration_update_from_response(response, effort: str) -> dict[str, Any]:
    """Use the last completed response's persisted provider-recorded baseline."""
    metadata = (
        response.get("metadata") if isinstance(response, dict)
        else getattr(response, "metadata", None)
    ) or {}
    state = metadata.get(REQUEST_KEY)
    if (
        not isinstance(state, dict)
        or set(state) != {"version", "model", "request_effort"}
        or type(state["version"]) is not int
        or state["version"] != 1
    ):
        raise ValueError("No recorded request configuration; do not infer effort from UI or API response")
    return configuration_update_message(
        state["model"], effort, request_effort=state["request_effort"]
    )


def validate_item(item: Any) -> dict[str, Any]:
    """Accept one bounded, documented effort update, not arbitrary API settings."""
    if (
        not isinstance(item, dict)
        or set(item) != {"type", "reasoning"}
        or item["type"] != "configuration_update"
        or not isinstance(item["reasoning"], dict)
        or set(item["reasoning"]) != {"effort"}
        or not isinstance(item["reasoning"]["effort"], str)
        or item["reasoning"]["effort"]
        not in {"none", "minimal", "low", "medium", "high", "xhigh", "max"}
    ):
        raise ValueError("configuration_update v1 accepts only an explicit reasoning effort")
    return copy.deepcopy(item)


def configuration_update_message(
    model: str, effort: str, *, request_effort: str | None
) -> dict[str, Any]:
    """Build a carrier with the original wire effort, not the current UI choice.

    Pass None only if the original request omitted effort. The caller persists
    this message in position; the provider never changes global/operator config.
    """
    item = validate_item({"type": "configuration_update", "reasoning": {"effort": effort}})
    _validate_model_effort(model, effort)
    if request_effort is not None:
        _validate_model_effort(model, request_effort)
    return {
        "role": "user",
        "content": "",
        "metadata": {
            KEY: {
                "version": 1, "model": model, "request_effort": request_effort,
                "item": item,
            },
        },
    }


def message_item(message: dict[str, Any]) -> dict[str, Any] | None:
    """Read owned JSON; reject mixed carriers instead of dropping their content."""
    metadata = message.get("metadata") or {}
    if KEY not in metadata:
        return None
    state = metadata[KEY]
    if (
        not isinstance(state, dict)
        or set(state) != {"version", "model", "request_effort", "item"}
        or type(state["version"]) is not int
        or state["version"] != 1
        or not isinstance(state["model"], str)
        or message.get("role") != "user"
        or message.get("content") != ""
        or message.get("tool_call_id")
        or message.get("tool_calls")
        or "openai:compaction" in metadata
        or "openai:output_items" in metadata
    ):
        raise ValueError("Invalid persisted configuration_update carrier")
    return validate_item(state["item"])


def compaction_restatement(message: dict[str, Any]) -> dict[str, Any] | None:
    """A derived checkpoint keeps returned output and the fresh update separate."""
    metadata = message.get("metadata") or {}
    if RESTATE_KEY not in metadata:
        return None
    state = metadata[RESTATE_KEY]
    compact = metadata.get("openai:compaction")
    if (
        not isinstance(state, dict)
        or set(state) != {"version", "model", "request_effort", "item"}
        or type(state["version"]) is not int
        or state["version"] != 1
        or not isinstance(compact, dict)
        or compact.get("model") != state["model"]
        or KEY in metadata
    ):
        raise ValueError("Invalid post-compaction configuration_update restatement")
    return validate_item(state["item"])


def _validate_model_effort(model: str, effort: str) -> None:
    from ._capabilities import get_request_policy

    if not isinstance(model, str) or not isinstance(effort, str):
        raise TypeError("configuration_update requires a model and effort string")
    policy = get_request_policy(model)
    if not policy.supports_configuration_update:
        raise ValueError("configuration_update requires an exact supported GPT-6 model")
    if policy.allowed_efforts is None or effort not in policy.allowed_efforts:
        raise ValueError(f"configuration_update effort {effort!r} is not supported by {model}")


def validate_request(
    params: dict[str, Any], messages=(), *, original_input=None
) -> list[dict[str, Any]]:
    """Gate the final wire body after all overrides, including SDK extra_body.

    Return updates in wire order. No configured value is rewritten, and no
    effective API setting is inferred from an echoed request-level effort.
    """
    body = {**params, **(params.get("extra_body") or {})}
    items = body.get("input")
    updates = [
        validate_item(item)
        for item in items if isinstance(item, dict)
        and item.get("type") == "configuration_update"
    ] if isinstance(items, list) else []
    recorded = []
    baseline = request_configuration(body)["request_effort"]
    for message in messages:
        item = message_item(message) or compaction_restatement(message)
        if item is not None:
            state = message["metadata"].get(KEY) or message["metadata"][RESTATE_KEY]
            if state["model"] != body.get("model"):
                raise ValueError("configuration_update history belongs to a different model")
            reasoning = body.get("reasoning") or {}
            if (
                not isinstance(reasoning, dict)
                or state["request_effort"] != baseline
            ):
                raise ValueError(
                    "configuration_update must preserve the original request-level effort"
                )
            recorded.append(item)
    if recorded and updates != recorded:
        raise ValueError("Request overrides cannot replace recorded configuration_update history")
    if recorded and original_input is not None and items != original_input:
        raise ValueError("Request overrides cannot reposition configuration_update history")
    if not updates:
        return []
    model = body.get("model")
    for item in updates:
        _validate_model_effort(model, item["reasoning"]["effort"])
    reasoning = body.get("reasoning") or {}
    if not isinstance(reasoning, dict) or reasoning.get("mode", "standard") != "standard":
        raise ValueError("configuration_update requires standard single-agent reasoning")
    # Reject other reasoning extensions as well: unrecognized multi-agent
    # options cannot silently inherit a single-agent eligibility claim.
    if set(reasoning) - {"mode", "effort", "context", "summary", "generate_summary"}:
        raise ValueError("configuration_update does not support reasoning extensions")
    if "multi_agent" in body and body["multi_agent"] != {"enabled": False}:
        raise ValueError("configuration_update requires single-agent mode")
    if reasoning.get("effort") is not None:
        _validate_model_effort(model, reasoning["effort"])
    if body.get("truncation") not in (None, "disabled"):
        raise ValueError("configuration_update cannot use automatic truncation")
    if body.get("context_management"):
        raise ValueError("configuration_update cannot use automatic compaction")
    return updates


def reject_standalone(params: dict[str, Any]) -> None:
    if any(
        isinstance(item, dict) and item.get("type") == "configuration_update"
        for item in params.get("input", [])
    ):
        raise ValueError(
            "Standalone /responses/compact rejects configuration_update history; "
            "use explicit Responses compaction"
        )


def compacted_update_message(model, output, updates, *, request_effort):
    """Keep the opaque returned window intact and restate the last explicit effort."""
    from .compaction import compacted_message

    message = compacted_message(model, output)
    message["metadata"][RESTATE_KEY] = {
        "version": 1,
        "model": model,
        "request_effort": request_effort,
        "item": validate_item(updates[-1]),
    }
    return message