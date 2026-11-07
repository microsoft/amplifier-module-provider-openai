"""Canonical Responses output conversion and safe stateless replay.

Provider instances own pricing and native-call state. This module owns the single
SDK/dict conversion, ordered wire representation and continuation safety checks.
"""

import copy
import json
import logging
from types import SimpleNamespace
from typing import Any

from amplifier_core import TextContent, ThinkingContent, ToolCallContent
from amplifier_core import llm_errors as kernel_errors
from amplifier_core.message_models import (
    TextBlock,
    ThinkingBlock,
    ToolCall,
    ToolCallBlock,
    Usage,
)

from ._constants import (
    METADATA_CONTINUATION_COUNT,
    METADATA_INCOMPLETE_REASON,
    METADATA_REASONING_ITEMS,
    METADATA_RESPONSE_ID,
    METADATA_STATUS,
    METADATA_TOOL_CALL_NAMESPACES,
    METADATA_TOOL_SEARCH_ITEMS,
)
from ._tool_search import (
    extract_function_call_namespaces,
    extract_hosted_tool_search_items,
)

logger = logging.getLogger(__name__)
METADATA_OUTPUT_ITEMS = "openai:output_items"


def field(value: Any, name: str, default: Any = None) -> Any:
    """Read a Responses field from either a wire dict or an SDK object."""
    return (
        value.get(name, default)
        if isinstance(value, dict)
        else getattr(value, name, default)
    )


def _wire_value(value: Any) -> Any:
    """Own wire values without inventing SDK defaults or dropping explicit nulls."""
    if hasattr(value, "model_dump"):
        dumped = value.model_dump(mode="json", exclude_unset=True, warnings=False)
        if not isinstance(dumped, (dict, list)):
            raise kernel_errors.LLMError(
                "Response output model_dump did not return JSON items", retryable=False
            )
        return _wire_value(dumped)
    if isinstance(value, dict):
        return {key: _wire_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_wire_value(item) for item in value]
    if isinstance(value, SimpleNamespace):
        return _wire_value(vars(value))
    return copy.deepcopy(value)


def response_items(output: list[Any]) -> list[dict[str, Any]]:
    """Own ordered output, including opaque fields, IDs, namespaces and phase.

    SDKs add unset defaults and local parsing annotations which are not API
    input fields. Exclude those while retaining explicit wire nulls. Normalize
    saved dicts too, so histories captured by older versions remain resumable.
    """
    items = [_wire_value(item) for item in output]
    if any(
        not isinstance(item, dict) or not isinstance(item.get("type"), str)
        for item in items
    ):
        raise kernel_errors.LLMError(
            "Response output contains a non-item value", retryable=False
        )
    for item in items:
        if item["type"] == "function_call":
            item.pop("parsed_arguments", None)
        elif item["type"] == "message":
            for part in item.get("content", []) or []:
                if isinstance(part, dict) and part.get("type") == "output_text":
                    part.pop("parsed", None)
    # Do not recursively strip these names: tool arguments and opaque fields
    # may legitimately use them. All changes above affect our owned copy only.
    return items


class FunctionCallTruncationError(kernel_errors.LLMError):
    """Truncated call arguments must never become executable empty arguments."""

    def __init__(
        self, message: str, *, provider: str | None = None, model: str | None = None
    ) -> None:
        super().__init__(message, provider=provider, model=model, retryable=False)


class UnsafeContinuationError(kernel_errors.LLMError):
    """Output cannot be replayed automatically without inventing execution."""

    def __init__(self, message: str) -> None:
        super().__init__(message, retryable=False)


def _function_call_fields(block: Any) -> tuple[str | None, str | None, str, Any, Any]:
    arguments = field(block, "input")
    if arguments is None:
        arguments = field(block, "arguments")
    return (
        field(block, "call_id"),
        field(block, "id"),
        field(block, "name", ""),
        arguments,
        field(block, "status"),
    )


def describe_incomplete_function_calls(output_items: list[Any]) -> list[dict[str, Any]]:
    """Detect truncated calls before the response-level raised-budget retry."""
    problems = []
    for block in output_items:
        if field(block, "type") not in {"tool_call", "function_call"}:
            continue
        call_id, item_id, name, arguments, status = _function_call_fields(block)
        reason = "status_incomplete" if status == "incomplete" else None
        if not reason and isinstance(arguments, str) and arguments.strip():
            try:
                json.loads(arguments)
            except json.JSONDecodeError:
                reason = "arguments_unparseable"
        if reason:
            problems.append(
                {
                    "reason": reason,
                    "call_id": call_id,
                    "item_id": item_id,
                    "name": name,
                    "arguments_len": len(arguments)
                    if isinstance(arguments, str)
                    else None,
                }
            )
    return problems


def parse_function_call_block(block: Any) -> tuple[str, str, dict[str, Any]]:
    """Parse calls by call_id; retain historical completed-malformed recovery.

    A completed malformed call remains a loudly logged tool-error-loop case.
    Incomplete responses are separately validated before conversion, including
    malformed calls which the endpoint failed to label incomplete.
    """
    call_id, item_id, name, arguments, status = _function_call_fields(block)
    if status == "incomplete":
        raise FunctionCallTruncationError(
            f"function_call '{name}' (call_id={call_id or item_id}) has status "
            "'incomplete' — its arguments were truncated by max_output_tokens and it cannot be executed"
        )
    tool_input = arguments
    if isinstance(tool_input, str):
        if not tool_input.strip():
            tool_input = {}
        else:
            try:
                tool_input = json.loads(tool_input)
            except json.JSONDecodeError:
                logger.warning(
                    "[PROVIDER] function_call '%s' (call_id=%s, status=%s) carries "
                    "unparseable JSON arguments (%d chars) on a non-incomplete call — "
                    "coercing to {} so the tool error surfaces to the model instead of killing the session. "
                    "Argument prefix: %.120s",
                    name,
                    call_id or item_id,
                    status,
                    len(arguments),
                    arguments,
                )
                tool_input = {}
    return (
        call_id or item_id or "",
        name,
        tool_input if isinstance(tool_input, dict) else {},
    )


def replay_output_items(
    output: list[Any], *, continuation: bool = False
) -> list[dict[str, Any]]:
    """Preserve history; a continuation additionally refuses outstanding host calls."""
    items = response_items(output)
    problems = describe_incomplete_function_calls(items)
    if any(
        continuation or problem["reason"] == "status_incomplete" for problem in problems
    ):
        raise FunctionCallTruncationError(
            "Cannot replay truncated function-call arguments"
        )
    for item in items:
        if continuation and item["type"] in {
            "function_call",
            "tool_call",
            "custom_tool_call",
            "local_shell_call",
            "shell_call",
            "apply_patch_call",
            "computer_call",
        }:
            raise UnsafeContinuationError(
                f"Cannot auto-continue {item['type']} (call_id={item.get('call_id')}): "
                "a real host tool result is required"
            )
        if (
            continuation
            and item["type"] == "tool_search_call"
            and item.get("execution") == "client"
        ):
            raise UnsafeContinuationError(
                "Cannot auto-continue client tool search without a real host result"
            )
    return items


class _ResponseOutputView:
    """Delegate response fields without SDK/raw-wrapper copy protocols."""

    def __init__(self, response: Any, output: list[Any]) -> None:
        self._response = response
        self.output = output

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    def output_text(self) -> str:
        return "".join(
            field(part, "text", "")
            for item in self.output
            if field(item, "type") == "message"
            for part in (field(item, "content", []) or [])
            if field(part, "type") == "output_text"
        )


def response_with_output(response: Any, output: list[Any]) -> Any:
    """Create a private response view without changing the SDK response."""
    if isinstance(response, dict):
        return {
            **response,
            "output": output,
            "output_text": _ResponseOutputView(response, output).output_text,
        }
    return _ResponseOutputView(response, output)


def normalized_usage(usage_obj: Any) -> Usage:
    """Normalize gross input once: cache writes are a subset, not extra."""
    input_details = field(usage_obj, "input_tokens_details")
    output_details = field(usage_obj, "output_tokens_details")
    cache_write = field(input_details, "cache_write_tokens")
    fresh_and_read = max(
        0, (field(usage_obj, "input_tokens", 0) or 0) - (cache_write or 0)
    )
    output = field(usage_obj, "output_tokens", 0) or 0
    return Usage(
        input_tokens=fresh_and_read,
        output_tokens=output,
        total_tokens=fresh_and_read + output,
        reasoning_tokens=field(output_details, "reasoning_tokens"),
        cache_read_tokens=field(input_details, "cached_tokens"),
        cache_write_tokens=cache_write,
    )


def merge_discarded_usage(usage: Usage, discarded_usage_objs: list[Any]) -> Usage:
    """Include every billed attempt, normalizing cache subsets once per attempt."""
    updates = {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "reasoning_tokens": usage.reasoning_tokens,
        "cache_read_tokens": usage.cache_read_tokens,
        "cache_write_tokens": usage.cache_write_tokens,
    }
    for obj in discarded_usage_objs:
        if obj is None:
            continue
        attempt = normalized_usage(obj)
        for name, value in updates.items():
            addition = getattr(attempt, name)
            if addition is not None:
                updates[name] = (value or 0) + addition
    updates["total_tokens"] = updates["input_tokens"] + updates["output_tokens"]
    return usage.model_copy(update=updates)


def extract_reasoning_text(reasoning_summary: Any) -> str | None:
    """Extract summaries without exposing encrypted state as display text."""
    if isinstance(reasoning_summary, list):
        text = "\n".join(
            filter(
                None,
                (
                    item if isinstance(item, str) else field(item, "text", "")
                    for item in reasoning_summary
                ),
            )
        )
    elif isinstance(reasoning_summary, str):
        text = reasoning_summary
    elif isinstance(reasoning_summary, dict) or hasattr(reasoning_summary, "text"):
        text = field(reasoning_summary, "text", str(reasoning_summary))
    else:
        text = None
    return text or None


def convert_response(
    response: Any,
    chat_response_class: type,
    *,
    native_call_ids: set[str] | None = None,
    native_call_types: dict[str, str] | None = None,
    computer_context: dict[str, Any] | None = None,
) -> Any:
    """The single ordinary/continued/native conversion, independent of pricing.

    State sets are request-local when called by completion. Native computer
    validation happens for the whole batch before returning executable calls.
    Unknown output items are preserved in metadata, not presented as host tools.
    """
    from .neutral_computer import CALLS, IMAGES, KEY, response_calls

    items = response_items(list(field(response, "output", []) or []))
    if field(response, "status") == "incomplete" and describe_incomplete_function_calls(
        items
    ):
        raise FunctionCallTruncationError(
            "Incomplete response contains truncated function-call arguments"
        )
    neutral_calls = (
        response_calls(
            items, computer_context, field(response, "model"), field(response, "status")
        )
        if computer_context is not None
        else {}
    )
    staged_ids, staged_types = set(), {}
    content, calls, events, texts, reasoning_ids = [], [], [], [], []
    for block in items:
        kind = block["type"]
        if kind == "message":
            parts = block.get("content", [])
            if isinstance(parts, str):
                parts = [{"type": "output_text", "text": parts}]
            for part in parts:
                if part.get("type") == "output_text":
                    text = part.get("text", "")
                    content.append(TextBlock(text=text))
                    events.append(TextContent(text=text, raw=part))
                    texts.append(text)
        elif kind == "reasoning":
            item_id, encrypted = block.get("id"), block.get("encrypted_content")
            if item_id:
                reasoning_ids.append(item_id)
            text = extract_reasoning_text(block.get("summary") or block.get("text"))
            if text or encrypted:
                content.append(
                    ThinkingBlock(
                        thinking=text or "",
                        signature=None,
                        visibility="internal",
                        content=[
                            {
                                "encrypted_content": encrypted,
                                "id": item_id,
                                "summary": text or None,
                            }
                        ],
                    )
                )
                events.append(ThinkingContent(text=text or ""))
        elif kind in {
            "tool_call",
            "function_call",
            "apply_patch_call",
            "computer_call",
        }:
            call_id, name, args, extra = "", "", {}, {}
            if kind in {"tool_call", "function_call"}:
                call_id, name, args = parse_function_call_block(block)
            else:
                call_id = block.get("call_id", "")
                if block.get("status") in {"incomplete", "in_progress"}:
                    raise UnsafeContinuationError(
                        f"Cannot execute incomplete {kind} (call_id={call_id})"
                    )
                if kind == "apply_patch_call":
                    name = "apply_patch"
                    operation = block.get("operation") or {}
                    args = {
                        key: operation.get(key, "") for key in ("type", "path", "diff")
                    }
                else:
                    record = neutral_calls.get(call_id)
                    name = record["alias"] if record else "computer"
                    if record:
                        args, extra = record["canonical"], {"metadata": {KEY: record}}
                    else:
                        actions = block.get("actions")
                        args = {
                            "actions": actions
                            if actions is not None
                            else ([block["action"]] if block.get("action") else [])
                        }
                        staged_types[call_id] = "computer"
                if (
                    kind == "apply_patch_call" or call_id not in neutral_calls
                ) and native_call_ids is not None:
                    staged_ids.add(call_id)
            content.append(ToolCallBlock(id=call_id, name=name, input=args, **extra))
            calls.append(ToolCall(id=call_id, name=name, arguments=args, **extra))
            events.append(
                ToolCallContent(id=call_id, name=name, arguments=args, raw=block)
            )

    metadata = {METADATA_OUTPUT_ITEMS: items}
    for key, source in ((METADATA_RESPONSE_ID, "id"), (METADATA_STATUS, "status")):
        if field(response, source) is not None:
            metadata[key] = field(response, source)
    if field(response, "status") == "incomplete":
        metadata[METADATA_INCOMPLETE_REASON] = field(
            field(response, "incomplete_details"), "reason"
        )
    if reasoning_ids:
        metadata[METADATA_REASONING_ITEMS] = reasoning_ids
    hosted = extract_hosted_tool_search_items(items)
    if hosted:
        metadata[METADATA_TOOL_SEARCH_ITEMS] = hosted
    namespaces = extract_function_call_namespaces(items)
    if namespaces:
        metadata[METADATA_TOOL_CALL_NAMESPACES] = namespaces
    if neutral_calls:
        metadata[CALLS] = copy.deepcopy(neutral_calls)
    if (
        computer_context
        and computer_context.get("observations")
        and field(response, "status") == "completed"
    ):
        metadata[IMAGES] = copy.deepcopy(computer_context["observations"])
    result = chat_response_class(
        content=content,
        tool_calls=calls or None,
        usage=normalized_usage(field(response, "usage")),
        finish_reason="length"
        if field(response, "status") == "incomplete"
        else field(response, "finish_reason"),
        content_blocks=events or None,
        text="\n\n".join(texts).strip() or None,
        output_text="".join(texts),
        metadata=metadata,
    )
    if native_call_ids is not None:
        native_call_ids.update(staged_ids)
    if native_call_types is not None:
        native_call_types.update(staged_types)
    return result


def convert_response_with_accumulated_output(
    final_response: Any,
    accumulated_output: list[Any],
    continuation_count: int,
    chat_response_class: type,
) -> Any:
    """Compatibility entry point; uses the same conversion as ordinary output.

    Provider completion calls its instance hook instead, preserving subclass
    overrides and request-local native tracking/endpoint-specific pricing.
    """
    result = convert_response(
        response_with_output(final_response, accumulated_output), chat_response_class
    )
    if continuation_count > 0:
        result.metadata[METADATA_CONTINUATION_COUNT] = continuation_count
    return result
