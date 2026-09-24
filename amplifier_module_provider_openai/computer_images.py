"""Request-only compatibility for ordinary images alongside computer history.

The native computer API rejects multiple ordinary image inputs, even when its
tool is merely available. The mounted computer tool also has a function schema
and accepts the recorded native action batches. Use that transport for these
requests, retaining every image and action; canonical session history is never
changed and this module never executes a tool.
"""

from __future__ import annotations

import copy
import json
from typing import Any


def _unsupported(reason: str) -> None:
    from amplifier_core import llm_errors as errors

    error = errors.InvalidRequestError(
        "This request contains multiple images and native computer-use history "
        f"that cannot safely use the function transport: {reason}. "
        "All images and saved history are preserved; no tool was replayed.",
        provider="openai",
        retryable=False,
    )
    error.code = "computer_multi_image_incompatible"
    raise error


def _ordinary_images(items: list[dict[str, Any]]) -> int:
    count = 0
    for item in items:
        if item.get("type") in (None, "message"):
            blocks = item.get("content")
        elif item.get("type") in ("function_call_output", "custom_tool_call_output"):
            blocks = item.get("output")
        else:
            continue
        if isinstance(blocks, list):
            count += sum(
                isinstance(block, dict) and block.get("type") == "input_image"
                for block in blocks
            )
    return count


def prepare_computer_images(
    params: dict[str, Any],
    source_tools: list[Any],
    *,
    protected_computer_history: bool = False,
    function_lineage: bool = False,
    retained_function_call_ids: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Return compatible params without modifying caller-owned input or tools.

    Native compact windows and unresolved safety checks must not be rewritten.
    Explicit native-only choices remain explicit, rather than silently changing
    their meaning. A bare native declaration supplies no executable fallback.
    """
    items = params.get("input")
    tools = params.get("tools") or []
    if not isinstance(items, list) or not all(isinstance(item, dict) for item in items):
        return params
    has_native = any(tool.get("type") == "computer" for tool in tools)
    native_items = [
        item
        for item in items
        if item.get("type") in {"computer_call", "computer_call_output"}
    ]
    if not (has_native or native_items) or (
        not function_lineage and _ordinary_images(items) <= 1
    ):
        return params
    if protected_computer_history or (
        native_items and any(item.get("type") == "compaction" for item in items)
    ):
        _unsupported("the native compacted window contains computer items")
    if params.get("previous_response_id"):
        _unsupported("server-retained history is opaque")
    choice = params.get("tool_choice")
    if isinstance(choice, dict) and (
        choice.get("type") == "computer"
        or any(tool.get("type") == "computer" for tool in choice.get("tools", []))
    ):
        _unsupported("the caller explicitly requires the native computer tool")

    fallback = next(
        (
            tool
            for tool in tools
            if tool.get("type") == "function" and tool.get("name") == "computer"
        ),
        None,
    )
    if fallback is None:
        for tool in source_tools:
            if (
                getattr(tool, "type", None) == "computer"
                and getattr(tool, "name", None) == "computer"
                and isinstance(getattr(tool, "parameters", None), dict)
            ):
                strict = getattr(tool, "strict", False)
                fallback = {
                    "type": "function",
                    "name": "computer",
                    "description": tool.description,
                    "parameters": copy.deepcopy(tool.parameters),
                    "strict": strict if isinstance(strict, bool) else False,
                }
                break
    if fallback is None:
        _unsupported("no mounted computer function schema was supplied")

    # Validate all pairs before projecting any part of the request. A safety
    # approval or an interrupted execution is never manufactured here.
    calls: dict[str, dict[str, Any]] = {}
    outputs: dict[str, dict[str, Any]] = {}
    for item in native_items:
        identity = item.get("call_id")
        if not isinstance(identity, str) or not identity:
            _unsupported("a computer item has no call ID")
        if item["type"] == "computer_call":
            if set(item) - {
                "type",
                "call_id",
                "actions",
                "status",
                "pending_safety_checks",
            }:
                _unsupported("a computer call contains opaque or unsupported fields")
            if item.get("pending_safety_checks") or item.get("status") != "completed":
                _unsupported(
                    "a computer call has pending safety checks or is unfinished"
                )
            if not isinstance(item.get("actions"), list) or identity in calls:
                _unsupported("a computer action batch or call ID is ambiguous")
            calls[identity] = item
        else:
            if set(item) - {"type", "call_id", "output"} or identity in outputs:
                _unsupported("a computer result contains opaque or unsupported fields")
            screenshot = item.get("output")
            if (
                not isinstance(screenshot, dict)
                or screenshot.get("type") != "computer_screenshot"
                or set(screenshot) - {"type", "image_url", "detail"}
                or not isinstance(screenshot.get("image_url"), str)
            ):
                _unsupported("a computer result is not a recorded screenshot")
            outputs[identity] = item
    # A native transport delta may contain only a result for a function call
    # already retained by that same server lineage. The caller must derive
    # these IDs from its validated full request, never from opaque history.
    retained = retained_function_call_ids if function_lineage else frozenset()
    if calls.keys() - outputs.keys() or outputs.keys() - calls.keys() - retained:
        _unsupported("a computer call/result pair is incomplete")

    projected = copy.deepcopy(params)
    projected["tools"] = [
        copy.deepcopy(tool) for tool in tools if tool.get("type") != "computer"
    ]
    if not any(
        tool.get("type") == "function" and tool.get("name") == "computer"
        for tool in projected["tools"]
    ):
        projected["tools"].append(copy.deepcopy(fallback))
    for index, item in enumerate(projected["input"]):
        if item.get("type") == "computer_call":
            projected["input"][index] = {
                "type": "function_call",
                "name": "computer",
                "call_id": item["call_id"],
                "arguments": json.dumps({"actions": item["actions"]}),
                "status": "completed",
            }
        elif item.get("type") == "computer_call_output":
            screenshot = item["output"]
            image = {"type": "input_image", "image_url": screenshot["image_url"]}
            if "detail" in screenshot:
                image["detail"] = screenshot["detail"]
            projected["input"][index] = {
                "type": "function_call_output",
                "call_id": item["call_id"],
                "output": [image],
            }
    return projected
