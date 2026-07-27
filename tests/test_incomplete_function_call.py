"""P4/P5 regression tests: truncated function_call handling + capability-based output budget.

Reproduces the bug shape observed live (parity baseline, pilot forensics Q3):
a response hits max_output_tokens mid-function_call (status "incomplete",
arguments truncated), the provider auto-continues fruitlessly, and the
stitched output surfaced `{}`-argument tool calls keyed by Responses-API
item ids (`fc_…`) instead of call ids (`call_…`) — unpairable outputs, then
a 400 killed the session.

Invariants asserted here (non-negotiable):
- no `{}`-argument call ever surfaces from a truncated/unparseable function_call
- surfaced tool-call ids are call_-keyed (call_id preferred over item id)
- incomplete-after-budget fails loud (FunctionCallTruncationError), never silent
- the continuation policy retries once with the budget raised to the model cap
  before failing
- default output budget comes from the model's capability table, not a fixed 4096
"""

import asyncio
import json
from pathlib import Path

import pytest
from amplifier_core.message_models import ChatRequest, Message
from openai.types.responses import Response

from amplifier_module_provider_openai import OpenAIProvider
from amplifier_module_provider_openai._capabilities import get_capabilities
from amplifier_module_provider_openai._response_handling import (
    FunctionCallTruncationError,
    convert_response_with_accumulated_output,
    describe_incomplete_function_calls,
    parse_function_call_block,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "responses"

TRUNCATED_ARGS = '{"file_path":"/workspace/graspologic/DISCREPANCIES.md","content":"# Documentation Discr'


def _base_payload() -> dict:
    with open(FIXTURES_DIR / "gpt-5-5-basic.json") as f:
        return json.load(f)


def _function_call_item(
    *,
    item_id: str = "fc_0d1fafae000000",
    call_id: str = "call_Ohhub2jBM8HU07mYExW9jDvD",
    name: str = "write_file",
    arguments: str = TRUNCATED_ARGS,
    status: str = "incomplete",
) -> dict:
    return {
        "type": "function_call",
        "id": item_id,
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
        "status": status,
    }


def _make_response(
    *,
    status: str = "completed",
    output: list | None = None,
    incomplete_reason: str | None = None,
    max_output_tokens: int | None = 4096,
) -> Response:
    payload = _base_payload()
    payload["status"] = status
    payload["max_output_tokens"] = max_output_tokens
    payload["incomplete_details"] = (
        {"reason": incomplete_reason} if incomplete_reason else None
    )
    if output is not None:
        payload["output"] = output
    return Response.model_validate(payload)


def _make_provider(config: dict | None = None) -> OpenAIProvider:
    return OpenAIProvider(
        api_key="test-key",
        config={"max_retries": 0, "use_streaming": False, **(config or {})},
    )


def _sequential_mock(responses: list[Response], captured: list[dict]):
    async def _mock(**kwargs):
        captured.append(kwargs)
        return responses[min(len(captured) - 1, len(responses) - 1)]

    return _mock


# ---------------------------------------------------------------------------
# parse_function_call_block — the converter contract
# ---------------------------------------------------------------------------


def test_parse_prefers_call_id_over_item_id():
    tool_id, name, args = parse_function_call_block(
        _function_call_item(status="completed", arguments='{"file_path": "/tmp/x"}')
    )
    assert tool_id == "call_Ohhub2jBM8HU07mYExW9jDvD"
    assert tool_id.startswith("call_")
    assert name == "write_file"
    assert args == {"file_path": "/tmp/x"}


def test_parse_incomplete_status_raises_never_surfaces_empty_args():
    with pytest.raises(FunctionCallTruncationError):
        parse_function_call_block(_function_call_item(status="incomplete"))


def test_parse_unparseable_arguments_raises():
    with pytest.raises(FunctionCallTruncationError):
        parse_function_call_block(
            _function_call_item(status="completed", arguments=TRUNCATED_ARGS)
        )


def test_parse_empty_arguments_is_a_legitimate_no_arg_call():
    tool_id, _, args = parse_function_call_block(
        _function_call_item(status="completed", arguments="")
    )
    assert args == {}
    assert tool_id.startswith("call_")


def test_describe_detects_both_truncation_shapes():
    items = [
        _function_call_item(status="incomplete", arguments='{"ok": true}'),
        _function_call_item(status="completed", arguments=TRUNCATED_ARGS),
        _function_call_item(status="completed", arguments='{"ok": true}'),
        {"type": "message", "content": [], "id": "msg_1", "role": "assistant"},
    ]
    problems = describe_incomplete_function_calls(items)
    assert len(problems) == 2
    assert {p["reason"] for p in problems} == {
        "status_incomplete",
        "arguments_unparseable",
    }


# ---------------------------------------------------------------------------
# Accumulated-output conversion (the stitching path that produced {} calls)
# ---------------------------------------------------------------------------


def test_accumulated_conversion_raises_on_stitched_incomplete_calls():
    """The exact bug shape: N incomplete function_calls accumulated across
    continuations. Before the fix these surfaced as executable calls with {}
    arguments keyed by fc_ item ids."""
    from amplifier_module_provider_openai import OpenAIChatResponse

    final = _make_response(
        status="incomplete",
        output=[_function_call_item()],
        incomplete_reason="max_output_tokens",
    )
    accumulated = [
        _function_call_item(item_id=f"fc_{i:08d}", call_id=f"call_trunc{i}")
        for i in range(6)
    ]
    with pytest.raises(FunctionCallTruncationError):
        convert_response_with_accumulated_output(
            final, accumulated, 5, OpenAIChatResponse
        )


def test_accumulated_conversion_valid_calls_are_call_id_keyed():
    from amplifier_module_provider_openai import OpenAIChatResponse

    final = _make_response(status="completed")
    accumulated = [
        _function_call_item(status="completed", arguments='{"path": "/tmp/y"}')
    ]
    result = convert_response_with_accumulated_output(
        final, accumulated, 1, OpenAIChatResponse
    )
    assert result.tool_calls is not None
    assert result.tool_calls[0].id == "call_Ohhub2jBM8HU07mYExW9jDvD"
    assert result.tool_calls[0].arguments == {"path": "/tmp/y"}


# ---------------------------------------------------------------------------
# Continuation policy (retry-at-cap, then loud failure)
# ---------------------------------------------------------------------------


def test_truncated_call_retries_once_with_budget_raised_to_model_cap():
    provider = _make_provider({"max_tokens": 4096})
    captured: list[dict] = []
    truncated = _make_response(
        status="incomplete",
        output=[_function_call_item()],
        incomplete_reason="max_output_tokens",
    )
    recovered = _make_response(
        status="completed",
        output=[
            _function_call_item(
                status="completed",
                arguments='{"file_path": "/workspace/DISCREPANCIES.md", "content": "done"}',
            )
        ],
    )
    provider.client.responses.create = _sequential_mock(  # pyright: ignore[reportAttributeAccessIssue]
        [truncated, recovered], captured
    )

    request = ChatRequest(messages=[Message(role="user", content="write the file")])
    result = asyncio.run(provider.complete(request))

    assert len(captured) == 2, "expected exactly one raised-budget retry"
    cap = get_capabilities(captured[1]["model"]).max_output_tokens
    assert captured[0]["max_output_tokens"] == 4096
    assert captured[1]["max_output_tokens"] == cap
    assert result.tool_calls is not None
    assert result.tool_calls[0].id.startswith("call_")
    assert result.tool_calls[0].arguments["content"] == "done"


def test_truncated_call_at_cap_fails_loud_no_empty_arg_calls():
    provider = _make_provider({"max_tokens": 4096})
    captured: list[dict] = []
    truncated = _make_response(
        status="incomplete",
        output=[_function_call_item()],
        incomplete_reason="max_output_tokens",
    )
    # Both the original attempt and the raised-budget retry come back truncated.
    provider.client.responses.create = _sequential_mock(  # pyright: ignore[reportAttributeAccessIssue]
        [truncated, truncated], captured
    )

    request = ChatRequest(messages=[Message(role="user", content="write the file")])
    with pytest.raises(FunctionCallTruncationError) as excinfo:
        asyncio.run(provider.complete(request))

    assert "write_file" in str(excinfo.value)
    assert len(captured) == 2, "one retry, then loud failure — no fruitless loop"


def test_truncated_call_already_at_cap_fails_without_retry():
    provider = _make_provider()  # P5 default = model cap
    captured: list[dict] = []
    truncated = _make_response(
        status="incomplete",
        output=[_function_call_item()],
        incomplete_reason="max_output_tokens",
        max_output_tokens=128_000,
    )
    provider.client.responses.create = _sequential_mock(  # pyright: ignore[reportAttributeAccessIssue]
        [truncated], captured
    )

    request = ChatRequest(messages=[Message(role="user", content="write the file")])
    with pytest.raises(FunctionCallTruncationError):
        asyncio.run(provider.complete(request))

    assert len(captured) == 1, "already at cap: no budget left to retry into"


# ---------------------------------------------------------------------------
# P5 — capability-derived default output budget
# ---------------------------------------------------------------------------


def test_default_max_output_tokens_comes_from_model_capabilities():
    provider = _make_provider()  # no max_tokens config
    captured: list[dict] = []
    provider.client.responses.create = _sequential_mock(  # pyright: ignore[reportAttributeAccessIssue]
        [_make_response(status="completed")], captured
    )
    request = ChatRequest(messages=[Message(role="user", content="hi")])
    asyncio.run(provider.complete(request))

    model = captured[0]["model"]
    caps = get_capabilities(model)
    assert captured[0]["max_output_tokens"] == caps.max_output_tokens
    assert captured[0]["max_output_tokens"] > 4096, (
        "the fixed 4096 default was the P4 trigger and a parity confound"
    )


def test_explicit_config_max_tokens_still_wins():
    provider = _make_provider({"max_tokens": 2048})
    captured: list[dict] = []
    provider.client.responses.create = _sequential_mock(  # pyright: ignore[reportAttributeAccessIssue]
        [_make_response(status="completed")], captured
    )
    request = ChatRequest(messages=[Message(role="user", content="hi")])
    asyncio.run(provider.complete(request))
    assert captured[0]["max_output_tokens"] == 2048


# ---------------------------------------------------------------------------
# Wire-path backstop — orphaned function_call never reaches the API unpaired
# ---------------------------------------------------------------------------


def test_orphaned_function_call_gets_synthetic_output():
    provider = _make_provider()
    messages = [
        {"role": "user", "content": "do the thing"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call_orphan1", "name": "write_file", "arguments": {"a": 1}}
            ],
        },
        # NOTE: no tool-result message for call_orphan1 — the orphan shape
        # that 400'd the session ("No tool output found for function call").
    ]
    wire = provider._convert_messages(messages)
    calls = [i for i in wire if i.get("type") == "function_call"]
    outputs = {
        i.get("call_id") for i in wire if i.get("type") == "function_call_output"
    }
    assert calls, "sanity: the function_call must be replayed"
    for call in calls:
        assert call["call_id"] in outputs, (
            f"orphaned function_call {call['call_id']} reached the wire unpaired"
        )


def test_paired_function_call_gets_no_synthetic_output():
    provider = _make_provider()
    messages = [
        {"role": "user", "content": "do the thing"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call_ok1", "name": "write_file", "arguments": {"a": 1}}
            ],
        },
        {
            "role": "tool",
            "content": "file written",
            "tool_call_id": "call_ok1",
            "name": "write_file",
        },
    ]
    wire = provider._convert_messages(messages)
    outputs = [i for i in wire if i.get("type") == "function_call_output"]
    assert len(outputs) == 1, "no duplicate/synthetic output for a paired call"
