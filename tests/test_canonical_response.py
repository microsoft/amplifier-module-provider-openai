"""Canonical output parity and lossless, fail-closed continuation contracts."""

import copy
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from amplifier_core import ChatRequest, Message, llm_errors
from openai.types.responses import Response

from amplifier_module_provider_openai import OpenAIChatResponse, OpenAIProvider
from amplifier_module_provider_openai._response_handling import (
    FunctionCallTruncationError,
    UnsafeContinuationError,
    convert_response_with_accumulated_output,
    replay_output_items,
    response_items,
    response_with_output,
)


def payload(output=None, *, status="completed", reason=None, index=0):
    return {
        "id": f"resp_{index}",
        "object": "response",
        "created_at": 1,
        "model": "gpt-5.4",
        "status": status,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "incomplete_details": {"reason": reason} if reason else None,
        "output": output or [],
        "usage": {
            "input_tokens": 100,
            "output_tokens": 10,
            "total_tokens": 110,
            "input_tokens_details": {"cached_tokens": 20, "cache_write_tokens": 5},
            "output_tokens_details": {"reasoning_tokens": 3},
        },
    }


def text(index=0, **kwargs):
    return {
        "type": "message",
        "id": f"msg_{index}",
        "role": "assistant",
        "status": "completed",
        "phase": "final_answer",
        "content": [
            {"type": "output_text", "text": f"text-{index}", "annotations": []}
        ],
        **kwargs,
    }


def reasoning(index=0):
    return {
        "type": "reasoning",
        "id": f"rs_{index}",
        "encrypted_content": f"cipher-{index}",
        "summary": [{"type": "summary_text", "text": f"reason-{index}"}],
    }


FUNCTION = {
    "type": "function_call",
    "id": "fc_1",
    "call_id": "call_function",
    "namespace": "files",
    "name": "read",
    "arguments": '{"path":"a"}',
    "status": "completed",
}
PATCH = {
    "type": "apply_patch_call",
    "id": "ap_1",
    "call_id": "call_patch",
    "status": "completed",
    "operation": {"type": "update_file", "path": "a", "diff": "@@ x"},
}
COMPUTER = {
    "type": "computer_call",
    "id": "cc_1",
    "call_id": "call_computer",
    "status": "completed",
    "actions": [{"type": "screenshot"}],
    "pending_safety_checks": [],
}


@pytest.mark.parametrize("item", [text(), reasoning(), FUNCTION, PATCH, COMPUTER])
@pytest.mark.parametrize("as_sdk", [False, True])
def test_sdk_dict_ordinary_continued_conversion_parity(item, as_sdk):
    wire = payload([item])
    sdk = Response.model_validate(wire)
    response = sdk if as_sdk else wire
    provider = OpenAIProvider(api_key="offline")
    ordinary = provider._convert_to_chat_response(response)
    continued = convert_response_with_accumulated_output(
        response, response_items(sdk.output), 2, OpenAIChatResponse
    )
    left, right = ordinary.model_dump(), continued.model_dump()
    # Pricing belongs to the instance/endpoint hook, not the pure compatibility helper.
    left["usage"]["cost_usd"] = right["usage"]["cost_usd"]
    right["metadata"].pop("openai:continuation_count")
    assert left == right
    assert ordinary.metadata["openai:output_items"] == [item]
    if ordinary.tool_calls:
        assert ordinary.content_blocks[0].id == ordinary.tool_calls[0].id
        assert ordinary.content_blocks[0].raw == item
    if item["type"] == "apply_patch_call":
        assert provider._native_call_ids == {"call_patch"}
    if item["type"] == "computer_call":
        assert provider._native_call_types == {"call_computer": "computer"}
    assert ordinary.usage.input_tokens == 95
    assert ordinary.usage.cache_write_tokens == 5


def test_sdk_omits_only_unset_defaults_preserves_explicit_nulls_and_unknown_fields():
    item = text()
    del item["phase"]
    item["future"] = {"opaque": [1, None, {"retained": True}]}
    sdk = Response.model_validate(payload([item]))
    assert response_items(sdk.output) == [item]
    explicit = Response.model_validate(payload([text(phase=None)]))
    assert response_items(explicit.output)[0]["phase"] is None


def test_continuation_preserves_order_phase_ids_ciphertext_and_opaque_items_without_mutation():
    output = [
        reasoning(),
        text(status="incomplete"),
        {
            "type": "tool_search_call",
            "id": "ts_1",
            "status": "completed",
            "arguments": {},
        },
        {"type": "tool_search_output", "call_id": "ts_1", "tools": []},
        {"type": "compaction", "id": "cmp_1", "encrypted_content": "opaque"},
        {"type": "future_hosted_item", "id": "future_1", "opaque": [None, "x"]},
    ]
    original = [{"role": "user", "content": "question"}]
    before = copy.deepcopy((original, output))
    provider = OpenAIProvider(api_key="offline")
    result = provider._build_continuation_input(original, output)
    assert result == original + output
    result[0]["content"] = "changed"
    result[-1]["opaque"].append("changed")
    assert (original, output) == before


@pytest.mark.parametrize("item", [FUNCTION, PATCH, COMPUTER])
def test_complete_host_call_is_preserved_for_history_but_refused_for_automatic_continuation(
    item,
):
    assert replay_output_items([item]) == [item]
    with pytest.raises(
        UnsafeContinuationError, match="real host tool result"
    ) as caught:
        replay_output_items([item], continuation=True)
    assert caught.value.retryable is False


@pytest.mark.parametrize(
    "item",
    [
        {**FUNCTION, "status": "incomplete"},
        {**FUNCTION, "arguments": '{"path":'},
    ],
)
def test_truncated_call_never_becomes_executable_even_without_item_status(item):
    provider = OpenAIProvider(api_key="offline")
    with pytest.raises(FunctionCallTruncationError):
        provider._convert_to_chat_response(payload([item], status="incomplete"))
    with pytest.raises(FunctionCallTruncationError):
        replay_output_items([item], continuation=True)


def test_completed_malformed_history_preserves_real_tool_error_feedback_path():
    item = {**FUNCTION, "arguments": '{"path":'}
    provider = OpenAIProvider(api_key="offline")
    reply = provider._convert_to_chat_response(payload([item]))
    assert reply.tool_calls[0].arguments == {}
    assert replay_output_items(reply.metadata["openai:output_items"]) == [item]


@pytest.mark.parametrize("kind", ["custom_tool_call", "local_shell_call", "shell_call"])
def test_other_host_call_families_cannot_auto_continue_without_real_results(kind):
    with pytest.raises(UnsafeContinuationError):
        replay_output_items(
            [{"type": kind, "call_id": "call", "input": "raw"}], continuation=True
        )


def test_native_state_commits_only_after_whole_batch_conversion_validates():
    provider = OpenAIProvider(api_key="offline")
    with pytest.raises(UnsafeContinuationError):
        provider._convert_to_chat_response(
            payload([PATCH, {**COMPUTER, "status": "incomplete"}])
        )
    assert provider._native_call_ids == set()
    assert provider._native_call_types == {}


class EndpointProvider(OpenAIProvider):
    """Subclasses must see both continued conversion and endpoint pricing."""

    def __init__(self, **kwargs):
        super().__init__(
            api_key="offline",
            config={"use_streaming": False, "max_retries": 0, **kwargs},
        )
        self.converted = []
        self.priced = []
        self.costs = []
        self._add_cost = self.costs.append
        self._guard_assembled_params_with_provider_count = AsyncMock(return_value=1)

    def _convert_to_chat_response(self, response, **options):
        self.converted.append(response_items(response.output))
        return super()._convert_to_chat_response(response, **options)

    def _compute_attempt_cost(self, response):
        self.priced.append(response.id)
        return Decimal("0.125")


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_kind", ["terminal", "disconnect", "cancel"])
async def test_adapter_error_observations_retain_confirmed_parent_without_duplicate_billing(
    failure_kind,
):
    import asyncio

    confirmed = Response.model_validate(payload([text()], index=91))
    unresolved = SimpleNamespace(model="gpt-5.4", usage=None)

    class ObservedEndpoint(EndpointProvider):
        observation_calls = 0

        def _error_response_observations(self, error):
            self.observation_calls += 1
            return (confirmed, unresolved)

    provider = ObservedEndpoint()
    hooks = SimpleNamespace(emit=AsyncMock())
    provider.coordinator = SimpleNamespace(hooks=hooks)
    failure = {
        "terminal": llm_errors.LLMError("successor failed", retryable=False),
        "disconnect": llm_errors.LLMError("successor outcome unknown", retryable=False),
        "cancel": asyncio.CancelledError(),
    }[failure_kind]
    provider.client.responses.create = AsyncMock(side_effect=failure)
    try:
        with pytest.raises(type(failure)) as caught:
            await provider.complete(
                ChatRequest(messages=[Message(role="user", content="go")])
            )
        assert provider.observation_calls == 1
        assert provider.priced == ["resp_91"]
        assert provider.costs == [Decimal("0.125")]
        assert caught.value.usage.input_tokens == 95
        assert caught.value.usage.output_tokens == 10
        assert caught.value.usage.cost_usd is None
        assert caught.value.known_cost_usd == Decimal("0.125")
        assert provider.converted == []
        error_events = [
            call.args[1]
            for call in hooks.emit.call_args_list
            if call.args[0] == "llm:response" and call.args[1].get("status") == "error"
        ]
        if failure_kind == "cancel":
            assert (
                error_events == []
            )  # cancellation remains cancellation, not an ordinary error event
        else:
            assert len(error_events) == 1
            assert error_events[0]["usage"]["cost_usd"] is None
            assert error_events[0]["known_cost_usd"] == "0.125"
    finally:
        await provider.close()


@pytest.mark.asyncio
async def test_adapter_error_observation_hook_is_not_called_after_success_billing():
    class ObservedEndpoint(EndpointProvider):
        def _error_response_observations(self, error):
            raise AssertionError(
                "successful billing must not consume adapter failure evidence"
            )

    provider = ObservedEndpoint()
    provider.client.responses.create = AsyncMock(
        return_value=Response.model_validate(payload([text()]))
    )
    try:
        result = await provider.complete(
            ChatRequest(messages=[Message(role="user", content="go")])
        )
        assert result.usage.cost_usd == Decimal("0.125")
        assert provider.costs == [Decimal("0.125")]
    finally:
        await provider.close()


@pytest.mark.parametrize("as_spec", [False, True])
def test_custom_tool_declaration_keeps_raw_input_contract_and_grammar(as_spec):
    from amplifier_core.message_models import ToolSpec

    definition = {
        "type": "custom",
        "name": "query",
        "description": "Raw query",
        "format": {"type": "grammar", "syntax": "lark", "definition": 'start: "query"'},
    }
    tool = (
        ToolSpec.model_validate({**definition, "parameters": {}})
        if as_spec
        else definition
    )
    provider = OpenAIProvider(api_key="offline")
    assert provider._convert_tools_from_request([tool]) == [definition]
    item = {
        "type": "custom_tool_call",
        "id": "ct_1",
        "call_id": "call_custom",
        "name": "query",
        "input": "query",
    }
    reply = provider._convert_to_chat_response(payload([item]))
    assert reply.metadata["openai:output_items"] == [item]
    assert (
        reply.tool_calls is None
    )  # raw custom execution belongs to the native adapter


@pytest.mark.asyncio
async def test_final_extra_body_input_override_continues_effective_history():
    overridden = [{"role": "user", "content": "effective final input"}]
    provider = EndpointProvider(
        extra_request_params={"extra_body": {"input": overridden}}
    )
    first = Response.model_validate(
        payload([text()], status="incomplete", reason="max_output_tokens")
    )
    second = Response.model_validate(payload([text(1)]))
    provider.client.responses.create = AsyncMock(side_effect=[first, second])
    try:
        await provider.complete(
            ChatRequest(messages=[Message(role="user", content="superseded")])
        )
        continued = provider.client.responses.create.call_args_list[1].kwargs
        assert continued["input"] == overridden + [text()]
        assert continued["extra_body"]["input"] == continued["input"]
    finally:
        await provider.close()


@pytest.mark.asyncio
async def test_cancellation_accounts_only_confirmed_previous_attempt_once():
    import asyncio

    provider = EndpointProvider()
    first = Response.model_validate(
        payload([text()], status="incomplete", reason="max_output_tokens")
    )
    provider.client.responses.create = AsyncMock(
        side_effect=[first, asyncio.CancelledError()]
    )
    try:
        with pytest.raises(asyncio.CancelledError) as caught:
            await provider.complete(
                ChatRequest(messages=[Message(role="user", content="go")])
            )
        assert provider.costs == [Decimal("0.125")]
        assert caught.value.usage.input_tokens == 95
        assert provider.client.responses.create.await_count == 2
    finally:
        await provider.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_status", ["completed", "failed", "cancelled"])
async def test_background_continuation_waits_for_same_response_terminal(
    terminal_status,
):
    provider = EndpointProvider(poll_interval=0)
    initial = Response.model_validate(
        payload([text()], status="incomplete", reason="max_output_tokens")
    )
    queued = Response.model_validate(payload([], status="queued", index=1))
    terminal = Response.model_validate(
        payload([text(1)], status=terminal_status, index=1)
    )
    provider.client.responses.create = AsyncMock(side_effect=[initial, queued])
    provider.client.responses.retrieve = AsyncMock(return_value=terminal)
    try:
        if terminal_status == "completed":
            result = await provider.complete(
                ChatRequest(messages=[Message(role="user", content="go")])
            )
            assert result.output_text == "text-0text-1"
        else:
            with pytest.raises(llm_errors.LLMError) as caught:
                await provider.complete(
                    ChatRequest(messages=[Message(role="user", content="go")])
                )
            assert caught.value.retryable is False
        provider.client.responses.retrieve.assert_awaited_once()
        assert provider.client.responses.retrieve.call_args.args == ("resp_1",)
        assert provider.client.responses.create.await_count == 2
    finally:
        await provider.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_status", ["completed", "failed", "cancelled"])
async def test_raised_budget_repair_settles_queued_response_without_regeneration(
    terminal_status,
):
    provider = EndpointProvider(poll_interval=0)
    initial = Response.model_validate(
        payload(
            [{**FUNCTION, "status": "incomplete", "arguments": '{"path":'}],
            status="incomplete",
            reason="max_output_tokens",
        )
    )
    queued = Response.model_validate(payload([], status="queued", index=1))
    terminal = Response.model_validate(
        payload([text(1)], status=terminal_status, index=1)
    )
    provider.client.responses.create = AsyncMock(side_effect=[initial, queued])
    provider.client.responses.retrieve = AsyncMock(return_value=terminal)
    try:
        if terminal_status == "completed":
            result = await provider.complete(
                ChatRequest(messages=[Message(role="user", content="go")]),
                max_tokens=4096,
            )
            assert result.output_text == "text-1"
            assert result.tool_calls is None
        else:
            with pytest.raises(llm_errors.LLMError) as caught:
                await provider.complete(
                    ChatRequest(messages=[Message(role="user", content="go")]),
                    max_tokens=4096,
                )
            assert caught.value.retryable is False
        provider.client.responses.retrieve.assert_awaited_once()
        assert provider.client.responses.create.await_count == 2
        assert provider.costs == [Decimal("0.250")]
    finally:
        await provider.close()


@pytest.mark.asyncio
async def test_terminal_error_mixed_missing_usage_retains_known_subtotal_not_false_total():
    provider = EndpointProvider()
    item = {**FUNCTION, "status": "incomplete", "arguments": '{"path":'}
    initial = Response.model_validate(
        payload([item], status="incomplete", reason="max_output_tokens")
    )
    unknown_wire = payload(
        [item], status="incomplete", reason="max_output_tokens", index=1
    )
    unknown_wire["usage"] = None
    provider.client.responses.create = AsyncMock(
        side_effect=[initial, Response.model_validate(unknown_wire)]
    )
    try:
        with pytest.raises(FunctionCallTruncationError) as caught:
            await provider.complete(
                ChatRequest(messages=[Message(role="user", content="go")]),
                max_tokens=4096,
            )
        assert caught.value.usage.cost_usd is None
        assert caught.value.known_cost_usd == Decimal("0.125")
        assert provider.costs == [Decimal("0.125")]
    finally:
        await provider.close()


@pytest.mark.asyncio
async def test_two_successive_truncations_preserve_validated_options_order_and_each_attempt_cost():
    provider = EndpointProvider(
        extra_request_params={
            "max_tool_calls": 7,
            "extra_body": {"gateway_field": "retained"},
            "extra_headers": {"X-Test": "value"},
            "service_tier": "priority",
        }
    )
    outputs = [
        [reasoning(i), text(i, status="incomplete" if i < 2 else "completed")]
        for i in range(3)
    ]
    responses = [
        Response.model_validate(
            payload(
                items,
                index=i,
                status="incomplete" if i < 2 else "completed",
                reason="max_output_tokens" if i < 2 else None,
            )
        )
        for i, items in enumerate(outputs)
    ]
    provider.client.responses.create = AsyncMock(side_effect=responses)
    request = ChatRequest(messages=[Message(role="user", content="go")])
    try:
        result = await provider.complete(request, timeout=42)
        calls = [
            call.kwargs for call in provider.client.responses.create.call_args_list
        ]
        assert len(calls) == 3
        assert calls[1]["input"] == calls[0]["input"] + outputs[0]
        assert calls[2]["input"] == calls[0]["input"] + outputs[0] + outputs[1]
        for call in calls[1:]:
            assert {k: v for k, v in call.items() if k != "input"} == {
                k: v for k, v in calls[0].items() if k != "input"
            }
        assert len(provider.converted) == 1
        assert provider.converted[0] == [item for output in outputs for item in output]
        assert result.output_text == "text-0text-1text-2"
        assert result.metadata["openai:continuation_count"] == 2
        assert result.usage.input_tokens == 285
        assert result.usage.output_tokens == 30
        assert result.usage.cache_write_tokens == 15
        assert result.usage.reasoning_tokens == 9
        assert result.usage.cost_usd == Decimal("0.375")
        assert provider.costs == [Decimal("0.375")]
        assert set(provider.priced) == {"resp_0", "resp_1", "resp_2"}
    finally:
        await provider.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", [TimeoutError("lost acknowledgement"), RuntimeError("disconnected")]
)
async def test_uncertain_continuation_is_not_partial_success_or_replayed(failure):
    provider = EndpointProvider(max_retries=3)
    initial = Response.model_validate(
        payload(
            [text(status="incomplete")], status="incomplete", reason="max_output_tokens"
        )
    )
    provider.client.responses.create = AsyncMock(side_effect=[initial, failure])
    try:
        with pytest.raises(llm_errors.LLMError, match="outcome is unknown") as caught:
            await provider.complete(
                ChatRequest(messages=[Message(role="user", content="go")])
            )
        assert caught.value.retryable is False
        assert provider.client.responses.create.await_count == 2
        assert provider.converted == []
        # Only the confirmed first response is priced; the lost response is
        # neither made free nor assigned invented tokens/cost.
        assert provider.costs == [Decimal("0.125")]
        assert caught.value.usage.input_tokens == 95
    finally:
        await provider.close()


@pytest.mark.asyncio
async def test_content_filter_does_not_start_an_automatic_paid_continuation():
    provider = EndpointProvider()
    initial = Response.model_validate(
        payload([text()], status="incomplete", reason="content_filter")
    )
    provider.client.responses.create = AsyncMock(return_value=initial)
    try:
        result = await provider.complete(
            ChatRequest(messages=[Message(role="user", content="go")])
        )
        assert result.finish_reason == "length"
        assert result.metadata["openai:incomplete_reason"] == "content_filter"
        assert provider.client.responses.create.await_count == 1
    finally:
        await provider.close()


@pytest.mark.asyncio
async def test_terminal_truncated_repair_bills_each_confirmed_attempt_once_and_emits_error_usage():
    provider = EndpointProvider(max_tokens=4096)
    hooks = SimpleNamespace(emit=AsyncMock())
    provider.coordinator = SimpleNamespace(hooks=hooks)
    truncated = {**FUNCTION, "status": "incomplete", "arguments": '{"path":'}
    attempts = [
        Response.model_validate(
            payload(
                [truncated],
                status="incomplete",
                reason="max_output_tokens",
                index=index,
            )
        )
        for index in range(2)
    ]
    provider.client.responses.create = AsyncMock(side_effect=attempts)
    try:
        with pytest.raises(FunctionCallTruncationError) as caught:
            await provider.complete(
                ChatRequest(messages=[Message(role="user", content="go")]),
                max_tokens=4096,
            )
        assert provider.client.responses.create.await_count == 2
        assert provider.converted == []
        assert provider.costs == [Decimal("0.250")]
        assert caught.value.usage.input_tokens == 190
        assert caught.value.usage.output_tokens == 20
        assert caught.value.usage.cache_write_tokens == 10
        errors = [
            call.args[1]
            for call in hooks.emit.call_args_list
            if call.args[0] == "llm:response" and call.args[1].get("status") == "error"
        ]
        assert len(errors) == 1
        assert errors[0]["usage"]["cost_usd"] == "0.250"
        assert errors[0]["usage"]["input_tokens"] == 190
    finally:
        await provider.close()


@pytest.mark.asyncio
async def test_missing_usage_on_terminal_refusal_never_invents_cost():
    provider = EndpointProvider()
    no_usage = payload([{**PATCH, "status": "in_progress"}])
    no_usage["usage"] = None
    provider.client.responses.create = AsyncMock(
        return_value=Response.model_validate(no_usage)
    )
    try:
        with pytest.raises(UnsafeContinuationError) as caught:
            await provider.complete(
                ChatRequest(messages=[Message(role="user", content="go")])
            )
        assert not hasattr(caught.value, "usage")
        assert provider.costs == []
    finally:
        await provider.close()


def test_response_view_handles_sdk_and_raw_wrappers_without_copy_or_mutating_original():
    from amplifier_module_provider_openai import _RawResponseObject

    for original in (
        Response.model_validate(payload([text()])),
        _RawResponseObject(payload([text()])),
    ):
        view = response_with_output(original, [text(1)])
        assert original.output[0] != view.output[0]
        assert view.id == original.id
        assert view.output_text == "text-1"


@pytest.mark.parametrize(
    "summary",
    [
        [{"text": "one"}, SimpleNamespace(text="two"), "three"],
        "one",
        {"text": "one"},
        SimpleNamespace(text="one"),
        [],
    ],
)
def test_reasoning_helper_matches_removed_duplicate_fallback(summary):
    from amplifier_module_provider_openai._response_handling import (
        extract_reasoning_text,
    )

    if isinstance(summary, list):
        expected = "\n".join(
            filter(
                None,
                (
                    item
                    if isinstance(item, str)
                    else item.get("text", "")
                    if isinstance(item, dict)
                    else getattr(item, "text", "")
                    for item in summary
                ),
            )
        )
    elif isinstance(summary, str):
        expected = summary
    elif isinstance(summary, dict):
        expected = summary.get("text", str(summary))
    else:
        expected = getattr(summary, "text", str(summary))
    assert extract_reasoning_text(summary) == (expected or None)
