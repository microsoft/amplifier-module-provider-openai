import copy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from amplifier_core.message_models import ChatRequest, Message

from amplifier_module_provider_openai import OpenAIProvider
from amplifier_module_provider_openai.native import NativeResponsesProvider


def make():
    p = NativeResponsesProvider.wrap(
        OpenAIProvider(api_key="fixture", config={"default_model": "gpt-6-astra"}),
        owner_getter=lambda: None,
    )
    request = ChatRequest(
        messages=[Message(role="user", content="before")], max_output_tokens=32
    )
    canonical = [m.model_dump() for m in request.messages]
    p._last_native_request = (request, {})
    output = [
        {
            "id": "c",
            "type": "compaction",
            "encrypted_content": "OPAQUE secret punctuation /+= 😁",
        },
        {
            "role": "user",
            "content": [{"type": "input_text", "text": "retained exactly"}],
        },
    ]
    api = AsyncMock(
        return_value=SimpleNamespace(
            content=json.dumps({"output": output}, ensure_ascii=False)
        )
    )
    client = SimpleNamespace(
        responses=SimpleNamespace(with_raw_response=SimpleNamespace(compact=api))
    )
    client.with_options = lambda **kwargs: client
    p._client = client
    return p, canonical, output, api


@pytest.mark.asyncio
async def test_compaction_preserves_full_exact_window_and_restores_without_network():
    p, messages, opaque, api = make()
    identity = {"instance": "selected", "model": "gpt-6-astra"}
    originals = copy.deepcopy(messages)
    await p.native_compact(canonical=messages, identity=identity)
    record = p.native_export_checkpoint()
    assert record["output"] == opaque and messages == originals
    assert set(api.call_args.kwargs) == {"model", "input"}
    q, _, _, unused = make()
    q.native_restore_checkpoint(record, canonical=messages, identity=identity)
    assert unused.await_count == 0
    # Include a new suffix in the original canonical and provider input forms.
    messages = messages + [Message(role="user", content="after").model_dump()]
    q.full_messages = messages
    request = ChatRequest(
        messages=[Message(**m) for m in messages], max_output_tokens=32
    )
    params, _, _ = q._assemble_initial_responses_params(request)
    result = q._apply_checkpoint(params, params)
    assert result["input"][: len(opaque)] == opaque
    assert len(result["input"]) == len(opaque) + 1
    assert "previous_response_id" not in result
    result["input"][0]["encrypted_content"] = "changed caller copy"
    assert q.native_export_checkpoint()["output"] == opaque
    assert "OPAQUE" not in str(q.native_status())


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["prefix", "identity", "config", "output", "format"])
async def test_restore_rejects_modified_source_config_identity_or_record(change):
    p, messages, _, _ = make()
    identity = {"instance": "one", "model": "gpt-6-astra"}
    await p.native_compact(canonical=messages, identity=identity)
    record = p.native_export_checkpoint()
    if change == "prefix":
        messages[0]["content"] = "rewritten"
    if change == "identity":
        identity = {**identity, "instance": "two"}
    if change == "config":
        p.config["reasoning_effort"] = "high"
    if change == "output":
        record["output"][0]["encrypted_content"] = "tampered"
    if change == "format":
        record["format"] = "summary"
    p.native_restore_checkpoint(record, canonical=messages, identity=identity)
    assert p.native_export_checkpoint() is None
    assert p.native_status()["checkpoint"].startswith("discarded")


@pytest.mark.asyncio
async def test_request_fitting_or_tool_schema_change_discards_opaque_without_summary():
    p, messages, _, _ = make()
    identity = {"instance": "one", "model": "gpt-6-astra"}
    await p.native_compact(canonical=messages, identity=identity)
    p.full_messages = messages
    params, _, _ = p._assemble_initial_responses_params(p._last_native_request[0])
    params["tools"] = [{"type": "function", "name": "different"}]
    assert p._apply_checkpoint(params, params) == params
    assert p.native_export_checkpoint() is None


@pytest.mark.asyncio
async def test_uncertain_native_outcome_prevents_compaction_or_restore():
    p, messages, _, api = make()
    p.request_uncertain = True
    with pytest.raises(ValueError, match="unknown"):
        await p.native_compact(canonical=messages, identity={})
    with pytest.raises(ValueError):
        p.native_restore_checkpoint({}, canonical=messages, identity={})
    assert api.await_count == 0


@pytest.mark.asyncio
async def test_budget_and_initial_guard_measure_exact_opaque_window_without_advancing_lineage():
    from amplifier_module_provider_openai.native import NATIVE_REQUEST

    p, messages, opaque, _ = make()
    identity = {"instance": "selected", "model": "gpt-6-astra"}
    await p.native_compact(canonical=messages, identity=identity)
    owner = SimpleNamespace(
        context=SimpleNamespace(get_messages=AsyncMock(return_value=messages))
    )
    p.owner_getter = lambda: owner
    request = p._last_native_request[0]
    p._provider_count_available = lambda: False
    measured = []
    original = p._estimated_input_tokens

    def estimate(params, **kwargs):
        measured.append(copy.deepcopy(params))
        return original(params, **kwargs)

    p._estimated_input_tokens = estimate
    before = p.native_export_checkpoint()
    budget = await p.request_budget(request, context_estimate=100)
    assert budget and measured[0]["input"] == opaque
    assert (
        not p.seen
        and p.previous_response_id is None
        and p.native_export_checkpoint() == before
    )
    p.full_messages = messages
    token = NATIVE_REQUEST.set(p)
    try:
        params, state, _ = p._assemble_initial_responses_params(request)
        assert params["input"] == opaque
        assert state["native_full_params"]["input"] != opaque
        p._commit_initial_assembly_state(state)
        assert p._native_full_params["input"] == state["native_full_params"]["input"]
        assert not p.seen and p.previous_response_id is None
    finally:
        NATIVE_REQUEST.reset(token)


@pytest.mark.asyncio
@pytest.mark.parametrize("new_user_after_reasoning", [False, True])
async def test_real_complete_guards_and_sends_same_opaque_payload_after_compaction(
    new_user_after_reasoning,
):
    import json

    if new_user_after_reasoning:
        p, messages, _, opaque, _ = await reasoning_checkpoint()
        messages = [*messages, {"role": "user", "content": "Repeat the value"}]
        request = ChatRequest(
            messages=[Message(**m) for m in messages], max_output_tokens=32
        )
        params, _, _ = p._assemble_initial_responses_params(request)
        expected_input = [*opaque, params["input"][-1]]
    else:
        p, messages, opaque, _ = make()
        await p.native_compact(
            canonical=messages, identity={"instance": "one", "model": "gpt-6-astra"}
        )
        request = p._last_native_request[0]
        expected_input = opaque
    owner = SimpleNamespace(
        context=SimpleNamespace(get_messages=AsyncMock(return_value=messages)),
        runtime=SimpleNamespace(emit=AsyncMock()),
        config={},
        native_job=lambda identity: None,
    )
    p.owner_getter = lambda: owner
    p._guard_assembled_params_with_provider_count = AsyncMock(return_value=None)
    p.socket = SimpleNamespace(
        send=AsyncMock(),
        close=AsyncMock(),
        recv=AsyncMock(
            side_effect=[
                json.dumps({"type": "response.created", "response": {"id": "new"}}),
                json.dumps(
                    {
                        "type": "response.completed",
                        "response": {
                            "id": "new",
                            "status": "completed",
                            "model": "gpt-6-astra",
                            "output": [],
                            "usage": {"input_tokens": 1, "output_tokens": 0},
                        },
                    }
                ),
            ]
        ),
    )
    await p.complete(request)
    sent = json.loads(p.socket.send.call_args.args[0])
    assert sent["input"] == expected_input and "previous_response_id" not in sent
    assert all(
        call.args[0]["input"] == expected_input
        for call in p._guard_assembled_params_with_provider_count.call_args_list
    )


@pytest.mark.asyncio
async def test_compaction_retains_actual_factory_instructions_and_public_usage():
    p, canonical, opaque, api = make()
    request, kwargs = p._last_native_request
    request = request.model_copy(
        update={
            "messages": [
                Message(role="system", content="Exact factory and host policy"),
                *request.messages,
            ]
        }
    )
    p._last_native_request = request, kwargs
    api.return_value = SimpleNamespace(
        content=json.dumps(
            {
                "output": opaque,
                "usage": {
                    "input_tokens": 12,
                    "output_tokens": 3,
                    "encrypted_content": "never public",
                    "total_tokens": float("inf"),
                },
            }
        )
    )
    result = await p.native_compact(canonical=canonical, identity={"instance": "one"})
    assert api.call_args.kwargs["instructions"] == "Exact factory and host policy"
    assert result["usage"] == {"input_tokens": 12, "output_tokens": 3}
    assert len(canonical) == 1 and canonical[0]["role"] == "user"
    p.full_messages = canonical
    params, _, _ = p._assemble_initial_responses_params(request)
    assert p._apply_checkpoint(params, params)["input"] == opaque
    changed = request.model_copy(
        update={
            "messages": [
                Message(role="system", content="Changed policy"),
                *request.messages[1:],
            ]
        }
    )
    params, _, _ = p._assemble_initial_responses_params(changed)
    assert p._apply_checkpoint(params, params) == params
    assert p.native_export_checkpoint() is None


async def reasoning_checkpoint():
    """A completed turn with tool evidence and real typed reasoning content."""
    from amplifier_core.message_models import ThinkingBlock

    p, _, opaque, api = make()
    canonical = [
        {"role": "user", "content": "Read the synthetic value"},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_call", "id": "call_fixture", "name": "read", "input": {}}
            ],
        },
        {"role": "tool", "tool_call_id": "call_fixture", "content": '{"value":17}'},
        {
            "role": "assistant",
            "content": [
                ThinkingBlock(
                    thinking="synthetic reasoning",
                    content=[
                        {
                            "id": "rs_fixture",
                            "encrypted_content": "ENC_fixture",
                            "summary": "synthetic reasoning",
                        }
                    ],
                ).model_dump(),
                {"type": "text", "text": "FINAL-B 17"},
            ],
        },
    ]
    request = ChatRequest(
        messages=[Message(**m) for m in canonical], max_output_tokens=32
    )
    p._last_native_request = request, {}
    identity = {"instance": "reasoning", "model": "gpt-6-astra"}
    await p.native_compact(canonical=canonical, identity=identity)
    record = p.native_export_checkpoint()
    assert any(
        item.get("type") == "reasoning" for item in api.call_args.kwargs["input"]
    )
    q, _, _, unused = make()
    q.native_restore_checkpoint(record, canonical=canonical, identity=identity)
    return q, canonical, record, opaque, unused


@pytest.mark.asyncio
async def test_new_user_reuses_restored_reasoning_checkpoint_without_resending_tool_result():
    from amplifier_module_provider_openai.native import NATIVE_REQUEST

    p, canonical, record, opaque, unused = await reasoning_checkpoint()
    original = copy.deepcopy(canonical)
    canonical = [*canonical, {"role": "user", "content": "Repeat the tag and value"}]
    request = ChatRequest(
        messages=[Message(**m) for m in canonical], max_output_tokens=32
    )
    params, _, _ = p._assemble_initial_responses_params(request)
    assert not any(item.get("type") == "reasoning" for item in params["input"])
    assert record["wireCount"] == len(params["input"])
    expected = [*opaque, params["input"][-1]]
    before = (
        copy.deepcopy(p.seen),
        copy.deepcopy(p.last_context),
        set(p._native_call_ids),
        dict(p._native_call_types),
    )
    assert p._plan_checkpoint(params, canonical)["input"] == expected
    assert (p.seen, p.last_context, p._native_call_ids, p._native_call_types) == before
    assert p.native_export_checkpoint() == record and canonical[:-1] == original
    assert unused.await_count == 0

    # Budgeting and the actual request planner must select the same payload.
    p.owner_getter = lambda: SimpleNamespace(
        context=SimpleNamespace(get_messages=AsyncMock(return_value=canonical))
    )
    p._provider_count_available = lambda: False
    measured = []
    estimate = p._estimated_input_tokens
    p._estimated_input_tokens = lambda params, **kwargs: (
        measured.append(copy.deepcopy(params)) or estimate(params, **kwargs)
    )
    assert await p.request_budget(request, context_estimate=100)
    assert measured[0]["input"] == expected
    p.full_messages = canonical
    token = NATIVE_REQUEST.set(p)
    try:
        planned, state, _ = p._assemble_initial_responses_params(request)
        assert planned["input"] == expected
        assert (
            p._apply_checkpoint(planned, state["native_full_params"])["input"]
            == expected
        )
    finally:
        NATIVE_REQUEST.reset(token)
    assert not any(item.get("type") == "function_call_output" for item in expected)
    assert p.native_export_checkpoint() == record


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change",
    ["canonical", "tool_result", "assistant", "instructions", "insert", "ephemeral"],
)
async def test_reasoning_expiry_does_not_relax_other_checkpoint_boundaries(change):
    p, canonical, record, _, _ = await reasoning_checkpoint()
    canonical = [*canonical, {"role": "user", "content": "Repeat the value"}]
    request = ChatRequest(
        messages=[Message(**m) for m in canonical], max_output_tokens=32
    )
    params, _, _ = p._assemble_initial_responses_params(request)
    if change == "canonical":
        canonical[0]["content"] = "different authority"
    elif change == "tool_result":
        next(
            item
            for item in params["input"]
            if item.get("type") == "function_call_output"
        )["output"] = "different result"
    elif change == "assistant":
        next(item for item in params["input"] if item.get("role") == "assistant")[
            "content"
        ] = "different text"
    elif change == "instructions":
        params["instructions"] = "different policy"
    elif change == "insert":
        params["input"].insert(0, {"role": "developer", "content": "new authority"})
    elif change == "ephemeral":
        canonical[-1]["metadata"] = {"ephemeral": True}
    assert p._plan_checkpoint(params, canonical) == params
    assert p.native_export_checkpoint() == record
    p.full_messages = canonical
    assert p._apply_checkpoint(params, params) == params
    assert p.native_export_checkpoint() is None
