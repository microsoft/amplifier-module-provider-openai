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
async def test_real_complete_guards_and_sends_same_opaque_payload_after_compaction():
    import json

    p, messages, opaque, _ = make()
    await p.native_compact(
        canonical=messages, identity={"instance": "one", "model": "gpt-6-astra"}
    )
    request = p._last_native_request[0]
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
    assert sent["input"] == opaque and "previous_response_id" not in sent
    assert all(
        call.args[0]["input"] == opaque
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
