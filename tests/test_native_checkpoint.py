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
