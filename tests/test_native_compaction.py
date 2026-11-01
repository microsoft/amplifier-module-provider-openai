import copy
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from amplifier_core import ChatRequest, Message, ToolSpec
from amplifier_core.llm_errors import ContextLengthError

from amplifier_module_provider_openai import OpenAIProvider
from amplifier_module_provider_openai.compaction import (
    canonical_usage,
    compacted_message,
)


def test_native_usage_has_disjoint_cache_write_tokens_and_unknown_cost():
    usage = canonical_usage(
        {
            "input_tokens": 1000,
            "output_tokens": 100,
            "total_tokens": 1100,
            "input_tokens_details": {"cached_tokens": 200, "cache_write_tokens": 300},
        }
    )
    assert usage == {
        "input_tokens": 700,
        "output_tokens": 100,
        "total_tokens": 800,
        "cache_read_tokens": 200,
        "cache_write_tokens": 300,
    }
    assert "cost_usd" not in usage


def fixture(**config):
    # A retained call's result may live inside opaque state. The returned
    # canonical window must not acquire a fabricated "missing result" item.
    window = [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "Keep fact ORBIT"}],
        },
        {
            "type": "function_call",
            "id": "fc1",
            "call_id": "c1",
            "name": "read",
            "arguments": "{}",
        },
        {"type": "compaction", "id": "cmp1", "encrypted_content": "opaque-fixture"},
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "Latest correction"}],
        },
    ]
    responses = SimpleNamespace(
        input_tokens=SimpleNamespace(
            count=AsyncMock(return_value=SimpleNamespace(input_tokens=100))
        ),
        compact=AsyncMock(
            return_value=SimpleNamespace(
                output=copy.deepcopy(window),
                usage={"input_tokens": 100, "output_tokens": 20},
            )
        ),
        create=AsyncMock(),
    )
    provider = OpenAIProvider(
        api_key="test",
        client=SimpleNamespace(
            base_url="https://api.openai.com/v1/", responses=responses
        ),
        config={"default_model": "gpt-6-astra", "enable_long_context": True, **config},
    )
    return provider, responses, window


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["gpt-6-astra", "gpt-6.1-sol", "gpt-5.6"])
@pytest.mark.parametrize("output", ["COMPLETED_RESULT", [{"type": "input_text", "text": "COMPLETED_RESULT"}]])
async def test_native_window_results_survive_count_dispatch_and_second_compaction(model, output):
    provider, responses, window = fixture(
        default_model=model, use_streaming=False, max_retries=0,
    )
    events = []
    async def observe(name, data):
        if name == "provider:request_attribution":
            events.append(copy.deepcopy(data))
    provider.coordinator = SimpleNamespace(hooks=SimpleNamespace(emit=observe))
    window.insert(2, {"type": "function_call_output", "call_id": "c1", "output": output})
    responses.compact.return_value.output = copy.deepcopy(window)

    async def count(**params):
        # Deterministic wire-size oracle, not vendor tokenization.
        payload = {key: value for key, value in params.items() if key != "timeout"}
        return SimpleNamespace(input_tokens=len(json.dumps(payload, sort_keys=True)))

    responses.input_tokens.count.side_effect = count
    responses.create.return_value = SimpleNamespace(
        status="completed", output=[], usage=SimpleNamespace(input_tokens=100),
    )
    first = await provider.compact_context(
        ChatRequest(messages=[Message(role="user", content="Original task")])
    )
    assert events[-1]["binding"] == "subset_matched"
    assert events[-1]["operation"] == "responses.compact"
    assert events[-1]["dispatch"] == first["measurement"]["compact_input"]
    original_carrier = copy.deepcopy(first["message"])
    tools = [ToolSpec(name="read", description="Read evidence", parameters={"type": "object", "properties": {}})]
    request = ChatRequest(
        messages=[
            Message(role="system", content="Accepted instruction: never repeat completed effects."),
            Message(**first["message"]),
            Message(role="assistant", content="", tool_calls=[
                {"id": "new-call", "name": "read", "arguments": {}},
            ]),
            Message(role="tool", tool_call_id="new-call", content="NEW_RESULT"),
            Message(role="user", content="Correction: keep the completed result."),
        ],
        tools=tools,
        max_output_tokens=1000,
    )
    original_request = request.model_dump()
    decision = await provider.request_budget(request, context_estimate=0)
    counted = responses.input_tokens.count.call_args.kwargs
    assert counted["input"][:len(window)] == window
    assert counted["model"] == model
    assert counted["instructions"] == "Accepted instruction: never repeat completed effects."
    fresh_output = next(item for item in counted["input"] if item.get("call_id") == "new-call" and item["type"] == "function_call_output")
    assert fresh_output["output"][-1]["prompt_cache_breakpoint"] == {"mode": "explicit"}
    assert decision["measurement"]["input_tokens"] == len(json.dumps(
        {key: value for key, value in counted.items() if key != "timeout"}, sort_keys=True,
    ))
    await provider.complete(request)
    sent = responses.create.call_args.kwargs
    for key in ("model", "input", "instructions", "tools"):
        assert sent[key] == counted[key]
    first_attribution = decision["measurement"]["attribution"]
    assert events[-1]["binding"] == "matched"
    assert events[-1]["dispatch"] == first_attribution["request"]
    assert events[-1]["attribution"]["count_id"] != first_attribution["count_id"]
    assert responses.create.await_count == 1
    assert request.model_dump() == original_request
    assert first["message"] == original_carrier

    # The next native operation receives the first window unchanged, including
    # results before opaque state; new suffix items remain ordinarily serialized.
    second_window = copy.deepcopy(window)
    second_window[3]["id"] = "cmp2"
    second_window.append({"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Second correction"}]})
    responses.compact.return_value.output = copy.deepcopy(second_window)
    second = await provider.compact_context(request)
    assert responses.compact.call_args.kwargs["input"][:len(window)] == window
    second_request = ChatRequest(messages=[
        Message(**second["message"]), Message(role="user", content="Continue twice"),
    ])
    second_decision = await provider.request_budget(second_request, context_estimate=0)
    second_counted = responses.input_tokens.count.call_args.kwargs
    assert second_counted["input"][:len(second_window)] == second_window
    assert second["message"]["metadata"]["openai:compaction"]["output"] == second_window
    assert responses.compact.await_count == 2
    assert responses.create.await_count == 1
    second_attribution = second_decision["measurement"]["attribution"]
    assert second_attribution["scope_id"] == first_attribution["scope_id"]
    assert second_attribution["request"]["payload_tag"] != first_attribution["request"]["payload_tag"]
    await provider.complete(second_request)
    assert responses.create.call_args.kwargs["input"] == second_counted["input"]
    assert events[-1]["binding"] == "matched"
    assert events[-1]["dispatch"] == second_attribution["request"]
    assert responses.create.await_count == 2
    print(json.dumps({
        "model": model, "output_shape": type(output).__name__,
        "first_window_sha256": hashlib.sha256(json.dumps(window, sort_keys=True).encode()).hexdigest(),
        "second_window_sha256": hashlib.sha256(json.dumps(second_window, sort_keys=True).encode()).hexdigest(),
        "first_count": decision["measurement"]["input_tokens"],
        "compact_calls": responses.compact.await_count,
        "completion_calls": responses.create.await_count,
    }, sort_keys=True))


@pytest.mark.asyncio
async def test_canonical_results_reach_real_sdk_json_and_incomplete_continuation_unchanged():
    import httpx
    from openai import AsyncOpenAI

    _, _, window = fixture()
    window.insert(2, {"type": "function_call_output", "call_id": "c1", "output": "SAME_RESULT"})
    window.append({"type": "function_call_output", "call_id": "retained", "output": [
        {"type": "input_text", "text": "RETAINED_AFTER_OPAQUE"},
    ]})
    captured = []
    events = []
    async def observe(name, data):
        if name == "provider:request_attribution":
            events.append(copy.deepcopy(data))

    async def handler(request):
        body = json.loads(request.content)
        captured.append((request.url.path, body))
        if request.url.path.endswith("/input_tokens"):
            return httpx.Response(200, json={"input_tokens": len(request.content)})
        generation = sum(path.endswith("/responses") for path, _ in captured)
        return httpx.Response(200, json={
            "id": f"resp_{generation}", "object": "response", "created_at": 1,
            "status": "incomplete" if generation == 1 else "completed",
            "incomplete_details": None, "model": "gpt-6.1-sol",
            "output": [], "usage": {"input_tokens": 100, "output_tokens": 1, "total_tokens": 101},
        })

    client = AsyncOpenAI(api_key="offline", base_url="https://api.openai.com/v1",
                         http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
                         max_retries=0)
    provider = OpenAIProvider(api_key="offline", client=client, config={
        "default_model": "gpt-6.1-sol", "use_streaming": False, "max_retries": 0,
    }, coordinator=SimpleNamespace(hooks=SimpleNamespace(emit=observe)))
    request = ChatRequest(messages=[
        Message(**compacted_message("gpt-6.1-sol", window)),
        Message(role="assistant", content="", tool_calls=[{"id": "new", "name": "read", "arguments": {}}]),
        Message(role="tool", tool_call_id="new", content="SAME_RESULT"),
        Message(role="user", content="Continue"),
    ])
    before = request.model_dump()
    try:
        await provider.complete(request)
    finally:
        await client.close()
    generations = [body for path, body in captured if path.endswith("/responses")]
    assert len(generations) == 2
    counts = [body for path, body in captured if path.endswith("/input_tokens")]
    assert counts
    assert len(events) == len(generations) == len(counts) == 2
    for event, count, generation in zip(events, counts, generations, strict=True):
        assert event["binding"] == "matched"
        assert event["dispatch"] == provider._request_attribution.describe(count)
        assert event["dispatch"] == provider._request_attribution.describe({
            key: generation[key] for key in provider._COUNT_FIELDS if key in generation
        })
    assert events[0]["dispatch_id"] != events[1]["dispatch_id"]
    for body in [*counts, *generations]:
        assert body["input"][:len(window)] == window
        fresh = next(item for item in body["input"] if item.get("call_id") == "new" and item["type"] == "function_call_output")
        assert fresh["output"] == [{"type": "input_text", "text": "SAME_RESULT",
                                   "prompt_cache_breakpoint": {"mode": "explicit"}}]
    assert request.model_dump() == before


@pytest.mark.asyncio
async def test_compact_returns_all_canonical_items_then_passes_them_as_is():
    provider, responses, window = fixture()
    request = ChatRequest(
        messages=[
            Message(role="system", content="Stable instructions"),
            Message(role="user", content="Original task"),
        ],
        max_output_tokens=1500,
    )
    before = request.model_dump()
    result = await provider.compact_context(request)
    params = responses.compact.call_args.kwargs
    assert set(params) == {"model", "input", "instructions", "timeout"}
    assert params["timeout"].read is None
    assert params["instructions"] == "Stable instructions"
    assert request.model_dump() == before
    continuation = ChatRequest(
        messages=[
            Message(**result["message"]),
            Message(role="user", content="Continue"),
        ]
    )
    wire = provider._budget_params(continuation)["input"]
    assert wire[:-1] == window
    assert len(wire) == len(window) + 1
    assert result["usage"] == {"input_tokens": 100, "output_tokens": 20}
    wire[0]["content"][0]["text"] = "caller mutation"
    assert result["message"]["metadata"]["openai:compaction"]["output"] == window
    responses.create.assert_not_called()


@pytest.mark.asyncio
async def test_repeat_compaction_sends_previous_entire_window_and_new_messages():
    provider, responses, window = fixture()
    first = await provider.compact_context(
        ChatRequest(messages=[Message(role="user", content="Original")])
    )
    await provider.compact_context(
        ChatRequest(
            messages=[
                Message(**first["message"]),
                Message(role="user", content="New steering"),
            ]
        )
    )
    assert responses.compact.call_args.kwargs["input"][:-1] == window


@pytest.mark.asyncio
async def test_oversized_native_input_is_rejected_before_compact_dispatch():
    provider, responses, _ = fixture()
    responses.input_tokens.count.return_value = SimpleNamespace(input_tokens=2_000_000)
    with pytest.raises(ContextLengthError):
        await provider.compact_context(
            ChatRequest(messages=[Message(role="user", content="Synthetic")])
        )
    responses.compact.assert_not_awaited()


def test_model_override_applies_to_budget_and_canonical_identity():
    provider, _, window = fixture()
    request = ChatRequest(
        model="gpt-5.6-terra", messages=[Message(role="user", content="Summary")]
    )
    assert provider._budget_params(request)["model"] == "gpt-5.6-terra"
    assert (
        provider._budget_params(request, model="gpt-6-astra")["model"] == "gpt-6-astra"
    )
    wrong = ChatRequest(
        model="gpt-5.6-terra",
        messages=[Message(**compacted_message("gpt-6-astra", window))],
    )
    with pytest.raises(ValueError, match="different model"):
        provider._budget_params(wrong)


def test_proxy_does_not_inherit_native_compaction_claim():
    provider, _, _ = fixture()
    provider._client.base_url = "https://proxy.invalid/v1"
    assert provider.supports_native_compaction() is False


def test_empty_or_noncompacted_output_fails_closed():
    for window in ([], [{"type": "message", "role": "user", "content": []}]):
        with pytest.raises(ValueError):
            compacted_message("gpt-6-astra", window)


def test_native_transport_validation_rejects_missing_payload_without_changing_window():
    provider, _, window = fixture()
    carrier = compacted_message(provider.default_model, window)
    original = copy.deepcopy(carrier)
    assert provider.validate_compacted_context(carrier)
    assert carrier == original
    assert not provider.validate_compacted_context(
        {"role": "user", "content": "Placeholder", "metadata": {}}
    )


def test_validation_uses_explicit_continuation_model_without_changing_provider_default():
    provider, _, window = fixture()
    default = provider.default_model
    selected = "gpt-5.6-terra" if default != "gpt-5.6-terra" else "gpt-6-astra"
    carrier = compacted_message(selected, window)
    assert provider.validate_compacted_context(carrier, model=selected)
    with pytest.raises(ValueError, match="different model"):
        provider.validate_compacted_context(carrier)
    assert provider.default_model == default


@pytest.mark.asyncio
async def test_sdk_defaults_do_not_become_unknown_input_parameters():
    from pydantic import BaseModel

    class RetainedItem(BaseModel):
        type: str
        role: str
        content: list
        status: str | None = None
        phase: str | None = None

    raw = {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": "Fact"}],
        "status": None,
    }
    item = RetainedItem(**raw)
    assert "phase" in item.model_dump()  # SDK-invented default, rejected by API.
    provider, responses, window = fixture()
    responses.compact.return_value.output = [item, window[2]]
    result = await provider.compact_context(
        ChatRequest(messages=[Message(role="user", content="Task")])
    )
    saved = result["message"]["metadata"]["openai:compaction"]["output"]
    assert saved[0] == raw  # Retain explicit null, omit only absent SDK defaults.
