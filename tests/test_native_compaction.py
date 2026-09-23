import copy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from amplifier_core import ChatRequest, Message
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


def fixture():
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
        config={"default_model": "gpt-6-astra", "enable_long_context": True},
    )
    return provider, responses, window


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
