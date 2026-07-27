"""Pathway-2 regression tests: chain-aware output pairing on the delta path.

Reproduces the second orphaned-call pathway (pathway-2 forensics, run
20260727T072546Z): a request that chains via previous_response_id sends only
the delta (tool outputs) while the function_call items live SERVER-SIDE in
the chained response. If an output is keyed by the Responses-API item id
(fc_…) instead of call_id (call_…), the server-side call finds no output and
the API 400s ("No tool output found for function call …"), killing the
session. Trial-3 proved truncation is NOT required: a COMPLETED,
successfully-executed tool call died this way.

Invariants:
- every tool call on the chained-from assistant turn has a
  function_call_output paired BY call_id in the delta input (synthesized
  error output for orphans);
- fc_-prefixed output ids never reach the wire on the chain path (dropped,
  loudly — they can pair with nothing server-side);
- stitched-turn dispatch keys by call_id even for completed calls
  (trial-3's exact flavor).
"""

import asyncio
import json
from pathlib import Path

from amplifier_core.message_models import ChatRequest, Message, ToolCallBlock
from openai.types.responses import Response

from amplifier_module_provider_openai import (
    OpenAIChatResponse,
    OpenAIProvider,
)
from amplifier_module_provider_openai._constants import METADATA_RESPONSE_ID
from amplifier_module_provider_openai._response_handling import (
    convert_response_with_accumulated_output,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "responses"


def _make_provider(config: dict | None = None) -> OpenAIProvider:
    return OpenAIProvider(
        api_key="test-key",
        config={"max_retries": 0, "use_streaming": False, **(config or {})},
    )


def _completed_response() -> Response:
    with open(FIXTURES_DIR / "gpt-5-5-basic.json") as f:
        return Response.model_validate(json.load(f))


# ---------------------------------------------------------------------------
# _enforce_chain_output_pairing — direct helper contract
# ---------------------------------------------------------------------------


def _run_pairing(provider, delta_input, chained_msg):
    return asyncio.run(provider._enforce_chain_output_pairing(delta_input, chained_msg))


def test_orphaned_chained_call_gets_synthesized_output_and_fc_dropped():
    """Trial-3's fatal shape: chained call keyed call_, output keyed fc_."""
    provider = _make_provider()
    chained_msg = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "call_uLhmISUqpuXS1pC54nDZqypj", "name": "load_skill"}],
    }
    delta_input = [
        {
            "type": "function_call_output",
            "call_id": "fc_07d52d8589554c84006",
            "output": "skill loaded ok",
        }
    ]
    result = _run_pairing(provider, delta_input, chained_msg)

    outputs = [i for i in result if i.get("type") == "function_call_output"]
    ids = {o["call_id"] for o in outputs}
    assert "call_uLhmISUqpuXS1pC54nDZqypj" in ids, "orphaned chained call not repaired"
    assert not any(i.startswith("fc_") for i in ids), (
        "fc_-keyed output reached the wire"
    )


def test_properly_paired_delta_is_untouched():
    provider = _make_provider()
    chained_msg = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "call_ok", "name": "bash"}],
    }
    delta_input = [
        {"type": "function_call_output", "call_id": "call_ok", "output": "done"},
        {"role": "user", "content": [{"type": "input_text", "text": "next"}]},
    ]
    result = _run_pairing(provider, [dict(i) for i in delta_input], chained_msg)
    outputs = [i for i in result if i.get("type") == "function_call_output"]
    assert len(outputs) == 1
    assert outputs[0]["call_id"] == "call_ok"
    assert outputs[0]["output"] == "done", "real result must not be replaced"


def test_chained_calls_from_content_blocks_are_recognized():
    """Assistant tool calls stored as content blocks (not the tool_calls field)."""
    provider = _make_provider()
    chained_msg = {
        "role": "assistant",
        "content": [
            {"type": "tool_call", "id": "call_blk", "name": "read_file", "input": {}}
        ],
    }
    result = _run_pairing(provider, [], chained_msg)
    outputs = [i for i in result if i.get("type") == "function_call_output"]
    assert len(outputs) == 1 and outputs[0]["call_id"] == "call_blk"
    assert "[error]" in outputs[0]["output"]


def test_no_chained_calls_and_clean_delta_is_noop():
    provider = _make_provider()
    delta_input = [{"role": "user", "content": [{"type": "input_text", "text": "hi"}]}]
    result = _run_pairing(
        provider, list(delta_input), {"role": "assistant", "content": "x"}
    )
    assert result == delta_input


# ---------------------------------------------------------------------------
# Stitched-dispatch regression — trial-3's flavor at the converter level
# ---------------------------------------------------------------------------


def test_stitched_completed_call_dispatches_by_call_id_not_item_id():
    """continuation_count=1, COMPLETED call with valid args and both ids:
    dispatch must key by call_id (pre-fix code keyed by item id — the fc_ leak)."""
    final = _completed_response()
    accumulated = [
        {
            "type": "function_call",
            "id": "fc_07d52d8589554c84006",
            "call_id": "call_uLhmISUqpuXS1pC54nDZqypj",
            "name": "load_skill",
            "arguments": '{"skill_name": "gitea"}',
            "status": "completed",
        }
    ]
    result = convert_response_with_accumulated_output(
        final, accumulated, 1, OpenAIChatResponse
    )
    assert result.tool_calls is not None
    assert result.tool_calls[0].id == "call_uLhmISUqpuXS1pC54nDZqypj"
    assert result.tool_calls[0].arguments == {"skill_name": "gitea"}


# ---------------------------------------------------------------------------
# Full chain path through complete() — the pairing runs on the wire input
# ---------------------------------------------------------------------------


def test_chain_delta_request_carries_repaired_pairing():
    provider = _make_provider({"enable_response_chaining": True})
    captured: list[dict] = []

    async def _mock(**kwargs):
        captured.append(kwargs)
        return _completed_response()

    provider.client.responses.create = _mock  # pyright: ignore[reportAttributeAccessIssue]

    request = ChatRequest(
        messages=[
            Message(role="user", content="find the discrepancies"),
            Message(
                role="assistant",
                content=[
                    ToolCallBlock(
                        id="call_uLhm",
                        name="load_skill",
                        input={"skill_name": "gitea"},
                    )
                ],
                metadata={METADATA_RESPONSE_ID: "resp_07d52d8589554c84006"},
            ),
            # The tool result came back keyed by the ITEM id — the bug shape.
            Message(
                role="tool",
                content="skill loaded",
                tool_call_id="fc_07d52d85",
                name="load_skill",
            ),
        ]
    )
    asyncio.run(provider.complete(request))

    assert captured, "no API call captured"
    params = captured[0]
    assert params.get("previous_response_id") == "resp_07d52d8589554c84006"
    outputs = [
        i
        for i in params["input"]
        if isinstance(i, dict) and i.get("type") == "function_call_output"
    ]
    ids = {o["call_id"] for o in outputs}
    assert "call_uLhm" in ids, "chained call has no paired output — would 400"
    assert not any(i.startswith("fc_") for i in ids), (
        "fc_-keyed output reached the wire"
    )
