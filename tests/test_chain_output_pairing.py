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
- an fc_-prefixed output id that matches NO chained call is dropped,
  loudly — it can pair with nothing server-side;
- an fc_-prefixed output id that MATCHES the chained turn's local record
  (both sides originate from the same ToolCallBlock.id, so this is the
  realistic mis-keyed state) keeps the REAL output: dropping it only for
  the orphan branch to synthesize an error under the SAME unpairable id
  destroys the tool result and repairs nothing;
- stitched-turn dispatch keys by call_id even for completed calls
  (trial-3's exact flavor).

The composed-state tests below run the WHOLE pairing function against each
realistic input state (the 3-scenario table), not one branch at a time —
the original defect lived exactly in the branches' composition.
"""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

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
# Native result envelopes count as paired
#
# apply_patch and computer_use REQUIRE their own output envelope; a generic
# function_call_output is not accepted for them. Counting only
# function_call_output made every chained native call look orphaned, so its
# genuine result was shipped alongside a synthesized "result missing" error
# for the SAME call_id -- two contradictory results for one call.
# ---------------------------------------------------------------------------


def test_apply_patch_call_output_counts_as_paired():
    """A successful native apply_patch result must not be called orphaned."""
    provider = _make_provider()
    chained_msg = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "call_patch1", "name": "apply_patch"}],
    }
    delta_input = [
        {
            "type": "apply_patch_call_output",
            "call_id": "call_patch1",
            "output": "M src/thing.py",
            "status": "completed",
        }
    ]
    result = _run_pairing(provider, [dict(i) for i in delta_input], chained_msg)

    assert not [i for i in result if i.get("type") == "function_call_output"], (
        "synthesized an error output for an apply_patch call that already had a "
        "real, successful result -- the model would see two contradictory "
        "results for one call and may re-apply an applied patch"
    )
    assert result == delta_input, "real apply_patch output was altered or dropped"


def test_computer_call_output_counts_as_paired():
    """computer_use has the same native-envelope exposure as apply_patch."""
    provider = _make_provider()
    chained_msg = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "call_cua1", "name": "computer"}],
    }
    delta_input = [
        {
            "type": "computer_call_output",
            "call_id": "call_cua1",
            "output": {"type": "computer_screenshot", "image_url": "data:,"},
        }
    ]
    result = _run_pairing(provider, [dict(i) for i in delta_input], chained_msg)

    assert not [i for i in result if i.get("type") == "function_call_output"]
    assert result == delta_input


def test_mixed_native_and_function_outputs_all_count():
    """A turn mixing envelope types pairs every call, synthesizing none."""
    provider = _make_provider()
    chained_msg = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": "call_patch1", "name": "apply_patch"},
            {"id": "call_bash1", "name": "bash"},
        ],
    }
    delta_input = [
        {
            "type": "apply_patch_call_output",
            "call_id": "call_patch1",
            "output": "A new.py",
            "status": "completed",
        },
        {"type": "function_call_output", "call_id": "call_bash1", "output": "ok"},
    ]
    result = _run_pairing(provider, [dict(i) for i in delta_input], chained_msg)

    assert result == delta_input
    assert not any("[error]" in str(i.get("output", "")) for i in result)


def test_orphaned_native_call_still_repaired():
    """The safety net still fires when a native result is genuinely absent."""
    provider = _make_provider()
    chained_msg = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "call_patch_missing", "name": "apply_patch"}],
    }
    result = _run_pairing(provider, [], chained_msg)

    outputs = [i for i in result if i.get("type") == "function_call_output"]
    assert len(outputs) == 1
    assert outputs[0]["call_id"] == "call_patch_missing"
    assert "[error]" in outputs[0]["output"]


def test_fc_keyed_native_output_is_not_dropped():
    """The fc_ drop stays scoped to function_call_output.

    Dropping a native output would destroy a real, otherwise unrecoverable
    tool result. An fc_-keyed native envelope is still not a valid pairing,
    so the orphan repair for the true call_id must still fire.
    """
    provider = _make_provider()
    chained_msg = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "call_patch2", "name": "apply_patch"}],
    }
    delta_input = [
        {
            "type": "apply_patch_call_output",
            "call_id": "fc_abc123",
            "output": "M src/thing.py",
            "status": "completed",
        }
    ]
    result = _run_pairing(provider, [dict(i) for i in delta_input], chained_msg)

    assert delta_input[0] in result, "real native output was destroyed"
    synthesized = [i for i in result if i.get("type") == "function_call_output"]
    assert len(synthesized) == 1
    assert synthesized[0]["call_id"] == "call_patch2"


# ---------------------------------------------------------------------------
# Composed-state contract — whole function, one test per realistic state.
# Expected ids and output call_ids originate from the SAME ToolCallBlock.id
# (outputs get tool_call_id copied from it), so both branches of the pairing
# function see one keyspace. The 3 realistic states:
#   A: fc_/fc_   — legacy mis-keyed record; output id matches the local call
#   B: call_/fc_ — output keyed by an id matching no chained call
#   C: call_/call_ — healthy
# ---------------------------------------------------------------------------


def test_composed_state_a_fc_call_and_fc_output_keeps_real_result():
    """Scenario A (the composition defect): chained call AND output both keyed
    by the same fc_ id. Pre-fix, the fc_-drop branch destroyed the real output
    and the orphan branch synthesized an error under the SAME unpairable id —
    still 400s AND the result is gone. The real output must be kept."""
    provider = _make_provider()
    chained_msg = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "fc_abc123", "name": "bash"}],
    }
    delta_input = [
        {
            "type": "function_call_output",
            "call_id": "fc_abc123",
            "output": "REAL RESULT",
        }
    ]
    result = _run_pairing(provider, delta_input, chained_msg)

    outputs = [i for i in result if i.get("type") == "function_call_output"]
    assert len(outputs) == 1, "exactly one output: no drop, no extra synthesis"
    assert outputs[0]["call_id"] == "fc_abc123"
    assert outputs[0]["output"] == "REAL RESULT", (
        "the real tool result was destroyed by the drop/synthesize composition"
    )
    assert "[error]" not in outputs[0]["output"]


def test_composed_state_b_call_expected_fc_output_drops_and_synthesizes():
    """Scenario B: chained call keyed call_, output keyed by an unmatched fc_
    id. The fc_ output pairs with nothing; drop it and synthesize an error
    for the orphaned call so the chained request cannot 400."""
    provider = _make_provider()
    chained_msg = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "call_xyz789", "name": "bash"}],
    }
    delta_input = [
        {
            "type": "function_call_output",
            "call_id": "fc_abc123",
            "output": "REAL RESULT",
        }
    ]
    result = _run_pairing(provider, delta_input, chained_msg)

    outputs = [i for i in result if i.get("type") == "function_call_output"]
    ids = {o["call_id"] for o in outputs}
    assert ids == {"call_xyz789"}, "fc_ output dropped; orphan repaired by call_id"
    assert "[error]" in outputs[0]["output"]


def test_composed_state_c_healthy_pairing_passes_through_untouched():
    """Scenario C: call_/call_ — nothing dropped, nothing synthesized."""
    provider = _make_provider()
    chained_msg = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "call_xyz789", "name": "bash"}],
    }
    delta_input = [
        {
            "type": "function_call_output",
            "call_id": "call_xyz789",
            "output": "REAL RESULT",
        }
    ]
    result = _run_pairing(provider, [dict(i) for i in delta_input], chained_msg)
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


# ---------------------------------------------------------------------------
# Canonical repair-event name
#
# "a tool call's result went missing and the provider patched over it" is an
# ECOSYSTEM concept, not an OpenAI one. The kernel registers it as
# provider:tool_sequence_repaired and six providers emit it -- including this
# one, for the message-level repair. An unregistered name gets no
# hooks-logging handler, so every emission is silently discarded before it
# reaches events.jsonl and the repair becomes invisible to every consumer.
# ---------------------------------------------------------------------------


def _make_provider_with_hooks() -> tuple[OpenAIProvider, AsyncMock]:
    emit = AsyncMock()
    coordinator = MagicMock()
    coordinator.hooks.emit = emit
    coordinator.get_capability = MagicMock(return_value=None)
    provider = OpenAIProvider(
        api_key="test-key",
        config={"max_retries": 0, "use_streaming": False},
        coordinator=coordinator,
    )
    return provider, emit


def test_emitted_repair_event_is_registered_in_the_kernel():
    """An unregistered event name is dropped before it reaches events.jsonl."""
    from amplifier_core.events import ALL_EVENTS

    assert "provider:tool_sequence_repaired" in ALL_EVENTS
    assert "provider:chain_pairing_repaired" not in ALL_EVENTS, (
        "if this name ever becomes registered, revisit whether the chain "
        "repair genuinely warrants a second name for one concept"
    )


def test_chain_repair_emits_the_canonical_event():
    provider, emit = _make_provider_with_hooks()
    chained_msg = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "call_missing1", "tool": "apply_patch"}],
    }
    _run_pairing(provider, [], chained_msg)

    assert emit.await_count == 1
    name, payload = emit.await_args.args[0], emit.await_args.args[1]
    assert name == "provider:tool_sequence_repaired", (
        "emitted a name the kernel does not register -- hooks-logging "
        "attaches no handler and the repair never reaches events.jsonl"
    )
    assert payload["provider"] == provider.name
    assert payload["repair_count"] == 1
    assert payload["repairs"] == [
        {"tool_call_id": "call_missing1", "tool_name": "apply_patch"}
    ]
    assert payload["repair_site"] == "chain_pairing", (
        "the two repair sites must stay distinguishable under one event name"
    )
    assert payload["repair_count"] == len(payload["repairs"])
    assert payload["dropped_count"] == 0
    assert payload["synthesized_for"] == ["call_missing1"]


def test_tool_name_read_from_the_canonical_name_key():
    """The canonical ToolCall shape carries the tool under "name".

    amplifier_core.message_models.ToolCall serializes to "name"; the
    streaming orchestrator writes "tool" (covered above). Both shapes
    genuinely reach this code, so both branches of the lookup are exercised.
    """
    provider, emit = _make_provider_with_hooks()
    chained_msg = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "call_canonical", "name": "read_file"}],
    }
    _run_pairing(provider, [], chained_msg)

    payload = emit.await_args.args[1]
    assert payload["repairs"] == [
        {"tool_call_id": "call_canonical", "tool_name": "read_file"}
    ]


def test_tool_name_falls_back_to_unknown_when_the_record_omits_it():
    provider, emit = _make_provider_with_hooks()
    chained_msg = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "call_noname"}],
    }
    _run_pairing(provider, [], chained_msg)

    payload = emit.await_args.args[1]
    assert payload["repairs"] == [
        {"tool_call_id": "call_noname", "tool_name": "unknown"}
    ]


def test_no_event_when_nothing_was_repaired():
    provider, emit = _make_provider_with_hooks()
    chained_msg = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "call_ok", "tool": "bash"}],
    }
    delta = [{"type": "function_call_output", "call_id": "call_ok", "output": "ok"}]
    _run_pairing(provider, delta, chained_msg)

    assert emit.await_count == 0, "a clean turn must stay silent"


def test_dropped_only_turn_reports_zero_repairs_and_one_drop():
    """Dropping is not synthesizing; the payload must say so explicitly.

    Every genuine call is correctly paired here, but a stray fc_-keyed
    output is dropped. repair_count must stay 0 and equal len(repairs) --
    the invariant sibling providers hold and cross-provider repair-volume
    aggregation depends on -- while dropped_count carries the real signal.
    The emission is deliberately not gated on repair_count > 0: a dropped
    output is precisely what went unobserved before this change.
    """
    provider, emit = _make_provider_with_hooks()
    chained_msg = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "call_x", "tool": "bash"}],
    }
    delta = [
        {"type": "function_call_output", "call_id": "call_x", "output": "ok"},
        {"type": "function_call_output", "call_id": "fc_stray", "output": "orphan"},
    ]
    _run_pairing(provider, delta, chained_msg)

    assert emit.await_count == 1, "a dropped output must still be reported"
    payload = emit.await_args.args[1]
    assert payload["repair_count"] == 0
    assert payload["repairs"] == []
    assert payload["repair_count"] == len(payload["repairs"]), (
        "repair_count must always equal len(repairs) — five sibling providers "
        "hold this invariant and cross-provider aggregation relies on it"
    )
    assert payload["dropped_count"] == 1
    assert payload["dropped_item_id_outputs"] == ["fc_stray"]


def test_multiple_orphans_report_every_repair():
    """repair_count > 1 at the chain site, over its own expected_names map."""
    provider, emit = _make_provider_with_hooks()
    chained_msg = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": "call_a", "tool": "apply_patch"},
            {"id": "call_b", "name": "read_file"},
            {"id": "call_c"},
        ],
    }
    delta = [{"type": "function_call_output", "call_id": "call_b", "output": "ok"}]
    _run_pairing(provider, delta, chained_msg)

    payload = emit.await_args.args[1]
    assert payload["repair_count"] == 2
    assert payload["repair_count"] == len(payload["repairs"])
    assert payload["repairs"] == [
        {"tool_call_id": "call_a", "tool_name": "apply_patch"},
        {"tool_call_id": "call_c", "tool_name": "unknown"},
    ]
    assert payload["dropped_count"] == 0


def test_content_block_tool_name_reaches_the_emitted_event():
    """Calls recorded as content blocks (no tool_calls field) name their tool.

    The content-block branch populates expected_names separately from the
    tool_calls branch. Exercised through the emission rather than the return
    value, because the payload is the only place that lookup surfaces.
    """
    provider, emit = _make_provider_with_hooks()
    chained_msg = {
        "role": "assistant",
        "content": [
            {"type": "tool_call", "id": "call_block1", "name": "write_file"},
        ],
    }
    _run_pairing(provider, [], chained_msg)

    payload = emit.await_args.args[1]
    assert payload["repair_count"] == 1
    assert payload["repairs"] == [
        {"tool_call_id": "call_block1", "tool_name": "write_file"}
    ]


def test_kept_fc_output_is_counted_as_kept_not_dropped():
    """An fc_-keyed output has TWO outcomes, and the payload must say which.

    dropped_count alone under-reports fc_ keying anomalies once the matching
    fc_/fc_ case is kept rather than dropped: the anomaly is just as real,
    just as much an upstream dispatch bug, and the outcome differs only in
    whether the payload survived. Counting it nowhere would re-hide exactly
    the signal this event was fixed to surface.
    """
    provider, emit = _make_provider_with_hooks()
    chained_msg = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "fc_abc123", "tool": "bash"}],
    }
    delta = [
        {"type": "function_call_output", "call_id": "fc_abc123", "output": "REAL"},
    ]
    _run_pairing(provider, delta, chained_msg)

    assert emit.await_count == 1, "a kept fc_ anomaly must still be reported"
    payload = emit.await_args.args[1]
    # Nothing was synthesized and nothing was discarded -- the real output
    # paired against the chained turn's (identically mis-keyed) local record.
    assert payload["repair_count"] == 0
    assert payload["repairs"] == []
    assert payload["repair_count"] == len(payload["repairs"])
    assert payload["dropped_count"] == 0
    assert payload["dropped_item_id_outputs"] == []
    assert payload["kept_count"] == 1
    assert payload["kept_item_id_outputs"] == ["fc_abc123"]
    assert payload["repair_site"] == "chain_pairing"


def test_total_fc_anomalies_are_dropped_plus_kept():
    """A turn carrying both outcomes at once must account for both.

    `dropped_count + kept_count` is the documented way to recover total fc_
    keying anomalies; a consumer reading dropped_count alone would see 1
    where 2 occurred.
    """
    provider, emit = _make_provider_with_hooks()
    chained_msg = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": "fc_kept1", "tool": "bash"},  # matches -> kept
            {"id": "call_ok", "tool": "read_file"},  # healthy pairing
        ],
    }
    delta = [
        {"type": "function_call_output", "call_id": "fc_kept1", "output": "REAL"},
        {"type": "function_call_output", "call_id": "call_ok", "output": "ok"},
        {"type": "function_call_output", "call_id": "fc_stray", "output": "orphan"},
    ]
    _run_pairing(provider, delta, chained_msg)

    payload = emit.await_args.args[1]
    assert payload["kept_count"] == 1
    assert payload["kept_item_id_outputs"] == ["fc_kept1"]
    assert payload["dropped_count"] == 1
    assert payload["dropped_item_id_outputs"] == ["fc_stray"]
    assert payload["dropped_count"] + payload["kept_count"] == 2, (
        "total fc_ keying anomalies must be recoverable from the payload"
    )
    # Neither anomaly orphaned a genuine call, so nothing was synthesized.
    assert payload["repair_count"] == 0
    assert payload["repair_count"] == len(payload["repairs"])
