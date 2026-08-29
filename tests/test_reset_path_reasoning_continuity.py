"""Tests for Change A: unified chain_will_attach predicate.

Bug: the provider answered "is the server holding my reasoning state for
this call?" with three different expressions (chain_active, store_enabled,
previous_response_id is not None) evaluated at three different points in the
request build. They disagreed on exactly the post-compaction chain-reset
path: message conversion was keyed on chain_active (still True post-reset,
so reasoning-item re-insertion was suppressed), while previous_response_id
was dropped by the reset and the include gate was ALSO keyed on chain_active
(so ciphertext was never requested either). Net effect: a post-compaction
request replayed function_call items with ZERO of their originating
reasoning items -- OpenAI's #1 documented Responses-API migration error --
executed on every post-compaction turn, silently (no 400; the model just
re-plans from scratch).

Fix: compute `chain_will_attach` ONCE, before message conversion, from
inputs already resolved by then (previous_response_id, the compaction reset,
chaining permission, store_enabled) and key every downstream decision on it:
message conversion's skip_reasoning_reinsertion, the previous_response_id
attach predicate, and (per probe P1, Change A5) the `include` gate.

Harness mirrors test_response_chaining.py / test_chain_reset_on_compaction.py
(AsyncMock client, asyncio.run(), no live API calls).
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import openai
from amplifier_core.message_models import (
    ChatRequest,
    Message,
    TextBlock,
    ThinkingBlock,
    ToolCallBlock,
)
from httpx import Request as HttpxRequest
from httpx import Response as HttpxResponse

from amplifier_module_provider_openai import OpenAIProvider
from amplifier_module_provider_openai._constants import METADATA_RESPONSE_ID

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_provider(**config_overrides) -> OpenAIProvider:
    config = {"max_retries": 0, "use_streaming": False, **config_overrides}
    return OpenAIProvider(api_key="test-key", config=config)


class DummyResponse:
    """Minimal response stub -- matches the shape _convert_to_chat_response() needs."""

    def __init__(self, response_id: str = "resp_new"):
        self.output = [
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text="Hi")],
            )
        ]
        self.usage = SimpleNamespace(input_tokens=1, output_tokens=1)
        self.status = "completed"
        self.id = response_id


def _captured_params(provider: OpenAIProvider) -> Any:
    mock = cast(AsyncMock, provider.client.responses.create)
    return mock.call_args.kwargs


def _all_calls(provider: OpenAIProvider) -> list[Any]:
    mock = cast(AsyncMock, provider.client.responses.create)
    return [call.kwargs for call in mock.call_args_list]


def _make_not_found_error(code: str = "response_not_found") -> openai.NotFoundError:
    req = HttpxRequest("GET", "https://api.openai.com/v1/responses")
    resp = HttpxResponse(404, request=req)
    return openai.NotFoundError(
        message=f"response not found: {code}",
        response=resp,
        body={"error": {"code": code, "message": "Response not found"}},
    )


def _make_context_length_error() -> openai.BadRequestError:
    req = HttpxRequest("POST", "https://api.openai.com/v1/responses")
    resp = HttpxResponse(400, request=req)
    return openai.BadRequestError(
        message="This model's maximum context length is 400000 tokens",
        response=resp,
        body={
            "error": {
                "code": "context_length_exceeded",
                "message": "This model's maximum context length is 400000 tokens",
            }
        },
    )


def _reasoning_thinking_block(tag: str) -> ThinkingBlock:
    return ThinkingBlock(
        thinking=f"reasoning-{tag}",
        content=[
            {
                "encrypted_content": f"ENC_{tag}",
                "id": f"rs_{tag}",
                "summary": f"reasoning-{tag}",
            }
        ],
    )


def _request_prior_thinking_and_call(
    response_id: str = "resp_precompaction",
) -> ChatRequest:
    """Same-turn, mid-tool-loop continuation: one user message started the
    turn; the assistant's ONE reply so far carries BOTH a ThinkingBlock
    (with ciphertext) AND a ToolCallBlock (response_id in metadata); the
    tool result is appended and we're asking the provider to continue --
    NO new user message yet (this is still the same README:502 "turn").

    Deliberately does NOT end in a fresh user turn: reasoning_replay_scope
    defaults to "turn" (Change D), which bounds replay to assistant turns
    SINCE THE LAST USER MESSAGE. A fixture ending in a new user turn would
    put the thinking+tool-call turn BEFORE that cutoff, making it invisible
    to the very fix these tests exist to prove -- the reset/retry tests here
    are about restoring reasoning for the CURRENT (not yet complete) turn,
    which is exactly the scenario Change A's headline bug hits (a
    mid-tool-loop request, post-compaction-reset or post-retry).
    """
    msgs = [
        Message(role="user", content="Hi"),
        Message(
            role="assistant",
            content=[
                _reasoning_thinking_block("t1"),
                ToolCallBlock(id="call_1", name="lookup", input={"key": "a"}),
            ],
            metadata={METADATA_RESPONSE_ID: response_id},
        ),
        Message(role="tool", content="tool result", tool_call_id="call_1"),
    ]
    return ChatRequest(messages=msgs)


def _multi_turn_request_with_current_turn_step(
    response_id: str = "resp_midturn",
) -> ChatRequest:
    """4 user-separated exchanges: 3 COMPLETED turns (each with its own
    reasoning + text reply) followed by a 4th, CURRENT turn that already has
    one in-flight tool-loop step (reasoning + tool call, response_id in
    metadata) awaiting its tool result. Used to prove turn-scoped bounding on
    a retry: only the current turn's reasoning (rs_t4) may survive, not the
    3 completed turns' (rs_t1/rs_t2/rs_t3)."""
    msgs = [
        Message(role="user", content="q1"),
        Message(
            role="assistant",
            content=[
                _reasoning_thinking_block("t1"),
                TextBlock(type="text", text="a1"),
            ],
        ),
        Message(role="user", content="q2"),
        Message(
            role="assistant",
            content=[
                _reasoning_thinking_block("t2"),
                TextBlock(type="text", text="a2"),
            ],
        ),
        Message(role="user", content="q3"),
        Message(
            role="assistant",
            content=[
                _reasoning_thinking_block("t3"),
                TextBlock(type="text", text="a3"),
            ],
        ),
        Message(role="user", content="q4"),
        Message(
            role="assistant",
            content=[
                _reasoning_thinking_block("t4"),
                ToolCallBlock(id="call_current", name="lookup", input={"key": "b"}),
            ],
            metadata={METADATA_RESPONSE_ID: response_id},
        ),
        Message(role="tool", content="tool result", tool_call_id="call_current"),
    ]
    return ChatRequest(messages=msgs)


def _reasoning_items(input_items: list[Any]) -> list[dict[str, Any]]:
    return [
        it
        for it in input_items
        if isinstance(it, dict) and it.get("type") == "reasoning"
    ]


def _function_call_items(input_items: list[Any]) -> list[dict[str, Any]]:
    return [
        it
        for it in input_items
        if isinstance(it, dict) and it.get("type") == "function_call"
    ]


# ---------------------------------------------------------------------------
# Post-compaction reset -- the headline regression (A1-A5)
# ---------------------------------------------------------------------------


def test_post_compaction_request_carries_reasoning_items():
    """The headline fix: post-compaction request's `input` contains >=1
    reasoning item WITH encrypted_content -- fails on main (chain_active
    still True post-reset suppresses reasoning re-insertion)."""
    provider = _make_provider(default_model="gpt-5.5")
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())
    provider._reset_chain_on_next_request = True

    asyncio.run(provider.complete(_request_prior_thinking_and_call()))

    params = _captured_params(provider)
    reasoning_items = _reasoning_items(params.get("input", []))
    assert len(reasoning_items) >= 1, (
        f"Post-compaction request must carry reasoning items; got input={params.get('input')}"
    )
    assert any(it.get("encrypted_content") for it in reasoning_items), (
        f"Post-compaction reasoning items must carry encrypted_content; got {reasoning_items}"
    )


def test_post_compaction_request_requests_include():
    provider = _make_provider(default_model="gpt-5.5")
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())
    provider._reset_chain_on_next_request = True

    asyncio.run(provider.complete(_request_prior_thinking_and_call()))

    params = _captured_params(provider)
    include = params.get("include", [])
    assert "reasoning.encrypted_content" in include, (
        f"Post-compaction request must request include=reasoning.encrypted_content; got {include}"
    )


def test_post_compaction_request_keeps_store_true():
    """store=True is preserved (deliberate spec deviation, see design §2.3):
    the reset response must still be retained server-side so the NEXT turn
    can chain from it -- store=False would 404 the following turn's chain
    attempt."""
    provider = _make_provider(default_model="gpt-5.5")
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())
    provider._reset_chain_on_next_request = True

    asyncio.run(provider.complete(_request_prior_thinking_and_call()))

    params = _captured_params(provider)
    assert params.get("store") is True, (
        f"Post-compaction request must keep store=True; got {params.get('store')}"
    )


def test_post_compaction_request_drops_previous_id():
    """Guards the existing reset behavior through the A1 reorder."""
    provider = _make_provider(default_model="gpt-5.5")
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())
    provider._reset_chain_on_next_request = True

    asyncio.run(provider.complete(_request_prior_thinking_and_call()))

    params = _captured_params(provider)
    assert "previous_response_id" not in params, (
        f"Post-compaction request must NOT carry previous_response_id; got {params.get('previous_response_id')}"
    )


def test_post_compaction_reasoning_precedes_function_calls():
    """The actual migration-error invariant: a reasoning item's index must
    precede its function_call item's index in `input`."""
    provider = _make_provider(default_model="gpt-5.5")
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())
    provider._reset_chain_on_next_request = True

    asyncio.run(provider.complete(_request_prior_thinking_and_call()))

    params = _captured_params(provider)
    input_items = params.get("input", [])
    reasoning_idx = next(
        (
            i
            for i, it in enumerate(input_items)
            if isinstance(it, dict) and it.get("type") == "reasoning"
        ),
        None,
    )
    fc_idx = next(
        (
            i
            for i, it in enumerate(input_items)
            if isinstance(it, dict) and it.get("type") == "function_call"
        ),
        None,
    )
    assert reasoning_idx is not None, f"No reasoning item found in input={input_items}"
    assert fc_idx is not None, f"No function_call item found in input={input_items}"
    assert reasoning_idx < fc_idx, (
        f"Reasoning item (index {reasoning_idx}) must precede its function_call "
        f"(index {fc_idx}); got input={input_items}"
    )


# ---------------------------------------------------------------------------
# Overflow retry (A6/A7)
# ---------------------------------------------------------------------------


def test_overflow_retry_restores_reasoning_items():
    provider = _make_provider(default_model="gpt-5.5")
    provider.client.responses.create = AsyncMock(
        side_effect=[_make_context_length_error(), DummyResponse()]
    )

    asyncio.run(provider.complete(_request_prior_thinking_and_call()))

    calls = _all_calls(provider)
    assert len(calls) == 2
    retry_input = calls[1].get("input", [])
    reasoning_items = _reasoning_items(retry_input)
    assert len(reasoning_items) >= 1, (
        f"Overflow retry must restore reasoning items; got input={retry_input}"
    )
    assert any(it.get("encrypted_content") for it in reasoning_items)


def test_overflow_retry_sets_include():
    provider = _make_provider(default_model="gpt-5.5")
    provider.client.responses.create = AsyncMock(
        side_effect=[_make_context_length_error(), DummyResponse()]
    )

    asyncio.run(provider.complete(_request_prior_thinking_and_call()))

    calls = _all_calls(provider)
    include = calls[1].get("include", [])
    assert "reasoning.encrypted_content" in include, (
        f"Overflow retry must request include; got include={include}"
    )


# ---------------------------------------------------------------------------
# 404 chain-invalidation retry (A6/A7)
# ---------------------------------------------------------------------------


def test_404_retry_restores_reasoning_items():
    provider = _make_provider(default_model="gpt-5.5")
    provider.client.responses.create = AsyncMock(
        side_effect=[_make_not_found_error("response_not_found"), DummyResponse()]
    )

    asyncio.run(provider.complete(_request_prior_thinking_and_call()))

    calls = _all_calls(provider)
    assert len(calls) == 2
    retry_input = calls[1].get("input", [])
    reasoning_items = _reasoning_items(retry_input)
    assert len(reasoning_items) >= 1, (
        f"404 chain-invalidation retry must restore reasoning items; got input={retry_input}"
    )


def test_404_retry_sets_include():
    provider = _make_provider(default_model="gpt-5.5")
    provider.client.responses.create = AsyncMock(
        side_effect=[_make_not_found_error("response_not_found"), DummyResponse()]
    )

    asyncio.run(provider.complete(_request_prior_thinking_and_call()))

    calls = _all_calls(provider)
    include = calls[1].get("include", [])
    assert "reasoning.encrypted_content" in include, (
        f"404 retry must request include; got include={include}"
    )


# ---------------------------------------------------------------------------
# Turn-bounded replay on the overflow retry (guards the re-inflation risk, §2.4)
# ---------------------------------------------------------------------------


def test_overflow_retry_reasoning_is_turn_bounded():
    """3 COMPLETED turns + 1 in-flight current-turn step, overflow -> the
    retry's restored reasoning must carry ONLY the current turn's item
    (rs_t4), not the 3 completed turns' (rs_t1/t2/t3). Without D's bound,
    the retry would re-inflate with every historical turn's ciphertext,
    risking re-triggering the very overflow it's trying to recover from."""
    provider = _make_provider(
        default_model="gpt-5.5"
    )  # default reasoning_replay_scope="turn"
    provider.client.responses.create = AsyncMock(
        side_effect=[_make_context_length_error(), DummyResponse()]
    )

    asyncio.run(provider.complete(_multi_turn_request_with_current_turn_step()))

    calls = _all_calls(provider)
    retry_input = calls[1].get("input", [])
    reasoning_ids = [it.get("id") for it in _reasoning_items(retry_input)]
    assert reasoning_ids == ["rs_t4"], (
        f"Overflow retry must be turn-bounded (only the current turn's reasoning); "
        f"got {reasoning_ids}"
    )
