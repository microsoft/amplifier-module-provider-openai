"""Tests for ephemeral tail messages and turn-scoped reasoning replay.

Bug: `reasoning_replay_scope="turn"` computed its cutoff as the max index of
any message with role=="user". Live post-compaction message lists always end
with 1-2 EPHEMERAL user-role messages -- context-simple's compaction notice
(metadata={"source": "context-compaction", "ephemeral": True}) and
loop-streaming's per-iteration hook-injection fallback
(metadata={"ephemeral": True}, e.g. hooks-status-context) -- so the cutoff
landed on the last message's own index and ZERO reasoning items were replayed
on every post-compaction request, on BOTH the chained-reset path and the
stateless path. The existing unit fixtures deliberately avoided this shape
(see tests/test_reset_path_reasoning_continuity.py:129-135 -- fixtures end
[user, assistant, tool]; live message lists end with ephemeral user tails).

Fix: ephemeral messages (metadata.ephemeral=True) are not turn boundaries --
they are regenerated tail content injected after the real conversation, not
a new user turn. Excluded from the "turn" scope's cutoff computation.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

from amplifier_core.message_models import (
    ChatRequest,
    Message,
    ThinkingBlock,
    ToolCallBlock,
)

from amplifier_module_provider_openai import OpenAIProvider
from amplifier_module_provider_openai._constants import METADATA_RESPONSE_ID

# ---------------------------------------------------------------------------
# Helpers (dict-based messages, matching tests/test_reasoning_replay_scope.py)
# ---------------------------------------------------------------------------


def _provider(**config_overrides) -> OpenAIProvider:
    config = {"max_retries": 0, "use_streaming": False, **config_overrides}
    return OpenAIProvider(api_key="test-key", config=config)


class _Block:
    """Minimal object stand-in for a ContentBlock (attribute access)."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _thinking_block(tag: str) -> _Block:
    return _Block(
        type="thinking",
        thinking=f"reasoning for {tag}",
        content=[
            {
                "encrypted_content": f"ENC_{tag}",
                "id": f"rs_{tag}",
                "summary": f"reasoning for {tag}",
            }
        ],
    )


def _thinking_turn(tag: str, text: str) -> dict[str, Any]:
    """A completed turn: thinking + text reply, no further tool step."""
    return {
        "role": "assistant",
        "content": [_thinking_block(tag), _Block(type="text", text=text)],
    }


def _thinking_turn_with_tool_call(
    tag: str, call_id: str, name: str, tool_input: dict[str, Any]
) -> dict[str, Any]:
    """A mid-tool-loop step: thinking + tool_call, no text reply yet --
    matches the live shape (assistant-with-thinking(t1)+tool_call)."""
    return {
        "role": "assistant",
        "content": [
            _thinking_block(tag),
            _Block(type="tool_call", id=call_id, name=name, input=tool_input),
        ],
    }


def _user_turn(text: str) -> dict[str, Any]:
    return {"role": "user", "content": text}


def _ephemeral_user(text: str, **extra_metadata) -> dict[str, Any]:
    """A regenerated tail injection: role=user, metadata.ephemeral=True.
    Mirrors context-simple's compaction notice and loop-streaming's
    per-iteration hook-status injection fallback."""
    return {
        "role": "user",
        "content": text,
        "metadata": {"ephemeral": True, **extra_metadata},
    }


def _tool_result(call_id: str, content: str = "tool result") -> dict[str, Any]:
    return {
        "role": "tool",
        "content": content,
        "tool_call_id": call_id,
        "tool_name": "lookup",
    }


def _reasoning_ids(converted: list[dict[str, Any]]) -> list[str]:
    return [
        m.get("id")
        for m in converted
        if isinstance(m, dict) and m.get("type") == "reasoning"
    ]


# ---------------------------------------------------------------------------
# 1. The headline regression: live post-compaction shape survives the fix
# ---------------------------------------------------------------------------


def test_turn_scope_survives_tail_ephemeral_injections():
    """LIVE post-compaction shape: one completed tool-loop step (thinking +
    tool_call), its tool result, then the two ephemeral tail user messages
    observed in live captures (compaction notice + hook-status injection).
    Converted input must contain exactly the reasoning item for that step
    (rs_t1) -- FAILS on current main (cutoff lands on the last ephemeral
    user's own index, so the emission gate `i > _reasoning_cutoff` is never
    satisfied and reasoning_ids comes back [])."""
    provider = _provider()  # default reasoning_replay_scope == "turn"
    assert provider.reasoning_replay_scope == "turn"

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        _user_turn("q1"),
        _thinking_turn_with_tool_call("t1", "call_1", "lookup", {"key": "a"}),
        _tool_result("call_1"),
        _ephemeral_user(
            "[Context compacted: summarized 40 earlier messages]",
            source="context-compaction",
        ),
        _ephemeral_user(
            "<status>cwd: /repo, git: clean</status>",
            source="hooks-status-context",
        ),
    ]
    converted = provider._convert_messages(messages, skip_reasoning_reinsertion=False)
    ids = _reasoning_ids(converted)
    assert ids == ["rs_t1"], (
        f"Ephemeral tail injections must not count as turn boundaries; got {ids}"
    )


# ---------------------------------------------------------------------------
# 2. The fix must not widen the window -- a genuine user turn still bounds it
# ---------------------------------------------------------------------------


def test_genuine_user_turn_still_bounds_replay_under_ephemeral_tail():
    """2 completed turns, the 2nd trailed by an ephemeral status injection.
    Exactly the 2nd turn's reasoning (rs_t2) must survive -- a real user
    message still cuts the window; the fix does not widen replay to 'all
    history' just because ephemeral messages are excluded from the count."""
    provider = _provider()

    messages = [
        _user_turn("q1"),
        _thinking_turn("t1", "a1"),
        _user_turn("q2"),
        _thinking_turn("t2", "a2"),
        _ephemeral_user(
            "<status>cwd: /repo, git: clean</status>",
            source="hooks-status-context",
        ),
    ]
    converted = provider._convert_messages(messages, skip_reasoning_reinsertion=False)
    ids = _reasoning_ids(converted)
    assert ids == ["rs_t2"], (
        f"A genuine user message must still bound turn-scoped replay "
        f"(fix must not widen the window); got {ids}"
    )


# ---------------------------------------------------------------------------
# 3. End-to-end reset-path proof, in the LIVE shape (mirrors
#    tests/test_reset_path_reasoning_continuity.py's harness)
# ---------------------------------------------------------------------------


def _make_reset_provider(**config_overrides) -> OpenAIProvider:
    config = {"max_retries": 0, "use_streaming": False, **config_overrides}
    return OpenAIProvider(api_key="test-key", config=config)


class _DummyResponse:
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


def _pydantic_reasoning_block(tag: str) -> ThinkingBlock:
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


def _live_post_compaction_request(
    response_id: str = "resp_precompaction",
) -> ChatRequest:
    """Same-turn, mid-tool-loop continuation (matches
    _request_prior_thinking_and_call() in test_reset_path_reasoning_continuity.py)
    with the two ephemeral tail messages the live wave observed appended --
    the shape the existing fixtures deliberately avoided (see that file,
    lines 129-135: fixtures end [user, assistant, tool])."""
    msgs = [
        Message(role="user", content="Hi"),
        Message(
            role="assistant",
            content=[
                _pydantic_reasoning_block("t1"),
                ToolCallBlock(id="call_1", name="lookup", input={"key": "a"}),
            ],
            metadata={METADATA_RESPONSE_ID: response_id},
        ),
        Message(role="tool", content="tool result", tool_call_id="call_1"),
        Message(
            role="user",
            content="[Context compacted: summarized 40 earlier messages]",
            metadata={"source": "context-compaction", "ephemeral": True},
        ),
        Message(
            role="user",
            content="<status>cwd: /repo, git: clean</status>",
            metadata={"ephemeral": True, "source": "hooks-status-context"},
        ),
    ]
    return ChatRequest(messages=msgs)


def test_reset_path_carries_reasoning_under_live_ephemeral_tail():
    """The post-compaction RESET request (provider._reset_chain_on_next_request
    = True, the same path proven by test_reset_path_reasoning_continuity.py)
    must carry the surviving reasoning item even with the live ephemeral tail
    appended. FAILS on current main -- the tail collapses the turn window."""
    provider = _make_reset_provider(default_model="gpt-5.5")
    provider.client.responses.create = AsyncMock(return_value=_DummyResponse())
    provider._reset_chain_on_next_request = True

    asyncio.run(provider.complete(_live_post_compaction_request()))

    mock = cast(AsyncMock, provider.client.responses.create)
    params = mock.call_args.kwargs
    input_items = params.get("input", [])
    reasoning_items = [
        it
        for it in input_items
        if isinstance(it, dict) and it.get("type") == "reasoning"
    ]
    assert len(reasoning_items) >= 1, (
        f"Post-compaction reset request (live ephemeral tail) must carry "
        f"reasoning items; got input={input_items}"
    )
    assert any(it.get("encrypted_content") for it in reasoning_items), (
        f"Surviving reasoning items must carry encrypted_content; got {reasoning_items}"
    )


# ---------------------------------------------------------------------------
# 4. Corner case: ALL user messages ephemeral -> cutoff=-1 (safe direction)
# ---------------------------------------------------------------------------


def test_all_ephemeral_users_yields_unbounded_cutoff():
    """If EVERY user-role message in the list happens to be ephemeral, the
    generator excluding them is empty and max(..., default=-1) resolves to
    -1 -- equivalent to scope="all" (every index qualifies). This is the
    safe direction: it means MORE reasoning gets replayed, never less, so
    there is no risk of the original zero-replay bug recurring for this
    input shape. Documented here as intended behavior, not an accident."""
    provider = _provider()

    messages = [
        _ephemeral_user("[compacted]", source="context-compaction"),
        _thinking_turn("t1", "a1"),
    ]
    converted = provider._convert_messages(messages, skip_reasoning_reinsertion=False)
    ids = _reasoning_ids(converted)
    assert ids == ["rs_t1"], (
        f"All-ephemeral-users edge case must still replay reasoning (cutoff=-1 "
        f"is the safe/more-permissive direction, not a regression); got {ids}"
    )


# ---------------------------------------------------------------------------
# 5. No-op for non-ephemeral / absent metadata -- behavior unchanged
# ---------------------------------------------------------------------------


def test_non_ephemeral_metadata_behaves_as_before():
    """A user message with metadata present but NOT ephemeral (or no
    metadata key at all) must behave exactly like today: it still counts as
    a turn boundary. Guards against the filter accidentally treating any
    metadata-bearing user message as ephemeral."""
    provider = _provider()

    messages = [
        {"role": "user", "content": "q1", "metadata": {"source": "user-typed"}},
        _thinking_turn("t1", "a1"),
        _user_turn("q2"),  # no metadata key at all
        _thinking_turn("t2", "a2"),
    ]
    converted = provider._convert_messages(messages, skip_reasoning_reinsertion=False)
    ids = _reasoning_ids(converted)
    assert ids == ["rs_t2"], (
        f"Non-ephemeral / absent metadata must not change turn-boundary "
        f"behavior; got {ids}"
    )
