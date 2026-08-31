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
    TextBlock,
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
    converted = provider._convert_messages(messages)
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
    converted = provider._convert_messages(messages)
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
    """The post-compaction stateless request must carry the surviving
    reasoning item even with the live ephemeral tail appended. FAILS
    without the fix -- the tail collapses the turn window."""
    provider = _make_reset_provider(default_model="gpt-5.5")
    provider.client.responses.create = AsyncMock(return_value=_DummyResponse())

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
    converted = provider._convert_messages(messages)
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
    converted = provider._convert_messages(messages)
    ids = _reasoning_ids(converted)
    assert ids == ["rs_t2"], (
        f"Non-ephemeral / absent metadata must not change turn-boundary "
        f"behavior; got {ids}"
    )


# ---------------------------------------------------------------------------
# 6. Pre-user ephemeral block (system-reminder redesign W1/W4.2 pinning)
#
# W1 (separate repo: amplifier-module-loop-streaming) moves the turn-start
# reminder block to BEFORE the real user message:
#   [..., block(ephemeral, reminder_placement="pre_user"), user(real), assistant, ...]
# instead of after it. No code change is required in THIS repo for that shape
# -- the `not ephemeral` filter already keeps the cutoff on the real user
# message regardless of whether the ephemeral block sits before or after it,
# because `max()` picks the highest qualifying index either way. These tests
# pin that fact so a future change is never free to assume otherwise, and so
# that dropping `metadata.ephemeral` from the block anywhere in the pipeline
# is caught here rather than silently degrading reasoning replay in prod.
# ---------------------------------------------------------------------------


def test_turn_scope_cutoff_ignores_a_pre_user_ephemeral_block():
    """A leading ephemeral reminder block (written BEFORE the real user
    message, per the system-reminder redesign) must not be mistaken for the
    turn boundary itself, and must not widen or narrow the turn-scoped
    replay window. The cutoff must land on the REAL user message's index --
    exactly the same place it would land with no block present at all.
    Passes at 8eb761f (documented as a pin, not a fail-before proof); exists
    so a future change that drops `metadata.ephemeral` from the pre-user
    block fails HERE, loudly."""
    provider = _provider()

    messages = [
        _user_turn("q1"),
        _thinking_turn("t1", "a1"),
        _ephemeral_user(
            "<system-reminders>...</system-reminders>",
            reminder_placement="pre_user",
        ),
        _user_turn("q2"),
        _thinking_turn("t2", "a2"),
    ]
    converted = provider._convert_messages(messages)
    ids = _reasoning_ids(converted)
    assert ids == ["rs_t2"], (
        f"Turn-scoped replay must be cut at the REAL user message (q2), not "
        f"widened by the preceding ephemeral reminder block, and not "
        f"collapsed to zero either; got {ids}"
    )

    # Comparison case: the identical shape MINUS the leading block must
    # produce the identical result -- proof the block is truly inert here.
    messages_no_block = [
        _user_turn("q1"),
        _thinking_turn("t1", "a1"),
        _user_turn("q2"),
        _thinking_turn("t2", "a2"),
    ]
    converted_no_block = provider._convert_messages(messages_no_block)
    assert _reasoning_ids(converted_no_block) == ids, (
        "The pre-user ephemeral block must be a no-op for cutoff placement"
    )


def test_pre_user_ephemeral_block_survives_conversion():
    """End-to-end: the block's `metadata.ephemeral` (and `persisted` /
    `reminder_placement`) must survive from the pydantic `Message` object,
    through `.model_dump()`, to the dicts `_convert_messages` actually
    inspects -- otherwise the cutoff filter above would silently stop
    seeing it. Uses the same reset-path harness as
    `test_reset_path_carries_reasoning_under_live_ephemeral_tail` above, but
    with the ephemeral block placed BEFORE turn 2's user message (the
    system-reminder redesign's `pre_user` placement) rather than after it.
    Turn 2's own mid-loop reasoning (rs_t2, emitted after the cutoff) must
    survive; turn 1's (rs_t1, before the cutoff) correctly does not -- that
    is the pre-existing turn-scope behavior, unaffected by this shape."""
    provider = _make_reset_provider(default_model="gpt-5.5")
    provider.client.responses.create = AsyncMock(return_value=_DummyResponse())

    msgs = [
        Message(role="user", content="Hi (turn 1)"),
        Message(
            role="assistant",
            content=[
                _pydantic_reasoning_block("t1"),
                TextBlock(type="text", text="a1"),
            ],
        ),
        Message(
            role="user",
            content="<system-reminders>...</system-reminders>",
            metadata={
                "ephemeral": True,
                "persisted": True,
                "reminder_placement": "pre_user",
            },
        ),
        Message(role="user", content="Hi (turn 2)"),
        Message(
            role="assistant",
            content=[
                _pydantic_reasoning_block("t2"),
                ToolCallBlock(id="call_1", name="lookup", input={"key": "a"}),
            ],
            metadata={METADATA_RESPONSE_ID: "resp_precompaction"},
        ),
        Message(role="tool", content="tool result", tool_call_id="call_1"),
    ]
    request = ChatRequest(messages=msgs)

    asyncio.run(provider.complete(request))

    mock = cast(AsyncMock, provider.client.responses.create)
    params = mock.call_args.kwargs
    input_items = params.get("input", [])
    reasoning_items = [
        it
        for it in input_items
        if isinstance(it, dict) and it.get("type") == "reasoning"
    ]
    assert any(it.get("id") == "rs_t2" for it in reasoning_items), (
        f"Turn 2's reasoning item must survive being requested with a "
        f"pre-user ephemeral block ahead of its own user message; the "
        f"block's metadata.ephemeral must have reached _convert_messages "
        f"intact for the cutoff to land correctly; got input={input_items}"
    )
    assert not any(it.get("id") == "rs_t1" for it in reasoning_items), (
        f"Turn 1's reasoning is out of scope (before the cutoff) and must "
        f"still be excluded -- unaffected by the pre-user block; "
        f"got {reasoning_items}"
    )
