"""Tests for Change D: bounded reasoning-blob replay (`reasoning_replay_scope`).

Bug: on the stateless replay path, every historical assistant turn's
ThinkingBlock ciphertext was re-inserted into `input` on every subsequent
request. Encrypted reasoning blobs measured ~1,200 chars each; unbounded
replay is linear in conversation length (>50% of payload by turn 4 in live
probing), while README.md's own citation of OpenAI guidance scopes reasoning
preservation to *within a turn* (the multiple API calls of one tool loop),
not across turns.

Fix: `reasoning_replay_scope` config (default "turn") bounds inline replay to
assistant turns since the last user message. "all" restores the old
unbounded behavior (escape hatch); "none" disables inline replay entirely.
One gate, computed once in `_convert_messages`, applied at the single
emission site -- collection is unaffected, so the orphan-stripping guard's
semantics are untouched.
"""

from __future__ import annotations

from typing import Any

from amplifier_module_provider_openai import OpenAIProvider

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _provider(**config_overrides) -> OpenAIProvider:
    config = {"max_retries": 0, "use_streaming": False, **config_overrides}
    return OpenAIProvider(api_key="test-key", config=config)


class _Block:
    """Minimal object stand-in for a ContentBlock (attribute access)."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _thinking_turn(tag: str, text: str) -> dict[str, Any]:
    """One assistant message: a ThinkingBlock (named-dict content, post-B1)
    with ciphertext + a text reply, tagged with a unique reasoning id."""
    return {
        "role": "assistant",
        "content": [
            _Block(
                type="thinking",
                thinking=f"reasoning for {tag}",
                content=[
                    {
                        "encrypted_content": f"ENC_{tag}",
                        "id": f"rs_{tag}",
                        "summary": f"reasoning for {tag}",
                    }
                ],
            ),
            _Block(type="text", text=text),
        ],
    }


def _user_turn(text: str) -> dict[str, Any]:
    return {"role": "user", "content": text}


def _reasoning_ids(converted: list[dict[str, Any]]) -> list[str]:
    return [
        m.get("id")
        for m in converted
        if isinstance(m, dict) and m.get("type") == "reasoning"
    ]


# ---------------------------------------------------------------------------
# Default ("turn") scope
# ---------------------------------------------------------------------------


def test_turn_scope_replays_only_since_last_user():
    """3 user-separated turns, each with reasoning; default ("turn") scope
    emits ONLY the last turn's reasoning items."""
    provider = _provider()  # default reasoning_replay_scope == "turn"
    assert provider.reasoning_replay_scope == "turn"

    messages = [
        _user_turn("q1"),
        _thinking_turn("t1", "a1"),
        _user_turn("q2"),
        _thinking_turn("t2", "a2"),
        _user_turn("q3"),
        _thinking_turn("t3", "a3"),
    ]
    converted = provider._convert_messages(messages, skip_reasoning_reinsertion=False)
    ids = _reasoning_ids(converted)
    assert ids == ["rs_t3"], (
        f"Turn-scoped replay must emit ONLY the last turn's reasoning item; got {ids}"
    )


def test_turn_scope_replays_all_tool_steps_in_turn():
    """One user message + 3 assistant tool-loop steps (no intervening user
    message) -> ALL 3 reasoning items emitted (README's "single turn spans
    multiple API calls" guarantee)."""
    provider = _provider()

    messages = [
        _user_turn("q1"),
        _thinking_turn("step1", "a1"),
        _thinking_turn("step2", "a2"),
        _thinking_turn("step3", "a3"),
    ]
    converted = provider._convert_messages(messages, skip_reasoning_reinsertion=False)
    ids = _reasoning_ids(converted)
    assert ids == ["rs_step1", "rs_step2", "rs_step3"], (
        f"All tool-loop steps within the current turn must be replayed; got {ids}"
    )


# ---------------------------------------------------------------------------
# "all" / "none" scopes
# ---------------------------------------------------------------------------


def test_all_scope_replays_everything():
    provider = _provider(reasoning_replay_scope="all")
    assert provider.reasoning_replay_scope == "all"

    messages = [
        _user_turn("q1"),
        _thinking_turn("t1", "a1"),
        _user_turn("q2"),
        _thinking_turn("t2", "a2"),
        _user_turn("q3"),
        _thinking_turn("t3", "a3"),
    ]
    converted = provider._convert_messages(messages, skip_reasoning_reinsertion=False)
    ids = _reasoning_ids(converted)
    assert ids == ["rs_t1", "rs_t2", "rs_t3"], (
        f"'all' scope must replay every turn's reasoning items; got {ids}"
    )


def test_none_scope_replays_nothing():
    provider = _provider(reasoning_replay_scope="none")
    assert provider.reasoning_replay_scope == "none"

    messages = [
        _user_turn("q1"),
        _thinking_turn("t1", "a1"),
        _user_turn("q2"),
        _thinking_turn("t2", "a2"),
    ]
    converted = provider._convert_messages(messages, skip_reasoning_reinsertion=False)
    ids = _reasoning_ids(converted)
    assert ids == [], f"'none' scope must replay zero reasoning items; got {ids}"


def test_invalid_scope_falls_back_to_turn_with_warning(caplog):
    import logging

    with caplog.at_level(logging.WARNING, logger="amplifier_module_provider_openai"):
        provider = _provider(reasoning_replay_scope="bogus")

    assert provider.reasoning_replay_scope == "turn", (
        f"Invalid scope must fall back to 'turn'; got {provider.reasoning_replay_scope!r}"
    )
    assert any(
        "reasoning_replay_scope" in record.message and "bogus" in record.message
        for record in caplog.records
    ), (
        f"Expected a warning naming the invalid scope; got messages="
        f"{[r.message for r in caplog.records]}"
    )


# ---------------------------------------------------------------------------
# Interaction with chaining
# ---------------------------------------------------------------------------


def test_scope_does_not_affect_chained_path():
    """When skip_reasoning_reinsertion=True (chain_will_attach), ZERO
    reasoning items are emitted regardless of reasoning_replay_scope -- the
    scope only bounds STATELESS replay; chaining suppresses replay entirely
    (server already holds the state)."""
    messages = [
        _user_turn("q1"),
        _thinking_turn("t1", "a1"),
        _user_turn("q2"),
        _thinking_turn("t2", "a2"),
    ]
    for scope in ("turn", "all", "none"):
        provider = _provider(reasoning_replay_scope=scope)
        converted = provider._convert_messages(
            messages, skip_reasoning_reinsertion=True
        )
        ids = _reasoning_ids(converted)
        assert ids == [], (
            f"Chained path (skip_reasoning_reinsertion=True) must emit zero "
            f"reasoning items under scope={scope!r}; got {ids}"
        )


# ---------------------------------------------------------------------------
# Config-key recognition
# ---------------------------------------------------------------------------


def test_scope_key_recognized_at_mount(caplog):
    """reasoning_replay_scope must be a recognized config key -- no
    'Unrecognized config key' warning when it's set."""
    import logging

    with caplog.at_level(logging.WARNING, logger="amplifier_module_provider_openai"):
        _provider(reasoning_replay_scope="turn")

    assert not any(
        "Unrecognized config key" in record.message for record in caplog.records
    ), (
        f"reasoning_replay_scope must be in _CONSUMED_CONFIG_KEYS; got warnings="
        f"{[r.message for r in caplog.records]}"
    )
