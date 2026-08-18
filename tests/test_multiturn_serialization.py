"""Wire-format regression tests for multi-turn assistant message serialization.

Guards the fix for the "Cannot determine type of 'item'" 400 that llama-server
(and vLLM's strict Pydantic validation) raise when assistant history is replayed
into the Responses API ``input`` array.

The provider must emit the canonical ``ResponseOutputMessage`` shape ("Form 2+"):
``{"type": "message", "id": ..., "role": "assistant", "status": "completed",
   "content": [{"type": "output_text", "text": ..., "annotations": []}]}``.

This shape was verified on the wire against the OpenAI Responses API, llama.cpp's
llama-server, and vLLM 0.19 as the single form every backend accepts:
- ``type`` is required by llama-server (item dispatch keys on it),
- ``id`` and ``status`` are required by vLLM (openai SDK ``ResponseOutputMessageParam``),
- ``annotations`` mirrors OpenAI's own output items and is accepted everywhere.

User/input messages must NOT be given ``type: "message"`` -- they resolve to the
permissive ``EasyInputMessage`` branch and adding output-message fields breaks them.
"""

from typing import Any

from amplifier_module_provider_openai import (
    OpenAIProvider,
    _build_assistant_message_item,
)


class TestAssistantMessageHelper:
    """The shared serializer must produce the canonical Form 2+ shape."""

    def test_single_text_part(self) -> None:
        item = _build_assistant_message_item(
            [{"type": "output_text", "text": "Hello!"}]
        )
        assert item["type"] == "message"
        assert item["role"] == "assistant"
        assert item["status"] == "completed"
        assert item["id"].startswith("msg_")
        assert item["content"] == [
            {"type": "output_text", "text": "Hello!", "annotations": []}
        ]

    def test_multi_part_preserves_order_and_annotations(self) -> None:
        item = _build_assistant_message_item(
            [
                {"type": "output_text", "text": "one"},
                {"type": "output_text", "text": "two", "annotations": [{"k": 1}]},
            ]
        )
        assert len(item["content"]) == 2
        assert item["content"][0] == {
            "type": "output_text",
            "text": "one",
            "annotations": [],
        }
        assert item["content"][1]["annotations"] == [{"k": 1}]

    def test_preserved_id_used_when_given(self) -> None:
        item = _build_assistant_message_item(
            [{"type": "output_text", "text": "x"}], message_id="msg_keep_me"
        )
        assert item["id"] == "msg_keep_me"

    def test_synthesized_ids_are_valid_strings(self) -> None:
        a = _build_assistant_message_item([{"type": "output_text", "text": "x"}])
        b = _build_assistant_message_item([{"type": "output_text", "text": "x"}])
        assert a["id"].startswith("msg_") and len(a["id"]) > 4
        assert a["id"] != b["id"]  # fresh id per call; fine for stateless replay

    def test_empty_content(self) -> None:
        item = _build_assistant_message_item([])
        assert item["type"] == "message"
        assert item["content"] == []


class TestBuildContinuationInput:
    """_build_continuation_input must emit assistant items in Form 2+."""

    def _provider(self) -> OpenAIProvider:
        return OpenAIProvider(api_key="test-key")

    def test_assistant_item_is_typed_message(self) -> None:
        provider = self._provider()
        original_input = [
            {"role": "user", "content": [{"type": "input_text", "text": "hi"}]}
        ]
        accumulated_output = [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "Once upon a time..."}],
            }
        ]
        result = provider._build_continuation_input(original_input, accumulated_output)
        assistant = [m for m in result if m.get("role") == "assistant"]
        assert len(assistant) == 1
        msg = assistant[0]
        assert msg["type"] == "message"
        assert msg["status"] == "completed"
        assert msg["id"].startswith("msg_")
        assert msg["content"][0] == {
            "type": "output_text",
            "text": "Once upon a time...",
            "annotations": [],
        }

    def test_user_input_left_untyped(self) -> None:
        provider = self._provider()
        original_input = [
            {"role": "user", "content": [{"type": "input_text", "text": "hi"}]}
        ]
        result = provider._build_continuation_input(original_input, [])
        # No assistant content -> unchanged; user item must not gain a type.
        assert result[0].get("type") is None


class TestConvertMessages:
    """_convert_messages must emit assistant items in Form 2+, users untyped."""

    def _provider(self) -> OpenAIProvider:
        return OpenAIProvider(api_key="test-key")

    def test_assistant_text_becomes_typed_message(self) -> None:
        provider = self._provider()
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "Say hello"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "Say goodbye"},
        ]
        result = provider._convert_messages(messages)
        assistant = [m for m in result if m.get("role") == "assistant"]
        assert len(assistant) == 1
        msg = assistant[0]
        assert msg["type"] == "message"
        assert msg["status"] == "completed"
        assert msg["id"].startswith("msg_")
        assert msg["content"][0] == {
            "type": "output_text",
            "text": "Hello!",
            "annotations": [],
        }

    def test_user_messages_stay_untyped(self) -> None:
        provider = self._provider()
        messages: list[dict[str, Any]] = [{"role": "user", "content": "Hello"}]
        result = provider._convert_messages(messages)
        assert all(
            m.get("type") != "message" for m in result if m.get("role") == "user"
        )


class _Block:
    """Minimal ContentBlock stand-in for the thinking-block serialization path."""

    def __init__(self, **kw: Any) -> None:
        self.__dict__.update(kw)


class TestReasoningOrphanStrip:
    """Orphaned reasoning items (no encrypted_content) must be stripped PER-ITEM.

    A reasoning item replayed without ``encrypted_content`` is an unpairable
    reference the Responses API rejects (bare ``rs_*`` id -> 404). A single
    assistant turn can mix usable and orphaned reasoning items (one thinking
    block carried ``encrypted_content``, another did not). The strip must drop
    only the orphans while keeping the usable items -- an all-or-nothing check
    keyed on ``any()`` kept the orphans whenever a single sibling was usable,
    still failing the request.

    The strip only runs when the assistant message carries
    ``METADATA_REASONING_ITEMS`` metadata (the shape a real reasoning response
    produces); tests set it explicitly to exercise that guarded path.
    """

    def _provider(self) -> OpenAIProvider:
        return OpenAIProvider(api_key="test-key")

    def _reasoning(self, result: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            m for m in result if isinstance(m, dict) and m.get("type") == "reasoning"
        ]

    def test_mixed_turn_drops_orphan_keeps_usable(self) -> None:
        from amplifier_module_provider_openai._constants import (
            METADATA_REASONING_ITEMS,
        )

        provider = self._provider()
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "metadata": {METADATA_REASONING_ITEMS: ["rs_hasenc", "rs_noenc"]},
                "content": [
                    _Block(
                        type="thinking",
                        thinking="planning A",
                        content=["ENCRYPTED_AAA", "rs_hasenc"],
                    ),
                    _Block(
                        type="thinking",
                        thinking="planning B",
                        content=["", "rs_noenc"],  # NO encrypted_content -> orphan
                    ),
                    _Block(type="text", text="Here is my answer."),
                ],
            },
            {"role": "user", "content": "continue"},
        ]
        reasoning = self._reasoning(provider._convert_messages(messages))
        ids = [r.get("id") for r in reasoning]
        # usable item kept, orphan dropped
        assert ids == ["rs_hasenc"]
        assert all(r.get("encrypted_content") for r in reasoning)

    def test_all_orphan_turn_strips_everything(self) -> None:
        from amplifier_module_provider_openai._constants import (
            METADATA_REASONING_ITEMS,
        )

        provider = self._provider()
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "metadata": {METADATA_REASONING_ITEMS: ["rs_orphan"]},
                "content": [
                    _Block(
                        type="thinking",
                        thinking="p",
                        content=["", "rs_orphan"],  # NO encrypted_content
                    ),
                    _Block(type="text", text="ans"),
                ],
            },
            {"role": "user", "content": "go"},
        ]
        reasoning = self._reasoning(provider._convert_messages(messages))
        assert reasoning == []  # all-or-nothing behavior preserved when all orphaned

    def test_all_usable_turn_keeps_all(self) -> None:
        from amplifier_module_provider_openai._constants import (
            METADATA_REASONING_ITEMS,
        )

        provider = self._provider()
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "metadata": {METADATA_REASONING_ITEMS: ["rs_a", "rs_b"]},
                "content": [
                    _Block(type="thinking", thinking="A", content=["ENC_A", "rs_a"]),
                    _Block(type="thinking", thinking="B", content=["ENC_B", "rs_b"]),
                    _Block(type="text", text="done"),
                ],
            },
            {"role": "user", "content": "go"},
        ]
        reasoning = self._reasoning(provider._convert_messages(messages))
        assert [r.get("id") for r in reasoning] == ["rs_a", "rs_b"]
        assert all(r.get("encrypted_content") for r in reasoning)
