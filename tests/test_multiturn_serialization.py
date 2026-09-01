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

import json
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

    def test_synthesized_ids_are_deterministic(self) -> None:
        """Byte-identical content MUST serialize to a byte-identical item.

        The serializer is deterministic by contract: same input, same output.
        A fresh ``uuid4()`` per call (the pre-fix behavior) violated that for
        no caller-visible reason. This inverts the old
        ``test_synthesized_ids_are_valid_strings``, which asserted the
        nondeterminism ("fresh id per call; fine for stateless replay") as
        correct behavior.
        """
        a = _build_assistant_message_item([{"type": "output_text", "text": "x"}])
        b = _build_assistant_message_item([{"type": "output_text", "text": "x"}])
        assert a["id"].startswith("msg_") and len(a["id"]) > 4
        assert a["id"] == b["id"]
        c = _build_assistant_message_item([{"type": "output_text", "text": "y"}])
        assert a["id"] != c["id"]

    def test_empty_content(self) -> None:
        item = _build_assistant_message_item([])
        assert item["type"] == "message"
        assert item["content"] == []

    def test_status_defaults_to_completed(self) -> None:
        item = _build_assistant_message_item([{"type": "output_text", "text": "x"}])
        assert item["status"] == "completed"

    def test_status_override_is_honored(self) -> None:
        item = _build_assistant_message_item(
            [{"type": "output_text", "text": "x"}], status="incomplete"
        )
        assert item["status"] == "incomplete"


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
        # The turn is replayed only because it was truncated, so it must be
        # reported as incomplete -- "completed" would contradict the request.
        assert msg["status"] == "incomplete"
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


class TestSerializationDeterminism:
    """``_convert_messages`` must be deterministic: same history in, same bytes out.

    Before the content-derived id change, every replayed assistant text
    message minted a fresh ``uuid4()`` on EVERY call to ``_convert_messages``
    (which runs fresh on every provider request), so two consecutive
    serializations of an UNCHANGED history produced different bytes -- for a
    reason unrelated to anything the caller changed.

    These tests pin determinism as a property of the serializer itself, not
    just of the ``_build_assistant_message_item`` helper in isolation, so a
    regression introduced anywhere else in ``_convert_messages`` is also
    caught.

    Scope note, so these tests are not mistaken for a performance claim:
    determinism here is a *correctness* property and a prerequisite for a
    prefix cache to ever cover replayed history items. It is not, by itself,
    evidence of any cache or cost benefit. On the deployment measured on
    2026-09-01, ``input`` items were not read back by the prompt cache at all
    (``cache_read`` equalled the ``instructions``+``tools`` head exactly, both
    before and after this change), so this change produced no measured cost
    effect there. That separate defect remains open.
    """

    def _provider(self) -> OpenAIProvider:
        return OpenAIProvider(api_key="[REDACTED:SECRET]")

    def test_request_serialization_is_byte_stable_across_consecutive_builds(
        self,
    ) -> None:
        """Two get-request cycles over UNCHANGED history must be byte-identical.

        The minimal reproduction of the nondeterminism this change removes: a
        3-message history with one completed assistant turn, serialized twice
        back-to-back in the same process. Pre-change this returned different
        bytes each time; the only difference was the synthesized ``msg_*`` id.
        """
        provider = self._provider()
        history: list[dict[str, Any]] = [
            {"role": "user", "content": "do the thing"},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Did it.\n- A\n- B"}],
            },
            {"role": "user", "content": "next"},
        ]
        a = provider._convert_messages(history)
        b = provider._convert_messages(history)
        assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)

    def test_shared_prefix_survives_history_growth(self) -> None:
        """Appending a turn must not perturb ANY earlier serialized item.

        Determinism has to hold across a GROWING history, not just repeated
        serializations of a fixed one: an earlier item whose serialized bytes
        change (even only its `id`) when a later turn is appended would mean
        the serializer's output depends on input it should be independent of.
        """
        provider = self._provider()
        base: list[dict[str, Any]] = [
            {"role": "user", "content": "do the thing"},
            {"role": "assistant", "content": [{"type": "text", "text": "Did it."}]},
            {"role": "user", "content": "next"},
        ]
        grown = base + [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Did that too."}],
            },
            {"role": "user", "content": "more"},
        ]
        a = provider._convert_messages(base)
        b = provider._convert_messages(grown)
        a_json = [json.dumps(x, sort_keys=True) for x in a]
        b_prefix_json = [json.dumps(y, sort_keys=True) for y in b[: len(a)]]
        assert a_json == b_prefix_json

    def test_duplicate_assistant_content_gets_distinct_but_stable_ids(self) -> None:
        """Two byte-identical assistant messages in one history get DISTINCT ids
        (so neither wire item collides with the other), but each is STABLE
        across repeated serialization passes of the same history.

        This is the reason the id derivation takes an occurrence index rather
        than being a pure function of (content, status) alone: two identical
        turns must not collapse to the same id, but replaying the same
        history twice must not reshuffle which occurrence gets which id.
        """
        provider = self._provider()
        history: list[dict[str, Any]] = [
            {"role": "user", "content": "say hi"},
            {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
            {"role": "user", "content": "say hi again"},
            {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
            {"role": "user", "content": "ok"},
        ]
        a = provider._convert_messages(history)
        b = provider._convert_messages(history)

        def _assistant_ids(result: list[dict[str, Any]]) -> list[str]:
            return [m["id"] for m in result if m.get("role") == "assistant"]

        ids_a = _assistant_ids(a)
        ids_b = _assistant_ids(b)
        assert len(ids_a) == 2
        assert ids_a[0] != ids_a[1]  # distinct within one pass
        assert ids_a == ids_b  # stable across passes (order-preserving)

    def test_post_compaction_shape_is_byte_stable_across_consecutive_builds(
        self,
    ) -> None:
        """A rebuilt (post-compaction) history list -- no persisted ids anywhere,
        just plain role/content dicts -- must still serialize byte-identically
        across two consecutive builds.

        Compaction hands the serializer a freshly rebuilt list of plain
        role/content dicts, with no ids carried over from before the
        boundary. Determinism must not depend on anything having been
        persisted, so this pins the fully-synthesized case explicitly.
        """
        provider = self._provider()
        compacted_history: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": "[compaction notice] earlier turns summarized",
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "Implemented the first toolkit module and supporting notes.",
                    }
                ],
            },
            {"role": "user", "content": "continue with the next module"},
        ]
        a = provider._convert_messages(compacted_history)
        b = provider._convert_messages(compacted_history)
        assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


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
            # NOTE: deliberately NO trailing new user message -- Change D's
            # default reasoning_replay_scope="turn" bounds replay to
            # assistant turns SINCE THE LAST USER MESSAGE. A trailing new
            # user turn would put this assistant turn's reasoning before
            # that cutoff, which is out of scope for what this test checks
            # (orphan-stripping), not a real assertion about D's boundary.
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
            # See note in test_mixed_turn_drops_orphan_keeps_usable: no
            # trailing new user turn, so Change D's turn-scope cutoff does
            # not exclude this assistant turn's reasoning.
        ]
        reasoning = self._reasoning(provider._convert_messages(messages))
        assert [r.get("id") for r in reasoning] == ["rs_a", "rs_b"]
        assert all(r.get("encrypted_content") for r in reasoning)
