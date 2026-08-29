"""Tests for Change B: ThinkingBlock named-dict encoding + back-compat reader.

Bug: two identical construction sites encoded ThinkingBlock.content as a
POSITIONAL list, ``[encrypted_content, reasoning_id]``. amplifier_foundation's
``sanitize_for_json`` silently drops ``None`` from lists, so every block
captured while chaining (encrypted_content=None, since ``include`` was never
requested on the chained path) collapsed ``[None, "rs_abc"]`` ->
``["rs_abc"]`` on persistence. Both read sites required ``len(content) >= 2``,
so the collapsed block was silently discarded on the very next read -- the
reasoning became permanently unreplayable after resume, with no warning.

Fix: encode as a named dict, ``[{"encrypted_content": ..., "id": ...,
"summary": ...}]``. A dict survives ``sanitize_for_json``'s key-dropping
without losing the *identity* of what remains (``{"id": "rs_abc"}`` is still
unambiguously an id after ``encrypted_content: None`` is dropped) -- a
positional list cannot make the same claim. ``_decode_reasoning_state`` reads
all three on-disk shapes (new named dict, old 2-element positional, old
1-element collapsed) so existing transcripts remain readable across the
upgrade.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from amplifier_module_provider_openai import (
    OpenAIChatResponse,
    OpenAIProvider,
    _decode_reasoning_state,
)
from amplifier_module_provider_openai._response_handling import (
    convert_response_with_accumulated_output,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _provider(**config_overrides) -> OpenAIProvider:
    config = {"max_retries": 0, "use_streaming": False, **config_overrides}
    return OpenAIProvider(api_key="test-key", config=config)


def _make_response_with_reasoning(
    *, encrypted_content=None, reasoning_id="rs_test_001", summary=None
):
    reasoning_block = SimpleNamespace(
        type="reasoning",
        id=reasoning_id,
        encrypted_content=encrypted_content,
        summary=summary,
    )
    message_block = SimpleNamespace(
        type="message",
        content=[SimpleNamespace(type="output_text", text="Hello")],
    )
    return SimpleNamespace(
        output=[reasoning_block, message_block],
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
        status="completed",
        id="resp_test",
    )


class _Block:
    """Minimal object stand-in for a ContentBlock (attribute access, not dict)."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


# ---------------------------------------------------------------------------
# B1 -- write encoding is a named dict (non-streaming + streaming builders)
# ---------------------------------------------------------------------------


def test_new_encoding_is_named_dict():
    """Non-streaming response with ciphertext -> ThinkingBlock.content is the
    named-dict form, not a positional list."""
    response = _make_response_with_reasoning(
        encrypted_content="enc_blob_xyz",
        reasoning_id="rs_named",
        summary=[SimpleNamespace(type="summary_text", text="thinking here")],
    )
    result = convert_response_with_accumulated_output(
        final_response=response,
        accumulated_output=list(response.output),
        continuation_count=0,
        chat_response_class=OpenAIChatResponse,
    )
    thinking_blocks = [
        b for b in result.content if getattr(b, "type", None) == "thinking"
    ]
    assert len(thinking_blocks) == 1
    block = thinking_blocks[0]
    assert block.content == [
        {
            "encrypted_content": "enc_blob_xyz",
            "id": "rs_named",
            "summary": "thinking here",
        }
    ], f"Expected named-dict content, got {block.content!r}"


def test_new_encoding_is_named_dict_streaming():
    """Same assertion via the accumulated-output (continuation/streaming
    merge) path -- `convert_response_with_accumulated_output` in
    _response_handling.py, exercised via its DICT-format branch (accumulated
    items merged from continuations are plain dicts, not SDK objects). This
    is a SEPARATE pair of ThinkingBlock construction sites from
    _convert_to_chat_response's in __init__.py -- discovered during
    implementation (not enumerated in the spec's original inventory, which
    covered only __init__.py) and fixed identically, since the spec's own
    stated rationale (named dict survives sanitize_for_json's key-dropping)
    applies verbatim to every construction site, not just the two the spec's
    investigation happened to read.
    """
    reasoning_block = {
        "type": "reasoning",
        "id": "rs_stream_named",
        "encrypted_content": "enc_stream_blob",
        "summary": [{"type": "summary_text", "text": "streamed thought"}],
    }
    message_block = {
        "type": "message",
        "content": [{"type": "output_text", "text": "hello"}],
    }
    final_response = SimpleNamespace(
        output=[reasoning_block, message_block],
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
        status="completed",
        id="resp_accum_test",
    )
    result = convert_response_with_accumulated_output(
        final_response=final_response,
        accumulated_output=[reasoning_block, message_block],
        continuation_count=1,
        chat_response_class=OpenAIChatResponse,
    )
    thinking_blocks = [
        b for b in result.content if getattr(b, "type", None) == "thinking"
    ]
    assert len(thinking_blocks) == 1, (
        f"Expected 1 ThinkingBlock from the accumulated-output dict path, "
        f"got {len(thinking_blocks)}. Content: {result.content}"
    )
    block = thinking_blocks[0]
    assert block.content == [
        {
            "encrypted_content": "enc_stream_blob",
            "id": "rs_stream_named",
            "summary": "streamed thought",
        }
    ], (
        f"Expected named-dict content from the accumulated-output dict path, got {block.content!r}"
    )


# ---------------------------------------------------------------------------
# The regression: named dict survives sanitize_for_json's key-dropping
# ---------------------------------------------------------------------------


def _sanitize_for_json_replica(value):
    """Local replica of amplifier_foundation.serialization.sanitize_for_json's
    None-dropping behavior (verified against
    amplifier-foundation/amplifier_foundation/serialization.py:18-86 at the
    time this spec was written: dict branch drops None VALUES, list branch
    drops None ENTRIES). Reimplemented here rather than taking a real
    dependency on amplifier_foundation from this provider module purely for
    a test -- amplifier_foundation is not (and should not become) a runtime
    dependency of amplifier_module_provider_openai.
    """
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            clean = _sanitize_for_json_replica(v)
            if clean is not None:
                out[k] = clean
        return out
    if isinstance(value, list):
        out_list = []
        for item in value:
            clean = _sanitize_for_json_replica(item)
            if clean is not None:
                out_list.append(clean)
        return out_list
    return value


def test_named_dict_survives_sanitize_for_json():
    """THE REGRESSION. A block with encrypted_content=None, sanitized via
    sanitize_for_json's documented None-dropping behavior, still yields its
    id when round-tripped through _decode_reasoning_state.

    Fails on the old positional encoding: sanitize_for_json's LIST branch
    drops None entries, collapsing [None, "rs_abc"] -> ["rs_abc"], a
    1-element list the old `len(content) >= 2` read guard discarded
    entirely -- silent, permanent data loss.
    """
    # Named-dict encoding, ciphertext absent (e.g. captured while chaining
    # before this fix, or a resume where `include` was never requested).
    block_content = [{"encrypted_content": None, "id": "rs_abc", "summary": "hi"}]
    sanitized = _sanitize_for_json_replica(block_content)

    # The dict survives key-dropping (encrypted_content key removed, since
    # its value is None) but the id keeps its unambiguous meaning.
    encrypted_content, reasoning_id, summary = _decode_reasoning_state(sanitized)
    assert reasoning_id == "rs_abc", (
        f"Expected id to survive sanitize_for_json round-trip; got "
        f"decoded=({encrypted_content!r}, {reasoning_id!r}, {summary!r}) from sanitized={sanitized!r}"
    )
    assert encrypted_content is None  # never fabricated
    assert summary == "hi"


# ---------------------------------------------------------------------------
# B2 -- back-compat reader: all three on-disk shapes
# ---------------------------------------------------------------------------


def test_legacy_positional_two_element_still_read():
    encrypted_content, reasoning_id, summary = _decode_reasoning_state(["ENC", "rs_a"])
    assert (encrypted_content, reasoning_id, summary) == ("ENC", "rs_a", None)


def test_legacy_collapsed_one_element_read_as_id():
    """A 1-element list is read as a bare id, NEVER as ciphertext."""
    encrypted_content, reasoning_id, summary = _decode_reasoning_state(["rs_a"])
    assert (encrypted_content, reasoning_id, summary) == (None, "rs_a", None)


def test_collapsed_non_rs_string_not_guessed():
    """A 1-element list whose string does NOT look like an rs_* id must not
    be fabricated into anything -- it is not recoverable state."""
    encrypted_content, reasoning_id, summary = _decode_reasoning_state(["ENC_no_id"])
    assert (encrypted_content, reasoning_id, summary) == (None, None, None)


def test_named_dict_missing_encrypted_key():
    """sanitize_for_json can drop the encrypted_content KEY entirely (dict
    branch also drops None values) -- must still decode the id."""
    encrypted_content, reasoning_id, summary = _decode_reasoning_state([{"id": "rs_a"}])
    assert (encrypted_content, reasoning_id, summary) == (None, "rs_a", None)


def test_collapsed_legacy_triggers_orphan_warning():
    """End-to-end: a collapsed legacy block (bare id, no ciphertext) is
    stripped by the existing orphan-stripping guard, AND warns -- silent loss
    becomes audible loss."""
    provider = _provider()
    from amplifier_module_provider_openai._constants import METADATA_REASONING_ITEMS

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "metadata": {METADATA_REASONING_ITEMS: ["rs_collapsed"]},
            "content": [
                _Block(
                    type="thinking",
                    thinking="",
                    content=["rs_collapsed"],  # collapsed legacy shape
                ),
                _Block(type="text", text="answer"),
            ],
        },
        {"role": "user", "content": "go"},
    ]
    result = provider._convert_messages(messages)
    reasoning_items = [
        m for m in result if isinstance(m, dict) and m.get("type") == "reasoning"
    ]
    assert reasoning_items == [], (
        f"Collapsed legacy block (bare id, no ciphertext) must be stripped "
        f"as an orphan; got {reasoning_items}"
    )


def test_empty_and_malformed_content():
    """None, [], a bare string, and a dict with nothing usable all decode to
    (None, None, None) without raising."""
    for bad in (None, [], "a string, not a list", [{}]):
        result = _decode_reasoning_state(bad)
        assert result == (None, None, None), (
            f"Expected safe no-op for {bad!r}, got {result}"
        )
