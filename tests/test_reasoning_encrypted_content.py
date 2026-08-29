"""Tests for ThinkingBlock creation when encrypted_content exists without summary text.

Bug: 4 locations gate ThinkingBlock creation on `if reasoning_text:` but when a
model reasons by default without explicit `reasoning.summary` config, the API
returns `encrypted_content` with NO summary text. The ThinkingBlock is never
created, `encrypted_content` is discarded, and multi-turn reasoning breaks.

Fix: ThinkingBlock must be created whenever we have EITHER displayable reasoning
text OR encrypted state to preserve for multi-turn.

Also tests auto-enabling reasoning summary for models that reason by default.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from amplifier_core.message_models import ChatRequest, Message, ThinkingBlock

from amplifier_module_provider_openai import OpenAIChatResponse, OpenAIProvider
from amplifier_module_provider_openai._response_handling import (
    convert_response_with_accumulated_output,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_provider(**config_overrides) -> OpenAIProvider:
    config = {"max_retries": 0, "use_streaming": False, **config_overrides}
    provider = OpenAIProvider(api_key="test-key", config=config)
    return provider


def _make_response_with_reasoning(
    *,
    encrypted_content=None,
    reasoning_id="rs_test_001",
    summary=None,
):
    """Create a DummyResponse with a reasoning block and a message block."""
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


def _get_call_kwargs(provider: OpenAIProvider) -> dict:
    """Extract the kwargs from the last client.responses.create call."""
    return provider.client.responses.create.await_args_list[-1].kwargs


# ---------------------------------------------------------------------------
# Test 1: encrypted_content only -> ThinkingBlock IS created
# ---------------------------------------------------------------------------


def test_thinking_block_created_with_encrypted_content_only():
    """When API returns reasoning with encrypted_content but no summary,
    ThinkingBlock IS created with content=[encrypted, id].

    This is the core bug fix: models that reason by default may return
    encrypted_content without any summary text. The ThinkingBlock must
    still be created to preserve reasoning state for multi-turn.
    """
    response = _make_response_with_reasoning(
        encrypted_content="encrypted_blob_abc123",
        reasoning_id="rs_enc_only",
        summary=None,  # No summary text at all
    )
    accumulated_output = list(response.output)

    result = convert_response_with_accumulated_output(
        final_response=response,
        accumulated_output=accumulated_output,
        continuation_count=0,
        chat_response_class=OpenAIChatResponse,
    )

    # Find ThinkingBlocks in content
    thinking_blocks = [b for b in result.content if isinstance(b, ThinkingBlock)]
    assert len(thinking_blocks) == 1, (
        f"Expected 1 ThinkingBlock from encrypted_content-only reasoning, "
        f"got {len(thinking_blocks)}. Content: {result.content}"
    )
    block = thinking_blocks[0]
    assert block.content == [
        {
            "encrypted_content": "encrypted_blob_abc123",
            "id": "rs_enc_only",
            "summary": None,
        }
    ]  # B1: named-dict encoding, not positional
    assert block.thinking == ""  # Empty string, not None — no summary text


# ---------------------------------------------------------------------------
# Test 2: summary only -> ThinkingBlock created (existing behavior)
# ---------------------------------------------------------------------------


def test_thinking_block_created_with_summary_only():
    """Existing behavior: summary text without encrypted_content still creates
    a ThinkingBlock."""
    response = _make_response_with_reasoning(
        encrypted_content=None,
        reasoning_id="rs_summary_only",
        summary=[SimpleNamespace(type="summary_text", text="I thought about it")],
    )
    accumulated_output = list(response.output)

    result = convert_response_with_accumulated_output(
        final_response=response,
        accumulated_output=accumulated_output,
        continuation_count=0,
        chat_response_class=OpenAIChatResponse,
    )

    thinking_blocks = [b for b in result.content if isinstance(b, ThinkingBlock)]
    assert len(thinking_blocks) == 1, (
        f"Expected 1 ThinkingBlock from summary-only reasoning, "
        f"got {len(thinking_blocks)}"
    )
    block = thinking_blocks[0]
    assert block.thinking == "I thought about it"
    assert block.content == [
        {
            "encrypted_content": None,
            "id": "rs_summary_only",
            "summary": "I thought about it",
        }
    ]  # B1: named-dict encoding, not positional


# ---------------------------------------------------------------------------
# Test 3: both encrypted_content and summary -> ThinkingBlock with all fields
# ---------------------------------------------------------------------------


def test_thinking_block_created_with_both():
    """Both encrypted_content and summary -> ThinkingBlock with all fields."""
    response = _make_response_with_reasoning(
        encrypted_content="encrypted_blob_xyz",
        reasoning_id="rs_both",
        summary=[SimpleNamespace(type="summary_text", text="Step-by-step reasoning")],
    )
    accumulated_output = list(response.output)

    result = convert_response_with_accumulated_output(
        final_response=response,
        accumulated_output=accumulated_output,
        continuation_count=0,
        chat_response_class=OpenAIChatResponse,
    )

    thinking_blocks = [b for b in result.content if isinstance(b, ThinkingBlock)]
    assert len(thinking_blocks) == 1
    block = thinking_blocks[0]
    assert block.thinking == "Step-by-step reasoning"
    assert block.content == [
        {
            "encrypted_content": "encrypted_blob_xyz",
            "id": "rs_both",
            "summary": "Step-by-step reasoning",
        }
    ]  # B1: named-dict encoding, not positional


# ---------------------------------------------------------------------------
# Test 4: dict format — encrypted_content only (covers _response_handling dict path)
# ---------------------------------------------------------------------------


def test_thinking_block_created_with_encrypted_content_only_dict_format():
    """Same as test 1 but using dict-format blocks (the second code path
    in _response_handling.py)."""
    reasoning_block = {
        "type": "reasoning",
        "id": "rs_dict_enc",
        "encrypted_content": "encrypted_dict_blob",
        "summary": None,
    }
    message_block = {
        "type": "message",
        "content": [{"type": "output_text", "text": "Hello dict"}],
    }
    response = SimpleNamespace(
        output=[reasoning_block, message_block],
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
        status="completed",
        id="resp_dict_test",
    )
    accumulated_output = [reasoning_block, message_block]

    result = convert_response_with_accumulated_output(
        final_response=response,
        accumulated_output=accumulated_output,
        continuation_count=0,
        chat_response_class=OpenAIChatResponse,
    )

    thinking_blocks = [b for b in result.content if isinstance(b, ThinkingBlock)]
    assert len(thinking_blocks) == 1, (
        f"Expected 1 ThinkingBlock from dict-format encrypted_content-only, "
        f"got {len(thinking_blocks)}. Content: {result.content}"
    )
    block = thinking_blocks[0]
    assert block.content == [
        {
            "encrypted_content": "encrypted_dict_blob",
            "id": "rs_dict_enc",
            "summary": None,
        }
    ]  # B1: named-dict encoding, not positional
    assert block.thinking == ""


# ---------------------------------------------------------------------------
# Test 5: auto-reasoning summary for implicit reasoning models
# ---------------------------------------------------------------------------


def test_auto_reasoning_summary_for_implicit_reasoning_model():
    """gpt-5.2-codex without explicit reasoning config gets
    `reasoning: {summary: "auto"}` in params so the API returns summaries
    for observability."""
    provider = _make_provider(default_model="gpt-5.2-codex")
    provider.client.responses.create = AsyncMock(
        return_value=_make_response_with_reasoning(
            encrypted_content="enc", summary=None
        )
    )

    request = ChatRequest(
        messages=[Message(role="user", content="Hello")],
        reasoning_effort=None,  # No explicit reasoning
    )
    asyncio.run(provider.complete(request))

    kwargs = _get_call_kwargs(provider)
    assert "reasoning" in kwargs, (
        "Implicit reasoning model (gpt-5.2-codex) should auto-set reasoning param "
        f"for summary, but 'reasoning' not in kwargs. Keys: {list(kwargs.keys())}"
    )
    assert kwargs["reasoning"] == {"summary": "auto"}


# ---------------------------------------------------------------------------
# Test 6: non-reasoning model does NOT get auto-reasoning
# ---------------------------------------------------------------------------


def test_no_auto_reasoning_for_non_reasoning_model():
    """gpt-4.1-mini doesn't get auto-reasoning."""
    provider = _make_provider(default_model="gpt-4.1-mini")
    provider.client.responses.create = AsyncMock(
        return_value=_make_response_with_reasoning(summary=None)
    )

    request = ChatRequest(
        messages=[Message(role="user", content="Hello")],
        reasoning_effort=None,
    )
    asyncio.run(provider.complete(request))

    kwargs = _get_call_kwargs(provider)
    assert "reasoning" not in kwargs, (
        "Non-reasoning model (gpt-4.1-mini) should NOT have reasoning param, "
        f"but got reasoning={kwargs.get('reasoning')}"
    )


# ---------------------------------------------------------------------------
# Test 7 (PR-B regression): encrypted_content include is suppressed when chaining
# ---------------------------------------------------------------------------


def test_encrypted_content_requested_under_chaining():
    """A5 (reasoning-continuity-fix spec, probe P1 PASS): include is now
    UNCONDITIONAL, even when chaining is active.

    Superseded regression for PR-B: PR-B originally dropped encrypted_content
    while chaining because it was assumed to bust the cache prefix. Probe P1
    (2026-08-29, probes/p1_include_chained_probe.py) measured this live on
    gpt-5.6-luna and found it cache-neutral (0-4 tokens out of ~1,500-1,600
    on hops with an actual prefix to cache) -- `include` only changes the
    *response* shape, not the *request* prefix caching keys on. So ciphertext
    is now always captured, letting every reset/resume path replay reasoning
    regardless of whether the response was chained (see Change B).

    Chaining is "auto" by default and resolves to True for gpt-5.5
    (supports_reasoning=True). So with default config on a reasoning model,
    include=[reasoning.encrypted_content] must be present in the API params.
    """
    provider = _make_provider(default_model="gpt-5.5")
    provider.client.responses.create = AsyncMock(
        return_value=_make_response_with_reasoning(
            encrypted_content="enc",
            reasoning_id="rs_test_chaining",
            summary=None,
        )
    )

    request = ChatRequest(
        messages=[Message(role="user", content="Hello")],
    )
    asyncio.run(provider.complete(request))

    kwargs = _get_call_kwargs(provider)
    include = kwargs.get("include", [])
    assert "reasoning.encrypted_content" in include, (
        "With chaining active (auto + reasoning model), "
        "include=reasoning.encrypted_content must still be sent (probe P1 PASS). "
        f"Got include={include}"
    )
