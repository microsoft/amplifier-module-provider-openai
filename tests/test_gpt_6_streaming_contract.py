"""Synthetic happy-path streaming contract coverage for GPT-6 Sol/Luna.

Uses the same `FakeEventStream`/`MockStreamContext`/`_make_event`/
`_make_item` fake-SDK-event pattern as tests/test_streaming.py (duplicated
here, following this repo's existing per-file convention rather than
cross-importing another test module). Does not touch the parser: this
only builds fixtures and asserts on its existing, already-tested output.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

import pytest
from amplifier_core import ModuleCoordinator
from amplifier_core.message_models import (
    ChatRequest,
    Message,
    TextBlock,
    ThinkingBlock,
    ToolCallBlock,
)

from amplifier_module_provider_openai import OpenAIProvider

# ---------------------------------------------------------------------------
# Helpers (mirrors tests/test_streaming.py's fake-stream pattern)
# ---------------------------------------------------------------------------


class FakeHooks:
    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    async def emit(self, name: str, payload: dict) -> None:
        self.events.append((name, payload))


class FakeCoordinator:
    def __init__(self):
        self.hooks = FakeHooks()


class MockStreamContext:
    def __init__(self, stream):
        self._stream = stream

    async def __aenter__(self):
        return self._stream

    async def __aexit__(self, *args):
        pass


class FakeEventStream:
    """Async-iterable stream yielding fixed synthetic SDK events, then
    returning a pre-built terminal response from get_final_response()."""

    def __init__(self, events, response):
        self._events = list(events)
        self._response = SimpleNamespace(headers={})
        self._final_response = response
        self._pos = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._pos >= len(self._events):
            raise StopAsyncIteration
        ev = self._events[self._pos]
        self._pos += 1
        return ev

    async def get_final_response(self):
        return self._final_response


def _make_event(type_str, **kwargs):
    return SimpleNamespace(type=type_str, **kwargs)


def _make_item(**kwargs):
    return SimpleNamespace(**kwargs)


def _simple_request() -> ChatRequest:
    return ChatRequest(messages=[Message(role="user", content="Hello")])


def _final_response(*, model: str) -> SimpleNamespace:
    """Terminal Responses-API object: reasoning + text + one tool call,
    with the full cache/reasoning usage detail fields the task asks for."""
    return SimpleNamespace(
        id="resp_gpt6_stream",
        model=model,
        status="completed",
        service_tier="default",
        output=[
            SimpleNamespace(
                type="reasoning",
                id="rs_1",
                encrypted_content=None,
                summary=[
                    SimpleNamespace(type="summary_text", text="Thinking it through.")
                ],
            ),
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text="Done.", raw=None)],
            ),
            SimpleNamespace(
                type="function_call",
                call_id="call_123",
                id="fc_123",
                name="search",
                arguments='{"query": "hello"}',
                status="completed",
            ),
        ],
        incomplete_details=None,
        usage=SimpleNamespace(
            input_tokens=1_200,
            output_tokens=340,
            output_tokens_details=SimpleNamespace(reasoning_tokens=200),
            input_tokens_details=SimpleNamespace(
                cached_tokens=500, cache_write_tokens=100
            ),
        ),
    )


def _events_for(model: str) -> list[SimpleNamespace]:
    return [
        # Reasoning block: added -> summary delta -> done
        _make_event(
            "response.output_item.added",
            output_index=0,
            item=_make_item(type="reasoning"),
        ),
        _make_event(
            "response.reasoning_summary_text.delta",
            output_index=0,
            delta="Thinking it through.",
        ),
        _make_event(
            "response.output_item.done",
            output_index=0,
            item=_make_item(type="reasoning"),
        ),
        # Text block: added -> delta -> done
        _make_event(
            "response.output_item.added",
            output_index=1,
            item=_make_item(type="message"),
        ),
        _make_event(
            "response.output_text.delta",
            output_index=1,
            delta="Done.",
        ),
        _make_event(
            "response.output_item.done",
            output_index=1,
            item=_make_item(type="message"),
        ),
        # Tool-call block: added (with name) -> (ignored arg delta) -> done
        _make_event(
            "response.output_item.added",
            output_index=2,
            item=_make_item(type="function_call", name="search"),
        ),
        _make_event(
            "response.function_call_arguments.delta",
            output_index=2,
            delta='{"query": "hello"}',
        ),
        _make_event(
            "response.output_item.done",
            output_index=2,
            item=_make_item(type="function_call", name="search"),
        ),
    ]


# ===========================================================================
# Happy-path streaming contract: block lifecycle, tool args, usage, cost
# ===========================================================================


@pytest.mark.parametrize("model", ["gpt-6-sol", "gpt-6-luna"])
def test_sol_luna_stream_happy_path_lifecycle_and_terminal_response(model: str) -> None:
    provider = OpenAIProvider(
        api_key="[REDACTED:SECRET]",
        config={
            "default_model": model,
            "use_streaming": True,
            "max_retries": 0,
        },
    )
    fake_coordinator = FakeCoordinator()
    provider.coordinator = cast(ModuleCoordinator, fake_coordinator)

    stream = FakeEventStream(_events_for(model), _final_response(model=model))
    provider.client.responses.stream = MagicMock(return_value=MockStreamContext(stream))

    result = asyncio.run(provider.complete(_simple_request()))

    # --- balanced stream lifecycle: no aborted event, 3 blocks each with
    # start (+delta) + end, in strict order ---------------------------------
    stream_events = [
        (name, payload)
        for name, payload in fake_coordinator.hooks.events
        if name.startswith("llm:stream_")
    ]
    names = [n for n, _ in stream_events]
    assert names == [
        "llm:stream_block_start",  # reasoning
        "llm:stream_block_delta",
        "llm:stream_block_end",
        "llm:stream_block_start",  # text
        "llm:stream_block_delta",
        "llm:stream_block_end",
        "llm:stream_block_start",  # tool_use (no delta -- silently ignored)
        "llm:stream_block_end",
    ], f"Got: {names}"
    assert "llm:stream_aborted" not in {n for n, _ in fake_coordinator.hooks.events}

    _, reasoning_start = stream_events[0]
    assert reasoning_start["block_type"] == "thinking"
    _, text_start = stream_events[3]
    assert text_start["block_type"] == "text"
    _, tool_start = stream_events[6]
    assert tool_start["block_type"] == "tool_use"
    assert tool_start["name"] == "search"

    # --- terminal response model: content blocks in output order -----------
    assert isinstance(result.content, list)
    thinking_blocks = [b for b in result.content if isinstance(b, ThinkingBlock)]
    text_blocks = [b for b in result.content if isinstance(b, TextBlock)]
    tool_blocks = [b for b in result.content if isinstance(b, ToolCallBlock)]
    assert len(thinking_blocks) == 1
    assert thinking_blocks[0].thinking == "Thinking it through."
    assert len(text_blocks) == 1
    assert text_blocks[0].text == "Done."

    # --- parsed tool args: JSON string arguments become a real dict --------
    assert len(tool_blocks) == 1
    assert tool_blocks[0].id == "call_123"
    assert tool_blocks[0].name == "search"
    assert tool_blocks[0].input == {"query": "hello"}
    assert result.tool_calls is not None
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].arguments == {"query": "hello"}

    # --- usage: cached/cache_write/reasoning tokens + cost ------------------
    usage = result.usage
    assert usage.reasoning_tokens == 200
    assert usage.cache_read_tokens == 500
    assert usage.cache_write_tokens == 100
    # input_tokens is normalized: raw gross (1200) minus cache_write (100).
    assert usage.input_tokens == 1_100
    assert usage.output_tokens == 340
    assert usage.cost_usd is not None
    assert isinstance(usage.cost_usd, Decimal)
