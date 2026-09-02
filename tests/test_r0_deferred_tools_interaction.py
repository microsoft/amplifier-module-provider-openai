"""Interaction tests for R0 (explicit prompt-cache mode) x deferred tool loading.

These two features landed independently and BOTH rewrite the params assembly:

* R0 (`prompt_cache_mode="explicit"`) PREPENDS a byte-constant developer
  sentinel at `input[0]` and may mark one further "stable" breakpoint.
* deferred loading (`tool_loading="deferred_namespace"`) rewrites the `tools`
  block into namespaces and APPENDS an `additional_tools` INPUT item at the
  tail for any tool that shows up mid-session.

They meet in `_prepare_request_params`, so the rebase that put them in the same
tree needs its own guardrail. Three things are asserted here:

1. **Both flags off -> byte-identical request.** The composite default must be
   what shipped before EITHER feature existed. Written as a literal expected
   payload (so a failure shows the diff) plus a sha256 over the canonical
   serialization, with negative controls proving each flag alone moves the
   bytes -- a digest nothing can break is not a guardrail.
2. **Ordering.** deferred appends at the tail BEFORE R0 prepends the sentinel,
   so the sentinel keeps slot 0 (its whole value is being a byte-constant
   prefix) and the discovered-tools item keeps the tail (`TS:893` -- once
   emitted its position must never move).
3. **No breakpoint ever lands on the discovered-tools item.** It is the one
   item in the array that grows mid-session; a breakpoint behind it would put
   a moving payload inside the cached prefix, which is the exact failure the
   append-shaped design exists to avoid.
"""

import asyncio
import hashlib
import json
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

from amplifier_core.message_models import ChatRequest, Message, ToolSpec

from amplifier_module_provider_openai import (
    PROMPT_CACHE_SENTINEL_TEXT,
    OpenAIProvider,
)

# Two tools that fall in two different namespaces, so the deferred arm is
# visibly different from the static one.
ROSTER: tuple[tuple[str, str], ...] = (
    ("read_file", "Read a file"),
    ("bash", "Run a shell command"),
)


def _tool_specs(names: tuple[tuple[str, str], ...] = ROSTER) -> list[ToolSpec]:
    return [
        ToolSpec(
            name=name,
            description=desc,
            parameters={"type": "object", "properties": {}},
        )
        for name, desc in names
    ]


def _make_provider(**config_overrides) -> OpenAIProvider:
    config = {"max_retries": 0, "use_streaming": False, **config_overrides}
    coordinator = MagicMock()
    coordinator.get_capability = MagicMock(return_value=None)
    coordinator.hooks = MagicMock()
    coordinator.hooks.emit = AsyncMock(return_value=None)
    return OpenAIProvider(api_key="test-key", config=config, coordinator=coordinator)


def _request(tools: list[ToolSpec] | None = None) -> ChatRequest:
    return ChatRequest(
        messages=[Message(role="user", content="Hello")],
        tools=tools if tools is not None else _tool_specs(),
    )


class DummyResponse:
    def __init__(self):
        self.output = [
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text="Hi")],
            )
        ]
        self.usage = SimpleNamespace(input_tokens=1, output_tokens=1)
        self.status = "completed"
        self.id = "resp_test"


def _run(provider: OpenAIProvider, request: ChatRequest) -> dict[str, Any]:
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())
    asyncio.run(provider.complete(request))
    return cast(AsyncMock, provider.client.responses.create).call_args.kwargs


def _canon(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _digest(obj: Any) -> str:
    return hashlib.sha256(_canon(obj).encode()).hexdigest()


# ---------------------------------------------------------------------------
# 1. BOTH FLAGS OFF -> BYTE-IDENTICAL
# ---------------------------------------------------------------------------


def _expected_default_params() -> dict[str, Any]:
    """The whole request this provider emitted before R0 and deferred loading.

    Literal, not derived: a payload computed from the code under test would
    pass no matter what that code did.
    """
    return {
        "model": "gpt-5.6-sol",
        "input": [
            {"role": "user", "content": [{"type": "input_text", "text": "Hello"}]}
        ],
        "store": False,
        "max_output_tokens": 128000,
        "include": ["reasoning.encrypted_content"],
        "tools": [
            {
                "type": "function",
                "name": name,
                "description": desc,
                "parameters": {"type": "object", "properties": {}},
            }
            for name, desc in ROSTER
        ],
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "prompt_cache_retention": "24h",
    }


def test_both_flags_off_request_is_byte_identical():
    """The composite default. Both features are strict no-ops together."""
    params = _run(_make_provider(), _request())

    assert params == _expected_default_params()
    assert _digest(params) == _digest(_expected_default_params())

    # And structurally: neither feature left a trace.
    assert all(
        PROMPT_CACHE_SENTINEL_TEXT not in _canon(item) for item in params["input"]
    )
    assert "prompt_cache_options" not in params
    assert not any(
        isinstance(i, dict) and i.get("type") == "additional_tools"
        for i in params["input"]
    )


def test_explicit_flags_are_also_off_when_the_other_feature_is_on():
    """Cross-check: turning ONE feature on must not switch the other on."""
    deferred_only = _run(_make_provider(tool_loading="deferred_namespace"), _request())
    assert "prompt_cache_options" not in deferred_only
    assert all(
        PROMPT_CACHE_SENTINEL_TEXT not in _canon(i) for i in deferred_only["input"]
    )

    explicit_only = _run(_make_provider(prompt_cache_mode="explicit"), _request())
    assert explicit_only["tools"] == _expected_default_params()["tools"]
    assert not any(
        isinstance(i, dict) and i.get("type") == "additional_tools"
        for i in explicit_only["input"]
    )


def test_negative_control_each_flag_alone_moves_the_bytes():
    """Proves the digest above discriminates. A guardrail that cannot fail is
    not a guardrail -- these are the two ways it must fail."""
    baseline = _digest(_run(_make_provider(), _request()))

    explicit = _digest(_run(_make_provider(prompt_cache_mode="explicit"), _request()))
    deferred = _digest(
        _run(_make_provider(tool_loading="deferred_namespace"), _request())
    )
    both = _digest(
        _run(
            _make_provider(
                prompt_cache_mode="explicit", tool_loading="deferred_namespace"
            ),
            _request(),
        )
    )

    assert explicit != baseline
    assert deferred != baseline
    assert both != baseline
    assert len({baseline, explicit, deferred, both}) == 4


# ---------------------------------------------------------------------------
# 2 + 3. BOTH FLAGS ON -> ORDERING AND BREAKPOINT PLACEMENT
# ---------------------------------------------------------------------------


def _both_on() -> OpenAIProvider:
    return _make_provider(
        prompt_cache_mode="explicit",
        prompt_cache_stable_breakpoint=True,
        tool_loading="deferred_namespace",
    )


def test_sentinel_keeps_slot_zero_and_discovered_tools_keep_the_tail():
    provider = _both_on()

    first = _run(provider, _request())
    assert first["input"][0]["role"] == "developer"
    assert first["input"][0]["content"][0]["text"] == PROMPT_CACHE_SENTINEL_TEXT
    assert first["prompt_cache_options"]["mode"] == "explicit"

    # A tool that appears mid-session rides an additional_tools item.
    late = _tool_specs(ROSTER + (("terminal_inspector", "Drive a terminal app"),))
    second = _run(provider, _request(late))

    # Sentinel still first: it is the cached prefix, it cannot be displaced.
    assert second["input"][0]["content"][0]["text"] == PROMPT_CACHE_SENTINEL_TEXT
    # Discovered tools still last (TS:893).
    tail = second["input"][-1]
    assert isinstance(tail, dict) and tail["type"] == "additional_tools"
    assert [t["name"] for t in tail["tools"]] == ["terminal_inspector"]
    # The namespaced tools block is the cached head: it did not move a byte.
    assert _canon(second["tools"]) == _canon(first["tools"])


def test_no_breakpoint_ever_lands_on_the_discovered_tools_item():
    """The additional_tools item is the one input item that GROWS mid-session.
    A breakpoint at or behind it would pull a moving payload into the cached
    prefix -- the exact failure the append-shaped design exists to avoid."""
    provider = _both_on()
    _run(provider, _request())
    late = _tool_specs(ROSTER + (("terminal_inspector", "Drive a terminal app"),))
    params = _run(provider, _request(late))

    items = params["input"]
    marked = [
        idx
        for idx, item in enumerate(items)
        if "prompt_cache_breakpoint" in _canon(item)
    ]
    assert items[-1]["type"] == "additional_tools"
    assert "prompt_cache_breakpoint" not in _canon(items[-1])
    # Every breakpoint sits strictly BEFORE the discovered-tools item, so the
    # growing payload is always outside the cached prefix.
    assert marked  # the sentinel at minimum
    assert max(marked) < len(items) - 1
    for idx in marked:
        assert items[idx].get("type") != "additional_tools"


def test_discovered_tools_item_promotes_the_last_history_message():
    """The one MEASURED behaviour delta of the two features combined, asserted
    rather than hidden.

    R0's stable-breakpoint heuristic skips the FINAL input item because that is
    the dynamic tail. In deferred mode the final item is the `additional_tools`
    item, so the last real history message is no longer final and becomes an
    eligible carrier -- a 2nd breakpoint fires in a short conversation where
    R0 alone would place only the sentinel.

    This is correct, not a defect: the `additional_tools` item is pinned to the
    tail on every subsequent request, so the prefix through that history
    message IS byte-identical next request, which is exactly what the cache
    matches on. It is recorded here because it costs one extra cache write
    (R0's own rig: explicit $0.006326 -> explicit+stable $0.007458) that an
    operator enabling BOTH flags would otherwise not see coming.
    """
    short = _request()

    r0_only = _run(
        _make_provider(
            prompt_cache_mode="explicit", prompt_cache_stable_breakpoint=True
        ),
        short,
    )
    r0_marks = [
        i
        for i, it in enumerate(r0_only["input"])
        if "prompt_cache_breakpoint" in _canon(it)
    ]
    assert r0_marks == [0]  # sentinel only: the user turn is the final item

    provider = _both_on()
    _run(provider, short)
    late = _tool_specs(ROSTER + (("terminal_inspector", "Drive a terminal app"),))
    both = _run(provider, _request(late))
    both_marks = [
        i
        for i, it in enumerate(both["input"])
        if "prompt_cache_breakpoint" in _canon(it)
    ]

    assert both_marks == [0, 1]
    assert both["input"][1]["role"] == "user"
    # Still inside R0's own <=2 breakpoint budget.
    assert len(both_marks) <= 2


def test_stable_breakpoint_never_selects_the_additional_tools_item():
    """With enough history for the stable-breakpoint heuristic to fire, it must
    still pick a real history item -- never the discovered-tools tail."""
    provider = _both_on()
    _run(provider, _request())

    late = _tool_specs(ROSTER + (("terminal_inspector", "Drive a terminal app"),))
    request = ChatRequest(
        messages=[
            Message(role="user", content="First"),
            Message(role="assistant", content="Ack"),
            Message(role="user", content="Second"),
        ],
        tools=late,
    )
    params = _run(provider, request)
    items = params["input"]

    assert items[-1]["type"] == "additional_tools"
    marked = [
        idx
        for idx, item in enumerate(items)
        if "prompt_cache_breakpoint" in _canon(item)
    ]
    assert 0 in marked  # sentinel
    assert len(items) - 1 not in marked  # never the discovered-tools item
    for idx in marked:
        assert items[idx].get("type") != "additional_tools"
