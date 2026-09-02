"""Tests for R0 -- `prompt_cache_mode` and the explicit cache-breakpoint sentinel.

WHAT R0 IS, from the measurement that justified it (probe P7 / arm A7,
gpt-5.6-terra, 3/3 reps, 2026-09-02):

  * A `prompt_cache_breakpoint` on a byte-constant `developer` item at
    `input[0]` writes a prefix that INCLUDES top-level `instructions` and
    `tools`: request 1 wrote 12,318-12,321 tokens against a ~12,330-token head;
    request 2, with a DIFFERENT user tail, read back the same count and wrote
    0 on the tail.

WHAT R0 IS NOT: a compaction remedy. Probe 7's pre-registered gate
(`G-P7-EXPLICIT`) FAILED -- a breakpoint written at the exact post-compaction
retained-prefix boundary still read back 0 after the shrink. Explicit mode does
not escape the grow-only cache.

The load-bearing default: `prompt_cache_mode="implicit"` must leave the request
BYTE-IDENTICAL to the pre-R0 shape. That is what
`test_default_mode_request_is_byte_identical_golden` locks down.
"""

import asyncio
import logging
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from amplifier_core import llm_errors as kernel_errors
from amplifier_core.message_models import ChatRequest, Message

from amplifier_module_provider_openai import (
    PROMPT_CACHE_SENTINEL_TEXT,
    OpenAIProvider,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_provider(**config_overrides) -> OpenAIProvider:
    config = {"max_retries": 0, "use_streaming": False, **config_overrides}
    return OpenAIProvider(api_key="test-key", config=config)


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


def _captured_params(provider: OpenAIProvider) -> Any:
    mock = cast(AsyncMock, provider.client.responses.create)
    return mock.call_args.kwargs


def _simple_request() -> ChatRequest:
    return ChatRequest(messages=[Message(role="user", content="Hello")])


def _multiturn_request() -> ChatRequest:
    """system + user + assistant + user -- the shape the golden below pins."""
    return ChatRequest(
        messages=[
            Message(role="system", content="You are helpful."),
            Message(role="user", content="First question"),
            Message(role="assistant", content="First answer"),
            Message(role="user", content="Second question"),
        ]
    )


def _run(provider: OpenAIProvider, request: ChatRequest, **kwargs) -> Any:
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())
    asyncio.run(provider.complete(request, **kwargs))
    return _captured_params(provider)


def _breakpoints(input_items: Any) -> list[tuple[int, int]]:
    """Every (item_index, block_index) carrying a prompt_cache_breakpoint."""
    found: list[tuple[int, int]] = []
    for i, item in enumerate(input_items):
        if not isinstance(item, dict):
            continue
        for j, block in enumerate(item.get("content") or []):
            if isinstance(block, dict) and "prompt_cache_breakpoint" in block:
                found.append((i, j))
    return found


# ---------------------------------------------------------------------------
# The default is a strict no-op (the gate this whole feature ships behind)
# ---------------------------------------------------------------------------


def test_default_mode_request_is_byte_identical_golden():
    """Default config must produce EXACTLY the pre-R0 request params.

    This golden is deliberately literal. If a change to this provider alters
    the default request shape at all, this test fails and the author has to
    say so out loud -- which is the point: every byte here is a prompt-cache
    prefix key, and a silent change to it is a silent cache-wide cold start.
    """
    provider = _make_provider(default_model="gpt-5.6-terra", reasoning_effort="high")
    params = _run(provider, _multiturn_request())

    assert params == {
        "model": "gpt-5.6-terra",
        "input": [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "First question"}],
            },
            {
                "type": "message",
                "id": "msg_8249fd89022b2bf7c6962570f1a2fd4c",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": "First answer",
                        "annotations": [],
                    }
                ],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "Second question"}],
            },
        ],
        "store": False,
        "instructions": "You are helpful.",
        "max_output_tokens": 128000,
        "reasoning": {"effort": "high", "summary": "detailed"},
        "include": ["reasoning.encrypted_content"],
        "prompt_cache_retention": "24h",
    }


def test_default_mode_emits_no_sentinel_and_no_options():
    provider = _make_provider(default_model="gpt-5.6-terra")
    params = _run(provider, _simple_request())
    assert "prompt_cache_options" not in params
    assert _breakpoints(params["input"]) == []
    assert all(PROMPT_CACHE_SENTINEL_TEXT not in str(item) for item in params["input"])


def test_default_mode_is_the_default():
    assert _make_provider().prompt_cache_mode == "implicit"


# ---------------------------------------------------------------------------
# Explicit mode -- request shape
# ---------------------------------------------------------------------------


def test_explicit_mode_injects_sentinel_at_input_zero():
    provider = _make_provider(
        default_model="gpt-5.6-terra", prompt_cache_mode="explicit"
    )
    params = _run(provider, _simple_request())

    sentinel = params["input"][0]
    assert sentinel == {
        "role": "developer",
        "content": [
            {
                "type": "input_text",
                "text": PROMPT_CACHE_SENTINEL_TEXT,
                "prompt_cache_breakpoint": {"mode": "explicit"},
            }
        ],
    }
    # The original conversation still follows, untouched.
    assert params["input"][1] == {
        "role": "user",
        "content": [{"type": "input_text", "text": "Hello"}],
    }


def test_explicit_mode_sends_prompt_cache_options_mode():
    provider = _make_provider(
        default_model="gpt-5.6-terra", prompt_cache_mode="explicit"
    )
    params = _run(provider, _simple_request())
    assert params["prompt_cache_options"] == {"mode": "explicit"}


def test_explicit_mode_preserves_configured_ttl():
    """`mode` is set by prompt_cache_mode; everything else the operator
    configured on prompt_cache_options survives."""
    provider = _make_provider(
        default_model="gpt-5.6-terra",
        prompt_cache_mode="explicit",
        prompt_cache_options={"ttl": "30m"},
    )
    params = _run(provider, _simple_request())
    assert params["prompt_cache_options"] == {"ttl": "30m", "mode": "explicit"}


def test_sentinel_is_byte_identical_across_requests_with_different_tails():
    """The whole mechanism depends on the sentinel being CONSTANT: A7's read
    hit came from request 2 carrying a different user tail behind the same
    sentinel."""
    provider = _make_provider(
        default_model="gpt-5.6-terra", prompt_cache_mode="explicit"
    )
    first = _run(provider, ChatRequest(messages=[Message(role="user", content="A")]))
    sentinel_a = first["input"][0]
    second = _run(
        provider,
        ChatRequest(messages=[Message(role="user", content="a different tail")]),
    )
    sentinel_b = second["input"][0]
    assert sentinel_a == sentinel_b
    assert first["input"][1] != second["input"][1]


# ---------------------------------------------------------------------------
# Explicit mode -- the second (stable-history) breakpoint
# ---------------------------------------------------------------------------


def test_stable_breakpoint_lands_on_last_user_item_before_the_tail():
    provider = _make_provider(
        default_model="gpt-5.6-terra",
        prompt_cache_mode="explicit",
        prompt_cache_stable_breakpoint=True,
    )
    params = _run(provider, _multiturn_request())
    input_items = params["input"]

    # [0]=sentinel, [1]=user "First question", [2]=assistant, [3]=user tail
    assert _breakpoints(input_items) == [(0, 0), (1, 0)]
    assert input_items[1]["content"][0]["text"] == "First question"
    # The dynamic tail must NOT carry one -- it changes every request.
    assert "prompt_cache_breakpoint" not in input_items[-1]["content"][0]


def test_stable_breakpoint_never_lands_on_an_assistant_carrier():
    """P7: a breakpoint on an assistant `output_text` block is accepted with
    HTTP 200 and silently writes NOTHING. Only input_text carriers count."""
    provider = _make_provider(
        default_model="gpt-5.6-terra",
        prompt_cache_mode="explicit",
        prompt_cache_stable_breakpoint=True,
    )
    request = ChatRequest(
        messages=[
            Message(role="user", content="Question"),
            Message(role="assistant", content="Answer"),
        ]
    )
    params = _run(provider, request)
    for item_idx, _ in _breakpoints(params["input"]):
        item = params["input"][item_idx]
        assert item.get("role") in ("user", "developer")
        assert item["content"][0]["type"] == "input_text"


def test_stable_breakpoint_is_off_by_default():
    """MEASURED, not assumed (rig 20260902-r0-validate): the second breakpoint
    wrote 2,240-2,264 tokens at the 1.25x write rate on every request and the
    next request still read back only the sentinel prefix. It costs and returns
    nothing, so it is off unless someone opts in to probe it again."""
    provider = _make_provider(
        default_model="gpt-5.6-terra", prompt_cache_mode="explicit"
    )
    assert provider.prompt_cache_stable_breakpoint is False
    params = _run(provider, _multiturn_request())
    assert _breakpoints(params["input"]) == [(0, 0)]


def test_stable_breakpoint_can_be_disabled_explicitly():
    provider = _make_provider(
        default_model="gpt-5.6-terra",
        prompt_cache_mode="explicit",
        prompt_cache_stable_breakpoint=False,
    )
    params = _run(provider, _multiturn_request())
    assert _breakpoints(params["input"]) == [(0, 0)]


def test_stable_breakpoint_accepts_string_bools():
    """The wizard/YAML write booleans as strings; `bool("false")` is True."""
    provider = _make_provider(
        default_model="gpt-5.6-terra",
        prompt_cache_mode="explicit",
        prompt_cache_stable_breakpoint="true",
    )
    assert provider.prompt_cache_stable_breakpoint is True
    params = _run(provider, _multiturn_request())
    assert len(_breakpoints(params["input"])) == 2

    off = _make_provider(
        default_model="gpt-5.6-terra",
        prompt_cache_mode="explicit",
        prompt_cache_stable_breakpoint="false",
    )
    assert off.prompt_cache_stable_breakpoint is False


def test_never_exceeds_the_breakpoint_budget():
    """<=3 explicit breakpoints is this provider's self-imposed budget (the
    API allows <=4 cache writes per request). We emit at most 2, on a long
    conversation as well as a short one."""
    provider = _make_provider(
        default_model="gpt-5.6-terra", prompt_cache_mode="explicit"
    )
    messages = [Message(role="system", content="sys")]
    for i in range(12):
        messages.append(Message(role="user", content=f"q{i}"))
        messages.append(Message(role="assistant", content=f"a{i}"))
    messages.append(Message(role="user", content="final"))
    params = _run(provider, ChatRequest(messages=messages))
    assert len(_breakpoints(params["input"])) == 1
    with_stable = _make_provider(
        default_model="gpt-5.6-terra",
        prompt_cache_mode="explicit",
        prompt_cache_stable_breakpoint=True,
    )
    params = _run(with_stable, ChatRequest(messages=messages))
    assert len(_breakpoints(params["input"])) == 2


def test_single_message_conversation_gets_only_the_sentinel():
    """Nothing but the tail exists, so there is no stable item to mark."""
    provider = _make_provider(
        default_model="gpt-5.6-terra", prompt_cache_mode="explicit"
    )
    params = _run(provider, _simple_request())
    assert _breakpoints(params["input"]) == [(0, 0)]


# ---------------------------------------------------------------------------
# Config surface
# ---------------------------------------------------------------------------


def test_invalid_prompt_cache_mode_fails_loud_at_mount():
    with pytest.raises(kernel_errors.InvalidRequestError) as exc:
        _make_provider(prompt_cache_mode="on")
    assert "prompt_cache_mode" in str(exc.value)


def test_per_call_kwarg_overrides_config_mode():
    provider = _make_provider(default_model="gpt-5.6-terra")
    params = _run(provider, _simple_request(), prompt_cache_mode="explicit")
    assert params["prompt_cache_options"] == {"mode": "explicit"}
    assert params["input"][0]["role"] == "developer"


def test_prompt_cache_mode_is_a_known_config_key():
    """No spurious 'unknown config key' warning for the new keys."""
    from amplifier_module_provider_openai import _KNOWN_CONFIG_KEYS

    assert {
        "prompt_cache_mode",
        "prompt_cache_stable_breakpoint",
        "reasoning_context",
    } <= _KNOWN_CONFIG_KEYS


# ---------------------------------------------------------------------------
# The ~10x guard: explicit options with zero breakpoints stays impossible
# ---------------------------------------------------------------------------


def test_options_explicit_alone_is_still_dropped_at_mount(caplog):
    """Unchanged pre-R0 behavior: asking for explicit MODE without asking for
    the breakpoint mechanism would disable caching entirely."""
    with caplog.at_level(logging.WARNING):
        provider = _make_provider(prompt_cache_options={"mode": "explicit"})
    assert provider.prompt_cache_options is None
    assert "prompt_cache_mode" in caplog.text


def test_options_explicit_is_kept_when_breakpoints_are_enabled():
    provider = _make_provider(
        prompt_cache_mode="explicit",
        prompt_cache_options={"mode": "explicit", "ttl": "30m"},
    )
    assert provider.prompt_cache_options == {"mode": "explicit", "ttl": "30m"}


def test_options_explicit_ttl_still_passes_through_when_mode_dropped():
    provider = _make_provider(prompt_cache_options={"mode": "explicit", "ttl": "30m"})
    assert provider.prompt_cache_options == {"ttl": "30m"}
