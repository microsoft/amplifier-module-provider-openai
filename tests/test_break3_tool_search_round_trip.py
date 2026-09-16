"""FAIL-BEFORE guard for break 3: hosted tool-search items must round-trip.

WHY THIS FILE EXISTS SEPARATELY FROM `test_tool_search_namespaces.py`
---------------------------------------------------------------------
`test_tool_search_namespaces.py` tests the *request-assembly* feature and
imports `_tool_search`, so at a tree without the feature it fails as a
collection error. That is a weak fail-before: it proves a module is absent,
not that a behaviour is wrong.

This file deliberately imports **nothing new**. It drives the provider's
public path only and asserts on the wire-level metadata key by its literal
string. At `main` (`f104e6c`) every test here fails as a real assertion --
the hosted items are silently dropped by the two

    if block_type not in {"tool_call", "function_call"}: continue

guards in `_response_handling` / `__init__`, exactly as break 3 describes.
Per `TS:854`, a dropped `tool_search_output` makes every tool it loaded cease
to exist for the model *and* breaks the cache forward, with no error raised.

The round trip has two halves and both are asserted, because either half alone
is useless:

1. **CAPTURE** -- the pair survives the response into `Message.metadata`.
2. **RE-EMISSION** -- the pair is accepted back as *input* items, verbatim, in
   wire order, ahead of the `function_call` they preceded on the wire.

Capture is deliberately **unconditional** (not gated on `tool_search.mode`): a
response that contains hosted items must round-trip them whatever this
process's config says, or a resumed session silently loses its loaded set.
"""

import asyncio
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from amplifier_core.message_models import ChatRequest, Message

from amplifier_module_provider_openai import OpenAIProvider

# The wire-level contract, written as a literal on purpose: this test must be
# able to fail on a tree where the constant does not exist yet.
TOOL_SEARCH_ITEMS_KEY = "openai:tool_search_items"

# Measured wire shape (probe `bub`, re-confirmed by `cal` 2026-09-06): hosted
# execution sets `execution: "server"` and `call_id: null` on BOTH items, and
# the `function_call` that follows carries `namespace` alongside an
# **unqualified** `name`.
HOSTED_OUTPUT: list[dict[str, Any]] = [
    {
        "type": "tool_search_call",
        "execution": "server",
        "call_id": None,
        "status": "completed",
        "arguments": {"paths": ["files"]},
    },
    {
        "type": "tool_search_output",
        "execution": "server",
        "call_id": None,
        "tools": [{"type": "function", "name": "glob", "namespace": "files"}],
    },
    {
        "type": "function_call",
        "call_id": "call_Ejwv",
        "name": "glob",
        "namespace": "files",
        "arguments": "{}",
    },
]


class _DummyResponse:
    def __init__(self, output: list[Any]):
        self.output = output
        self.usage = SimpleNamespace(input_tokens=1, output_tokens=1)
        self.status = "completed"
        self.id = "resp_break3"


def _provider(**config_overrides: Any) -> OpenAIProvider:
    config = {"max_retries": 0, "use_streaming": False, **config_overrides}
    coordinator = MagicMock()
    coordinator.get_capability = MagicMock(return_value=None)
    coordinator.hooks = MagicMock()
    coordinator.hooks.emit = AsyncMock(return_value=None)
    client = SimpleNamespace(
        base_url="https://api.openai.com/v1",
        responses=SimpleNamespace(
            input_tokens=SimpleNamespace(
                count=AsyncMock(return_value=SimpleNamespace(input_tokens=1))
            ),
            create=AsyncMock(),
            stream=MagicMock(),
        ),
        close=AsyncMock(),
    )
    return OpenAIProvider(
        api_key="test-key", client=client, config=config, coordinator=coordinator
    )


def _complete_with_hosted_output(provider: OpenAIProvider):
    provider.client.responses.create = AsyncMock(
        return_value=_DummyResponse(output=[dict(i) for i in HOSTED_OUTPUT])
    )
    return asyncio.run(
        provider.complete(
            ChatRequest(messages=[Message(role="user", content="find files")])
        )
    )


# ---------------------------------------------------------------------------
# HALF 1 -- CAPTURE
# ---------------------------------------------------------------------------


def test_hosted_pair_survives_the_response_into_metadata():
    """FAIL-BEFORE: at main the pair is dropped and the key never appears."""
    response = _complete_with_hosted_output(_provider())
    captured = (response.metadata or {}).get(TOOL_SEARCH_ITEMS_KEY)
    assert captured is not None, (
        "hosted tool_search_call/tool_search_output pair was silently dropped "
        "(break 3). Per TS:854 the loaded tools now cease to exist next turn."
    )
    assert [item["type"] for item in captured] == [
        "tool_search_call",
        "tool_search_output",
    ]


def test_capture_preserves_the_null_call_id_and_the_loaded_set():
    """`call_id: null` is real hosted wire state (break 6) and must survive."""
    response = _complete_with_hosted_output(_provider())
    captured = (response.metadata or {})[TOOL_SEARCH_ITEMS_KEY]
    assert "call_id" in captured[0] and captured[0]["call_id"] is None
    assert captured[0]["execution"] == "server"
    # `tools[]` IS the loaded set (TS:854). Losing it loses the tools.
    assert captured[1]["tools"] == [
        {"type": "function", "name": "glob", "namespace": "files"}
    ]


def test_capture_is_unconditional_and_not_gated_on_config():
    """A resumed session must not lose its loaded set because a flag is off."""
    default_arm = _complete_with_hosted_output(_provider())
    enabled_arm = _complete_with_hosted_output(
        _provider(tool_search={"mode": "namespaced"})
    )
    assert (default_arm.metadata or {}).get(TOOL_SEARCH_ITEMS_KEY) == (
        enabled_arm.metadata or {}
    ).get(TOOL_SEARCH_ITEMS_KEY)
    assert (default_arm.metadata or {}).get(TOOL_SEARCH_ITEMS_KEY) is not None


# ---------------------------------------------------------------------------
# HALF 2 -- RE-EMISSION (accepted back as INPUT items)
# ---------------------------------------------------------------------------


def _history_from(captured: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"role": "user", "content": "find files"},
        {
            "role": "assistant",
            "content": [],
            "tool_calls": [{"id": "call_Ejwv", "name": "glob", "arguments": {}}],
            "metadata": {TOOL_SEARCH_ITEMS_KEY: captured},
        },
    ]


def test_hosted_pair_is_re_emitted_as_input_in_wire_order():
    """FAIL-BEFORE: at main nothing re-emits these, so the types are absent."""
    provider = _provider()
    captured = (_complete_with_hosted_output(provider).metadata or {}).get(
        TOOL_SEARCH_ITEMS_KEY
    ) or []
    converted = provider._convert_messages(_history_from(captured))
    types = [item.get("type") for item in converted if isinstance(item, dict)]
    assert "tool_search_call" in types, (
        "captured pair was never re-emitted into `input`; the model re-searches "
        "every turn (break 3)."
    )
    assert "tool_search_output" in types
    assert types.index("tool_search_call") < types.index("tool_search_output")
    assert types.index("tool_search_output") < types.index("function_call")


def test_re_emitted_items_are_byte_identical_to_what_was_captured():
    """Round-trip fidelity: the API accepts these back only as it emitted them."""
    provider = _provider()
    captured = (_complete_with_hosted_output(provider).metadata or {}).get(
        TOOL_SEARCH_ITEMS_KEY
    ) or []
    converted = provider._convert_messages(_history_from(captured))
    replayed = [
        item
        for item in converted
        if isinstance(item, dict)
        and item.get("type") in {"tool_search_call", "tool_search_output"}
    ]

    def canon(obj: Any) -> str:
        return json.dumps(
            obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        )

    # Non-emptiness is asserted FIRST and separately: `[] == []` would otherwise
    # make this test pass vacuously on a tree that captures nothing at all.
    assert len(captured) == 2, "nothing was captured to round-trip (break 3)"
    assert len(replayed) == 2, "captured pair was never re-emitted (break 3)"
    assert [canon(i) for i in replayed] == [canon(i) for i in captured]


def test_round_trip_survives_a_second_turn():
    """The loaded set is monotone (TS:854): it must persist past one turn."""
    provider = _provider()
    captured = (_complete_with_hosted_output(provider).metadata or {}).get(
        TOOL_SEARCH_ITEMS_KEY
    ) or []
    history = _history_from(captured)
    history.append({"role": "tool", "tool_call_id": "call_Ejwv", "content": "ok"})
    history.append({"role": "user", "content": "now edit it"})
    converted = provider._convert_messages(history)
    types = [item.get("type") for item in converted if isinstance(item, dict)]
    assert types.count("tool_search_call") == 1
    assert types.count("tool_search_output") == 1
