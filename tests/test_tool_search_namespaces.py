"""Tests for `tool_loading: static | deferred_namespace` (OpenAI-only).

Three things these tests exist to catch, in priority order:

1. **Default byte-identity.** The default (`static`) request must be
   byte-for-byte what shipped before this feature existed. This is the one
   assertion that protects every existing deployment, and it is written as a
   literal expected payload plus a sha256 over the serialized `tools` array so
   a "harmless" refactor of the emission site cannot slide past it.
2. **The deferred request shape.** Namespaces sorted, members sorted,
   `defer_loading` on everything except the always-loaded set, `tool_search`
   last, native tools untouched, reserved names refused.
3. **Discovery appends, never mutates.** A tool registered mid-session must
   leave the `tools` block byte-identical and ride an `additional_tools` INPUT
   item instead -- the whole cache argument for this lever collapses otherwise.
"""

import asyncio
import hashlib
import json
import logging
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from amplifier_core.message_models import ChatRequest, Message, ToolSpec

from amplifier_module_provider_openai import OpenAIProvider
from amplifier_module_provider_openai._constants import METADATA_TOOL_SEARCH_ITEMS
from amplifier_module_provider_openai._tool_search import (
    _WARNED_UNLISTED,
    RESERVED_NAMESPACE_NAMES,
    ToolSearchConfigError,
    normalize_namespaces,
)

# A roster deliberately drawn from the real 14-tool set the census measured, so
# the namespace table under test is the one that would actually ship.
ROSTER: tuple[tuple[str, str], ...] = (
    ("read_file", "Read a file"),
    ("write_file", "Write a file"),
    ("edit_file", "Edit a file"),
    ("grep", "Search file contents"),
    ("glob", "Match file paths"),
    ("bash", "Run a shell command"),
    ("todo", "Track tasks"),
    ("web_search", "Search the web"),
    ("web_fetch", "Fetch a URL"),
    ("delegate", "Delegate to a sub-agent"),
    ("load_skill", "Load a skill"),
    ("recipes", "Run a recipe"),
    ("mode", "Switch mode"),
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
    return OpenAIProvider(
        api_key="[REDACTED:SECRET]", config=config, coordinator=coordinator
    )


def _request(tools: list[ToolSpec] | None = None) -> ChatRequest:
    return ChatRequest(
        messages=[Message(role="user", content="Hello")],
        tools=tools if tools is not None else _tool_specs(),
    )


class DummyResponse:
    def __init__(self, output: list[Any] | None = None):
        self.output = (
            output
            if output is not None
            else [
                SimpleNamespace(
                    type="message",
                    content=[SimpleNamespace(type="output_text", text="Hi")],
                )
            ]
        )
        self.usage = SimpleNamespace(input_tokens=1, output_tokens=1)
        self.status = "completed"
        self.id = "resp_test"


def _captured(provider: OpenAIProvider) -> dict[str, Any]:
    return cast(AsyncMock, provider.client.responses.create).call_args.kwargs


def _run(provider: OpenAIProvider, request: ChatRequest) -> dict[str, Any]:
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())
    asyncio.run(provider.complete(request))
    return _captured(provider)


def _canon(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


# ---------------------------------------------------------------------------
# 1. DEFAULT BYTE-IDENTITY -- the guardrail for every existing deployment
# ---------------------------------------------------------------------------


def _expected_static_tools() -> list[dict[str, Any]]:
    """The flat shape this provider emitted before deferred loading existed."""
    return [
        {
            "type": "function",
            "name": name,
            "description": desc,
            "parameters": {"type": "object", "properties": {}},
        }
        for name, desc in ROSTER
    ]


def test_default_config_emits_the_pre_existing_flat_tool_block():
    params = _run(_make_provider(), _request())
    assert params["tools"] == _expected_static_tools()


def test_default_config_tools_sha256_is_pinned():
    """Pin the serialized bytes, not just the structure.

    A refactor that reorders keys, coerces a None, or adds a "harmless" field
    would still satisfy a structural assertion while zeroing the prompt cache
    in production (probe `12v`: a 2.1% byte change dropped cached_tokens from
    18,387 to 0).
    """
    params = _run(_make_provider(), _request())
    digest = hashlib.sha256(_canon(params["tools"]).encode()).hexdigest()
    expected = hashlib.sha256(_canon(_expected_static_tools()).encode()).hexdigest()
    assert digest == expected


def test_default_config_sends_no_tool_search_surface_at_all():
    params = _run(_make_provider(), _request())
    blob = _canon(params)
    for token in ("tool_search", "defer_loading", "namespace", "additional_tools"):
        assert token not in blob, f"{token!r} leaked into a default-config request"


def test_explicit_static_matches_the_default():
    default = _run(_make_provider(), _request())["tools"]
    explicit = _run(_make_provider(tool_loading="static"), _request())["tools"]
    assert _canon(default) == _canon(explicit)


def test_unknown_mode_fails_loud_at_construction():
    with pytest.raises(ValueError, match="tool_loading"):
        _make_provider(tool_loading="deferred")


def test_flat_mode_is_not_offered():
    """`flat` is refused on purpose -- 1,270 tok vs 7,573 for the namespace form."""
    with pytest.raises(ValueError, match="tool_loading"):
        _make_provider(tool_loading="flat")


# ---------------------------------------------------------------------------
# 2. THE DEFERRED REQUEST SHAPE
# ---------------------------------------------------------------------------


def _deferred_tools() -> list[dict[str, Any]]:
    return _run(_make_provider(tool_loading="deferred_namespace"), _request())["tools"]


def test_deferred_emits_namespaces_then_tool_search_last():
    tools = _deferred_tools()
    assert tools[-1] == {"type": "tool_search"}, "tool_search must activate the block"
    kinds = [t["type"] for t in tools[:-1]]
    assert set(kinds) == {"namespace"}
    names = [t["name"] for t in tools[:-1]]
    assert names == sorted(names), "namespace order must not depend on dict iteration"
    assert names == ["delegation", "files", "internet", "knowledge", "shell"]


def test_deferred_never_uses_a_reserved_namespace_name():
    """`web`/`browser`/`python`/`computer` are HTTP 400 on every request."""
    tools = _deferred_tools()
    emitted = {t["name"] for t in tools if t.get("type") == "namespace"}
    assert not (emitted & RESERVED_NAMESPACE_NAMES)


def test_delegate_is_split_into_its_own_namespace():
    """Loading is per-NAMESPACE, and `delegate` alone is 3,602 tok.

    Grouping it with `load_skill` would drag the whole agent catalog in on 18%
    of sessions for free -- so the split is a measured cost decision, not
    taxonomy, and it is asserted rather than left to a config comment.
    """
    tools = _deferred_tools()
    by_ns = {
        t["name"]: [m["name"] for m in t["tools"]]
        for t in tools
        if t.get("type") == "namespace"
    }
    assert by_ns["delegation"] == ["delegate"]
    assert "delegate" not in by_ns["knowledge"]


def test_members_are_deferred_except_the_always_loaded_set():
    tools = _deferred_tools()
    deferred, loaded = set(), set()
    for t in tools:
        if t.get("type") != "namespace":
            continue
        for member in t["tools"]:
            (deferred if member.get("defer_loading") else loaded).add(member["name"])
    # bash 90.7% / todo 83.3% of sessions -- deferring them buys a search
    # round-trip in ~9 of 10 sessions to save 905 tokens.
    assert loaded == {"bash", "todo"}
    assert "read_file" in deferred and "delegate" in deferred


def test_members_are_sorted_within_a_namespace():
    tools = _deferred_tools()
    for t in tools:
        if t.get("type") == "namespace":
            names = [m["name"] for m in t["tools"]]
            assert names == sorted(names)


def test_member_definitions_are_unchanged_apart_from_defer_loading():
    tools = _deferred_tools()
    static_by_name = {t["name"]: t for t in _expected_static_tools()}
    for t in tools:
        if t.get("type") != "namespace":
            continue
        for member in t["tools"]:
            expected = dict(static_by_name[member["name"]])
            actual = {k: v for k, v in member.items() if k != "defer_loading"}
            assert actual == expected


def test_always_loaded_is_configurable():
    provider = _make_provider(
        tool_loading="deferred_namespace", tool_search_always_loaded=["bash"]
    )
    tools = _run(provider, _request())["tools"]
    loaded = {
        m["name"]
        for t in tools
        if t.get("type") == "namespace"
        for m in t["tools"]
        if not m.get("defer_loading")
    }
    assert loaded == {"bash"}


def test_unlisted_tool_falls_through_flat_and_warns(caplog):
    """Fail OPEN and loudly. A tool that silently vanished into a namespace is
    exactly the class of silent downgrade this provider warns about elsewhere."""
    caplog.set_level(logging.WARNING, logger="amplifier_module_provider_openai")
    _WARNED_UNLISTED.clear()  # the warning is once-per-set per process
    provider = _make_provider(tool_loading="deferred_namespace")
    tools = _run(provider, _request(_tool_specs(ROSTER + (("frobnicate", "?"),))))[
        "tools"
    ]
    flat = [t for t in tools if t.get("type") == "function"]
    assert [t["name"] for t in flat] == ["frobnicate"]
    assert "defer_loading" not in flat[0]
    assert any("frobnicate" in r.getMessage() for r in caplog.records)


def test_deferred_forces_tool_choice_auto():
    """`tool_choice` against a not-yet-discovered tool is unspecified (unprobed)."""
    provider = _make_provider(tool_loading="deferred_namespace")
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())
    asyncio.run(provider.complete(_request(), tool_choice="required"))
    assert _captured(provider)["tool_choice"] == "auto"


def test_static_does_not_force_tool_choice():
    provider = _make_provider()
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())
    asyncio.run(provider.complete(_request(), tool_choice="required"))
    assert _captured(provider)["tool_choice"] == "required"


# ---------------------------------------------------------------------------
# 3. DISCOVERY APPENDS, NEVER MUTATES
# ---------------------------------------------------------------------------


def test_mid_session_tool_appends_and_leaves_the_tools_block_byte_identical():
    provider = _make_provider(tool_loading="deferred_namespace")

    first = _run(provider, _request())
    assert not any(
        isinstance(i, dict) and i.get("type") == "additional_tools"
        for i in first["input"]
    )

    late = _tool_specs(ROSTER + (("terminal_inspector", "Drive a terminal app"),))
    second = _run(provider, _request(late))

    # The block is the cached head. It must not move by a single byte.
    assert _canon(second["tools"]) == _canon(first["tools"])
    assert "terminal_inspector" not in _canon(second["tools"])

    items = [
        i
        for i in second["input"]
        if isinstance(i, dict) and i.get("type") == "additional_tools"
    ]
    assert len(items) == 1
    item = items[0]
    assert item["role"] == "developer"  # TS:870-890
    assert [t["name"] for t in item["tools"]] == ["terminal_inspector"]
    # TS:893 -- the item must sit at the tail, never drift earlier.
    assert second["input"][-1] is item


def test_static_mode_never_emits_additional_tools():
    provider = _make_provider()
    _run(provider, _request())
    second = _run(
        provider, _request(_tool_specs(ROSTER + (("terminal_inspector", "x"),)))
    )
    assert not any(
        isinstance(i, dict) and i.get("type") == "additional_tools"
        for i in second["input"]
    )
    assert "terminal_inspector" in _canon(second["tools"])


def test_tool_leaving_the_roster_rebuilds_the_block_loudly(caplog):
    """The one deliberate cache rebuild: correctness over a cached prefix."""
    caplog.set_level(logging.WARNING, logger="amplifier_module_provider_openai")
    provider = _make_provider(tool_loading="deferred_namespace")
    _run(provider, _request())
    shrunk = tuple(t for t in ROSTER if t[0] != "web_search")
    tools = _run(provider, _request(_tool_specs(shrunk)))["tools"]
    assert "web_search" not in _canon(tools)
    assert any("web_search" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# 4. HOSTED tool_search ITEMS SURVIVE THE ROUND TRIP
# ---------------------------------------------------------------------------

_HOSTED_OUTPUT = [
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
        "tools": [
            {"type": "function", "name": "glob", "namespace": "files"},
        ],
    },
    {
        "type": "function_call",
        "call_id": "call_Ejwv",
        "name": "glob",
        "namespace": "files",
        "arguments": "{}",
    },
]


def test_hosted_items_are_captured_into_metadata():
    """Without this they hit `if block_type not in {"tool_call","function_call"}:
    continue` and vanish -- and per TS:854 every loaded tool then ceases to
    exist on the next request."""
    provider = _make_provider(tool_loading="deferred_namespace")
    provider.client.responses.create = AsyncMock(
        return_value=DummyResponse(output=list(_HOSTED_OUTPUT))
    )
    response = asyncio.run(provider.complete(_request()))
    captured = (response.metadata or {}).get(METADATA_TOOL_SEARCH_ITEMS)
    assert captured is not None
    assert [i["type"] for i in captured] == [
        "tool_search_call",
        "tool_search_output",
    ]
    assert captured[1]["tools"][0]["name"] == "glob"
    # `call_id: null` is real wire state (hosted execution) and must survive.
    assert "call_id" in captured[0] and captured[0]["call_id"] is None


def test_hosted_items_are_replayed_into_input_before_the_function_call():
    provider = _make_provider(tool_loading="deferred_namespace")
    history = [
        {"role": "user", "content": "find files"},
        {
            "role": "assistant",
            "content": [],
            "tool_calls": [{"id": "call_Ejwv", "name": "glob", "arguments": {}}],
            "metadata": {METADATA_TOOL_SEARCH_ITEMS: list(_HOSTED_OUTPUT[:2])},
        },
    ]
    converted = provider._convert_messages(history)
    types = [i.get("type") for i in converted if isinstance(i, dict)]
    assert "tool_search_call" in types and "tool_search_output" in types
    assert types.index("tool_search_call") < types.index("tool_search_output")
    assert types.index("tool_search_output") < types.index("function_call")


def test_namespaced_function_call_name_stays_unqualified():
    """`function_call` carries `name: "glob"` AND `namespace: "files"`; the loop
    dispatches on `name`, so the provider must never emit `files/glob`."""
    provider = _make_provider(tool_loading="deferred_namespace")
    provider.client.responses.create = AsyncMock(
        return_value=DummyResponse(output=list(_HOSTED_OUTPUT))
    )
    response = asyncio.run(provider.complete(_request()))
    assert response.tool_calls is not None
    assert [tc.name for tc in response.tool_calls] == ["glob"]


# ---------------------------------------------------------------------------
# 5. NAMESPACE TABLE VALIDATION
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("reserved", sorted(RESERVED_NAMESPACE_NAMES))
def test_reserved_namespace_names_are_refused_at_construction(reserved):
    with pytest.raises(ToolSearchConfigError, match="RESERVED"):
        normalize_namespaces([{"name": reserved, "members": ["web_fetch"]}])


def test_a_tool_cannot_live_in_two_namespaces():
    with pytest.raises(ToolSearchConfigError, match="two namespaces"):
        normalize_namespaces(
            [
                {"name": "a", "members": ["read_file"]},
                {"name": "b", "members": ["read_file"]},
            ]
        )


def test_custom_table_is_frozen_into_sorted_order():
    table = normalize_namespaces(
        [
            {"name": "zeta", "members": ["b", "a"]},
            {"name": "alpha", "members": ["d", "c"]},
        ]
    )
    assert [ns["name"] for ns in table] == ["alpha", "zeta"]
    assert table[0]["members"] == ("c", "d")


def test_bad_table_fails_at_provider_construction():
    with pytest.raises(ValueError):
        _make_provider(
            tool_loading="deferred_namespace",
            tool_namespaces=[{"name": "web", "members": ["web_fetch"]}],
        )


def test_the_unlisted_warning_fires_once_per_set_not_per_request(caplog):
    """A deployment mistake to fix once, not a per-request event."""
    caplog.set_level(logging.WARNING, logger="amplifier_module_provider_openai")
    _WARNED_UNLISTED.clear()
    provider = _make_provider(tool_loading="deferred_namespace")
    roster = ROSTER + (("frobnicate", "?"),)
    _run(provider, _request(_tool_specs(roster)))
    _run(provider, _request(_tool_specs(roster)))
    hits = [r for r in caplog.records if "frobnicate" in r.getMessage()]
    assert len(hits) == 1


def test_function_shaped_apply_patch_is_namespaced_not_unlisted(caplog):
    """Native apply_patch is a passthrough shape and never reaches grouping; the
    FUNCTION-shaped fallback must still land in a namespace rather than warning
    on every request."""
    caplog.set_level(logging.WARNING, logger="amplifier_module_provider_openai")
    _WARNED_UNLISTED.clear()
    provider = _make_provider(tool_loading="deferred_namespace")
    tools = _run(provider, _request(_tool_specs(ROSTER + (("apply_patch", "Patch"),))))[
        "tools"
    ]
    files = next(t for t in tools if t.get("name") == "files")
    assert "apply_patch" in [m["name"] for m in files["tools"]]
    assert not any("apply_patch" in r.getMessage() for r in caplog.records)


def test_native_tool_shapes_pass_through_untouched():
    """`{"type": "computer"}` / hosted tool dicts are vendor-owned: never
    namespaced, never deferred, never reordered relative to each other."""
    from amplifier_module_provider_openai._tool_search import (
        DEFAULT_ALWAYS_LOADED,
        DEFAULT_NAMESPACES,
        build_namespaced_tools,
    )

    flat = [
        {"type": "computer"},
        {"type": "apply_patch"},
        {
            "type": "function",
            "name": "read_file",
            "description": "Read",
            "parameters": {},
        },
    ]
    out = build_namespaced_tools(
        flat, DEFAULT_NAMESPACES, frozenset(DEFAULT_ALWAYS_LOADED)
    )
    assert out[0] == {"type": "computer"}
    assert out[1] == {"type": "apply_patch"}
    assert out[-1] == {"type": "tool_search"}
