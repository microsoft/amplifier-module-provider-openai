"""Focused lowering tests for context-simple's canonical instruction layout."""

from __future__ import annotations

import asyncio
import copy
import importlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from amplifier_core.message_models import (
    ChatRequest,
    Message,
    ThinkingBlock,
    ToolCallBlock,
)

from amplifier_module_provider_openai import OpenAIProvider


class _Response:
    def __init__(self) -> None:
        self.output: list[Any] = []
        self.usage = SimpleNamespace(input_tokens=1, output_tokens=1)
        self.status = "completed"
        self.id = "resp_instruction_layout"


def _provider() -> OpenAIProvider:
    return OpenAIProvider(
        api_key="test-key", config={"max_retries": 0, "use_streaming": False}
    )


def _descriptor(
    placement: str,
    *,
    binding: str = "live",
    target: dict[str, Any] | None = None,
    deferred_origin: bool = False,
    disposition: str = "pending",
    authority: str | None = "authoritative",
) -> dict[str, Any]:
    descriptor: dict[str, Any] = {
        "version": 1,
        "source": "test-source",
        "key": f"{binding}-{placement}",
        "binding": binding,
        "placement": placement,
    }
    if authority is not None:
        descriptor["authority"] = authority
    if binding == "fixed":
        descriptor.update(
            entry_id=f"session:test-source:{binding}-{placement}",
            event_key=f"{binding}-{placement}",
            session_id="session",
            target=target,
            order=1,
            disposition=disposition,
        )
        if deferred_origin:
            descriptor["deferred_origin"] = True
    elif target is not None:
        descriptor["target"] = target
    return descriptor


def _instruction(
    content: str,
    placement: str,
    *,
    binding: str = "live",
    target: dict[str, Any] | None = None,
    deferred_origin: bool = False,
    disposition: str = "pending",
    authority: str | None = "authoritative",
) -> Message:
    return Message(
        role="system",
        content=content,
        metadata={
            "amplifier:instruction": _descriptor(
                placement,
                binding=binding,
                target=target,
                deferred_origin=deferred_origin,
                disposition=disposition,
                authority=authority,
            )
        },
    )


def _request(messages: list[Message]) -> ChatRequest:
    return ChatRequest(messages=messages)


def _params(provider: OpenAIProvider) -> dict[str, Any]:
    return cast(AsyncMock, provider.client.responses.create).call_args.kwargs


def test_provider_advertises_v1_instruction_layout() -> None:
    provider = _provider()
    assert provider.instruction_layout_version == 1
    assert provider.instruction_layout_authority_v1 is True


def test_authorityless_historical_descriptor_defaults_to_authoritative() -> None:
    provider = _provider()
    provider.client.responses.create = AsyncMock(return_value=_Response())
    request = _request(
        [
            _instruction("head", "head", authority=None),
            _instruction("positioned", "before_human", authority=None),
            Message(role="user", content="human"),
        ]
    )

    asyncio.run(provider.complete(request))

    assert _params(provider)["input"][0]["role"] == "developer"


def test_legacy_unmarked_system_and_developer_transforms_are_unchanged() -> None:
    provider = _provider()
    provider.client.responses.create = AsyncMock(return_value=_Response())
    request = _request(
        [
            Message(role="system", content="legacy global"),
            Message(role="developer", content="legacy developer"),
            Message(role="user", content="actual human"),
        ]
    )

    asyncio.run(provider.complete(request))

    params = _params(provider)
    assert params["instructions"] == "legacy global"
    assert params["input"] == [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "<context_file>\nlegacy developer\n</context_file>",
                }
            ],
        },
        {"role": "user", "content": [{"type": "input_text", "text": "actual human"}]},
    ]


def test_head_is_global_while_fixed_and_live_inline_records_keep_position() -> None:
    provider = _provider()
    provider.client.responses.create = AsyncMock(return_value=_Response())
    canonical = [
        Message(role="system", content="legacy head"),
        _instruction("resolved head", "head"),
        _instruction(
            "deferred fixed head",
            "head",
            binding="fixed",
            target={"kind": "conversation_head", "session_id": "session"},
            deferred_origin=True,
        ),
        _instruction(
            '<system-reminder source="old">fixed old input</system-reminder>',
            "before_human",
            binding="fixed",
            target={"input_id": "input-1", "message_id": "human-1", "origin": "human"},
        ),
        Message(role="user", content="old human"),
        Message(
            role="assistant",
            content=[
                ThinkingBlock(
                    thinking="plan",
                    content=[
                        {
                            "encrypted_content": "ENC",
                            "id": "rs_turn",
                            "summary": "plan",
                        }
                    ],
                ),
                ToolCallBlock(id="call_1", name="lookup", input={"q": "one"}),
            ],
        ),
        Message(role="tool", content="result", tool_call_id="call_1"),
        _instruction(
            '<system-reminder source="tail">tail machine content</system-reminder>',
            "tail",
            binding="fixed",
            target={"after_message_id": "tool-message"},
        ),
        _instruction("live new input", "before_human"),
        Message(role="user", content="new human"),
        Message(
            role="assistant",
            content=[
                ThinkingBlock(
                    thinking="current plan",
                    content=[
                        {
                            "encrypted_content": "ENC_CURRENT",
                            "id": "rs_current",
                            "summary": "current plan",
                        }
                    ],
                )
            ],
        ),
    ]
    request = _request(canonical)
    original = copy.deepcopy(request.model_dump())

    asyncio.run(provider.complete(request))

    params = _params(provider)
    assert (
        params["instructions"] == "legacy head\n\nresolved head\n\ndeferred fixed head"
    )
    input_items = params["input"]
    assert [item.get("type") or item.get("role") for item in input_items] == [
        "developer",
        "user",
        "function_call",
        "function_call_output",
        "developer",
        "developer",
        "user",
        "reasoning",
    ]
    assert input_items[0]["content"][0]["text"].startswith(
        '<system-reminder source="old">'
    )
    assert input_items[4] == {
        "role": "developer",
        "content": [
            {
                "type": "input_text",
                "text": '<system-reminder source="tail">tail machine content</system-reminder>',
            }
        ],
    }
    assert input_items[5]["content"][0]["text"] == "live new input"
    assert input_items[-1]["id"] == "rs_current"
    assert input_items[2]["call_id"] == input_items[3]["call_id"] == "call_1"
    assert request.model_dump() == original


def test_inline_machine_carrier_does_not_bound_reasoning_replay() -> None:
    provider = _provider()
    messages = [
        {"role": "user", "content": "human"},
        {
            "role": "assistant",
            "content": [
                SimpleNamespace(
                    type="thinking",
                    thinking="first",
                    content=[
                        {
                            "encrypted_content": "ENC_FIRST",
                            "id": "rs_first",
                            "summary": "first",
                        }
                    ],
                )
            ],
        },
        {
            "role": "_amplifier_inline_instruction",
            "content": "machine placement",
        },
        {
            "role": "assistant",
            "content": [
                SimpleNamespace(
                    type="thinking",
                    thinking="second",
                    content=[
                        {
                            "encrypted_content": "ENC_SECOND",
                            "id": "rs_second",
                            "summary": "second",
                        }
                    ],
                )
            ],
        },
    ]

    converted = provider._convert_messages(messages)

    assert [item["id"] for item in converted if item.get("type") == "reasoning"] == [
        "rs_first",
        "rs_second",
    ]
    assert converted[2] == {
        "role": "developer",
        "content": [{"type": "input_text", "text": "machine placement"}],
    }


def test_delivered_fixed_records_keep_their_resolved_placement() -> None:
    provider = _provider()
    provider.client.responses.create = AsyncMock(return_value=_Response())
    request = _request(
        [
            _instruction(
                "delivered head",
                "head",
                binding="fixed",
                target={"kind": "conversation_head", "session_id": "session"},
                disposition="delivered",
            ),
            _instruction(
                "delivered before H1",
                "before_human",
                binding="fixed",
                target={"input_id": "input-1", "message_id": "human-1", "origin": "human"},
                disposition="delivered",
            ),
            Message(role="user", content="H1"),
            _instruction(
                "delivered tail",
                "tail",
                binding="fixed",
                target={"after_message_id": "human-1"},
                disposition="delivered",
            ),
        ]
    )

    asyncio.run(provider.complete(request))

    params = _params(provider)
    assert params["instructions"] == "delivered head"
    assert params["input"] == [
        {
            "role": "developer",
            "content": [{"type": "input_text", "text": "delivered before H1"}],
        },
        {"role": "user", "content": [{"type": "input_text", "text": "H1"}]},
        {
            "role": "developer",
            "content": [{"type": "input_text", "text": "delivered tail"}],
        },
    ]


@pytest.mark.parametrize(
    ("descriptor", "error_type"),
    [
        (
            {
                "version": 2,
                "source": "s",
                "key": "k",
                "binding": "live",
                "placement": "head",
            },
            ValueError,
        ),
        (
            {
                "version": 1,
                "source": "s",
                "key": "k",
                "binding": "live",
                "placement": "head",
                "unknown": True,
            },
            ValueError,
        ),
        (
            {
                "version": 1,
                "source": "s",
                "key": "k",
                "binding": "live",
                "placement": "tail",
                "target": {"after_message_id": "m1", "unknown": True},
            },
            ValueError,
        ),
        (
            {
                "version": 1,
                "source": "s",
                "key": "k",
                "binding": "fixed",
                "placement": "head",
            },
            ValueError,
        ),
        (None, TypeError),
    ],
)
def test_malformed_or_unknown_marked_descriptor_fails_before_dispatch(
    descriptor: Any,
    error_type: type[Exception],
) -> None:
    provider = _provider()
    provider.client.responses.create = AsyncMock(return_value=_Response())
    request = _request(
        [
            Message(
                role="system",
                content="must not be silently moved",
                metadata={"amplifier:instruction": descriptor},
            ),
            Message(role="user", content="human"),
        ]
    )

    with pytest.raises(error_type, match="amplifier:instruction"):
        asyncio.run(provider.complete(request))

    cast(AsyncMock, provider.client.responses.create).assert_not_awaited()


def test_marked_non_system_message_fails_before_dispatch() -> None:
    provider = _provider()
    provider.client.responses.create = AsyncMock(return_value=_Response())
    request = _request(
        [
            Message(
                role="user",
                content="forged instruction carrier",
                metadata={"amplifier:instruction": _descriptor("head")},
            )
        ]
    )

    with pytest.raises(ValueError, match="canonical system"):
        asyncio.run(provider.complete(request))

    cast(AsyncMock, provider.client.responses.create).assert_not_awaited()


@pytest.mark.parametrize("disposition", ["retired", "anchor_pruned"])
def test_terminal_fixed_records_fail_before_dispatch(disposition: str) -> None:
    provider = _provider()
    provider.client.responses.create = AsyncMock(return_value=_Response())
    request = _request(
        [
            _instruction(
                "terminal record",
                "head",
                binding="fixed",
                target={"kind": "conversation_head", "session_id": "session"},
                disposition=disposition,
            )
        ]
    )

    with pytest.raises(ValueError, match="fixed amplifier:instruction descriptor has invalid fields"):
        asyncio.run(provider.complete(request))

    cast(AsyncMock, provider.client.responses.create).assert_not_awaited()


def test_synthetic_fixed_before_human_anchor_fails_before_dispatch() -> None:
    provider = _provider()
    provider.client.responses.create = AsyncMock(return_value=_Response())
    request = _request(
        [
            _instruction(
                "invalid fixed anchor",
                "before_human",
                binding="fixed",
                target={
                    "input_id": "synthetic-input",
                    "message_id": "synthetic-message",
                    "origin": "synthetic",
                },
            )
        ]
    )

    with pytest.raises(
        ValueError, match="fixed amplifier:instruction target does not match placement"
    ):
        asyncio.run(provider.complete(request))

    cast(AsyncMock, provider.client.responses.create).assert_not_awaited()


@pytest.mark.parametrize("authority", [True, "untrusted"])
def test_invalid_instruction_authority_fails_before_dispatch(authority: Any) -> None:
    provider = _provider()
    provider.client.responses.create = AsyncMock(return_value=_Response())
    request = _request([_instruction("bad", "head", authority=authority)])
    original = copy.deepcopy(request.model_dump())

    with pytest.raises(ValueError, match="invalid v1 fields"):
        asyncio.run(provider.complete(request))

    cast(AsyncMock, provider.client.responses.create).assert_not_awaited()
    assert request.model_dump() == original


def test_every_marked_descriptor_is_validated_before_tool_preflight() -> None:
    provider = _provider()
    provider.client.responses.create = AsyncMock(return_value=_Response())
    request = _request(
        [
            _instruction("valid", "head"),
            _instruction("invalid", "tail", authority="untrusted"),
            Message(
                role="assistant",
                content=[ToolCallBlock(id="missing", name="tool", input={})],
            ),
        ]
    )
    original = copy.deepcopy(request.model_dump())

    with pytest.raises(ValueError, match="invalid v1 fields"):
        asyncio.run(provider.complete(request))

    cast(AsyncMock, provider.client.responses.create).assert_not_awaited()
    assert request.model_dump() == original


def test_marked_v1_incomplete_tool_batch_fails_before_legacy_repair() -> None:
    provider = _provider()
    provider.client.responses.create = AsyncMock(return_value=_Response())
    request = _request(
        [
            _instruction("head", "head"),
            Message(
                role="assistant",
                content=[ToolCallBlock(id="missing", name="tool", input={})],
            ),
            Message(role="user", content="must not be preceded by a repair"),
        ]
    )
    original = copy.deepcopy(request.model_dump())

    with pytest.raises(ValueError, match="splits a tool-call/tool-result batch"):
        asyncio.run(provider.complete(request))

    cast(AsyncMock, provider.client.responses.create).assert_not_awaited()
    assert request.model_dump() == original


def test_v1_top_level_tool_name_alias_serializes_paired_result_unchanged() -> None:
    provider = _provider()
    provider.client.responses.create = AsyncMock(return_value=_Response())
    request = _request(
        [
            _instruction("head", "head"),
            Message(
                role="assistant",
                content="",
                tool_calls=[
                    {
                        "id": "call-1",
                        "tool": "lookup",
                        "arguments": {"query": "value"},
                    }
                ],
            ),
            Message(role="tool", content="result", tool_call_id="call-1"),
        ]
    )
    original = copy.deepcopy(request.model_dump())

    asyncio.run(provider.complete(request))

    payload = _params(provider)["input"]
    assert {"type": "function_call", "call_id": "call-1", "name": "lookup",
            "arguments": '{"query": "value"}'} in payload
    assert {
        "type": "function_call_output",
        "call_id": "call-1",
        "output": "result",
    } in payload
    assert request.model_dump() == original


@pytest.mark.parametrize(
    ("native_name", "native_input", "expected_type", "expected_output_type"),
    [
        (
            "apply_patch",
            {"type": "update_file", "path": "file.py", "diff": "@@"},
            "apply_patch_call",
            "apply_patch_call_output",
        ),
        (
            "computer",
            {"actions": [{"type": "click", "x": 1, "y": 2}]},
            "computer_call",
            "computer_call_output",
        ),
    ],
)
def test_v1_hybrid_tool_calls_emit_content_once_and_preserve_native_pairing(
    native_name: str,
    native_input: dict[str, Any],
    expected_type: str,
    expected_output_type: str,
) -> None:
    provider = _provider()
    provider.client.responses.create = AsyncMock(return_value=_Response())
    request = _request(
        [
            _instruction("head", "head"),
            Message(
                role="assistant",
                content=[
                    ToolCallBlock(id="ordinary", name="lookup", input={"q": "value"}),
                    ToolCallBlock(
                        id="native", name=native_name, input=native_input
                    ),
                ],
                tool_calls=[
                    {"id": "ordinary", "name": "lookup", "arguments": {"q": "value"}},
                    {"id": "native", "name": native_name, "arguments": native_input},
                ],
            ),
            Message(role="tool", content="ordinary result", tool_call_id="ordinary"),
            Message(role="tool", content="native result", tool_call_id="native"),
        ]
    )
    original = copy.deepcopy(request.model_dump())

    asyncio.run(provider.complete(request))

    payload = _params(provider)["input"]
    ordinary_calls = [
        item
        for item in payload
        if item.get("call_id") == "ordinary"
        and item.get("type") == "function_call"
    ]
    native_calls = [
        item
        for item in payload
        if item.get("call_id") == "native" and item.get("type") == expected_type
    ]
    ordinary_outputs = [
        item
        for item in payload
        if item.get("call_id") == "ordinary"
        and item.get("type") == "function_call_output"
    ]
    native_outputs = [
        item
        for item in payload
        if item.get("call_id") == "native"
        and item.get("type") == expected_output_type
    ]
    assert len(ordinary_calls) == len(native_calls) == 1
    assert len(ordinary_outputs) == len(native_outputs) == 1
    assert not [
        item
        for item in payload
        if item.get("call_id") == "native" and item.get("type") == "function_call"
    ]
    assert request.model_dump() == original


def test_private_completion_detects_marked_layout_with_legacy_messages() -> None:
    provider = _provider()
    provider.client.responses.create = AsyncMock(return_value=_Response())
    request = _request(
        [
            _instruction("marked head", "head"),
            Message(role="system", content="legacy head"),
            Message(role="user", content="human"),
        ]
    )
    original = copy.deepcopy(request.model_dump())

    asyncio.run(provider._complete_chat_request(request))

    params = _params(provider)
    assert params["instructions"] == "marked head\n\nlegacy head"
    assert request.model_dump() == original


class _ContextCoordinator:
    """Minimal coordinator for the opt-in real context-simple seam."""

    def __init__(self) -> None:
        self.capabilities: dict[str, object] = {}

    def register_capability(self, name: str, value: object) -> None:
        self.capabilities[name] = value

    def get_capability(self, name: str) -> object | None:
        return self.capabilities.get(name)

    async def process_hook_result(
        self, result: Any, *, event: str, hook_name: str
    ) -> Any:
        return result


@pytest.fixture
def context_simple_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Any, Any]:
    """Load a caller-selected context-simple checkout for the seam test."""
    configured = os.environ.get("CONTEXT_SIMPLE_SOURCE")
    if not configured:
        pytest.skip("set CONTEXT_SIMPLE_SOURCE to run the real context-simple seam")

    source = Path(configured).resolve()
    package_root = source / "amplifier_module_context_simple"
    if not (package_root / "__init__.py").is_file():
        pytest.fail(f"CONTEXT_SIMPLE_SOURCE is not a context-simple checkout: {source}")

    loaded = sys.modules.get("amplifier_module_context_simple")
    if loaded is not None:
        loaded_path = Path(loaded.__file__).resolve()
        if package_root not in loaded_path.parents:
            pytest.fail(
                "context-simple was imported from a different checkout before this seam test"
            )
    else:
        monkeypatch.syspath_prepend(str(source))

    context_module = importlib.import_module("amplifier_module_context_simple")
    instructions_module = importlib.import_module(
        "amplifier_module_context_simple.instructions"
    )
    assert package_root in Path(context_module.__file__).resolve().parents
    return context_module, instructions_module


def _context_input(context: Any, assembly: Any, input_id: str, content: str) -> Any:
    """Bind and admit a real human input, returning context's anchor."""

    async def add() -> Any:
        with assembly.input_scope("human", input_id):
            await context.add_message({"role": "user", "content": content})
        return context.messages[-1]["metadata"]["amplifier:input"]

    return add()


def _input_texts(payload: dict[str, Any]) -> list[str]:
    return [
        item["content"][0]["text"]
        for item in payload["input"]
        if isinstance(item.get("content"), list)
        and item["content"]
        and item["content"][0].get("type") == "input_text"
    ]


@pytest.mark.asyncio
async def test_real_context_retained_instruction_lifecycle_and_checkpoint(
    context_simple_modules: tuple[Any, Any],
) -> None:
    """Exercise context publish/accept/checkpoint with only a captured SDK payload."""
    context_module, instructions_module = context_simple_modules
    provider = _provider()
    provider.client.responses.create = AsyncMock(return_value=_Response())
    live = {"content": "live state request 1"}

    def attach_assembly(context: Any) -> Any:
        assembly = instructions_module.InstructionAssembly(
            context, _ContextCoordinator(), session_id="session"
        )
        context._instruction_assembly = assembly
        return assembly

    async def lower_and_accept(
        context: Any,
        assembly: Any,
        request_id: str,
        anchor: Any,
        response_text: str,
    ) -> dict[str, Any]:
        async with assembly.turn(f"turn-{request_id}", anchor), assembly.request(
            {
                "turn_id": f"turn-{request_id}",
                "request_id": request_id,
                "llm_step_id": f"step-{request_id}",
                "input_anchor": anchor,
                "tail_anchor": None,
                "completed_batches": [],
            },
            provider,
        ):
            view = await context.get_messages_for_request()
            await provider.complete(_request([Message(**message) for message in view]))
            payload = copy.deepcopy(_params(provider))
            await assembly.accept_response(
                request_id, {"role": "assistant", "content": response_text}
            )
            return payload

    context = context_module.SimpleContextManager(compaction_notice_enabled=False)
    assembly = attach_assembly(context)
    lease = assembly.register(
        "live-source",
        lambda _scope: [
            {
                "key": "current",
                "content": live["content"],
                "placement": "before_human",
            }
        ],
    )
    h1 = await _context_input(context, assembly, "h1", "human 1")
    head_id = lease.publish(
        "head", "fixed head", target={"kind": "conversation_head", "session_id": "session"}, retain_history=True
    )
    before_id = lease.publish("before", "fixed before H1", target=h1, retain_history=True)
    tail_id = lease.publish(
        "tail", "fixed tail H1", target={"after_message_id": h1["message_id"]}, retain_history=True
    )

    first_payload = await lower_and_accept(context, assembly, "request-1", h1, "reply 1")
    assert first_payload["instructions"] == "fixed head"
    assert _input_texts(first_payload) == [
        "live state request 1",
        "fixed before H1",
        "human 1",
        "fixed tail H1",
    ]
    assert any(
        message["role"] == "assistant" and message["content"] == "reply 1"
        for message in context.messages
    )

    live["content"] = "live state request 2"
    h2 = await _context_input(context, assembly, "h2", "human 2")
    second_payload = await lower_and_accept(
        context, assembly, "request-2", h2, "reply 2"
    )
    second_texts = _input_texts(second_payload)
    assert second_payload["instructions"] == "fixed head"
    assert second_texts.count("fixed before H1") == second_texts.count("fixed tail H1") == 1
    assert second_texts.index("fixed before H1") < second_texts.index("human 1")
    assert second_texts.index("fixed tail H1") > second_texts.index("human 1")
    assert second_texts.index("live state request 2") + 1 == second_texts.index("human 2")
    assert "live state request 1" not in second_texts

    checkpoint = await context.get_messages()
    retained = {
        message["metadata"]["amplifier:instruction"]["entry_id"]: message[
            "metadata"
        ]["amplifier:instruction"]
        for message in checkpoint
        if "amplifier:instruction" in message.get("metadata", {})
    }
    assert set(retained) == {head_id, before_id, tail_id}
    assert {entry["disposition"] for entry in retained.values()} == {"delivered"}
    assert retained[before_id]["target"] == h1
    retained_checkpoint = copy.deepcopy(retained)

    restored = context_module.SimpleContextManager(compaction_notice_enabled=False)
    restored_assembly = attach_assembly(restored)
    await restored.restore_host_checkpoint(checkpoint)
    live["content"] = "live state request 3"
    restored_assembly.register(
        "live-source",
        lambda _scope: [
            {
                "key": "current",
                "content": live["content"],
                "placement": "before_human",
            }
        ],
    )
    h3 = await _context_input(restored, restored_assembly, "h3", "human 3")
    third_payload = await lower_and_accept(
        restored, restored_assembly, "request-3", h3, "reply 3"
    )
    third_texts = _input_texts(third_payload)
    assert third_payload["instructions"] == "fixed head"
    assert third_texts.count("fixed before H1") == third_texts.count("fixed tail H1") == 1
    assert third_texts.index("live state request 3") + 1 == third_texts.index("human 3")
    assert "live state request 2" not in third_texts
    restored_retained = {
        message["metadata"]["amplifier:instruction"]["entry_id"]: message[
            "metadata"
        ]["amplifier:instruction"]
        for message in await restored.get_messages()
        if "amplifier:instruction" in message.get("metadata", {})
    }
    assert restored_retained == retained_checkpoint


def test_unchanged_head_prefix_is_stable_when_inline_suffix_changes() -> None:
    provider = _provider()
    provider.client.responses.create = AsyncMock(return_value=_Response())

    first = _request(
        [
            _instruction("stable head", "head"),
            Message(role="user", content="human"),
            _instruction("volatile one", "tail"),
        ]
    )
    asyncio.run(provider.complete(first))
    first_params = copy.deepcopy(_params(provider))

    provider.client.responses.create.reset_mock()
    second = _request(
        [
            _instruction("stable head", "head"),
            Message(role="user", content="human"),
            _instruction("volatile two", "tail"),
        ]
    )
    asyncio.run(provider.complete(second))
    second_params = _params(provider)

    assert (
        first_params["instructions"] == second_params["instructions"] == "stable head"
    )
    assert first_params["input"][:1] == second_params["input"][:1]
    assert first_params["input"][-1]["content"][0]["text"] == "volatile one"
    assert second_params["input"][-1]["content"][0]["text"] == "volatile two"
