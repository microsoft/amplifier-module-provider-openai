"""Tests for OpenAI's native `computer` (computer-use) tool support.

Mirrors tests/test_apply_patch_integration.py -- apply_patch is the exact
precedent this follows: a native tool type needed (1) a declaration that
survives to the wire un-flattened, (2) call_id tracking so results route to
the right envelope, and (3) a native result envelope instead of the generic
stringified function_call_output.

`computer` differs from apply_patch in two verified ways (see _constants.py
and __init__.py comments for the live-API evidence):
- It must be declared completely bare: {"type": "computer"} with NO other
  fields. apply_patch still carries name/description implicitly; computer
  rejects even display_width/display_height/environment/display_width_px.
- Its result envelope carries an image (computer_screenshot), not a string.

Fixtures under tests/fixtures/computer_use/ are copies of the captures at
amplifier-bundle-computer-use/tests/fixtures/captures/openai-turn0.json and
openai-turn1.json (real traffic captured against gpt-5.6).
"""

# pyright: reportAttributeAccessIssue=false

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from amplifier_module_provider_openai import OpenAIProvider
from amplifier_module_provider_openai._constants import NATIVE_TOOL_TYPES

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "computer_use"


def _load_fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES_DIR / name).read_text(encoding="utf-8"))


# --- Fixtures / helpers ---


def _make_provider(**overrides: Any) -> OpenAIProvider:
    """Create a minimal provider instance for unit testing."""
    coordinator = MagicMock()
    coordinator.get_capability = MagicMock(return_value=None)
    coordinator.hooks = MagicMock()
    coordinator.hooks.emit = AsyncMock()
    config = {**overrides}
    provider = OpenAIProvider(
        api_key="test-key", config=config, coordinator=coordinator
    )
    return provider


def _make_computer_tool_spec() -> MagicMock:
    """Mock a ToolSpec carrying the `computer` native type as an extra attr.

    Mirrors what amplifier-module-loop-streaming's native_tool_spec mechanism
    produces: a ToolSpec (extra="allow") whose `type` attribute rides along
    from the tool's native_tool_spec = {"type": "computer"} declaration.
    """
    spec = MagicMock()
    spec.name = "computer"
    spec.description = "Control the computer"
    spec.parameters = {"type": "object", "properties": {}}
    spec.type = "computer"
    return spec


# --- Test NATIVE_TOOL_TYPES includes computer ---


class TestNativeToolTypes:
    def test_computer_in_native_tool_types(self) -> None:
        assert "computer" in NATIVE_TOOL_TYPES


# --- Test _convert_tools_from_request: declaration is bare ---


class TestConvertToolsFromRequestComputer:
    def test_native_computer_sends_bare_type_only(self) -> None:
        """The computer tool must be declared as exactly {"type": "computer"} --
        no name, description, or parameters leaked onto the wire."""
        provider = _make_provider()
        tool_spec = _make_computer_tool_spec()

        result = provider._convert_tools_from_request([tool_spec])

        computer_tools = [t for t in result if t.get("type") == "computer"]
        assert len(computer_tools) == 1
        assert computer_tools[0] == {"type": "computer"}, (
            "computer declaration must be completely bare -- live API evidence "
            "shows display_width/display_height/environment/display_width_px "
            "each independently trigger 400 Unknown parameter"
        )

    def test_computer_tool_not_flattened_to_function(self) -> None:
        """Regression guard for blocker #2: ToolSpec objects must not fall
        through to the generic hasattr(tool, "name") function-tool branch."""
        provider = _make_provider()
        tool_spec = _make_computer_tool_spec()

        result = provider._convert_tools_from_request([tool_spec])

        function_tools = [t for t in result if t.get("type") == "function"]
        assert len(function_tools) == 0
        assert not any(t.get("name") == "computer" for t in result)

    def test_other_tools_unaffected(self) -> None:
        """Non-computer tools are converted normally (no interaction with the
        new branch)."""
        provider = _make_provider()
        other_tool = MagicMock()
        other_tool.name = "read_file"
        other_tool.description = "Read files"
        other_tool.parameters = {"type": "object", "properties": {}}
        other_tool.type = None  # explicit: no native type extra

        result = provider._convert_tools_from_request([other_tool])
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["name"] == "read_file"

    def test_dict_form_computer_tool_still_passes_through(self) -> None:
        """Belt-and-braces: a raw dict {"type": "computer"} (the pre-existing
        passthrough path for dicts) must also still work now that "computer"
        is in NATIVE_TOOL_TYPES."""
        provider = _make_provider()
        result = provider._convert_tools_from_request([{"type": "computer"}])
        assert result == [{"type": "computer"}]

    def test_apply_patch_still_uses_name_based_gate_unaffected(self) -> None:
        """Regression guard: the new `type`-attribute branch must not
        interfere with apply_patch's existing name+capability-gated branch
        (apply_patch ToolSpecs, as built today, carry no `type` extra)."""
        provider = _make_provider()
        provider._apply_patch_native = True
        tool_spec = MagicMock()
        tool_spec.name = "apply_patch"
        tool_spec.description = "Apply V4A patches"
        tool_spec.parameters = {"type": "object", "properties": {}}
        tool_spec.type = MagicMock()  # MagicMock auto-attr, not == "computer"

        result = provider._convert_tools_from_request([tool_spec])
        native_tools = [t for t in result if t.get("type") == "apply_patch"]
        assert len(native_tools) == 1


# --- Prove the declaration reaches the wire un-flattened (not just a unit test) ---


class TestWireBodyComputerDeclaration:
    """Asserts on the actual serialized request body passed to the OpenAI
    SDK client, not just the internal conversion helper -- a passing unit
    test sitting on a wrong wire is exactly the failure this project keeps
    hitting."""

    @pytest.mark.asyncio
    async def test_computer_tool_reaches_client_create_call_bare(self) -> None:
        """End-to-end through provider.complete(): capture the exact kwargs
        passed to the OpenAI client for a computer-tool request and assert
        on the serialized "tools" wire value directly.

        A request declaring the `computer` tool routes through
        client.responses.with_raw_response.create(**params) (see
        OpenAIProvider._create_response -- the raw-JSON fallback for
        OpenAI's pending_safety_checks SDK defect), not the plain
        client.responses.create(**params) a non-computer request uses.

        Response-side post-processing (cost computation, rate-limit header
        parsing, ChatResponse construction) is exercised separately and
        thoroughly by TestConvertResponseComputerCall below; it is stubbed
        out here so this test isolates exactly one thing: does the native
        declaration survive, un-flattened, all the way to the real client
        call boundary.
        """
        from amplifier_core.message_models import ChatRequest, Message, ToolSpec

        # use_streaming=False routes through the blocking client.responses.create()
        # path instead of the client.responses.stream() async-context-manager
        # path (the default) -- this test only needs to assert on the outbound
        # request body, not exercise the streaming transport.
        provider = _make_provider(use_streaming=False)
        # A real ToolSpec (extra="allow"), not a MagicMock -- this is what
        # ChatRequest validation actually requires, and it's also the real
        # shape amplifier-module-loop-streaming's native_tool_spec mechanism
        # produces: a `type="computer"` extra attribute riding alongside
        # name/description/parameters.
        tool_spec = ToolSpec(
            name="computer",
            description="Control the computer",
            parameters={"type": "object", "properties": {}},
        )
        tool_spec.type = "computer"  # extra="allow" -- rides along as an attribute

        captured_params: dict[str, Any] = {}

        class _CapturedAndAborted(Exception):
            """Sentinel: raised the instant params are captured, to abort
            before any response post-processing (cost computation, retries,
            continuation handling) runs -- none of that is what this test
            is proving."""

        async def fake_create(**kwargs: Any) -> Any:
            captured_params.update(kwargs)
            raise _CapturedAndAborted()

        fake_client = MagicMock()
        # Computer-tool requests go through with_raw_response, not create()
        # directly -- see OpenAIProvider._create_response.
        fake_client.responses.with_raw_response.create = AsyncMock(
            side_effect=fake_create
        )
        provider._client = fake_client

        request = ChatRequest(
            model="gpt-5.6-sol",
            messages=[Message(role="user", content="take a screenshot")],
            tools=[tool_spec],
        )

        with pytest.raises(Exception) as exc_info:
            await provider.complete(request)
        # The provider wraps arbitrary exceptions from the API call in its
        # own error type; walk the cause chain to confirm it's really our
        # sentinel and not some unrelated failure masking a real bug.
        cause: BaseException | None = exc_info.value
        seen: list[type] = []
        while cause is not None:
            seen.append(type(cause))
            if isinstance(cause, _CapturedAndAborted):
                break
            cause = cause.__cause__
        assert cause is not None and isinstance(cause, _CapturedAndAborted), (
            f"expected the abort sentinel somewhere in the exception chain, "
            f"got chain of types: {seen!r}"
        )

        assert "tools" in captured_params, (
            "tools must reach the actual client.responses.create(**params) call"
        )
        wire_tools = captured_params["tools"]
        assert wire_tools == [{"type": "computer"}], (
            f"expected the bare native declaration on the wire, got: {wire_tools!r}"
        )


# --- Test _convert_to_chat_response (computer_call parsing, from real fixtures) ---


class TestConvertResponseComputerCall:
    def test_parses_computer_call_screenshot_action_dict_form(self) -> None:
        """openai-turn0.json: a computer_call whose actions batch is a single
        {"type": "screenshot"} action."""
        provider = _make_provider()
        provider._native_call_ids = set()
        provider._native_call_types = {}

        block = _load_fixture("openai-turn0.json")

        mock_response = _make_mock_response(output=[block])
        chat_response = provider._convert_to_chat_response(mock_response)

        assert chat_response.tool_calls is not None
        assert len(chat_response.tool_calls) == 1
        tc = chat_response.tool_calls[0]
        assert tc.name == "computer"
        assert tc.id == "call_1JvtBj79pTBZDKWno500zMsn"
        assert tc.arguments == {"actions": [{"type": "screenshot"}]}

        assert "call_1JvtBj79pTBZDKWno500zMsn" in provider._native_call_ids
        assert (
            provider._native_call_types["call_1JvtBj79pTBZDKWno500zMsn"] == "computer"
        )

    def test_parses_computer_call_move_action_dict_form(self) -> None:
        """openai-turn1.json: real captured batched action, {"type": "move",
        "keys": null, "x": 426, "y": 87}."""
        provider = _make_provider()
        provider._native_call_ids = set()
        provider._native_call_types = {}

        block = _load_fixture("openai-turn1.json")
        mock_response = _make_mock_response(output=[block])
        chat_response = provider._convert_to_chat_response(mock_response)

        assert chat_response.tool_calls is not None
        tc = chat_response.tool_calls[0]
        assert tc.name == "computer"
        assert tc.id == "call_1warEZSWaXDHw1TSSeigeHxv"
        assert tc.arguments == {
            "actions": [{"type": "move", "keys": None, "x": 426, "y": 87}]
        }

    def test_parses_computer_call_object_form(self) -> None:
        """Same as dict form, but via SDK-object-shaped blocks (MagicMock
        standing in for the pydantic response object), exercising the
        `hasattr(block, "type")` branch instead of the dict branch."""
        provider = _make_provider()
        provider._native_call_ids = set()
        provider._native_call_types = {}

        action = MagicMock()
        action.type = "click"
        action.x = 10
        action.y = 20
        action.button = "left"
        action.model_dump = MagicMock(
            return_value={"type": "click", "x": 10, "y": 20, "button": "left"}
        )

        block = MagicMock()
        block.type = "computer_call"
        block.call_id = "call_obj_123"
        block.actions = [action]
        block.action = None

        mock_response = _make_mock_response(output=[block])
        chat_response = provider._convert_to_chat_response(mock_response)

        assert chat_response.tool_calls is not None
        tc = chat_response.tool_calls[0]
        assert tc.name == "computer"
        assert tc.id == "call_obj_123"
        assert tc.arguments == {
            "actions": [{"type": "click", "x": 10, "y": 20, "button": "left"}]
        }
        assert "call_obj_123" in provider._native_call_ids
        assert provider._native_call_types["call_obj_123"] == "computer"

    def test_apply_patch_call_parsing_unaffected(self) -> None:
        """Regression guard: apply_patch_call parsing (existing behavior)
        must be untouched by the new computer_call branch."""
        provider = _make_provider()
        provider._native_call_ids = set()
        provider._native_call_types = {}

        block = {
            "type": "apply_patch_call",
            "call_id": "call_ap_1",
            "operation": {
                "type": "update_file",
                "path": "src/main.py",
                "diff": "@@ -1 +1 @@\n-old\n+new",
            },
        }
        mock_response = _make_mock_response(output=[block])
        chat_response = provider._convert_to_chat_response(mock_response)

        assert chat_response.tool_calls is not None
        tc = chat_response.tool_calls[0]
        assert tc.name == "apply_patch"
        assert "call_ap_1" in provider._native_call_ids
        # apply_patch call_ids must NOT be classified as "computer"
        assert provider._native_call_types.get("call_ap_1") != "computer"


def _make_mock_response(output: list[Any]) -> MagicMock:
    mock_response = MagicMock()
    mock_response.model = "gpt-5.6-sol"
    mock_response.output = output
    mock_response.usage = MagicMock()
    mock_response.usage.input_tokens = 10
    mock_response.usage.output_tokens = 5
    mock_response.usage.total_tokens = 15
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 5
    mock_response.usage.output_tokens_details = None
    mock_response.usage.input_tokens_details = None
    mock_response.id = "resp_computer_test"
    mock_response.status = "completed"
    mock_response.incomplete_details = None
    mock_response.finish_reason = None
    mock_response.output_text = None
    return mock_response


# --- Test _convert_messages (computer_call_output with image envelope) ---


class TestConvertMessagesComputerCallOutput:
    def test_native_computer_result_uses_computer_call_output_with_image(self) -> None:
        """Tool results for native computer calls must use computer_call_output
        with an image envelope -- not a stringified blob."""
        provider = _make_provider()
        provider._native_call_ids = {"call_abc123"}
        provider._native_call_types = {"call_abc123": "computer"}

        png_b64 = base64.b64encode(b"fake-png-bytes").decode("ascii")
        messages = [
            {
                "role": "tool",
                "tool_call_id": "call_abc123",
                "content": png_b64,
                "tool_name": "computer",
            }
        ]

        result = provider._convert_messages(messages)

        outputs = [
            m
            for m in result
            if isinstance(m, dict) and m.get("type") == "computer_call_output"
        ]
        assert len(outputs) == 1
        item = outputs[0]
        assert item["call_id"] == "call_abc123"
        assert item["output"]["type"] == "computer_screenshot"
        assert item["output"]["image_url"] == f"data:image/png;base64,{png_b64}"
        assert item["output"]["detail"] == "original"

        # Must NOT be stringified into a generic function_call_output.
        assert not any(
            isinstance(m, dict) and m.get("type") == "function_call_output"
            for m in result
        )

    def test_computer_result_accepts_image_block_list_content(self) -> None:
        """Also accept the ImageBlock-shaped content list, mirroring the
        existing role=="user" image conversion."""
        provider = _make_provider()
        provider._native_call_ids = {"call_img_1"}
        provider._native_call_types = {"call_img_1": "computer"}

        png_b64 = base64.b64encode(b"another-fake-png").decode("ascii")
        messages = [
            {
                "role": "tool",
                "tool_call_id": "call_img_1",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": png_b64,
                        },
                    }
                ],
                "tool_name": "computer",
            }
        ]

        result = provider._convert_messages(messages)
        outputs = [
            m
            for m in result
            if isinstance(m, dict) and m.get("type") == "computer_call_output"
        ]
        assert len(outputs) == 1
        assert outputs[0]["output"]["image_url"] == f"data:image/png;base64,{png_b64}"

    def test_computer_result_without_image_fails_loud(self) -> None:
        """Fail-loud requirement: a computer_call tool result with no image
        data must raise, not silently degrade into a text/empty envelope."""
        provider = _make_provider()
        provider._native_call_ids = {"call_no_image"}
        provider._native_call_types = {"call_no_image": "computer"}

        messages = [
            {
                "role": "tool",
                "tool_call_id": "call_no_image",
                "content": "",
                "tool_name": "computer",
            }
        ]

        with pytest.raises(ValueError, match="did not contain image data"):
            provider._convert_messages(messages)

    def test_apply_patch_result_unaffected_by_computer_branch(self) -> None:
        """Regression guard: existing apply_patch tool-result handling (no
        _native_call_types entry set) must be completely untouched."""
        provider = _make_provider()
        provider._native_call_ids = {"call_abc123"}
        # No _native_call_types entry -- default/legacy behavior expected.

        messages = [
            {
                "role": "tool",
                "tool_call_id": "call_abc123",
                "content": "M src/main.py",
                "tool_name": "apply_patch",
            }
        ]
        result = provider._convert_messages(messages)
        patch_outputs = [
            m
            for m in result
            if isinstance(m, dict) and m.get("type") == "apply_patch_call_output"
        ]
        assert len(patch_outputs) == 1
        assert patch_outputs[0]["call_id"] == "call_abc123"
        assert patch_outputs[0]["output"] == "M src/main.py"


# --- Test historical replay of computer_call (multi-turn correctness) ---


class TestComputerCallHistoryReplay:
    def test_computer_call_replayed_as_computer_call_not_function_call(self) -> None:
        """A stored assistant tool_call block for `computer` (dict form) must
        be replayed as a native computer_call item, not flattened to
        function_call -- otherwise turn 2+ of any computer-use conversation
        silently degrades."""
        provider = _make_provider()
        provider._native_call_ids = set()
        provider._native_call_types = {}

        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_call",
                        "id": "call_native_computer_1",
                        "name": "computer",
                        "input": {"actions": [{"type": "screenshot"}]},
                    }
                ],
            }
        ]

        result = provider._convert_messages(messages)

        function_call_items = [
            m
            for m in result
            if isinstance(m, dict) and m.get("type") == "function_call"
        ]
        assert len(function_call_items) == 0

        computer_call_items = [
            m
            for m in result
            if isinstance(m, dict) and m.get("type") == "computer_call"
        ]
        assert len(computer_call_items) == 1
        item = computer_call_items[0]
        assert item["call_id"] == "call_native_computer_1"
        assert item["actions"] == [{"type": "screenshot"}]
        assert "call_native_computer_1" in provider._native_call_ids
        assert provider._native_call_types["call_native_computer_1"] == "computer"

    def test_computer_call_output_type_correct_for_historical_native_call(self) -> None:
        """End-to-end multi-turn correctness: after replaying a historical
        computer_call, its paired tool-result message must use
        computer_call_output (with an image envelope), not
        function_call_output."""
        provider = _make_provider()
        provider._native_call_ids = set()
        provider._native_call_types = {}

        png_b64 = base64.b64encode(b"replay-png").decode("ascii")
        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_call",
                        "id": "call_native_computer_2",
                        "name": "computer",
                        "input": {"actions": [{"type": "screenshot"}]},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_native_computer_2",
                "content": png_b64,
                "tool_name": "computer",
            },
        ]

        result = provider._convert_messages(messages)

        func_outputs = [
            m
            for m in result
            if isinstance(m, dict) and m.get("type") == "function_call_output"
        ]
        assert len(func_outputs) == 0

        outputs = [
            m
            for m in result
            if isinstance(m, dict) and m.get("type") == "computer_call_output"
        ]
        assert len(outputs) == 1
        assert outputs[0]["call_id"] == "call_native_computer_2"
        assert outputs[0]["output"]["image_url"] == f"data:image/png;base64,{png_b64}"

    def test_computer_call_replayed_as_computer_call_object_form(self) -> None:
        """Same detection, but via the ContentBlock-object branch (a real
        ToolCallBlock instance rather than a stored dict)."""
        from amplifier_core.message_models import ToolCallBlock

        provider = _make_provider()
        provider._native_call_ids = set()
        provider._native_call_types = {}

        block = ToolCallBlock(
            id="call_native_computer_obj",
            name="computer",
            input={"actions": [{"type": "wait"}]},
        )
        messages = [{"role": "assistant", "content": [block]}]

        result = provider._convert_messages(messages)

        computer_call_items = [
            m
            for m in result
            if isinstance(m, dict) and m.get("type") == "computer_call"
        ]
        assert len(computer_call_items) == 1
        assert computer_call_items[0]["call_id"] == "call_native_computer_obj"
        assert computer_call_items[0]["actions"] == [{"type": "wait"}]
        assert "call_native_computer_obj" in provider._native_call_ids
        assert provider._native_call_types["call_native_computer_obj"] == "computer"

    def test_function_mode_computer_named_tool_stays_function_call(self) -> None:
        """A tool_call literally named "computer" but WITHOUT the native
        {"actions": [...]} input shape must NOT be promoted -- guards against
        over-eager detection by name alone."""
        provider = _make_provider()
        provider._native_call_ids = set()
        provider._native_call_types = {}

        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_call",
                        "id": "call_func_computer",
                        "name": "computer",
                        "input": {"some_other_shape": True},
                    }
                ],
            }
        ]
        result = provider._convert_messages(messages)

        computer_call_items = [
            m
            for m in result
            if isinstance(m, dict) and m.get("type") == "computer_call"
        ]
        assert len(computer_call_items) == 0
        function_call_items = [
            m
            for m in result
            if isinstance(m, dict) and m.get("type") == "function_call"
        ]
        assert len(function_call_items) == 1
        assert function_call_items[0]["name"] == "computer"
        assert "call_func_computer" not in provider._native_call_ids


# --- Test _create_response: the raw-JSON fallback for OpenAI's SDK defect ---
#
# Live GA `computer_call` responses omit `pending_safety_checks` entirely.
# Every openai-python release checked (2.8.1 installed, 2.52.0 latest)
# declares it a required field with no default on `ResponseComputerToolCall`,
# so `client.responses.create()`'s automatic parsing raises a real
# `pydantic.ValidationError` against a real captured response -- independent
# of anything this provider does. These tests exercise the SDK's *actual*
# installed `openai.types.responses.Response` model (no mocking away the
# boundary that broke) to prove the fallback works against the real defect,
# not a stand-in for it.


def _envelope_for(computer_call_block: dict[str, Any]) -> dict[str, Any]:
    """Wrap a captured `computer_call` fixture block in a full Responses-API
    envelope, matching the shape a live `responses.create()` call returns."""
    return {
        "id": "resp_test_computer_use",
        "created_at": 1234567890.0,
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "metadata": {},
        "model": "gpt-5.6-sol",
        "object": "response",
        "output": [computer_call_block],
        "parallel_tool_calls": True,
        "temperature": 1.0,
        "tool_choice": "auto",
        "tools": [{"type": "computer"}],
        "top_p": 1.0,
        "status": "completed",
        "usage": {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
    }


class _FakeRawAPIResponse:
    """Stands in for the SDK's `AsyncAPIResponse` (what
    `client.responses.with_raw_response.create()` returns), but calls the
    REAL installed `openai.types.responses.Response` model to parse --
    exercising the actual SDK boundary that the pending_safety_checks
    defect breaks, not a mock of it."""

    def __init__(self, body: dict[str, Any]) -> None:
        self._body = body

    async def parse(self) -> Any:
        from openai.types.responses import Response

        return Response.model_validate(self._body)

    async def json(self) -> Any:
        return self._body


class TestCreateResponseComputerUseFallback:
    def test_declares_computer_tool_detection(self) -> None:
        from amplifier_module_provider_openai import _params_declare_computer_tool

        assert _params_declare_computer_tool({"tools": [{"type": "computer"}]})
        assert not _params_declare_computer_tool({"tools": [{"type": "function"}]})
        assert not _params_declare_computer_tool({"tools": []})
        assert not _params_declare_computer_tool({})

    @pytest.mark.asyncio
    async def test_real_sdk_typed_model_rejects_live_computer_call_payload(
        self,
    ) -> None:
        """Ground-truth regression guard: proves the SDK defect is real, using
        the real installed openai SDK types (not a mock). If this test ever
        starts failing because parsing *succeeds*, the SDK has fixed the
        upstream bug and the fallback in `_create_response` may be safe to
        remove."""
        from openai.types.responses import Response

        block = _load_fixture("openai-turn1.json")
        envelope = _envelope_for(block)

        with pytest.raises(ValidationError) as exc_info:
            Response.model_validate(envelope)

        errors = exc_info.value.errors()
        assert any(
            "pending_safety_checks" in str(err["loc"]) and err["type"] == "missing"
            for err in errors
        ), (
            "expected the known pending_safety_checks-required-but-missing "
            f"defect; got errors: {errors!r}"
        )

    @pytest.mark.asyncio
    async def test_falls_back_to_raw_json_when_typed_parse_fails(self) -> None:
        """The actual fix: `_create_response` recovers from the real SDK
        ValidationError by reading the raw JSON body, and the result is
        consumable by `_convert_to_chat_response` exactly like a normal
        parsed response would be."""
        provider = _make_provider()
        provider._native_call_ids = set()
        provider._native_call_types = {}

        block = _load_fixture("openai-turn1.json")
        envelope = _envelope_for(block)

        fake_client = MagicMock()
        fake_client.responses.with_raw_response.create = AsyncMock(
            return_value=_FakeRawAPIResponse(envelope)
        )
        provider._client = fake_client

        params = {"model": "gpt-5.6-sol", "tools": [{"type": "computer"}]}
        response = await provider._create_response(params)

        # Not the typed SDK model -- the raw-JSON wrapper.
        assert type(response).__name__ == "_RawResponseObject"

        chat_response = provider._convert_to_chat_response(response)
        assert chat_response.tool_calls is not None
        assert len(chat_response.tool_calls) == 1
        call = chat_response.tool_calls[0]
        assert call.name == "computer"
        assert call.arguments == {
            "actions": [{"type": "move", "keys": None, "x": 426, "y": 87}]
        }
        assert chat_response.usage is not None
        assert chat_response.usage.input_tokens == 100
        assert chat_response.usage.output_tokens == 20

    @pytest.mark.asyncio
    async def test_non_computer_request_uses_plain_create_unchanged(self) -> None:
        """Scope guard: a request that does NOT declare the `computer` tool
        must never touch `with_raw_response` -- only the computer-use path
        changes behavior."""
        provider = _make_provider()

        sentinel = MagicMock(name="typed_response")
        fake_client = MagicMock()
        fake_client.responses.create = AsyncMock(return_value=sentinel)
        fake_client.responses.with_raw_response.create = AsyncMock(
            side_effect=AssertionError(
                "with_raw_response.create must not be called for a "
                "non-computer-use request"
            )
        )
        provider._client = fake_client

        params = {"model": "gpt-5.6-sol", "tools": [{"type": "function"}]}
        result = await provider._create_response(params)

        assert result is sentinel
        fake_client.responses.create.assert_awaited_once_with(**params)
        fake_client.responses.with_raw_response.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_fails_loud_when_raw_body_missing_output_field(self) -> None:
        """Fail-loud requirement: if the raw JSON body doesn't even have the
        shape we expect, raise clearly -- never silently return an empty or
        partial ChatResponse."""
        provider = _make_provider()

        malformed_envelope = {"id": "resp_bad", "status": "completed"}  # no "output"
        fake_client = MagicMock()
        fake_client.responses.with_raw_response.create = AsyncMock(
            return_value=_FakeRawAPIResponse(malformed_envelope)
        )
        provider._client = fake_client

        params = {"model": "gpt-5.6-sol", "tools": [{"type": "computer"}]}

        with pytest.raises(RuntimeError, match="missing the expected 'output'"):
            await provider._create_response(params)
