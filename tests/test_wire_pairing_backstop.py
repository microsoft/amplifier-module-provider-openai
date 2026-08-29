"""Wire-path pairing backstop -- cross-vocabulary coverage.

`_convert_messages` ends with a last-resort backstop: any `function_call`
item replayed into the input with no paired output gets a synthesized
"[error] result missing" output, so the request cannot 400 on an orphan.
That check recognizes the full `_PAIRED_OUTPUT_ITEM_TYPES` vocabulary
(function_call_output, apply_patch_call_output, computer_call_output) --
NOT function_call_output alone.

This module is the ONE place asserting the backstop across ALL three
envelope types together (mixed native + function outputs on one turn), and
carries the `computer_call_output` case ported from the now-deleted
`tests/test_chain_output_pairing.py` (the chain-specific pairing helper
`_enforce_chain_output_pairing` this file used to test was removed with the
`previous_response_id` code path -- see the stateless-only refactor). The
per-envelope conversion-shape tests for apply_patch and computer_use live in
`test_apply_patch_integration.py::TestWirePathPairingCountsNativeOutputs`
and `test_computer_use_integration.py` respectively.
"""

from __future__ import annotations

# pyright: reportAttributeAccessIssue=false
from typing import Any
from unittest.mock import MagicMock

from amplifier_module_provider_openai import OpenAIProvider


def _make_provider(**overrides: Any) -> OpenAIProvider:
    coordinator = MagicMock()
    coordinator.get_capability = MagicMock(return_value=None)
    coordinator.hooks = MagicMock()
    return OpenAIProvider(
        api_key="test-key", config={**overrides}, coordinator=coordinator
    )


def test_computer_call_output_satisfies_wire_path_pairing() -> None:
    """A real computer_call_output pairs its call; nothing is synthesized.

    Mirrors TestWirePathPairingCountsNativeOutputs (apply_patch) for the
    other native envelope type sharing _PAIRED_OUTPUT_ITEM_TYPES.
    """
    provider = _make_provider()
    provider._native_call_ids = {"call_cua1"}
    provider._native_call_types = {"call_cua1": "computer"}

    messages = [
        {"role": "user", "content": "click it"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_call",
                    "id": "call_cua1",
                    "name": "computer",
                    "input": {"actions": [{"type": "click", "x": 1, "y": 2}]},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_cua1",
            "content": "data:image/png;base64,AAAA",
            "tool_name": "computer",
        },
    ]

    result = provider._convert_messages(messages)

    computer_outputs = [
        m
        for m in result
        if isinstance(m, dict) and m.get("type") == "computer_call_output"
    ]
    assert len(computer_outputs) == 1, "the real native computer result must survive"

    synthesized = [
        m
        for m in result
        if isinstance(m, dict)
        and m.get("type") == "function_call_output"
        and "[error]" in str(m.get("output", ""))
    ]
    assert not synthesized, (
        "synthesized a 'result missing' error for a call that already had a "
        "real, successful native computer_call_output result"
    )


def test_mixed_native_and_function_outputs_all_pair_at_the_wire() -> None:
    """A turn mixing apply_patch, computer, and plain function calls pairs
    every call at the wire backstop -- synthesizing none of them."""
    provider = _make_provider()
    provider._native_call_ids = {"call_patch1", "call_cua1"}
    provider._native_call_types = {"call_cua1": "computer"}

    messages = [
        {"role": "user", "content": "do three things"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_call",
                    "id": "call_patch1",
                    "name": "apply_patch",
                    "input": {"type": "update_file", "path": "a.py", "diff": "x"},
                },
                {
                    "type": "tool_call",
                    "id": "call_cua1",
                    "name": "computer",
                    "input": {"actions": [{"type": "click", "x": 1, "y": 2}]},
                },
                {
                    "type": "tool_call",
                    "id": "call_bash1",
                    "name": "bash",
                    "input": {"command": "ls"},
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_patch1",
            "content": "M a.py",
            "tool_name": "apply_patch",
        },
        {
            "role": "tool",
            "tool_call_id": "call_cua1",
            "content": "data:image/png;base64,AAAA",
            "tool_name": "computer",
        },
        {
            "role": "tool",
            "tool_call_id": "call_bash1",
            "content": "total 0",
            "tool_name": "bash",
        },
    ]

    result = provider._convert_messages(messages)

    by_type: dict[str, list[dict]] = {}
    for m in result:
        if isinstance(m, dict) and "type" in m:
            by_type.setdefault(m["type"], []).append(m)

    assert len(by_type.get("apply_patch_call_output", [])) == 1
    assert len(by_type.get("computer_call_output", [])) == 1
    function_outputs = by_type.get("function_call_output", [])
    assert [o["call_id"] for o in function_outputs] == ["call_bash1"], (
        "the plain function call's real output must be present"
    )
    assert not any("[error]" in str(o.get("output", "")) for o in function_outputs), (
        "no orphan should have been declared -- every call had a real, "
        "correctly-typed paired output"
    )
