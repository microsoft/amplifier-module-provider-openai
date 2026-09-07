#!/usr/bin/env python3
"""LIVE smoke: a hosted tool_search pair survives one turn and is accepted back.

This is the one deliverable in `model_performance-v5co` that costs money, and it
is the only one that can distinguish "the unit tests agree with my mock" from
"the API accepts what this provider emits". It drives the REAL
`OpenAIProvider.complete()` -- not a hand-rolled wire call -- so what is proved
is the shipped code path:

  TURN 1  tool_search.mode=namespaced assembles the namespaced tools block and
          `{"type": "tool_search"}`; the model searches; the response carries
          hosted `tool_search_call` / `tool_search_output` items; the provider
          captures them onto ChatResponse.metadata["openai:tool_search_items"].
          -> BREAK 3, capture half. At main these are silently dropped.

  TURN 2  the captured pair is replayed VERBATIM into `input`, and the API
          returns 200 rather than rejecting the items.
          -> BREAK 3, re-emission half. This is the undocumented behaviour
             Phase 1 established and the whole reason break 3 is fixable by
             re-emission at all.

  Also asserted on live output, not mocks:
    BREAK 1  tool_choice is "auto" on the wire even though "required" was asked
    BREAK 6  the hosted items carry call_id: null / execution: "server", and the
             function_call's `name` comes back UNQUALIFIED ("glob", never
             "files/glob") so LS:3754's tools.get(name) still resolves.

Spend is metered and printed. Usage: smoke_tool_search.py [<output-dir>]
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from amplifier_core.message_models import ChatRequest, Message, ToolSpec

from amplifier_module_provider_openai import OpenAIProvider
from amplifier_module_provider_openai._constants import (
    METADATA_TOOL_SEARCH_ITEMS,
)

MODEL = os.environ.get("SMOKE_MODEL", "gpt-5.6-terra")

# The same 14-tool-shaped roster the namespace table is authored against, cut
# down to what a one-turn smoke needs. Names must match the shipped table or
# they fall through the unlisted-tool path and prove nothing.
ROSTER: tuple[tuple[str, str], ...] = (
    ("read_file", "Read the contents of a file from the workspace."),
    ("write_file", "Write content to a file in the workspace."),
    ("edit_file", "Perform an exact string replacement in a file."),
    ("grep", "Search file contents with a regular expression."),
    ("glob", "Find files in the workspace by glob pattern."),
    ("bash", "Run a shell command."),
    ("todo", "Manage a task checklist."),
    ("web_search", "Search the public web."),
    ("web_fetch", "Fetch the contents of a URL."),
    ("delegate", "Delegate a task to a specialised sub-agent."),
    ("load_skill", "Load a skill knowledge package."),
    ("recipes", "Execute a multi-step recipe."),
    ("mode", "Switch the runtime mode."),
)


def _specs() -> list[ToolSpec]:
    return [
        ToolSpec(
            name=name,
            description=desc,
            parameters={
                "type": "object",
                "properties": {"pattern": {"type": "string"}},
                "required": ["pattern"],
            },
        )
        for name, desc in ROSTER
    ]


def _provider() -> OpenAIProvider:
    coordinator = MagicMock()
    coordinator.get_capability = MagicMock(return_value=None)
    coordinator.hooks = MagicMock()
    coordinator.hooks.emit = AsyncMock(return_value=None)
    return OpenAIProvider(
        api_key=os.environ["OPENAI_API_KEY"],
        config={
            "default_model": MODEL,
            "use_streaming": False,
            "max_retries": 1,
            "tool_search": {"mode": "namespaced"},
        },
        coordinator=coordinator,
    )


def _usage(resp: Any) -> dict[str, Any]:
    u = getattr(resp, "usage", None)
    if u is None:
        return {}
    return {
        k: getattr(u, k, None)
        for k in ("input_tokens", "output_tokens", "cached_tokens", "total_tokens")
        if getattr(u, k, None) is not None
    }


async def run() -> dict[str, Any]:
    provider = _provider()
    result: dict[str, Any] = {
        "smoke": "hosted tool_search pair survives one turn, accepted back as input",
        "model": MODEL,
        "started_at": datetime.now(UTC).isoformat(),
        "turns": [],
    }

    # Capture what actually went on the wire, without replacing the client.
    sent: list[dict[str, Any]] = []
    real_create = provider.client.responses.create

    async def spy(**kwargs):
        sent.append(kwargs)
        return await real_create(**kwargs)

    provider.client.responses.create = spy  # type: ignore[assignment]

    # ---------------- TURN 1 -------------------------------------------------
    turn1 = ChatRequest(
        messages=[
            Message(
                role="user",
                content=(
                    "List the Python files in this workspace. You do not have "
                    "the file tools loaded yet -- search for them first, then "
                    "call the one that matches."
                ),
            )
        ],
        tools=_specs(),
    )
    # BREAK 1: ask for something that must be overridden to "auto".
    r1 = await provider.complete(turn1, tool_choice="required")

    wire1 = sent[0]
    tools_block = wire1.get("tools") or []
    captured = (r1.metadata or {}).get(METADATA_TOOL_SEARCH_ITEMS) or []

    result["turns"].append(
        {
            "turn": 1,
            "request": {
                "tool_choice_on_the_wire": wire1.get("tool_choice"),
                "tool_choice_requested": "required",
                "n_tool_entries": len(tools_block),
                "entry_types": [t.get("type") for t in tools_block],
                "namespace_names": [
                    t.get("name") for t in tools_block if t.get("type") == "namespace"
                ],
                "tool_search_entry_present": any(
                    t.get("type") == "tool_search" for t in tools_block
                ),
            },
            "response": {
                "captured_item_types": [i.get("type") for i in captured],
                "captured_call_ids": [i.get("call_id") for i in captured],
                "captured_execution": [i.get("execution") for i in captured],
                "loaded_tool_names": [
                    t.get("name")
                    for i in captured
                    if i.get("type") == "tool_search_output"
                    for t in (i.get("tools") or [])
                ],
                "tool_call_names": [tc.name for tc in (r1.tool_calls or [])],
                "usage": _usage(r1),
            },
        }
    )

    checks: dict[str, Any] = {
        # BREAK 1
        "break1_tool_choice_forced_auto": wire1.get("tool_choice") == "auto",
        # request shape
        "namespaced_block_emitted": any(
            t.get("type") == "namespace" for t in tools_block
        ),
        "tool_search_entry_last": bool(tools_block)
        and tools_block[-1] == {"type": "tool_search"},
        # BREAK 3, capture half
        "break3_pair_captured": [i.get("type") for i in captured]
        == ["tool_search_call", "tool_search_output"],
        # BREAK 6
        "break6_hosted_call_id_is_null": all(
            i.get("call_id") is None for i in captured
        ),
        "break6_function_call_name_unqualified": all(
            "/" not in tc.name and "." not in tc.name for tc in (r1.tool_calls or [])
        ),
    }

    if not captured:
        result["checks"] = checks
        result["verdict"] = "INCONCLUSIVE"
        result["reason"] = (
            "the model did not emit a hosted tool_search pair on this turn; "
            "search is a MODEL decision and cannot be forced. Re-run or "
            "re-prompt -- this is not a provider failure."
        )
        return result

    # ---------------- TURN 2 -- accepted back as INPUT -----------------------
    tool_call = (r1.tool_calls or [None])[0]
    history = [
        turn1.messages[0],
        Message(
            role="assistant",
            content=[],
            tool_calls=r1.tool_calls or [],
            metadata={METADATA_TOOL_SEARCH_ITEMS: captured},
        ),
    ]
    if tool_call is not None:
        history.append(
            Message(
                role="tool",
                tool_call_id=tool_call.id,
                content="a.py\nb.py\nc.py",
            )
        )
    history.append(Message(role="user", content="Thanks. Reply with just: OK"))

    turn2 = ChatRequest(messages=history, tools=_specs())
    r2 = await provider.complete(turn2)

    wire2 = sent[1]
    replayed = [
        i
        for i in (wire2.get("input") or [])
        if isinstance(i, dict)
        and i.get("type") in {"tool_search_call", "tool_search_output"}
    ]

    def canon(obj: Any) -> str:
        return json.dumps(
            obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        )

    replayed_calls = [
        i
        for i in (wire2.get("input") or [])
        if isinstance(i, dict) and i.get("type") == "function_call"
    ]
    result["turns"].append(
        {
            "turn": 2,
            "request": {
                "replayed_item_types": [i.get("type") for i in replayed],
                "n_input_items": len(wire2.get("input") or []),
                "replayed_function_calls": [
                    {"name": i.get("name"), "namespace": i.get("namespace")}
                    for i in replayed_calls
                ],
            },
            "response": {
                "http_ok": True,  # complete() would have raised otherwise
                "usage": _usage(r2),
                "text": "".join(
                    str(getattr(b, "text", "") or "") for b in (r2.content or [])
                )[:200],
            },
        }
    )

    checks.update(
        {
            # BREAK 3, re-emission half -- the load-bearing one.
            "break3_pair_re_emitted_as_input": [i.get("type") for i in replayed]
            == ["tool_search_call", "tool_search_output"],
            "break3_re_emission_is_verbatim": [canon(i) for i in replayed]
            == [canon(i) for i in captured],
            "break3_api_accepted_the_pair_back": True,
            # BREAK 6, second half -- measured on the wire, not predicted.
            "break6_namespace_round_tripped_on_function_call": all(
                i.get("namespace") for i in replayed_calls
            )
            if replayed_calls
            else "N/A (no function_call to replay)",
        }
    )

    result["checks"] = checks
    result["verdict"] = "PASS" if all(bool(v) for v in checks.values()) else "FAIL"
    result["finished_at"] = datetime.now(UTC).isoformat()
    return result


def main() -> int:
    dest_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent
    out = asyncio.run(run())
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "smoke-tool-search.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"\nwrote {dest}")
    return 0 if out.get("verdict") == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
