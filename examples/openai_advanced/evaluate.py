"""Run finite mock examples: python -m examples.openai_advanced.evaluate.

Prints JSON to stdout. No network, paid requests, configured-client creation,
filesystem writes, services or host-setting changes. Test commands are separate.
"""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import gettempdir

from .hosted_agents import (
    HostedAgentBoundary,
    ModelEvidence,
    build_request,
    inspect_sdk,
)
from .multiplex import Custody, MockSocket, Multiplex
from .programmatic import (
    PROGRAM,
    ProgrammaticRun,
    ToolReceipt,
    compare_mock_aggregation,
    encoded,
)

SOURCES = [
    "https://developers.openai.com/api/docs/guides/tools-programmatic-tool-calling.md",
    "https://developers.openai.com/api/docs/guides/websocket-mode.md",
    "https://developers.openai.com/api/docs/guides/responses-multi-agent.md",
    "https://developers.openai.com/api/docs/guides/async-tool-calling.md",
]


def programmatic_example() -> dict:
    comparison = compare_mock_aggregation()
    caller = {"type": "program", "caller_id": "mock_program"}
    script = iter([
        {
            "status": "completed",
            "output": [
                {"type": "reasoning", "encrypted_content": "mock_opaque_reasoning"},
                {"type": "program", "call_id": "mock_program", "code": PROGRAM,
                 "fingerprint": "mock_opaque_fingerprint"},
                *[
                    {"type": "function_call", "name": name, "call_id": name,
                     "arguments": '{"sku":"sku_123"}', "caller": caller}
                    for name in ("get_inventory", "get_demand")
                ],
            ],
        },
        {
            "status": "completed",
            "output": [
                {"type": "program_output", "call_id": "mock_program",
                 "result": encoded(comparison["result"]), "status": "completed"},
            ],
        },
        {
            "status": "completed",
            "output": [
                {"type": "message", "role": "assistant", "status": "completed",
                 "content": [{"type": "output_text", "text": "Mock shortage: 9 units."}]},
            ],
        },
    ])
    calls = []

    def dispatch(call):
        calls.append(call["call_id"])
        name = "available_units" if call["name"] == "get_inventory" else "requested_units"
        return ToolReceipt(
            "completed", f"mock_host:{call['call_id']}",
            {"sku": "sku_123", name: 42 if name == "available_units" else 51},
        )

    run = ProgrammaticRun("explicit-mock-model", True, [
        {"role": "user", "content": "Compare the explicit mock SKU snapshots."},
    ])
    text = run.run(lambda _: next(script), dispatch, max_responses=3)
    assert text == "Mock shortage: 9 units."
    assert len(calls) == 2 and len(run.requests) == 3
    return {
        **comparison,
        "mock_final": text,
        "mock_response_count": len(run.requests),
        "model_generation_quality": "not evaluated; all responses are fixtures",
        "hosted_javascript_execution": "not evaluated; never executes locally",
        "rollout": (
            "Keep opt-in research only. Two compact lookups do not justify replacing "
            "existing host tool_exec. Qualify large-output tasks, evidence preservation "
            "and durable receipts before any production integration."
        ),
    }


def multiplex_example() -> dict:
    socket = MockSocket()
    mux = Multiplex(socket)
    owners = {name: Custody(f"mock-owner-{name}", f"mock-token-{name}")
              for name in ("planner", "research")}
    # Identity strings only: the prototype does not create or access these dirs.
    base = Path(gettempdir()) / "openai-multiplex-identity-only"
    for stream, owner in owners.items():
        mux.register(stream, owner, workdir=str(base / stream),
                     provenance=("explicit-mock-commission", stream), request_limit=2)
        mux.submit(stream, owner, request_id=f"mock-{stream}",
                   model="explicit-mock-model", input=[{"role": "user", "content": stream}])
    mux.pump()
    interleaving = [
        ("response.created", "planner"),
        ("response.created", "research"),
        ("response.output_text.delta", "research"),
        ("response.output_text.delta", "planner"),
        ("response.completed", "research"),
        ("response.completed", "planner"),
    ]
    for kind, stream in interleaving:
        event = {"type": kind, "stream_id": stream}
        if kind == "response.output_text.delta":
            event.update({"response_id": f"resp_{stream}", "delta": stream})
        else:
            event["response"] = {"id": f"resp_{stream}", "status": "completed"}
        socket.feed(event)
        mux.receive()
    states = {
        stream: mux.snapshot(stream, owner).requests[f"mock-{stream}"].state
        for stream, owner in owners.items()
    }
    assert states == {"planner": "completed", "research": "completed"}
    return {
        "fixture": "one JSON socket mock, two owned streams, interleaved frames",
        "socket_connections": 1,
        "create_frames": len(socket.sent),
        "states": states,
        "native_provider_modified": False,
        "rollout": (
            "Keep isolated. Actual native transport has a single generation owner. "
            "Production requires a host-owned per-stream admission/custody/journal "
            "adapter; this caller-supplied identity reducer is not that adapter."
        ),
    }


def hosted_example() -> dict:
    request, negotiation = build_request(
        ModelEvidence("explicit-mock-model", True, "explicit mock capability fixture"),
        [{"role": "user", "content": "Compare mock proposal alpha and beta."}],
        transport="websocket",
    )
    boundary = HostedAgentBoundary()
    boundary.receive({"type": "response.created", "response": {"id": "resp_mock"}}, None)
    proposals = {
        "alpha": {"estimated_weeks": 6, "risk": "medium"},
        "beta": {"estimated_weeks": 8, "risk": "low"},
    }
    host_calls = []

    def dispatch(call):
        host_calls.append(call)
        proposal = json.loads(call["arguments"])["proposal"]
        return ToolReceipt(
            "completed", f"mock-host:{call['call_id']}", proposals[proposal],
        )

    for index, proposal in enumerate(proposals):
        agent = {"agent_name": f"/root/{proposal}_reviewer"}
        boundary.receive({
            "type": "response.output_item.done", "agent": agent, "output_index": index,
            "item": {
                "type": "function_call", "status": "completed", "name": "get_proposal",
                "arguments": encoded({"proposal": proposal}),
                "call_id": f"mock_call_{proposal}", "agent": agent,
            },
        }, dispatch)
    frames = []
    for _ in proposals:
        boundary.send_next(frames.append, lambda _: True)
        boundary.receive({"type": "response.inject.created", "response_id": "resp_mock"}, None)
    boundary.receive({
        "type": "response.completed",
        "response": {
            "id": "resp_mock", "status": "completed",
            "output": [
                {"type": "message", "role": "assistant", "status": "completed",
                 "phase": "final_answer", "agent": {"agent_name": "/root"},
                 "content": [{"type": "output_text", "text": "Mock root synthesis."}]},
            ],
        },
    }, None)
    assert boundary.final_text() == "Mock root synthesis."
    injections = list(boundary.calls.values())
    assert all(item.status == "accepted" for item in injections)
    assert len(host_calls) == 2
    return {
        "fixture": "attributed beta JSON events, no hosted agent actually ran",
        "request_candidate": request,
        "negotiation": negotiation,
        "injection_frames": frames,
        "injection_status": "accepted",
        "result_applied": None,
        "host_tool_calls": len(host_calls),
        "attributed_agents": [call["agent"]["agent_name"] for call in host_calls],
        "mock_proposals": proposals,
        "root_final": boundary.final_text(),
        "documented_model_scope": "GPT-6.1 Sol and GPT-5.6 variants as of source retrieval",
        "rollout": (
            "Do not enable by default. SDK/model/account qualification, beta "
            "negotiation, durable host approval/custody/budget receipts, and "
            "implicit per-agent compaction compatibility remain required."
        ),
    }


def evaluate() -> dict:
    return {
        "classification": "offline mock prototype evaluation, NOT production qualification",
        "sources_retrieved_local_date": "2026-11-06",
        "official_sources": SOURCES,
        "paid_api_calls": 0,
        "sdk_compatibility": inspect_sdk(),
        "sdk_floor_3_29": "not tested in this interpreter; rerun in qualified floor environment",
        "programmatic": programmatic_example(),
        "multiplex": multiplex_example(),
        "hosted_multi_agent": hosted_example(),
        "checks": [
            "python -m unittest discover -s tests/experiments -v",
            "python -m examples.openai_advanced.evaluate",
            "ruff check examples/openai_advanced tests/experiments",
        ],
        "global_limitations": [
            "No model-quality, actual latency, cost or billed-token measurements.",
            "In-memory ledgers cannot establish restart-safe exactly-once execution.",
            "Caller-supplied fixture evidence and custody do not confer host authority.",
            "No provider mount/default, credentials, live socket, or host settings changed.",
        ],
    }


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2, sort_keys=True, allow_nan=False))