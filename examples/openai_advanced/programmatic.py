"""Opt-in, offline-evaluable Responses programmatic tool-calling boundary.

Source (retrieved 2026-11-06):
https://developers.openai.com/api/docs/guides/tools-programmatic-tool-calling.md

This is not mounted by the provider. The host supplies response generation and
per-call admission/execution. No generated JavaScript is executed locally.
The in-memory ledger is NOT durable exactly-once delivery across restarts.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field


class ProtocolFailure(RuntimeError):
    """Stop and retain evidence; never retry a client call implicitly."""


def encoded(value: object) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True, allow_nan=False)


def read_only_tools() -> list[dict]:
    """Two structured functions; async and programmatic must not be combined."""
    functions = []
    for name, quantity in (
        ("get_inventory", "available_units"),
        ("get_demand", "requested_units"),
    ):
        functions.append(
            {
                "type": "function",
                "name": name,
                "description": f"Read mock SKU snapshot: sku and {quantity}.",
                "parameters": {
                    "type": "object",
                    "properties": {"sku": {"type": "string"}},
                    "required": ["sku"],
                    "additionalProperties": False,
                },
                "output_schema": {
                    "type": "object",
                    "properties": {
                        "sku": {"type": "string"},
                        quantity: {"type": "number"},
                    },
                    "required": ["sku", quantity],
                    "additionalProperties": False,
                },
                "allowed_callers": ["programmatic"],
                "strict": True,
            }
        )
    return [*functions, {"type": "programmatic_tool_calling"}]


PROGRAM = """const [stock, demand] = await Promise.all([
  tools.get_inventory({sku: "sku_123"}),
  tools.get_demand({sku: "sku_123"})
]);
if (stock.sku !== demand.sku) throw new Error("Mismatched SKU");
text(JSON.stringify({
  sku: stock.sku,
  available_units: stock.available_units,
  requested_units: demand.requested_units,
  shortage_units: Math.max(demand.requested_units - stock.available_units, 0)
}));"""


@dataclass(frozen=True)
class ToolReceipt:
    """Host-owned outcome, not a tool's self-attestation."""

    status: str  # completed, denied, failed or unknown
    receipt_id: str
    result: dict | None = None


@dataclass
class ProgrammaticRun:
    """Stateless ordered continuation with an inspectable non-replay ledger.

    ``respond`` is an existing host-controlled generation boundary, not a new
    client, credential source, or raw transport. ``dispatch`` must perform
    current approval/custody/budget checks for EVERY call. This prototype only
    permits the two read-only schema-checked functions above.
    """

    model: str
    capability: bool | None
    input_items: list[dict]
    history: list[dict] = field(init=False)
    receipts: dict[str, ToolReceipt] = field(default_factory=dict, init=False)
    identities: dict[str, str] = field(default_factory=dict, init=False)
    programs: dict[str, dict] = field(default_factory=dict, init=False)
    requests: list[dict] = field(default_factory=list, init=False)
    failure: str | None = field(default=None, init=False)
    final_text: str | None = field(default=None, init=False)

    def __post_init__(self):
        if not isinstance(self.model, str) or not self.model:
            raise ValueError("An explicitly selected model is required")
        if self.capability is not True:
            raise ProtocolFailure("Programmatic capability unsupported or unknown")
        self.history = deepcopy(self.input_items)

    def _fail(self, reason: str):
        self.failure = reason
        raise ProtocolFailure(reason)

    def _validate_call(self, call: dict) -> dict:
        if call.get("name") not in {"get_inventory", "get_demand"}:
            self._fail("Only the two read-only functions are admitted")
        if call.get("status", "completed") != "completed":
            self._fail("Incomplete function call cannot be dispatched")
        if call.get("async"):
            self._fail("Async tools cannot be called programmatically")
        if not isinstance(call.get("call_id"), str) or not call["call_id"]:
            self._fail("Missing function call identity")
        caller = call.get("caller", {})
        if caller.get("type") != "program" or caller.get("caller_id") not in self.programs:
            self._fail("Nested function does not identify a known program")
        try:
            arguments = json.loads(call["arguments"])
        except (KeyError, TypeError, ValueError):
            self._fail("Invalid function arguments")
        if (
            not isinstance(arguments, dict)
            or set(arguments) != {"sku"}
            or not isinstance(arguments["sku"], str)
            or not arguments["sku"]
        ):
            self._fail("Expected exactly one nonempty sku")
        return arguments

    def _validate_result(self, call: dict, arguments: dict, result: dict):
        field_name = (
            "available_units" if call["name"] == "get_inventory" else "requested_units"
        )
        if (
            not isinstance(result, dict)
            or set(result) != {"sku", field_name}
            or result["sku"] != arguments["sku"]
            or isinstance(result[field_name], bool)
            or not isinstance(result[field_name], (int, float))
        ):
            self._fail("Host result did not match output_schema and requested SKU")
        # Reject NaN and Infinity as well as non-JSON values.
        try:
            encoded(result)
        except (TypeError, ValueError):
            self._fail("Host result was not finite JSON")

    def run(
        self,
        respond: Callable[[dict], dict],
        dispatch: Callable[[dict], ToolReceipt],
        *,
        max_responses: int,
    ) -> str:
        """Run a bounded protocol exercise; no catch/retry of unknown effects."""
        if self.failure is not None or self.requests:
            raise ProtocolFailure("This run already started; reconcile, do not replay")
        if type(max_responses) is not int or max_responses < 1:
            raise ValueError("A positive response budget is required")
        for _ in range(max_responses):
            request = {
                "model": self.model,
                "store": False,
                "input": deepcopy(self.history),
                "tools": read_only_tools(),
            }
            self.requests.append(deepcopy(request))
            try:
                response = respond(request)
            except Exception:  # noqa: BLE001 - arbitrary host errors leave outcome unknown
                self._fail("Response outcome unknown; reconcile before continuation")
            output = response.get("output", [])
            if not isinstance(output, list) or any(not isinstance(i, dict) for i in output):
                self._fail("Invalid response output")
            # Preserve opaque fingerprint, reasoning encrypted_content, caller,
            # program_output and any other replayable returned fields verbatim.
            self.history.extend(deepcopy(output))
            if response.get("status") != "completed":
                self._fail("Response incomplete or failed; no client calls dispatched")
            for item in output:
                if item.get("type") == "program":
                    program_id = item.get("call_id")
                    if not program_id or not item.get("fingerprint"):
                        self._fail("Program lacks call_id or opaque fingerprint")
                    if program_id in self.programs and self.programs[program_id] != item:
                        self._fail("Program identity changed")
                    self.programs[program_id] = deepcopy(item)
                elif item.get("type") == "program_output":
                    if item.get("call_id") not in self.programs:
                        self._fail("Output does not identify a known program")
                    if item.get("status") != "completed":
                        self._fail("Program failed after client calls; do not repeat calls")
            calls = [i for i in output if i.get("type") == "function_call"]
            # Validate the entire batch before invoking any function.
            arguments = [self._validate_call(call) for call in calls]
            ids = [call["call_id"] for call in calls]
            if len(ids) != len(set(ids)):
                self._fail("Duplicate call identity in one batch")
            if any(call_id in self.identities for call_id in ids):
                self._fail("Repeated call identity; completed/unknown calls are not replayed")
            for call, args in zip(calls, arguments, strict=True):
                call_id = call["call_id"]
                digest = hashlib.sha256(encoded(call).encode()).hexdigest()
                self.identities[call_id] = digest
                # Record uncertainty BEFORE crossing the host boundary.
                self.receipts[call_id] = ToolReceipt("unknown", "dispatch-not-settled")
                try:
                    receipt = dispatch(deepcopy(call))
                except Exception:  # noqa: BLE001 - never retry arbitrary host failures
                    self._fail("Client call outcome unknown; do not repeat")
                if not isinstance(receipt, ToolReceipt) or not receipt.receipt_id:
                    self._fail("Missing host receipt; client call outcome unknown")
                self.receipts[call_id] = receipt
                if receipt.status != "completed":
                    self._fail(f"Client call {receipt.status}; no automatic replay")
                self._validate_result(call, args, receipt.result)
                self.history.append(
                    {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": encoded(receipt.result),
                        "caller": deepcopy(call["caller"]),
                    }
                )
            if calls:
                continue
            messages = [
                i for i in output
                if i.get("type") == "message"
                and i.get("role") == "assistant"
                and i.get("status") == "completed"
                # Phase-less messages retain the ordinary Responses contract.
                # Explicit commentary never completes a model turn.
                and i.get("phase") in {None, "final_answer"}
            ]
            if messages:
                self.final_text = "".join(
                    p.get("text", p.get("refusal", ""))
                    for message in messages
                    for p in message.get("content", [])
                    if p.get("type") in {"output_text", "refusal"}
                )
                return self.final_text
            # A completed program is NOT the final assistant answer.
        self._fail("Response budget exhausted before final assistant message")


def compare_mock_aggregation() -> dict:
    """Deterministic correctness/bytes only; not tokens, model quality or latency."""
    raw = [
        {"sku": "sku_123", "available_units": 42},
        {"sku": "sku_123", "requested_units": 51},
    ]
    result = {
        **raw[0],
        "requested_units": raw[1]["requested_units"],
        "shortage_units": max(raw[1]["requested_units"] - raw[0]["available_units"], 0),
    }
    return {
        "fixture": "explicit mock, no API requests",
        "result": result,
        "raw_tool_payload_bytes": len(encoded(raw).encode()),
        "reduced_payload_bytes": len(encoded(result).encode()),
        "boundary_counts": {
            "direct_parallel": {"host_tool_executions": 2, "tool_result_batches": 1},
            "host_tool_exec": {"host_tool_executions": 2, "tool_result_batches": 1},
            "native_programmatic": {"host_tool_executions": 2, "tool_result_batches": 1},
        },
        "interpretation": (
            "This compact case has no demonstrated context-size win. Native PTC "
            "adds hosted program continuation; existing host tool_exec already "
            "aggregates two tools. Byte counts exclude protocol envelopes. "
            "Stateless replay still retains all raw outputs; bytes are not billed "
            "tokens or proof of model-visible context size. No latency/quality claim."
        ),
    }