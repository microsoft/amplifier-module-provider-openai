"""Read-only hosted Multi-agent request/event/injection prototype; NOT mounted.

Official contracts, retrieved 2026-11-06:
https://developers.openai.com/api/docs/guides/responses-multi-agent.md
https://developers.openai.com/api/docs/guides/async-tool-calling.md

The host retains actual generation, approval, tool execution, custody and budgets.
This reducer never constructs a client, reads credentials, or sends network data.
Missing beta SDK/model/account support remains a named incompatibility, not a
reason to fall back to an unguarded raw request. In-memory receipts are not
restart-durable. API acceptance does not prove the agent consumed a result.
"""

from __future__ import annotations

import importlib
import inspect
import json
from collections import deque
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field

from .programmatic import ProtocolFailure, ToolReceipt, encoded

BETA = "responses_multi_agent=v1"
MULTI_AGENT_SOURCE = "https://developers.openai.com/api/docs/guides/responses-multi-agent.md"


def proposal_tool() -> dict:
    return {
        "type": "function",
        "name": "get_proposal",
        "description": "Read the explicit mock snapshot for a proposal.",
        "parameters": {
            "type": "object",
            "properties": {"proposal": {"type": "string", "enum": ["alpha", "beta"]}},
            "required": ["proposal"],
            "additionalProperties": False,
        },
        "strict": True,
    }


@dataclass(frozen=True)
class ModelEvidence:
    """Host-supplied exact-model evidence, not inferred from a family prefix."""

    model: str
    supported: bool | None
    source: str
    account_verified: bool | None = None


def build_request(
    evidence: ModelEvidence,
    input_items: list[dict],
    *,
    transport: str,
    options: dict | None = None,
) -> tuple[dict, dict]:
    """Return a candidate plus negotiation metadata; submit only through host.

    Documented eligible models on 2026-11-06 are GPT-6.1 Sol and GPT-5.6
    variants. This function intentionally does not hardcode that list as
    account availability, and does not change the selected model.
    """
    if evidence.supported is not True or not evidence.model or not evidence.source:
        raise ProtocolFailure("Exact-model hosted Multi-agent capability unsupported or unknown")
    if transport not in {"http", "websocket"}:
        raise ValueError("Expected http or websocket")
    options = deepcopy(options or {})
    if "max_tool_calls" in options:
        raise ProtocolFailure("Hosted Multi-agent does not support max_tool_calls")
    if "summary" in options.get("reasoning", {}):
        raise ProtocolFailure("Hosted Multi-agent does not support reasoning.summary")
    if options.pop("standalone_compact", False):
        raise ProtocolFailure("Hosted Multi-agent does not support /responses/compact")
    multi = options.pop("multi_agent", {"enabled": True, "max_concurrent_subagents": 3})
    if multi.get("enabled") is not True:
        raise ProtocolFailure("This experiment requires explicit multi_agent.enabled")
    concurrency = multi.get("max_concurrent_subagents", 3)
    if type(concurrency) is not int or concurrency < 1:
        raise ProtocolFailure("max_concurrent_subagents must be a positive integer")
    tools = options.pop("tools", [proposal_tool()])
    # This prototype admits exactly the bounded read-only schema, not arbitrary
    # model-provided tools or a different host's tool registry.
    if tools != [proposal_tool()]:
        if any(t.get("async") for t in tools):
            if options.get("parallel_tool_calls", True):
                raise ProtocolFailure("Do not combine async and parallel tools in Multi-agent")
            if any("programmatic" in t.get("allowed_callers", []) for t in tools):
                raise ProtocolFailure("Async tools cannot be called programmatically")
        raise ProtocolFailure("Only the read-only get_proposal schema is admitted")
    allowed_options = {
        "reasoning", "context_management", "parallel_tool_calls", "instructions"
    }
    if set(options) - allowed_options:
        raise ProtocolFailure("Unsupported experiment options; do not silently forward")
    request = {
        "model": evidence.model,
        "store": False,
        "input": deepcopy(input_items),
        "multi_agent": deepcopy(multi),
        "tools": deepcopy(tools),
        **options,
    }
    if transport == "http":
        request["betas"] = [BETA]
        negotiation = {"boundary": "configured_client.beta.responses.create", "betas": [BETA]}
    else:
        request["type"] = "response.create"
        negotiation = {
            "boundary": "configured_client.beta.responses.connect",
            "extra_headers": {"OpenAI-Beta": BETA},
        }
    return request, {
        **negotiation,
        "model_evidence": evidence.source,
        "account_verified": evidence.account_verified,
        "submission": "not attempted",
        "implicit_server_compaction": True,
    }


def inspect_sdk(configured_client: object | None = None) -> dict:
    """Finite local compatibility inspection: no client creation or API probe."""
    report = {
        "live_api": "not called",
        "account_model_availability": "unknown",
        "configured_client": "not supplied" if configured_client is None else "supplied",
        "production_host": {
            "automatic_mount": False,
            "per_agent_approval_custody_budget_adapter": "not integrated",
            "durable_injection_journal": "not integrated",
        },
    }
    try:
        sdk = importlib.import_module("openai")
    except ImportError:
        return {**report, "sdk": "missing", "beta_responses": False}
    report["sdk"] = sdk.__version__
    try:
        beta = importlib.import_module("openai.resources.beta.responses")
        resource = beta.Responses
        create = inspect.signature(resource.create).parameters
        report["beta_responses"] = True
        report["beta_create_multi_agent"] = "multi_agent" in create
        report["beta_create_betas"] = "betas" in create
        report["beta_websocket_connect"] = callable(getattr(resource, "connect", None))
    except (ImportError, AttributeError):
        report["beta_responses"] = False
        report["reason"] = "Installed SDK lacks beta Responses resource; no raw bypass"
    if configured_client is not None:
        resource = getattr(getattr(configured_client, "beta", None), "responses", None)
        report["configured_beta_create"] = callable(getattr(resource, "create", None))
        report["configured_beta_connect"] = callable(getattr(resource, "connect", None))
    return report


@dataclass
class Injection:
    key: tuple[str, str, str]  # response, attributed agent, original API call ID
    call: dict
    output: dict
    host_receipt: ToolReceipt
    status: str = "prepared"
    applied: bool | None = None  # No API receipt alone proves consumption.
    acknowledgement: dict | None = None


@dataclass
class HostedAgentBoundary:
    """One response tree; only complete read-only function items reach the host.

    Serialize injection submission because the documented acknowledgement has
    response_id but no call_id. Other agents/events may still interleave. This
    sacrifices some throughput to avoid guessing which receipt an ack belongs
    to. Wire IDs remain unchanged; local ledger keys include the agent.
    """

    response_id: str | None = None
    output: list[dict] = field(default_factory=list)
    attribution: dict[int, str] = field(default_factory=dict)
    deltas: list[tuple[str, str]] = field(default_factory=list)
    calls: dict[tuple[str, str, str], Injection | None] = field(default_factory=dict)
    host_receipts: dict[tuple[str, str, str], ToolReceipt] = field(default_factory=dict)
    wire_owners: dict[str, str] = field(default_factory=dict)
    pending: deque = field(default_factory=deque)
    inflight: tuple[str, str, str] | None = None
    continuation_input: list[dict] = field(default_factory=list)
    completed_response: dict | None = None
    blocked: str | None = None

    def _fail(self, reason: str):
        self.blocked = reason
        raise ProtocolFailure(reason)

    def receive(self, event: dict, dispatch: Callable[[dict], ToolReceipt]) -> None:
        if self.blocked:
            raise ProtocolFailure(self.blocked)
        kind = event.get("type")
        if kind == "response.created":
            rid = event["response"]["id"]
            if self.response_id is not None:
                self._fail("Response identity already bound")
            self.response_id = rid
        elif kind in {"response.output_item.added", "response.output_item.done"}:
            item = deepcopy(event["item"])
            agent = item.get("agent", {}).get("agent_name")
            event_agent = event.get("agent", {}).get("agent_name")
            if agent is not None and event_agent is not None and agent != event_agent:
                self._fail("Conflicting event and item attribution")
            agent = agent or event_agent
            if not isinstance(agent, str) or not (
                agent == "/root" or agent.startswith("/root/")
            ):
                self._fail("Missing or invalid agent attribution; never assume root")
            index = event.get("output_index")
            if index is not None:
                if index in self.attribution and self.attribution[index] != agent:
                    self._fail("Output index changed agent")
                self.attribution[index] = agent
            if kind == "response.output_item.done":
                self.output.append(item)
                # Hosted multi_agent_call/output and encrypted agent_message
                # are retained, NOT run as application tools.
                if item.get("type") == "function_call":
                    self._execute(item, agent, dispatch)
        elif kind == "response.output_text.delta":
            agent = event.get("agent", {}).get("agent_name")
            indexed_agent = self.attribution.get(event.get("output_index"))
            if agent is not None and indexed_agent is not None and agent != indexed_agent:
                self._fail("Delta conflicts with output index agent attribution")
            agent = agent or indexed_agent
            if not isinstance(agent, str) or not (
                agent == "/root" or agent.startswith("/root/")
            ):
                self._fail("Unattributed text cannot be presented as root output")
            self.deltas.append((agent, event["delta"]))
        elif kind in {"response.inject.created", "response.inject.failed"}:
            self._acknowledge(event)
        elif kind == "response.completed":
            if event["response"].get("id") != self.response_id:
                self._fail("Completion response identity mismatch")
            if event["response"].get("status") != "completed":
                self._fail("Completion event lacks completed response status")
            self.completed_response = deepcopy(event["response"])
        elif kind in {"response.failed", "response.incomplete", "error"}:
            self.disconnect()
            self._fail("Response failed; retain host receipts and reconcile")

    def _execute(self, item: dict, agent: str, dispatch: Callable[[dict], ToolReceipt]):
        if self.response_id is None:
            self._fail("Function call arrived before response.created")
        if item.get("name") != "get_proposal" or item.get("status") != "completed":
            self._fail("Expected a completed read-only get_proposal call")
        call_id = item.get("call_id")
        if not isinstance(call_id, str) or not call_id:
            self._fail("Missing API call_id")
        key = (self.response_id, agent, call_id)
        if key in self.calls:
            self._fail("Repeated tool call; do not replay host work")
        if call_id in self.wire_owners and self.wire_owners[call_id] != agent:
            self._fail("Cross-agent wire call_id collision cannot be disambiguated")
        try:
            args = json.loads(item["arguments"])
        except (KeyError, TypeError, ValueError):
            self._fail("Invalid proposal arguments")
        if not isinstance(args, dict) or set(args) != {"proposal"}:
            self._fail("Expected exactly one proposal argument")
        if args["proposal"] not in {"alpha", "beta"}:
            self._fail("Unknown proposal")
        self.calls[key] = None  # record potential effect before host entry
        self.host_receipts[key] = ToolReceipt("unknown", "dispatch-not-settled")
        self.wire_owners[call_id] = agent
        attributed_call = deepcopy(item)
        attributed_call.setdefault("agent", {"agent_name": agent})
        try:
            receipt = dispatch(attributed_call)
        except Exception:  # noqa: BLE001 - host call may already have effects
            self._fail("Host call outcome unknown; never replay")
        if isinstance(receipt, ToolReceipt):
            self.host_receipts[key] = receipt
        if (
            not isinstance(receipt, ToolReceipt)
            or receipt.status != "completed"
            or not receipt.receipt_id
        ):
            self._fail("Host did not return a completed admission/execution receipt")
        result = receipt.result
        if (
            not isinstance(result, dict)
            or set(result) != {"estimated_weeks", "risk"}
            or type(result["estimated_weeks"]) is not int
            or result["estimated_weeks"] < 0
            or result["risk"] not in {"low", "medium", "high"}
        ):
            self._fail("Invalid structured proposal result")
        output = {
            "type": "function_call_output",
            "call_id": call_id,  # never namespace or rewrite the wire call ID
            "output": encoded(result),
        }
        injection = Injection(key, deepcopy(item), output, receipt)
        self.calls[key] = injection
        self.pending.append(key)

    def send_next(self, send: Callable[[dict], None], admit: Callable[[dict], bool]) -> dict:
        """Cross a supplied host send gate once; no automatic retries."""
        if self.blocked or self.inflight is not None or not self.pending:
            raise ProtocolFailure("No safely sendable injection")
        key = self.pending[0]
        injection = self.calls[key]
        envelope = {
            "type": "response.inject",
            "response_id": self.response_id,
            "input": [deepcopy(injection.output)],
        }
        # The synchronous gate receives a separate snapshot so it cannot alter
        # the candidate that is actually sent.
        try:
            allowed = admit(deepcopy(envelope))
        except Exception:  # noqa: BLE001 - fail closed on any admission error
            allowed = False
        if allowed is not True:
            raise ProtocolFailure("Host send admission refused; no send attempted")
        self.pending.popleft()
        self.inflight = key
        injection.status = "unknown"  # before the uncertain send boundary
        try:
            send(deepcopy(envelope))
        except Exception:  # noqa: BLE001 - send outcome unknown, never replay
            self._fail("Injection send outcome unknown; do not replay")
        injection.status = "awaiting_ack"
        return envelope

    def _acknowledge(self, event: dict):
        if self.inflight is None or event.get("response_id") != self.response_id:
            self._fail("Unmatched injection acknowledgement")
        injection = self.calls[self.inflight]
        injection.acknowledgement = deepcopy(event)
        self.inflight = None
        if event["type"] == "response.inject.created":
            injection.status = "accepted"
        else:
            if event.get("input") != [injection.output]:
                self._fail("Failed injection did not return the original output")
            code = event.get("error", {}).get("code")
            if code == "response_already_completed":
                injection.status = "continuation_required"
                # Carry the existing output, NEVER execute the tool again.
                self.continuation_input.extend(deepcopy(event["input"]))
            else:
                injection.status = "rejected"
                self._fail(f"Injection rejected: {code}")

    def disconnect(self) -> None:
        if self.inflight is not None:
            self.calls[self.inflight].status = "unknown"
        self.blocked = "Disconnected; unresolved effects require owner reconciliation"

    def final_text(self) -> str:
        if self.blocked or self.completed_response is None or self.inflight or self.pending:
            raise ProtocolFailure("Response/injection receipts are not settled")
        if self.continuation_input:
            raise ProtocolFailure("Continue with retained outputs before selecting final answer")
        messages = [
            item for item in self.completed_response.get("output", [])
            if item.get("type") == "message"
            and item.get("role") == "assistant"
            and item.get("status") == "completed"
            and item.get("agent", {}).get("agent_name") == "/root"
            and item.get("phase") == "final_answer"
        ]
        if not messages:
            raise ProtocolFailure("No attributed root final answer")
        return "".join(
            part.get("text", part.get("refusal", ""))
            for item in messages
            for part in item.get("content", [])
            if part.get("type") in {"output_text", "refusal"}
        )