"""Offline beta Multi-agent contract fixtures, not evidence of API entitlement."""

import unittest
from unittest.mock import Mock, patch

from examples.openai_advanced.hosted_agents import (
    BETA,
    HostedAgentBoundary,
    ModelEvidence,
    build_request,
    inspect_sdk,
    proposal_tool,
)
from examples.openai_advanced.programmatic import ProtocolFailure, ToolReceipt


def created():
    return {"type": "response.created", "response": {"id": "resp_mock"}}


def call(agent="/root/reviewer", call_id="call_a", proposal="alpha"):
    return {
        "type": "response.output_item.done", "output_index": 0,
        "agent": {"agent_name": agent},
        "item": {
            "type": "function_call", "name": "get_proposal",
            "call_id": call_id, "id": f"fc_{call_id}", "status": "completed",
            "arguments": '{"proposal":"' + proposal + '"}',
            "agent": {"agent_name": agent},
        },
    }


def dispatch(_):
    return ToolReceipt("completed", "host:readonly", {"estimated_weeks": 6, "risk": "medium"})


def ack():
    return {"type": "response.inject.created", "response_id": "resp_mock", "sequence_number": 42}


def completed():
    def message(agent, phase, text):
        return {
            "type": "message", "role": "assistant", "status": "completed",
            "agent": {"agent_name": agent}, "phase": phase,
            "content": [{"type": "output_text", "text": text}],
        }
    return {
        "type": "response.completed",
        "response": {
            "id": "resp_mock", "status": "completed",
            "output": [
                message("/root/reviewer", "final_answer", "Child answer."),
                message("/root", "commentary", "Working."),
                message("/root", "final_answer", "Root synthesis."),
            ],
        },
    }


class HostedAgentTests(unittest.TestCase):
    def new_boundary(self):
        boundary = HostedAgentBoundary()
        boundary.receive(created(), dispatch)
        return boundary

    def test_request_beta_and_model_evidence_not_availability(self):
        # Explicit mock model; this proves request shape, not a real model list.
        evidence = ModelEvidence("explicit-mock-model", True, "mock capability fixture")
        http, negotiation = build_request(evidence, [], transport="http")
        self.assertEqual(http["betas"], [BETA])
        self.assertTrue(http["multi_agent"]["enabled"])
        self.assertTrue(negotiation["implicit_server_compaction"])
        self.assertIsNone(negotiation["account_verified"])
        ws, negotiation = build_request(evidence, [], transport="websocket")
        self.assertNotIn("betas", ws)
        self.assertEqual(negotiation["extra_headers"], {"OpenAI-Beta": BETA})
        self.assertEqual(ws["type"], "response.create")

    def test_unknown_capability_does_not_build_request(self):
        for supported in (False, None):
            with self.assertRaisesRegex(ProtocolFailure, "unknown"):
                build_request(ModelEvidence("gpt-6-something", supported, "unknown"),
                              [], transport="http")

    def test_incompatible_features_are_rejected_not_stripped(self):
        for options in (
            {"max_tool_calls": 1},
            {"reasoning": {"summary": "auto"}},
            {"standalone_compact": True},
            {"model": "silently-replace-model"},
            {"multi_agent": {"enabled": True, "max_concurrent_subagents": 0}},
            {"multi_agent": {"enabled": False}},
        ):
            with self.subTest(options=options), self.assertRaises(ProtocolFailure):
                build_request(ModelEvidence("mock", True, "fixture"), [], transport="http",
                              options=options)

    def test_async_parallel_and_async_programmatic_fail(self):
        for parallel, callers in ((True, ["direct"]), (False, ["programmatic"])):
            tool = proposal_tool()
            tool.update({"async": True, "allowed_callers": callers})
            with self.assertRaisesRegex(ProtocolFailure, "async|Async"):
                build_request(ModelEvidence("mock", True, "fixture"), [], transport="http",
                              options={"tools": [tool], "parallel_tool_calls": parallel})

    def test_hosted_actions_and_encrypted_agent_messages_are_not_client_calls(self):
        boundary = self.new_boundary()
        run = Mock(side_effect=AssertionError("must not execute hosted action"))
        for kind in ("multi_agent_call", "multi_agent_call_output", "agent_message"):
            boundary.receive(
                {"type": "response.output_item.done", "agent": {"agent_name": "/root"},
                 "item": {"type": kind, "call_id": "hosted_call", "opaque": "preserved",
                          "agent": {"agent_name": "/root"}}},
                run,
            )
        run.assert_not_called()
        self.assertEqual([i["type"] for i in boundary.output],
                         ["multi_agent_call", "multi_agent_call_output", "agent_message"])
        self.assertEqual(boundary.output[-1]["opaque"], "preserved")

    def test_attribution_and_root_final_selection(self):
        boundary = self.new_boundary()
        boundary.receive(call(), dispatch)
        boundary.receive({"type": "response.output_text.delta", "output_index": 0,
                          "delta": "Child text"}, dispatch)
        boundary.send_next(lambda _: None, lambda _: True)
        boundary.receive(ack(), dispatch)
        boundary.receive(completed(), dispatch)
        self.assertEqual(boundary.deltas, [("/root/reviewer", "Child text")])
        self.assertEqual(boundary.final_text(), "Root synthesis.")
        injection = next(iter(boundary.calls.values()))
        self.assertEqual(injection.status, "accepted")
        self.assertIsNone(injection.applied)  # accepted is NOT consumption proof

    def test_injection_envelope_exact_and_acks_serialized(self):
        boundary = self.new_boundary()
        boundary.receive(call(), dispatch)
        second = call("/root/checker", "call_b", "beta")
        second["output_index"] = 1
        boundary.receive(second, dispatch)
        sent = []
        first = boundary.send_next(sent.append, lambda _: True)
        self.assertEqual(first, {
            "type": "response.inject", "response_id": "resp_mock",
            "input": [{"type": "function_call_output", "call_id": "call_a",
                       "output": '{"estimated_weeks":6,"risk":"medium"}'}],
        })
        with self.assertRaisesRegex(ProtocolFailure, "safely sendable"):
            boundary.send_next(sent.append, lambda _: True)
        boundary.receive(ack(), dispatch)
        boundary.send_next(sent.append, lambda _: True)
        boundary.receive(ack(), dispatch)
        self.assertEqual(len(sent), 2)
        self.assertEqual(list(boundary.calls), [
            ("resp_mock", "/root/reviewer", "call_a"),
            ("resp_mock", "/root/checker", "call_b"),
        ])

    def test_late_injection_returns_existing_output_without_reexecution(self):
        boundary = self.new_boundary()
        called = Mock(side_effect=dispatch)
        boundary.receive(call(), called)
        envelope = boundary.send_next(lambda _: None, lambda _: True)
        boundary.receive({
            "type": "response.inject.failed", "response_id": "resp_mock",
            "input": envelope["input"], "sequence_number": 43,
            "error": {"code": "response_already_completed"},
        }, called)
        boundary.receive(completed(), called)
        self.assertEqual(boundary.continuation_input, envelope["input"])
        self.assertEqual(called.call_count, 1)
        with self.assertRaisesRegex(ProtocolFailure, "Continue with retained"):
            boundary.final_text()

    def test_unknown_send_never_replays(self):
        boundary = self.new_boundary()
        boundary.receive(call(), dispatch)

        def unknown(_):
            raise TimeoutError("send may have reached server")

        with self.assertRaisesRegex(ProtocolFailure, "unknown"):
            boundary.send_next(unknown, lambda _: True)
        self.assertEqual(next(iter(boundary.calls.values())).status, "unknown")
        with self.assertRaises(ProtocolFailure):
            boundary.send_next(lambda _: self.fail("must not send"), lambda _: True)

    def test_stop_at_send_gate_does_not_send(self):
        boundary = self.new_boundary()
        boundary.receive(call(), dispatch)
        sent = Mock()
        with self.assertRaisesRegex(ProtocolFailure, "refused"):
            boundary.send_next(sent, lambda _: False)
        sent.assert_not_called()
        self.assertEqual(next(iter(boundary.calls.values())).status, "prepared")

    def test_disconnect_after_send_is_unknown_not_accepted(self):
        boundary = self.new_boundary()
        boundary.receive(call(), dispatch)
        boundary.send_next(lambda _: None, lambda _: True)
        boundary.disconnect()
        self.assertEqual(next(iter(boundary.calls.values())).status, "unknown")
        with self.assertRaises(ProtocolFailure):
            boundary.final_text()

    def test_no_final_before_all_injection_receipts(self):
        boundary = self.new_boundary()
        boundary.receive(call(), dispatch)
        boundary.send_next(lambda _: None, lambda _: True)
        boundary.receive(completed(), dispatch)
        with self.assertRaises(ProtocolFailure):
            boundary.final_text()
        boundary.receive(ack(), dispatch)
        self.assertEqual(boundary.final_text(), "Root synthesis.")

    def test_function_call_before_created_cannot_dispatch(self):
        with self.assertRaisesRegex(ProtocolFailure, "before response.created"):
            HostedAgentBoundary().receive(call(), lambda _: self.fail("no dispatch"))

    def test_event_only_attribution_reaches_host_dispatch(self):
        boundary = self.new_boundary()
        event = call()
        del event["item"]["agent"]
        called = Mock(side_effect=dispatch)
        boundary.receive(event, called)
        self.assertEqual(called.call_args.args[0]["agent"], {"agent_name": "/root/reviewer"})
        self.assertNotIn("agent", boundary.output[0])  # original replay item is retained

    def test_host_denial_receipt_is_retained(self):
        boundary = self.new_boundary()
        with self.assertRaisesRegex(ProtocolFailure, "completed admission"):
            boundary.receive(call(), lambda _: ToolReceipt("denied", "host:denied"))
        self.assertEqual(next(iter(boundary.host_receipts.values())).status, "denied")

    def test_child_final_without_root_final_is_not_success(self):
        boundary = self.new_boundary()
        event = completed()
        event["response"]["output"] = event["response"]["output"][:1]
        boundary.receive(event, dispatch)
        with self.assertRaisesRegex(ProtocolFailure, "No attributed root final"):
            boundary.final_text()

    def test_missing_or_conflicting_attribution_is_not_default_root(self):
        for event in (
            {"type": "response.output_text.delta", "delta": "whose text?"},
            {"type": "response.output_item.done", "agent": {"agent_name": "/root"},
             "item": {"type": "message", "agent": {"agent_name": "/root/child"}}},
        ):
            with self.subTest(event=event), self.assertRaises(ProtocolFailure):
                self.new_boundary().receive(event, dispatch)

    def test_delta_cannot_override_recorded_child_attribution(self):
        for agent in ("/root", "/rooted", ""):
            boundary = self.new_boundary()
            boundary.receive(call(), dispatch)
            with self.subTest(agent=agent), self.assertRaises(ProtocolFailure):
                boundary.receive({
                    "type": "response.output_text.delta", "output_index": 0,
                    "agent": {"agent_name": agent}, "delta": "child text",
                }, dispatch)
            self.assertEqual(boundary.deltas, [])

    def test_root_final_refusal_is_preserved(self):
        boundary = self.new_boundary()
        event = completed()
        event["response"]["output"][-1]["content"] = [
            {"type": "refusal", "refusal": "Cannot answer this request."},
        ]
        boundary.receive(event, dispatch)
        self.assertEqual(boundary.final_text(), "Cannot answer this request.")

    def test_duplicate_and_cross_agent_wire_ids_do_not_repeat_calls(self):
        for agent in ("/root/reviewer", "/root/checker"):
            boundary = self.new_boundary()
            called = Mock(side_effect=dispatch)
            boundary.receive(call(), called)
            event = call(agent)
            event["output_index"] = 1
            with self.assertRaises(ProtocolFailure):
                boundary.receive(event, called)
            self.assertEqual(called.call_count, 1)

    def test_bad_arguments_and_denied_host_receipts_stop(self):
        for event in (call(proposal="unknown"), call()):
            with self.subTest(event=event), self.assertRaises(ProtocolFailure):
                self.new_boundary().receive(
                    event, lambda _: ToolReceipt("denied", "host:denied")
                )

    def test_mismatched_ack_and_rejected_response_id_stop(self):
        boundary = self.new_boundary()
        boundary.receive(call(), dispatch)
        envelope = boundary.send_next(lambda _: None, lambda _: True)
        with self.assertRaisesRegex(ProtocolFailure, "response_not_found"):
            boundary.receive({
                "type": "response.inject.failed", "response_id": "resp_mock",
                "input": envelope["input"], "error": {"code": "response_not_found"},
            }, dispatch)
        other = self.new_boundary()
        with self.assertRaisesRegex(ProtocolFailure, "Unmatched"):
            other.receive(ack(), dispatch)

    def test_local_sdk_probe_never_creates_a_client_or_calls_create(self):
        client = Mock()
        report = inspect_sdk(client)
        self.assertEqual(report["live_api"], "not called")
        self.assertEqual(report["account_model_availability"], "unknown")
        client.beta.responses.create.assert_not_called()
        client.beta.responses.connect.assert_not_called()

    def test_sdk_floor_with_missing_beta_seam_is_explicit(self):
        # Simulates an SDK lacking the beta module; not a claim that 3.29 lacks it.
        from types import SimpleNamespace

        def missing_beta(name):
            if name == "openai":
                return SimpleNamespace(__version__="explicit-sdk-fixture")
            if name == "openai.resources.beta.responses":
                raise ImportError("explicit fixture")
            raise AssertionError(f"Unexpected import probe: {name}")

        with patch("examples.openai_advanced.hosted_agents.importlib.import_module",
                   side_effect=missing_beta):
            report = inspect_sdk()
        self.assertFalse(report["beta_responses"])
        self.assertIn("no raw bypass", report["reason"])


if __name__ == "__main__":
    unittest.main()