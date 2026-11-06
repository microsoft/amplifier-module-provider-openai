"""Offline protocol tests; all response and host tool values are explicit mocks."""

import json
import unittest
from copy import deepcopy

from examples.openai_advanced.programmatic import (
    PROGRAM,
    ProgrammaticRun,
    ProtocolFailure,
    ToolReceipt,
    compare_mock_aggregation,
    encoded,
    read_only_tools,
)


def responses():
    caller = {"type": "program", "caller_id": "program_call"}
    return [
        {
            "status": "completed",
            "output": [
                {"type": "reasoning", "id": "r1", "encrypted_content": "opaque_reasoning"},
                {
                    "type": "program", "id": "p1", "call_id": "program_call",
                    "code": PROGRAM, "fingerprint": "opaque_fingerprint",
                },
                *[
                    {
                        "type": "function_call", "id": name, "call_id": name,
                        "name": name, "arguments": '{"sku":"sku_123"}',
                        "caller": deepcopy(caller),
                    }
                    for name in ("get_inventory", "get_demand")
                ],
            ],
        },
        {
            "status": "completed",
            "output": [
                {
                    "type": "program_output", "id": "po1", "call_id": "program_call",
                    "result": encoded(compare_mock_aggregation()["result"]),
                    "status": "completed",
                }
            ],
        },
        {
            "status": "completed",
            "output": [
                {
                    "type": "message", "id": "m1", "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "Mock shortage: 9 units."}],
                }
            ],
        },
    ]


def host_dispatch(call):
    field = "available_units" if call["name"] == "get_inventory" else "requested_units"
    return ToolReceipt(
        "completed", f"host:{call['call_id']}",
        {"sku": "sku_123", field: 42 if field == "available_units" else 51},
    )


def exercise(script=None, dispatch=host_dispatch, max_responses=4):
    script = iter(script if script is not None else responses())
    run = ProgrammaticRun(
        "explicit-mock-model", True, [{"role": "user", "content": "Compare mock SKU."}]
    )
    text = run.run(lambda _: next(script), dispatch, max_responses=max_responses)
    return run, text


class ProgrammaticTests(unittest.TestCase):
    def test_complete_history_and_caller_preserved(self):
        run, text = exercise()
        self.assertEqual(text, "Mock shortage: 9 units.")
        self.assertEqual(len(run.receipts), 2)
        # Program completion causes a THIRD response, not an early final answer.
        self.assertEqual(len(run.requests), 3)
        history = run.requests[2]["input"]
        self.assertEqual(
            [x.get("type") for x in history],
            [None, "reasoning", "program", "function_call", "function_call",
             "function_call_output", "function_call_output", "program_output"],
        )
        self.assertEqual(history[1]["encrypted_content"], "opaque_reasoning")
        self.assertEqual(history[2]["fingerprint"], "opaque_fingerprint")
        self.assertEqual(history[5]["caller"], history[3]["caller"])
        self.assertEqual(history[6]["caller"], history[4]["caller"])
        self.assertEqual(json.loads(history[5]["output"])["available_units"], 42)
        self.assertEqual(history[7], responses()[1]["output"][0])

    def test_tools_retain_programmatic_metadata(self):
        tools = read_only_tools()
        self.assertEqual(tools[2], {"type": "programmatic_tool_calling"})
        for tool in tools[:2]:
            self.assertEqual(tool["allowed_callers"], ["programmatic"])
            self.assertEqual(tool["output_schema"]["type"], "object")
            self.assertNotIn("async", tool)

    def test_no_generated_javascript_is_executed_locally(self):
        script = responses()
        script[0]["output"][1]["code"] = "throw new Error('must not run locally')"
        run, text = exercise(script)
        self.assertIn("shortage", text)
        self.assertIn("must not run locally", run.history[2]["code"])

    def test_program_failure_retains_client_receipts_and_never_replays(self):
        script = responses()
        script[1]["output"][0]["status"] = "incomplete"
        calls = []
        run = ProgrammaticRun("mock", True, [])
        it = iter(script)
        with self.assertRaisesRegex(ProtocolFailure, "Program failed"):
            run.run(lambda _: next(it), lambda call: calls.append(call) or host_dispatch(call),
                    max_responses=4)
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(run.receipts), 2)
        self.assertEqual(run.history[-1]["status"], "incomplete")
        with self.assertRaisesRegex(ProtocolFailure, "reconcile"):
            run.run(lambda _: next(it), host_dispatch, max_responses=4)
        self.assertEqual(len(calls), 2)

    def test_unknown_host_call_is_recorded_before_dispatch(self):
        run = ProgrammaticRun("mock", True, [])
        it = iter(responses())

        def uncertain(_):
            raise TimeoutError("remote result may exist")

        with self.assertRaisesRegex(ProtocolFailure, "unknown"):
            run.run(lambda _: next(it), uncertain, max_responses=4)
        self.assertEqual(run.receipts["get_inventory"].status, "unknown")
        self.assertNotIn("get_demand", run.receipts)

    def test_denied_call_is_not_result_or_retried(self):
        with self.assertRaisesRegex(ProtocolFailure, "denied"):
            exercise(dispatch=lambda _: ToolReceipt("denied", "host:denied"))

    def test_entire_batch_is_validated_before_execution(self):
        for mutation in ("unknown_tool", "wrong_program", "async", "duplicate_id", "bad_args"):
            with self.subTest(mutation=mutation):
                script = responses()
                second = script[0]["output"][3]
                if mutation == "unknown_tool":
                    second["name"] = "delete_inventory"
                elif mutation == "wrong_program":
                    second["caller"]["caller_id"] = "other_program"
                elif mutation == "async":
                    second["async"] = True
                elif mutation == "duplicate_id":
                    second["call_id"] = "get_inventory"
                else:
                    second["arguments"] = '{"sku":"sku_123","unapproved":true}'
                called = []
                with self.assertRaises(ProtocolFailure):
                    exercise(script, dispatch=lambda call, called=called: called.append(call))
                self.assertEqual(called, [])

    def test_repeated_call_identity_is_not_executed_twice(self):
        script = responses()
        script.insert(1, deepcopy(script[0]))
        called = []
        with self.assertRaisesRegex(ProtocolFailure, "Repeated call"):
            exercise(script, dispatch=lambda call: called.append(call) or host_dispatch(call))
        self.assertEqual(len(called), 2)

    def test_historical_duplicate_rejects_whole_batch_before_any_new_call(self):
        script = responses()
        new_call = deepcopy(script[0]["output"][2])
        new_call["call_id"] = "new_call"
        script.insert(1, {
            "status": "completed",
            "output": [new_call, deepcopy(script[0]["output"][2])],
        })
        called = []
        with self.assertRaisesRegex(ProtocolFailure, "Repeated call"):
            exercise(script, dispatch=lambda item: called.append(item) or host_dispatch(item))
        self.assertEqual(len(called), 2)

    def test_commentary_only_continues_and_is_not_merged_with_final_answer(self):
        script = responses()
        commentary = {
            "type": "message", "role": "assistant", "status": "completed",
            "phase": "commentary", "content": [{"type": "output_text", "text": "Working."}],
        }
        script[1]["output"].append(commentary)
        script[2]["output"].insert(0, deepcopy(commentary))
        script[2]["output"][-1]["phase"] = "final_answer"
        run, text = exercise(script)
        self.assertEqual(text, "Mock shortage: 9 units.")
        self.assertEqual(len(run.requests), 3)
        self.assertEqual(run.requests[2]["input"][-1], commentary)

    def test_incomplete_response_dispatches_nothing(self):
        script = responses()
        script[0]["status"] = "incomplete"
        calls = []
        with self.assertRaisesRegex(ProtocolFailure, "incomplete"):
            exercise(script, dispatch=lambda call: calls.append(call))
        self.assertEqual(calls, [])

    def test_result_schema_rejects_wrong_sku_nonfinite_and_bool(self):
        for result in (
            {"sku": "wrong", "available_units": 42},
            {"sku": "sku_123", "available_units": True},
            {"sku": "sku_123", "available_units": float("nan")},
            {"sku": "sku_123", "available_units": 42, "extra": "ignored?"},
        ):
            with self.subTest(result=result), self.assertRaises(ProtocolFailure):
                exercise(dispatch=lambda _, result=result: ToolReceipt("completed", "host:r", result))

    def test_capability_unknown_fails_before_any_request(self):
        for capability in (False, None):
            with self.assertRaisesRegex(ProtocolFailure, "unknown"):
                ProgrammaticRun("selected-model", capability, [])

    def test_program_output_alone_needs_final_message(self):
        with self.assertRaisesRegex(ProtocolFailure, "budget exhausted"):
            exercise(max_responses=2)

    def test_refusal_is_a_final_message(self):
        script = responses()
        script[2]["output"][0]["content"] = [{"type": "refusal", "refusal": "Cannot answer."}]
        self.assertEqual(exercise(script)[1], "Cannot answer.")

    def test_fixture_comparison_does_not_claim_an_efficiency_win(self):
        report = compare_mock_aggregation()
        self.assertEqual(report["result"]["shortage_units"], 9)
        self.assertEqual(report["boundary_counts"]["direct_parallel"],
                         report["boundary_counts"]["native_programmatic"])
        self.assertIn("no demonstrated", report["interpretation"])
        self.assertEqual(report["fixture"], "explicit mock, no API requests")


if __name__ == "__main__":
    unittest.main()