"""Executable demonstrations have deterministic mock outcomes, never live proof."""

import unittest

from examples.openai_advanced.evaluate import evaluate


class EvaluationTests(unittest.TestCase):
    def test_all_three_experiments_have_outcomes_and_rollout_decisions(self):
        report = evaluate()
        self.assertEqual(report["paid_api_calls"], 0)
        self.assertEqual(report["programmatic"]["mock_response_count"], 3)
        self.assertEqual(report["programmatic"]["result"]["shortage_units"], 9)
        self.assertEqual(report["multiplex"]["states"],
                         {"planner": "completed", "research": "completed"})
        self.assertEqual(report["hosted_multi_agent"]["injection_status"], "accepted")
        self.assertIsNone(report["hosted_multi_agent"]["result_applied"])
        self.assertEqual(report["hosted_multi_agent"]["host_tool_calls"], 2)
        self.assertEqual(len(report["hosted_multi_agent"]["injection_frames"]), 2)
        self.assertEqual(report["hosted_multi_agent"]["attributed_agents"],
                         ["/root/alpha_reviewer", "/root/beta_reviewer"])
        self.assertEqual(report["hosted_multi_agent"]["root_final"], "Mock root synthesis.")
        for experiment in ("programmatic", "multiplex", "hosted_multi_agent"):
            self.assertIn("rollout", report[experiment])
        self.assertEqual(len(report["official_sources"]), 4)
        self.assertIn("not tested", report["sdk_floor_3_29"])


if __name__ == "__main__":
    unittest.main()