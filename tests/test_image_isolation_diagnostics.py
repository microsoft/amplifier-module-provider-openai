"""Synthetic stdlib-only controls; no proc probes, product imports or sockets."""
import ast
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).with_name("image_outcome_baseline.py")
SPEC = importlib.util.spec_from_file_location("image_isolation_fixture", SOURCE)
helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helper)
HEADER = "Iface\tDestination\tGateway\tFlags\tRefCnt\tUse\tMetric\tMask\tMTU\tWindow\tIRTT\n"
ROW = "lo\t00000000\t00000000\t0001\t0\t0\t0\t00000000\t0\t0\t0\n"
STATUS = "NoNewPrivs:\t1\n" + "".join(key + ":\t0000000000000000\n" for key in
    ("CapEff", "CapPrm", "CapBnd", "CapAmb", "CapInh"))
DEV = "Inter-| Receive | Transmit\n face |bytes packets |bytes packets\n lo: 0 0 0\n"


def inputs(route=HEADER):
    return {"status": STATUS, "interfaces": DEV, "routes": route}


class DiagnosticControls(unittest.TestCase):
    def test_header_only_does_not_invent_route_rows(self):
        evidence = helper.route_structure(HEADER)
        self.assertEqual(evidence["classification"], "header_only")
        self.assertEqual(evidence["splitlinesCount"], 1)
        self.assertEqual(evidence["structuredRouteRows"], 0)
        self.assertTrue(evidence["originalLineCountPredicate"])
        self.assertFalse(evidence["admissionAuthority"])

    def test_extra_blank_lines_remain_original_refusal(self):
        for text in (HEADER + "\n", HEADER + "  \n", "\n" + HEADER):
            with self.subTest(text=text):
                evidence = helper.route_structure(text)
                self.assertEqual(evidence["blankLines"], 1)
                self.assertFalse(evidence["originalLineCountPredicate"])
                self.assertFalse(helper.original_guard_observations(inputs(text))[-1][2])

    def test_actual_rows_remain_refused_without_addresses_in_evidence(self):
        evidence = helper.route_structure(HEADER + ROW)
        self.assertEqual(evidence["classification"], "route_rows")
        self.assertEqual(evidence["structuredRouteRows"], 1)
        self.assertFalse(helper.original_guard_observations(inputs(HEADER + ROW))[-1][2])
        self.assertNotIn("00000000", json.dumps(evidence))

    def test_empty_and_malformed_are_not_called_empty_valid_tables(self):
        for text, category in (("", "empty"), ("\n", "blank_only"),
                               ("not-a-header\n", "malformed_header"),
                               (HEADER + "bad row\n", "malformed_rows")):
            with self.subTest(category=category):
                observed = helper.route_structure(text)
                self.assertEqual(observed["classification"], category)
                self.assertFalse(observed["admissionAuthority"])
        # Preserve and expose the OLD predicate's limitation, not a new grant.
        self.assertTrue(helper.route_structure("not-a-header\n")["originalLineCountPredicate"])
        self.assertFalse(helper.route_structure("")["originalLineCountPredicate"])

    def test_line_shape_and_row_output_bounds_are_reported(self):
        observed = helper.route_structure(HEADER.replace("\n", "\r\n") + "\n\n")
        self.assertEqual(observed["splitlinesCount"], 3)
        self.assertEqual(observed["blankLines"], 2)
        self.assertEqual(observed["carriageReturns"], 1)
        self.assertEqual(observed["terminalNewlines"], 3)
        observed = helper.route_structure(HEADER + ROW * 40)
        self.assertEqual(observed["structuredRouteRows"], 40)
        self.assertEqual(len(observed["rows"]), 32)
        self.assertTrue(observed["rowsCapped"])

    def test_truncated_unreadable_and_invalid_text_do_not_qualify(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture"
            for payload, limit, reason in ((HEADER.encode() + ROW.encode(), len(HEADER), "truncated"),
                                            (b"\xff", 16, "invalid_utf8")):
                path.write_bytes(payload)
                text, observation = helper.bounded_text(path, limit)
                self.assertIsNone(text)
                self.assertEqual(observation["error"], reason)
                self.assertFalse(observation["complete"])
                self.assertFalse(helper.original_guard_observations(inputs(text))[-1][2])
            text, observation = helper.bounded_text(Path(directory) / "absent", 16)
            self.assertIsNone(text)
            self.assertFalse(observation["complete"])

    def test_diagnostics_written_before_each_original_refusal(self):
        cases = [
            ({**inputs(), "status": STATUS.replace("NoNewPrivs:\t1", "NoNewPrivs:\t0")}, "privilege_regain_not_blocked", 1),
            ({**inputs(), "status": STATUS.replace("CapEff:\t0000000000000000", "CapEff:\t0000000000000001")}, "retained_capability", 2),
            ({**inputs(), "interfaces": DEV + " eth0: 0 0 0\n"}, "unexpected_network_interface", 3),
            (inputs(HEADER + "\n"), "unexpected_network_route", 4),
        ]
        for observed, reason, checks in cases:
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as directory:
                output = Path(directory)
                evidence = {"guards": [], "routes": helper.route_structure(observed["routes"])}
                with self.assertRaisesRegex(RuntimeError, "^" + reason + "$"):
                    helper.enforce_original_isolation(evidence, observed, output)
                retained = json.loads((output / "isolation.json").read_text())
                self.assertEqual(retained["firstRefusal"], reason)
                self.assertEqual(len(retained["guards"]), checks)
                self.assertEqual(retained["guards"][-1]["outcome"], "refused")
                self.assertFalse(retained["guards"][-1]["observedPredicate"])

    def test_optional_write_failure_cannot_replace_original_refusal(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(Path, "write_text", side_effect=OSError):
            bad = {**inputs(), "status": STATUS.replace("NoNewPrivs:\t1", "NoNewPrivs:\t0")}
            with self.assertRaisesRegex(RuntimeError, "^privilege_regain_not_blocked$"):
                helper.enforce_original_isolation({"guards": []}, bad, Path(directory))
            with self.assertRaisesRegex(RuntimeError, "^unexpected_network_route$"):
                helper.enforce_original_isolation({"guards": []}, inputs(HEADER + "\n"), Path(directory))
            with self.assertRaisesRegex(RuntimeError, "^isolation_evidence_write_failed$"):
                helper.enforce_original_isolation({"guards": []}, inputs(), Path(directory))

    def test_main_prints_original_refusal_when_both_receipt_writes_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory).resolve() / "openai-image-outcome"
            output = work / "evidence"
            output.mkdir(parents=True)
            environment = {"GITHUB_ACTIONS": "true", "GITHUB_REPOSITORY": "microsoft/amplifier-module-provider-openai",
                           "GITHUB_REF": "refs/heads/test/image-isolation-diagnostics-run-ad2410",
                           "IMAGE_BASELINE_OFFLINE": "1", "RUNNER_TEMP": directory}
            stream = io.StringIO()
            with patch.dict(helper.os.environ, environment, clear=True), \
                 patch.object(helper.sys, "argv", ["fixture", "--diagnostic-only"]), \
                 patch.object(helper.sys, "flags", SimpleNamespace(isolated=1)), \
                 patch.object(helper.sys, "prefix", str(work / "runtime")), \
                 patch.object(helper, "isolation_observations", return_value=({"guards": []}, inputs(HEADER + "\n"))), \
                 patch.object(helper, "git", side_effect=AssertionError("Must stop before source/provider stage")), \
                 patch.object(Path, "write_text", side_effect=OSError), redirect_stdout(stream):
                self.assertEqual(helper.main(), 1)
            result = json.loads(stream.getvalue())
            self.assertEqual(result["status"], "BLOCKED")
            self.assertEqual(result["reason"], "unexpected_network_route")
            self.assertEqual(result["receiptWriteError"], "result_evidence_write_failed")
            self.assertFalse(result["testsExecuted"])

    def test_original_good_predicates_are_not_safety_qualification(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = {"guards": [], "safetyQualified": False, "productExecutionAuthorized": False}
            helper.enforce_original_isolation(evidence, inputs(), Path(directory))
            self.assertEqual(len(evidence["guards"]), 4)
            self.assertTrue(all(row["observedPredicate"] for row in evidence["guards"]))
            self.assertFalse(evidence["safetyQualified"])
            self.assertFalse(evidence["productExecutionAuthorized"])

    def test_diagnostic_entry_point_cannot_import_or_execute_product_tests(self):
        tree = ast.parse(SOURCE.read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                modules = [alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module or ""]
                self.assertFalse(any(name.startswith(("amplifier_", "pytest", "openai")) for name in modules))
        self.assertNotIn("pytest.main(", SOURCE.read_text())
        self.assertNotIn("importlib.import_module(", SOURCE.read_text())


if __name__ == "__main__":
    unittest.main(verbosity=2)
