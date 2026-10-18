"""Synthetic stdlib-only controls; no proc probes, product imports or sockets."""
import ast
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import socket
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

SOURCE = Path(__file__).with_name("image_outcome_baseline.py")
SPEC = importlib.util.spec_from_file_location("image_isolation_fixture", SOURCE)
helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helper)
HEADER = "Iface\tDestination\tGateway\tFlags\tRefCnt\tUse\tMetric\tMask\tMTU\tWindow\tIRTT\n"
ROW = "lo\t00000000\t00000000\t0001\t0\t0\t0\t00000000\t0\t0\t0\n"
STATUS = "NoNewPrivs:\t1\n" + "".join(key + ":\t0000000000000000\n" for key in
    ("CapEff", "CapPrm", "CapBnd", "CapAmb", "CapInh"))
DEV = ("Inter-| Receive | Transmit\n"
       " face |bytes packets errs drop fifo frame compressed multicast|bytes packets errs drop fifo colls carrier compressed\n"
       " lo: " + " ".join(["0"] * 16) + "\n")


def inputs(route=HEADER):
    return {"status": STATUS, "interfaces": DEV, "routes": route}


def observations(raw):
    """Synthetic reproduction of complete reads and the observed down-loopback."""
    evidence = {"guards": [], "safetyQualified": False, "productAcceptance": False}
    for key, limit in (("status", 32768), ("interfaces", 16384), ("routes", 16384)):
        evidence[key] = {"complete": True, "bytesObserved": len(raw[key].encode()),
                         "limit": limit, "error": None}
    evidence["interfaces"].update(names=["lo"], count=1, linkFlags=[
        {"interface": "lo", "status": "observed", "flags": 8, "up": False, "loopback": True}])
    evidence["namespace"] = {"current": "net:[200]", "retainedParent": "net:[100]", "comparison": "different",
                             "parentRead": {"complete": True, "error": None, "identityBound": True,
                                            "limit": 1024, "bytesObserved": 134}}
    return evidence


def case_fixture():
    """Synthetic one-to-one evidence for the immutable product test identities."""
    nodes = list(helper.EXPECTED_NODE_IDS)
    reports = [{"nodeid": node, "when": phase, "outcome": "passed"}
               for node in nodes for phase in ("setup", "call", "teardown")]
    tree = ET.Element("testsuites")
    suite = ET.SubElement(tree, "testsuite", tests="32", failures="0", errors="0", skipped="0")
    for node in nodes:
        ET.SubElement(suite, "testcase", classname="tests.test_image_operation",
                      name=node.split("::", 1)[1])
    return nodes, reports, tree


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
                           "GITHUB_REF": "refs/heads/test/image-empty-route-baseline-run-ad2410",
                           "IMAGE_BASELINE_OFFLINE": "1", "RUNNER_TEMP": directory}
            stream = io.StringIO()
            with patch.dict(helper.os.environ, environment, clear=True), \
                 patch.object(helper.sys, "argv", ["fixture"]), \
                 patch.object(helper.sys, "flags", SimpleNamespace(isolated=1)), \
                 patch.object(helper.sys, "prefix", str(work / "runtime")), \
                 patch.object(helper, "isolation_observations", return_value=(observations(inputs(HEADER + "\n")), inputs(HEADER + "\n"))), \
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

    def test_baseline_entry_point_guards_and_audits_before_product_imports(self):
        source = SOURCE.read_text()
        tree = ast.parse(source)
        for node in tree.body:  # Importing the helper to run controls is stdlib-only.
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                modules = [alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module or ""]
                self.assertFalse(any(name.startswith(("amplifier_", "pytest", "openai")) for name in modules))
        main = source[source.index("def main():"):]
        self.assertLess(main.index("enforce_isolation("), main.index("installed_payload_mismatch"))
        self.assertLess(main.index("installed_payload_mismatch"), main.index("sys.addaudithook(audit)"))
        self.assertLess(main.index("sys.addaudithook(audit)"), main.index("importlib.import_module(name)"))
        self.assertLess(main.index("importlib.import_module(name)"), main.index("pytest.main("))
        self.assertEqual(helper.BASELINE, "32db886dd681a03f9fd7c1fdd2a4093a60d054c5")
        self.assertEqual(helper.TEST_BLOB, "787e1308c9d17f5f9f4bae3a278ff2acc55507b6")
        self.assertEqual(helper.TEST.name, "test_image_operation.py")
        self.assertNotIn("DIAGNOSTIC_ONLY", main)

    def test_corrected_guard_accepts_complete_observed_zero_byte_input(self):
        raw = inputs("")
        evidence = observations(raw)
        self.assertFalse(helper.original_guard_observations(raw)[-1][2])
        with tempfile.TemporaryDirectory() as directory:
            helper.enforce_isolation(evidence, raw, Path(directory))
            retained = json.loads((Path(directory) / "isolation.json").read_text())
        self.assertTrue(all(row["observedPredicate"] for row in retained["guards"]))
        self.assertEqual(retained["routes"]["bytesObserved"], 0)
        self.assertFalse(retained["originalPredicates"][-1]["observedPredicate"])
        self.assertFalse(retained["originalPredicates"][-1]["admissionAuthority"])
        self.assertFalse(retained["productAcceptance"])

    def test_corrected_guard_accepts_only_strict_header_only_forms(self):
        for text in (HEADER, HEADER.rstrip("\n"), HEADER.replace("\n", "\r\n"),
                     " ".join(helper.ROUTE_HEADER) + "\n"):
            with self.subTest(text=text):
                raw = inputs(text)
                self.assertTrue(all(row[2] for row in helper.corrected_guard_observations(observations(raw), raw)))

    def test_corrected_guard_rejects_blank_malformed_extra_and_actual_rows(self):
        for text in ("\n", " ", "\t\r\n", "malformed\n", HEADER + "\n", HEADER + " \n",
                     "\n" + HEADER, HEADER + ROW, ROW, HEADER + "bad row\n", HEADER + HEADER):
            with self.subTest(text=text), tempfile.TemporaryDirectory() as directory:
                raw = inputs(text)
                with self.assertRaisesRegex(RuntimeError, "^unexpected_network_route$"):
                    helper.enforce_isolation(observations(raw), raw, Path(directory))

    def test_corrected_guard_refuses_any_incomplete_or_inconsistent_read(self):
        for key in ("status", "interfaces", "routes"):
            for change in ({"complete": False}, {"error": "PermissionError"}, {"bytesObserved": -1},
                           {"bytesObserved": True}, {"limit": 100000}, {"bytesObserved": 16385}):
                with self.subTest(key=key, change=change):
                    raw = inputs("")
                    evidence = observations(raw)
                    evidence[key].update(change)
                    self.assertFalse(helper.corrected_guard_observations(evidence, raw)[0][2])
            raw = inputs("")
            evidence = observations(raw)
            raw[key] = None  # Never use the remaining complete-looking prefix/metadata.
            self.assertFalse(helper.corrected_guard_observations(evidence, raw)[0][2])

    def test_corrected_guard_requires_distinct_complete_bound_namespace(self):
        for changes in ({"current": None}, {"retainedParent": None}, {"current": "net:[100]"},
                        {"comparison": "unknown"}, {"comparison": "same"}, {"current": "malformed"},
                        {"parentRead": {"complete": False}},
                        {"parentRead": {"complete": True, "error": "invalid_or_unbound_identity", "identityBound": True, "limit": 1024, "bytesObserved": 134}},
                        {"parentRead": {"complete": True, "error": None, "identityBound": True, "limit": 1024, "bytesObserved": 0}},
                        {"parentRead": {"complete": True, "error": None, "identityBound": False, "limit": 1024, "bytesObserved": 134}}):
            with self.subTest(changes=changes), tempfile.TemporaryDirectory() as directory:
                raw = inputs("")
                evidence = observations(raw)
                evidence["namespace"].update(changes)
                with self.assertRaisesRegex(RuntimeError, "^network_namespace_not_separate$"):
                    helper.enforce_isolation(evidence, raw, Path(directory))

    def test_parent_namespace_binding_refuses_missing_or_mismatched_run_identity(self):
        raw = inputs("")
        source = "a" * 40
        record = {"networkNamespace": "net:[100]", "source": source, "runId": "123", "attempt": "1"}
        good_env = {"GITHUB_SHA": source, "GITHUB_RUN_ID": "123", "GITHUB_RUN_ATTEMPT": "1"}
        for environment, accepted in ((good_env, True), ({}, False),
                                      ({**good_env, "GITHUB_RUN_ID": "124"}, False),
                                      ({**good_env, "GITHUB_SHA": "bad"}, False)):
            with self.subTest(environment=environment):
                serialized = json.dumps(record)
                reads = [(raw[key], observations(raw)[key]) for key in ("status", "interfaces", "routes")]
                reads.append((serialized, {"complete": True, "error": None, "limit": 1024,
                                           "bytesObserved": len(serialized)}))
                with patch.dict(helper.os.environ, environment, clear=True), \
                     patch.object(helper, "bounded_text", side_effect=reads), \
                     patch.object(helper, "namespace_identity", return_value="net:[200]"), \
                     patch.object(socket, "socket", side_effect=OSError("synthetic unavailable flags")):
                    evidence, _ = helper.isolation_observations(Path("synthetic-work"))
                self.assertEqual(evidence["namespace"]["parentRead"]["identityBound"], accepted)
                self.assertEqual(evidence["namespace"]["retainedParent"], "net:[100]" if accepted else None)
                self.assertEqual(evidence["namespace"]["comparison"], "different" if accepted else "unknown")
                self.assertEqual(evidence["interfaces"]["linkFlags"][0]["status"], "unavailable")

    def test_corrected_guard_refuses_nonloopback_or_unobserved_up_flags(self):
        for changes in ({"status": "unavailable"}, {"flags": 9, "up": True}, {"flags": 0, "loopback": False},
                        {"flags": -8}, {"flags": True}, {"flags": "8"}, {"up": True},
                        {"loopback": False}, {"interface": "eth0"}):
            with self.subTest(changes=changes), tempfile.TemporaryDirectory() as directory:
                raw = inputs("")
                evidence = observations(raw)
                evidence["interfaces"]["linkFlags"][0].update(changes)
                with self.assertRaisesRegex(RuntimeError, "^loopback_not_down$"):
                    helper.enforce_isolation(evidence, raw, Path(directory))
        raw = {**inputs(""), "interfaces": DEV + " eth0: 0 0 0\n"}
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "^unexpected_network_interface$"):
                helper.enforce_isolation(observations(raw), raw, Path(directory))

    def test_corrected_empty_route_never_bypasses_privilege_denials(self):
        for key in ("NoNewPrivs", "CapEff", "CapPrm", "CapBnd", "CapAmb", "CapInh"):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                raw = inputs("")
                raw["status"] = (STATUS.replace("NoNewPrivs:\t1", "NoNewPrivs:\t0") if key == "NoNewPrivs" else
                                 STATUS.replace(key + ":\t0000000000000000", key + ":\t0000000000000001"))
                reason = "privilege_regain_not_blocked" if key == "NoNewPrivs" else "retained_capability"
                with self.assertRaisesRegex(RuntimeError, "^" + reason + "$"):
                    helper.enforce_isolation(observations(raw), raw, Path(directory))

    def test_corrected_guard_refuses_duplicate_and_malformed_status_fields(self):
        for key in ("NoNewPrivs", "CapEff", "CapPrm", "CapBnd", "CapAmb", "CapInh"):
            value = "1" if key == "NoNewPrivs" else "0000000000000000"
            line = key + ":\t" + value + "\n"
            variants = [
                STATUS.replace(line, ""),  # Missing is not a zero capability.
                STATUS + line,  # Even duplicate equal observations are ambiguous.
                key + ":\t0000000000000001\n" + STATUS,
                STATUS.replace(line, key + ":\t-0\n"),
                STATUS.replace(line, key + ":\t+0\n"),
                STATUS.replace(line, key + ":\t0x0000000000000000\n"),
            ]
            for text in variants:
                with self.subTest(key=key, text=text), tempfile.TemporaryDirectory() as directory:
                    raw = {**inputs(""), "status": text}
                    reason = "privilege_regain_not_blocked" if key == "NoNewPrivs" else "retained_capability"
                    with self.assertRaisesRegex(RuntimeError, "^" + reason + "$"):
                        helper.enforce_isolation(observations(raw), raw, Path(directory))

    def test_corrected_guard_refuses_hidden_duplicate_or_malformed_interface_rows(self):
        lines = DEV.splitlines()
        for text in ("\n".join(["eth0: 0", *lines[1:]]) + "\n",
                     "\n".join([lines[0], "eth0: 0", lines[2]]) + "\n",
                     DEV + lines[2] + "\n", DEV + "\n", DEV + " eth0: 0\n",
                     DEV.replace("lo:", "eth0:"), DEV.replace("Receive", "Unknown"),
                     "\n".join([*lines[:2], " lo: 0 0 0"]) + "\n",
                     DEV.replace(" lo: 0", " lo: -0", 1)):
            with self.subTest(text=text), tempfile.TemporaryDirectory() as directory:
                raw = {**inputs(""), "interfaces": text}
                with self.assertRaisesRegex(RuntimeError, "^unexpected_network_interface$"):
                    helper.enforce_isolation(observations(raw), raw, Path(directory))

    def test_corrected_guard_preserves_refusal_when_evidence_write_fails(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(Path, "write_text", side_effect=OSError):
            raw = inputs(HEADER + ROW)
            with self.assertRaisesRegex(RuntimeError, "^unexpected_network_route$"):
                helper.enforce_isolation(observations(raw), raw, Path(directory))
            raw = inputs("")
            with self.assertRaisesRegex(RuntimeError, "^isolation_evidence_write_failed$"):
                helper.enforce_isolation(observations(raw), raw, Path(directory))

    def test_refused_main_never_reaches_source_or_provider_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory).resolve() / "openai-image-outcome"
            output = work / "evidence"
            output.mkdir(parents=True)
            raw = inputs(HEADER + ROW)
            environment = {"GITHUB_ACTIONS": "true", "GITHUB_REPOSITORY": "microsoft/amplifier-module-provider-openai",
                           "GITHUB_REF": "refs/heads/test/image-empty-route-baseline-run-ad2410",
                           "IMAGE_BASELINE_OFFLINE": "1", "RUNNER_TEMP": directory}
            with patch.dict(helper.os.environ, environment, clear=True), \
                 patch.object(helper.sys, "argv", ["fixture"]), \
                 patch.object(helper.sys, "flags", SimpleNamespace(isolated=1)), \
                 patch.object(helper.sys, "prefix", str(work / "runtime")), \
                 patch.object(helper, "isolation_observations", return_value=(observations(raw), raw)), \
                 patch.object(helper, "git", side_effect=AssertionError("No source/provider stage")), \
                 patch.object(helper.importlib, "import_module", side_effect=AssertionError("No product import")), \
                 redirect_stdout(io.StringIO()):
                self.assertEqual(helper.main(), 1)
            result = json.loads((output / "result.json").read_text())
            self.assertEqual(result["reason"], "unexpected_network_route")
            self.assertFalse(result["productImports"])
            self.assertFalse(result["testsExecuted"])
            self.assertIsNone(result["networkAttempts"])

    def test_fixed_case_identities_match_unchanged_literal_parametrization(self):
        # Independently expand this fixed source's literal decorators, without
        # importing pytest, collecting tests, or running any provider fixture.
        expected = []
        for node in ast.parse(helper.TEST.read_text()).body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or not node.name.startswith("test_"):
                continue
            ids = [""]
            for decorator in reversed(node.decorator_list):
                if isinstance(decorator, ast.Call) and ast.unparse(decorator.func) == "pytest.mark.parametrize":
                    self.assertEqual(len(decorator.args), 2)
                    self.assertFalse(decorator.keywords)
                    values = ast.literal_eval(decorator.args[1])
                    self.assertTrue(all(isinstance(value, (bool, str)) for value in values))
                    ids = [prefix + ("-" if prefix else "") + str(value) for prefix in ids for value in values]
            expected.extend("tests/test_image_operation.py::" + node.name + ("[" + value + "]" if value else "")
                            for value in ids)
        self.assertEqual(len(expected), 32)
        self.assertEqual(len(set(expected)), 32)
        self.assertEqual(tuple(expected), helper.EXPECTED_NODE_IDS)

    def test_exact_evidence_accepts_all_32_unique_cases_in_any_report_order(self):
        nodes, reports, tree = case_fixture()
        result = helper.exact_case_evidence(list(reversed(nodes)), list(reversed(reports)), tree)
        self.assertTrue(result["allRequiredExecuted"])
        self.assertEqual(result["expectedCount"], 32)
        for phase in ("collection", "calls", "junit"):
            self.assertEqual(result[phase]["count"], 32)
            self.assertEqual(result[phase]["unique"], 32)
            self.assertTrue(result[phase]["exact"])
        self.assertTrue(result["junit"]["countsMatch"])
        self.assertEqual(result["counts"], {"tests": 32, "failures": 0, "errors": 0, "skipped": 0})

    def test_missing_parameter_variant_cannot_pass_a_complete_function_set(self):
        nodes, reports, tree = case_fixture()
        missing = nodes.pop(0)  # Other variants of this same function remain.
        reports = [report for report in reports if report["nodeid"] != missing]
        suite = tree.find("testsuite")
        suite.remove(suite[0])
        suite.set("tests", "31")
        result = helper.exact_case_evidence(nodes, reports, tree)
        self.assertFalse(result["allRequiredExecuted"])
        for phase in ("collection", "calls", "junit"):
            self.assertEqual(result[phase]["count"], 31)
            self.assertEqual(result[phase]["missing"], [missing])
        self.assertEqual(result["counts"]["tests"], 31)

    def test_collected_duplicate_or_unexpected_variant_cannot_replace_expected_case(self):
        for replacement in (helper.EXPECTED_NODE_IDS[1], "tests/test_image_operation.py::test_legacy_images_also_sends_once[unexpected]"):
            with self.subTest(replacement=replacement):
                nodes, reports, tree = case_fixture()
                missing = nodes[0]
                nodes[0] = replacement
                result = helper.exact_case_evidence(nodes, reports, tree)
                self.assertFalse(result["allRequiredExecuted"])
                self.assertEqual(result["collection"]["count"], 32)
                self.assertEqual(result["collection"]["missing"], [missing])
                self.assertTrue(result["collection"]["duplicate"] or result["collection"]["unexpected"])

    def test_call_reports_require_exactly_one_call_per_expected_identity(self):
        for change in ("missing", "duplicate", "unexpected", "unexpected_setup"):
            with self.subTest(change=change):
                nodes, reports, tree = case_fixture()
                call = next(report for report in reports if report["when"] == "call")
                if change == "missing":
                    reports.remove(call)
                elif change == "duplicate":
                    reports.append(dict(call))
                else:
                    reports.append({**call, "nodeid": "tests/test_image_operation.py::unexpected",
                                    "when": "call" if change == "unexpected" else "setup"})
                result = helper.exact_case_evidence(nodes, reports, tree)
                self.assertFalse(result["allRequiredExecuted"])
                if change == "missing":
                    self.assertEqual(result["calls"]["missing"], [call["nodeid"]])
                elif change == "duplicate":
                    self.assertEqual(result["calls"]["duplicate"], [call["nodeid"]])
                else:
                    self.assertEqual(len(result["unexpectedReports"]), 1)

    def test_junit_identities_must_match_every_expected_parameter_once(self):
        for change in ("missing", "duplicate", "unexpected", "wrong_class", "missing_name"):
            with self.subTest(change=change):
                nodes, reports, tree = case_fixture()
                suite = tree.find("testsuite")
                if change == "missing":
                    suite.remove(suite[0])
                    suite.set("tests", "31")
                elif change == "duplicate":
                    suite[0].set("name", suite[1].get("name"))
                elif change == "unexpected":
                    suite[0].set("name", "test_legacy_images_also_sends_once[unexpected]")
                elif change == "wrong_class":
                    suite[0].set("classname", "other.test_image_operation")
                else:
                    del suite[0].attrib["name"]
                result = helper.exact_case_evidence(nodes, reports, tree)
                self.assertFalse(result["allRequiredExecuted"])
                self.assertTrue(result["junit"]["missing"])
                if change in {"wrong_class", "missing_name"}:
                    self.assertEqual(len(result["junit"]["unmapped"]), 1)

    def test_junit_declared_counts_must_equal_actual_case_and_outcome_counts(self):
        for key in ("tests", "failures", "errors", "skipped"):
            for value in ("", "invalid", "1", None):
                with self.subTest(key=key, value=value):
                    nodes, reports, tree = case_fixture()
                    suite = tree.find("testsuite")
                    if value is None:
                        del suite.attrib[key]
                    else:
                        suite.set(key, value)
                    result = helper.exact_case_evidence(nodes, reports, tree)
                    self.assertFalse(result["allRequiredExecuted"])
                    self.assertFalse(result["junit"]["countsMatch"])
                    self.assertEqual(result["counts"]["tests"], 32)  # Preserve observed, not declared totals.

    def test_product_failure_keeps_complete_evidence_and_nonzero_failure_count(self):
        nodes, reports, tree = case_fixture()
        next(report for report in reports if report["when"] == "call")["outcome"] = "failed"
        suite = tree.find("testsuite")
        suite.set("failures", "1")
        ET.SubElement(suite[0], "failure", message="synthetic expected product red").text = "retained assertion"
        result = helper.exact_case_evidence(nodes, reports, tree)
        self.assertTrue(result["allRequiredExecuted"])  # Executed is not passed.
        self.assertEqual(result["counts"], {"tests": 32, "failures": 1, "errors": 0, "skipped": 0})
        self.assertEqual(len(result["junit"]["identities"]), 32)
        self.assertIn("retained assertion", ET.tostring(tree, encoding="unicode"))
        main = SOURCE.read_text().split("def main():", 1)[1]
        self.assertIn('not any(counts[key] for key in ("failures", "errors", "skipped"))', main)
        self.assertIn('exit_code = 0 if result["status"] == "PASS" else (code or 1)', main)

    def test_junit_incomplete_empty_or_multiple_suites_never_qualify(self):
        for change in ("empty", "multiple", "nested"):
            with self.subTest(change=change):
                nodes, reports, tree = case_fixture()
                if change == "empty":
                    tree = ET.Element("testsuites")
                elif change == "multiple":
                    ET.SubElement(tree, "testsuite", tests="0", failures="0", errors="0", skipped="0")
                else:
                    suite = tree.find("testsuite")
                    row = suite[0]
                    suite.remove(row)
                    ET.SubElement(suite, "unexpected-wrapper").append(row)
                result = helper.exact_case_evidence(nodes, reports, tree)
                self.assertFalse(result["allRequiredExecuted"])
                self.assertFalse(result["junit"]["countsMatch"])

    def test_junit_skips_and_errors_remain_visible_without_becoming_passes(self):
        for tag, key in (("skipped", "skipped"), ("error", "errors")):
            with self.subTest(tag=tag):
                nodes, reports, tree = case_fixture()
                suite = tree.find("testsuite")
                suite.set(key, "1")
                ET.SubElement(suite[0], tag, message="synthetic non-pass")
                result = helper.exact_case_evidence(nodes, reports, tree)
                self.assertEqual(result["counts"][key], 1)
                self.assertEqual(result["counts"]["tests"], 32)


if __name__ == "__main__":
    unittest.main(verbosity=2)
