"""Installed-wheel image baseline with an observed-empty-route guard correction.

Run only in the repository's admitted hosted execution branch. Product/test bytes
remain those of BASELINE. Complete empty or recognized header-only route input
requires separate namespace, down loopback and zero-privilege evidence as well.
The original line-count predicate remains diagnostic, never admission authority.
"""
from __future__ import annotations

from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import importlib
import importlib.metadata as metadata
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile

BASELINE = "32db886dd681a03f9fd7c1fdd2a4093a60d054c5"
FOUNDATION = "8d26eceb9b7cafe6e93346f8f4294596bd45dcd1"
PRODUCT_TREE = "bc160ea4272efef1a4870724769c369e3793be94"
TEST_BLOB = "787e1308c9d17f5f9f4bae3a278ff2acc55507b6"
ROOT = Path(__file__).resolve().parents[1]
PRODUCT = "amplifier_module_provider_openai"
TEST = ROOT / "tests/test_image_operation.py"
ROUTE_HEADER = "Iface Destination Gateway Flags RefCnt Use Metric Mask MTU Window IRTT".split()
# Fixed identities for TEST_BLOB, including pytest's literal parameter IDs.
# Do not derive the expected set from collection: a deselected variant is missing
# evidence, even when every function name still appears in the remaining cases.
EXPECTED_NODE_IDS = tuple("tests/test_image_operation.py::" + name for name in (
    "test_legacy_images_also_sends_once[server]",
    "test_legacy_images_also_sends_once[disconnect]",
    "test_legacy_images_also_sends_once[redirect]",
    "test_responses_uses_distinct_models_bound_account_and_final_image[False]",
    "test_responses_uses_distinct_models_bound_account_and_final_image[True]",
    "test_sdk_500_is_one_request_and_cannot_reuse_plan",
    "test_changed_request_plan_or_binding_never_sends[prompt]",
    "test_changed_request_plan_or_binding_never_sends[model]",
    "test_changed_request_plan_or_binding_never_sends[plan]",
    "test_changed_request_plan_or_binding_never_sends[account]",
    "test_changed_request_plan_or_binding_never_sends[endpoint]",
    "test_changed_request_plan_or_binding_never_sends[project]",
    "test_changed_request_plan_or_binding_never_sends[organization]",
    "test_changed_request_plan_or_binding_never_sends[headers]",
    "test_missing_native_capability_controls_and_endpoint_refuse_without_call",
    "test_cancelled_send_stays_consumed",
    "test_redirect_does_not_repeat_generation_or_change_bound_endpoint",
    "test_parse_failure_retains_paid_image_and_safe_provenance_once[oversized_text]",
    "test_parse_failure_retains_paid_image_and_safe_provenance_once[text]",
    "test_parse_failure_retains_paid_image_and_safe_provenance_once[output_count]",
    "test_parse_failure_retains_paid_image_and_safe_provenance_once[usage]",
    "test_parse_failure_retains_paid_image_and_safe_provenance_once[model]",
    "test_requested_geometry_failure_keeps_safe_image_evidence_once[False-responses]",
    "test_requested_geometry_failure_keeps_safe_image_evidence_once[False-images]",
    "test_requested_geometry_failure_keeps_safe_image_evidence_once[True-responses]",
    "test_requested_geometry_failure_keeps_safe_image_evidence_once[True-images]",
    "test_incomplete_response_retains_completed_image_without_continuation",
    "test_image_disconnect_retains_uncertain_dispatch_without_replay[responses]",
    "test_image_disconnect_retains_uncertain_dispatch_without_replay[images]",
    "test_nonterminal_image_response_remains_unknown_but_received[queued]",
    "test_nonterminal_image_response_remains_unknown_but_received[in_progress]",
    "test_nonterminal_image_response_remains_unknown_but_received[unrecognized-state]",
))


def require(condition, code):
    if not condition:
        raise RuntimeError(code)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def clean_text(text, work):
    for value, replacement in ((str(ROOT), "[CHECKOUT]"), (str(work), "[WORK]"),
                               (os.environ.get("HOME", ""), "[HOME]")):
        if value:
            text = text.replace(value, replacement)
    return re.sub(r"/home/[^/\s]+", "[HOME]", text)


def route_structure(text):
    """Describe format, not route addresses; never use this to grant admission."""
    lines = text.splitlines()  # Deliberately no strip/filter: preserve the refusal.
    recognized = bool(lines) and lines[0].split() == ROUTE_HEADER
    rows = []
    for index, line in enumerate(lines[1:] if recognized else lines, start=1 if recognized else 0):
        fields = line.split()
        valid = (len(fields) == 11 and re.fullmatch(r"[A-Za-z0-9_.:-]{1,15}", fields[0])
                 and all(re.fullmatch(r"[0-9A-Fa-f]{8}", fields[i]) for i in (1, 2, 7))
                 and re.fullmatch(r"[0-9A-Fa-f]{1,8}", fields[3])
                 and all(re.fullmatch(r"[0-9]+", fields[i]) for i in (4, 5, 6, 8, 9, 10)))
        rows.append({"index": index, "fields": len(fields),
                     "kind": "blank" if not fields else "route" if valid else "malformed"})
    route_count = sum(row["kind"] == "route" for row in rows)
    malformed = sum(row["kind"] == "malformed" for row in rows)
    classification = ("empty" if not text else "blank_only" if not text.strip() else
                      "malformed_header" if not recognized else "malformed_rows" if malformed else
                      "route_rows" if route_count else "header_only")
    return {"classification": classification, "header": "recognized" if recognized else "unrecognized_or_absent",
            "splitlinesCount": len(lines), "blankLines": sum(not line.strip() for line in lines),
            "terminalNewlines": len(text) - len(text.rstrip("\n")),
            "carriageReturns": text.count("\r"), "structuredRouteRows": route_count,
            "malformedRows": malformed, "rows": rows[:32], "rowsCapped": len(rows) > 32,
            "originalLineCountPredicate": len(lines) == 1, "admissionAuthority": False}


def bounded_text(path, limit):
    try:
        with path.open("rb") as stream:
            data = stream.read(limit + 1)
        complete = len(data) <= limit
        try:
            text = data[:limit].decode("utf-8")
        except UnicodeDecodeError:
            return None, {"complete": False, "error": "invalid_utf8", "bytesObserved": len(data)}
        return text if complete else None, {
            "complete": complete, "bytesObserved": len(data), "limit": limit,
            "error": None if complete else "truncated"}
    except OSError as error:
        return None, {"complete": False, "error": type(error).__name__, "errno": error.errno}


def namespace_identity():
    try:
        value = os.readlink("/proc/self/ns/net")
        return value if re.fullmatch(r"net:\[[0-9]{1,20}\]", value) else None
    except OSError:
        return None


def isolation_observations(work):
    """Bounded current-process probes only; optional probe failures stay explicit."""
    evidence = {"schema": "image-isolation-observation.v1", "guards": [],
                "safetyQualified": False, "productAcceptance": False}
    inputs = {}
    for key, path, limit in (("status", "/proc/self/status", 32768),
                             ("interfaces", "/proc/net/dev", 16384),
                             ("routes", "/proc/net/route", 16384)):
        inputs[key], evidence[key] = bounded_text(Path(path), limit)
    fields = dict(line.split(":", 1) for line in (inputs["status"] or "").splitlines() if ":" in line)
    observed_fields = {}
    for key in ("NoNewPrivs", "CapEff", "CapPrm", "CapBnd", "CapAmb", "CapInh"):
        value = fields.get(key, "").strip()
        pattern = r"[01]" if key == "NoNewPrivs" else r"[0-9a-fA-F]{1,32}"
        observed_fields[key] = value if re.fullmatch(pattern, value) else None
    evidence["status"]["fields"] = observed_fields
    names = {line.split(":", 1)[0].strip() for line in (inputs["interfaces"] or "").splitlines()[2:]}
    evidence["interfaces"]["names"] = sorted(name if re.fullmatch(r"[A-Za-z0-9_.:-]{1,15}", name)
                                              else "[invalid-interface]" for name in names)[:32]
    evidence["interfaces"]["count"] = len(names)
    flags = []
    for name in sorted(names)[:32]:
        if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,15}", name):
            continue
        try:
            # This queries flags without bind/connect/send. No address is read.
            import fcntl
            import socket
            import struct
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
                data = fcntl.ioctl(probe.fileno(), 0x8913, struct.pack("256s", name.encode()))
            value = struct.unpack_from("H", data, 16)[0]
            flags.append({"interface": name, "flags": value, "up": bool(value & 1),
                          "loopback": bool(value & 8), "status": "observed"})
        except Exception as error:
            # A diagnostic ioctl/import/decoding failure is not the isolation
            # predicate. Preserve it without hiding a subsequent guard refusal.
            flags.append({"interface": name, "status": "unavailable", "error": type(error).__name__})
    evidence["interfaces"]["linkFlags"] = flags
    if inputs["routes"] is not None:
        evidence["routes"].update(route_structure(inputs["routes"]))
    parent_text, parent_read = bounded_text(work / "evidence/parent-namespace.json", 1024)
    parent = None
    if parent_text is not None:
        try:
            saved = json.loads(parent_text)
            value = saved.get("networkNamespace")
            matching = all(
                isinstance(os.environ.get(env), str)
                and re.fullmatch(pattern, os.environ[env])
                and saved.get(key) == os.environ[env]
                for key, env, pattern in (("source", "GITHUB_SHA", r"[0-9a-f]{40}"),
                                         ("runId", "GITHUB_RUN_ID", r"[1-9][0-9]*"),
                                         ("attempt", "GITHUB_RUN_ATTEMPT", r"[1-9][0-9]*")))
            if matching and isinstance(value, str) and re.fullmatch(r"net:\[[0-9]{1,20}\]", value):
                parent = value
            else:
                parent_read["error"] = "invalid_or_unbound_identity"
        except (ValueError, AttributeError):
            parent_read["error"] = "invalid_identity_record"
    own = namespace_identity()
    parent_read["identityBound"] = parent is not None
    evidence["namespace"] = {"current": own, "retainedParent": parent, "parentRead": parent_read,
                            "comparison": "unknown" if own is None or parent is None else
                            "same" if own == parent else "different",
                            "admissionAuthority": False}
    return evidence, inputs


def original_guard_observations(inputs):
    """Historical predicates only; no longer authoritative for admission."""
    status = dict(line.split(":", 1) for line in (inputs["status"] or "").splitlines() if ":" in line)
    nnp = status.get("NoNewPrivs", "").strip()
    caps = [status.get(key, "").strip() for key in ("CapEff", "CapPrm", "CapBnd", "CapAmb", "CapInh")]
    try:
        caps_zero = all(int(value, 16) == 0 for value in caps)
    except ValueError:
        caps_zero = False
    names = {line.split(":", 1)[0].strip() for line in (inputs["interfaces"] or "").splitlines()[2:]}
    lines = None if inputs["routes"] is None else len(inputs["routes"].splitlines())
    return [
        ("privilege_regain_not_blocked", "NoNewPrivs == 1", inputs["status"] is not None and nnp == "1"),
        ("retained_capability", "five capability fields == 0", inputs["status"] is not None and caps_zero),
        ("unexpected_network_interface", "interfaces == {lo}", inputs["interfaces"] is not None and names == {"lo"}),
        ("unexpected_network_route", "raw splitlines count == 1", lines == 1),
    ]


def corrected_guard_observations(evidence, inputs):
    def complete(key, limit):
        text, read = inputs.get(key), evidence.get(key, {})
        return (isinstance(text, str) and read.get("complete") is True
                and "error" in read and read["error"] is None
                and read.get("limit") == limit and type(read.get("bytesObserved")) is int
                and read["bytesObserved"] == len(text.encode("utf-8")) <= limit)

    reads_complete = all(complete(key, limit) for key, limit in
                         (("status", 32768), ("interfaces", 16384), ("routes", 16384)))
    namespace = evidence.get("namespace", {})
    current, parent = namespace.get("current"), namespace.get("retainedParent")
    parent_read = namespace.get("parentRead", {})
    separate = (all(isinstance(value, str) and re.fullmatch(r"net:\[[0-9]{1,20}\]", value)
                    for value in (current, parent)) and current != parent
                and namespace.get("comparison") == "different"
                and parent_read.get("complete") is True
                and "error" in parent_read and parent_read["error"] is None
                and parent_read.get("identityBound") is True
                and parent_read.get("limit") == 1024 and type(parent_read.get("bytesObserved")) is int
                and 0 < parent_read["bytesObserved"] <= 1024)
    # Do not reuse the historical dict/int parser for admission: repeated keys
    # or permissive integer spellings can conceal contradictory observations.
    status_lines = (inputs.get("status") or "").splitlines()
    def status_value(key, pattern):
        values = [line.split(":", 1)[1].strip() for line in status_lines
                  if ":" in line and line.split(":", 1)[0].strip() == key]
        return values[0] if len(values) == 1 and re.fullmatch(pattern, values[0]) else None

    nnp = status_value("NoNewPrivs", r"[01]")
    caps = [status_value(key, r"[0-9A-Fa-f]{16}")
            for key in ("CapEff", "CapPrm", "CapBnd", "CapAmb", "CapInh")]
    caps_zero = all(value is not None and int(value, 16) == 0 for value in caps)
    interface_lines = (inputs.get("interfaces") or "").splitlines()
    receive = "bytes packets errs drop fifo frame compressed multicast".split()
    transmit = "bytes packets errs drop fifo colls carrier compressed".split()
    interface_shape = (
        len(interface_lines) == 3
        and [part.strip() for part in interface_lines[0].split("|")] == ["Inter-", "Receive", "Transmit"]
        and [part.split() for part in interface_lines[1].split("|")] == [["face"], receive, transmit]
        and re.fullmatch(r"\s*lo:\s*[0-9]+(?:[ \t]+[0-9]+){15}[ \t]*", interface_lines[2]) is not None)
    interfaces = evidence.get("interfaces", {})
    flags = interfaces.get("linkFlags", [])
    down = (interfaces.get("names") == ["lo"] and interfaces.get("count") == 1
            and len(flags) == 1 and flags[0].get("interface") == "lo"
            and flags[0].get("status") == "observed" and type(flags[0].get("flags")) is int
            and flags[0]["flags"] >= 0 and bool(flags[0]["flags"] & 8) and not flags[0]["flags"] & 1
            and flags[0].get("up") is False and flags[0].get("loopback") is True)
    text = inputs.get("routes")
    no_routes = (isinstance(text, str) and (text == "" or
                 (len(text.splitlines()) == 1 and text.split() == ROUTE_HEADER
                  and re.fullmatch(r"[A-Za-z \t]+(?:\r?\n)?", text) is not None)))
    return [
        ("incomplete_isolation_observation", "all required reads complete, bounded and error-free", reads_complete),
        ("network_namespace_not_separate", "current namespace differs from bound complete parent identity", bool(separate)),
        ("privilege_regain_not_blocked", "one well-formed NoNewPrivs field == 1", nnp == "1"),
        ("retained_capability", "five unique well-formed capability fields == 0", caps_zero),
        ("unexpected_network_interface", "recognized interface headers and exactly one well-formed lo row", interface_shape),
        ("loopback_not_down", "exactly one observed loopback interface, flags DOWN", bool(down)),
        ("unexpected_network_route", "complete zero bytes OR strict recognized header only, no extra rows", no_routes),
    ]


def _enforce_guards(evidence, guards, output):
    for code, expected, passed in guards:
        evidence["guards"].append({"reasonCode": code, "expected": expected,
                                   "observedPredicate": passed, "outcome": "pass" if passed else "refused"})
        evidence["firstRefusal"] = None if passed else code
        # Persist all bounded observations BEFORE an admission require can raise.
        try:
            temporary = output / "isolation.json.tmp"
            temporary.write_text(json.dumps(evidence, indent=2) + "\n")
            temporary.replace(output / "isolation.json")
        except OSError:
            # Keep evaluating only the already-captured original predicates.
            # Evidence failure cannot hide a later isolation refusal.
            evidence["writeError"] = "isolation_evidence_write_failed"
        require(passed, code)
    require("writeError" not in evidence, "isolation_evidence_write_failed")


def enforce_original_isolation(evidence, inputs, output):
    """Historical fixture controls only; main never uses these for admission."""
    _enforce_guards(evidence, original_guard_observations(inputs), output)


def enforce_isolation(evidence, inputs, output):
    evidence["policy"] = "complete-empty-or-header-only-with-separate-down-loopback.v1"
    evidence["originalPredicates"] = [
        {"reasonCode": code, "expected": expected, "observedPredicate": passed, "admissionAuthority": False}
        for code, expected, passed in original_guard_observations(inputs)]
    _enforce_guards(evidence, corrected_guard_observations(evidence, inputs), output)


def exact_case_evidence(collected, reports, tree):
    """Reconcile this fixed 32-case file; never turn an incomplete subset green."""
    expected = Counter(EXPECTED_NODE_IDS)

    def compare(observed):
        counts = Counter(observed)
        return {"count": len(observed), "unique": len(counts),
                "missing": sorted((expected - counts).elements()),
                "duplicate": sorted(node for node, count in counts.items() if count > 1),
                "unexpected": sorted(node for node in counts if node not in expected),
                "exact": len(observed) == 32 and counts == expected}

    calls = [report["nodeid"] for report in reports if report["when"] == "call"]
    cases = list(tree.iter("testcase"))
    identities = [{"classname": case.get("classname"), "name": case.get("name")} for case in cases]
    mapped = ["tests/test_image_operation.py::" + case["name"]
              for case in identities if case["classname"] == "tests.test_image_operation"
              and isinstance(case["name"], str)]
    junit = compare(mapped)
    junit.update(caseCount=len(cases), identities=identities, mappedNodeIds=mapped,
                 unmapped=[case for case in identities if case["classname"] != "tests.test_image_operation"
                           or not isinstance(case["name"], str)])
    counts = {key: len(list(tree.iter(tag))) for key, tag in (
        ("tests", "testcase"), ("failures", "failure"), ("errors", "error"), ("skipped", "skipped"))}
    suites = list(tree.iter("testsuite"))
    attributes = [{key: suite.get(key) for key in counts} for suite in suites]
    # The unchanged pytest invocation writes one suite with direct case children.
    # Reconcile all four declared counters against the actual XML, not just tests.
    counters_match = (len(suites) == 1 and len(suites[0].findall("testcase")) == len(cases)
                      and all(attributes[0][key] is not None and re.fullmatch(r"[0-9]+", attributes[0][key])
                              and int(attributes[0][key]) == value for key, value in counts.items()))
    junit.update(declaredCounters=attributes, countsMatch=counters_match)
    collection, call_evidence = compare(collected), compare(calls)
    unexpected_reports = [report for report in reports if report["nodeid"] not in expected]
    complete = (len(EXPECTED_NODE_IDS) == len(expected) == 32
                and all(count == 1 for count in expected.values())
                and collection["exact"] and call_evidence["exact"] and not unexpected_reports
                and junit["exact"] and len(cases) == 32 and not junit["unmapped"] and counters_match)
    return {"expectedNodeIds": list(EXPECTED_NODE_IDS), "expectedCount": 32,
            "collection": collection, "calls": call_evidence, "junit": junit,
            "unexpectedReports": unexpected_reports, "counts": counts,
            "allRequiredExecuted": complete}


def main():
    # These checks precede every write. The workflow creates the offline namespace.
    require(os.environ.get("GITHUB_ACTIONS") == "true", "host_execution_refused")
    require(os.environ.get("GITHUB_REPOSITORY") == "microsoft/amplifier-module-provider-openai", "wrong_repository")
    require(os.environ.get("GITHUB_REF") == "refs/heads/test/image-empty-route-baseline-run-ad2410", "wrong_execution_branch")
    require(sys.argv[1:] == [], "fixed_baseline_arguments_required")
    require(os.environ.get("IMAGE_BASELINE_OFFLINE") == "1", "offline_namespace_required")
    temp = Path(os.environ["RUNNER_TEMP"]).resolve()
    work = temp / "openai-image-outcome"
    output = work / "evidence"
    require(work.is_dir() and output.is_dir(), "preparation_missing")
    require(sys.flags.isolated and Path(sys.prefix).resolve() == work / "runtime", "wrong_interpreter")
    result = {"status": "BLOCKED", "testBaseline": BASELINE, "foundation": FOUNDATION,
              "source": os.environ.get("GITHUB_SHA"), "runId": os.environ.get("GITHUB_RUN_ID"),
              "runAttempt": os.environ.get("GITHUB_RUN_ATTEMPT"), "testsExecuted": False,
              "productImports": False, "networkAttempts": None,
              "limits": "Offline single-SDK provider baseline; not downstream, newest-SDK or live image acceptance."}
    collected, reports, network_events = [], [], []
    log = io.StringIO()
    exit_code = 1
    try:
        observation, inputs = isolation_observations(work)
        result["isolationDiagnostics"] = observation
        enforce_isolation(observation, inputs, output)
        require(git("rev-parse", "HEAD") == result["source"], "checkout_identity_mismatch")
        require(git("rev-parse", "HEAD:" + PRODUCT) == PRODUCT_TREE, "product_changed_before_baseline")
        require(git("rev-parse", "HEAD:tests/test_image_operation.py") == TEST_BLOB, "regressions_changed")
        require(not git("status", "--porcelain", "--untracked-files=no"), "tracked_source_dirty")
        result["sourceTree"] = git("rev-parse", "HEAD^{tree}")
        result["testSha256"] = digest(TEST)
        result["lockSha256"] = digest(ROOT / "uv.lock")
        result["versions"] = {name: metadata.version(name) for name in (
            "amplifier-core", "amplifier-foundation", "amplifier-module-provider-openai",
            "openai", "httpx", "Pillow", "pytest", "pytest-asyncio")}
        require(result["versions"]["openai"] == "3.29.0", "wrong_sdk")
        require(result["versions"]["amplifier-core"] == "2.0.2", "wrong_core")
        foundation = metadata.distribution("amplifier-foundation")
        direct = json.loads(foundation.read_text("direct_url.json") or "{}")
        require(direct.get("vcs_info", {}).get("commit_id") == FOUNDATION, "wrong_foundation_revision")
        require(not direct.get("dir_info", {}).get("editable"), "editable_foundation")
        wheels = list((work / "wheels").glob("*.whl"))
        require(len(wheels) == 1, "wrong_wheel_count")
        wheel = wheels[0]
        dist = metadata.distribution("amplifier-module-provider-openai")
        direct = json.loads(dist.read_text("direct_url.json") or "{}")
        require(direct.get("url") == wheel.as_uri() and not direct.get("dir_info"), "nonwheel_provider")
        count = 0
        with zipfile.ZipFile(wheel) as archive:
            for member in archive.infolist():
                if not member.filename.startswith(PRODUCT + "/") or member.is_dir():
                    continue
                installed = Path(dist.locate_file(member.filename)).resolve()
                require(installed.is_relative_to(work / "runtime"), "payload_outside_environment")
                require(installed.read_bytes() == archive.read(member) == (ROOT / member.filename).read_bytes(),
                        "installed_payload_mismatch")
                count += 1
        require(count > 0, "empty_provider_payload")
        result.update(wheelSha256=digest(wheel), installedPayloadFiles=count,
                      dependencyVersions=sorted((d.metadata["Name"], d.version) for d in metadata.distributions()))
        # No ambient API keys, proxy, Python paths or credential configuration.
        keep = {key: os.environ[key] for key in ("PATH", "LANG") if key in os.environ}
        os.environ.clear()
        os.environ.update(keep, HOME=str(work / "home"), TMPDIR=str(work / "tmp"),
                          PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
        for name in ("home", "tmp"):
            (work / name).mkdir(exist_ok=True)

        def audit(event, args):
            if event in {"socket.connect", "socket.sendto", "socket.getaddrinfo"}:
                network_events.append(event)
                raise PermissionError("offline_image_baseline_network_refused")
        sys.addaudithook(audit)
        result["socketAuditInstalled"] = True
        modules = ("amplifier_core", "amplifier_core.models", "amplifier_foundation.operation_routing", PRODUCT)
        for name in modules:
            importlib.import_module(name)
        result["productImports"] = True

        def origins():
            paths = {}
            for name, module in list(sys.modules.items()):
                if name.startswith(("amplifier_core", "amplifier_foundation", PRODUCT)) and module is not None:
                    path = getattr(module, "__file__", None)
                    require(path and Path(path).resolve().is_relative_to(work / "runtime"), "source_import_detected")
                    paths[name] = str(Path(path).resolve().relative_to(work / "runtime"))
            return paths
        result["originsBefore"] = origins()
        class Evidence:
            def pytest_collection_finish(self, session):
                collected.extend(item.nodeid for item in session.items)
            def pytest_runtest_logreport(self, report):
                reports.append({"nodeid": report.nodeid, "when": report.when, "outcome": report.outcome})

        import pytest
        os.chdir(work)
        with redirect_stdout(log), redirect_stderr(log):
            code = int(pytest.main([str(TEST), "--rootdir=" + str(ROOT), "--import-mode=importlib",
                "-o", "addopts=", "-o", "pythonpath=", "-o", "asyncio_mode=strict", "-o", "xfail_strict=true",
                "-p", "pytest_asyncio.plugin", "--junitxml=" + str(work / "junit.xml"),
                "--basetemp=" + str(work / "pytest-tmp"), "-q", "-ra", "--tb=short"], plugins=[Evidence()]))
        result.update(pytestExit=code, testsExecuted=True, collectedNodeIds=collected, reports=reports,
                      networkAttempts=len(network_events))
        result["originsAfter"] = origins()
        xml = (work / "junit.xml").read_text()
        require(len(xml.encode()) <= 2 * 1024 * 1024, "junit_too_large")
        (output / "junit.xml").write_text(clean_text(xml, work))
        tree = ET.fromstring(xml)
        case_evidence = exact_case_evidence(collected, reports, tree)
        result["caseEvidence"] = case_evidence
        counts = case_evidence["counts"]
        result["counts"] = counts
        result["allRequiredExecuted"] = case_evidence["allRequiredExecuted"]
        result["status"] = "PASS" if (code == 0 and result["allRequiredExecuted"] and not network_events
            and not any(counts[key] for key in ("failures", "errors", "skipped"))) else "FAIL"
        exit_code = 0 if result["status"] == "PASS" else (code or 1)
    except Exception as error:
        result["reason"] = str(error) if type(error) is RuntimeError else type(error).__name__
        result.update(collectedNodeIds=collected, reports=reports,
                      networkAttempts=len(network_events) if result.get("socketAuditInstalled") else None)
    finally:
        try:
            (output / "pytest.log").write_text(clean_text(log.getvalue()[-128000:], work))
        except OSError:
            result["logWriteError"] = "pytest_log_write_failed"
            if exit_code == 0:
                result.update(status="BLOCKED", reason="pytest_log_write_failed")
            exit_code = 1
        try:
            (output / "result.json").write_text(clean_text(json.dumps(result, indent=2), work) + "\n")
        except OSError:
            result["receiptWriteError"] = "result_evidence_write_failed"
            if exit_code == 0:
                result.update(status="BLOCKED", reason="result_evidence_write_failed")
            exit_code = 1
        print(json.dumps({"status": result["status"], "testsExecuted": result["testsExecuted"],
                          "counts": result.get("counts"),
                          "reason": result.get("reason"),
                          "receiptWriteError": result.get("receiptWriteError")}))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
