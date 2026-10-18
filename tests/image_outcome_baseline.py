"""Fixed installed-wheel baseline for the retained image-outcome regressions.

Run only in the repository's admitted hosted execution branch. Product/test bytes
remain those of BASELINE; CI plumbing may differ. A failing regression is a failure,
not an expected-green result. No live provider service is permitted.
"""
from __future__ import annotations

import ast
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
PRODUCT_TREE = "f9cf9c89cd7fdf96964552a016fdf3c4fe74692d"
TEST_BLOB = "787e1308c9d17f5f9f4bae3a278ff2acc55507b6"
ROOT = Path(__file__).resolve().parents[1]
PRODUCT = "amplifier_module_provider_openai"
TEST = ROOT / "tests/test_image_operation.py"


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


def main():
    # These checks precede every write. The workflow creates the offline namespace.
    require(os.environ.get("GITHUB_ACTIONS") == "true", "host_execution_refused")
    require(os.environ.get("GITHUB_REPOSITORY") == "microsoft/amplifier-module-provider-openai", "wrong_repository")
    require(os.environ.get("GITHUB_REF") == "refs/heads/test/image-outcome-run-ad2410", "wrong_execution_branch")
    require(os.environ.get("IMAGE_BASELINE_OFFLINE") == "1", "offline_namespace_required")
    temp = Path(os.environ["RUNNER_TEMP"]).resolve()
    work = temp / "openai-image-outcome"
    output = work / "evidence"
    require(work.is_dir() and output.is_dir(), "preparation_missing")
    require(sys.flags.isolated and Path(sys.prefix).resolve() == work / "runtime", "wrong_interpreter")
    result = {"status": "BLOCKED", "testBaseline": BASELINE, "foundation": FOUNDATION,
              "source": os.environ.get("GITHUB_SHA"), "runId": os.environ.get("GITHUB_RUN_ID"),
              "runAttempt": os.environ.get("GITHUB_RUN_ATTEMPT"), "testsExecuted": False,
              "limits": "Offline single-SDK provider baseline; not downstream, newest-SDK or live image acceptance."}
    collected, reports, network_events = [], [], []
    log = io.StringIO()
    exit_code = 1
    try:
        process_status = dict(line.split(":", 1) for line in Path("/proc/self/status").read_text().splitlines() if ":" in line)
        require(process_status["NoNewPrivs"].strip() == "1", "privilege_regain_not_blocked")
        require(all(int(process_status[key].strip(), 16) == 0
                    for key in ("CapEff", "CapPrm", "CapBnd", "CapAmb", "CapInh")), "retained_capability")
        # /proc/net is scoped to this process's network namespace. Do not infer
        # it from the runner's pre-existing sysfs mount.
        interfaces = {line.split(":", 1)[0].strip()
                      for line in Path("/proc/net/dev").read_text().splitlines()[2:]}
        require(interfaces == {"lo"}, "unexpected_network_interface")
        require(len(Path("/proc/net/route").read_text().splitlines()) == 1, "unexpected_network_route")
        result["isolation"] = {"interfaces": sorted(interfaces), "ipv4Routes": 0,
                               "noNewPrivileges": True, "capabilities": "empty"}
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
        modules = ("amplifier_core", "amplifier_core.models", "amplifier_foundation.operation_routing", PRODUCT)
        for name in modules:
            importlib.import_module(name)

        def origins():
            paths = {}
            for name, module in list(sys.modules.items()):
                if name.startswith(("amplifier_core", "amplifier_foundation", PRODUCT)) and module is not None:
                    path = getattr(module, "__file__", None)
                    require(path and Path(path).resolve().is_relative_to(work / "runtime"), "source_import_detected")
                    paths[name] = str(Path(path).resolve().relative_to(work / "runtime"))
            return paths
        result["originsBefore"] = origins()
        definitions = {node.name for node in ast.parse(TEST.read_text()).body
                       if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_")}

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
        counts = {key: len(list(tree.iter(tag))) for key, tag in (
            ("tests", "testcase"), ("failures", "failure"), ("errors", "error"), ("skipped", "skipped"))}
        result["counts"] = counts
        functions = {node.rsplit("::", 1)[-1].split("[", 1)[0] for node in collected}
        called = {row["nodeid"] for row in reports if row["when"] == "call"}
        result["allRequiredExecuted"] = functions == definitions and called == set(collected) and bool(collected)
        result["status"] = "PASS" if (code == 0 and result["allRequiredExecuted"] and not network_events
            and not any(counts[key] for key in ("failures", "errors", "skipped"))) else "FAIL"
        exit_code = 0 if result["status"] == "PASS" else (code or 1)
    except Exception as error:
        result["reason"] = str(error) if type(error) is RuntimeError else type(error).__name__
        result.update(collectedNodeIds=collected, reports=reports, networkAttempts=len(network_events))
    finally:
        (output / "pytest.log").write_text(clean_text(log.getvalue()[-128000:], work))
        (output / "result.json").write_text(clean_text(json.dumps(result, indent=2), work) + "\n")
        print(json.dumps({"status": result["status"], "testsExecuted": result["testsExecuted"],
                          "counts": result.get("counts"), "reason": result.get("reason")}))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
