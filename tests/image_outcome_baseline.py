"""Diagnostic-only successor to the blocked installed-wheel image baseline.

Run only in the repository's admitted hosted execution branch. Product/test bytes
remain those of BASELINE. The four original isolation predicates are unchanged;
format classifications are observations, never permission to proceed. Even when
the guards and wheel attestation pass, this increment never imports a provider.
"""
from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
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


def route_structure(text):
    """Describe format, not route addresses; never use this to grant admission."""
    lines = text.splitlines()  # Deliberately no strip/filter: preserve the refusal.
    header = "Iface Destination Gateway Flags RefCnt Use Metric Mask MTU Window IRTT".split()
    recognized = bool(lines) and lines[0].split() == header
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
                "safetyQualified": False, "productExecutionAuthorized": False}
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
            matching = all(saved.get(key) == os.environ.get(env) for key, env in (
                ("source", "GITHUB_SHA"), ("runId", "GITHUB_RUN_ID"), ("attempt", "GITHUB_RUN_ATTEMPT")))
            if matching and isinstance(value, str) and re.fullmatch(r"net:\[[0-9]{1,20}\]", value):
                parent = value
            else:
                parent_read["error"] = "invalid_or_unbound_identity"
        except (ValueError, AttributeError):
            parent_read["error"] = "invalid_identity_record"
    own = namespace_identity()
    evidence["namespace"] = {"current": own, "retainedParent": parent, "parentRead": parent_read,
                            "comparison": "unknown" if own is None or parent is None else
                            "same" if own == parent else "different",
                            "admissionAuthority": False}
    return evidence, inputs


def original_guard_observations(inputs):
    """Same four admission predicates as a35e9; incomplete reads never qualify."""
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


def enforce_original_isolation(evidence, inputs, output):
    for code, expected, passed in original_guard_observations(inputs):
        evidence["guards"].append({"reasonCode": code, "expected": expected,
                                   "observedPredicate": passed, "outcome": "pass" if passed else "refused"})
        evidence["firstRefusal"] = None if passed else code
        # Persist all bounded observations BEFORE the original require can raise.
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


def main():
    # These checks precede every write. The workflow creates the offline namespace.
    require(os.environ.get("GITHUB_ACTIONS") == "true", "host_execution_refused")
    require(os.environ.get("GITHUB_REPOSITORY") == "microsoft/amplifier-module-provider-openai", "wrong_repository")
    require(os.environ.get("GITHUB_REF") == "refs/heads/test/image-isolation-diagnostics-run-ad2410", "wrong_execution_branch")
    require(sys.argv[1:] == ["--diagnostic-only"], "diagnostic_only_required")
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
              "limits": "Isolation diagnostics only; socket audit and product tests are never entered."}
    exit_code = 1
    try:
        observation, inputs = isolation_observations(work)
        result["isolationDiagnostics"] = observation
        enforce_original_isolation(observation, inputs, output)
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
        # This distinct diagnostic increment never calls providers, even when the
        # original guards and native wheel/source attestation all pass.
        result.update(status="DIAGNOSTIC_ONLY", reason="product_execution_not_admitted",
                      diagnosticComplete=True)
        exit_code = 0
    except Exception as error:
        result["reason"] = str(error) if type(error) is RuntimeError else type(error).__name__
    finally:
        try:
            (output / "result.json").write_text(clean_text(json.dumps(result, indent=2), work) + "\n")
        except OSError:
            result["receiptWriteError"] = "result_evidence_write_failed"
            if exit_code == 0:
                result.update(status="BLOCKED", reason="result_evidence_write_failed")
            exit_code = 1
        print(json.dumps({"status": result["status"], "testsExecuted": result["testsExecuted"],
                          "reason": result.get("reason"),
                          "receiptWriteError": result.get("receiptWriteError")}))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
