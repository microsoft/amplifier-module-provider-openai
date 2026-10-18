"""Fixed hosted lock-only prerequisite; never imports or builds product packages.

Only the five frozen pyproject files enter the resolver directory. This helper
belongs to a temporary test branch, not the provider's shipping implementation.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import selectors
import shutil
import signal
import subprocess
import sys
import time
import tomllib
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/imagegen-lock"
REPOSITORY = "microsoft/amplifier-module-provider-openai"
RUN_REF = "refs/heads/test/imagegen-lock-run-ad2410"
UV_VERSION = "0.9.18"
LOG_LIMIT = 128 * 1024
LOCK_LIMIT = 2 * 1024 * 1024
CORE_URL = "https://files.pythonhosted.org/packages/92/7f/80c9d27f76a3a1f0168ba4410e289d2eac21b01f5728e61be39df6ed9333/amplifier_core-2.0.2-cp311-abi3-manylinux_2_17_aarch64.manylinux2014_aarch64.whl"
CORE_HASH = "sha256:cba63bf80af198819ca648edbac4998735d2656fd19af0a569ef1849547a97ac"
INPUTS = {
    "pyproject.toml": "93fd4394d3d947631715f357af60bd31488cfec26ee9ee407e6f8043f63b76b0",
    "sources/amplifier-foundation/pyproject.toml": "fb8bae0a840e7191c2d55f570cc84bf47a78b4d398bf9b262205f9b482f961b4",
    "sources/amplifier-module-provider-openai/pyproject.toml": "de372b596596be3b10ecb09ccea7911bbf050e65d9eefa1797ef4493dc864c9e",
    "sources/amplifier-module-provider-gemini/pyproject.toml": "e7e7c18803a92a439aa6cfc2819eac19a70415318ecbafb004cdd3cd3d399d48",
    "sources/amplifier-bundle-imagegen/pyproject.toml": "b5c5b669ce783b226e1832e71966776d1e2700e15aab2955d60bf0800090084f",
}
LOCAL = {
    "amplifier-foundation": ("1.0.0", "sources/amplifier-foundation"),
    "amplifier-module-provider-openai": ("1.0.0", "sources/amplifier-module-provider-openai"),
    "amplifier-module-provider-gemini": ("1.0.0", "sources/amplifier-module-provider-gemini"),
    "amplifier-module-tool-image": ("0.1.0", "sources/amplifier-bundle-imagegen"),
}
PINS = {
    "amplifier-core": "2.0.2", "openai": "3.29.0", "google-genai": "2.30.1",
    "Pillow": "12.3.0", "httpx": "0.28.1", "pytest": "9.1.1",
    "pytest-asyncio": "1.4.0", "pytest-timeout": "2.4.0",
}


def require(value, reason):
    if not value:
        raise RuntimeError(reason)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_bytes(path, limit):
    require(not path.is_symlink() and path.is_file(), "missing_or_unsafe_file")
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    require(len(data) <= limit, "oversized_file")
    return data


def check_tree(directory, *, locked=False):
    expected = set(INPUTS) | ({"uv.lock"} if locked else set())
    files = set()
    directories = {str(parent) for name in expected for parent in PurePosixPath(name).parents if str(parent) != "."}
    for path in directory.rglob("*"):
        require(not path.is_symlink(), "symlink_in_manifest_tree")
        name = path.relative_to(directory).as_posix()
        if path.is_dir():
            require(name in directories, "extra_directory_in_manifest_tree")
        else:
            require(path.is_file() and name in expected, "extra_file_in_manifest_tree")
            files.add(name)
    require(files == expected, "incomplete_manifest_tree")
    for name, expected_hash in INPUTS.items():
        data = file_bytes(directory / name, 8192)
        require(digest(data) == expected_hash, "manifest_hash_mismatch")
        value = tomllib.loads(data.decode())
        project = value.get("project", {})
        require(not project.get("dynamic") and isinstance(project.get("name"), str)
                and isinstance(project.get("version"), str), "dynamic_or_incomplete_metadata")
        config = value.get("tool", {}).get("uv", {})
        require(not any(key.startswith(("no-binary", "index", "extra-index", "find-links")) for key in config),
                "manifest_resolver_override")
        if name == "pyproject.toml":
            require(config.get("package") is False, "root_not_virtual")
            sources = config.get("sources")
            require(isinstance(sources, dict) and set(sources) == set(LOCAL), "wrong_local_sources")
            for package, (_, relative) in LOCAL.items():
                require(sources[package] == {"path": relative, "editable": False}, "unsafe_local_source")
                require(not PurePosixPath(relative).is_absolute() and ".." not in PurePosixPath(relative).parts,
                        "out_of_tree_source")
            require(config.get("override-dependencies") == ["amplifier-foundation==1.0.0"], "wrong_foundation_override")


def public_file_url(value):
    parsed = urlsplit(value)
    return (parsed.scheme == "https" and parsed.netloc == "files.pythonhosted.org"
            and not parsed.username and not parsed.password and not parsed.query and not parsed.fragment)


def validate_lock(data):
    lock = tomllib.loads(data.decode())
    packages = lock.get("package")
    require(lock.get("version") == 1 and isinstance(packages, list) and 1 <= len(packages) <= 256,
            "invalid_lock_packages")
    selected = {}
    inventory = []
    for package in packages:
        name, version, source = package.get("name"), package.get("version"), package.get("source")
        require(isinstance(name, str) and re.fullmatch(r"[a-z0-9][a-z0-9-]{0,99}", name)
                and isinstance(version, str) and re.fullmatch(r"[A-Za-z0-9.!+_-]{1,80}", version)
                and isinstance(source, dict), "invalid_package_identity")
        if name in LOCAL:
            expected_version, path = LOCAL[name]
            require(version == expected_version and source == {"directory": path}, "wrong_candidate_source")
        elif name == "imagegen-local-baseline-ad2410":
            require(version == "0.0.0" and source == {"virtual": "."}, "wrong_virtual_root")
        elif name == "amplifier-core":
            require(version == "2.0.2" and set(source) == {"url"}
                    and source["url"] in {CORE_URL, CORE_URL + "#" + CORE_HASH.replace(":", "=", 1)},
                    "wrong_core_source")
            wheels = package.get("wheels")
            require(isinstance(wheels, list) and len(wheels) == 1
                    and wheels[0].get("url") == CORE_URL and wheels[0].get("hash") == CORE_HASH,
                    "wrong_core_wheel_hash")
            require("sdist" not in package, "core_source_build_not_allowed")
        else:
            require(set(source) == {"registry"} and source["registry"].rstrip("/") == "https://pypi.org/simple",
                    "nonpublic_or_unexpected_selected_source")
        for artifact in [*package.get("wheels", []), *([package["sdist"]] if "sdist" in package else [])]:
            require(isinstance(artifact, dict) and public_file_url(artifact.get("url", ""))
                    and re.fullmatch(r"sha256:[a-f0-9]{64}", artifact.get("hash", "")), "unsafe_registry_artifact")
        selected.setdefault(name, []).append(package)
        inventory.append({"name": name, "version": version, "source": source})
    for name, version in {**{key: value[0] for key, value in LOCAL.items()}, **{key.lower(): value for key, value in PINS.items()},
                          "imagegen-local-baseline-ad2410": "0.0.0"}.items():
        require(len(selected.get(name, [])) == 1 and selected[name][0]["version"] == version, "missing_or_changed_direct_pin")
    return inventory


def run_command(arguments, *, cwd, env, seconds, tail):
    """Bound process time and retained bytes without buffering unrestricted logs."""
    started = time.monotonic()
    process = subprocess.Popen(arguments, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
    print(json.dumps({"resolverPid": process.pid, "stage": "check" if "--check" in arguments else "lock"}), flush=True)
    timed_out = False
    total = 0
    with selectors.DefaultSelector() as selector:
        selector.register(process.stdout, selectors.EVENT_READ)
        try:
            while selector.get_map():
                if time.monotonic() - started >= seconds:
                    timed_out = True
                    break
                for key, _ in selector.select(0.1):
                    chunk = os.read(key.fileobj.fileno(), 16384)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    total += len(chunk)
                    tail.extend(chunk)
                    del tail[:-LOG_LIMIT]
            if not timed_out:
                try:
                    process.wait(timeout=max(0.01, seconds - (time.monotonic() - started)))
                except subprocess.TimeoutExpired:
                    timed_out = True
        finally:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            process.wait()
            process.stdout.close()
    return {"exitCode": process.returncode, "timedOut": timed_out,
            "seconds": round(time.monotonic() - started, 3), "outputBytes": total}


def main():
    require(os.environ.get("GITHUB_ACTIONS") == "true", "github_hosted_job_required")
    require(os.environ.get("GITHUB_REPOSITORY") == REPOSITORY, "wrong_repository")
    require(os.environ.get("GITHUB_EVENT_NAME") == "push" and os.environ.get("GITHUB_REF") == RUN_REF, "wrong_execution_event")
    temp = Path(os.environ["RUNNER_TEMP"]).resolve()
    require(temp.is_dir() and temp.is_absolute(), "invalid_runner_temp")
    output = temp / "imagegen-manifest-lock-evidence"
    require(output.is_dir() and not output.is_symlink(), "missing_initial_receipt")
    work = temp / "imagegen-manifest-lock"
    result = {"schema": "imagegen-manifest-lock.v1", "status": "BLOCKED", "phase": "preflight",
              "source": os.environ.get("GITHUB_SHA"), "runId": os.environ.get("GITHUB_RUN_ID"),
              "attempt": os.environ.get("GITHUB_RUN_ATTEMPT"), "inputs": INPUTS,
              "resolution": None, "check": None, "lockPublished": False,
              "productBuildsOrTestsRequested": False, "limits": "Lock-only prerequisite, not installation or product acceptance."}
    tail = bytearray()
    code = 1
    lock_data = None
    try:
        require(re.fullmatch(r"[a-f0-9]{40}", result["source"] or "")
                and re.fullmatch(r"[0-9]+", result["runId"] or "") and result["attempt"] == "1", "invalid_job_identity")
        initial = json.loads(file_bytes(output / "lock-result.json", 8192))
        require(all(initial.get(key) == result[key] for key in ("source", "runId", "attempt")),
                "initial_receipt_identity_mismatch")
        elapsed = time.time() - initial.get("startedAt", 0)
        require(0 <= elapsed < 420, "job_evidence_reserve_exhausted")
        # Resolve and --check share at most 300 seconds. Also finish by seven
        # minutes after the initial receipt, leaving time for evidence/upload.
        deadline = time.monotonic() + min(300, 420 - elapsed)
        result.update(startedAt=initial["startedAt"], sharedResolverBudgetSeconds=round(min(300, 420 - elapsed), 3))
        def remaining():
            seconds = deadline - time.monotonic()
            require(seconds > 0, "shared_resolver_deadline_exhausted")
            return seconds
        def checkpoint(phase):
            result["phase"] = phase
            (output / "lock-result.json").write_text(json.dumps(result, indent=2) + "\n")
        require(sys.flags.isolated and sys.version_info[:2] == (3, 13), "wrong_python")
        require(not work.exists(), "work_directory_already_exists")
        work.mkdir(mode=0o700)
        for name in ("home", "cache", "tmp", "project"):
            (work / name).mkdir(mode=0o700)
        uv_name = shutil.which("uv")
        require(uv_name, "uv_missing")
        uv, python = str(Path(uv_name).resolve()), str(Path(sys.executable).resolve())
        env = {"PATH": os.pathsep.join(dict.fromkeys((str(Path(uv).parent), str(Path(python).parent), "/usr/bin", "/bin"))),
               "HOME": str(work / "home"), "TMPDIR": str(work / "tmp"), "LANG": "C.UTF-8",
               "UV_CACHE_DIR": str(work / "cache"), "UV_NO_CONFIG": "1", "GIT_TERMINAL_PROMPT": "0",
               "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_ASKPASS": "/bin/false"}
        version = subprocess.check_output([uv, "--version"], env=env, text=True, timeout=10).strip()
        require(re.fullmatch(r"uv 0\.9\.18(?: \([^\n]{1,100}\))?", version), "wrong_uv")
        git = lambda *args: subprocess.check_output(["git", "-C", str(ROOT), *args], env=env, text=True, timeout=10).strip()
        require(git("rev-parse", "HEAD") == result["source"] and not git("status", "--porcelain"), "source_changed")
        result.update(uvVersion=version, pythonVersion=sys.version.split()[0], helperSha256=digest(Path(__file__).read_bytes()))
        check_tree(FIXTURE)
        project = work / "project"
        for name in INPUTS:
            target = project / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(file_bytes(FIXTURE / name, 8192))
        check_tree(project)
        command = [uv, "lock", "--no-config", "--no-build", "--python", python, "--no-python-downloads",
                   "--keyring-provider", "disabled", "--default-index", "https://pypi.org/simple"]
        checkpoint("resolve")
        result["resolution"] = run_command(command, cwd=project, env=env, seconds=remaining(), tail=tail)
        require(result["resolution"]["exitCode"] == 0 and not result["resolution"]["timedOut"], "lock_resolution_failed")
        check_tree(project, locked=True)
        lock_data = file_bytes(project / "uv.lock", LOCK_LIMIT)
        result["selectedPackages"] = validate_lock(lock_data)
        checkpoint("consistency_check")
        result["check"] = run_command([*command, "--check"], cwd=project, env=env, seconds=remaining(), tail=tail)
        require(result["check"]["exitCode"] == 0 and not result["check"]["timedOut"], "lock_consistency_failed")
        check_tree(project, locked=True)
        require(file_bytes(project / "uv.lock", LOCK_LIMIT) == lock_data, "lock_changed_during_check")
        check_tree(FIXTURE)
        require(not git("status", "--porcelain"), "checkout_changed")
        result.update(status="PASS", phase="validated", lockSha256=digest(lock_data), lockBytes=len(lock_data))
        code = 0
    except Exception as error:
        result.update(status="BLOCKED", reasonCode=str(error) if type(error) is RuntimeError else type(error).__name__)
    finally:
        text = tail.decode("utf-8", "replace")
        for path in (str(ROOT), str(work), str(temp), str(Path.home())):
            text = text.replace(path, "[JOB]")
        text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)
        text = re.sub(r"https?://[^\s/@]+:[^\s/@]+@", "https://[REDACTED]@", text)
        result["logTruncated"] = sum((result.get(stage) or {}).get("outputBytes", 0)
                                     for stage in ("resolution", "check")) > LOG_LIMIT
        try:
            (output / "lock.log").write_bytes(text.encode()[-LOG_LIMIT:])
            if code == 0:
                staged = output / ".validated-lock.tmp"
                with staged.open("xb") as stream:
                    stream.write(lock_data)
                    stream.flush()
                    os.fsync(stream.fileno())
                require(not (output / "uv.lock").exists(), "published_lock_already_exists")
                os.replace(staged, output / "uv.lock")
                result["lockPublished"] = True
        except (OSError, RuntimeError):
            result.update(status="BLOCKED", evidenceWriteError="lock_or_log_write_failed")
            code = 1
        try:
            (output / "lock-result.json").write_text(json.dumps(result, indent=2) + "\n")
        except OSError:
            result.update(status="BLOCKED", evidenceWriteError="result_write_failed")
            code = 1
        print(json.dumps({key: result.get(key) for key in ("status", "phase", "reasonCode", "lockPublished", "evidenceWriteError")}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
