"""Trusted-test runner: sanitize credentials and deny network before imports.

Use python -I -S tests/run_offline.py [--receipt FILE] [pytest args]. This
process audit is not a native-code or child-process security sandbox.
"""

import atexit
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

root = Path(__file__).resolve().parents[1]
command = list(sys.orig_argv)
revision = subprocess.run(
    ["git", "-C", str(root), "rev-parse", "HEAD"],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
inputs = sorted(
    [
        *root.glob("amplifier_module_provider_openai/**/*.py"),
        *root.glob("tests/**/*.py"),
        *root.glob("contracts/openai-media-evidence.v1.json"),
        root / "pyproject.toml",
        root / "uv.lock",
    ]
)


def hashes():
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in inputs
    }


input_hashes = hashes()
attempts = []
loopback = []
allow_loopback = "--allow-loopback" in sys.argv
if allow_loopback:
    sys.argv.remove("--allow-loopback")
receipt = root / ".media-offline-receipt.json"
if "--receipt" in sys.argv:
    i = sys.argv.index("--receipt")
    receipt = Path(sys.argv[i + 1]).resolve()
    del sys.argv[i : i + 2]
for key in list(os.environ):
    if any(
        word in key.upper()
        for word in ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
    ):
        os.environ.pop(key, None)
for key in ("PYTEST_ADDOPTS", "PYTHONPATH", "PYTHONHOME"):
    os.environ.pop(key, None)
owned_home = tempfile.TemporaryDirectory(prefix="openai-offline-")
for key in ("HOME", "USERPROFILE", "XDG_CONFIG_HOME", "XDG_CACHE_HOME"):
    os.environ[key] = owned_home.name
result = {
    "revision": revision,
    "command": command,
    "input_sha256": input_hashes,
    "attempts": attempts,
    "permitted_loopback": loopback,
    "exit_code": None,
    "shutdown_accounted": False,
}


def guard(event, args):
    if allow_loopback and event == "socket.getaddrinfo" and args[0] == "127.0.0.1":
        loopback.append({"event": event})
        return
    if (
        allow_loopback
        and event == "socket.connect"
        and isinstance(args[1], tuple)
        and args[1][0] == "127.0.0.1"
    ):
        loopback.append({"event": event})
        return
    if (
        event == "socket.getaddrinfo"
        or event == "socket.sendto"
        or (event == "socket.connect" and getattr(args[0], "family", None) in (2, 10))
    ):
        attempts.append({"event": event})
        raise RuntimeError("Offline test network access denied")


sys.addaudithook(guard)
import site  # guard before site and dependency imports

site.main()
sys.path.insert(0, str(root))
from importlib import metadata

result["packages"] = {}
for package in ("amplifier-core", "amplifier-foundation", "openai", "Pillow"):
    try:
        dist = metadata.distribution(package)
        result["packages"][package] = {
            "version": dist.version,
            "direct_url": dist.read_text("direct_url.json"),
        }
    except metadata.PackageNotFoundError:
        result["packages"][package] = None


def finalize():
    result["inputs_unchanged"] = hashes() == input_hashes
    owned_home.cleanup()
    result["shutdown_accounted"] = True
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(result, indent=2) + "\n")


atexit.register(finalize)
import pytest  # guard before pytest and plugins

try:
    result["exit_code"] = int(pytest.main(sys.argv[1:]))
finally:
    if attempts:
        result["exit_code"] = 1
raise SystemExit(result["exit_code"])
