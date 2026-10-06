"""Pytest configuration for module tests.

Behavioral tests use inheritance from amplifier-core base classes.
See tests/test_behavioral.py for the inherited tests.

The amplifier-core pytest plugin provides fixtures automatically:
- module_path: Detected path to this module
- module_type: Detected type (provider, tool, hook, etc.)
- provider_module, tool_module, etc.: Mounted module instances
"""

import ipaddress
import socket

import pytest


@pytest.fixture(autouse=True)
def offline_network_is_loopback_only(request, monkeypatch):
    """Fail before DNS/connect rather than accidentally contact a hosted API.

    An SDK added native counting to previously mocked-generation tests. Those
    optional unmocked counts must be unavailable offline, not real account calls.
    Explicitly marked live tests keep their separately admitted behavior.
    """
    if request.node.get_closest_marker("live"):
        return
    original_resolve = socket.getaddrinfo
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex

    def loopback(host):
        if host == "localhost":
            return True
        try:
            return ipaddress.ip_address(host).is_loopback
        except ValueError:
            return False

    def resolve(host, *args, **kwargs):
        if not loopback(host):
            raise socket.gaierror("Offline tests permit loopback only")
        return original_resolve(host, *args, **kwargs)

    def connect(sock, address):
        if isinstance(address, tuple) and not loopback(address[0]):
            raise OSError("Offline tests permit loopback only")
        return original_connect(sock, address)

    def connect_ex(sock, address):
        if isinstance(address, tuple) and not loopback(address[0]):
            raise OSError("Offline tests permit loopback only")
        return original_connect_ex(sock, address)

    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)


@pytest.fixture(autouse=True)
def offline_contract_credentials(request, monkeypatch):
    """Mount real providers for offline inherited contracts, without secrets."""
    if request.node.path.name in {"test_behavioral.py", "test_validation.py"}:
        monkeypatch.setenv("OPENAI_API_KEY", "offline-contract-placeholder")
