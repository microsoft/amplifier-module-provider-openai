"""Mount rollback regressions using the installed Core coordinator."""

import asyncio
from types import SimpleNamespace

import pytest
import pytest_asyncio
from amplifier_core import AmplifierSession
from amplifier_module_provider_openai import mount
from amplifier_module_provider_openai.decisions import register_decision_backend


@pytest_asyncio.fixture
async def coordinator():
    session = AmplifierSession(
        {
            "session": {
                "orchestrator": {"module": "unused-fixture-loop"},
                "context": {"module": "unused-fixture-context"},
            },
            "providers": [],
        }
    )
    try:
        yield session.coordinator
    finally:
        await session.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [RuntimeError, asyncio.CancelledError])
@pytest.mark.parametrize("replace", [False, True])
async def test_failed_registration_unpublishes_owned_backend(
    coordinator, monkeypatch, error, replace
):
    prior, replacement = object(), object()
    coordinator.register_capability("decision.backends", {"unrelated": prior})
    seen = []
    original = coordinator.register_contributor

    def fail(channel, name, contributor):
        if name == "provider-openai-decisions":
            seen.append(coordinator.get_capability("decision.backends")["fixture"])
            if replace:
                coordinator.register_capability(
                    "decision.backends", {"unrelated": prior, "fixture": replacement}
                )
            raise error("synthetic observer registration failure")
        return original(channel, name, contributor)

    monkeypatch.setattr(coordinator, "register_contributor", fail)
    with pytest.raises(error):
        register_decision_backend(
            coordinator,
            SimpleNamespace(client=object()),
            {"decisions": {"enabled": True, "id": "fixture", "model": "fixture-model"}},
        )
    current = coordinator.get_capability("decision.backends")
    assert current == (
        {"unrelated": prior, "fixture": replacement}
        if replace
        else {"unrelated": prior}
    ), "failed registration left its backend published"
    assert seen[0].closed, "a retained failed backend must not remain executable"


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [RuntimeError, asyncio.CancelledError])
@pytest.mark.parametrize("replace", [False, True])
async def test_later_mount_failure_unpublishes_owned_backend(
    coordinator, monkeypatch, error, replace
):
    prior, replacement = object(), object()
    coordinator.register_capability("decision.backends", {"unrelated": prior})
    seen = []
    original = coordinator.register_contributor

    def fail(channel, name, contributor):
        if channel == "session.cost" and name == "provider-openai":
            seen.append(coordinator.get_capability("decision.backends")["fixture"])
            if replace:
                coordinator.register_capability(
                    "decision.backends", {"unrelated": prior, "fixture": replacement}
                )
            raise error("synthetic later mount failure")
        return original(channel, name, contributor)

    monkeypatch.setattr(coordinator, "register_contributor", fail)
    with pytest.raises(error):
        await mount(
            coordinator,
            {
                "api_key": "fixture-not-a-real-key",
                "decisions": {
                    "enabled": True,
                    "id": "fixture",
                    "model": "fixture-model",
                },
            },
        )
    current = coordinator.get_capability("decision.backends")
    assert current == (
        {"unrelated": prior, "fixture": replacement}
        if replace
        else {"unrelated": prior}
    ), "later failed mount left its backend published"
    assert seen[0].closed, "a retained failed backend must not remain executable"


@pytest.mark.asyncio
async def test_rollback_closes_provider_even_when_a_removal_fails(
    coordinator, monkeypatch
):
    from unittest.mock import AsyncMock

    from amplifier_module_provider_openai import OpenAIProvider, images

    closed = AsyncMock()
    monkeypatch.setattr(OpenAIProvider, "close", closed)

    def image_registration(*args):
        def remove():
            raise RuntimeError("synthetic cleanup failure")

        return remove

    monkeypatch.setattr(images, "register_image_backend", image_registration)
    original = coordinator.register_contributor
    cancellation = asyncio.CancelledError("synthetic cancelled mount")

    def fail(channel, name, callback):
        if channel == "session.cost":
            raise cancellation
        return original(channel, name, callback)

    monkeypatch.setattr(coordinator, "register_contributor", fail)
    with pytest.raises(asyncio.CancelledError) as caught:
        await mount(
            coordinator,
            {
                "api_key": "synthetic",
                "decisions": {"enabled": True, "id": "fixture", "model": "fixture"},
            },
        )
    assert caught.value is cancellation
    assert coordinator.get_capability("decision.backends") == {}
    closed.assert_awaited_once()
    assert any("RuntimeError" in note for note in cancellation.__notes__)
