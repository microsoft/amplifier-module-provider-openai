import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from amplifier_decisions import DecisionRequest, Question
from amplifier_module_provider_openai.decisions import (
    OpenAIDecisionBackend,
    register_decision_backend,
)
from openai import AsyncOpenAI

pytestmark = pytest.mark.asyncio


async def test_metadata_does_not_claim_compatible_chat_providers_have_decisions():
    from amplifier_module_provider_openai import OpenAIProvider

    class CompatibleProvider(OpenAIProvider):
        name = "other-chat-endpoint"

    ordinary = OpenAIProvider.__new__(OpenAIProvider)
    compatible = CompatibleProvider.__new__(CompatibleProvider)
    assert ordinary.get_decision_info()["available"] is True
    assert compatible.get_decision_info()["available"] is False
    assert not hasattr(ordinary, "_client")


def request(**kwargs):
    return DecisionRequest(
        "Synthetic form",
        (
            Question("present", "predicate", "Is Save visible?"),
            Question(
                "target", "choice", "Choose a target", {"a": "Save", "b": "Cancel"}
            ),
            Question(
                "progress",
                "score",
                "Rate progress",
                levels=("None", "Partial", "Complete"),
            ),
        ),
        allow_external_state=True,
        **kwargs,
    )


def payload():
    return {
        "model": "gpt-6-luna",
        "answers": [
            {"name": "present", "type": "predicate", "probability": 0.9},
            {
                "name": "target",
                "type": "choice",
                "choice": "a",
                "confidence": 0.8,
                "probabilities": [
                    {"value": "a", "probability": 0.9},
                    {"value": "b", "probability": 0.1},
                ],
            },
            {
                "name": "progress",
                "type": "score",
                "score": 1.1,
                "confidence": 0.8,
                "probabilities": [
                    {"value": 0, "probability": 0.1},
                    {"value": 1, "probability": 0.7},
                    {"value": 2, "probability": 0.2},
                ],
            },
        ],
        "usage": {"input_tokens": 5, "output_tokens": 0, "total_tokens": 5},
    }


async def test_same_provider_client_account_endpoint_and_transport_without_ambient_key(
    monkeypatch,
):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    seen = []

    def handle(req):
        seen.append(req)
        return httpx.Response(
            200, json=payload(), headers={"x-request-id": "fixture-id"}
        )

    client = AsyncOpenAI(
        api_key="fixture-account-key",
        base_url="https://configured.test/v1/",
        organization="fixture-org",
        project="fixture-project",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
    )
    backend = OpenAIDecisionBackend(
        SimpleNamespace(client=client), {"model": "gpt-6-luna"}
    )
    try:
        result = await backend.decide(request())
        assert all(answer.status == "answered" for answer in result.answers.values())
        assert result.answers["progress"].value == 1.1
        assert result.request_id == "fixture-id" and result.usage["input_tokens"] == 5
        assert (
            len(seen) == 1
            and str(seen[0].url) == "https://configured.test/v1/decisions"
        )
        assert seen[0].headers["Authorization"] == "Bearer fixture-account-key"
        assert seen[0].headers["OpenAI-Organization"] == "fixture-org"
        assert seen[0].headers["OpenAI-Project"] == "fixture-project"
        assert set(json.loads(seen[0].content)) == {"model", "input", "questions"}
        assert "fixture-account-key" not in repr(result)
        await backend.close()
        assert not client.is_closed()
    finally:
        await client.close()


@pytest.mark.parametrize("status", [400, 401, 429, 500])
async def test_sdk_errors_do_not_retry_or_disclose_payload(status):
    seen = []

    def handle(req):
        seen.append(req)
        return httpx.Response(status, json={"error": {"message": "PRIVATE INPUT"}})

    async with AsyncOpenAI(
        api_key="fixture",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
    ) as client:
        backend = OpenAIDecisionBackend(
            SimpleNamespace(client=client), {"model": "gpt-6-luna"}
        )
        result = await backend.decide(request())
        assert len(seen) == 1 and "PRIVATE" not in repr(result)
        assert result.answers["target"].reason == f"http_{status}"


async def test_closed_disabled_and_unconsented_do_not_initialize_client():
    class Provider:
        @property
        def client(self):
            raise AssertionError("Client must remain lazy")

    registry = {}
    coordinator = SimpleNamespace(
        get_capability=registry.get, register_capability=registry.__setitem__
    )
    register_decision_backend(coordinator, Provider(), {})()
    assert registry == {}
    cleanup = register_decision_backend(
        coordinator, Provider(), {"decisions": {"enabled": True, "model": "gpt-6-luna"}}
    )
    backend = registry["decision.backends"]["openai-decisions"]
    result = await backend.decide(DecisionRequest("private", request().questions))
    assert result.answers["target"].status == "invalid"
    replacement = object()
    registry["decision.backends"]["openai-decisions"] = replacement
    cleanup()
    assert registry["decision.backends"]["openai-decisions"] is replacement
    assert (await backend.decide(request())).answers["target"].reason == "closed"


async def test_cancellation_is_not_converted_to_fallback():
    async def handle(req):
        raise asyncio.CancelledError

    async with AsyncOpenAI(
        api_key="fixture",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
    ) as client:
        backend = OpenAIDecisionBackend(
            SimpleNamespace(client=client), {"model": "gpt-6-luna"}
        )
        with pytest.raises(asyncio.CancelledError):
            await backend.decide(request())


async def test_named_models_share_account_and_close_without_removing_other_owners():
    registry = {}
    coordinator = SimpleNamespace(get_capability=registry.get, register_capability=registry.__setitem__)
    provider = SimpleNamespace(client=object())
    cleanup = register_decision_backend(coordinator, provider, {'decisions': {
        'enabled': True, 'backends': {
            'browser': {'enabled': True, 'model': 'browser-model'},
            'desktop': {'enabled': True, 'model': 'desktop-model'},
            'disabled': {'enabled': False, 'model': 'unused'},
        }}})
    backends = registry['decision.backends']
    assert set(backends) == {'browser', 'desktop'}
    assert backends['browser'].provider is provider
    assert backends['desktop'].provider is provider
    other = object()
    registry['decision.backends']['other'] = other
    cleanup()
    assert registry['decision.backends'] == {'other': other}
