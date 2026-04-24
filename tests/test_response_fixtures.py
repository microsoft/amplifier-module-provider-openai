"""Fixture-replay tests for gpt-5.5 responses.

Replays response payloads recorded from the live API through the provider's
parser and asserts on the resulting ChatResponse. Mock tests assert on the
mock's return value (whatever we set it to); fixture replay catches
response-shape drift that mocks cannot.

Maintenance policy: the parser must accept unknown optional fields
gracefully. Tests assert on the shape of fields we use, not the absence
of fields we don't.
"""

import asyncio
import json
from pathlib import Path

from amplifier_core.message_models import ChatRequest, Message
from openai.types.responses import Response

from amplifier_module_provider_openai import OpenAIProvider


FIXTURES_DIR = Path(__file__).parent / "fixtures" / "responses"


def _load_fixture(name: str) -> Response:
    with open(FIXTURES_DIR / name) as f:
        payload = json.load(f)
    return Response.model_validate(payload)


def _make_async_returning(value):
    async def _mock(**_kwargs):
        return value

    return _mock


def _run_through_parser(fixture_name: str):
    response = _load_fixture(fixture_name)
    provider = OpenAIProvider(
        api_key="test-key",
        config={"max_retries": 0, "use_streaming": False},
    )
    provider.client.responses.create = _make_async_returning(response)
    request = ChatRequest(messages=[Message(role="user", content="test")])
    return asyncio.run(provider.complete(request))


# ---------------------------------------------------------------------------
# Basic gpt-5.5 response
# ---------------------------------------------------------------------------


def test_gpt_5_5_basic_loads_as_response():
    response = _load_fixture("gpt-5-5-basic.json")
    assert isinstance(response, Response)
    assert response.status == "completed"
    assert response.model.startswith("gpt-5.5")


def test_gpt_5_5_basic_parses_to_chat_response():
    result = _run_through_parser("gpt-5-5-basic.json")
    assert result is not None
    assert result.content is not None


def test_gpt_5_5_basic_preserves_text():
    result = _run_through_parser("gpt-5-5-basic.json")
    text = "".join(getattr(b, "text", "") or "" for b in result.content or [])
    assert "Paris" in text


# ---------------------------------------------------------------------------
# gpt-5.5 with reasoning — rs_* IDs and reasoning_tokens round-trip
# ---------------------------------------------------------------------------


def test_gpt_5_5_reasoning_has_expected_blocks():
    response = _load_fixture("gpt-5-5-reasoning.json")
    output_types = [getattr(b, "type", None) for b in response.output]
    assert "reasoning" in output_types
    assert "message" in output_types


def test_gpt_5_5_reasoning_round_trip():
    result = _run_through_parser("gpt-5-5-reasoning.json")
    assert result is not None
    text = "".join(getattr(b, "text", "") or "" for b in result.content or [])
    # Recorded answer to "13 * 17" was 221.
    assert "221" in text


def test_gpt_5_5_reasoning_tokens_present():
    response = _load_fixture("gpt-5-5-reasoning.json")
    assert response.usage.output_tokens_details is not None
    assert response.usage.output_tokens_details.reasoning_tokens > 0
