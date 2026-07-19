"""Deterministic, zero-network regression test for issue #321.

Bug (before fix): when response-chaining is active, the OpenAI provider built
an outbound request that carried BOTH the full local `input` message array
(the entire conversation history) AND a `previous_response_id` at the same
time. Because previous_response_id already tells the server to load the full
prior request+response as server-side state, re-sending the whole local
history double-counts every prior token server-side.

Fix (issue #321): when previous_response_id is attached, `input` is trimmed to
the DELTA only -- the messages added after the chained assistant turn. The
prior history lives in server-side state referenced by previous_response_id.

This test asserts the FIXED behavior:
  - previous_response_id is still attached (chaining still works), AND
  - input no longer contains the already-chained prior turns (it is the delta).

ZERO NETWORK CALLS: `provider.client.responses.create` is monkeypatched with
an AsyncMock before any call is made. No real OpenAI SDK network request is
possible -- the mock captures kwargs and returns a canned, in-memory
DummyResponse synchronously.

Run with:
    cd /Users/salil/Development/msft/gh-issues-support/amplifier-issue-321/provider-openai
    uv run pytest tests/test_issue_321_double_context_proof.py -s -v
"""

import asyncio
import json
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

from amplifier_core.message_models import ChatRequest, Message

from amplifier_module_provider_openai import OpenAIProvider
from amplifier_module_provider_openai._constants import METADATA_RESPONSE_ID


class DummyResponse:
    """Minimal in-memory response stub - matches the shape
    _convert_to_chat_response() needs. No network I/O involved.
    """

    def __init__(self, response_id: str = "resp_test_dummy"):
        self.output = [
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text="Hi")],
            )
        ]
        self.usage = SimpleNamespace(input_tokens=1, output_tokens=1)
        self.status = "completed"
        self.id = response_id


def _make_provider(**config_overrides) -> OpenAIProvider:
    config = {"max_retries": 0, "use_streaming": False, **config_overrides}
    return OpenAIProvider(api_key="test-key-never-used", config=config)


def _request_two_plus_turns(prior_response_id: str = "resp_TEST123") -> ChatRequest:
    """3-message ChatRequest: prior user, prior assistant (carrying the chaining
    metadata), and a new user message. The first two turns are already covered
    by prior_response_id server-side; only the new user turn is the delta.
    """
    msgs = [
        Message(role="user", content="What's the capital of France?"),
        Message(
            role="assistant",
            content="Paris.",
            metadata={METADATA_RESPONSE_ID: prior_response_id},
        ),
        Message(role="user", content="And its population?"),
    ]
    return ChatRequest(messages=msgs)


def _flatten_input_text(input_items: list[Any]) -> str:
    """Serialize the whole input array to a searchable string so we can assert
    which turns are (and are not) present regardless of the exact item shape."""
    try:
        return json.dumps(input_items, default=str).lower()
    except Exception:
        return str(input_items).lower()


def test_chaining_trims_input_to_delta_when_previous_response_id_attached():
    """REGRESSION (issue #321): chain_active=True => outbound params attach
    previous_response_id AND send only the delta in `input` (the new user turn),
    NOT the full prior history that is already covered server-side.
    """
    provider = _make_provider(
        default_model="gpt-5-mini",
        enable_response_chaining=True,
        enable_state=True,
    )

    # --- Monkeypatch: capture params, never touch the network -------------
    captured: dict[str, Any] = {}

    async def _fake_create(*args, **kwargs):
        captured.update(kwargs)
        return DummyResponse(response_id="resp_from_mock")

    provider.client.responses.create = AsyncMock(side_effect=_fake_create)
    provider.client.responses.stream = AsyncMock(
        side_effect=RuntimeError(
            "responses.stream() should not be called in this test path"
        )
    )

    request = _request_two_plus_turns("resp_TEST123")

    # --- Execute (no network: client methods are mocked above) ------------
    asyncio.run(provider.complete(request))

    mock = cast(AsyncMock, provider.client.responses.create)
    assert mock.await_count == 1, (
        f"Expected exactly one call to the mocked create(); got {mock.await_count}."
    )
    assert captured, "No params were captured - mock was not invoked as expected."

    # --- Chaining is still active -----------------------------------------
    assert captured.get("previous_response_id") == "resp_TEST123", (
        f"Expected previous_response_id='resp_TEST123', got "
        f"{captured.get('previous_response_id')!r}"
    )

    # --- The core issue #321 fix assertions -------------------------------
    assert "input" in captured, f"params missing 'input': {sorted(captured.keys())}"
    assert isinstance(captured["input"], list)

    flat = _flatten_input_text(captured["input"])

    # The new user turn (the delta) MUST be present.
    assert "population" in flat, (
        f"Delta message ('And its population?') missing from input: {captured['input']}"
    )
    # The already-chained prior turns MUST NOT be re-sent (that was the bug).
    assert "capital of france" not in flat, (
        "BUG #321 present: prior user turn re-sent in input despite "
        f"previous_response_id being attached. input={captured['input']}"
    )
    assert "paris" not in flat, (
        "BUG #321 present: prior assistant turn re-sent in input despite "
        f"previous_response_id being attached. input={captured['input']}"
    )
    # Delta is exactly the one new user turn (no developer messages in this case).
    assert len(captured["input"]) == 1, (
        f"Expected input trimmed to the 1-message delta, got "
        f"{len(captured['input'])}: {captured['input']}"
    )

    # --- Evidence printout -------------------------------------------------
    print("\n=== ISSUE #321 FIX EVIDENCE ===")
    print("params['previous_response_id']:", captured["previous_response_id"])
    print("len(params['input']) (delta):", len(captured["input"]))
    print("input contains 'population' (delta present):", "population" in flat)
    print(
        "input contains 'capital of france' (should be False):",
        "capital of france" in flat,
    )
    print("input contains 'paris' (should be False):", "paris" in flat)
    print("===============================")


if __name__ == "__main__":
    test_chaining_trims_input_to_delta_when_previous_response_id_attached()
    print("PASS (run as standalone script)")
