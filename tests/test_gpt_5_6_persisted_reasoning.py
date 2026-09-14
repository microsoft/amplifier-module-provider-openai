"""Tests for reasoning.context (GPT-5.6 "persisted reasoning") passthrough.

Shape verified against the GPT-5.6 model guide (developers.openai.com,
fetched 2026-07-14): reasoning.context in {"auto", "current_turn", "all_turns"}.
"current_turn" trims rendered reasoning context on long agent loops -- the
documented mitigation for context_length_exceeded.

Forwarding is UNGATED: whenever the caller supplies `reasoning.context` in an
explicit `reasoning` dict, it is forwarded as-is (same stance as `mode` --
an explicit reasoning dict is a deliberate provider-specific override, and
the caller owns the consequences). The historical `enable_reasoning_context`
flag gate and the chain/store gate (both tied to the now-removed
`previous_response_id` chaining path) are gone -- the provider is
stateless-only. `_validate_reasoning_context` still rejects bad values.
"""

import asyncio
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from amplifier_core import llm_errors as kernel_errors
from amplifier_core.message_models import ChatRequest, Message, ThinkingBlock

from amplifier_module_provider_openai import (
    OpenAIChatResponse,
    OpenAIProvider,
    _validate_reasoning_context,
)
from amplifier_module_provider_openai._response_handling import (
    convert_response_with_accumulated_output,
)


def _make_provider(**config_overrides) -> OpenAIProvider:
    config = {"max_retries": 0, "use_streaming": False, **config_overrides}
    return OpenAIProvider(api_key="test-key", config=config)


def _simple_request() -> ChatRequest:
    return ChatRequest(messages=[Message(role="user", content="Hello")])


class DummyResponse:
    def __init__(self):
        self.output = [
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text="Hi")],
            )
        ]
        self.usage = SimpleNamespace(input_tokens=1, output_tokens=1)
        self.status = "completed"
        self.id = "resp_test"


def _captured_params(provider: OpenAIProvider) -> Any:
    mock = cast(AsyncMock, provider.client.responses.create)
    return mock.call_args.kwargs


# ---------------------------------------------------------------------------
# reasoning.context validator (value-shape only; unaffected by the flag/chain
# gates, which apply only at the forwarding site).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ok",
    [
        None,
        "high",
        {},
        {"context": "auto"},
        {"context": "current_turn"},
        {"context": "all_turns"},
    ],
)
def test_validate_reasoning_context_accepts(ok):
    _validate_reasoning_context(ok)  # must not raise


@pytest.mark.parametrize(
    "bad", [{"context": "later"}, {"context": ""}, {"context": "turn"}]
)
def test_validate_reasoning_context_rejects(bad):
    with pytest.raises(kernel_errors.InvalidRequestError):
        _validate_reasoning_context(bad)


# ---------------------------------------------------------------------------
# reasoning.context passthrough into the API call -- flag + chain/store gated
# ---------------------------------------------------------------------------


def test_reasoning_context_forwarded_ungated():
    """An explicit reasoning.context value is forwarded unconditionally --
    no flag, no chain/store gate. The provider is stateless-only; an
    explicit `reasoning` dict is a deliberate override the caller owns."""
    provider = _make_provider(default_model="gpt-5.6-sol")
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())
    asyncio.run(
        provider.complete(
            _simple_request(),
            reasoning={"effort": "high", "context": "current_turn"},
        )
    )
    reasoning = _captured_params(provider)["reasoning"]
    assert reasoning["context"] == "current_turn"
    assert reasoning["effort"] == "high"


def test_reasoning_context_absent_when_not_set():
    provider = _make_provider(default_model="gpt-5.6-sol")
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())
    asyncio.run(provider.complete(_simple_request(), reasoning={"effort": "medium"}))
    assert "context" not in _captured_params(provider)["reasoning"]


def test_reasoning_context_forwarded_even_with_leftover_legacy_config():
    """Stale `enable_reasoning_context` config (recognized-but-inert) must
    not suppress forwarding -- the flag gate no longer exists."""
    provider = _make_provider(
        default_model="gpt-5.6-sol", enable_reasoning_context=False
    )
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())
    asyncio.run(
        provider.complete(
            _simple_request(),
            reasoning={"effort": "high", "context": "current_turn"},
        )
    )
    reasoning = _captured_params(provider)["reasoning"]
    assert reasoning["context"] == "current_turn"


def test_reasoning_context_forwarded_on_continuation():
    """reasoning.context must survive an incomplete->continuation sequence.

    It lives inside the `reasoning` dict, already forwarded via the existing
    `if "reasoning" in params` continuation-forward line.
    """
    provider = _make_provider(default_model="gpt-5.6-sol")
    incomplete_resp = SimpleNamespace(
        status="incomplete", id="resp_incomplete", output=[], incomplete_details=None
    )
    provider.client.responses.create = AsyncMock(
        side_effect=[incomplete_resp, DummyResponse()]
    )
    asyncio.run(
        provider.complete(
            _simple_request(),
            reasoning={"effort": "high", "context": "current_turn"},
        )
    )

    calls = provider.client.responses.create.call_args_list
    assert len(calls) == 2
    for call in calls:
        assert call.kwargs["reasoning"]["context"] == "current_turn"


# ---------------------------------------------------------------------------
# `reasoning_context` -- the first-class config key (R0 companion fix)
#
# WHY IT EXISTS: before this key, `context` could only ride inside the LEGACY
# `reasoning` dict -- which the canonical `reasoning_effort` key outranks. An
# operator setting BOTH (the normal config shape) had their context silently
# dropped: `reasoning_param` was rebuilt from the effort alone and the legacy
# dict was never consulted. `test_config_context_survives_canonical_effort`
# is that exact failure.
#
# WHY IT IS SAFE TO SEND: measured live on our own stateless manual-replay
# path (t8p, gpt-5.6-terra, store=false, no chaining, reasoning items replayed
# inline in `input`): with no `context` field the API's effective mode is
# `all_turns`; with `context="current_turn"` it is honored and echoed back,
# 200 completed. Capture root:
# .amplifier/evaluation/treatment-validation/20260902-t8p-reasoning-context/
# ---------------------------------------------------------------------------


def test_config_reasoning_context_reaches_the_request():
    provider = _make_provider(
        default_model="gpt-5.6-sol",
        reasoning_effort="high",
        reasoning_context="current_turn",
    )
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())
    asyncio.run(provider.complete(_simple_request()))
    reasoning = _captured_params(provider)["reasoning"]
    assert reasoning["context"] == "current_turn"
    assert reasoning["effort"] == "high"


def test_config_context_survives_canonical_effort():
    """THE REGRESSION: `reasoning_effort` (canonical) wins over the legacy
    `reasoning` dict, so context set only in that dict never reached the wire.
    The first-class key composes with the canonical effort path instead."""
    provider = _make_provider(
        default_model="gpt-5.6-sol",
        reasoning_effort="medium",
        reasoning={"effort": "high", "context": "current_turn"},
    )
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())
    asyncio.run(provider.complete(_simple_request()))
    reasoning = _captured_params(provider)["reasoning"]
    # The legacy dict is ignored (documented, warned about at mount) ...
    assert reasoning["effort"] == "medium"
    assert "context" not in reasoning

    provider2 = _make_provider(
        default_model="gpt-5.6-sol",
        reasoning_effort="medium",
        reasoning_context="current_turn",
    )
    provider2.client.responses.create = AsyncMock(return_value=DummyResponse())
    asyncio.run(provider2.complete(_simple_request()))
    reasoning2 = _captured_params(provider2)["reasoning"]
    # ... but the first-class key composes with it.
    assert reasoning2 == {
        "effort": "medium",
        "summary": "detailed",
        "context": "current_turn",
    }


def test_explicit_dict_context_wins_over_config():
    """A caller-supplied `reasoning` dict is the strongest signal; the config
    default must never overwrite it."""
    provider = _make_provider(
        default_model="gpt-5.6-sol", reasoning_context="all_turns"
    )
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())
    asyncio.run(
        provider.complete(
            _simple_request(),
            reasoning={"effort": "high", "context": "current_turn"},
        )
    )
    assert _captured_params(provider)["reasoning"]["context"] == "current_turn"


def test_per_call_reasoning_context_kwarg_overrides_config():
    provider = _make_provider(
        default_model="gpt-5.6-sol",
        reasoning_effort="high",
        reasoning_context="all_turns",
    )
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())
    asyncio.run(provider.complete(_simple_request(), reasoning_context="current_turn"))
    assert _captured_params(provider)["reasoning"]["context"] == "current_turn"


def test_config_reasoning_context_absent_by_default():
    """Default config must not add the field -- the pre-R0 request shape."""
    provider = _make_provider(default_model="gpt-5.6-sol", reasoning_effort="high")
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())
    asyncio.run(provider.complete(_simple_request()))
    assert "context" not in _captured_params(provider)["reasoning"]


def test_invalid_config_reasoning_context_fails_loud_at_mount():
    with pytest.raises(kernel_errors.InvalidRequestError):
        _make_provider(default_model="gpt-5.6-sol", reasoning_context="later")


def test_reasoning_context_without_a_reasoning_param_warns_and_sends_nothing(caplog):
    """A config key that silently does nothing is the failure mode this repo
    already warns about for every inert key. Same treatment here."""
    import logging

    provider = _make_provider(default_model="gpt-5.4", reasoning_context="current_turn")
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())
    with caplog.at_level(logging.WARNING):
        asyncio.run(provider.complete(_simple_request()))
    params = _captured_params(provider)
    assert "reasoning" not in params
    assert "reasoning_context" in caplog.text


def test_config_reasoning_context_forwarded_on_continuation():
    provider = _make_provider(
        default_model="gpt-5.6-sol",
        reasoning_effort="high",
        reasoning_context="current_turn",
    )
    incomplete_resp = SimpleNamespace(
        status="incomplete", id="resp_incomplete", output=[], incomplete_details=None
    )
    provider.client.responses.create = AsyncMock(
        side_effect=[incomplete_resp, DummyResponse()]
    )
    asyncio.run(provider.complete(_simple_request()))
    calls = provider.client.responses.create.call_args_list
    assert len(calls) == 2
    for call in calls:
        assert call.kwargs["reasoning"]["context"] == "current_turn"


# ---------------------------------------------------------------------------
# GPT-5.6 Luna/Terra stateless replay defaults and reasoning-item fidelity
# ---------------------------------------------------------------------------


def _thinking_message(tag: str) -> Message:
    return Message(
        role="assistant",
        content=[
            ThinkingBlock(
                thinking=f"summary {tag}",
                content=[
                    {
                        "encrypted_content": f"ENC_{tag}",
                        "id": f"rs_{tag}",
                        "summary": f"summary {tag}",
                    }
                ],
            )
        ],
    )


def _multi_turn_reasoning_request() -> ChatRequest:
    return ChatRequest(
        messages=[
            Message(role="user", content="first human"),
            _thinking_message("first"),
            Message(role="user", content="second human"),
            _thinking_message("second"),
        ]
    )


def _replayed_reasoning_ids(params: dict[str, Any]) -> list[str]:
    return [
        item["id"]
        for item in params["input"]
        if item.get("type") == "reasoning"
    ]


@pytest.mark.parametrize(
    "selected_model",
    ["gpt-5.6-luna", "gpt-5.6-terra-2026-09-01"],
)
def test_effective_per_call_model_selects_luna_or_terra_all_turns(
    selected_model: str,
) -> None:
    request = _multi_turn_reasoning_request()

    older_default = _make_provider(default_model="gpt-5.5")
    older_default.client.responses.create = AsyncMock(return_value=DummyResponse())
    asyncio.run(older_default.complete(request, model=selected_model))
    assert _replayed_reasoning_ids(_captured_params(older_default)) == [
        "rs_first",
        "rs_second",
    ]


def test_effective_per_call_model_keeps_older_and_sol_models_turn_scoped() -> None:
    request = _multi_turn_reasoning_request()

    luna_default = _make_provider(default_model="gpt-5.6-luna")
    luna_default.client.responses.create = AsyncMock(return_value=DummyResponse())
    asyncio.run(luna_default.complete(request, model="gpt-5.5"))
    assert _replayed_reasoning_ids(_captured_params(luna_default)) == ["rs_second"]

    sol_default = _make_provider(default_model="gpt-5.6-sol")
    sol_default.client.responses.create = AsyncMock(return_value=DummyResponse())
    asyncio.run(sol_default.complete(request))
    assert _replayed_reasoning_ids(_captured_params(sol_default)) == ["rs_second"]


@pytest.mark.parametrize(
    ("config_scope", "per_call_scope", "expected"),
    [
        (None, None, ["rs_first", "rs_second"]),
        ("turn", None, ["rs_second"]),
        ("none", None, []),
        ("none", "all", ["rs_first", "rs_second"]),
        ("all", "turn", ["rs_second"]),
    ],
)
def test_explicit_or_per_call_replay_scope_overrides_luna_default(
    config_scope: str | None, per_call_scope: str | None, expected: list[str]
) -> None:
    provider = _make_provider(
        default_model="gpt-5.6-luna", reasoning_replay_scope=config_scope
    )
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())

    asyncio.run(
        provider.complete(
            _multi_turn_reasoning_request(),
            **(
                {"reasoning_replay_scope": per_call_scope}
                if per_call_scope is not None
                else {}
            ),
        )
    )

    assert _replayed_reasoning_ids(_captured_params(provider)) == expected


def test_invalid_per_call_replay_scope_warns_before_turn_fallback(caplog) -> None:
    provider = _make_provider(default_model="gpt-5.6-luna")
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())
    asyncio.run(
        provider.complete(_multi_turn_reasoning_request(), reasoning_replay_scope="typo")
    )
    assert _replayed_reasoning_ids(_captured_params(provider)) == ["rs_second"]
    assert "Unknown reasoning_replay_scope" in caplog.text


def _reasoning_item(*, as_dict: bool) -> Any:
    item = {
        "type": "reasoning",
        "id": "rs_fidelity",
        "encrypted_content": "ENC_FIDELITY",
        "summary": [],
        "content": [{"type": "reasoning_text", "text": "verbatim reasoning"}],
        "status": "completed",
        "unrecognized_extra": "must not be persisted",
    }
    return item if as_dict else SimpleNamespace(**item)


def _response_with_reasoning(*, as_dict: bool) -> SimpleNamespace:
    return SimpleNamespace(
        output=[_reasoning_item(as_dict=as_dict)],
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        status="completed",
        id="resp_fidelity",
    )


def _captured_reasoning_state(response: Any) -> dict[str, Any]:
    thinking = next(
        block for block in response.content if getattr(block, "type", None) == "thinking"
    )
    return thinking.content[0]


@pytest.mark.parametrize("as_dict", [False, True], ids=["sdk", "raw_dict"])
def test_both_response_capture_paths_preserve_reasoning_content_and_status(
    as_dict: bool,
) -> None:
    raw_response = _response_with_reasoning(as_dict=as_dict)
    provider = _make_provider()

    native_state = _captured_reasoning_state(
        provider._convert_to_chat_response(raw_response)
    )
    accumulated_state = _captured_reasoning_state(
        convert_response_with_accumulated_output(
            final_response=raw_response,
            accumulated_output=list(raw_response.output),
            continuation_count=0,
            chat_response_class=OpenAIChatResponse,
        )
    )
    expected = {
        "encrypted_content": "ENC_FIDELITY",
        "id": "rs_fidelity",
        "summary": None,
        "content": [{"type": "reasoning_text", "text": "verbatim reasoning"}],
        "status": "completed",
    }
    assert native_state == accumulated_state == expected


def test_luna_replays_prior_reasoning_content_across_a_new_human_message():
    provider = _make_provider(
        default_model="gpt-5.6-luna",
        reasoning={"effort": "low", "summary": "detailed"},
    )
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())
    captured = provider._convert_to_chat_response(
        _response_with_reasoning(as_dict=False)
    )
    request = ChatRequest(
        messages=[
            Message(role="user", content="first human"),
            Message(role="assistant", content=captured.content),
            Message(role="user", content="new human"),
        ]
    )

    asyncio.run(provider.complete(request))

    params = _captured_params(provider)
    assert params["store"] is False
    assert params["reasoning"] == {"effort": "low", "summary": "detailed"}
    assert "context" not in params["reasoning"]
    assert [
        item
        for item in params["input"]
        if item.get("type") == "reasoning"
    ] == [
        {
            "type": "reasoning",
            "id": "rs_fidelity",
            "encrypted_content": "ENC_FIDELITY",
            "summary": [],
            "content": [{"type": "reasoning_text", "text": "verbatim reasoning"}],
            "status": "completed",
        }
    ]
