"""Tests for `extra_request_params` -- the documented escape hatch for
Responses API parameters this provider does not model.

Contract: dict[str, Any], settings-only (never a ConfigField), merged into
`params` LAST (after every provider-computed key -- user always wins),
applied on both the primary request and the incomplete-continuation
request. Clobbers are warned once per key per provider instance.
"""

import asyncio
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from amplifier_core.message_models import ChatRequest, Message

from amplifier_module_provider_openai import OpenAIProvider


def _make_provider(**config_overrides) -> OpenAIProvider:
    config = {"max_retries": 0, "use_streaming": False, **config_overrides}
    return OpenAIProvider(api_key="test-key", config=config)


def _simple_request() -> ChatRequest:
    return ChatRequest(messages=[Message(role="user", content="Hello")])


class DummyResponse:
    def __init__(self, response_id: str = "resp_test"):
        self.output = [
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text="Hi")],
            )
        ]
        self.usage = SimpleNamespace(input_tokens=1, output_tokens=1)
        self.status = "completed"
        self.id = response_id


def _captured_params(provider: OpenAIProvider) -> Any:
    mock = cast(AsyncMock, provider.client.responses.create)
    return mock.call_args.kwargs


# ---------------------------------------------------------------------------
# 1. Non-dict value -> ValueError at construction, naming the key
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["nope", ["a"], 42])
def test_non_dict_value_raises_at_construction(bad):
    with pytest.raises(ValueError, match="extra_request_params"):
        _make_provider(extra_request_params=bad)


# ---------------------------------------------------------------------------
# 2. Absent / None / {} -> no-op, no warning, params unaffected
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("absent_value", [None, {}])
def test_absent_or_empty_is_a_noop(absent_value, caplog):
    import logging

    caplog.set_level(logging.WARNING, logger="amplifier_module_provider_openai")
    if absent_value is None:
        provider = _make_provider()
    else:
        provider = _make_provider(extra_request_params=absent_value)
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())
    asyncio.run(provider.complete(_simple_request()))

    assert not any(
        "extra_request_params" in r.message
        for r in caplog.records
        if r.levelno == logging.WARNING
    )


# ---------------------------------------------------------------------------
# 3. Novel key -> present in request kwargs, no warning (nothing clobbered)
# ---------------------------------------------------------------------------


def test_novel_key_reaches_the_wire_with_no_warning(caplog):
    import logging

    caplog.set_level(logging.WARNING, logger="amplifier_module_provider_openai")
    provider = _make_provider(extra_request_params={"seed": 42})
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())
    asyncio.run(provider.complete(_simple_request()))

    assert _captured_params(provider)["seed"] == 42
    assert not any(
        "extra_request_params" in r.message
        for r in caplog.records
        if r.levelno == logging.WARNING
    )


# ---------------------------------------------------------------------------
# 4. Collision -> user wins, exactly one warning naming the key
# ---------------------------------------------------------------------------


def test_collision_user_wins_and_warns_once(caplog):
    import logging

    caplog.set_level(logging.WARNING, logger="amplifier_module_provider_openai")
    provider = _make_provider(
        temperature=0.2, extra_request_params={"temperature": 0.9}
    )
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())
    asyncio.run(provider.complete(_simple_request()))

    assert _captured_params(provider)["temperature"] == 0.9

    warnings = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and "extra_request_params" in r.message
    ]
    assert len(warnings) == 1
    assert "temperature" in warnings[0].message


# ---------------------------------------------------------------------------
# 5. Multi-key collision -> one warning naming all clobbered keys, sorted
# ---------------------------------------------------------------------------


def test_multi_key_collision_names_all_sorted(caplog):
    import logging

    caplog.set_level(logging.WARNING, logger="amplifier_module_provider_openai")
    provider = _make_provider(
        temperature=0.2,
        truncation="auto",
        extra_request_params={"temperature": 0.9, "truncation": "disabled"},
    )
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())
    asyncio.run(provider.complete(_simple_request()))

    warnings = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and "extra_request_params" in r.message
    ]
    assert len(warnings) == 1
    idx_temp = warnings[0].message.find("temperature")
    idx_trunc = warnings[0].message.find("truncation")
    assert idx_temp != -1 and idx_trunc != -1
    assert idx_temp < idx_trunc, "clobbered keys must be named sorted"


# ---------------------------------------------------------------------------
# 6. Warning does not repeat on a second request with the same collision
# ---------------------------------------------------------------------------


def test_collision_warning_does_not_repeat_across_requests(caplog):
    import logging

    caplog.set_level(logging.WARNING, logger="amplifier_module_provider_openai")
    provider = _make_provider(extra_request_params={"temperature": 0.9})
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())

    asyncio.run(provider.complete(_simple_request(), temperature=0.1))
    asyncio.run(provider.complete(_simple_request(), temperature=0.1))

    warnings = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and "extra_request_params" in r.message
    ]
    assert len(warnings) == 1


# ---------------------------------------------------------------------------
# 7 & 8. store override -- including overriding background mode's forced True
# ---------------------------------------------------------------------------


def test_store_true_overrides_stateless_default():
    provider = _make_provider(extra_request_params={"store": True})
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())
    asyncio.run(provider.complete(_simple_request()))
    assert _captured_params(provider)["store"] is True


def test_store_false_overrides_background_mode_forced_true():
    """Documents that last-merge really is last: extra_request_params can
    override even background mode's forced store=True."""
    provider = _make_provider(extra_request_params={"store": False})
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())
    asyncio.run(provider.complete(_simple_request(), background=True))
    assert _captured_params(provider)["store"] is False


# ---------------------------------------------------------------------------
# 9 & 10. Continuation regression guard
# ---------------------------------------------------------------------------


def test_extra_request_params_survive_continuation():
    provider = _make_provider(extra_request_params={"seed": 42})
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
        assert call.kwargs.get("seed") == 42, (
            "extra_request_params must apply to the continuation request too"
        )


def test_instructions_still_propagate_alongside_extras():
    provider = _make_provider(extra_request_params={"seed": 42})
    incomplete_resp = SimpleNamespace(
        status="incomplete", id="resp_incomplete", output=[], incomplete_details=None
    )
    provider.client.responses.create = AsyncMock(
        side_effect=[incomplete_resp, DummyResponse()]
    )
    request = ChatRequest(
        messages=[
            Message(role="system", content="be terse"),
            Message(role="user", content="Hi"),
        ]
    )
    asyncio.run(provider.complete(request))

    calls = provider.client.responses.create.call_args_list
    assert len(calls) == 2
    for call in calls:
        assert call.kwargs.get("seed") == 42
        assert call.kwargs.get("instructions") == "be terse"
