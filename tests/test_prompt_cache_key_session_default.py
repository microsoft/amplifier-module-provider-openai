"""Tests for the session-scoped `prompt_cache_key` default.

Background (see README "prompt_cache_key" section and the treatment's PR
body for the full evidence chain): OpenAI's implicit prompt cache is
best-effort and routing-dependent. Without a `prompt_cache_key`, requests
are routed unhinted and can fall back to the most-replicated prefix under
process churn / history rewriting. `prompt_cache_key` is OpenAI's
documented lever for this -- a stable key keeps a logical conversation
pinned to one cache shard -- but the provider previously only ever sourced
it from static config, so in practice it was always None.

This module verifies the new fallback chain:

    per-call kwarg > config value (including the "" opt-out) > session
    identity (`self.coordinator.session_id`) > None

Precedence is unchanged for the first two links; only the third link (the
session-identity default) is new. See `_default_prompt_cache_key`'s
docstring in `amplifier_module_provider_openai/__init__.py` for exactly
which coordinator attribute is used and why it is a suitable identity.
"""

import asyncio
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

from amplifier_core.message_models import ChatRequest, Message

from amplifier_module_provider_openai import OpenAIProvider

# ---------------------------------------------------------------------------
# Helpers (mirrors test_cache_params.py / test_cache_defaults.py pattern)
# ---------------------------------------------------------------------------


def _make_coordinator(session_id: Any = "session-aaa") -> MagicMock:
    """A minimal coordinator stub exposing just what the provider touches.

    `session_id` defaults to a plain string (the normal case: a real
    Amplifier session). Pass a non-string (or omit the attribute entirely
    via `spec=[]`) to simulate a coordinator that exposes no usable
    session identity.
    """
    coordinator = MagicMock()
    coordinator.session_id = session_id
    coordinator.get_capability = MagicMock(return_value=None)
    coordinator.hooks = MagicMock()
    coordinator.hooks.emit = AsyncMock()
    return coordinator


def _make_provider(coordinator: Any = None, **config_overrides) -> OpenAIProvider:
    config = {"max_retries": 0, "use_streaming": False, **config_overrides}
    return OpenAIProvider(api_key="test-key", config=config, coordinator=coordinator)


def _simple_request() -> ChatRequest:
    return ChatRequest(messages=[Message(role="user", content="Hello")])


class DummyResponse:
    """Minimal response stub (mirrors test_cache_params.py)."""

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
    """Return the kwargs dict passed to the mocked create() call."""
    mock = cast(AsyncMock, provider.client.responses.create)
    return mock.call_args.kwargs


def _mock_create(provider: OpenAIProvider) -> None:
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())


# ---------------------------------------------------------------------------
# _default_prompt_cache_key() -- direct unit tests of the helper
# ---------------------------------------------------------------------------


def test_helper_returns_none_without_coordinator():
    """No coordinator mounted -> no session identity reachable -> None."""
    provider = _make_provider(coordinator=None)
    assert provider._default_prompt_cache_key() is None


def test_helper_returns_session_id_when_coordinator_present():
    """coordinator.session_id (a string) is used verbatim."""
    provider = _make_provider(coordinator=_make_coordinator("sess-123"))
    assert provider._default_prompt_cache_key() == "sess-123"


def test_helper_returns_none_when_session_id_not_a_string():
    """A coordinator whose `session_id` is not a plain string (e.g. an
    unconfigured MagicMock attribute, standing in for a coordinator
    implementation that never sets one) yields None -- never a stringified
    mock repr sent to the API as a routing key.
    """
    coordinator = MagicMock()
    del coordinator.session_id  # accessing it now raises AttributeError
    coordinator.get_capability = MagicMock(return_value=None)
    coordinator.hooks = MagicMock()
    coordinator.hooks.emit = AsyncMock()
    provider = _make_provider(coordinator=coordinator)
    assert provider._default_prompt_cache_key() is None


def test_helper_returns_none_when_session_id_empty_string():
    """An empty-string session_id (defensive edge case) also yields None."""
    provider = _make_provider(coordinator=_make_coordinator(""))
    assert provider._default_prompt_cache_key() is None


# ---------------------------------------------------------------------------
# End-to-end: default is present, stable, and distinct per session
# ---------------------------------------------------------------------------


def test_default_present_when_coordinator_has_session_id():
    """No config, no kwarg, coordinator with session_id -> field IS sent,
    carrying the session identity."""
    provider = _make_provider(coordinator=_make_coordinator("sess-abc-123"))
    _mock_create(provider)
    asyncio.run(provider.complete(_simple_request()))
    assert _captured_params(provider)["prompt_cache_key"] == "sess-abc-123"


def test_default_absent_without_coordinator():
    """No config, no coordinator -> field absent (pre-existing behavior,
    unhinted routing -- regression guard alongside
    test_cache_params.py::test_prompt_cache_key_omitted_when_none).
    """
    provider = _make_provider(coordinator=None)
    _mock_create(provider)
    asyncio.run(provider.complete(_simple_request()))
    assert "prompt_cache_key" not in _captured_params(provider)


def test_default_stable_across_consecutive_request_builds():
    """The same provider (one session) must send the identical
    prompt_cache_key on every request it builds -- the whole point of a
    cache-routing hint is consistency across the session's requests."""
    provider = _make_provider(coordinator=_make_coordinator("sess-stable"))
    _mock_create(provider)

    asyncio.run(provider.complete(_simple_request()))
    first = _captured_params(provider)["prompt_cache_key"]

    _mock_create(provider)  # fresh mock for the second build, not a reused one
    asyncio.run(provider.complete(_simple_request()))
    second = _captured_params(provider)["prompt_cache_key"]

    assert first == second == "sess-stable"


def test_default_distinct_across_different_sessions():
    """Two providers standing in for two different Amplifier sessions
    (distinct coordinator.session_id) must NOT collapse to the same
    routing key."""
    provider_a = _make_provider(coordinator=_make_coordinator("sess-AAA"))
    provider_b = _make_provider(coordinator=_make_coordinator("sess-BBB"))
    _mock_create(provider_a)
    _mock_create(provider_b)

    asyncio.run(provider_a.complete(_simple_request()))
    asyncio.run(provider_b.complete(_simple_request()))

    key_a = _captured_params(provider_a)["prompt_cache_key"]
    key_b = _captured_params(provider_b)["prompt_cache_key"]

    assert key_a == "sess-AAA"
    assert key_b == "sess-BBB"
    assert key_a != key_b


# ---------------------------------------------------------------------------
# Precedence: explicit config / kwarg ALWAYS wins over the session default
# ---------------------------------------------------------------------------


def test_config_value_wins_over_session_default():
    """An explicit config prompt_cache_key must NOT be overridden by the
    session-derived default, even when a session identity is reachable."""
    provider = _make_provider(
        coordinator=_make_coordinator("sess-should-not-be-used"),
        prompt_cache_key="explicit-config-key",
    )
    _mock_create(provider)
    asyncio.run(provider.complete(_simple_request()))
    assert _captured_params(provider)["prompt_cache_key"] == "explicit-config-key"


def test_kwarg_value_wins_over_session_default():
    """A per-call kwarg must NOT be overridden by the session default,
    even with no config value and a reachable session identity."""
    provider = _make_provider(coordinator=_make_coordinator("sess-should-not-be-used"))
    _mock_create(provider)
    asyncio.run(
        provider.complete(_simple_request(), prompt_cache_key="explicit-kwarg-key")
    )
    assert _captured_params(provider)["prompt_cache_key"] == "explicit-kwarg-key"


def test_explicit_empty_string_config_opts_out_even_with_session_identity():
    """The documented opt-out: config prompt_cache_key="" must send nothing
    at all, even though a session identity IS reachable. This is the case
    that requires distinguishing "configured empty" from "never
    configured" -- both are falsy, but only the latter should trigger the
    new default.
    """
    provider = _make_provider(
        coordinator=_make_coordinator("sess-should-not-be-used"),
        prompt_cache_key="",
    )
    _mock_create(provider)
    asyncio.run(provider.complete(_simple_request()))
    assert "prompt_cache_key" not in _captured_params(provider)


def test_explicit_none_config_opts_out_even_with_session_identity():
    """config prompt_cache_key=None is the same explicit opt-out as ""."""
    provider = _make_provider(
        coordinator=_make_coordinator("sess-should-not-be-used"),
        prompt_cache_key=None,
    )
    _mock_create(provider)
    asyncio.run(provider.complete(_simple_request()))
    assert "prompt_cache_key" not in _captured_params(provider)


def test_kwarg_none_opts_out_even_with_session_identity():
    """A per-call kwarg explicitly set to None also opts out for that call,
    even with a reachable session identity and no config value."""
    provider = _make_provider(coordinator=_make_coordinator("sess-should-not-be-used"))
    _mock_create(provider)
    asyncio.run(provider.complete(_simple_request(), prompt_cache_key=None))
    assert "prompt_cache_key" not in _captured_params(provider)


# ---------------------------------------------------------------------------
# Continuation calls inherit the session-derived default too
# ---------------------------------------------------------------------------


def test_continuation_inherits_session_default():
    """A continuation call (incomplete -> completed) must carry the same
    session-derived prompt_cache_key as the initial call -- otherwise the
    continuation would land on a different cache shard than the request
    it is completing.
    """
    provider = _make_provider(
        coordinator=_make_coordinator("sess-continuation"),
        default_model="gpt-5.4",
    )

    incomplete_resp = SimpleNamespace(
        status="incomplete",
        id="resp_incomplete",
        output=[],
        incomplete_details=None,
    )
    completed_resp = DummyResponse()

    provider.client.responses.create = AsyncMock(
        side_effect=[incomplete_resp, completed_resp]
    )
    asyncio.run(provider.complete(_simple_request()))

    calls = provider.client.responses.create.call_args_list
    assert len(calls) == 2
    for call in calls:
        assert call.kwargs.get("prompt_cache_key") == "sess-continuation"
