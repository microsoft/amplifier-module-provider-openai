"""Phase-2 tests: gpt-5.6 reasoning.mode ("pro") and prompt_cache_options.

Shapes verified live against gpt-5.6-sol on 2026-07-14:
- reasoning.mode in {"standard", "pro"} ("pro" = extended internal reasoning).
- prompt_cache_options {"mode": "implicit"|"explicit", "ttl": "30m"}, which
  COEXISTS with prompt_cache_retention (both are echoed together -- it is NOT a
  replacement/deprecation of prompt_cache_retention).

Also covers the D2 guardrail (spec section 2.4): a live probe on 2026-08-28
confirmed that `prompt_cache_options.mode == "explicit"` with zero
`prompt_cache_breakpoint` markers in `input` disables prompt caching entirely
(cache_write_tokens == 0 AND cached_tokens == 0 on every request). Since this
provider never attaches breakpoints, an operator setting explicit mode today
silently converts a ~95% cache-read workload into 100% full-price input. The
guardrail downgrades to implicit and warns once per provider instance.
"""

import asyncio
import logging
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import pytest
from amplifier_core import llm_errors as kernel_errors
from amplifier_core.message_models import ChatRequest, Message
from amplifier_module_provider_openai import (
    OpenAIProvider,
    _input_has_cache_breakpoint,
    _validate_prompt_cache_options,
    _validate_reasoning_mode,
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
# reasoning.mode validator
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ok", [None, "high", {}, {"mode": "standard"}, {"mode": "pro"}]
)
def test_validate_reasoning_mode_accepts(ok):
    _validate_reasoning_mode(ok)  # must not raise


@pytest.mark.parametrize("bad", [{"mode": "turbo"}, {"mode": "ultra"}, {"mode": ""}])
def test_validate_reasoning_mode_rejects(bad):
    with pytest.raises(kernel_errors.InvalidRequestError):
        _validate_reasoning_mode(bad)


# ---------------------------------------------------------------------------
# prompt_cache_options validator
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ok",
    [
        {"mode": "implicit"},
        {"mode": "explicit"},
        {"ttl": "30m"},
        {"mode": "explicit", "ttl": "30m"},
        {},
    ],
)
def test_validate_prompt_cache_options_accepts(ok):
    _validate_prompt_cache_options(ok)  # must not raise


@pytest.mark.parametrize("bad", [{"mode": "auto"}, {"mode": "zzz"}, "explicit", 123])
def test_validate_prompt_cache_options_rejects(bad):
    with pytest.raises(kernel_errors.InvalidRequestError):
        _validate_prompt_cache_options(bad)


# ---------------------------------------------------------------------------
# reasoning.mode passthrough into the API call
# ---------------------------------------------------------------------------


def test_reasoning_mode_pro_forwarded():
    provider = _make_provider(default_model="gpt-5.6-sol")
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())
    asyncio.run(
        provider.complete(
            _simple_request(), reasoning={"effort": "high", "mode": "pro"}
        )
    )
    reasoning = _captured_params(provider)["reasoning"]
    assert reasoning["mode"] == "pro"
    assert reasoning["effort"] == "high"


def test_reasoning_mode_absent_when_not_set():
    provider = _make_provider(default_model="gpt-5.6-sol")
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())
    asyncio.run(provider.complete(_simple_request(), reasoning={"effort": "medium"}))
    assert "mode" not in _captured_params(provider)["reasoning"]


# ---------------------------------------------------------------------------
# prompt_cache_options passthrough + coexistence with retention
# ---------------------------------------------------------------------------


def test_prompt_cache_options_forwarded_from_config():
    """explicit mode passes through verbatim when the D2 precondition (>=1
    prompt_cache_breakpoint marker in input) is satisfied.

    No breakpoint-attachment mechanism ships in this provider today (see the
    D2 guardrail tests below), so the precondition is simulated via patch --
    this test's job is to verify passthrough + retention coexistence, not the
    guardrail itself.
    """
    provider = _make_provider(
        default_model="gpt-5.6-sol", prompt_cache_options={"mode": "explicit"}
    )
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())
    with patch(
        "amplifier_module_provider_openai._input_has_cache_breakpoint",
        return_value=True,
    ):
        asyncio.run(provider.complete(_simple_request()))
    params = _captured_params(provider)
    assert params["prompt_cache_options"] == {"mode": "explicit"}
    # Coexistence: the default "24h" retention is still sent alongside it.
    assert params["prompt_cache_retention"] == "24h"


def test_prompt_cache_options_omitted_when_none():
    provider = _make_provider(default_model="gpt-5.6-sol")
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())
    asyncio.run(provider.complete(_simple_request()))
    assert "prompt_cache_options" not in _captured_params(provider)


def test_prompt_cache_options_kwarg_overrides_config():
    """Per-call kwarg overrides config; D2 precondition simulated (see above)."""
    provider = _make_provider(
        default_model="gpt-5.6-sol", prompt_cache_options={"mode": "implicit"}
    )
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())
    with patch(
        "amplifier_module_provider_openai._input_has_cache_breakpoint",
        return_value=True,
    ):
        asyncio.run(
            provider.complete(
                _simple_request(), prompt_cache_options={"mode": "explicit"}
            )
        )
    assert _captured_params(provider)["prompt_cache_options"] == {"mode": "explicit"}


def test_prompt_cache_options_forwarded_on_continuation():
    """prompt_cache_options must survive an incomplete->continuation sequence.

    Mirrors test_cache_params.test_continuation_inherits_cache_params: if the
    continuation-forwarding block ever drops the field, this catches it (the
    continuation call would otherwise land on a different cache policy). D2
    precondition simulated (see test_prompt_cache_options_forwarded_from_config).
    """
    provider = _make_provider(
        default_model="gpt-5.6-sol", prompt_cache_options={"mode": "explicit"}
    )
    incomplete_resp = SimpleNamespace(
        status="incomplete", id="resp_incomplete", output=[], incomplete_details=None
    )
    provider.client.responses.create = AsyncMock(
        side_effect=[incomplete_resp, DummyResponse()]
    )
    with patch(
        "amplifier_module_provider_openai._input_has_cache_breakpoint",
        return_value=True,
    ):
        asyncio.run(provider.complete(_simple_request()))

    calls = provider.client.responses.create.call_args_list
    assert len(calls) == 2
    for call in calls:
        assert call.kwargs.get("prompt_cache_options") == {"mode": "explicit"}


# ---------------------------------------------------------------------------
# D2 guardrail: explicit mode with zero prompt_cache_breakpoint markers
# (spec section 2.4 -- https://github.com/microsoft/amplifier-module-provider-openai)
# ---------------------------------------------------------------------------

_GUARD_LOGGER = "amplifier_module_provider_openai"


def test_input_has_cache_breakpoint_false_for_plain_input():
    """Pure-function unit test: ordinary converted input has no markers."""
    plain_input = [
        {"role": "user", "content": [{"type": "input_text", "text": "hi"}]},
        {"type": "function_call_output", "call_id": "c1", "output": "ok"},
    ]
    assert _input_has_cache_breakpoint(plain_input) is False


def test_input_has_cache_breakpoint_detects_item_level_marker():
    """Pure-function unit test: a top-level item can carry the marker."""
    marked_input = [
        {
            "type": "function_call_output",
            "call_id": "c1",
            "output": "ok",
            "prompt_cache_breakpoint": {"mode": "explicit"},
        }
    ]
    assert _input_has_cache_breakpoint(marked_input) is True


def test_input_has_cache_breakpoint_detects_content_block_marker():
    """Pure-function unit test: a nested content block can carry the marker."""
    marked_input = [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "hi",
                    "prompt_cache_breakpoint": {"mode": "explicit"},
                }
            ],
        }
    ]
    assert _input_has_cache_breakpoint(marked_input) is True


def test_explicit_mode_no_breakpoints_downgraded_and_warns(caplog):
    """(a) explicit mode + no breakpoints -> guard fires: mode stripped, warned.

    This is the fail-before/pass-after regression case: before the D2
    guardrail existed, `prompt_cache_options` was forwarded verbatim and this
    assertion would fail (mode would still be "explicit").
    """
    caplog.set_level(logging.WARNING, logger=_GUARD_LOGGER)
    provider = _make_provider(
        default_model="gpt-5.6-sol", prompt_cache_options={"mode": "explicit"}
    )
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())
    asyncio.run(provider.complete(_simple_request()))

    params = _captured_params(provider)
    # Nothing else was in the dict once "mode" is stripped, so the field is
    # omitted entirely -- matching the "don't send the field" convention used
    # everywhere else in this module.
    assert "prompt_cache_options" not in params
    assert any(
        "disables prompt caching entirely" in r.message
        for r in caplog.records
        if r.levelno == logging.WARNING
    )


def test_explicit_mode_ttl_preserved_after_downgrade(caplog):
    """Stripping `mode` must not throw away sibling keys like `ttl`."""
    caplog.set_level(logging.WARNING, logger=_GUARD_LOGGER)
    provider = _make_provider(
        default_model="gpt-5.6-sol",
        prompt_cache_options={"mode": "explicit", "ttl": "30m"},
    )
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())
    asyncio.run(provider.complete(_simple_request()))

    assert _captured_params(provider)["prompt_cache_options"] == {"ttl": "30m"}


def test_implicit_mode_no_warning_and_passthrough(caplog):
    """(b) implicit/default config -> no warning, options passed through unchanged."""
    caplog.set_level(logging.WARNING, logger=_GUARD_LOGGER)
    provider = _make_provider(
        default_model="gpt-5.6-sol", prompt_cache_options={"mode": "implicit"}
    )
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())
    asyncio.run(provider.complete(_simple_request()))

    assert _captured_params(provider)["prompt_cache_options"] == {"mode": "implicit"}
    assert not any(
        "disables prompt caching entirely" in r.message for r in caplog.records
    )


def test_no_prompt_cache_options_no_warning(caplog):
    """(b) no prompt_cache_options configured at all -> no warning, no field sent."""
    caplog.set_level(logging.WARNING, logger=_GUARD_LOGGER)
    provider = _make_provider(default_model="gpt-5.6-sol")
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())
    asyncio.run(provider.complete(_simple_request()))

    assert "prompt_cache_options" not in _captured_params(provider)
    assert not any(
        "disables prompt caching entirely" in r.message for r in caplog.records
    )


def test_explicit_mode_with_breakpoints_present_mode_kept(caplog):
    """When >=1 breakpoint marker is present, explicit mode is respected as-is.

    No breakpoint-attachment mechanism ships in this provider today, so this
    is exercised by forcing the detector's result -- proving the guard is
    conditioned on breakpoint presence, not unconditionally stripping
    "explicit".
    """
    caplog.set_level(logging.WARNING, logger=_GUARD_LOGGER)
    provider = _make_provider(
        default_model="gpt-5.6-sol", prompt_cache_options={"mode": "explicit"}
    )
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())
    with patch(
        "amplifier_module_provider_openai._input_has_cache_breakpoint",
        return_value=True,
    ):
        asyncio.run(provider.complete(_simple_request()))

    assert _captured_params(provider)["prompt_cache_options"] == {"mode": "explicit"}
    assert not any(
        "disables prompt caching entirely" in r.message for r in caplog.records
    )


def test_explicit_mode_warning_emitted_once_per_session_not_per_request(caplog):
    """The warning must fire at most once per provider instance, across many calls."""
    caplog.set_level(logging.WARNING, logger=_GUARD_LOGGER)
    provider = _make_provider(
        default_model="gpt-5.6-sol", prompt_cache_options={"mode": "explicit"}
    )
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())

    asyncio.run(provider.complete(_simple_request()))
    asyncio.run(provider.complete(_simple_request()))
    asyncio.run(provider.complete(_simple_request()))

    warnings = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING
        and "disables prompt caching entirely" in r.message
    ]
    assert len(warnings) == 1
    # The guard keeps stripping "mode" on every request, not just the first --
    # one-shot logging must never regress into one-shot enforcement.
    assert (
        "prompt_cache_options" not in provider.client.responses.create.call_args.kwargs
    )


def test_explicit_mode_kwarg_override_no_breakpoints_downgraded(caplog):
    """A per-call kwarg setting explicit mode is guarded the same as config."""
    caplog.set_level(logging.WARNING, logger=_GUARD_LOGGER)
    provider = _make_provider(
        default_model="gpt-5.6-sol", prompt_cache_options={"mode": "implicit"}
    )
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())
    asyncio.run(
        provider.complete(_simple_request(), prompt_cache_options={"mode": "explicit"})
    )

    assert "prompt_cache_options" not in _captured_params(provider)
    assert any(
        "disables prompt caching entirely" in r.message
        for r in caplog.records
        if r.levelno == logging.WARNING
    )
