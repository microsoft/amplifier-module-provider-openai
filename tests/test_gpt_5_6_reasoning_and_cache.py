"""Phase-2 tests: gpt-5.6 reasoning.mode ("pro") and prompt_cache_options.

Shapes verified live against gpt-5.6-sol on 2026-07-14:
- reasoning.mode in {"standard", "pro"} ("pro" = extended internal reasoning).
- prompt_cache_options {"mode": "implicit"|"explicit", "ttl": "30m"}, which
  COEXISTS with prompt_cache_retention (both are echoed together -- it is NOT a
  replacement/deprecation of prompt_cache_retention).

Also covers the D2 guardrail (spec section 2.4): a live probe on 2026-08-28
confirmed that `prompt_cache_options.mode == "explicit"` with zero
`prompt_cache_breakpoint` markers in `input` disables prompt caching entirely
(cache_write_tokens == 0 AND cached_tokens == 0 on every request). Even with
automatic Luna/Terra tool-result boundaries, some requests have no eligible
results. Session-wide explicit-only mode can therefore disable caching. The
guardrail is now validated ONCE AT MOUNT (not scanned per-request): mode is
downgraded to implicit and a warning fires exactly once per provider
instance. A caller who bypasses this via per-call kwargs owns the
consequences (same stance as the other explicit-override escape hatches).
"""

import asyncio
import copy
import logging
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from amplifier_core import llm_errors as kernel_errors
from amplifier_core.message_models import ChatRequest, Message

from amplifier_module_provider_openai import (
    OpenAIProvider,
    _validate_prompt_cache_options,
    _validate_reasoning_mode,
)


def _make_provider(**config_overrides) -> OpenAIProvider:
    config = {"max_retries": 0, "use_streaming": False, **config_overrides}
    client = SimpleNamespace(
        base_url="https://api.openai.com/v1",
        responses=SimpleNamespace(
            input_tokens=SimpleNamespace(
                count=AsyncMock(return_value=SimpleNamespace(input_tokens=1))
            ),
            create=AsyncMock(),
            stream=MagicMock(),
        ),
        close=AsyncMock(),
    )
    return OpenAIProvider(api_key="test-key", client=client, config=config)


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


@pytest.mark.parametrize("model,marked", [
    ("gpt-5.6-luna", True), ("gpt-5.6-terra-preview", True),
    ("unrecognized-model", False), ("gpt-5.6-sol", False),
])
def test_effective_model_and_caller_input_cache_contract(model, marked):
    """Final normalization applies to overrides but never mutates their objects."""
    supplied = [{"type": "function_call_output", "call_id": "c", "output": "café\n"}]
    before = copy.deepcopy(supplied)
    provider = _make_provider(
        default_model="gpt-5.6-luna",
        extra_request_params={"model": model, "input": supplied},
    )
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())
    asyncio.run(provider.complete(_simple_request(), model="gpt-5.6-terra"))
    params = _captured_params(provider)
    assert params["model"] == model
    assert supplied == before
    output = params["input"][0]["output"]
    assert isinstance(output, list) is marked
    if marked:
        assert output == [{"type": "input_text", "text": "café\n",
                           "prompt_cache_breakpoint": {"mode": "explicit"}}]
    else:
        assert output == "café\n"


def test_cache_boundaries_copy_only_eligible_blocks_and_are_idempotent():
    from amplifier_module_provider_openai import _add_tool_output_cache_breakpoints

    items = [
        {"type": "reasoning", "encrypted_content": "opaque"},
        {"type": "function_call_output", "call_id": "a", "output": ""},
        {"type": "function_call_output", "call_id": "b", "output": [
            {"type": "input_text", "text": "earlier"},
            {"type": "input_image", "image_url": "synthetic"},
            {"type": "input_text", "text": "last"},
        ]},
        {"type": "function_call_output", "call_id": "c", "output": [
            {"type": "input_text", "text": "caller", "prompt_cache_breakpoint": None}
        ]},
        {"type": "function_call_output", "call_id": "d", "output": [
            {"type": "input_text", "text": ["not supported"]}
        ]},
    ]
    before = copy.deepcopy(items)
    result = _add_tool_output_cache_breakpoints(items)
    assert items == before
    assert result[0] is items[0]
    assert result[1]["output"][0]["prompt_cache_breakpoint"] == {"mode": "explicit"}
    assert "prompt_cache_breakpoint" not in result[2]["output"][0]
    assert result[2]["output"][1] is items[2]["output"][1]
    assert result[2]["output"][2]["prompt_cache_breakpoint"] == {"mode": "explicit"}
    assert result[3] is items[3]
    assert result[4] is items[4]
    assert _add_tool_output_cache_breakpoints(result) is result


def test_cache_boundaries_reach_every_continuation_with_defaults_unchanged():
    supplied = [{"type": "function_call_output", "call_id": str(i), "output": f"result-{i}"}
                for i in range(7)]
    provider = _make_provider(
        default_model="gpt-5.6-terra", extra_request_params={"input": supplied},
    )
    incomplete = SimpleNamespace(
        status="incomplete", id="incomplete", output=[], incomplete_details=None,
    )
    provider.client.responses.create = AsyncMock(side_effect=[incomplete, DummyResponse()])
    asyncio.run(provider.complete(_simple_request()))
    calls = provider.client.responses.create.call_args_list
    assert len(calls) == 2
    for call in calls:
        assert [item["output"][0]["prompt_cache_breakpoint"] for item in call.kwargs["input"]] == [
            {"mode": "explicit"}
        ] * 7
        assert call.kwargs.get("prompt_cache_options") is None
        assert call.kwargs["prompt_cache_retention"] == provider.prompt_cache_retention
    assert all(isinstance(item["output"], str) for item in supplied)


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
    """implicit mode (unaffected by the D2 mount-time gate) passes through
    verbatim, coexisting with prompt_cache_retention."""
    provider = _make_provider(
        default_model="gpt-5.6-sol", prompt_cache_options={"mode": "implicit"}
    )
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())
    asyncio.run(provider.complete(_simple_request()))
    params = _captured_params(provider)
    assert params["prompt_cache_options"] == {"mode": "implicit"}
    # Coexistence: the default "24h" retention is still sent alongside it.
    assert params["prompt_cache_retention"] == "24h"


def test_prompt_cache_options_omitted_when_none():
    provider = _make_provider(default_model="gpt-5.6-sol")
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())
    asyncio.run(provider.complete(_simple_request()))
    assert "prompt_cache_options" not in _captured_params(provider)


def test_prompt_cache_options_kwarg_overrides_config():
    """Per-call kwarg overrides config. Explicit mode via kwargs bypasses the
    mount-time-only D2 gate (documented residual gap)."""
    provider = _make_provider(
        default_model="gpt-5.6-sol", prompt_cache_options={"mode": "implicit"}
    )
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())
    asyncio.run(
        provider.complete(_simple_request(), prompt_cache_options={"mode": "explicit"})
    )
    assert _captured_params(provider)["prompt_cache_options"] == {"mode": "explicit"}


def test_prompt_cache_options_forwarded_on_continuation():
    """prompt_cache_options must survive an incomplete->continuation sequence.

    Mirrors test_cache_params.test_continuation_inherits_cache_params: if the
    continuation-forwarding block ever drops the field, this catches it (the
    continuation call would otherwise land on a different cache policy).
    Uses the kwarg-based explicit mode (bypasses the mount-time-only D2
    gate) so the forwarded value is not stripped by the guardrail.
    """
    provider = _make_provider(default_model="gpt-5.6-sol")
    incomplete_resp = SimpleNamespace(
        status="incomplete", id="resp_incomplete", output=[], incomplete_details=None
    )
    provider.client.responses.create = AsyncMock(
        side_effect=[incomplete_resp, DummyResponse()]
    )
    asyncio.run(
        provider.complete(_simple_request(), prompt_cache_options={"mode": "explicit"})
    )

    calls = provider.client.responses.create.call_args_list
    assert len(calls) == 2
    for call in calls:
        assert call.kwargs.get("prompt_cache_options") == {"mode": "explicit"}


# ---------------------------------------------------------------------------
# D2 guardrail: explicit mode with zero prompt_cache_breakpoint markers
# (spec section 2.4 -- https://github.com/microsoft/amplifier-module-provider-openai)
# ---------------------------------------------------------------------------

_GUARD_LOGGER = "amplifier_module_provider_openai"


def test_explicit_mode_no_breakpoints_downgraded_and_warns_at_mount(caplog):
    """(a) explicit mode + no breakpoints -> guard fires AT MOUNT: mode
    stripped, warned once at construction (not scanned per-request anymore).
    """
    caplog.set_level(logging.WARNING, logger=_GUARD_LOGGER)
    provider = _make_provider(
        default_model="gpt-5.6-sol", prompt_cache_options={"mode": "explicit"}
    )

    # Nothing else was in the dict once "mode" is stripped, so the resolved
    # attribute is None -- matching the "don't send the field" convention.
    assert provider.prompt_cache_options is None
    assert any(
        "disables prompt caching entirely" in r.message
        for r in caplog.records
        if r.levelno == logging.WARNING
    )

    provider.client.responses.create = AsyncMock(return_value=DummyResponse())
    asyncio.run(provider.complete(_simple_request()))
    assert "prompt_cache_options" not in _captured_params(provider)


def test_explicit_mode_ttl_preserved_after_downgrade_at_mount(caplog):
    """Stripping `mode` at mount must not throw away sibling keys like `ttl`."""
    caplog.set_level(logging.WARNING, logger=_GUARD_LOGGER)
    provider = _make_provider(
        default_model="gpt-5.6-sol",
        prompt_cache_options={"mode": "explicit", "ttl": "30m"},
    )
    assert provider.prompt_cache_options == {"ttl": "30m"}

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


def test_explicit_mode_warning_emitted_once_at_mount_not_per_request(caplog):
    """The warning fires exactly once, at construction -- never re-evaluated
    per request (there is no runtime scan of `input` anymore)."""
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
    assert (
        "prompt_cache_options" not in provider.client.responses.create.call_args.kwargs
    )


def test_explicit_mode_kwarg_override_bypasses_mount_validation(caplog):
    """A per-call kwarg setting explicit mode is NOT gated -- mount validation
    only sees config, not kwargs. Documented residual gap: the caller
    bypassing mount validation via kwargs owns the consequences (same stance
    as every other explicit-override escape hatch in this provider)."""
    caplog.set_level(logging.WARNING, logger=_GUARD_LOGGER)
    provider = _make_provider(
        default_model="gpt-5.6-sol", prompt_cache_options={"mode": "implicit"}
    )
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())
    caplog.clear()
    asyncio.run(
        provider.complete(_simple_request(), prompt_cache_options={"mode": "explicit"})
    )

    assert _captured_params(provider)["prompt_cache_options"] == {"mode": "explicit"}
    assert not any(
        "disables prompt caching entirely" in r.message
        for r in caplog.records
        if r.levelno == logging.WARNING
    )
