"""Phase-2 tests: gpt-5.6 reasoning.mode ("pro") and prompt_cache_options.

Shapes verified live against gpt-5.6-sol on 2026-07-14:
- reasoning.mode in {"standard", "pro"} ("pro" = extended internal reasoning).
- prompt_cache_options {"mode": "implicit"|"explicit", "ttl": "30m"}, which
  COEXISTS with prompt_cache_retention (both are echoed together -- it is NOT a
  replacement/deprecation of prompt_cache_retention).

Also covers the explicit-mode guardrail: explicit caching with no
`prompt_cache_breakpoint` markers disables prompt caching. Luna and Terra
automatically mark eligible function results; unsupported default models retain
the existing mount-time downgrade and warning.
"""

import asyncio
import copy
import logging
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

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


class DummyStream:
    def __init__(self, response: DummyResponse):
        self.response = response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    def __aiter__(self):
        async def events():
            yield SimpleNamespace(type="response.completed", response=self.response)

        return events()

    async def get_final_response(self):
        return self.response


class CapturingStreamFactory:
    def __init__(self, response: DummyResponse):
        self.response = response
        self.params: dict[str, Any] | None = None

    def __call__(self, **params):
        self.params = params
        return DummyStream(self.response)


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
# Luna/Terra function-output cache breakpoints
# ---------------------------------------------------------------------------


def test_tool_output_breakpoint_helper_preserves_input_and_marks_last_text_block():
    """Only supported function outputs are copied and marked."""
    from amplifier_module_provider_openai import _add_tool_output_cache_breakpoints

    original_items = [
        {"role": "developer", "content": [{"type": "input_text", "text": "global"}]},
        {"type": "reasoning", "encrypted_content": "opaque"},
        {
            "type": "function_call_output",
            "call_id": "string",
            "output": "Unicode: café\n",
        },
        {"type": "function_call_output", "call_id": "empty", "output": ""},
        {
            "type": "function_call_output",
            "call_id": "mixed",
            "output": [
                {"type": "input_text", "text": "earlier"},
                {"type": "output_text", "text": "leave unchanged"},
                {"type": "input_text", "text": "last"},
            ],
        },
        {
            "type": "function_call_output",
            "call_id": "caller-marker",
            "output": [
                {
                    "type": "input_text",
                    "text": "caller owns this",
                    "prompt_cache_breakpoint": None,
                }
            ],
        },
        {
            "type": "function_call_output",
            "call_id": "nontext",
            "output": [{"type": "output_text", "text": "not an input anchor"}],
        },
        {"type": "message", "content": [{"type": "output_text", "text": "answer"}]},
    ]
    original = copy.deepcopy(original_items)

    marked = _add_tool_output_cache_breakpoints(original_items)

    assert original_items == original
    assert marked is not original_items
    assert marked[0] is original_items[0]
    assert marked[1] is original_items[1]
    assert marked[7] is original_items[7]
    assert marked[2]["output"] == [
        {
            "type": "input_text",
            "text": "Unicode: café\n",
            "prompt_cache_breakpoint": {"mode": "explicit"},
        }
    ]
    assert marked[3]["output"][0]["text"] == ""
    assert marked[3]["output"][0]["prompt_cache_breakpoint"] == {"mode": "explicit"}
    assert "prompt_cache_breakpoint" not in marked[4]["output"][0]
    assert marked[4]["output"][1] == {"type": "output_text", "text": "leave unchanged"}
    assert marked[4]["output"][2]["prompt_cache_breakpoint"] == {"mode": "explicit"}
    assert marked[5] is original_items[5]
    assert marked[5]["output"][0]["prompt_cache_breakpoint"] is None
    assert marked[6] is original_items[6]
    unchanged = [original_items[5], original_items[6]]
    assert _add_tool_output_cache_breakpoints(unchanged) is unchanged


def test_tool_output_breakpoint_helper_marks_every_historical_function_result():
    from amplifier_module_provider_openai import _add_tool_output_cache_breakpoints

    items = [
        {"type": "function_call_output", "call_id": str(index), "output": f"result {index}"}
        for index in range(7)
    ]

    marked = _add_tool_output_cache_breakpoints(items)

    assert [
        item["output"][0]["prompt_cache_breakpoint"] for item in marked
    ] == [{"mode": "explicit"}] * 7


def test_tool_output_breakpoint_model_predicate_is_luna_terra_only():
    from amplifier_module_provider_openai import _supports_tool_output_cache_breakpoints

    assert _supports_tool_output_cache_breakpoints("gpt-5.6-luna")
    assert _supports_tool_output_cache_breakpoints("gpt-5.6-luna-2026-09-01")
    assert _supports_tool_output_cache_breakpoints("gpt-5.6-terra")
    assert _supports_tool_output_cache_breakpoints("gpt-5.6-terra-preview")
    assert not _supports_tool_output_cache_breakpoints("gpt-5.6-sol")
    assert not _supports_tool_output_cache_breakpoints("gpt-5.5")
    assert not _supports_tool_output_cache_breakpoints("unrecognized-model")


@pytest.mark.parametrize(
    ("default_model", "request_model", "expected_marked"),
    [
        ("gpt-5.6-sol", "gpt-5.6-terra", True),
        ("gpt-5.6-luna", "gpt-5.6-sol", False),
        ("gpt-5.6-luna", "gpt-5.5", False),
        ("gpt-5.6-luna", "unrecognized-model", False),
    ],
)
def test_tool_output_breakpoints_use_effective_per_call_model(
    default_model, request_model, expected_marked
):
    source_input = [{"type": "function_call_output", "call_id": "call", "output": "result"}]
    provider = _make_provider(
        default_model=default_model, extra_request_params={"input": source_input}
    )
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())

    asyncio.run(provider.complete(_simple_request(), model=request_model))

    captured_input = _captured_params(provider)["input"]
    output = captured_input[0]["output"]
    assert (isinstance(output, list)) is expected_marked
    if not expected_marked:
        assert captured_input == source_input
    assert source_input[0]["output"] == "result"


def test_tool_output_breakpoints_honor_extra_request_param_model_and_input():
    supplied_input = [{"type": "function_call_output", "call_id": "call", "output": "result"}]
    provider = _make_provider(
        default_model="gpt-5.6-sol",
        extra_request_params={"model": "gpt-5.6-terra", "input": supplied_input},
    )
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())

    asyncio.run(provider.complete(_simple_request(), model="gpt-5.6-luna"))

    params = _captured_params(provider)
    assert params["model"] == "gpt-5.6-terra"
    assert params["input"][0]["output"][0]["prompt_cache_breakpoint"] == {
        "mode": "explicit"
    }
    assert supplied_input[0]["output"] == "result"


def test_tool_output_breakpoints_reach_streaming_sdk_params():
    source_input = [{"type": "function_call_output", "call_id": "call", "output": "result"}]
    provider = OpenAIProvider(
        api_key="[REDACTED:SECRET]",
        config={
            "max_retries": 0,
            "use_streaming": True,
            "default_model": "gpt-5.6-luna",
            "extra_request_params": {"input": source_input},
        },
    )
    stream = CapturingStreamFactory(DummyResponse())
    provider._client = SimpleNamespace(responses=SimpleNamespace(stream=stream))

    asyncio.run(provider.complete(_simple_request()))

    assert stream.params is not None
    assert stream.params["input"][0]["output"][0]["prompt_cache_breakpoint"] == {
        "mode": "explicit"
    }


def test_tool_output_breakpoints_are_idempotent_on_continuation():
    source_input = [{"type": "function_call_output", "call_id": "call", "output": "result"}]
    provider = _make_provider(
        default_model="gpt-5.6-luna", extra_request_params={"input": source_input}
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
        output = call.kwargs["input"][0]["output"]
        assert output == [
            {
                "type": "input_text",
                "text": "result",
                "prompt_cache_breakpoint": {"mode": "explicit"},
            }
        ]


# ---------------------------------------------------------------------------
# D2 guardrail: explicit mode with zero prompt_cache_breakpoint markers
# (spec section 2.4 -- https://github.com/microsoft/amplifier-module-provider-openai)
# ---------------------------------------------------------------------------

_GUARD_LOGGER = "amplifier_module_provider_openai"


@pytest.mark.parametrize("effective_model", ["gpt-5.6-terra", "gpt-5.6-sol"])
def test_session_explicit_guard_remains_safe_without_tool_results(caplog, effective_model):
    caplog.set_level(logging.WARNING, logger=_GUARD_LOGGER)
    provider = _make_provider(
        default_model="gpt-5.6-terra",
        prompt_cache_options={"mode": "explicit", "ttl": "30m"},
    )

    assert provider.prompt_cache_options == {"ttl": "30m"}
    assert any(
        "disables prompt caching entirely" in record.message for record in caplog.records
    )
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())
    asyncio.run(provider.complete(_simple_request(), model=effective_model))
    assert _captured_params(provider)["prompt_cache_options"] == {"ttl": "30m"}


def test_non_string_structured_tool_text_is_unchanged():
    from amplifier_module_provider_openai import _add_tool_output_cache_breakpoints

    items = [{"type": "function_call_output", "call_id": "call",
              "output": [{"type": "input_text", "text": ["not a string"]}]}]
    assert _add_tool_output_cache_breakpoints(items) == items


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
