"""Tests for boolean-ish provider-config parsing (fix/config-bool-parsing).

THE BUG: several boolean-ish config keys either passed the raw config value
straight through with no coercion, or coerced it with `bool(raw)`. The
app-cli wizard writes `field_type="boolean"` values as the STRINGS "true" /
"false" (not Python bools), and hand-edited YAML commonly quotes booleans
too. `bool("false")` is True -- any non-empty string is truthy in Python --
so a config author who wrote `enable_long_context: "false"` got the
*opposite* of what they asked for: a silently-enabled ~2x-cost setting.

This file exercises:
  (a) the truthiness-bug failure shape for `enable_long_context`
  (b) per-key string/bool/absent coercion for every affected key
  (c) mount-time fail-loud on garbage values
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from amplifier_core.message_models import ChatRequest, Message

from amplifier_module_provider_openai import OpenAIProvider

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_provider(**config_overrides) -> OpenAIProvider:
    config = {"max_retries": 0, "use_streaming": False, **config_overrides}
    return OpenAIProvider(api_key="test-key", config=config)


class DummyResponse:
    """Minimal response stub -- matches the shape _convert_to_chat_response() needs."""

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


def _captured_params(provider: OpenAIProvider):
    return provider.client.responses.create.call_args.kwargs


# ---------------------------------------------------------------------------
# (a) The truthiness-bug failure shape: enable_long_context="false" (STRING)
#     must NOT enable long context. Retargeted from the historical
#     enable_response_chaining example (that config key/code path is gone --
#     the provider is stateless-only) onto the current live boolean key that
#     shares the exact same truthiness hazard.
# ---------------------------------------------------------------------------


def test_string_false_disables_long_context():
    """config enable_long_context="false" (string) must resolve to False.

    Pre-fix pattern: `bool("false")` is True, so this silently enabled a
    ~2x-cost setting for exactly the operators who asked it to be off.
    """
    provider = _make_provider(default_model="gpt-5.6-sol", enable_long_context="false")
    assert provider.enable_long_context is False, (
        "enable_long_context='false' (string) must resolve to False; got "
        f"{provider.enable_long_context!r}"
    )


def test_string_true_enables_long_context():
    """Symmetric case: enable_long_context="true" (string) resolves to True."""
    provider = _make_provider(default_model="gpt-5.6-sol", enable_long_context="true")
    assert provider.enable_long_context is True


# ---------------------------------------------------------------------------
# (c) garbage value -> mount-time error naming key + accepted values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "garbage", ["flase", "yes-ish", "1x", "0x", "disabled", "off", "TrueFalse"]
)
def test_long_context_garbage_value_fails_loud_at_mount(garbage):
    with pytest.raises(ValueError) as exc_info:
        _make_provider(default_model="gpt-5.6-sol", enable_long_context=garbage)
    message = str(exc_info.value)
    assert "enable_long_context" in message, (
        f"Error must name the config key; got: {message}"
    )
    assert repr(garbage) in message or garbage in message, (
        f"Error must name the received value {garbage!r}; got: {message}"
    )
    assert "true" in message.lower() and "false" in message.lower(), (
        f"Error must name the accepted values; got: {message}"
    )


# ---------------------------------------------------------------------------
# (b) Per-key coercion: string "false" -> False, string "true" -> True,
#     real bool passthrough, absent -> default.
#
# Audited config keys affected by the same anti-pattern (see PR body for the
# full audit table): enable_state, raw, filtered, enable_long_context,
# enable_reasoning_context, use_streaming, retry_jitter.
#
# enable_state / enable_reasoning_context remain covered here even though
# their config surface is slated for removal in a later change -- the
# attributes still exist and are still parsed via _parse_config_bool today.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key,default",
    [
        ("enable_state", False),
        ("raw", False),
        ("filtered", True),
        ("enable_long_context", False),
        ("enable_reasoning_context", False),
        ("use_streaming", True),
    ],
)
class TestPerKeyBoolCoercion:
    """Constructor-level coercion for every affected simple-boolean config key."""

    def test_string_false_is_false(self, key, default):
        provider = _make_provider(**{key: "false"})
        assert getattr(provider, key) is False, (
            f"{key}='false' (string) must resolve to False; got {getattr(provider, key)!r}"
        )

    def test_string_true_is_true(self, key, default):
        provider = _make_provider(**{key: "true"})
        assert getattr(provider, key) is True, (
            f"{key}='true' (string) must resolve to True; got {getattr(provider, key)!r}"
        )

    def test_string_case_and_whitespace_insensitive(self, key, default):
        provider_false = _make_provider(**{key: " FALSE "})
        provider_true = _make_provider(**{key: " TRUE "})
        assert getattr(provider_false, key) is False
        assert getattr(provider_true, key) is True

    def test_real_bool_passthrough(self, key, default):
        provider_true = _make_provider(**{key: True})
        provider_false = _make_provider(**{key: False})
        assert getattr(provider_true, key) is True
        assert getattr(provider_false, key) is False

    def test_absent_uses_default(self, key, default):
        # NOTE: cannot use the shared _make_provider() helper here for
        # `use_streaming` -- it always forces use_streaming=False in its
        # base config (the convention this test suite uses everywhere else
        # to avoid exercising the streaming code path), which would make
        # "absent" indistinguishable from "explicitly False". Construct
        # directly with a config that omits `key` entirely instead.
        config = {"max_retries": 0}
        if key != "use_streaming":
            config["use_streaming"] = False
        provider = OpenAIProvider(api_key="test-key", config=config)
        assert getattr(provider, key) is default, (
            f"{key} absent from config should default to {default!r}; "
            f"got {getattr(provider, key)!r}"
        )

    def test_garbage_value_fails_loud_at_mount(self, key, default):
        with pytest.raises(ValueError) as exc_info:
            _make_provider(**{key: "not-a-bool"})
        message = str(exc_info.value)
        assert key in message, f"Error must name the config key {key!r}; got: {message}"
        assert "not-a-bool" in message, (
            f"Error must name the received value; got: {message}"
        )
        assert "true" in message.lower() and "false" in message.lower(), (
            f"Error must name the accepted values; got: {message}"
        )


def test_retry_jitter_string_false_is_false():
    """retry_jitter lives on RetryConfig, not a plain self.<key> attribute.

    RetryConfig.jitter is "numeric compat" (0.2 enabled / 0.0 disabled per
    the Rust binding's own docstring), not a literal bool -- assert
    truthiness, not identity.
    """
    provider = _make_provider(retry_jitter="false")
    assert not provider._retry_config.jitter, (
        f"retry_jitter='false' (string) must resolve to disabled (falsy); "
        f"got {provider._retry_config.jitter!r}"
    )


def test_retry_jitter_string_true_is_true():
    provider = _make_provider(retry_jitter="true")
    assert provider._retry_config.jitter, (
        f"retry_jitter='true' (string) must resolve to enabled (truthy); "
        f"got {provider._retry_config.jitter!r}"
    )


def test_retry_jitter_absent_defaults_true():
    provider = _make_provider()
    assert provider._retry_config.jitter, (
        f"retry_jitter absent should default to enabled (truthy); "
        f"got {provider._retry_config.jitter!r}"
    )


def test_retry_jitter_garbage_fails_loud_at_mount():
    with pytest.raises(ValueError, match="retry_jitter"):
        _make_provider(retry_jitter="nope")


# ---------------------------------------------------------------------------
# Direct unit coverage of the shared helper.
# ---------------------------------------------------------------------------


def test_parse_config_bool_unit():
    from amplifier_module_provider_openai import _parse_config_bool

    assert _parse_config_bool("k", None, True) is True
    assert _parse_config_bool("k", None, False) is False
    assert _parse_config_bool("k", "", True) is True
    assert _parse_config_bool("k", True, False) is True
    assert _parse_config_bool("k", False, True) is False
    assert _parse_config_bool("k", "true", False) is True
    assert _parse_config_bool("k", "false", True) is False
    assert _parse_config_bool("k", "1", False) is True
    assert _parse_config_bool("k", "0", True) is False
    assert _parse_config_bool("k", "yes", False) is True
    assert _parse_config_bool("k", "no", True) is False
    assert _parse_config_bool("k", "  TrUe  ", False) is True
    with pytest.raises(ValueError, match="k"):
        _parse_config_bool("k", "garbage", False)


# ---------------------------------------------------------------------------
# Behavioral confirmation: the provider is always stateless (store=False)
# regardless of legacy enable_state, since store is no longer read from
# config/kwargs at all -- only background mode can force it True.
# ---------------------------------------------------------------------------


def test_enable_state_no_longer_affects_store():
    provider = _make_provider(default_model="gpt-5-mini", enable_state="true")
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())

    asyncio.run(
        provider.complete(ChatRequest(messages=[Message(role="user", content="Hi")]))
    )

    params = _captured_params(provider)
    assert params["store"] is False, (
        "store must be False on every non-background request regardless of "
        f"legacy enable_state; got store={params.get('store')!r}"
    )


def test_raw_string_false_does_not_enable_raw_payload_events():
    """self.raw gates raw-payload event emission; 'false' string must disable it."""
    provider = _make_provider(raw="false")
    assert provider.raw is False
