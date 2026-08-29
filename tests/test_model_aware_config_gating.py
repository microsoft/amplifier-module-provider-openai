"""Tests for model-aware ConfigField gating and the unknown-key mount sweep.

Covers three hardening changes, all triggered by the gpt-5.6-luna defect
chain (the wizard offered `prompt_cache_retention: in_memory` for every
model, including gpt-5.6, which the request path then silently drops on
every session):

1. `prompt_cache_retention`, `text_verbosity`, and `enable_reasoning_context`
   are now gated via `requires_model` + `show_when`, so the wizard only
   offers values a model actually supports.
2. The `_drop_unsupported_in_memory_retention` warning leads with omission
   as the recommended remedy, and names `prompt_cache_options.ttl` as the
   gpt-5.6-specific replacement mechanism.
3. A generic unknown-config-key sweep at construction time warns (once, with
   a nearest-match suggestion) about keys this provider does not recognize,
   while staying silent on every legitimate config.
"""

import asyncio
import difflib
import logging
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

from amplifier_core.message_models import ChatRequest, Message

from amplifier_module_provider_openai import OpenAIProvider

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_provider(**config_overrides) -> OpenAIProvider:
    config = {"max_retries": 0, "use_streaming": False, **config_overrides}
    return OpenAIProvider(api_key="test-key", config=config)


def _simple_request() -> ChatRequest:
    return ChatRequest(messages=[Message(role="user", content="Hello")])


class _DummyResponse:
    """Minimal response stub (mirrors test_cache_params.py's DummyResponse)."""

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


def _field(provider: OpenAIProvider, field_id: str):
    info = provider.get_info()
    field = next((f for f in info.config_fields if f.id == field_id), None)
    assert field is not None, f"ConfigField {field_id!r} missing from get_info()"
    return field


def _should_show_field(field: dict[str, Any], collected_config: dict[str, Any]) -> bool:
    """Local re-implementation of amplifier_app_cli's
    ``provider_config_utils._should_show_field`` -- the actual wizard
    consumer of ``ConfigField.show_when``. Copied (not imported) because
    this package does not depend on amplifier-app-cli; kept in lockstep
    with the predicate vocabulary verified against that module's source
    during this change (contains / not_contains / startswith /
    not_startswith / exact, all case-insensitive).
    """
    show_when = field.get("show_when")
    if not show_when:
        return True
    for key, expected_value in show_when.items():
        actual_value = str(collected_config.get(key, "")).lower()
        expected_str = str(expected_value).lower()
        if expected_str.startswith("not_contains:"):
            if expected_str[13:] in actual_value:
                return False
        elif expected_str.startswith("contains:"):
            if expected_str[9:] not in actual_value:
                return False
        elif expected_str.startswith("not_startswith:"):
            if actual_value.startswith(expected_str[15:]):
                return False
        elif expected_str.startswith("startswith:"):
            if not actual_value.startswith(expected_str[11:]):
                return False
        else:
            if actual_value != expected_str:
                return False
    return True


def _field_as_dict(field) -> dict[str, Any]:
    """ConfigField is a pydantic BaseModel; the wizard consumes it as a dict."""
    return field.model_dump()


# ---------------------------------------------------------------------------
# 1a. get_info() gating metadata
# ---------------------------------------------------------------------------


class TestConfigFieldGatingMetadata:
    """The three model-sensitive fields must carry requires_model=True and
    the show_when predicate that hides/shows them per model family."""

    def test_prompt_cache_retention_requires_model_and_hides_for_5_6(self):
        field = _field(_make_provider(), "prompt_cache_retention")
        assert field.requires_model is True
        assert field.show_when == {"default_model": "not_contains:gpt-5.6"}

    def test_text_verbosity_requires_model_and_shows_only_for_5_6(self):
        field = _field(_make_provider(), "text_verbosity")
        assert field.requires_model is True
        assert field.show_when == {"default_model": "contains:gpt-5.6"}

    def test_untouched_fields_keep_no_gating(self):
        """Do-NOT-touch scope check: reasoning_effort keeps its existing
        requires_model=True with NO show_when (all values valid somewhere,
        per the task's explicit exclusion), and prompt_cache_key/
        enable_response_chaining are untouched (no requires_model)."""
        provider = _make_provider()
        reasoning_effort = _field(provider, "reasoning_effort")
        assert reasoning_effort.requires_model is True
        assert reasoning_effort.show_when is None

        prompt_cache_key = _field(provider, "prompt_cache_key")
        assert prompt_cache_key.requires_model is False
        assert prompt_cache_key.show_when is None


# ---------------------------------------------------------------------------
# 1b. show_when consumer simulation -- actual show/hide behavior per model
# ---------------------------------------------------------------------------


class TestShowWhenConsumerSimulation:
    """Drive the three gated ConfigFields through the real wizard predicate
    logic (`_should_show_field`) for representative models, mirroring how
    amplifier_app_cli.provider_config_utils.configure_provider evaluates
    post-model-selection fields."""

    GPT_5_6_MODELS = ("gpt-5.6", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna")
    NON_5_6_MODELS = ("gpt-5.4", "gpt-5.5", "gpt-5.5-pro", "gpt-4o", "gpt-5-mini")

    def _gated_fields(self, provider: OpenAIProvider) -> dict[str, dict[str, Any]]:
        info = provider.get_info()
        return {
            f.id: _field_as_dict(f)
            for f in info.config_fields
            if f.id in ("prompt_cache_retention", "text_verbosity")
        }

    def test_prompt_cache_retention_hidden_for_every_5_6_tier(self):
        fields = self._gated_fields(_make_provider())
        for model in self.GPT_5_6_MODELS:
            collected = {"default_model": model}
            assert (
                _should_show_field(fields["prompt_cache_retention"], collected) is False
            ), f"prompt_cache_retention should be hidden for {model}"

    def test_prompt_cache_retention_shown_for_non_5_6_models(self):
        fields = self._gated_fields(_make_provider())
        for model in self.NON_5_6_MODELS:
            collected = {"default_model": model}
            assert (
                _should_show_field(fields["prompt_cache_retention"], collected) is True
            ), f"prompt_cache_retention should be shown for {model}"

    def test_text_verbosity_shown_only_for_5_6_tiers(self):
        fields = self._gated_fields(_make_provider())
        for model in self.GPT_5_6_MODELS:
            assert (
                _should_show_field(fields["text_verbosity"], {"default_model": model})
                is True
            ), f"text_verbosity should be shown for {model}"
        for model in self.NON_5_6_MODELS:
            assert (
                _should_show_field(fields["text_verbosity"], {"default_model": model})
                is False
            ), f"text_verbosity should be hidden for {model}"

    def test_gpt_5_6_alias_bare_form_also_gates_correctly(self):
        """README: the bare 'gpt-5.6' alias resolves to gpt-5.6-sol, but the
        wizard gates on the STRING the user selected (before resolution) --
        it must still match the 'gpt-5.6' substring."""
        fields = self._gated_fields(_make_provider())
        collected = {"default_model": "gpt-5.6"}
        assert _should_show_field(fields["prompt_cache_retention"], collected) is False
        assert _should_show_field(fields["text_verbosity"], collected) is True


# ---------------------------------------------------------------------------
# 2. Warning text accuracy -- in_memory retention drop
# ---------------------------------------------------------------------------


class TestInMemoryDropWarningWording:
    def test_leads_with_omission_for_gpt_5_5(self, caplog):
        """For gpt-5.5 (no prompt_cache_options), the remedy must lead with
        omission and keep '24h' only as the secondary legacy-compatible
        alternative -- not the primary suggested fix.

        The drop (and its warning) happens at request time, inside
        _build_params -- NOT at construction -- so this must actually
        issue a completion call, mirroring test_cache_params.py.
        """
        caplog.set_level(logging.WARNING, logger="amplifier_module_provider_openai")
        provider = _make_provider(
            prompt_cache_retention="in_memory", default_model="gpt-5.5"
        )
        provider.client.responses.create = AsyncMock(return_value=_DummyResponse())
        asyncio.run(provider.complete(_simple_request()))

        assert "prompt_cache_retention" not in _captured_params(provider)

        warnings = [
            r.message
            for r in caplog.records
            if r.levelno == logging.WARNING
            and "Dropping prompt_cache_retention" in r.message
        ]
        assert len(warnings) == 1
        message = warnings[0]

        omit_idx = message.find("Omit")
        pass_24h_idx = message.find("pass '24h'")
        assert omit_idx != -1, f"expected 'Omit' remedy in message: {message!r}"
        assert pass_24h_idx != -1, f"expected '24h' alternative in message: {message!r}"
        assert omit_idx < pass_24h_idx, (
            f"omission must be presented before the '24h' alternative: {message!r}"
        )
        # gpt-5.5 has no prompt_cache_options -- must not be mentioned for it.
        assert "prompt_cache_options" not in message

    def test_names_prompt_cache_options_for_gpt_5_6(self, caplog):
        """For gpt-5.6, the doc-correct remedy (prompt_cache_options.ttl)
        must be named explicitly, still leading with omission."""
        caplog.set_level(logging.WARNING, logger="amplifier_module_provider_openai")
        provider = _make_provider(
            prompt_cache_retention="in_memory", default_model="gpt-5.6-luna"
        )
        provider.client.responses.create = AsyncMock(return_value=_DummyResponse())
        asyncio.run(provider.complete(_simple_request()))

        assert "prompt_cache_retention" not in _captured_params(provider)

        warnings = [
            r.message
            for r in caplog.records
            if r.levelno == logging.WARNING
            and "Dropping prompt_cache_retention" in r.message
        ]
        assert len(warnings) == 1
        message = warnings[0]

        assert "prompt_cache_options.ttl" in message
        omit_idx = message.find("Omit")
        options_idx = message.find("prompt_cache_options.ttl")
        pass_24h_idx = message.find("pass '24h'")
        assert -1 not in (omit_idx, options_idx, pass_24h_idx)
        assert omit_idx < pass_24h_idx, (
            f"omission must still lead over the '24h' alternative: {message!r}"
        )

    def test_existing_gpt_5_5_regression_assertions_still_hold(self, caplog):
        """Guards the pre-existing test_cache_params.py assertions (message
        contains 'in_memory' and the model id) against wording drift."""
        caplog.set_level(logging.WARNING, logger="amplifier_module_provider_openai")
        provider = _make_provider(
            prompt_cache_retention="in_memory", default_model="gpt-5.5"
        )
        provider.client.responses.create = AsyncMock(return_value=_DummyResponse())
        asyncio.run(provider.complete(_simple_request()))

        assert "prompt_cache_retention" not in _captured_params(provider)
        assert any(
            "in_memory" in r.message and "gpt-5.5" in r.message
            for r in caplog.records
            if r.levelno == logging.WARNING
        )


# ---------------------------------------------------------------------------
# 3. Unknown-config-key sweep at construction ("mount time")
# ---------------------------------------------------------------------------

_UNKNOWN_KEY_MARKER = "Unrecognized config key"


class TestUnknownConfigKeySweep:
    def test_typo_key_warns_with_suggestion(self, caplog):
        """A typo'd key with an obvious near neighbor gets a suggestion."""
        caplog.set_level(logging.WARNING, logger="amplifier_module_provider_openai")
        _make_provider(promt_cache_retention="24h")

        matches = [
            r.message for r in caplog.records if _UNKNOWN_KEY_MARKER in r.message
        ]
        assert len(matches) == 1, f"expected exactly one sweep warning, got: {matches}"
        message = matches[0]
        assert "'promt_cache_retention'" in message
        assert "prompt_cache_retention" in message
        assert "did you mean" in message.lower()

    def test_unmatched_key_warns_without_fabricating_a_suggestion(self, caplog):
        """A key with no close match is still named, but gets no bogus
        'did you mean' guess."""
        caplog.set_level(logging.WARNING, logger="amplifier_module_provider_openai")
        _make_provider(totally_unrelated_gibberish_xyz=True)

        matches = [
            r.message for r in caplog.records if _UNKNOWN_KEY_MARKER in r.message
        ]
        assert len(matches) == 1
        message = matches[0]
        assert "'totally_unrelated_gibberish_xyz'" in message
        # difflib must not invent a suggestion for a key this far from every
        # known key -- sanity-check via the real difflib call used by the code.
        from amplifier_module_provider_openai import _KNOWN_CONFIG_KEYS

        assert (
            difflib.get_close_matches(
                "totally_unrelated_gibberish_xyz", _KNOWN_CONFIG_KEYS, n=1
            )
            == []
        )

    def test_multiple_unknown_keys_combined_into_one_warning(self, caplog):
        caplog.set_level(logging.WARNING, logger="amplifier_module_provider_openai")
        _make_provider(promt_cache_retention="24h", bogus_setting_key=1)

        matches = [
            r.message for r in caplog.records if _UNKNOWN_KEY_MARKER in r.message
        ]
        assert len(matches) == 1, "expected ONE combined warning, not one per key"
        message = matches[0]
        assert "'promt_cache_retention'" in message
        assert "'bogus_setting_key'" in message

    def test_effort_key_alone_does_not_trigger_generic_sweep(self, caplog):
        """The dedicated 'effort' guard fires; the generic sweep must NOT
        ALSO fire for the same recognized-inert key (no double warning)."""
        caplog.set_level(logging.WARNING, logger="amplifier_module_provider_openai")
        _make_provider(effort="high")

        assert any("not consumed" in r.message for r in caplog.records)
        assert not any(_UNKNOWN_KEY_MARKER in r.message for r in caplog.records)

    def test_full_legitimate_settings_shaped_config_is_silent(self, caplog):
        """A real, settings-shaped config carrying every documented key PLUS
        the infrastructure keys an app/kernel may place alongside them
        (default_model, priority, source, id, api_key) must produce NO
        unknown-key warnings whatsoever. This is the primary
        no-false-positives guarantee the sweep must uphold."""
        caplog.set_level(logging.WARNING, logger="amplifier_module_provider_openai")
        real_config = {
            # infrastructure / entry metadata that may accompany config
            "id": "openai-primary",
            "module": "provider-openai",
            "source": "~/.amplifier/settings.yaml",
            "api_key": "${OPENAI_API_KEY}",
            "priority": 10,
            # every module-consumed key, one representative value each
            "base_url": "https://api.openai.com/v1",
            "default_model": "gpt-5.6-sol",
            "max_tokens": None,
            "temperature": 0.7,
            "reasoning": None,
            "reasoning_effort": "low",
            "reasoning_summary": "detailed",
            "truncation": None,
            "enable_state": False,
            "raw": False,
            "timeout": 600.0,
            "filtered": True,
            "prompt_cache_key": "",
            "prompt_cache_retention": None,
            "prompt_cache_options": None,
            "safety_identifier": None,
            "enable_response_chaining": "auto",
            "poll_interval": 5,
            "background_timeout": 3600,
            "enable_long_context": False,
            "use_streaming": True,
            "max_retries": 5,
            "min_retry_delay": 1.0,
            "max_retry_delay": 60.0,
            "retry_jitter": True,
            "max_concurrent_requests": 5,
            "thinking_budget_tokens": 0,
            "thinking_budget_buffer": 1024,
        }
        # text_verbosity / enable_reasoning_context are gpt-5.6-only, and
        # default_model above IS gpt-5.6-sol, so they're legitimate here too.
        real_config["text_verbosity"] = None
        real_config["enable_reasoning_context"] = False

        OpenAIProvider(api_key="test-key", config=real_config)

        matches = [
            r.message for r in caplog.records if _UNKNOWN_KEY_MARKER in r.message
        ]
        assert matches == [], (
            f"unexpected unknown-key warning(s) on a legitimate config: {matches}"
        )

    def test_clean_minimal_config_is_silent(self, caplog):
        """The bar case: an empty/minimal config must not warn either."""
        caplog.set_level(logging.WARNING, logger="amplifier_module_provider_openai")
        _make_provider()
        matches = [
            r.message for r in caplog.records if _UNKNOWN_KEY_MARKER in r.message
        ]
        assert matches == []


# ---------------------------------------------------------------------------
# EXTRA_KNOWN_CONFIG_KEYS: the subclass extension point
#
# Fixes the provider-azure-openai false positive: AzureOpenAIProvider
# SUBCLASSES OpenAIProvider (see amplifier-module-provider-azure-openai's
# `_create_azure_provider`) and passes its own config straight through the
# same `config` dict this constructor reads. Before this extension point,
# every legitimate azure_endpoint/api_version/use_managed_identity/etc
# config tripped the generic sweep -- valid config, false positive, exactly
# the failure mode the sweep was built to prevent.
# ---------------------------------------------------------------------------


class TestExtraKnownConfigKeysExtensionPoint:
    """A subclass declares `EXTRA_KNOWN_CONFIG_KEYS` to add its own
    recognized keys to the sweep, without altering base-class behavior."""

    @staticmethod
    def _make_subclass_provider(extra_keys, **config_overrides):
        class _ConsumerProvider(OpenAIProvider):
            EXTRA_KNOWN_CONFIG_KEYS = frozenset(extra_keys)

        config = {"max_retries": 0, "use_streaming": False, **config_overrides}
        return _ConsumerProvider(api_key="test-key", config=config)

    def test_default_is_empty_frozenset(self):
        """Direct use of OpenAIProvider (and any subclass that doesn't
        override the attribute) is completely unaffected."""
        assert OpenAIProvider.EXTRA_KNOWN_CONFIG_KEYS == frozenset()

    def test_subclass_declared_key_is_silent(self, caplog):
        """A key the subclass declares via EXTRA_KNOWN_CONFIG_KEYS never
        warns, even though the base class doesn't consume it."""
        caplog.set_level(logging.WARNING, logger="amplifier_module_provider_openai")
        self._make_subclass_provider(
            {"azure_endpoint"},
            azure_endpoint="https://example.openai.azure.com",
        )

        matches = [
            r.message for r in caplog.records if _UNKNOWN_KEY_MARKER in r.message
        ]
        assert matches == [], f"consumer-declared key should never warn: {matches}"

    def test_subclass_typo_in_own_key_still_warns_with_suggestion(self, caplog):
        """A genuine typo of the subclass's OWN declared key must still
        warn, with a did-you-mean suggestion drawn from the merged set."""
        caplog.set_level(logging.WARNING, logger="amplifier_module_provider_openai")
        self._make_subclass_provider(
            {"azure_endpoint"},
            azure_endpont="https://example.openai.azure.com",
        )

        matches = [
            r.message for r in caplog.records if _UNKNOWN_KEY_MARKER in r.message
        ]
        assert len(matches) == 1
        message = matches[0]
        assert "'azure_endpont'" in message
        assert "azure_endpoint" in message
        assert "did you mean" in message.lower()

    def test_subclass_genuine_unknown_key_still_warns(self, caplog):
        """A subclass's own unrelated typo (no close match anywhere) still
        warns without fabricating a suggestion."""
        caplog.set_level(logging.WARNING, logger="amplifier_module_provider_openai")
        self._make_subclass_provider(
            {"azure_endpoint"}, totally_bogus_azure_key_xyz=True
        )

        matches = [
            r.message for r in caplog.records if _UNKNOWN_KEY_MARKER in r.message
        ]
        assert len(matches) == 1
        assert "'totally_bogus_azure_key_xyz'" in matches[0]

    def test_direct_use_still_warns_for_subclass_only_keys(self, caplog):
        """Guard against accidentally widening the base class: a DIRECT
        OpenAIProvider instance (no subclass override) must still warn on
        keys that are only known via some other subclass's extension."""
        caplog.set_level(logging.WARNING, logger="amplifier_module_provider_openai")
        _make_provider(azure_endpoint="https://example.openai.azure.com")

        matches = [
            r.message for r in caplog.records if _UNKNOWN_KEY_MARKER in r.message
        ]
        assert len(matches) == 1
        assert "'azure_endpoint'" in matches[0]

    def test_warn_unknown_config_keys_unit_merges_extra_keys(self, caplog):
        """Unit-level check of the free function's `extra_known_keys` param
        directly, independent of any subclassing."""
        from amplifier_module_provider_openai import _warn_unknown_config_keys

        caplog.set_level(logging.WARNING, logger="amplifier_module_provider_openai")
        _warn_unknown_config_keys({"custom_key": 1}, frozenset({"custom_key"}))

        matches = [
            r.message for r in caplog.records if _UNKNOWN_KEY_MARKER in r.message
        ]
        assert matches == []


# ---------------------------------------------------------------------------
# _KNOWN_CONFIG_KEYS bookkeeping
# ---------------------------------------------------------------------------


def test_known_config_keys_has_no_accidental_overlap_gaps():
    """Sanity check on the KNOWN_KEYS bookkeeping itself: the three
    constituent sets (consumed / recognized-inert / infrastructure) are
    non-overlapping, so the audit trail in the module stays legible."""
    from amplifier_module_provider_openai import (
        _CONSUMED_CONFIG_KEYS,
        _DEPRECATED_ALIAS_CONFIG_KEYS,
        _INFRASTRUCTURE_CONFIG_KEYS,
        _KNOWN_CONFIG_KEYS,
        _RECOGNIZED_INERT_CONFIG_KEYS,
    )

    assert _CONSUMED_CONFIG_KEYS.isdisjoint(_RECOGNIZED_INERT_CONFIG_KEYS)
    assert _CONSUMED_CONFIG_KEYS.isdisjoint(_INFRASTRUCTURE_CONFIG_KEYS)
    assert _CONSUMED_CONFIG_KEYS.isdisjoint(_DEPRECATED_ALIAS_CONFIG_KEYS)
    assert _RECOGNIZED_INERT_CONFIG_KEYS.isdisjoint(_INFRASTRUCTURE_CONFIG_KEYS)
    assert _KNOWN_CONFIG_KEYS == (
        _CONSUMED_CONFIG_KEYS
        | _RECOGNIZED_INERT_CONFIG_KEYS
        | _DEPRECATED_ALIAS_CONFIG_KEYS
        | _INFRASTRUCTURE_CONFIG_KEYS
    )
    # 28 keys the survey counted: 32 - 5 removed (enable_state,
    # enable_reasoning_context, enable_response_chaining,
    # thinking_budget_tokens, thinking_budget_buffer) + 1 added
    # (extra_request_params), audited against every `self.config.get(...)`
    # call site in the constructor and request path.
    assert len(_CONSUMED_CONFIG_KEYS) == 28
