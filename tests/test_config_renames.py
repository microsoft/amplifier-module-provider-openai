"""Tests for the max_tokens->max_output_tokens and filtered->hide_dated_models
config-key renames (back-compat alias + one-shot deprecation warning).

Both renames share the same `_read_renamed_config()` helper: the new key
always wins when present (even when both are set); the old key still works
but emits exactly one deprecation warning naming both names; neither old key
name triggers the generic unknown-key sweep.
"""

import logging

from amplifier_module_provider_openai import OpenAIProvider

_LOGGER = "amplifier_module_provider_openai"
_UNKNOWN_MARKER = "Unrecognized config key"


def _make_provider(**config_overrides) -> OpenAIProvider:
    config = {"max_retries": 0, "use_streaming": False, **config_overrides}
    return OpenAIProvider(api_key="test-key", config=config)


# ---------------------------------------------------------------------------
# max_tokens -> max_output_tokens
# ---------------------------------------------------------------------------


def test_max_output_tokens_new_key_no_warning(caplog):
    caplog.set_level(logging.WARNING, logger=_LOGGER)
    provider = _make_provider(max_output_tokens=2048)
    assert provider.max_output_tokens == 2048
    assert not any(
        "max_output_tokens" in r.message or "max_tokens" in r.message
        for r in caplog.records
        if r.levelno == logging.WARNING
    )


def test_max_tokens_old_key_one_deprecation_warning(caplog):
    caplog.set_level(logging.WARNING, logger=_LOGGER)
    provider = _make_provider(max_tokens=2048)
    assert provider.max_output_tokens == 2048

    warnings = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and "deprecated" in r.message
    ]
    assert len(warnings) == 1
    assert "max_tokens" in warnings[0].message
    assert "max_output_tokens" in warnings[0].message


def test_max_output_tokens_both_set_new_wins_with_warning(caplog):
    caplog.set_level(logging.WARNING, logger=_LOGGER)
    provider = _make_provider(max_tokens=1111, max_output_tokens=2222)
    assert provider.max_output_tokens == 2222

    warnings = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and "BOTH" in r.message
    ]
    assert len(warnings) == 1
    assert "max_tokens" in warnings[0].message
    assert "max_output_tokens" in warnings[0].message


def test_max_output_tokens_neither_set_is_none():
    provider = _make_provider()
    assert provider.max_output_tokens is None


def test_max_tokens_old_key_never_triggers_unknown_sweep(caplog):
    caplog.set_level(logging.WARNING, logger=_LOGGER)
    _make_provider(max_tokens=2048)
    assert not any(
        _UNKNOWN_MARKER in r.message and "max_tokens" in r.message
        for r in caplog.records
    )


# ---------------------------------------------------------------------------
# filtered -> hide_dated_models
# ---------------------------------------------------------------------------


def test_hide_dated_models_new_key_no_warning(caplog):
    caplog.set_level(logging.WARNING, logger=_LOGGER)
    provider = _make_provider(hide_dated_models=False)
    assert provider.hide_dated_models is False
    assert not any(
        "deprecated" in r.message
        for r in caplog.records
        if r.levelno == logging.WARNING
    )


def test_filtered_old_key_one_deprecation_warning(caplog):
    caplog.set_level(logging.WARNING, logger=_LOGGER)
    provider = _make_provider(filtered=False)
    assert provider.hide_dated_models is False

    warnings = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and "deprecated" in r.message
    ]
    assert len(warnings) == 1
    assert "filtered" in warnings[0].message
    assert "hide_dated_models" in warnings[0].message


def test_hide_dated_models_both_set_new_wins_with_warning(caplog):
    caplog.set_level(logging.WARNING, logger=_LOGGER)
    provider = _make_provider(filtered=False, hide_dated_models=True)
    assert provider.hide_dated_models is True

    warnings = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and "BOTH" in r.message
    ]
    assert len(warnings) == 1
    assert "filtered" in warnings[0].message
    assert "hide_dated_models" in warnings[0].message


def test_hide_dated_models_neither_set_defaults_true():
    provider = _make_provider()
    assert provider.hide_dated_models is True


def test_filtered_false_string_yields_false_via_parse_config_bool(caplog):
    """`_parse_config_bool` still applies AFTER the rename resolution --
    the string 'false' must resolve to the actual bool False, not a
    truthy non-empty string."""
    caplog.set_level(logging.WARNING, logger=_LOGGER)
    provider = _make_provider(filtered="false")
    assert provider.hide_dated_models is False


def test_filtered_old_key_never_triggers_unknown_sweep(caplog):
    caplog.set_level(logging.WARNING, logger=_LOGGER)
    _make_provider(filtered=False)
    assert not any(
        _UNKNOWN_MARKER in r.message and "filtered" in r.message for r in caplog.records
    )


# ---------------------------------------------------------------------------
# Deprecated aliases stay `known` (bookkeeping regression guard)
# ---------------------------------------------------------------------------


def test_deprecated_alias_keys_are_a_distinct_frozenset():
    from amplifier_module_provider_openai import (
        _CONSUMED_CONFIG_KEYS,
        _DEPRECATED_ALIAS_CONFIG_KEYS,
        _KNOWN_CONFIG_KEYS,
    )

    assert _DEPRECATED_ALIAS_CONFIG_KEYS == frozenset({"max_tokens", "filtered"})
    assert _DEPRECATED_ALIAS_CONFIG_KEYS.isdisjoint(_CONSUMED_CONFIG_KEYS)
    assert _DEPRECATED_ALIAS_CONFIG_KEYS <= _KNOWN_CONFIG_KEYS
