"""Tests for the targeted migration-warning mechanism (removed config keys).

Config surface V2 removed five config keys entirely: enable_response_chaining,
enable_state, enable_reasoning_context, thinking_budget_tokens,
thinking_budget_buffer. Each gets its OWN targeted warning (naming the key
and what to do instead) via `_INERT_CONFIG_KEY_MESSAGES` -- never the
generic "Unrecognized config key" sweep, and never both.
"""

import logging

import pytest

from amplifier_module_provider_openai import OpenAIProvider

_LOGGER = "amplifier_module_provider_openai"
_UNKNOWN_MARKER = "Unrecognized config key"

_REMOVED_KEYS_AND_SUBSTRINGS = [
    ("enable_response_chaining", "always stateless"),
    ("enable_state", "store is managed automatically"),
    ("enable_reasoning_context", "legacy `reasoning` dict"),
    ("thinking_budget_tokens", "no longer adjusts max_output_tokens"),
    ("thinking_budget_buffer", "see thinking_budget_tokens"),
]


def _make_provider(**config_overrides) -> OpenAIProvider:
    config = {"max_retries": 0, "use_streaming": False, **config_overrides}
    return OpenAIProvider(api_key="test-key", config=config)


def _warnings_naming_key(records, key):
    """Records whose message is ABOUT `key` (i.e. "Config key '<key>' is
    ..."), not merely mentioning it in passing (e.g. thinking_budget_buffer's
    message references thinking_budget_tokens by name)."""
    marker = f"Config key '{key}' is"
    return [r for r in records if r.levelno == logging.WARNING and marker in r.message]


@pytest.mark.parametrize("key,substring", _REMOVED_KEYS_AND_SUBSTRINGS)
def test_removed_key_produces_exactly_one_targeted_warning(key, substring, caplog):
    caplog.set_level(logging.WARNING, logger=_LOGGER)
    _make_provider(**{key: True})

    matching = _warnings_naming_key(caplog.records, key)
    assert len(matching) == 1, (
        f"{key!r} must produce exactly one warning naming it; got {len(matching)}: "
        f"{[r.message for r in caplog.records]}"
    )
    assert substring in matching[0].message, (
        f"{key!r}'s warning must be targeted (contain {substring!r}); "
        f"got: {matching[0].message!r}"
    )


@pytest.mark.parametrize("key,substring", _REMOVED_KEYS_AND_SUBSTRINGS)
def test_removed_key_never_double_warns_via_unknown_sweep(key, substring, caplog):
    """A removed key is still `known` (via _RECOGNIZED_INERT_CONFIG_KEYS), so
    it must never ALSO appear in a generic 'Unrecognized config key' message."""
    caplog.set_level(logging.WARNING, logger=_LOGGER)
    _make_provider(**{key: True})

    assert not any(
        _UNKNOWN_MARKER in r.message and key in r.message for r in caplog.records
    ), f"{key!r} must not appear in an 'Unrecognized config key' warning"


def test_effort_inert_guard_still_fires_with_original_meaning(caplog):
    """Regression guard for the KEEP: the pre-existing 'effort' inert-key
    warning still fires, now sharing the same _INERT_CONFIG_KEY_MESSAGES
    mechanism as the five removed keys."""
    caplog.set_level(logging.WARNING, logger=_LOGGER)
    _make_provider(effort="high")

    matching = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and "effort" in r.message
    ]
    assert len(matching) == 1
    assert "reasoning_effort" in matching[0].message
    assert not any(_UNKNOWN_MARKER in r.message for r in caplog.records)


def test_all_five_removed_keys_together_produce_exactly_five_warnings(caplog):
    caplog.set_level(logging.WARNING, logger=_LOGGER)
    config = {key: True for key, _ in _REMOVED_KEYS_AND_SUBSTRINGS}
    _make_provider(**config)

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    for key, substring in _REMOVED_KEYS_AND_SUBSTRINGS:
        matching = _warnings_naming_key(warnings, key)
        assert len(matching) == 1, f"{key!r} should warn exactly once; got {matching}"
        assert substring in matching[0].message

    assert len(warnings) == 5, (
        f"Expected exactly 5 warnings (one per removed key); got {len(warnings)}: "
        f"{[r.message for r in warnings]}"
    )
    assert not any(_UNKNOWN_MARKER in r.message for r in warnings)
