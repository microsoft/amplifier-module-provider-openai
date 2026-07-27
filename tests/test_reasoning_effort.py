"""Phase 2: reasoning_effort support tests.

Verifies the precedence chain:
  kwargs["reasoning"] > request.reasoning_effort > config default > None

And that reasoning_effort maps to the correct OpenAI reasoning param format.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from amplifier_core.message_models import ChatRequest, Message

from amplifier_module_provider_openai import OpenAIProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class DummyResponse:
    """Minimal response stub."""

    def __init__(self):
        self.output = [
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text="Hi")],
            )
        ]
        self.usage = SimpleNamespace(input_tokens=10, output_tokens=5)
        self.status = "completed"
        self.id = "resp_test"


def _make_provider(**config_overrides) -> OpenAIProvider:
    config = {"max_retries": 0, "use_streaming": False, **config_overrides}
    provider = OpenAIProvider(api_key="test-key", config=config)
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())
    return provider


def _request_with_effort(effort: str | None) -> ChatRequest:
    return ChatRequest(
        messages=[Message(role="user", content="Hello")],
        reasoning_effort=effort,
    )


def _get_call_kwargs(provider: OpenAIProvider) -> dict:
    """Extract the kwargs from the last client.responses.create call."""
    return provider.client.responses.create.await_args_list[-1].kwargs


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_reasoning_effort_high():
    """reasoning_effort='high' -> reasoning={'effort': 'high', 'summary': ...}."""
    provider = _make_provider()
    asyncio.run(provider.complete(_request_with_effort("high")))

    kwargs = _get_call_kwargs(provider)
    assert "reasoning" in kwargs
    assert kwargs["reasoning"]["effort"] == "high"
    assert "summary" in kwargs["reasoning"]


def test_reasoning_effort_low():
    """reasoning_effort='low' -> reasoning={'effort': 'low', 'summary': ...}."""
    provider = _make_provider()
    asyncio.run(provider.complete(_request_with_effort("low")))

    kwargs = _get_call_kwargs(provider)
    assert "reasoning" in kwargs
    assert kwargs["reasoning"]["effort"] == "low"
    assert "summary" in kwargs["reasoning"]


def test_reasoning_effort_medium():
    """reasoning_effort='medium' -> reasoning={'effort': 'medium', 'summary': ...}."""
    provider = _make_provider()
    asyncio.run(provider.complete(_request_with_effort("medium")))

    kwargs = _get_call_kwargs(provider)
    assert "reasoning" in kwargs
    assert kwargs["reasoning"]["effort"] == "medium"
    assert "summary" in kwargs["reasoning"]


def test_reasoning_effort_none_no_reasoning_param():
    """reasoning_effort=None and no config -> no reasoning param sent.
    Uses a non-reasoning model since reasoning-capable models (gpt-5.*, o-series,
    codex) now auto-set reasoning={summary: 'auto'} for observability."""
    provider = _make_provider(default_model="gpt-4.1-mini")  # non-reasoning model
    asyncio.run(provider.complete(_request_with_effort(None)))

    kwargs = _get_call_kwargs(provider)
    assert "reasoning" not in kwargs


def test_kwargs_reasoning_overrides_request_effort():
    """kwargs['reasoning'] takes precedence over request.reasoning_effort."""
    provider = _make_provider()
    request = _request_with_effort("medium")

    # Pass kwargs reasoning that should override
    asyncio.run(
        provider.complete(request, reasoning={"effort": "high", "summary": "concise"})
    )

    kwargs = _get_call_kwargs(provider)
    assert kwargs["reasoning"]["effort"] == "high"
    assert kwargs["reasoning"]["summary"] == "concise"


def test_config_default_used_when_no_request_effort():
    """Config reasoning default is used when request.reasoning_effort is None."""
    provider = _make_provider(reasoning="low")
    asyncio.run(provider.complete(_request_with_effort(None)))

    kwargs = _get_call_kwargs(provider)
    assert kwargs["reasoning"]["effort"] == "low"


def test_request_effort_overrides_config_default():
    """request.reasoning_effort overrides config default."""
    provider = _make_provider(reasoning="low")
    asyncio.run(provider.complete(_request_with_effort("high")))

    kwargs = _get_call_kwargs(provider)
    assert kwargs["reasoning"]["effort"] == "high"


def test_reasoning_summary_from_config():
    """reasoning_summary config is used in the built reasoning param."""
    provider = _make_provider(reasoning_summary="concise")
    asyncio.run(provider.complete(_request_with_effort("high")))

    kwargs = _get_call_kwargs(provider)
    assert kwargs["reasoning"]["summary"] == "concise"


def test_reasoning_effort_with_extended_thinking():
    """reasoning_effort sets reasoning, extended_thinking doesn't override it."""
    provider = _make_provider()
    request = _request_with_effort("low")

    # extended_thinking=True but reasoning already set by reasoning_effort
    asyncio.run(provider.complete(request, extended_thinking=True))

    kwargs = _get_call_kwargs(provider)
    # reasoning_effort="low" already set reasoning, so it should be "low"
    assert kwargs["reasoning"]["effort"] == "low"


def test_reasoning_effort_xhigh():
    """reasoning_effort='xhigh' -> reasoning={'effort': 'xhigh', 'summary': ...}."""
    provider = _make_provider()
    asyncio.run(provider.complete(_request_with_effort("xhigh")))
    kwargs = _get_call_kwargs(provider)
    assert "reasoning" in kwargs
    assert kwargs["reasoning"]["effort"] == "xhigh"
    assert "summary" in kwargs["reasoning"]


def test_reasoning_effort_none_explicit():
    """reasoning_effort='none' explicitly set -> reasoning={'effort': 'none', 'summary': ...}.
    This is different from reasoning_effort=None (Python None = not set).
    GPT-5.4 uses 'none' as a string value meaning 'no reasoning'."""
    provider = _make_provider()
    asyncio.run(provider.complete(_request_with_effort("none")))
    kwargs = _get_call_kwargs(provider)
    assert "reasoning" in kwargs
    assert kwargs["reasoning"]["effort"] == "none"
    assert "summary" in kwargs["reasoning"]


# ---------------------------------------------------------------------------
# Canonical `reasoning_effort` config key (portable kernel key)
# ---------------------------------------------------------------------------


def test_config_reasoning_effort_honored_on_normal_path():
    """Canonical config reasoning_effort='high' -> reasoning={'effort': 'high'}
    on the NORMAL request path (no extended_thinking kwarg needed)."""
    provider = _make_provider(reasoning_effort="high")
    asyncio.run(provider.complete(_request_with_effort(None)))

    kwargs = _get_call_kwargs(provider)
    assert "reasoning" in kwargs
    assert kwargs["reasoning"]["effort"] == "high"
    assert "summary" in kwargs["reasoning"]


def test_config_reasoning_effort_none_stays_inert():
    """reasoning_effort='none' (the provisioned ConfigField default) must NOT
    start injecting a reasoning param — absence keeps today's behavior."""
    provider = _make_provider(reasoning_effort="none")
    asyncio.run(provider.complete(_request_with_effort(None)))

    kwargs = _get_call_kwargs(provider)
    # Default model (gpt-5.6-sol) has default_reasoning_effort=None, so no
    # reasoning param is auto-injected either.
    assert "reasoning" not in kwargs


def test_config_reasoning_effort_absent_unchanged():
    """No effort-family config at all -> no reasoning param (existing behavior)."""
    provider = _make_provider()
    asyncio.run(provider.complete(_request_with_effort(None)))

    kwargs = _get_call_kwargs(provider)
    assert "reasoning" not in kwargs


def test_request_effort_overrides_config_reasoning_effort():
    """request.reasoning_effort='low' beats config reasoning_effort='high'."""
    provider = _make_provider(reasoning_effort="high")
    asyncio.run(provider.complete(_request_with_effort("low")))

    kwargs = _get_call_kwargs(provider)
    assert kwargs["reasoning"]["effort"] == "low"


def test_kwargs_reasoning_effort_honored_on_normal_path():
    """Per-call kwargs['reasoning_effort'] works WITHOUT extended_thinking."""
    provider = _make_provider()
    asyncio.run(provider.complete(_request_with_effort(None), reasoning_effort="xhigh"))

    kwargs = _get_call_kwargs(provider)
    assert kwargs["reasoning"]["effort"] == "xhigh"


def test_config_reasoning_effort_wins_over_legacy_reasoning(caplog):
    """Both config keys set -> canonical reasoning_effort wins, with a warning."""
    import logging

    with caplog.at_level(logging.WARNING):
        provider = _make_provider(reasoning_effort="high", reasoning="low")
    asyncio.run(provider.complete(_request_with_effort(None)))

    kwargs = _get_call_kwargs(provider)
    assert kwargs["reasoning"]["effort"] == "high"
    assert any(
        "reasoning_effort" in r.message and "wins" in r.message for r in caplog.records
    )


def test_legacy_reasoning_config_still_works_alone():
    """config reasoning='low' alone keeps working (legacy alias, unchanged)."""
    provider = _make_provider(reasoning="low")
    asyncio.run(provider.complete(_request_with_effort(None)))

    kwargs = _get_call_kwargs(provider)
    assert kwargs["reasoning"]["effort"] == "low"


def test_config_reasoning_effort_capability_gated_no_op(caplog):
    """Non-reasoning model + config reasoning_effort -> loud no-op, no param."""
    import logging

    provider = _make_provider(
        default_model="gpt-4.1-mini",  # supports_reasoning=False
        reasoning_effort="high",
    )
    with caplog.at_level(logging.WARNING):
        asyncio.run(provider.complete(_request_with_effort(None)))

    kwargs = _get_call_kwargs(provider)
    assert "reasoning" not in kwargs
    assert any(
        "reasoning_effort" in r.message and "does not support reasoning" in r.message
        for r in caplog.records
    )


def test_config_reasoning_effort_composes_with_capability_max_tokens():
    """Combined P5 + canonical-effort behavior in ONE request: with no
    max_tokens config, the output budget derives from the model's capability
    limit (P5, PR #54) AND config reasoning_effort injects the reasoning
    param — the two are orthogonal and must both land in the same params.
    """
    from amplifier_module_provider_openai._capabilities import get_capabilities

    provider = _make_provider(reasoning_effort="high")  # no max_tokens config
    asyncio.run(provider.complete(_request_with_effort(None)))

    kwargs = _get_call_kwargs(provider)
    # P5: capability-derived output budget (gpt-5.6-sol default model)
    expected_max = get_capabilities(provider.default_model).max_output_tokens
    assert kwargs["max_output_tokens"] == expected_max
    # Canonical effort key honored on the same request
    assert kwargs["reasoning"]["effort"] == "high"
    assert "summary" in kwargs["reasoning"]


def test_config_reasoning_effort_normalized():
    """Mixed case / whitespace values are normalized ('  High ' -> 'high')."""
    provider = _make_provider(reasoning_effort="  High ")
    asyncio.run(provider.complete(_request_with_effort(None)))

    kwargs = _get_call_kwargs(provider)
    assert kwargs["reasoning"]["effort"] == "high"


# ---------------------------------------------------------------------------
# Mount-time validation of config reasoning_effort
# ---------------------------------------------------------------------------


def test_invalid_config_reasoning_effort_fails_loud_at_mount():
    """Unknown value ('ultra') raises at provider construction (mount time),
    not as an API 400 mid-session."""
    import pytest

    with pytest.raises(ValueError, match="reasoning_effort.*ultra"):
        OpenAIProvider(api_key="test-key", config={"reasoning_effort": "ultra"})


def test_gpt_5_5_pro_disallowed_config_effort_fails_at_mount():
    """gpt-5.5-pro accepts only {medium, high, xhigh}; 'low' fails at mount."""
    import pytest

    with pytest.raises(ValueError, match="gpt-5.5-pro"):
        OpenAIProvider(
            api_key="test-key",
            config={"default_model": "gpt-5.5-pro", "reasoning_effort": "low"},
        )


def test_gpt_5_5_pro_allowed_config_effort_mounts():
    """gpt-5.5-pro + reasoning_effort='high' constructs fine."""
    provider = OpenAIProvider(
        api_key="test-key",
        config={"default_model": "gpt-5.5-pro", "reasoning_effort": "high"},
    )
    assert provider.reasoning_effort == "high"


# ---------------------------------------------------------------------------
# Inert effort-family key warning
# ---------------------------------------------------------------------------


def test_unconsumed_effort_key_warns_at_init(caplog):
    """config 'effort' (Anthropic-style alias) is inert here -> loud warning."""
    import logging

    with caplog.at_level(logging.WARNING):
        provider = _make_provider(effort="high")

    assert any(
        "'effort'" in r.message and "not consumed" in r.message for r in caplog.records
    )

    # And it must stay inert: no reasoning param is injected by it.
    asyncio.run(provider.complete(_request_with_effort(None)))
    kwargs = _get_call_kwargs(provider)
    assert "reasoning" not in kwargs


def test_no_effort_warnings_for_clean_config(caplog):
    """A config without effort-family keys emits no effort warnings."""
    import logging

    with caplog.at_level(logging.WARNING):
        _make_provider()

    assert not any(
        "not consumed" in r.message or "canonical" in r.message for r in caplog.records
    )


def test_gpt54_without_effort_still_includes_encrypted_content():
    """GPT-5.4 stateless path (chaining off) -> include=[reasoning.encrypted_content] IS sent.

    GPT-5.4 is a reasoning-capable model (supports_reasoning=True) that CAN produce
    reasoning tokens even without explicit effort. Without include=[reasoning.encrypted_content],
    reasoning token content is lost when store=false (Amplifier's default), causing
    orphaned reasoning references (70 occurrences observed on test device).

    PR-B note: gpt-5.4 now defaults to response chaining (auto mode, supports_reasoning=True),
    which suppresses encrypted_content include in favour of server-side state. This test
    validates the stateless fallback path, exercised when enable_response_chaining=False
    (ZDR opt-out / regulated-industry deployments).

    Regression test: the include-guard incorrectly gated on default_reasoning_effort
    (None for GPT-5.4), but reasoning-capable models CAN produce tokens even without
    explicit effort. The guard must use supports_reasoning (the capability flag) instead,
    matching the Anthropic provider's pattern of always preserving thinking content for
    capable models.
    """
    # Disable chaining to exercise the stateless reasoning path (ZDR opt-out / pre-PR-B baseline)
    provider = _make_provider(default_model="gpt-5.4", enable_response_chaining=False)
    asyncio.run(provider.complete(_request_with_effort(None)))
    kwargs = _get_call_kwargs(provider)
    assert "include" in kwargs, (
        "GPT-5.4 stateless path (chaining=False): include=[reasoning.encrypted_content] must be "
        "sent even without explicit reasoning_effort, to prevent silent reasoning token loss."
    )
    assert kwargs["include"] == ["reasoning.encrypted_content"]
