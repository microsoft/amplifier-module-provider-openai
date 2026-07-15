"""Tests for capability-driven reasoning.effort gating.

PR #47 added "max" to the global reasoning_effort wizard choices, but "max" is
GPT-5.6-only -- a gpt-5.5 (or gpt-5.4, o-series, ...) request with effort="max"
would return an opaque API 400. `_validate_reasoning_effort` makes this
capability-driven and composes (AND) with the existing
`_validate_gpt_5_5_pro_effort`.
"""

import asyncio
from unittest.mock import AsyncMock

import pytest
from amplifier_core import llm_errors as kernel_errors
from amplifier_core.message_models import ChatRequest, Message

from amplifier_module_provider_openai import (
    OpenAIProvider,
    _validate_gpt_5_5_pro_effort,
    _validate_reasoning_effort,
)


# ---------------------------------------------------------------------------
# gpt-5.5 rejects 'max' and 'minimal'
# ---------------------------------------------------------------------------


class TestGPT55RejectsMax:
    def test_string_rejected(self):
        with pytest.raises(kernel_errors.InvalidRequestError) as exc:
            _validate_reasoning_effort("gpt-5.5", "max")
        msg = str(exc.value)
        assert "max" in msg
        assert "gpt-5.5" in msg

    def test_dict_rejected(self):
        with pytest.raises(kernel_errors.InvalidRequestError):
            _validate_reasoning_effort("gpt-5.5", {"effort": "max"})


class TestGPT55RejectsMinimal:
    def test_string_rejected(self):
        with pytest.raises(kernel_errors.InvalidRequestError):
            _validate_reasoning_effort("gpt-5.5", "minimal")

    def test_dict_rejected(self):
        with pytest.raises(kernel_errors.InvalidRequestError):
            _validate_reasoning_effort("gpt-5.5", {"effort": "minimal"})


class TestGPT55Accepts:
    @pytest.mark.parametrize("effort", ["none", "low", "medium", "high", "xhigh", None])
    def test_accepted(self, effort):
        _validate_reasoning_effort("gpt-5.5", effort)  # must not raise


# ---------------------------------------------------------------------------
# gpt-5.6 accepts 'max', still rejects 'minimal'
# ---------------------------------------------------------------------------


class TestGPT56:
    def test_accepts_max(self):
        _validate_reasoning_effort("gpt-5.6-sol", "max")  # must not raise

    def test_rejects_minimal(self):
        with pytest.raises(kernel_errors.InvalidRequestError):
            _validate_reasoning_effort("gpt-5.6-sol", "minimal")


# ---------------------------------------------------------------------------
# Permissive models (allowed_reasoning_efforts=None) -- always a no-op
# ---------------------------------------------------------------------------


class TestPermissiveModels:
    @pytest.mark.parametrize("model", ["gpt-5.4", "gpt-4o", "o3", "gpt-5.3-codex"])
    @pytest.mark.parametrize("effort", ["max", "minimal", "low"])
    def test_no_op(self, model, effort):
        _validate_reasoning_effort(model, effort)  # must not raise


# ---------------------------------------------------------------------------
# Integration -- zero network traffic on rejection path
# ---------------------------------------------------------------------------


def test_5_5_max_blocks_network():
    provider = OpenAIProvider(
        api_key="test-key",
        config={
            "max_retries": 0,
            "use_streaming": False,
            "default_model": "gpt-5.5",
            "reasoning_effort": "max",
        },
    )
    mock_create = AsyncMock()
    provider.client.responses.create = mock_create

    request = ChatRequest(
        messages=[Message(role="user", content="ok")],
        reasoning_effort="max",
    )

    with pytest.raises(kernel_errors.InvalidRequestError):
        asyncio.run(provider.complete(request))
    mock_create.assert_not_called()


# ---------------------------------------------------------------------------
# Composition with the gpt-5.5-pro validator
# ---------------------------------------------------------------------------


def test_5_5_pro_and_family_compose():
    """Both validators run in sequence for gpt-5.5-pro (mirrors the composition
    order in _build_params: pro validator first, then the family validator).

    "low" passes the family gate ({none,low,medium,high,xhigh} includes "low")
    but fails the narrower pro-specific gate ({medium,high,xhigh}) -- so the
    pro validator is the one that must raise. "medium" passes both.
    """
    # Family gate alone does NOT reject "low" for the gpt-5.5 family.
    _validate_reasoning_effort("gpt-5.5-pro", "low")  # must not raise

    # The pro-specific validator is the one that rejects it.
    with pytest.raises(kernel_errors.InvalidRequestError):
        _validate_gpt_5_5_pro_effort("gpt-5.5-pro", "low")

    # "medium" passes both the family gate and the pro gate.
    _validate_gpt_5_5_pro_effort("gpt-5.5-pro", "medium")  # must not raise
    _validate_reasoning_effort("gpt-5.5-pro", "medium")  # must not raise

    # End-to-end: the full pipeline (pro validator first, then family
    # validator) rejects "low" and accepts "medium" for gpt-5.5-pro.
    provider = OpenAIProvider(
        api_key="test-key",
        config={
            "max_retries": 0,
            "use_streaming": False,
            "default_model": "gpt-5.5-pro",
        },
    )
    provider.client.responses.create = AsyncMock()
    with pytest.raises(kernel_errors.InvalidRequestError):
        asyncio.run(
            provider.complete(
                ChatRequest(
                    messages=[Message(role="user", content="ok")],
                    reasoning_effort="low",
                )
            )
        )
    provider.client.responses.create.assert_not_called()
