"""Tests for the gpt-5.5-pro reasoning.effort pre-flight validator.

Verified against the live API on 2026-04-24:
    Allowed:  {medium, high, xhigh}
    Rejected: {minimal, none, low}

The validator catches disallowed values BEFORE any API call so callers
get a clear error instead of an opaque HTTP 400.
"""

import asyncio
from unittest.mock import AsyncMock

import pytest
from amplifier_core import llm_errors as kernel_errors
from amplifier_core.message_models import ChatRequest, Message

from amplifier_module_provider_openai import (
    OpenAIProvider,
    _validate_gpt_5_5_pro_effort,
)


# ---------------------------------------------------------------------------
# Reject set
# ---------------------------------------------------------------------------


class TestRejectsBelowMedium:
    @pytest.mark.parametrize("effort", ["minimal", "none", "low"])
    def test_string_rejected(self, effort):
        with pytest.raises(kernel_errors.InvalidRequestError):
            _validate_gpt_5_5_pro_effort("gpt-5.5-pro", effort)

    @pytest.mark.parametrize("effort", ["minimal", "none", "low"])
    def test_dict_rejected(self, effort):
        with pytest.raises(kernel_errors.InvalidRequestError):
            _validate_gpt_5_5_pro_effort("gpt-5.5-pro", {"effort": effort})


# ---------------------------------------------------------------------------
# Allow set
# ---------------------------------------------------------------------------


class TestAllowsMediumOrAbove:
    @pytest.mark.parametrize("effort", ["medium", "high", "xhigh"])
    def test_string_accepted(self, effort):
        _validate_gpt_5_5_pro_effort("gpt-5.5-pro", effort)

    @pytest.mark.parametrize("effort", ["medium", "high", "xhigh"])
    def test_dict_accepted(self, effort):
        _validate_gpt_5_5_pro_effort("gpt-5.5-pro", {"effort": effort})


# ---------------------------------------------------------------------------
# Unset / partial dicts
# ---------------------------------------------------------------------------


class TestAllowsUnset:
    def test_none_param(self):
        _validate_gpt_5_5_pro_effort("gpt-5.5-pro", None)

    def test_dict_with_explicit_none_effort(self):
        _validate_gpt_5_5_pro_effort("gpt-5.5-pro", {"effort": None})

    def test_dict_without_effort_key(self):
        _validate_gpt_5_5_pro_effort("gpt-5.5-pro", {"summary": "auto"})


# ---------------------------------------------------------------------------
# Dated snapshots
# ---------------------------------------------------------------------------


class TestDatedSnapshots:
    def test_dated_snapshot_rejected(self):
        with pytest.raises(kernel_errors.InvalidRequestError):
            _validate_gpt_5_5_pro_effort("gpt-5.5-pro-2026-04-23", "low")

    def test_dated_snapshot_accepted(self):
        _validate_gpt_5_5_pro_effort("gpt-5.5-pro-2026-04-23", "high")


# ---------------------------------------------------------------------------
# Non-pro / other models — validator must be a no-op
# ---------------------------------------------------------------------------


class TestNonProModels:
    @pytest.mark.parametrize(
        "model",
        ["gpt-5.5", "gpt-5.5-2026-04-23", "gpt-5.4", "gpt-5.4-pro", "gpt-4o", "o3"],
    )
    @pytest.mark.parametrize("effort", ["minimal", "none", "low", "medium", "high"])
    def test_no_op_on_non_pro_models(self, model, effort):
        _validate_gpt_5_5_pro_effort(model, effort)


# ---------------------------------------------------------------------------
# Error message contract
# ---------------------------------------------------------------------------


class TestErrorMessage:
    def test_error_message_lists_allowed_set(self):
        with pytest.raises(kernel_errors.InvalidRequestError) as exc:
            _validate_gpt_5_5_pro_effort("gpt-5.5-pro", "low")
        msg = str(exc.value)
        assert "gpt-5.5-pro" in msg
        assert "low" in msg
        for allowed in ("medium", "high", "xhigh"):
            assert allowed in msg


# ---------------------------------------------------------------------------
# Integration — zero network traffic on rejection path
# ---------------------------------------------------------------------------


class TestNoNetworkOnRejection:
    """Validator must raise BEFORE the SDK is called."""

    def test_string_low_blocks_network(self):
        provider = OpenAIProvider(
            api_key="test-key",
            config={
                "max_retries": 0,
                "use_streaming": False,
                "default_model": "gpt-5.5-pro",
            },
        )
        mock_create = AsyncMock()
        provider.client.responses.create = mock_create

        request = ChatRequest(
            messages=[Message(role="user", content="ok")],
            reasoning_effort="low",
        )

        with pytest.raises(kernel_errors.InvalidRequestError):
            asyncio.run(provider.complete(request))
        mock_create.assert_not_called()

    def test_dict_low_blocks_network(self):
        provider = OpenAIProvider(
            api_key="test-key",
            config={
                "max_retries": 0,
                "use_streaming": False,
                "default_model": "gpt-5.5-pro",
                "reasoning": {"effort": "low", "summary": "auto"},
            },
        )
        mock_create = AsyncMock()
        provider.client.responses.create = mock_create

        request = ChatRequest(messages=[Message(role="user", content="ok")])
        with pytest.raises(kernel_errors.InvalidRequestError):
            asyncio.run(provider.complete(request))
        mock_create.assert_not_called()
