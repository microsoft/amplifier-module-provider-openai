"""Tests for _cost.py: compute_cost() for provider-openai.

Covers:
  (a) Known model: correct Decimal cost for input tokens
  (b) Output tokens cost
  (c) REQUIRED: Cached request does NOT double-charge
  (d) Unknown model returns None
  (e) None != Decimal('0')
  (f) Result type is always Decimal, never float
  (g) Cache-only (fresh_input=0)
  (h) Models returning None until rates added
  (i) gpt-5.5 pricing
  (j) gpt-5.5-pro: no cache discount
"""

from decimal import Decimal

import pytest

from amplifier_module_provider_openai._cost import compute_cost


# ---------------------------------------------------------------------------
# (a) Known model: gpt-5.4 input cost
# ---------------------------------------------------------------------------
def test_known_model_input_cost():
    """gpt-5.4: 1M input (no cache) → $2.50."""
    result = compute_cost("gpt-5.4", prompt_tokens=1_000_000)
    assert result == Decimal("2.50"), f"Expected Decimal('2.50'), got {result!r}"


# ---------------------------------------------------------------------------
# (b) Output tokens cost
# ---------------------------------------------------------------------------
def test_known_model_output_cost():
    """gpt-5.4: 1M output → $15.00."""
    result = compute_cost("gpt-5.4", completion_tokens=1_000_000)
    assert result == Decimal("15.00"), f"Expected Decimal('15.00'), got {result!r}"


# ---------------------------------------------------------------------------
# (c) REQUIRED: Cached request does NOT double-charge
# ---------------------------------------------------------------------------
def test_cached_request_does_not_double_charge():
    """gpt-5.4: 1M prompt_tokens, 1M cached_tokens → $0.25 (cache_read only).

    fresh_input = 1M - 1M = 0
    cost = 0 × $2.50/M + 0 × $15.00/M + 1M × $0.25/M = $0.25
    """
    result = compute_cost("gpt-5.4", prompt_tokens=1_000_000, cached_tokens=1_000_000)
    assert result == Decimal("0.25"), (
        f"Expected Decimal('0.25') (cache_read only, no double-charge), got {result!r}"
    )


# ---------------------------------------------------------------------------
# (d) Unknown model returns None
# ---------------------------------------------------------------------------
def test_unknown_model_returns_none():
    """An unrecognised model must return None (not 0, not raise)."""
    result = compute_cost("gpt-unknown-9999", prompt_tokens=1_000_000)
    assert result is None


# ---------------------------------------------------------------------------
# (e) None != Decimal('0')
# ---------------------------------------------------------------------------
def test_unknown_distinct_from_zero():
    """None returned for unknown model must not equal Decimal('0')."""
    result = compute_cost("gpt-unknown-9999", prompt_tokens=0)
    assert result is None
    assert result != Decimal("0")


# ---------------------------------------------------------------------------
# (f) Result type is Decimal, not float
# ---------------------------------------------------------------------------
def test_result_type_is_decimal():
    """compute_cost must return a Decimal, not a float."""
    result = compute_cost("gpt-5.4", prompt_tokens=1_000)
    assert isinstance(result, Decimal)
    assert not isinstance(result, float)


# ---------------------------------------------------------------------------
# (g) Cache-only: prompt_tokens == cached_tokens → fresh_input = 0
# ---------------------------------------------------------------------------
def test_cache_only_no_fresh_input():
    """When prompt_tokens == cached_tokens, fresh cost is 0, only cache_read cost."""
    result = compute_cost("gpt-5.5", prompt_tokens=500_000, cached_tokens=500_000)
    expected = Decimal("500000") * Decimal("0.50") / Decimal("1000000")
    assert result == expected


# ---------------------------------------------------------------------------
# (h) Models returning None (not yet priced)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "model",
    [
        "gpt-5.3-codex",
        "gpt-5.2",
        "gpt-5.2-pro",
        "gpt-5.1",
        "gpt-5.1-codex",
        "gpt-5-mini",
    ],
)
def test_unpriced_models_return_none(model):
    """Models listed as 'return None until rates are added' must return None."""
    result = compute_cost(model, prompt_tokens=1_000_000)
    assert result is None, f"Expected None for unpriced model {model!r}, got {result!r}"


# ---------------------------------------------------------------------------
# (i) gpt-5.5 pricing: $5/$30/$0.50
# ---------------------------------------------------------------------------
def test_gpt_55_pricing():
    """gpt-5.5: 1M fresh input → $5.00, 1M output → $30.00, 1M cached → $0.50."""
    assert compute_cost("gpt-5.5", prompt_tokens=1_000_000) == Decimal("5.00")
    assert compute_cost("gpt-5.5", completion_tokens=1_000_000) == Decimal("30.00")
    assert compute_cost("gpt-5.5", cached_tokens=1_000_000) == Decimal("0.50")


# ---------------------------------------------------------------------------
# (j) gpt-5.5-pro: no cache discount (cache_read_per_m = 0.00)
# ---------------------------------------------------------------------------
def test_gpt_55_pro_no_cache_discount():
    """gpt-5.5-pro: cached_tokens=1M → $0 (no cache discount)."""
    result = compute_cost("gpt-5.5-pro", cached_tokens=1_000_000)
    assert result == Decimal("0"), f"Expected Decimal('0'), got {result!r}"
