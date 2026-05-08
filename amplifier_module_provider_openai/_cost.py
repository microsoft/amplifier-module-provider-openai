"""OpenAI pricing rates and cost computation.

Verification date: 2026-05-06
Source: https://openai.com/api/pricing

Only models in the display name registry are included.
Unknown models return None — DO NOT default to $0.00.

Usage
-----
    from amplifier_module_provider_openai._cost import compute_cost
    from decimal import Decimal

    cost = compute_cost(
        "gpt-5.4",
        prompt_tokens=1_000,
        completion_tokens=200,
        cached_tokens=100,
    )
    # Returns Decimal or None if the model is not recognised.

Notes
-----
- O-series: completion_tokens already includes reasoning_tokens (no extra handling needed).
- No cache write cost for OpenAI (unlike Anthropic).
- cached_tokens subtraction happens INSIDE compute_cost to prevent call-site double-charging.
"""

from __future__ import annotations

from decimal import Decimal

# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------

_PER_M = Decimal("1_000_000")

# _RATES maps model-id → {
#   "input_per_m":       Decimal,  # fresh input tokens, per 1M
#   "output_per_m":      Decimal,  # output/completion tokens, per 1M
#   "cache_read_per_m":  Decimal,  # cached input tokens, per 1M (0.00 = no discount)
# }
#
# Rates are in USD.
# Unknown models → return None (DO NOT default to $0.00).
#
# TODO: gpt-5.3-codex, gpt-5.2, gpt-5.2-pro, gpt-5.1, gpt-5.1-codex, gpt-5-mini
#       not yet on pricing page; these models return None until rates are added.
_RATES: dict[str, dict[str, Decimal]] = {
    # ------------------------------------------------------------------
    # GPT 5.5 (DEFAULT)  ($5.00 / $30.00, cache_read $0.50)
    # ------------------------------------------------------------------
    "gpt-5.5": {
        "input_per_m": Decimal("5.00"),
        "output_per_m": Decimal("30.00"),
        "cache_read_per_m": Decimal("0.50"),
    },
    # ------------------------------------------------------------------
    # GPT 5.5 Pro  ($30.00 / $180.00, no cache discount)
    # ------------------------------------------------------------------
    "gpt-5.5-pro": {
        "input_per_m": Decimal("30.00"),
        "output_per_m": Decimal("180.00"),
        "cache_read_per_m": Decimal("0.00"),
    },
    # ------------------------------------------------------------------
    # GPT 5.4 (Azure default)  ($2.50 / $15.00, cache_read $0.25)
    # ------------------------------------------------------------------
    "gpt-5.4": {
        "input_per_m": Decimal("2.50"),
        "output_per_m": Decimal("15.00"),
        "cache_read_per_m": Decimal("0.25"),
    },
    # ------------------------------------------------------------------
    # GPT 5.4 Pro  ($30.00 / $180.00, no cache discount)
    # ------------------------------------------------------------------
    "gpt-5.4-pro": {
        "input_per_m": Decimal("30.00"),
        "output_per_m": Decimal("180.00"),
        "cache_read_per_m": Decimal("0.00"),
    },
    # ------------------------------------------------------------------
    # o3 Deep Research  ($10.00 / $40.00, cache_read $5.00)
    # ------------------------------------------------------------------
    "o3-deep-research": {
        "input_per_m": Decimal("10.00"),
        "output_per_m": Decimal("40.00"),
        "cache_read_per_m": Decimal("5.00"),
    },
    "o3-deep-research-2025-06-26": {
        "input_per_m": Decimal("10.00"),
        "output_per_m": Decimal("40.00"),
        "cache_read_per_m": Decimal("5.00"),
    },
    # ------------------------------------------------------------------
    # o4-mini Deep Research  ($2.00 / $8.00, cache_read $0.275)
    # ------------------------------------------------------------------
    "o4-mini-deep-research": {
        "input_per_m": Decimal("2.00"),
        "output_per_m": Decimal("8.00"),
        "cache_read_per_m": Decimal("0.275"),
    },
    "o4-mini-deep-research-2025-06-26": {
        "input_per_m": Decimal("2.00"),
        "output_per_m": Decimal("8.00"),
        "cache_read_per_m": Decimal("0.275"),
    },
}


def compute_cost(
    model: str,
    *,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    cached_tokens: int = 0,
) -> Decimal | None:
    """Compute the cost of an OpenAI API call in USD.

    Args:
        model: The model ID (e.g. 'gpt-5.4').
        prompt_tokens: Total prompt tokens (TOTAL, includes cached).
            This is response.usage.prompt_tokens.
        completion_tokens: Completion tokens used.
        cached_tokens: Number of prompt tokens served from cache.
            This is response.usage.prompt_tokens_details.cached_tokens.

    Returns:
        Decimal cost in USD, or None if the model is not in the pricing table.

    Note:
        cached_tokens subtraction happens inside this function to prevent
        call-site double-charging.  Callers pass the raw API fields directly.
    """
    rates = _RATES.get(model)
    if rates is None:
        return None
    # Subtract cached from total INSIDE the function to prevent call-site double-charging.
    # Clamp to 0: if caller passes only cached_tokens without matching prompt_tokens,
    # fresh_input should not go negative.
    fresh_input = max(0, prompt_tokens - cached_tokens)
    cost = Decimal(fresh_input) * rates["input_per_m"] / _PER_M
    cost += Decimal(completion_tokens) * rates["output_per_m"] / _PER_M
    if cached_tokens:
        cost += Decimal(cached_tokens) * rates["cache_read_per_m"] / _PER_M
    return cost
