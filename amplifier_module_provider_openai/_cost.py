"""OpenAI pricing rates and cost computation.

Verification date: 2026-09-01
Source: https://developers.openai.com/api/docs/pricing

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
- Cache-write cost: most OpenAI models have none (writes are free, reads discounted).
  GPT-5.6 (Sol/Terra/Luna) is the exception -- it bills cache-WRITE tokens at 1.25x the
  input rate and reports them as usage.input_tokens_details.cache_write_tokens (Responses API)
  / usage.prompt_tokens_details.cache_write_tokens (Chat Completions). Rate entries that omit
  "cache_write_per_m" bill any write tokens as ordinary input (correct for pre-5.6 models,
  which never emit the field). Verified against live gpt-5.6-sol usage on 2026-07-14.
- cached_tokens / cache_write_tokens subtraction happens INSIDE compute_cost to prevent
  call-site double-charging.
- Snapshot aliasing: the Responses API echoes back a dated snapshot id in response.model
  (e.g. "gpt-5.5-2026-04-23") rather than the alias ("gpt-5.5"). _find_rates() strips the
  YYYY-MM-DD suffix and falls back to the family alias automatically — no duplicate entries
  needed. To pin a re-priced snapshot, add an explicit dated entry; exact match wins.
"""

from __future__ import annotations

import re
from decimal import Decimal

from ._capabilities import get_capabilities

# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------

_PER_M = Decimal("1_000_000")

# Long-context re-rating threshold. A request whose INPUT (prompt) token count
# exceeds the model's `long_context_pricing_threshold` bills the ENTIRE request --
# input, output, cached, AND cache-write tokens -- at the long-context rates in
# _LONG_RATES (whole-request re-rating, not marginal-on-the-overage). Boundary is
# strict: a request exactly AT the threshold is still short-context (OpenAI prices
# "<=272K" short, ">272K" long). Measured on input tokens only; output tokens do
# not count toward the threshold.
#
# The threshold is NOT redefined here -- it is read per-model from the single
# source of truth, ModelCapabilities.long_context_pricing_threshold in
# _capabilities.py (272_000 for gpt-5.6; None for models with no split, which
# therefore never re-rate). Source, verification date, and rationale live there.

# Matches OpenAI dated-snapshot suffix: "<family>-YYYY-MM-DD".
# Used by _find_rates() to fall back from a snapshot id to the family alias.
_SNAPSHOT_RE = re.compile(r"^(?P<base>.+)-\d{4}-\d{2}-\d{2}$")

# _RATES maps model-id → {
#   "input_per_m":       Decimal,  # fresh input tokens, per 1M
#   "output_per_m":      Decimal,  # output/completion tokens, per 1M
#   "cache_read_per_m":  Decimal,  # cached input tokens, per 1M (0.00 = no discount)
#   "cache_write_per_m": Decimal,  # OPTIONAL. cache-WRITE tokens, per 1M (GPT-5.6 only).
#                                  # Omit for models with no distinct write price -- write
#                                  # tokens are then billed as ordinary input.
# }
#
# Rates are in USD.
# Unknown models → return None (DO NOT default to $0.00).
# Dated snapshots are handled by _find_rates() — no duplicate entries needed here.
#
# TODO: gpt-5.3-codex, gpt-5.2, gpt-5.2-pro, gpt-5.1, gpt-5.1-codex, gpt-5-mini
#       not yet on pricing page; these models return None until rates are added.
_RATES: dict[str, dict[str, Decimal]] = {
    # ------------------------------------------------------------------
    # GPT 6 Astra  (rolling out 2026-09-03)
    # Standard-tier SHORT-context rates (≤272,000 input tokens):
    #   input $10.00 / cached $1.00 / cache-write $12.50 / output $50.00 per 1M
    # Source: https://developers.openai.com/api/docs/pricing (verified 2026-09-03)
    # Source: https://developers.openai.com/api/docs/models/gpt-6-astra.md
    # Batch/Flex/Fast and regional uplifts are NOT represented here.
    # Long-context (>272K input) rates are in _LONG_RATES below.
    # ------------------------------------------------------------------
    "gpt-6-astra": {
        "input_per_m": Decimal("10.00"),
        "output_per_m": Decimal("50.00"),
        "cache_read_per_m": Decimal("1.00"),
        "cache_write_per_m": Decimal("12.50"),
    },
    # ------------------------------------------------------------------
    # GPT 5.6 family: Sol / Terra / Luna  (GA 2026-07-09)
    # Sol $4/$20, Terra $2/$12, Luna $0.20/$1.20 per 1M
    # (input/output, short-context, Standard tier).
    # cache_read = 0.1x input; cache_write = 1.25x input (GPT-5.6 bills writes).
    # Field verified live: usage.input_tokens_details.cache_write_tokens (Responses API).
    # Sol pricing is promotional through at least 2026-11-21; recheck after that date.
    # Alias "gpt-5.6" -> API echoes "gpt-5.6-sol" in response.model, so keying the
    # canonical tier ids is sufficient; no bare-alias entry required for cost.
    # The rates below are SHORT-context (<=272K input tokens). Requests whose input
    # exceeds the model's ModelCapabilities.long_context_pricing_threshold re-rate the
    # whole request at _LONG_RATES (see compute_cost).
    # ------------------------------------------------------------------
    "gpt-5.6-sol": {
        "input_per_m": Decimal("4.00"),
        "output_per_m": Decimal("20.00"),
        "cache_read_per_m": Decimal("0.40"),
        "cache_write_per_m": Decimal("5.00"),
    },
    "gpt-5.6-terra": {
        "input_per_m": Decimal("2.00"),
        "output_per_m": Decimal("12.00"),
        "cache_read_per_m": Decimal("0.20"),
        "cache_write_per_m": Decimal("2.50"),
    },
    "gpt-5.6-luna": {
        "input_per_m": Decimal("0.20"),
        "output_per_m": Decimal("1.20"),
        "cache_read_per_m": Decimal("0.02"),
        "cache_write_per_m": Decimal("0.25"),
    },
    # ------------------------------------------------------------------
    # GPT 5.5 (DEFAULT)  ($5.00 / $30.00, cache_read $0.50)
    # ------------------------------------------------------------------
    "gpt-5.5": {
        "input_per_m": Decimal("5.00"),
        "output_per_m": Decimal("30.00"),
        "cache_read_per_m": Decimal("0.50"),
    },
    # ------------------------------------------------------------------
    # GPT 5.5 Pro  ($30.00 / $180.00)
    # Pro models do not support prompt caching — API never returns
    # cached tokens.  cache_read_per_m is 0.00 (dead rate, never applied).
    # Source: https://developers.openai.com/api/docs/models/gpt-5.5-pro
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
    # GPT 5.4 Pro  ($30.00 / $180.00)
    # Pro models do not support prompt caching — API never returns
    # cached tokens.  cache_read_per_m is 0.00 (dead rate, never applied).
    # Source: https://developers.openai.com/api/docs/models/gpt-5.4-pro
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
    # ------------------------------------------------------------------
    # o4-mini Deep Research  ($2.00 / $8.00, cache_read $0.275)
    # ------------------------------------------------------------------
    "o4-mini-deep-research": {
        "input_per_m": Decimal("2.00"),
        "output_per_m": Decimal("8.00"),
        "cache_read_per_m": Decimal("0.275"),
    },
}


# _LONG_RATES: long-context rates, applied to the WHOLE request when input
# tokens exceed the model's ModelCapabilities.long_context_pricing_threshold.
# Same four keys as _RATES entries.
#
# GPT-6 Astra long-context (>272K input) Standard-tier rates:
#   input $20.00 / cached $2.00 / cache-write $25.00 / output $75.00 per 1M
#   (2x input/cached/write, 1.5x output vs short-context)
#   Source: https://developers.openai.com/api/docs/pricing (verified 2026-09-03)
#
# GPT-5.6 long-context rates:
# Relative to short-context: input / cached / cache-write are 2x, output is 1.5x.
# Absolute rates read directly off the pricing page (not derived from multipliers):
#   sol   input $8.00  cached $0.80  cache-write $10.00  output $30.00
#   terra input $4.00  cached $0.40  cache-write  $5.00  output $18.00
#   luna  input $0.40  cached $0.04  cache-write  $0.50  output  $1.80
# Source: https://developers.openai.com/api/docs/pricing (verified 2026-09-01).
_LONG_RATES: dict[str, dict[str, Decimal]] = {
    # GPT-6 Astra long-context (>272K input tokens) Standard-tier rates.
    # Source: https://developers.openai.com/api/docs/pricing (verified 2026-09-03)
    "gpt-6-astra": {
        "input_per_m": Decimal("20.00"),
        "output_per_m": Decimal("75.00"),
        "cache_read_per_m": Decimal("2.00"),
        "cache_write_per_m": Decimal("25.00"),
    },
    "gpt-5.6-sol": {
        "input_per_m": Decimal("8.00"),
        "output_per_m": Decimal("30.00"),
        "cache_read_per_m": Decimal("0.80"),
        "cache_write_per_m": Decimal("10.00"),
    },
    "gpt-5.6-terra": {
        "input_per_m": Decimal("4.00"),
        "output_per_m": Decimal("18.00"),
        "cache_read_per_m": Decimal("0.40"),
        "cache_write_per_m": Decimal("5.00"),
    },
    "gpt-5.6-luna": {
        "input_per_m": Decimal("0.40"),
        "output_per_m": Decimal("1.80"),
        "cache_read_per_m": Decimal("0.04"),
        "cache_write_per_m": Decimal("0.50"),
    },
}


def _find_rates(
    model: str, table: dict[str, dict[str, Decimal]] | None = None
) -> dict[str, Decimal] | None:
    """Look up pricing rates, falling back from snapshot id to family alias.

    The OpenAI Responses API echoes back the dated snapshot id in response.model
    (e.g. 'gpt-5.5-2026-04-23'), not the alias the caller configured ('gpt-5.5').
    We do a two-level lookup:

      1. Exact match — lets an individual snapshot be listed explicitly in the table
         if OpenAI ever re-prices it differently from the family.
      2. Strip the YYYY-MM-DD suffix and retry against the family alias.

    Args:
        model: model id or dated snapshot id.
        table: rate table to search. Defaults to _RATES (short-context). Pass
            _LONG_RATES to resolve long-context rates for the same model.

    Returns None (not a fabricated $0.00) when neither resolves.
    """
    if table is None:
        table = _RATES
    rates = table.get(model)
    if rates is not None:
        return rates
    m = _SNAPSHOT_RE.match(model)
    if m is None:
        return None
    return table.get(m.group("base"))


def compute_cost(
    model: str,
    *,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    cached_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> Decimal | None:
    """Compute the cost of an OpenAI API call in USD.

    Args:
        model: The model ID (e.g. 'gpt-5.4') or dated snapshot id
            (e.g. 'gpt-5.4-2026-03-05') as returned by the Responses API.
        prompt_tokens: Total prompt tokens (TOTAL, includes cached AND cache-write).
            This is response.usage.prompt_tokens (Chat) / usage.input_tokens (Responses).
        completion_tokens: Completion tokens used.
        cached_tokens: Number of prompt tokens served from cache (billed at
            cache_read_per_m). usage.{prompt,input}_tokens_details.cached_tokens.
        cache_write_tokens: Number of prompt tokens written to cache this call.
            GPT-5.6 only; billed at cache_write_per_m (1.25x input) when the model
            has that rate. usage.{prompt,input}_tokens_details.cache_write_tokens.
            Models without a cache_write_per_m rate never emit this field and bill
            it as ordinary input.

    Returns:
        Decimal cost in USD, or None if the model is not in the pricing table.

    Note:
        cached_tokens / cache_write_tokens subtraction happens inside this function
        to prevent call-site double-charging. Callers pass the raw API fields directly.
    """
    rates = _find_rates(model)
    if rates is None:
        return None

    # Long-context re-rating: when input tokens exceed the model's long-context
    # pricing threshold, the ENTIRE request (input, output, cached, cache-write)
    # bills at the long-context rates. Two independent gates:
    #   1. threshold -- read per-model from the single source of truth,
    #      ModelCapabilities.long_context_pricing_threshold (272_000 for gpt-5.6,
    #      None for models with no documented split, which therefore never re-rate).
    #   2. _LONG_RATES -- only GPT-5.6 tiers have long rates modelled, so a model
    #      with a threshold but no long rates (e.g. gpt-5.4) keeps its single
    #      short-context rate set unchanged.
    threshold = get_capabilities(model).long_context_pricing_threshold
    if threshold is not None and prompt_tokens > threshold:
        long_rates = _find_rates(model, _LONG_RATES)
        if long_rates is not None:
            rates = long_rates

    cache_write_rate = rates.get("cache_write_per_m")
    if cache_write_rate is None:
        # Model has no distinct cache-write price: any write tokens are ordinary
        # input (pre-5.6 models never emit the field, so this is also the historical
        # path, byte-for-byte unchanged).
        fresh_input = max(0, prompt_tokens - cached_tokens)
        write_cost = Decimal(0)
    else:
        # GPT-5.6: cache-write tokens are a re-rated subset of prompt_tokens
        # (billed at 1.25x input INSTEAD of the input rate, not on top of it).
        fresh_input = max(0, prompt_tokens - cached_tokens - cache_write_tokens)
        write_cost = Decimal(cache_write_tokens) * cache_write_rate / _PER_M

    cost = Decimal(fresh_input) * rates["input_per_m"] / _PER_M
    cost += write_cost
    cost += Decimal(completion_tokens) * rates["output_per_m"] / _PER_M
    if cached_tokens:
        cost += Decimal(cached_tokens) * rates["cache_read_per_m"] / _PER_M
    return cost
