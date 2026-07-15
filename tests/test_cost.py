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

from amplifier_module_provider_openai._cost import _RATES, _find_rates, compute_cost


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


# ---------------------------------------------------------------------------
# (k) o3-deep-research pricing: $10/$40/$5.00  (corrected from $2/$8/$0.50)
# ---------------------------------------------------------------------------
def test_o3_deep_research_input_cost():
    """o3-deep-research: 1M fresh input -> $10.00."""
    result = compute_cost("o3-deep-research", prompt_tokens=1_000_000)
    assert result == Decimal("10.00"), f"Expected Decimal('10.00'), got {result!r}"


def test_o3_deep_research_output_cost():
    """o3-deep-research: 1M output -> $40.00."""
    result = compute_cost("o3-deep-research", completion_tokens=1_000_000)
    assert result == Decimal("40.00"), f"Expected Decimal('40.00'), got {result!r}"


def test_o3_deep_research_cache_read_cost():
    """o3-deep-research: 1M cached -> $5.00 (50% of $10 input, matching o3 pattern)."""
    result = compute_cost("o3-deep-research", cached_tokens=1_000_000)
    assert result == Decimal("5.00"), f"Expected Decimal('5.00'), got {result!r}"


def test_o3_deep_research_dated_alias():
    """o3-deep-research-2025-06-26: same rates as o3-deep-research."""
    assert compute_cost(
        "o3-deep-research-2025-06-26", prompt_tokens=1_000_000
    ) == Decimal("10.00")
    assert compute_cost(
        "o3-deep-research-2025-06-26", completion_tokens=1_000_000
    ) == Decimal("40.00")
    assert compute_cost(
        "o3-deep-research-2025-06-26", cached_tokens=1_000_000
    ) == Decimal("5.00")


# ---------------------------------------------------------------------------
# (l) o4-mini-deep-research stays at $2/$8/$0.275  (correct, unchanged)
# ---------------------------------------------------------------------------
def test_o4_mini_deep_research_unchanged():
    """o4-mini-deep-research must remain at $2/$8/$0.275 -- NOT changed by this fix."""
    assert compute_cost("o4-mini-deep-research", prompt_tokens=1_000_000) == Decimal(
        "2.00"
    )
    assert compute_cost(
        "o4-mini-deep-research", completion_tokens=1_000_000
    ) == Decimal("8.00")
    assert compute_cost("o4-mini-deep-research", cached_tokens=1_000_000) == Decimal(
        "0.275"
    )


# ---------------------------------------------------------------------------
# (p) _find_rates: snapshot fallback — API echoes snapshot ID, not alias
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "snapshot,expected_input,expected_output,expected_cache",
    [
        ("gpt-5.5-2026-04-23", Decimal("5.00"), Decimal("30.00"), Decimal("0.50")),
        (
            "gpt-5.5-pro-2026-04-23",
            Decimal("30.00"),
            Decimal("180.00"),
            Decimal("0.00"),
        ),
        ("gpt-5.4-2026-03-05", Decimal("2.50"), Decimal("15.00"), Decimal("0.25")),
        (
            "gpt-5.4-pro-2026-03-05",
            Decimal("30.00"),
            Decimal("180.00"),
            Decimal("0.00"),
        ),
        (
            "o3-deep-research-2025-06-26",
            Decimal("10.00"),
            Decimal("40.00"),
            Decimal("5.00"),
        ),
        (
            "o4-mini-deep-research-2025-06-26",
            Decimal("2.00"),
            Decimal("8.00"),
            Decimal("0.275"),
        ),
    ],
)
def test_dated_snapshot_resolves_via_fallback(
    snapshot, expected_input, expected_output, expected_cache
):
    """Dated snapshots (what the API echoes back) resolve to the family alias rates."""
    assert _find_rates(snapshot) is not None, f"{snapshot} should resolve via fallback"
    assert compute_cost(snapshot, prompt_tokens=1_000_000) == expected_input
    assert compute_cost(snapshot, completion_tokens=1_000_000) == expected_output
    assert compute_cost(snapshot, cached_tokens=1_000_000) == expected_cache


def test_unknown_family_snapshot_returns_none():
    """A snapshot whose family alias is not in _RATES must return None, not $0."""
    assert _find_rates("gpt-6-2026-08-01") is None
    assert compute_cost("gpt-6-2026-08-01", prompt_tokens=1_000_000) is None


@pytest.mark.parametrize(
    "garbage",
    [
        "",
        "not-a-model",
        "gpt-5.5-",
        "gpt-5.5-2026",
        "gpt-5.5-2026-04",
        "gpt-5.5-2026-04-23-preview",
    ],
)
def test_non_snapshot_garbage_returns_none(garbage):
    """Strings that don't match the YYYY-MM-DD suffix pattern return None."""
    assert _find_rates(garbage) is None


def test_exact_match_wins_over_family_fallback():
    """An explicit dated entry in _RATES overrides the family alias (exact-match-wins)."""
    sentinel = {
        "input_per_m": Decimal("999.00"),
        "output_per_m": Decimal("999.00"),
        "cache_read_per_m": Decimal("999.00"),
    }
    _RATES["gpt-5.5-2099-01-01"] = sentinel
    try:
        assert _find_rates("gpt-5.5-2099-01-01") is sentinel
        assert compute_cost("gpt-5.5-2099-01-01", prompt_tokens=1_000_000) == Decimal(
            "999.00"
        )
    finally:
        del _RATES["gpt-5.5-2099-01-01"]


# ---------------------------------------------------------------------------
# (m) Dated snapshot aliases — API returns snapshot ID, not alias
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "snapshot,input_cost,output_cost,cache_cost",
    [
        ("gpt-5.5-2026-04-23", Decimal("5.00"), Decimal("30.00"), Decimal("0.50")),
        (
            "gpt-5.5-pro-2026-04-23",
            Decimal("30.00"),
            Decimal("180.00"),
            Decimal("0.00"),
        ),
        ("gpt-5.4-2026-03-05", Decimal("2.50"), Decimal("15.00"), Decimal("0.25")),
        (
            "gpt-5.4-pro-2026-03-05",
            Decimal("30.00"),
            Decimal("180.00"),
            Decimal("0.00"),
        ),
    ],
)
def test_dated_snapshot_matches_alias_pricing(
    snapshot, input_cost, output_cost, cache_cost
):
    """Dated snapshot IDs (what the API echoes back) must resolve to the same rates as alias."""
    assert compute_cost(snapshot, prompt_tokens=1_000_000) == input_cost, (
        f"{snapshot} input"
    )
    assert compute_cost(snapshot, completion_tokens=1_000_000) == output_cost, (
        f"{snapshot} output"
    )
    assert compute_cost(snapshot, cached_tokens=1_000_000) == cache_cost, (
        f"{snapshot} cache"
    )


# ---------------------------------------------------------------------------
# (q) GPT-5.6 (Sol / Terra / Luna): base pricing + cache-WRITE billing (1.25x)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "model,inp,out,cread,cwrite,l_inp,l_out,l_cread,l_cwrite",
    [
        (
            "gpt-5.6-sol",
            Decimal("5.00"),
            Decimal("30.00"),
            Decimal("0.50"),
            Decimal("6.25"),
            Decimal("10.00"),
            Decimal("45.00"),
            Decimal("1.00"),
            Decimal("12.50"),
        ),
        (
            "gpt-5.6-terra",
            Decimal("2.50"),
            Decimal("15.00"),
            Decimal("0.25"),
            Decimal("3.125"),
            Decimal("5.00"),
            Decimal("22.50"),
            Decimal("0.50"),
            Decimal("6.25"),
        ),
        (
            "gpt-5.6-luna",
            Decimal("1.00"),
            Decimal("6.00"),
            Decimal("0.10"),
            Decimal("1.25"),
            Decimal("2.00"),
            Decimal("9.00"),
            Decimal("0.20"),
            Decimal("2.50"),
        ),
    ],
)
def test_gpt_56_family_rates(
    model, inp, out, cread, cwrite, l_inp, l_out, l_cread, l_cwrite
):
    """Each GPT-5.6 tier prices the four token classes at its SHORT rate for
    requests with input <=272K, and re-rates the WHOLE request at its LONG rate
    when input exceeds the 272K threshold (see _LONG_RATES / compute_cost).

    Short-context assertions use a 200K count (<=272K, so no re-rate) scaled from
    the per-1M rate; long-context assertions use counts >272K to trigger the split.
    """
    _M = Decimal(1_000_000)

    # --- short context: input at/below the 272K threshold keeps the standard rate ---
    assert compute_cost(model, prompt_tokens=200_000) == Decimal(200_000) * inp / _M
    # output / cache-read carry 0 prompt tokens -> below threshold -> short rate.
    assert compute_cost(model, completion_tokens=1_000_000) == out
    assert compute_cost(model, cached_tokens=1_000_000) == cread
    assert (
        compute_cost(model, prompt_tokens=200_000, cache_write_tokens=200_000)
        == Decimal(200_000) * cwrite / _M
    )

    # --- long context: input above 272K re-rates the entire request ---
    # 1M fresh input -> long input rate.
    assert compute_cost(model, prompt_tokens=1_000_000) == l_inp
    # 300K fresh input (>272K, so long) + 1M output at the long output rate.
    assert (
        compute_cost(model, prompt_tokens=300_000, completion_tokens=1_000_000)
        == Decimal(300_000) * l_inp / _M + l_out
    )
    # 1M all-cached input (>272K) -> long cache-read rate on the whole request.
    assert (
        compute_cost(model, prompt_tokens=1_000_000, cached_tokens=1_000_000) == l_cread
    )
    # 1M all cache-write input (>272K) -> long cache-write rate.
    assert (
        compute_cost(model, prompt_tokens=1_000_000, cache_write_tokens=1_000_000)
        == l_cwrite
    )


def test_gpt_56_no_triple_charge_mixed_turn():
    """A turn mixing fresh + cached + written tokens charges each bucket once (Sol).

    fresh = 1000 - 200 cached - 300 written = 500 @ $5/M   = 0.0025
    write = 300 @ $6.25/M                                  = 0.001875
    read  = 200 @ $0.50/M                                  = 0.0001
    out   = 100 @ $30/M                                    = 0.003
    total                                                  = 0.007475
    """
    result = compute_cost(
        "gpt-5.6-sol",
        prompt_tokens=1_000,
        completion_tokens=100,
        cached_tokens=200,
        cache_write_tokens=300,
    )
    assert result == Decimal("0.007475"), f"got {result!r}"


def test_gpt_56_golden_from_real_usage():
    """Reality-derived golden: exact token counts captured from live gpt-5.6-sol.

    Source: two real Responses-API calls with an identical >1024-token prefix
    (see projects/gpt-5-6-support/ground-truth/usage-real.json). These are the
    numbers the API actually returned -- not synthetic.

    WRITE call: input=2655, cache_write=2652, cached=0, output=5
        fresh=3 @5 + 2652 @6.25 + 5 @30 = 0.000015 + 0.016575 + 0.000150 = 0.016740
    READ  call: input=2655, cache_write=0, cached=2652, output=5
        fresh=3 @5 + 2652 @0.50 + 5 @30 = 0.000015 + 0.001326 + 0.000150 = 0.001491
    """
    write = compute_cost(
        "gpt-5.6-sol",
        prompt_tokens=2655,
        completion_tokens=5,
        cached_tokens=0,
        cache_write_tokens=2652,
    )
    read = compute_cost(
        "gpt-5.6-sol",
        prompt_tokens=2655,
        completion_tokens=5,
        cached_tokens=2652,
        cache_write_tokens=0,
    )
    assert write == Decimal("0.016740"), f"WRITE got {write!r}"
    assert read == Decimal("0.001491"), f"READ got {read!r}"
    # The write turn MUST cost more than billing those tokens as plain input --
    # this is the 1.25x premium, and guards against silent regression to $5 input.
    plain_input = compute_cost(
        "gpt-5.6-sol", prompt_tokens=2655, completion_tokens=5, cached_tokens=0
    )
    assert write is not None and plain_input is not None
    assert write > plain_input


def test_gpt_56_alias_snapshot_resolves():
    """A dated gpt-5.6-sol snapshot resolves to the tier rates via the fallback --
    through BOTH the short table (_RATES) and the long table (_LONG_RATES)."""
    # short context: 200K input (<=272K) -> sol short input $5/M -> $1.00.
    assert compute_cost("gpt-5.6-sol-2026-07-09", prompt_tokens=200_000) == Decimal(
        "1.00"
    )
    # long context: 1M input (>272K) -> snapshot must also resolve through _LONG_RATES
    # to sol's long input rate $10/M -> $10.00.
    assert compute_cost("gpt-5.6-sol-2026-07-09", prompt_tokens=1_000_000) == Decimal(
        "10.00"
    )


def test_long_context_rerate_requires_both_gates():
    """Long-context re-rating fires only when the model has BOTH a threshold AND
    modelled long rates. The two gates are independent:

    - gpt-5.4: has a 272K threshold but NO _LONG_RATES entry -> never re-rates,
      keeps its single short rate even at 1M input.
    - gpt-5.5: threshold is None -> never re-rates regardless of input size.
    """
    # gpt-5.4 short input $2.50/M applies even at 1M input (no long rates modelled).
    assert compute_cost("gpt-5.4", prompt_tokens=1_000_000) == Decimal("2.50")
    # gpt-5.5 short input $5.00/M applies even at 1M input (threshold is None).
    assert compute_cost("gpt-5.5", prompt_tokens=1_000_000) == Decimal("5.00")


def test_cache_write_ignored_for_models_without_write_rate():
    """Pre-5.6 models have no cache_write_per_m: any cache_write_tokens are billed
    as ordinary input (historical path unchanged, no crash)."""
    # gpt-5.5 has no cache_write rate; passing cache_write_tokens must not error and
    # bills the full prompt as input (fresh = prompt - cached, write ignored).
    result = compute_cost("gpt-5.5", prompt_tokens=1_000, cache_write_tokens=300)
    assert result == Decimal("1000") * Decimal("5") / Decimal("1000000")
