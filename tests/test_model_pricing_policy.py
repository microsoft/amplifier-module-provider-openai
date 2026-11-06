"""Independent Standard-pricing oracles, verified 2026-11-06.

Sources:
https://developers.openai.com/api/docs/pricing.md
https://developers.openai.com/api/docs/guides/prompt-caching.md
https://developers.openai.com/api/docs/models/gpt-5.6-sol.md (bare alias)
https://developers.openai.com/api/docs/models/gpt-5.1-codex.md (retired pricing)

Expected values are literal source rates or independently calculated totals,
never generated from production rates or model-family multipliers.
"""

from decimal import Decimal

import pytest

from amplifier_module_provider_openai._cost import _LONG_RATES, _RATES, compute_cost

MODERN_MODELS = (
    "gpt-5.6", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna",
    "gpt-6-astra", "gpt-6-sol", "gpt-6-luna", "gpt-6.1-sol", "gpt-6.1-terra",
)


@pytest.mark.parametrize("model,short,long", [
    ("gpt-5.6", ("4", "20", ".4", "5", "10"), ("8", "30", ".8", "10", "20")),
    ("gpt-5.6-sol", ("4", "20", ".4", "5", "10"), ("8", "30", ".8", "10", "20")),
    ("gpt-5.6-terra", ("2", "12", ".2", "2.5", "5"), ("4", "18", ".4", "5", "10")),
    ("gpt-5.6-luna", (".2", "1.2", ".02", ".25", ".5"), (".4", "1.8", ".04", ".5", "1")),
    ("gpt-6-astra", ("10", "50", "1", "12.5", "25"), ("20", "75", "2", "25", "50")),
    ("gpt-6-sol", ("2", "10", ".2", "2.5", "5"), ("4", "15", ".4", "5", "10")),
    ("gpt-6-luna", (".1", ".5", ".01", ".125", ".25"), (".2", ".75", ".02", ".25", ".5")),
    ("gpt-6.1-sol", ("2", "10", ".1", "2.5", "5"), ("4", "15", ".2", "5", "10")),
    ("gpt-6.1-terra", ("1", "5", ".05", "1.25", "2.5"), ("2", "7.5", ".1", "2.5", "5")),
])
def test_exact_modern_source_rate_oracles(model, short, long):
    keys = (
        "input_per_m", "output_per_m", "cache_read_per_m",
        "cache_write_per_m", "cache_write_12h_per_m",
    )
    assert tuple(_RATES[model][key] for key in keys) == tuple(map(Decimal, short))
    assert tuple(_LONG_RATES[model][key] for key in keys) == tuple(map(Decimal, long))


@pytest.mark.parametrize("model,short,long", [
    ("gpt-5.6", "1.328", "2.456008"),
    ("gpt-5.6-sol", "1.328", "2.456008"),
    ("gpt-5.6-terra", ".704", "1.288004"),
    ("gpt-5.6-luna", ".0704", ".1288004"),
    ("gpt-6-astra", "3.32", "6.14002"),
    ("gpt-6-sol", ".664", "1.228004"),
    ("gpt-6-luna", ".0332", ".0614002"),
    ("gpt-6.1-sol", ".654", "1.208004"),
    ("gpt-6.1-terra", ".327", ".604002"),
])
def test_mixed_lifetime_whole_request_boundary_oracles(model, short, long):
    # At threshold: 122K fresh + 100K cached + 20K 30m + 30K 12h; 20K output.
    # Above threshold: one additional fresh token; ALL five classes re-rate.
    usage = {
        "completion_tokens": 20_000,
        "cached_tokens": 100_000,
        "cache_write_tokens": 50_000,
        "cache_write_tokens_details": {
            "ttl_30m_tokens": 20_000, "ttl_12h_tokens": 30_000,
        },
    }
    assert compute_cost(model, prompt_tokens=272_000, **usage) == Decimal(short)
    assert compute_cost(model, prompt_tokens=272_001, **usage) == Decimal(long)


@pytest.mark.parametrize("model,write_30m,write_12h", [
    ("gpt-5.6", ".06", ".12"),
    ("gpt-5.6-sol", ".06", ".12"),
    ("gpt-5.6-terra", ".03", ".06"),
    ("gpt-5.6-luna", ".003", ".006"),
    ("gpt-6-astra", ".15", ".3"),
    ("gpt-6-sol", ".03", ".06"),
    ("gpt-6-luna", ".0015", ".003"),
    ("gpt-6.1-sol", ".03", ".06"),
    ("gpt-6.1-terra", ".015", ".03"),
])
def test_single_lifetime_write_oracles(model, write_30m, write_12h):
    # Keep the existing Terra reader's single-bucket behavior: if the bucket
    # equals the measured aggregate, no writes remain for the omitted bucket.
    assert compute_cost(
        model, prompt_tokens=12_000, cache_write_tokens=12_000,
        cache_write_tokens_details={"ttl_30m_tokens": 12_000},
    ) == Decimal(write_30m)
    assert compute_cost(
        model, prompt_tokens=12_000, cache_write_tokens=12_000,
        cache_write_tokens_details={"ttl_12h_tokens": 12_000},
    ) == Decimal(write_12h)


@pytest.mark.parametrize("model,short,long", [
    ("gpt-5.4", ".98", "1.810005"),
    ("gpt-5.5", "1.96", "3.62001"),
    ("gpt-5.4-pro", "11.76", "21.72006"),
    ("gpt-5.4-2026-03-05", ".98", "1.810005"),
    ("gpt-5.5-2026-04-23", "1.96", "3.62001"),
])
def test_legacy_long_context_boundary_oracles(model, short, long):
    assert compute_cost(
        model, prompt_tokens=272_000, completion_tokens=20_000,
    ) == Decimal(short)
    assert compute_cost(
        model, prompt_tokens=272_001, completion_tokens=20_000,
    ) == Decimal(long)


@pytest.mark.parametrize("model,input_rate,output_rate,cached_rate", [
    ("gpt-5.4", "5", "22.5", ".5"),
    ("gpt-5.5", "10", "45", "1"),
    ("gpt-5.4-pro", "60", "270", "0"),
    ("gpt-5.5-pro", "60", "270", "0"),
])
def test_legacy_long_context_source_rates(model, input_rate, output_rate, cached_rate):
    assert _LONG_RATES[model] == {
        "input_per_m": Decimal(input_rate),
        "output_per_m": Decimal(output_rate),
        "cache_read_per_m": Decimal(cached_rate),
    }


@pytest.mark.parametrize("model,input_cost,cache_cost", [
    ("gpt-5.4", "5", ".5"),
    ("gpt-5.5", "10", "1"),
])
def test_long_input_and_cache_only_oracles(model, input_cost, cache_cost):
    assert compute_cost(model, prompt_tokens=1_000_000) == Decimal(input_cost)
    assert compute_cost(
        model, prompt_tokens=1_000_000, cached_tokens=1_000_000,
    ) == Decimal(cache_cost)


@pytest.mark.parametrize("model,input_rate,output_rate,cached_rate", [
    ("gpt-5.4-mini", ".75", "4.5", ".075"),
    ("gpt-5.4-nano", ".2", "1.25", ".02"),
    ("gpt-5.3-codex", "1.75", "14", ".175"),
    ("gpt-5.2", "1.75", "14", ".175"),
    ("gpt-5.2-pro", "21", "168", "0"),
    ("gpt-5.1", "1.25", "10", ".125"),
    ("gpt-5.1-codex", "1.25", "10", ".125"),
    ("gpt-5", "1.25", "10", ".125"),
    ("gpt-5-mini", ".25", "2", ".025"),
    ("gpt-5-nano", ".05", ".4", ".005"),
    ("gpt-5-pro", "15", "120", "0"),
    ("gpt-4.1", "2", "8", ".5"),
    ("gpt-4.1-mini", ".4", "1.6", ".1"),
    ("gpt-4o", "2.5", "10", "1.25"),
    ("gpt-4o-mini", ".15", ".6", ".075"),
    ("o3-pro", "20", "80", "0"),
    ("o3", "2", "8", ".5"),
])
def test_documented_legacy_price_oracles(model, input_rate, output_rate, cached_rate):
    assert compute_cost(model, prompt_tokens=1_000_000) == Decimal(input_rate)
    assert compute_cost(model, completion_tokens=1_000_000) == Decimal(output_rate)
    assert compute_cost(
        model, prompt_tokens=1_000_000, cached_tokens=1_000_000,
    ) == Decimal(cached_rate)


@pytest.mark.parametrize("model", MODERN_MODELS)
@pytest.mark.parametrize("details", [
    None, {}, [], "30m",
    {"ttl_30m_tokens": 49, "ttl_12h_tokens": 50},
    {"ttl_30m_tokens": 101, "ttl_12h_tokens": 0},
    {"ttl_30m_tokens": -1, "ttl_12h_tokens": 101},
    {"ttl_30m_tokens": True, "ttl_12h_tokens": 99},
    {"ttl_30m_tokens": 50.0, "ttl_12h_tokens": 50},
    {"ttl_30m_tokens": "50", "ttl_12h_tokens": 50},
    {"ttl_30m_tokens": None, "ttl_12h_tokens": 100},
    {"ttl_30m_tokens": 100, "ttl_24h_tokens": 0},
])
def test_missing_or_invalid_lifetime_evidence_stays_unknown(model, details):
    assert compute_cost(
        model, prompt_tokens=200, cache_write_tokens=100,
        cache_write_tokens_details=details,
    ) is None


@pytest.mark.parametrize("model", MODERN_MODELS)
@pytest.mark.parametrize("field", [
    "prompt_tokens", "completion_tokens", "cached_tokens", "cache_write_tokens",
])
@pytest.mark.parametrize("invalid", [None, -1, True, 1.5, "1"])
def test_invalid_modern_counts_stay_unknown(model, field, invalid):
    counts = {
        "prompt_tokens": 200, "completion_tokens": 20,
        "cached_tokens": 0, "cache_write_tokens": 0,
    }
    counts[field] = invalid
    assert compute_cost(model, **counts) is None


@pytest.mark.parametrize("model", MODERN_MODELS)
def test_overlapping_counts_stay_unknown(model):
    assert compute_cost(model, prompt_tokens=100, cached_tokens=101) is None
    assert compute_cost(
        model, prompt_tokens=100, cached_tokens=50, cache_write_tokens=51,
        cache_write_tokens_details={"ttl_30m_tokens": 51, "ttl_12h_tokens": 0},
    ) is None


@pytest.mark.parametrize("model", MODERN_MODELS)
def test_zero_is_known_but_contradictory_zero_writes_are_unknown(model):
    assert compute_cost(model) == Decimal(0)
    assert compute_cost(
        model, cache_write_tokens_details={"ttl_30m_tokens": 0, "ttl_12h_tokens": 0},
    ) == Decimal(0)
    assert compute_cost(
        model, cache_write_tokens_details={"ttl_30m_tokens": 0, "ttl_12h_tokens": 1},
    ) is None


@pytest.mark.parametrize("tier,short,long", [
    ("default", ".327", ".604002"),
    ("flex", ".1635", ".302001"),
    ("fast", ".654", "1.208004"),
    ("priority", ".654", "1.208004"),
])
def test_gpt_6_service_tier_oracles_remain_endpoint_agnostic(tier, short, long):
    usage = {
        "completion_tokens": 20_000, "cached_tokens": 100_000,
        "cache_write_tokens": 50_000,
        "cache_write_tokens_details": {
            "ttl_30m_tokens": 20_000, "ttl_12h_tokens": 30_000,
        },
        "service_tier": tier,
    }
    assert compute_cost("gpt-6.1-terra", prompt_tokens=272_000, **usage) == Decimal(short)
    assert compute_cost("gpt-6.1-terra", prompt_tokens=272_001, **usage) == Decimal(long)


@pytest.mark.parametrize("tier", [None, "", "auto", "ultrafast", "batch", 1])
def test_unpriced_or_missing_gpt_6_tier_is_unknown(tier):
    assert compute_cost("gpt-6-astra", prompt_tokens=100, service_tier=tier) is None


def test_historical_readers_and_legacy_tiers_are_preserved():
    # No lifetime buckets are needed for a read with an explicit zero write count.
    assert compute_cost(
        "gpt-5.6-sol", prompt_tokens=2655, completion_tokens=5, cached_tokens=2652,
    ) == Decimal(".0011728")
    assert compute_cost(
        "gpt-5.6-sol-2026-07-09", prompt_tokens=200_000,
    ) == Decimal(".8")
    # Pre-5.6 write counts remain ordinary input; legacy tiers remain unchanged.
    assert compute_cost(
        "gpt-5.5", prompt_tokens=1000, cache_write_tokens=300,
        cache_write_tokens_details={"unrecognized": 300}, service_tier="fast",
    ) == Decimal(".005")
    # Historical cached-only calls still use the legacy clamping behavior.
    assert compute_cost("gpt-5.4", cached_tokens=1000) == Decimal(".00025")


@pytest.mark.parametrize("model", [
    "gpt-5.7", "gpt-6", "gpt-6.1", "gpt-6-terra", "gpt-6.1-luna",
    "gpt-6-astra-2026-11-06", "gpt-6.1-sol-preview", "unpriced-unknown",
])
def test_undocumented_ids_never_get_inferred_prices(model):
    assert compute_cost(model) is None
    assert compute_cost(model, prompt_tokens=1_000_000) is None