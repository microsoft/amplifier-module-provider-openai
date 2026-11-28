"""Tests for _cost.py: compute_cost() for provider-openai.

Covers:
  (a) Known model: correct Decimal cost for input tokens
  (b) Output tokens cost
  (c) REQUIRED: Cached request does NOT double-charge
  (d) Unknown model returns None
  (e) None != Decimal('0')
  (f) Result type is always Decimal, never float
  (g) Cache-only (fresh_input=0)
  (h) Documented legacy pricing
  (i) gpt-5.5 pricing
  (j) gpt-5.5-pro: no cache discount
"""

from decimal import Decimal

import pytest

from amplifier_module_provider_openai._cost import _RATES, _find_rates, compute_cost

# Exact specialized rows, not base-model capability or price inheritance.
# pricing.md, Healthcare Standard/Fast, verified 2026-11-25; USD per 1M.
HEALTHCARE_PRICES = [
    ("gpt-6.2-astra-healthcare", ("8", "40", ".40", "10", "20"),
     ("16", "60", ".80", "20", "40"), ("16", "80", ".80"), ("32", "120", "1.60")),
    ("gpt-6.2-sol-healthcare", ("2", "10", ".10", "2.50", "5"),
     ("4", "15", ".20", "5", "10"), ("4", "20", ".20"), ("8", "30", ".40")),
    ("gpt-6-astra-healthcare", ("10", "50", "1"), ("20", "75", "2"),
     ("20", "100", "2"), ("40", "150", "4")),
    ("gpt-6.1-sol-healthcare", ("2", "10", ".10"), ("4", "15", ".20"),
     ("4", "20", ".20"), ("8", "30", ".40")),
]


@pytest.fixture(autouse=True)
def healthcare_verification_clock(request, monkeypatch):
    """Arithmetic fixtures stay at source verification time, not wall time."""
    if not request.node.name.startswith("test_healthcare"):
        return
    from datetime import datetime

    from amplifier_module_provider_openai import _cost

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 11, 25, tzinfo=tz)

    monkeypatch.setattr(_cost, "datetime", Clock)


@pytest.mark.parametrize("model,short,long,fast_short,fast_long", HEALTHCARE_PRICES)
@pytest.mark.parametrize("prompt", [271_999, 272_000, 272_001])
@pytest.mark.parametrize("tier", ["default", "fast", "priority"])
def test_healthcare_exact_rates(model, short, long, fast_short, fast_long, prompt, tier):
    rates = (short if prompt <= 272_000 else long) if tier == "default" else (
        fast_short if prompt <= 272_000 else fast_long
    )
    inp, out, read = map(Decimal, rates[:3])
    assert compute_cost(
        model, prompt_tokens=prompt, completion_tokens=1_000_000,
        cached_tokens=100, cache_write_tokens=0, service_tier=tier,
    ) == (Decimal(prompt - 100) * inp + 100 * read + 1_000_000 * out) / 1_000_000


@pytest.mark.parametrize("model,short,long,fast_short,fast_long", HEALTHCARE_PRICES[:2])
@pytest.mark.parametrize("prompt", [1000, 272_000, 272_001])
@pytest.mark.parametrize("split", [
    {"ttl_30m_tokens": 300, "ttl_12h_tokens": 0},
    {"ttl_30m_tokens": 0, "ttl_12h_tokens": 300},
    {"ttl_30m_tokens": 100, "ttl_12h_tokens": 200},
])
def test_healthcare_standard_disjoint_writes(
    model, short, long, fast_short, fast_long, prompt, split,
):
    inp, out, read, write30, write12 = map(Decimal, short if prompt <= 272_000 else long)
    assert compute_cost(
        model, prompt_tokens=prompt, completion_tokens=50, cached_tokens=200,
        cache_write_tokens=300, cache_write_tokens_details=split,
    ) == (
        Decimal(prompt - 500) * inp + 50 * out + 200 * read
        + split["ttl_30m_tokens"] * write30 + split["ttl_12h_tokens"] * write12
    ) / 1_000_000


@pytest.mark.parametrize("model", [row[0] for row in HEALTHCARE_PRICES])
@pytest.mark.parametrize("evidence", [
    *({field: value} for field in (
        "prompt_tokens", "completion_tokens", "cached_tokens", "cache_write_tokens",
    ) for value in (None, False, -1, "0", 0.0)),
    *({"service_tier": tier} for tier in (
        None, False, "Default", "standard", "flex", "batch", "ultrafast", "unknown",
    )),
    {"cached_tokens": 1001}, {"cached_tokens": 900, "cache_write_tokens": 101},
    {"cache_write_tokens": 1},
    *({"cache_write_tokens_details": split} for split in (
        False, [], {"unknown": 0}, {"ttl_30m_tokens": False},
        {"ttl_12h_tokens": -1}, {"ttl_30m_tokens": 1},
    )),
])
def test_healthcare_uncertain_usage_unknown(model, evidence):
    counts = {"prompt_tokens": 1000, "completion_tokens": 10,
              "cached_tokens": 0, "cache_write_tokens": 0}
    counts.update(evidence)
    assert compute_cost(model, **counts) is None


@pytest.mark.parametrize("model", [row[0] for row in HEALTHCARE_PRICES])
@pytest.mark.parametrize("tier", ["default", "fast", "priority"])
def test_healthcare_unpublished_writes_unknown(model, tier):
    if model.startswith("gpt-6.2-") and tier == "default":
        return  # Published Standard writes are exercised above.
    assert compute_cost(
        model, prompt_tokens=1000, cache_write_tokens=1, service_tier=tier,
        cache_write_tokens_details={"ttl_30m_tokens": 1, "ttl_12h_tokens": 0},
    ) is None


@pytest.mark.parametrize("model", [row[0] for row in HEALTHCARE_PRICES])
@pytest.mark.parametrize("suffix", ["-2026-11-24", "-preview", " "])
def test_healthcare_no_invented_alias(model, suffix):
    assert compute_cost(model + suffix, prompt_tokens=1000) is None


@pytest.mark.parametrize("day,known", [
    ("2026-11-23", False), ("2026-11-25", True),
    ("2026-12-31", True), ("2027-01-01", False),
])
@pytest.mark.parametrize("model", [row[0] for row in HEALTHCARE_PRICES])
@pytest.mark.parametrize("tier", ["default", "fast"])
def test_healthcare_promotion_requires_reverification(monkeypatch, day, known, model, tier):
    from datetime import datetime

    from amplifier_module_provider_openai import _cost

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime.fromisoformat(day).replace(tzinfo=tz)

    monkeypatch.setattr(_cost, "datetime", Clock)
    result = compute_cost(model, prompt_tokens=1000, service_tier=tier)
    assert isinstance(result, Decimal) if known else result is None


@pytest.mark.parametrize("model", [row[0] for row in HEALTHCARE_PRICES])
@pytest.mark.parametrize("field", ["input_tokens", "output_tokens", "cached_tokens",
                                  "cache_write_tokens"])
@pytest.mark.parametrize("value", ["absent", None, False, -1, "0"])
def test_healthcare_provider_preserves_missing_usage(model, field, value):
    from types import SimpleNamespace

    from amplifier_module_provider_openai import OpenAIProvider

    usage = {"input_tokens": 1000, "output_tokens": 100}
    details = {"cached_tokens": 100, "cache_write_tokens": 0}
    target = usage if field in usage else details
    if value == "absent":
        target.pop(field)
    else:
        target[field] = value
    response = SimpleNamespace(
        model=model, service_tier="default",
        usage=SimpleNamespace(**usage, input_tokens_details=SimpleNamespace(**details)),
    )
    provider = OpenAIProvider(config={"api_key": "offline", "default_model": "gpt-5.5"})
    assert provider._compute_attempt_cost(response) is None


@pytest.mark.parametrize("endpoint", [
    None, "https://api.openai.com/v1", "https://eu.api.openai.com/v1",
    "https://gateway.example/v1", "https://example.openai.azure.com",
])
def test_healthcare_provider_global_endpoint_only(endpoint):
    from types import SimpleNamespace

    from amplifier_module_provider_openai import OpenAIProvider

    provider = OpenAIProvider(config={
        "api_key": "offline", "base_url": endpoint, "default_model": "gpt-5.5",
    })
    response = SimpleNamespace(
        model="gpt-6.2-sol-healthcare", service_tier="default",
        usage=SimpleNamespace(
            input_tokens=1000, output_tokens=100,
            input_tokens_details=SimpleNamespace(cached_tokens=100, cache_write_tokens=0),
        ),
    )
    expected = Decimal(".00281") if endpoint in (None, "https://api.openai.com/v1") else None
    assert provider._compute_attempt_cost(response) == expected
    response.model = "unpublished-healthcare"
    assert provider._compute_attempt_cost(response) is None


@pytest.mark.parametrize("tier", ["default", "fast"])
@pytest.mark.parametrize("split", [
    {"ttl_30m_tokens": 300}, {"ttl_12h_tokens": 300},
    {"ttl_30m_tokens": 100, "ttl_12h_tokens": 199},
    {"ttl_30m_tokens": 300, "ttl_12h_tokens": None},
])
def test_healthcare_incomplete_lifetime_counts_unknown(tier, split):
    assert compute_cost(
        "gpt-6.2-sol-healthcare", prompt_tokens=1000,
        cache_write_tokens=300, cache_write_tokens_details=split, service_tier=tier,
    ) is None


@pytest.mark.parametrize("evidence", [
    {"background_options": {"completion_window": "2h"}},
    {"multi_agent": {"enabled": True}},
])
def test_healthcare_unpriced_modes_unknown(evidence):
    from types import SimpleNamespace

    from amplifier_module_provider_openai import (
        OpenAIProvider,
        _has_unpriced_gpt_6_2_mode,
    )

    response = SimpleNamespace(
        model="gpt-6.2-sol-healthcare", service_tier="default", **evidence,
        usage=SimpleNamespace(
            input_tokens=1000, output_tokens=100,
            input_tokens_details=SimpleNamespace(cached_tokens=100, cache_write_tokens=0),
        ),
    )
    assert _has_unpriced_gpt_6_2_mode(vars(response))
    provider = OpenAIProvider(config={"api_key": "offline"})
    assert provider._compute_attempt_cost(response) is None


@pytest.mark.parametrize("model", [row[0] for row in HEALTHCARE_PRICES])
def test_healthcare_sdk_reported_usage_not_requested_model(model):
    from openai.types.responses import Response

    from amplifier_module_provider_openai import OpenAIProvider
    from amplifier_module_provider_openai._capabilities import (
        GPT_6_MODELS,
        get_request_policy,
    )

    response = Response.model_construct(
        model=model, service_tier="default",
        usage={
            "input_tokens": 1000, "output_tokens": 100, "total_tokens": 1100,
            "input_tokens_details": {"cached_tokens": 100, "cache_write_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    )
    provider = OpenAIProvider(config={"api_key": "offline", "default_model": "unpriced"})
    expected = compute_cost(model, prompt_tokens=1000, completion_tokens=100, cached_tokens=100)
    assert expected is not None
    assert provider._compute_attempt_cost(response) == expected
    # Observed unknown ID must not reuse the configured selector.
    provider.default_model = model
    response.model = "unpublished-healthcare"
    assert provider._compute_attempt_cost(response) is None
    assert model not in GPT_6_MODELS
    assert get_request_policy(model).cache_ttls == frozenset()
    # An SDK-created default is not reported wire usage.
    response.model = model
    response.usage.input_tokens_details.model_fields_set.discard("cache_write_tokens")
    assert provider._compute_attempt_cost(response) is None


@pytest.mark.parametrize("split,expected", [
    ({"ttl_30m_tokens": 100, "ttl_12h_tokens": 200}, Decimal(".00327")),
    ({"ttl_30m_tokens": 300}, None),
    ({"ttl_30m_tokens": -1, "ttl_12h_tokens": 301}, None),
])
def test_healthcare_raw_response_cache_write_breakdown(split, expected):
    from amplifier_module_provider_openai import OpenAIProvider, _RawResponseObject

    response = _RawResponseObject({
        "model": "gpt-6.2-sol-healthcare", "service_tier": "default",
        "usage": {
            "input_tokens": 1000, "output_tokens": 100,
            "input_tokens_details": {
                "cached_tokens": 200, "cache_write_tokens": 300,
                "cache_write_tokens_details": split,
            },
        },
    })
    provider = OpenAIProvider(config={"api_key": "offline"})
    assert provider._compute_attempt_cost(response) == expected


@pytest.mark.parametrize("counts,expected", [
    ({"prompt_tokens": 1000}, "0.005"),
    ({"completion_tokens": 100}, "0.003"),
    ({"prompt_tokens": 1000, "cached_tokens": 1000}, "0.0005"),
    ({}, "0"),
    ({"prompt_tokens": 1000, "cached_tokens": 100, "completion_tokens": 100}, "0.00755"),
    ({"cache_write_tokens_details": {}}, "0"),
    ({"cache_write_tokens_details": {"ttl_30m_tokens": 0, "ttl_12h_tokens": 0}}, "0"),
])
def test_chat_latest_standard_exact_rates(counts, expected):
    result = compute_cost("chat-latest", **counts)
    assert isinstance(result, Decimal)
    assert result == Decimal(expected)


@pytest.mark.parametrize("field", [
    "prompt_tokens", "completion_tokens", "cached_tokens", "cache_write_tokens",
])
@pytest.mark.parametrize("value", [None, False, True, -1, 0.0, "0"])
def test_chat_latest_price_evidence_unknown(field, value):
    assert compute_cost("chat-latest", **{field: value}) is None


@pytest.mark.parametrize("evidence", [
    {"prompt_tokens": 1, "cached_tokens": 2},
    {"cache_write_tokens": 1},
    *({"service_tier": tier} for tier in (
        None, "flex", "priority", "fast", "ultrafast", "unknown", "Default", " default ", False,
    )),
    *({"cache_write_tokens_details": split} for split in (
        [], False, "", {"unknown": 0}, {"ttl_30m_tokens": 1},
        {"ttl_12h_tokens": -1}, {"ttl_30m_tokens": None},
        {"ttl_12h_tokens": False}, {"ttl_30m_tokens": "0"}, {"ttl_30m_tokens": 0.0},
    )),
])
def test_chat_latest_unsupported_price_evidence_unknown(evidence):
    assert compute_cost("chat-latest", **evidence) is None


@pytest.mark.parametrize("model", [
    "chat-latest-2099-01-01", "chat-latest-preview", "Chat-Latest", " chat-latest ",
])
def test_chat_latest_no_guessed_snapshot_rates(model):
    assert _find_rates(model) is None
    assert compute_cost(model, prompt_tokens=1000) is None


# ---------------------------------------------------------------------------
# (a) Known model: gpt-5.4 input cost
# ---------------------------------------------------------------------------
def test_known_model_input_cost():
    """gpt-5.4: 1M input crosses 272K and uses the $5 long-context rate."""
    result = compute_cost("gpt-5.4", prompt_tokens=1_000_000)
    assert result == Decimal("5.00"), f"Expected Decimal('5.00'), got {result!r}"


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
    """gpt-5.4: 1M prompt_tokens, 1M cached_tokens → $0.50 (long cache_read only).

    fresh_input = 1M - 1M = 0
    cost = 0 × $5.00/M + 0 × $22.50/M + 1M × $0.50/M = $0.50
    """
    result = compute_cost("gpt-5.4", prompt_tokens=1_000_000, cached_tokens=1_000_000)
    assert result == Decimal("0.50"), (
        f"Expected Decimal('0.50') (cache_read only, no double-charge), got {result!r}"
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
    assert result != Decimal(0)


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
    assert result == Decimal("0.50")


# ---------------------------------------------------------------------------
# (h) Documented prices replace the stale "not yet priced" TODO.
# https://developers.openai.com/api/docs/pricing.md (2026-11-06)
# https://developers.openai.com/api/docs/models/gpt-5.1-codex.md
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "model,expected",
    [
        ("gpt-5.3-codex", "1.75"),
        ("gpt-5.2", "1.75"),
        ("gpt-5.2-pro", "21.00"),
        ("gpt-5.1", "1.25"),
        ("gpt-5.1-codex", "1.25"),
        ("gpt-5-mini", "0.25"),
    ],
)
def test_previously_unpriced_models_use_documented_rates(model, expected):
    """Current source prices are known; this does not assert model availability."""
    result = compute_cost(model, prompt_tokens=1_000_000)
    assert result == Decimal(expected)


# ---------------------------------------------------------------------------
# (i) gpt-5.5 pricing: $5/$30/$0.50
# ---------------------------------------------------------------------------
def test_gpt_55_pricing():
    """1M input uses long pricing; output alone and cached-only legacy calls do not."""
    assert compute_cost("gpt-5.5", prompt_tokens=1_000_000) == Decimal("10.00")
    assert compute_cost("gpt-5.5", completion_tokens=1_000_000) == Decimal("30.00")
    assert compute_cost("gpt-5.5", cached_tokens=1_000_000) == Decimal("0.50")


# ---------------------------------------------------------------------------
# (j) gpt-5.5-pro: no cache discount (cache_read_per_m = 0.00)
# ---------------------------------------------------------------------------
def test_gpt_55_pro_no_cache_discount():
    """gpt-5.5-pro: cached_tokens=1M → $0 (no cache discount)."""
    result = compute_cost("gpt-5.5-pro", cached_tokens=1_000_000)
    assert result == Decimal(0), f"Expected Decimal('0'), got {result!r}"


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
    assert compute_cost(snapshot, prompt_tokens=200_000) == expected_input / 5
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
        assert compute_cost("gpt-5.5-2099-01-01", prompt_tokens=1_000) == Decimal(
            "0.999"
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
    assert compute_cost(snapshot, prompt_tokens=200_000) == input_cost / 5, (
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
            Decimal("4.00"),
            Decimal("20.00"),
            Decimal("0.40"),
            Decimal("5.00"),
            Decimal("8.00"),
            Decimal("30.00"),
            Decimal("0.80"),
            Decimal("10.00"),
        ),
        (
            "gpt-5.6-terra",
            Decimal("2.00"),
            Decimal("12.00"),
            Decimal("0.20"),
            Decimal("2.50"),
            Decimal("4.00"),
            Decimal("18.00"),
            Decimal("0.40"),
            Decimal("5.00"),
        ),
        (
            "gpt-5.6-luna",
            Decimal("0.20"),
            Decimal("1.20"),
            Decimal("0.02"),
            Decimal("0.25"),
            Decimal("0.40"),
            Decimal("1.80"),
            Decimal("0.04"),
            Decimal("0.50"),
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
    # Output alone keeps short pricing; cached tokens must be a subset of input.
    assert compute_cost(model, completion_tokens=1_000_000) == out
    assert compute_cost(model, prompt_tokens=200_000, cached_tokens=200_000) == cread / 5
    assert (
        compute_cost(
            model, prompt_tokens=200_000, cache_write_tokens=200_000,
            cache_write_tokens_details={"ttl_30m_tokens": 200_000, "ttl_12h_tokens": 0},
        )
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
        compute_cost(
            model, prompt_tokens=1_000_000, cache_write_tokens=1_000_000,
            cache_write_tokens_details={"ttl_30m_tokens": 1_000_000, "ttl_12h_tokens": 0},
        )
        == l_cwrite
    )


def test_gpt_56_no_triple_charge_mixed_turn():
    """A turn mixing fresh + cached + written tokens charges each bucket once (Sol).

    fresh = 1000 - 200 cached - 300 written = 500 @ $4/M   = 0.002
    write = 300 @ $5/M                                     = 0.0015
    read  = 200 @ $0.40/M                                  = 0.00008
    out   = 100 @ $20/M                                    = 0.002
    total                                                  = 0.00558
    """
    result = compute_cost(
        "gpt-5.6-sol",
        prompt_tokens=1_000,
        completion_tokens=100,
        cached_tokens=200,
        cache_write_tokens=300,
        cache_write_tokens_details={"ttl_30m_tokens": 300, "ttl_12h_tokens": 0},
    )
    assert result == Decimal("0.00558"), f"got {result!r}"


def test_gpt_56_golden_from_real_usage():
    """Current-rate golden using exact token counts captured from live gpt-5.6-sol.

    Source: two real Responses-API calls with an identical >1024-token prefix
    (see projects/gpt-5-6-support/ground-truth/usage-real.json). The API supplied
    the usage/token evidence; expected costs below are computed at current rates.

    WRITE call: input=2655, cache_write=2652, cached=0, output=5.
        Historical capture has no lifetime evidence: cost is now unknown.
    READ  call: input=2655, cache_write=0, cached=2652, output=5
        fresh=3 @4 + 2652 @0.40 + 5 @20 = 0.000012 + 0.0010608 + 0.000100 = 0.0011728
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
    assert write is None
    assert read == Decimal("0.0011728"), f"READ got {read!r}"


def test_gpt_56_alias_snapshot_resolves():
    """A dated gpt-5.6-sol snapshot resolves to the tier rates via the fallback --
    through BOTH the short table (_RATES) and the long table (_LONG_RATES)."""
    # short context: 200K input (<=272K) -> sol short input $4/M -> $0.80.
    assert compute_cost("gpt-5.6-sol-2026-07-09", prompt_tokens=200_000) == Decimal(
        "0.80"
    )
    # long context: 1M input (>272K) -> snapshot must also resolve through _LONG_RATES
    # to sol's long input rate $8/M -> $8.00.
    assert compute_cost("gpt-5.6-sol-2026-07-09", prompt_tokens=1_000_000) == Decimal(
        "8.00"
    )


def test_gpt_54_and_55_use_documented_long_context_rates():
    """Current pricing documents both thresholds and both long-context rates.

    https://developers.openai.com/api/docs/pricing.md (2026-11-06).
    """
    assert compute_cost("gpt-5.4", prompt_tokens=1_000_000) == Decimal("5.00")
    assert compute_cost("gpt-5.5", prompt_tokens=1_000_000) == Decimal("10.00")


def test_cache_write_ignored_for_models_without_write_rate():
    """Pre-5.6 models have no cache_write_per_m: any cache_write_tokens are billed
    as ordinary input (historical path unchanged, no crash)."""
    # gpt-5.5 has no cache_write rate; passing cache_write_tokens must not error and
    # bills the full prompt as input (fresh = prompt - cached, write ignored).
    result = compute_cost("gpt-5.5", prompt_tokens=1_000, cache_write_tokens=300)
    assert result == Decimal(1000) * Decimal(5) / Decimal(1000000)


# ---------------------------------------------------------------------------
# (r) Golden: exact measured numbers from the cache-write input-accounting
# defect investigation (see test_cache_write_input_accounting.py). This is
# a historical raw-count fixture. The input-accounting bug lived in how the provider mapped
# raw usage fields onto the public Usage.input_tokens contract, NOT in cost
# math. compute_cost() must always be fed the RAW vendor total (prompt_tokens
# = fresh + cache_read + cache_write combined), never the normalized
# Usage.input_tokens value.
# ---------------------------------------------------------------------------
def test_historical_measured_write_without_lifetime_is_unknown():
    """gpt-5.6-sol, short context: input=9028 (raw), cache_write=9025,
    cached=0, output=7. Token counts are from a live capture without lifetime data.

    Do not manufacture 30m evidence for this old write capture.
    """
    result = compute_cost(
        "gpt-5.6-sol",
        prompt_tokens=9_028,
        completion_tokens=7,
        cached_tokens=0,
        cache_write_tokens=9_025,
    )
    assert result is None
