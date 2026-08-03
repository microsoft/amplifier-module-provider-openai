"""Model capabilities lookup — single source of truth for per-model decisions.

Provides a frozen dataclass `ModelCapabilities` and a `get_capabilities()`
function that returns the correct capabilities for any known (or unknown)
model identifier.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = ["ModelCapabilities", "get_capabilities"]

_GPT5_TAGS: tuple[str, ...] = (
    "tools",
    "reasoning",
    "streaming",
    "json_mode",
    "vision",
)


@dataclass(frozen=True)
class ModelCapabilities:
    """Immutable per-model capability descriptor."""

    family: str
    context_window: int = 200_000
    max_output_tokens: int = 128_000
    supports_reasoning: bool = False
    default_reasoning_effort: str | None = None
    supports_vision: bool = True
    supports_streaming: bool = True
    capability_tags: tuple[str, ...] = ("tools", "streaming", "json_mode")
    long_context_pricing_threshold: int | None = (
        None  # Input tokens above this = 2x/1.5x pricing
    )
    supports_in_memory_retention: bool = True
    """Whether the model accepts `prompt_cache_retention="in_memory"`.

    Per OpenAI docs (Feb 2026), gpt-5.5 and gpt-5.5-pro REJECT "in_memory" — their
    default and only supported retention is "24h". Setting this False causes the
    provider to drop the field with a warning rather than send a value the API
    will reject. Default True preserves existing behavior for all other models
    (gpt-5.4, gpt-5.2, gpt-5.1*, gpt-5, gpt-5-codex, gpt-4.1, o-series, etc.).
    """

    supports_24h_retention: bool = True
    """Whether the model accepts `prompt_cache_retention="24h"`.

    Mirrors `supports_in_memory_retention`. Default True reflects empirical
    behavior: the Feb-2026 smoke test confirmed gpt-4o, gpt-5.x, and
    o-series all accept "24h" even when the cookbook list does not formally
    enumerate them. Future families that prove to reject "24h" should set
    this False on their branch; the provider will then drop the field with
    a warning rather than send a value the API will reject.
    """

    supports_native_apply_patch: bool = True
    """Whether the model accepts OpenAI's native `{"type": "apply_patch"}` tool.

    PHILOSOPHY (as of 2026-07, superseding the earlier version-gated rule):
    OpenAI publishes no official support matrix for this tool type, and new
    model releases have consistently added support rather than dropped it.
    Per explicit product direction: DEFAULT TO SUPPORTED. Only exclude
    models/families we have DIRECT EMPIRICAL PROOF reject the native tool
    shape. A wrong optimistic guess is a loud, cheap failure — the
    Responses API rejects an unsupported native tool with a clear
    "Tool 'apply_patch' is not supported with <model>" error at call time.
    That is an acceptable, deliberately-deferred trade-off: no retry/
    fallback-on-rejection mechanism is built for it. If/when a future model
    is discovered to reject it, add it to the named-exclusion list below —
    do not build reactive detection machinery preemptively.

    This inverts the prior rule's default (previously: default False,
    version-gated opt-in for gpt-5.1+). That rule was itself found to be
    wrong in one place: it blanket-excluded every "-chat-latest" alias,
    which produced a false negative for gpt-5.3-chat-latest (empirically
    SUPPORTED). The new rule fixes that — chat-latest aliases default True
    like everything else and are excluded only by name when proven to fail.

    Named/family exclusions (every one of these is a live-API-confirmed
    rejection, not a guess):

    | Result | Models |
    |---|---|
    | NOT supported | gpt-5 (bare), gpt-5-mini, gpt-5-nano, gpt-5-codex, gpt-5-pro, gpt-5-chat-latest (all parse to gpt-5.0 generation, minor == 0) |
    | NOT supported | gpt-5.1-chat-latest (named exception — despite minor == 1, this specific alias is confirmed rejected) |
    | NOT supported | o-series: o1, o1-pro, o3, o3-pro, o3-mini, o4-mini |
    | NOT supported | deep-research: o3-deep-research, o4-mini-deep-research |
    | NOT supported | gpt-4.x: gpt-4.1, gpt-4.1-mini, gpt-4.1-nano, gpt-4o, gpt-4o-mini |
    | SUPPORTED | every gpt-5.1+ variant not named above — INCLUDING mini/nano sizes: gpt-5.1, gpt-5.1-codex, gpt-5.1-codex-mini, gpt-5.1-codex-max, gpt-5.2, gpt-5.2-codex, gpt-5.2-pro, gpt-5.3-codex, gpt-5.3-chat-latest, gpt-5.4, gpt-5.4-mini, gpt-5.4-nano, gpt-5.4-pro, gpt-5.5 |
    | SUPPORTED (default) | any "unknown" model id that isn't a gpt-4.x prefix — genuinely novel/untested models (a hypothetical gpt-6, a new chat-latest alias, etc.) |

    Key findings baked into the rule:

    - Model SIZE (mini/nano) is NOT a valid exclusion signal once
      minor >= 1 — gpt-5.4-mini and gpt-5.4-nano are both confirmed
      supported. This disproves the naive "exclude all -mini/-nano" rule.
    - "-chat-latest" is NOT excluded as a blanket class anymore. Only the
      one alias proven to fail (gpt-5.1-chat-latest) is named explicitly.
      Future chat-latest aliases default True and are added to the
      exclusion list only if/when proven to fail.
    - gpt-4.x models are NOT swept into the new "unknown defaults True"
      trade-off — they were tested and confirmed to reject the tool, so
      they get an explicit, narrow, model_id-prefix-based override (see
      `get_capabilities()`) rather than silently flipping True.
    - TRADE-OFF, explicitly accepted: the broad "unknown" bucket (any
      unrecognized model_id, minus the gpt-4.x carve-out) now defaults
      True. An untested future model that happens to reject the tool will
      fail loudly at the API boundary the first time it's used with
      apply_patch — this is intentional and preferred over maintaining a
      denylist that goes stale, per explicit product direction.

    Default True for the dataclass. `get_capabilities()` sets explicit
    False only for the family/name combinations proven above; every other
    branch either inherits the dataclass default or computes the gpt-5
    version-gated value (minor >= 1 and not the gpt-5.1-chat-latest named
    exception).
    """

    supports_native_computer_use: bool = False
    """Whether the model accepts OpenAI's native `{"type": "computer"}` tool.

    THE INVERSE DISTRIBUTION from `supports_native_apply_patch` \u2014 same
    version-gating machinery (`_detect_family` / `_parse_gpt5_version`),
    opposite default, because the live evidence points the opposite way.

    Live-API evidence (2026-08-03, bare `{"type": "computer"}` tool \u2014 no
    `display_width`/`display_height`/`environment` sub-fields; the GA
    `computer` tool takes no config on declaration \u2014 declared against
    `https://api.openai.com/v1/responses` with `max_output_tokens=16`):

    | Result | Models (live-probed) |
    |---|---|
    | SUPPORTED | gpt-5.4, gpt-5.4-mini, gpt-5.4-pro |
    | SUPPORTED | gpt-5.5, gpt-5.5-pro |
    | SUPPORTED | gpt-5.6 |
    | NOT supported | gpt-5.4-nano (proves size, not just version, gates this) |
    | NOT supported | gpt-5 (bare), gpt-5.1, gpt-5.2, gpt-5.3-chat-latest |

    NOT live-probed (no signal, answered by rule only): `gpt-5.3` returned
    `model_not_found`, not a tool rejection - it is absent from the table
    above deliberately. A 404 says nothing about tool support. Same
    treatment as the deep-research and -nano generalizations noted below.
    | NOT supported | gpt-5-mini (bare, gpt-5.0 generation) |
    | NOT supported | gpt-4o, gpt-4.1 |
    | NOT supported | o-series: o1, o3-mini |

    DEFAULT POLICY (explicit decision, not an implementation convenience):
    unclassifiable models get False. The probe found computer-use is the
    EXCEPTION, not the rule - 3 of 11 probed models accept it - so defaulting
    True would produce frequent loud failures. The accepted cost is the inverse:
    a genuinely-supporting future model this module cannot yet classify reports
    False, and its caller simply does not offer the tool - no error, no log.
    This is why `supports_native_apply_patch` defaults True and this defaults
    False: opposite calls on opposite evidence, not an inconsistency.

    REQUIREMENT ON CALLERS: treat False as "do not offer a computer tool",
    never as "fall back to another vendor's tool type". Falling back reproduces
    the exact downstream defect this field exists to remove.

    Rule: SUPPORTED only when `minor >= 4` AND the model_id is not a
    "-nano" tier. The "-nano" exclusion is a model_id substring check
    (not an enumerated list) so it generalizes to future nano releases
    the same way `_detect_family`/`_parse_gpt5_version` generalize version
    bumps \u2014 proven for gpt-5.4-nano, extended by pattern rather than by
    adding a name every release.

    Default False for the dataclass and for every other family/branch
    (o-series, deep-research, gpt-4.x, gpt-5-mini, gpt-5 through 5.3) \u2014
    inherited, not named one by one, because the majority result here is
    False (the OPPOSITE distribution from apply_patch's majority-True
    result). Sending `computer` to an unsupported model fails loud and
    immediately (`Tool 'computer' is not supported with <model>`, HTTP
    400) \u2014 same fail-closed-at-the-API-boundary property apply_patch
    accepts \u2014 but a narrow default avoids exercising that failure path
    for the common case, where the common case is "does not support it."

    This flag governs ONLY whether the model accepts the native wire tool
    type. It carries no opinion on geometry, coordinate mapping, or action
    decoding \u2014 those are downstream/consumer concerns.
    """


def _detect_family(model_id: str) -> str:
    """Classify *model_id* into a capability family.

    Order matters — deep-research must be checked before o-series because
    deep-research model IDs start with "o3-" / "o4-".
    """
    if "deep-research" in model_id:
        return "deep-research"
    if model_id.startswith("gpt-5-mini") or model_id.startswith("gpt-5.0-mini"):
        return "gpt-5-mini"
    if model_id.startswith("gpt-5"):
        return "gpt-5"
    if re.match(r"^o\d", model_id):
        return "o-series"
    return "unknown"


def _parse_gpt5_version(model_id: str) -> tuple[int, int]:
    """Extract ``(major, minor)`` from a gpt-5 model id.

    Examples::

        gpt-5.4          -> (5, 4)
        gpt-5.4-pro      -> (5, 4)
        gpt-5.3-codex    -> (5, 3)
        gpt-5-mini       -> (5, 0)   # handled by family detection, but safe

    Returns ``(0, 0)`` when parsing fails.
    """
    m = re.match(r"gpt-(\d+)(?:\.(\d+))?", model_id)
    if not m:
        return (0, 0)
    major = int(m.group(1))
    minor = int(m.group(2)) if m.group(2) else 0
    return (major, minor)


def _detect_version(model_id: str, family: str) -> tuple[int, int]:
    """Extract ``(major, minor)`` version from a model ID.

    Uses *family* to short-circuit parsing for non-GPT families.
    For GPT families, delegates to ``_parse_gpt5_version``.

    Examples::

        _detect_version("gpt-5.4", "gpt-5")       -> (5, 4)
        _detect_version("gpt-5.4-pro", "gpt-5")   -> (5, 4)
        _detect_version("gpt-5.3-codex", "gpt-5") -> (5, 3)
        _detect_version("gpt-5-mini", "gpt-5-mini") -> (5, 0)
        _detect_version("o3", "o-series")          -> (0, 0)

    Returns ``(0, 0)`` for non-GPT families or when parsing fails.
    """
    if not family.startswith("gpt-"):
        return (0, 0)
    return _parse_gpt5_version(model_id)


def get_capabilities(model_id: str) -> ModelCapabilities:
    """Return capabilities for *model_id*.

    Version-gated logic for the gpt-5 family:
    - 5.4+ (or unknown sub-version): 1.05M context, reasoning, no explicit effort, 272K pricing threshold
    - 5.3: 400K context, reasoning, no explicit effort
    - 5.2 and below: 200K context, reasoning, implicit effort
    """
    family = _detect_family(model_id)

    if family == "deep-research":
        return ModelCapabilities(
            family="deep-research",
            context_window=200_000,
            max_output_tokens=128_000,
            supports_reasoning=True,
            default_reasoning_effort=None,
            supports_vision=False,
            supports_streaming=False,
            capability_tags=("deep_research", "web_search", "reasoning"),
            supports_native_apply_patch=False,  # confirmed: o3-deep-research, o4-mini-deep-research rejected
        )

    if family == "o-series":
        return ModelCapabilities(
            family="o-series",
            context_window=200_000,
            max_output_tokens=100_000,
            supports_reasoning=True,
            default_reasoning_effort="medium",
            supports_vision=False,
            supports_streaming=True,
            capability_tags=("tools", "reasoning", "streaming"),
            supports_native_apply_patch=False,  # confirmed: o1, o1-pro, o3, o3-pro, o3-mini, o4-mini rejected
        )

    if family == "gpt-5-mini":
        return ModelCapabilities(
            family="gpt-5-mini",
            context_window=128_000,
            max_output_tokens=64_000,
            supports_reasoning=False,
            default_reasoning_effort=None,
            supports_vision=True,
            supports_streaming=True,
            capability_tags=("tools", "streaming", "json_mode", "vision", "fast"),
            supports_native_apply_patch=False,  # confirmed: bare "gpt-5-mini" (gpt-5.0 gen) rejected
        )

    if family == "gpt-5":
        major, minor = _parse_gpt5_version(model_id)

        # Default-True-with-named-exclusions native apply_patch support —
        # empirical, live-API basis. See ModelCapabilities.
        # supports_native_apply_patch docstring for the full evidence table.
        # Rule: minor >= 1 (i.e. gpt-5.1 and later) is SUPPORTED, which
        # covers the entire gpt-5.0 generation (bare gpt-5, gpt-5-nano,
        # gpt-5-codex, gpt-5-pro, gpt-5-chat-latest all parse to minor == 0
        # via `_parse_gpt5_version` and are therefore NOT supported), plus a
        # single NAMED exception: gpt-5.1-chat-latest is confirmed NOT
        # supported despite minor == 1 — this is the one proven failure
        # among chat-latest aliases, called out by name rather than via a
        # blanket "-chat-latest" rule (which would incorrectly also exclude
        # gpt-5.3-chat-latest, confirmed SUPPORTED).
        # Model SIZE (mini/nano) does NOT affect this rule once minor >= 1 —
        # gpt-5.4-mini and gpt-5.4-nano are both confirmed supported.
        supports_apply_patch = minor >= 1 and model_id != "gpt-5.1-chat-latest"

        # Native `computer` tool support — empirical, live-API basis, 2026-08-03.
        # See ModelCapabilities.supports_native_computer_use docstring for the
        # full evidence table. Rule (inverse distribution from apply_patch):
        # minor >= 4 AND not a "-nano" tier. Confirmed live: gpt-5.4,
        # gpt-5.4-mini, gpt-5.4-pro, gpt-5.5, gpt-5.5-pro, gpt-5.6 all SUPPORTED;
        # gpt-5.4-nano confirmed NOT supported despite minor == 4 — size, not
        # just version, gates this tool (unlike apply_patch, where mini/nano
        # made no difference once minor >= 1).
        supports_computer_use = minor >= 4 and "-nano" not in model_id

        # gpt-5.5 — verified against live API 2026-04-24.
        # 1M context, ~4x input / 3x output pricing vs 5.4. Reasoning blocks,
        # rs_* IDs, and encrypted_content carry the same shape as 5.4.
        # long_context_pricing_threshold is left None: the public pricing page
        # is not API-derivable, and 5.5's threshold (if any) differs from
        # 5.4's 272K given the price step. Callers see the full 1M context.
        if minor == 5:
            return ModelCapabilities(
                family="gpt-5",
                context_window=1_000_000,
                max_output_tokens=128_000,
                supports_reasoning=True,
                default_reasoning_effort=None,
                supports_vision=True,
                supports_streaming=True,
                capability_tags=_GPT5_TAGS,
                long_context_pricing_threshold=None,
                supports_in_memory_retention=False,  # 5.5 default is "24h", "in_memory" rejected
                supports_native_apply_patch=supports_apply_patch,
                supports_native_computer_use=supports_computer_use,
            )

        # gpt-5.6 (Sol / Terra / Luna) -- GA 2026-07-09.
        #
        # context_window=900_000 is EMPIRICALLY MEASURED, not the marketing number.
        # OpenAI advertises ~1.05M, but a live binary-search probe of gpt-5.6-sol
        # (2026-07-14) found the real hard ceiling is ~908K-928K: a 907,812-token
        # input succeeds, 928,125 returns HTTP 400 context_length_exceeded. The
        # nominal 1.05M is NOT deliverable -- reporting it caused real
        # context_length_exceeded crashes on long sessions. 900_000 sits safely
        # below the measured success point (headroom for output tokens). All three
        # tiers share this ceiling (they differ in price/latency, see _cost.py).
        #
        # in_memory retention is REJECTED by the API ("compatible only with 24h
        # extended prompt caching"), so supports_in_memory_retention=False routes
        # callers to 24h via the existing _drop_unsupported_in_memory_retention gate.
        #
        # long_context_pricing_threshold=272_000 (mirrors gpt-5.4): gpt-5.6 has a
        # documented short/long-context price split (~2x) at this boundary. get_info()
        # reports the THRESHOLD as the default context_window, so unpinned sessions
        # compact against the standard-priced 272K window and don't silently incur 2x
        # billing. Callers who need the full measured 900K opt in via
        # enable_long_context=True (accepting the ~2x price) -- identical to gpt-5.4.
        if minor == 6:
            return ModelCapabilities(
                family="gpt-5",
                context_window=900_000,
                max_output_tokens=128_000,
                supports_reasoning=True,
                default_reasoning_effort=None,
                supports_vision=True,
                supports_streaming=True,
                capability_tags=_GPT5_TAGS,
                long_context_pricing_threshold=272_000,
                supports_in_memory_retention=False,
                supports_native_apply_patch=supports_apply_patch,
                supports_native_computer_use=supports_computer_use,
            )

        if minor >= 4 or (major, minor) == (0, 0):
            # 5.4+ or unparseable version — assume latest
            return ModelCapabilities(
                family="gpt-5",
                context_window=1_050_000,
                max_output_tokens=128_000,
                supports_reasoning=True,
                default_reasoning_effort=None,
                supports_vision=True,
                supports_streaming=True,
                capability_tags=_GPT5_TAGS,
                long_context_pricing_threshold=272_000,
                supports_native_apply_patch=supports_apply_patch,
                supports_native_computer_use=supports_computer_use,
            )

        if minor == 3:
            return ModelCapabilities(
                family="gpt-5",
                context_window=400_000,
                max_output_tokens=128_000,
                supports_reasoning=True,
                default_reasoning_effort=None,
                supports_vision=True,
                supports_streaming=True,
                capability_tags=_GPT5_TAGS,
                supports_native_apply_patch=supports_apply_patch,
            )

        # 5.2 and below — covers gpt-5 (bare), gpt-5.1, gpt-5.1-codex, gpt-5.2,
        # gpt-5.2-codex, and "-chat-latest" variants of any of these versions.
        # supports_apply_patch correctly resolves each: gpt-5 bare -> False
        # (minor 0), gpt-5.1/gpt-5.2* -> True, gpt-5.1-chat-latest -> False
        # (chat-latest exclusion).
        return ModelCapabilities(
            family="gpt-5",
            context_window=200_000,
            max_output_tokens=128_000,
            supports_reasoning=True,
            supports_native_apply_patch=supports_apply_patch,
            default_reasoning_effort="medium",
            supports_vision=True,
            supports_streaming=True,
            capability_tags=_GPT5_TAGS,
        )

    # unknown — every OTHER capability (context_window, max_output_tokens,
    # supports_reasoning, supports_vision, supports_streaming, capability_tags,
    # retention flags) keeps its exact existing "unknown" bucket default.
    # supports_native_apply_patch is the ONE field that now differs by
    # model_id within this bucket:
    #
    #   - gpt-4.x models (gpt-4.1, gpt-4.1-mini, gpt-4.1-nano, gpt-4o,
    #     gpt-4o-mini) are confirmed empirically NOT supported. This is a
    #     narrow, model_id-prefix-based override — NOT a new family branch —
    #     deliberately, so family stays "unknown" for these models exactly as
    #     before (see test_gpt_4_1_mini_does_not_reason in test_capabilities.py,
    #     which asserts family == "unknown" for gpt-4.1-mini) and every other
    #     capability field is untouched. A dedicated "gpt-4" family branch was
    #     considered and rejected: it would require re-deriving all the other
    #     "unknown"-bucket defaults inside the new branch to avoid silently
    #     changing them — strictly more surface area for the same outcome.
    #   - Every other unrecognized model_id (genuinely novel/untested models —
    #     a hypothetical gpt-6, a new provider alias, etc.) now defaults True
    #     via the dataclass default. This is the accepted "loud failure if
    #     wrong" trade-off: sending the native tool to a model that silently
    #     doesn't support it fails immediately and clearly at the API
    #     boundary, per explicit product direction to default-optimistic and
    #     only exclude proven failures.
    if model_id.startswith("gpt-4"):
        return ModelCapabilities(family="unknown", supports_native_apply_patch=False)

    return ModelCapabilities(family="unknown")
