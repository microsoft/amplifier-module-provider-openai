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
    - 5.4+ (or unknown sub-version): 272K context, reasoning, no explicit effort
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
        )

    if family == "gpt-5":
        major, minor = _parse_gpt5_version(model_id)

        if minor >= 4 or (major, minor) == (0, 0):
            # 5.4+ or unparseable version — assume latest
            return ModelCapabilities(
                family="gpt-5",
                context_window=272_000,
                max_output_tokens=128_000,
                supports_reasoning=True,
                default_reasoning_effort=None,
                supports_vision=True,
                supports_streaming=True,
                capability_tags=_GPT5_TAGS,
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
            )

        # 5.2 and below
        return ModelCapabilities(
            family="gpt-5",
            context_window=200_000,
            max_output_tokens=128_000,
            supports_reasoning=True,
            default_reasoning_effort="medium",
            supports_vision=True,
            supports_streaming=True,
            capability_tags=_GPT5_TAGS,
        )

    # unknown — conservative defaults
    return ModelCapabilities(family="unknown")
