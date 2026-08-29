"""Constants for OpenAI provider.

This module defines constants used across the OpenAI provider implementation,
following the principle of single source of truth.
"""

# Metadata keys for OpenAI Responses API state
# These keys are namespaced with "openai:" to prevent collisions with other providers
METADATA_RESPONSE_ID = "openai:response_id"
METADATA_STATUS = "openai:status"
METADATA_INCOMPLETE_REASON = "openai:incomplete_reason"
METADATA_REASONING_ITEMS = "openai:reasoning_items"
METADATA_CONTINUATION_COUNT = "openai:continuation_count"

# Default configuration values
# gpt-5.6-sol (the GPT-5.6 flagship; alias "gpt-5.6" also resolves to it) is the
# default: same input/output pricing as gpt-5.5 ($5/$30) with the newer model's
# capabilities. Note gpt-5.6 bills cache-WRITE tokens at 1.25x input (automatic on
# prompts >1024 tokens) and rejects "in_memory" retention (auto-dropped to 24h).
DEFAULT_MODEL = "gpt-5.6-sol"
DEFAULT_MAX_TOKENS = 4096
DEFAULT_REASONING_SUMMARY = "detailed"
DEFAULT_DEBUG_TRUNCATE_LENGTH = 180
DEFAULT_TIMEOUT = 600.0  # 10 minutes
# `None` (omit the field) is the cache-friendly default. OpenAI's
# `truncation="auto"` silently drops oldest messages when context fills,
# which rewrites the cached prefix and busts prompt caching — listed on
# OpenAI's caching-troubleshooting checklist as a top cause of low hit
# rates. With `None`, the API errors loudly on overflow instead of
# silently degrading. Opt back into the old behavior with
# `config={"truncation": "auto"}`.
DEFAULT_TRUNCATION: str | None = None

# Default prompt-cache retention. OpenAI's per-model server-side default
# is "in_memory" (5–10 min) for gpt-5.4 and below, "24h" for gpt-5.5+.
# Forcing "24h" everywhere stabilizes cache lifetime across the curated
# model list. Models that reject "in_memory" are gated by
# `ModelCapabilities.supports_in_memory_retention`.
DEFAULT_PROMPT_CACHE_RETENTION: str | None = "24h"

# Maximum number of continuation attempts for incomplete responses
# This prevents infinite loops while being generous enough for legitimate large responses
MAX_CONTINUATION_ATTEMPTS = 5

# Deep research / background mode constants
DEFAULT_POLL_INTERVAL = 5.0  # seconds between status polls
DEFAULT_BACKGROUND_TIMEOUT = (
    1800.0  # 30 minutes for background requests (deep research can be slow)
)

# Native tool types that should be passed through to OpenAI without conversion
# These are OpenAI-hosted tools, not user-defined function tools
#
# `computer` (OpenAI's computer-use tool) is unusual among this set: live
# Responses API traffic against gpt-5.6 confirms it accepts *zero* declaration
# fields. `{"type": "computer"}` alone -> 200. Adding any of `display_width`,
# `display_height`, `environment`, or `display_width_px` -> 400
# "Unknown parameter" (the opposite of Anthropic's `computer_20251124`, which
# *requires* `display_width_px`/`display_height_px`). See
# `_convert_tools_from_request` for where this is enforced.
NATIVE_TOOL_TYPES = frozenset(
    {
        "web_search_preview",
        "web_search_preview_2025_03_11",
        "web_search",
        "file_search",
        "code_interpreter",
        "apply_patch",
        "computer",
    }
)

# Deep research model identifiers
DEEP_RESEARCH_MODELS = frozenset(
    {
        "o3-deep-research",
        "o3-deep-research-2025-06-26",
        "o4-mini-deep-research",
        "o4-mini-deep-research-2025-06-26",
    }
)

# Background response status values
BACKGROUND_STATUS_QUEUED = "queued"
BACKGROUND_STATUS_IN_PROGRESS = "in_progress"
BACKGROUND_STATUS_SEARCHING = "searching"
BACKGROUND_STATUS_COMPLETED = "completed"
BACKGROUND_STATUS_FAILED = "failed"
BACKGROUND_STATUS_CANCELLED = "cancelled"

# Non-terminal statuses that require continued polling
BACKGROUND_POLLING_STATUSES = frozenset(
    {
        BACKGROUND_STATUS_QUEUED,
        BACKGROUND_STATUS_IN_PROGRESS,
        BACKGROUND_STATUS_SEARCHING,
    }
)
