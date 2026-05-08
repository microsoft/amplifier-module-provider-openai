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
DEFAULT_MODEL = "gpt-5.5"
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
# model list. Models that reject "24h" are gated by
# `ModelCapabilities.supports_24h_retention`.
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
NATIVE_TOOL_TYPES = frozenset(
    {
        "web_search_preview",
        "web_search_preview_2025_03_11",
        "web_search",
        "file_search",
        "code_interpreter",
        "apply_patch",
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

# Hook event emitted when the server rejects previous_response_id.
# Caller observability: lets dashboards count chain breaks vs. cache hits.
RESPONSE_CHAIN_INVALIDATED = "provider:response_chain_invalidated"

# OpenAI error codes that signal "previous_response_id is unknown/expired/foreign-key".
# Detected against error.body["error"]["code"] (preferred) or, as a fallback,
# substring match against the raw error message. Keep this set narrow — we do
# NOT want to swallow generic 4xx errors as "chain invalidations".
RESPONSE_NOT_FOUND_ERROR_CODES = frozenset(
    {
        "response_not_found",
        "previous_response_not_found",
    }
)
