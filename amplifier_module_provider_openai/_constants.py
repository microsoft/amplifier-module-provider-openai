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
DEFAULT_MODEL = "gpt-5.1-codex"
DEFAULT_MAX_TOKENS = 4096
DEFAULT_REASONING_SUMMARY = "detailed"
DEFAULT_DEBUG_TRUNCATE_LENGTH = 180
DEFAULT_TIMEOUT = 600.0  # 10 minutes
DEFAULT_TRUNCATION = "auto"  # Automatic context management

# Maximum number of continuation attempts for incomplete responses
# This prevents infinite loops while being generous enough for legitimate large responses
MAX_CONTINUATION_ATTEMPTS = 5

# Deep research / background mode constants
DEFAULT_POLL_INTERVAL = 5.0  # seconds between status polls
DEFAULT_BACKGROUND_TIMEOUT = 1800.0  # 30 minutes for background requests (deep research can be slow)

# Native tool types that should be passed through to OpenAI without conversion
# These are OpenAI-hosted tools, not user-defined function tools
NATIVE_TOOL_TYPES = frozenset({
    "web_search_preview",
    "web_search_preview_2025_03_11",
    "web_search",
    "file_search",
    "code_interpreter",
})

# Deep research model identifiers
DEEP_RESEARCH_MODELS = frozenset({
    "o3-deep-research",
    "o3-deep-research-2025-06-26",
    "o4-mini-deep-research",
    "o4-mini-deep-research-2025-06-26",
})

# Background response status values
BACKGROUND_STATUS_QUEUED = "queued"
BACKGROUND_STATUS_IN_PROGRESS = "in_progress"
BACKGROUND_STATUS_SEARCHING = "searching"
BACKGROUND_STATUS_COMPLETED = "completed"
BACKGROUND_STATUS_FAILED = "failed"
BACKGROUND_STATUS_CANCELLED = "cancelled"

# Non-terminal statuses that require continued polling
BACKGROUND_POLLING_STATUSES = frozenset({
    BACKGROUND_STATUS_QUEUED,
    BACKGROUND_STATUS_IN_PROGRESS,
    BACKGROUND_STATUS_SEARCHING,
})
