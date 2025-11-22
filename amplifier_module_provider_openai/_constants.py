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
DEFAULT_TIMEOUT = 300.0  # 5 minutes
DEFAULT_TRUNCATION = "auto"  # Automatic context management

# Maximum number of continuation attempts for incomplete responses
# This prevents infinite loops while being generous enough for legitimate large responses
MAX_CONTINUATION_ATTEMPTS = 5
