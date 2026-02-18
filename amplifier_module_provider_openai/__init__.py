"""
OpenAI provider module for Amplifier.
Integrates with OpenAI's Responses API.
"""

__all__ = ["mount", "OpenAIProvider"]

# Amplifier module metadata
__amplifier_module_type__ = "provider"

import asyncio
import json
import logging
import os
import time
from typing import Any
from typing import cast

import openai
from amplifier_core import ConfigField
from amplifier_core import ModelInfo
from amplifier_core import ModuleCoordinator
from amplifier_core import ProviderInfo
from amplifier_core import TextContent
from amplifier_core import ThinkingContent
from amplifier_core import ToolCallContent
from amplifier_core import llm_errors as kernel_errors
from amplifier_core.events import PROVIDER_RETRY
from amplifier_core.message_models import ChatRequest
from amplifier_core.message_models import ChatResponse
from amplifier_core.message_models import ToolCall
from amplifier_core.utils.retry import RetryConfig, retry_with_backoff
from openai import AsyncOpenAI

from ._constants import BACKGROUND_POLLING_STATUSES
from ._constants import BACKGROUND_STATUS_COMPLETED
from ._constants import BACKGROUND_STATUS_FAILED
from ._constants import DEFAULT_BACKGROUND_TIMEOUT
from ._constants import DEFAULT_DEBUG_TRUNCATE_LENGTH
from ._constants import DEFAULT_MAX_TOKENS
from ._constants import DEFAULT_MODEL
from ._constants import DEFAULT_POLL_INTERVAL
from ._constants import DEFAULT_REASONING_SUMMARY
from ._constants import DEFAULT_TIMEOUT
from ._constants import DEFAULT_TRUNCATION
from ._constants import DEEP_RESEARCH_MODELS
from ._constants import MAX_CONTINUATION_ATTEMPTS
from ._constants import METADATA_CONTINUATION_COUNT
from ._constants import METADATA_INCOMPLETE_REASON
from ._constants import METADATA_REASONING_ITEMS
from ._constants import METADATA_RESPONSE_ID
from ._constants import METADATA_STATUS
from ._constants import NATIVE_TOOL_TYPES
from ._response_handling import convert_response_with_accumulated_output
from ._response_handling import extract_reasoning_text

logger = logging.getLogger(__name__)


class OpenAIChatResponse(ChatResponse):
    """ChatResponse with additional fields for streaming UI compatibility."""

    content_blocks: list[TextContent | ThinkingContent | ToolCallContent] | None = None
    text: str | None = None
    # Per OpenAI docs: "response.output_text is the safest way to retrieve the final answer"
    # Exposed directly for tools like deep_research that need reliable text extraction
    output_text: str | None = None


async def mount(coordinator: ModuleCoordinator, config: dict[str, Any] | None = None):
    """Mount the OpenAI provider."""
    config = config or {}

    # Get API key from config or environment
    api_key = config.get("api_key") or os.environ.get("OPENAI_API_KEY")

    if not api_key:
        logger.warning("No API key found for OpenAI provider")
        return None

    provider = OpenAIProvider(api_key=api_key, config=config, coordinator=coordinator)
    await coordinator.mount("providers", provider, name="openai")
    logger.info("Mounted OpenAIProvider (Responses API)")

    # Return cleanup function
    async def cleanup():
        if hasattr(provider.client, "close"):
            await provider.client.close()

    return cleanup


class OpenAIProvider:
    """OpenAI Responses API integration."""

    name = "openai"
    api_label = "OpenAI"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        config: dict[str, Any] | None = None,
        coordinator: ModuleCoordinator | None = None,
        client: AsyncOpenAI | None = None,
    ):
        """Initialize OpenAI provider with Responses API client.

        The SDK client is created lazily on first use, allowing get_info()
        to work without valid credentials.
        """
        self._api_key = api_key
        self._client: AsyncOpenAI | None = client  # Lazy init if None
        self.config = config or {}
        self.coordinator = coordinator

        # Configuration with sensible defaults (from _constants.py - single source of truth)
        self.base_url = self.config.get(
            "base_url", None
        )  # Optional custom endpoint (None = OpenAI default)
        self.default_model = self.config.get("default_model", DEFAULT_MODEL)
        self.max_tokens = self.config.get("max_tokens", DEFAULT_MAX_TOKENS)
        self.temperature = self.config.get(
            "temperature", None
        )  # None = not sent (some models don't support it)
        self.reasoning = self.config.get(
            "reasoning", None
        )  # None = not sent (minimal|low|medium|high)
        self.reasoning_summary = self.config.get(
            "reasoning_summary", DEFAULT_REASONING_SUMMARY
        )
        self.truncation = self.config.get(
            "truncation", DEFAULT_TRUNCATION
        )  # Automatic context management
        self.enable_state = self.config.get("enable_state", False)
        self.debug = self.config.get(
            "debug", False
        )  # Enable full request/response logging
        self.raw_debug = self.config.get(
            "raw_debug", False
        )  # Enable ultra-verbose raw API I/O logging
        self.debug_truncate_length = self.config.get(
            "debug_truncate_length", DEFAULT_DEBUG_TRUNCATE_LENGTH
        )
        self.timeout = self.config.get("timeout", DEFAULT_TIMEOUT)
        self.filtered = self.config.get(
            "filtered", True
        )  # Filter to curated model list by default

        # Deep research / background mode configuration
        self.poll_interval = self.config.get("poll_interval", DEFAULT_POLL_INTERVAL)
        self.background_timeout = self.config.get(
            "background_timeout", DEFAULT_BACKGROUND_TIMEOUT
        )

        # Provider priority for selection (lower = higher priority)
        self.priority = self.config.get("priority", 100)

        # Retry configuration — delegates to shared retry_with_backoff() from amplifier-core.
        # Backward compat: retry_jitter used to be bool (True/False). Now it's a float
        # for RetryConfig.jitter (0.0-1.0). Accept both: True→0.2, False→0.0, float→as-is.
        jitter_cfg = self.config.get("retry_jitter", 0.2)
        if isinstance(jitter_cfg, bool):
            jitter_cfg = 0.2 if jitter_cfg else 0.0
        self._retry_config = RetryConfig(
            max_retries=int(self.config.get("max_retries", 5)),
            min_delay=float(self.config.get("min_retry_delay", 1.0)),
            max_delay=float(self.config.get("max_retry_delay", 60.0)),
            jitter=float(jitter_cfg),
        )

        # Track tool call IDs that have been repaired with synthetic results.
        # This prevents infinite loops when the same missing tool results are
        # detected repeatedly across LLM iterations (since synthetic results
        # are injected into request.messages but not persisted to message store).
        self._repaired_tool_ids: set[str] = set()

        # Apply patch native mode detection — set during tool conversion
        self._apply_patch_native = False
        self._native_call_ids: set[str] = set()

    @property
    def client(self) -> AsyncOpenAI:
        """Lazily initialize the OpenAI client on first access."""
        if self._client is None:
            if self._api_key is None:
                raise ValueError("api_key or client must be provided for API calls")
            self._client = AsyncOpenAI(
                api_key=self._api_key, base_url=self.base_url, max_retries=0
            )
        return self._client

    def get_info(self) -> ProviderInfo:
        """Get provider metadata."""
        return ProviderInfo(
            id="openai",
            display_name="OpenAI",
            credential_env_vars=["OPENAI_API_KEY"],
            capabilities=["streaming", "tools", "reasoning", "batch", "json_mode"],
            defaults={
                "model": "gpt-5.1",
                "max_tokens": 16384,
                "temperature": None,  # Model default
                "timeout": 600.0,
                "context_window": 400000,
                "max_output_tokens": 128000,
            },
            config_fields=[
                ConfigField(
                    id="api_key",
                    display_name="API Key",
                    field_type="secret",
                    prompt="Enter your OpenAI API key",
                    env_var="OPENAI_API_KEY",
                ),
                ConfigField(
                    id="base_url",
                    display_name="API Base URL",
                    field_type="text",
                    prompt="API base URL",
                    env_var="OPENAI_BASE_URL",
                    required=False,
                    default="https://api.openai.com/v1",
                ),
                ConfigField(
                    id="reasoning_effort",
                    display_name="Reasoning Effort",
                    field_type="choice",
                    prompt="Select reasoning effort level",
                    choices=["none", "minimal", "low", "medium", "high"],
                    default="none",
                    required=False,
                    requires_model=True,  # Shown after model selection
                ),
            ],
        )

    async def list_models(self) -> list[ModelInfo]:
        """
        List available OpenAI models.

        Queries the OpenAI API for available models and filters to GPT-5+ series
        and deep research models.
        Raises exception if API query fails (no fallback - caller handles empty lists).
        """
        # Query OpenAI models API - let exceptions propagate
        models_response = await self.client.models.list()
        models = []

        import re as regex_module

        for model in models_response.data:
            model_id = model.id

            # Check if this is a deep research model
            is_deep_research = model_id in DEEP_RESEARCH_MODELS or model_id.startswith(
                ("o3-deep-research", "o4-mini-deep-research")
            )

            # Filter to GPT-5+ series models or deep research models
            if not (
                model_id.startswith("gpt-5")
                or model_id.startswith("gpt-6")
                or is_deep_research
            ):
                continue

            # Skip dated versions when filtered (e.g., gpt-5-2025-08-07) - duplicates of aliases
            # But always include deep research aliases (o3-deep-research, o4-mini-deep-research)
            if (
                self.filtered
                and not is_deep_research
                and regex_module.search(r"-\d{4}-\d{2}-\d{2}$", model_id)
            ):
                continue

            # Generate display name from model ID
            display_name = self._model_id_to_display_name(model_id)

            # Determine capabilities based on model type
            if is_deep_research:
                capabilities = ["deep_research", "web_search", "reasoning"]
                context_window = 200000
                max_output_tokens = 100000
                defaults = {"max_tokens": 32768, "background": True}
            else:
                capabilities = ["tools", "reasoning", "streaming", "json_mode"]
                if "mini" in model_id or "nano" in model_id:
                    capabilities.append("fast")
                context_window = 400000
                max_output_tokens = 128000
                defaults = {"max_tokens": 16384, "reasoning_effort": "none"}

            models.append(
                ModelInfo(
                    id=model_id,
                    display_name=display_name,
                    context_window=context_window,
                    max_output_tokens=max_output_tokens,
                    capabilities=capabilities,
                    defaults=defaults,
                )
            )

        # Sort alphabetically by display name
        return sorted(models, key=lambda m: m.display_name.lower())

    def _model_id_to_display_name(self, model_id: str) -> str:
        """Convert model ID to display name with proper capitalization.

        Examples:
            gpt-5.1 -> GPT 5.1
            gpt-5.1-codex -> GPT-5.1 codex
            gpt-5-mini -> GPT-5 mini
            o3-deep-research -> o3 Deep Research
            o4-mini-deep-research -> o4-mini Deep Research
        """
        # Known display name mappings
        display_names = {
            "gpt-5.1": "GPT 5.1",
            "gpt-5.1-codex": "GPT-5.1 codex",
            "gpt-5-mini": "GPT-5 mini",
            "o3-deep-research": "o3 Deep Research",
            "o3-deep-research-2025-06-26": "o3 Deep Research (2025-06-26)",
            "o4-mini-deep-research": "o4-mini Deep Research",
            "o4-mini-deep-research-2025-06-26": "o4-mini Deep Research (2025-06-26)",
        }

        if model_id in display_names:
            return display_names[model_id]

        # Handle deep research model variants
        if "deep-research" in model_id:
            # Extract base model (o3, o4-mini, etc.) and format nicely
            if model_id.startswith("o3-deep-research"):
                suffix = model_id.replace("o3-deep-research", "")
                return f"o3 Deep Research{suffix}"
            if model_id.startswith("o4-mini-deep-research"):
                suffix = model_id.replace("o4-mini-deep-research", "")
                return f"o4-mini Deep Research{suffix}"

        # Generate from ID: capitalize GPT, keep rest lowercase
        if model_id.startswith("gpt-"):
            parts = model_id.split("-", 1)
            if len(parts) == 2:
                return f"GPT-{parts[1]}"
        return model_id

    def _model_may_reason(self, model_name: str) -> bool:
        """Check if a model may produce reasoning output by default.

        Models like gpt-5.2-codex and o-series reason without explicit
        reasoning_effort being set.  We need to request encrypted_content
        for these models even when no reasoning param is in the request.
        """
        if not model_name:
            return False
        name = model_name.lower()
        # Models known to reason by default
        reasoning_prefixes = ("o1", "o3", "o4", "codex", "gpt-5")
        return any(name.startswith(p) or f"-{p}" in name for p in reasoning_prefixes)

    def _build_continuation_input(
        self, original_input: list, accumulated_output: list
    ) -> list:
        """Build input for continuation call in stateless mode.

        Instead of using previous_response_id (requires store:true), we include
        the accumulated output in the next request's input to preserve context.
        This allows continuation to work in stateless mode.

        Per OpenAI Responses API docs: "context += response.output" - the API
        accepts output items (reasoning, message, tool_call) directly in the
        input array for continuation.

        Args:
            original_input: The original input messages from the first call
            accumulated_output: Output items accumulated from incomplete response(s)

        Returns:
            New input array with accumulated output included for continuation
        """
        # Start with original input (the conversation so far)
        continuation_input = list(original_input)

        # Convert accumulated output to assistant messages for input
        # Extract text from message blocks and reasoning summaries
        assistant_content = []

        for item in accumulated_output:
            if hasattr(item, "type"):
                item_type = item.type
                if item_type == "message":
                    # Extract text from message content
                    content = getattr(item, "content", [])
                    for content_item in content:
                        if (
                            hasattr(content_item, "type")
                            and content_item.type == "output_text"
                        ):
                            text = getattr(content_item, "text", "")
                            if text:
                                assistant_content.append(
                                    {"type": "output_text", "text": text}
                                )
                elif item_type == "reasoning":
                    # For reasoning, we can't really include it in input as text
                    # The reasoning trace is internal and not meant for reinsertion
                    # Skip for now - continuation will lose reasoning context
                    pass
                elif item_type in {"tool_call", "function_call"}:
                    # Tool calls - we'd need to include these but this is complex
                    # For now, skip - incomplete with tool calls is edge case
                    pass
            else:
                # Dictionary format
                item_type = item.get("type")
                if item_type == "message":
                    content = item.get("content", [])
                    for content_item in content:
                        if content_item.get("type") == "output_text":
                            text = content_item.get("text", "")
                            if text:
                                assistant_content.append(
                                    {"type": "output_text", "text": text}
                                )

        # If we extracted any assistant content, add as assistant message
        if assistant_content:
            continuation_input.append(
                {"role": "assistant", "content": assistant_content}
            )

        return continuation_input

    def _truncate_values(self, obj: Any, max_length: int | None = None) -> Any:
        """Recursively truncate string values in nested structures.

        Preserves structure, only truncates leaf string values longer than max_length.
        Uses self.debug_truncate_length if max_length not specified.

        Args:
            obj: Any JSON-serializable structure (dict, list, primitives)
            max_length: Maximum string length (defaults to self.debug_truncate_length)

        Returns:
            Structure with truncated string values
        """
        if max_length is None:
            max_length = self.debug_truncate_length

        # Type guard: max_length is guaranteed to be int after this point
        assert max_length is not None, (
            "max_length should never be None after initialization"
        )

        if isinstance(obj, str):
            if len(obj) > max_length:
                return (
                    obj[:max_length] + f"... (truncated {len(obj) - max_length} chars)"
                )
            return obj
        if isinstance(obj, dict):
            return {k: self._truncate_values(v, max_length) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._truncate_values(item, max_length) for item in obj]
        return obj  # Numbers, booleans, None pass through unchanged

    def _find_missing_tool_results(self, messages: list) -> list[tuple[str, str, dict]]:
        """Find tool calls without matching results.

        Scans conversation for assistant tool calls and validates each has
        a corresponding tool result message. Returns missing pairs.

        Excludes tool call IDs that have already been repaired with synthetic
        results to prevent infinite detection loops.

        Returns:
            List of (call_id, tool_name, tool_arguments) tuples for unpaired calls
        """
        from amplifier_core.message_models import Message

        tool_calls = {}  # {call_id: (name, args)}
        tool_results = set()  # {call_id}

        for msg in messages:
            # Check assistant messages for ToolCallBlock in content
            if msg.role == "assistant" and isinstance(msg.content, list):
                for block in msg.content:
                    if hasattr(block, "type") and block.type == "tool_call":
                        tool_calls[block.id] = (block.name, block.input)

            # Check tool messages for tool_call_id
            elif (
                msg.role == "tool" and hasattr(msg, "tool_call_id") and msg.tool_call_id
            ):
                tool_results.add(msg.tool_call_id)

        # Exclude IDs that have already been repaired to prevent infinite loops
        return [
            (call_id, name, args)
            for call_id, (name, args) in tool_calls.items()
            if call_id not in tool_results and call_id not in self._repaired_tool_ids
        ]

    def _create_synthetic_result(self, call_id: str, tool_name: str):
        """Create synthetic error result for missing tool response.

        This is a BACKUP for when tool results go missing AFTER execution.
        The orchestrator should handle tool execution errors at runtime,
        so this should only trigger on context/parsing bugs.
        """
        from amplifier_core.message_models import Message

        return Message(
            role="tool",
            content=(
                f"[SYSTEM ERROR: Tool result missing from conversation history]\n\n"
                f"Tool: {tool_name}\n"
                f"Call ID: {call_id}\n\n"
                f"This indicates the tool result was lost after execution.\n"
                f"Likely causes: context compaction bug, message parsing error, or state corruption.\n\n"
                f"The tool may have executed successfully, but the result was lost.\n"
                f"Please acknowledge this error and offer to retry the operation."
            ),
            tool_call_id=call_id,
            name=tool_name,
        )

    async def complete(self, request: ChatRequest, **kwargs) -> ChatResponse:
        """Generate completion using Responses API.

        Args:
            request: Typed chat request with messages, tools, config
            **kwargs: Provider-specific options (override request fields)

        Returns:
            ChatResponse with content blocks, tool calls, usage
        """
        # VALIDATE AND REPAIR: Check for missing tool results (backup safety net)
        missing = self._find_missing_tool_results(request.messages)

        if missing:
            logger.warning(
                f"[PROVIDER] OpenAI: Detected {len(missing)} missing tool result(s). "
                f"Injecting synthetic errors. This indicates a bug in context management. "
                f"Tool IDs: {[call_id for call_id, _, _ in missing]}"
            )

            # Inject synthetic results and track repaired IDs to prevent infinite loops
            for call_id, tool_name, _ in missing:
                synthetic = self._create_synthetic_result(call_id, tool_name)
                request.messages.append(synthetic)
                # Track this ID so we don't detect it as missing again in future iterations
                self._repaired_tool_ids.add(call_id)

            # Emit observability event
            if self.coordinator and hasattr(self.coordinator, "hooks"):
                await self.coordinator.hooks.emit(
                    "provider:tool_sequence_repaired",
                    {
                        "provider": self.name,
                        "repair_count": len(missing),
                        "repairs": [
                            {"tool_call_id": call_id, "tool_name": tool_name}
                            for call_id, tool_name, _ in missing
                        ],
                    },
                )

        return await self._complete_chat_request(request, **kwargs)

    def parse_tool_calls(self, response: ChatResponse) -> list[ToolCall]:
        """
        Parse tool calls from ChatResponse.

        Args:
            response: Typed chat response

        Returns:
            List of tool calls from the response
        """
        if not response.tool_calls:
            return []
        return response.tool_calls

    async def _complete_chat_request(
        self, request: ChatRequest, **kwargs
    ) -> ChatResponse:
        """Handle ChatRequest format with developer message conversion.

        Args:
            request: ChatRequest with messages
            **kwargs: Additional parameters

        Returns:
            ChatResponse with content blocks
        """
        logger.info(
            f"[PROVIDER] Received ChatRequest with {len(request.messages)} messages"
        )
        logger.info(f"[PROVIDER] Message roles: {[m.role for m in request.messages]}")

        message_list = list(request.messages)

        # Separate messages by role
        system_msgs = [m for m in message_list if m.role == "system"]
        developer_msgs = [m for m in message_list if m.role == "developer"]
        conversation = [
            m for m in message_list if m.role in ("user", "assistant", "tool")
        ]

        logger.info(
            f"[PROVIDER] Separated: {len(system_msgs)} system, {len(developer_msgs)} developer, {len(conversation)} conversation"
        )

        # Combine system messages as instructions
        instructions = (
            "\n\n".join(
                m.content if isinstance(m.content, str) else "" for m in system_msgs
            )
            if system_msgs
            else None
        )

        # Convert all messages (developer + conversation) to Responses API format
        # Developer messages become XML-wrapped user messages, tools are batched
        all_messages_for_conversion = []

        # Add developer messages first
        for dev_msg in developer_msgs:
            all_messages_for_conversion.append(dev_msg.model_dump())

        # Add conversation messages
        for conv_msg in conversation:
            all_messages_for_conversion.append(conv_msg.model_dump())

        # Convert to OpenAI Responses API message format
        input_messages = self._convert_messages(all_messages_for_conversion)
        logger.info(
            f"[PROVIDER] Converted {len(all_messages_for_conversion)} messages to {len(input_messages)} API messages"
        )

        # Check for previous response metadata to preserve reasoning state across turns
        previous_response_id = None
        if message_list:
            # Look at the last assistant message for metadata
            for msg in reversed(message_list):
                if msg.role == "assistant":
                    # Check if message has our metadata
                    msg_dict = msg.model_dump() if hasattr(msg, "model_dump") else msg
                    if isinstance(msg_dict, dict) and msg_dict.get("metadata"):
                        metadata = msg_dict["metadata"]
                        prev_id = metadata.get(METADATA_RESPONSE_ID)
                        if prev_id:
                            previous_response_id = prev_id
                            logger.info(
                                f"[PROVIDER] Found previous_response_id={prev_id} "
                                f"from last assistant message - will preserve reasoning state"
                            )
                            break

        # Prepare request parameters per Responses API spec
        params = {
            "model": kwargs.get("model", self.default_model),
            "input": input_messages,  # Array of message objects, not text string
        }

        # Check for background mode (used for deep research and long-running requests)
        # Background mode requires store=True per OpenAI API requirements
        background_mode = kwargs.get("background", False)

        # Auto-enable background mode for deep research models
        model_name = kwargs.get("model", self.default_model)
        if model_name in DEEP_RESEARCH_MODELS or model_name.startswith(
            ("o3-deep-research", "o4-mini-deep-research")
        ):
            # Deep research models should use background mode by default
            background_mode = kwargs.get("background", True)
            logger.info(
                f"[PROVIDER] Deep research model detected: {model_name}, background={background_mode}"
            )

        # Determine store parameter early (needed for previous_response_id logic)
        # Background mode requires store=True
        store_enabled = kwargs.get("store", self.enable_state)
        if background_mode:
            store_enabled = True  # Background mode requires store=True
            logger.info("[PROVIDER] Background mode enabled, forcing store=True")
        params["store"] = store_enabled

        # Add previous_response_id ONLY if store is enabled (server-side state)
        # With store=False, we rely on explicit reasoning re-insertion instead
        if previous_response_id and store_enabled:
            params["previous_response_id"] = previous_response_id
            logger.debug("[PROVIDER] Using previous_response_id (store=True)")
        elif previous_response_id and not store_enabled:
            logger.debug(
                "[PROVIDER] Skipping previous_response_id (store=False). "
                "Relying on explicit reasoning re-insertion from metadata/content."
            )

        if instructions:
            params["instructions"] = instructions

        if request.max_output_tokens:
            params["max_output_tokens"] = request.max_output_tokens
        elif max_tokens := kwargs.get("max_tokens", self.max_tokens):
            params["max_output_tokens"] = max_tokens

        if request.temperature is not None:
            params["temperature"] = request.temperature
        elif temperature := kwargs.get("temperature", self.temperature):
            params["temperature"] = temperature

        # Phase 2: Reasoning parameter precedence chain
        # kwargs["reasoning"] > request.reasoning_effort > config default > None
        reasoning_param = kwargs.get("reasoning", getattr(request, "reasoning", None))
        if reasoning_param is None and request.reasoning_effort:
            reasoning_param = {
                "effort": request.reasoning_effort,
                "summary": self.reasoning_summary,
            }
        if reasoning_param is None:
            reasoning_param = self.reasoning
        if reasoning_param:
            # Handle both dict format ({"effort": "low", "summary": "auto"}) and string format ("low")
            if isinstance(reasoning_param, dict):
                # Dict format: use as-is, but apply defaults for missing keys
                params["reasoning"] = {
                    "effort": reasoning_param.get("effort", "medium"),
                    "summary": reasoning_param.get("summary", self.reasoning_summary),
                }
            else:
                # String format: use as effort level with default summary
                params["reasoning"] = {
                    "effort": reasoning_param,
                    "summary": self.reasoning_summary,  # Verbosity: auto|concise|detailed
                }
            logger.info(f"[PROVIDER] Setting reasoning: {params['reasoning']}")

        # CRITICAL: Request encrypted_content for ANY model that may reason, not just when explicitly configured.
        # Models like gpt-5.2-codex and o-series reason by default without "reasoning" appearing in params.
        if not store_enabled:
            model_may_reason = "reasoning" in params or self._model_may_reason(
                model_name
            )
            if model_may_reason:
                params["include"] = kwargs.get(
                    "include", ["reasoning.encrypted_content"]
                )
                logger.debug(
                    "[PROVIDER] Requesting encrypted_content (store=False, model may reason: %s)",
                    model_name,
                )

        # Add tools if provided (from request or kwargs)
        # Native tools (web_search_preview, file_search, code_interpreter) can be passed via kwargs["tools"]
        tools_list = list(request.tools) if request.tools else []
        native_tools = kwargs.get("tools", [])
        logger.info(
            f"[PROVIDER] Tools from request: {len(list(request.tools) if request.tools else [])}, native_tools from kwargs: {native_tools}"
        )
        if native_tools:
            tools_list.extend(native_tools)

        if tools_list:
            params["tools"] = self._convert_tools_from_request(tools_list)
            # Add tool-related parameters per Responses API spec
            params["tool_choice"] = kwargs.get("tool_choice", "auto")
            params["parallel_tool_calls"] = kwargs.get("parallel_tool_calls", True)
            # max_tool_calls limits how many tool calls the model can make
            # Important for deep research to prevent excessive searching that consumes token budget
            if max_tool_calls := kwargs.get("max_tool_calls"):
                params["max_tool_calls"] = max_tool_calls

        # Add truncation parameter for automatic context management
        if self.truncation:
            params["truncation"] = kwargs.get("truncation", self.truncation)

        # Add background mode parameter for long-running requests (deep research)
        if background_mode:
            params["background"] = True

        logger.info(
            f"[PROVIDER] {self.api_label} API call - model: {params['model']}, has_instructions: {bool(instructions)}, tools: {len(tools_list)}, background={background_mode}"
        )

        thinking_enabled = bool(kwargs.get("extended_thinking"))
        thinking_budget = None
        if thinking_enabled:
            if "reasoning" not in params:
                params["reasoning"] = {
                    "effort": kwargs.get("reasoning_effort")
                    or self.config.get("reasoning_effort", "high"),
                    "summary": self.reasoning_summary,  # Verbosity: auto|concise|detailed
                }

            budget_tokens = (
                kwargs.get("thinking_budget_tokens")
                or self.config.get("thinking_budget_tokens")
                or 0
            )
            buffer_tokens = kwargs.get("thinking_budget_buffer") or self.config.get(
                "thinking_budget_buffer", 1024
            )

            if budget_tokens:
                thinking_budget = budget_tokens
                target_tokens = budget_tokens + buffer_tokens
                if params.get("max_output_tokens"):
                    params["max_output_tokens"] = max(
                        params["max_output_tokens"], target_tokens
                    )
                else:
                    params["max_output_tokens"] = target_tokens

            logger.info(
                "[PROVIDER] Extended thinking enabled (effort=%s, budget=%s, buffer=%s)",
                params["reasoning"]["effort"],
                thinking_budget or "default",
                buffer_tokens,
            )

        # Auto-enable reasoning summary for models that reason by default.
        # Without this, models like gpt-5.2-codex return encrypted_content but no
        # summary text, making reasoning invisible for observability/debugging.
        # Placed AFTER extended_thinking so it doesn't interfere with effort-based reasoning.
        if self._model_may_reason(model_name) and "reasoning" not in params:
            params["reasoning"] = {"summary": "auto"}

        # Emit llm:request event
        if self.coordinator and hasattr(self.coordinator, "hooks"):
            # INFO level: Summary only
            await self.coordinator.hooks.emit(
                "llm:request",
                {
                    "provider": self.name,
                    "model": params["model"],
                    "message_count": len(message_list),
                    "has_instructions": bool(instructions),
                    "reasoning_enabled": params.get("reasoning") is not None,
                    "thinking_enabled": thinking_enabled,
                    "thinking_budget": thinking_budget,
                    "background_mode": background_mode,
                },
            )

            # DEBUG level: Full request payload with truncated values (if debug enabled)
            if self.debug:
                await self.coordinator.hooks.emit(
                    "llm:request:debug",
                    {
                        "lvl": "DEBUG",
                        "provider": self.name,
                        "request": self._truncate_values(params),
                    },
                )

            # RAW level: Complete params dict as sent to OpenAI API (if debug AND raw_debug enabled)
            if self.debug and self.raw_debug:
                await self.coordinator.hooks.emit(
                    "llm:request:raw",
                    {
                        "lvl": "DEBUG",
                        "provider": self.name,
                        "params": params,  # Complete untruncated params
                    },
                )

        start_time = time.time()

        # Use appropriate timeout for background mode (deep research can take minutes)
        effective_timeout = self.background_timeout if background_mode else self.timeout
        poll_interval = kwargs.get("poll_interval", self.poll_interval)

        # Call provider API with shared retry_with_backoff from amplifier-core.
        # Error translation happens inside _do_complete() so that retry_with_backoff
        # sees LLMError (and checks retryable) rather than raw SDK exceptions.

        async def _do_complete():
            """Single API call attempt with SDK → kernel error translation."""
            try:
                return await asyncio.wait_for(
                    self.client.responses.create(**params),
                    timeout=effective_timeout,
                )
            except openai.RateLimitError as e:
                retry_after = None
                if hasattr(e, "response") and e.response is not None:
                    ra_header = e.response.headers.get("retry-after")
                    if ra_header:
                        try:
                            retry_after = float(ra_header)
                        except (ValueError, TypeError):
                            pass
                # Fail-fast: if retry_after exceeds max_delay, mark non-retryable
                # so retry_with_backoff raises immediately instead of sleeping.
                retryable = True
                if (
                    retry_after is not None
                    and retry_after > self._retry_config.max_delay
                ):
                    retryable = False
                raise kernel_errors.RateLimitError(
                    str(e),
                    provider=self.name,
                    status_code=429,
                    retryable=retryable,
                    retry_after=retry_after,
                ) from e
            except openai.AuthenticationError as e:
                raise kernel_errors.AuthenticationError(
                    str(e),
                    provider=self.name,
                    status_code=getattr(e, "status_code", 401),
                ) from e
            except openai.BadRequestError as e:
                msg = str(e).lower()
                if (
                    "context length" in msg
                    or "too many tokens" in msg
                    or "maximum context" in msg
                ):
                    raise kernel_errors.ContextLengthError(
                        str(e),
                        provider=self.name,
                        status_code=400,
                    ) from e
                elif "content filter" in msg or "safety" in msg or "blocked" in msg:
                    raise kernel_errors.ContentFilterError(
                        str(e),
                        provider=self.name,
                        status_code=400,
                    ) from e
                else:
                    raise kernel_errors.InvalidRequestError(
                        str(e),
                        provider=self.name,
                        status_code=400,
                    ) from e
            except openai.APIStatusError as e:
                status = getattr(e, "status_code", 500)
                if status == 403:
                    raise kernel_errors.AccessDeniedError(
                        str(e),
                        provider=self.name,
                        status_code=403,
                    ) from e
                if status == 404:
                    raise kernel_errors.NotFoundError(
                        str(e),
                        provider=self.name,
                        status_code=404,
                    ) from e
                if status >= 500:
                    raise kernel_errors.ProviderUnavailableError(
                        str(e),
                        provider=self.name,
                        status_code=status,
                        retryable=True,
                    ) from e
                raise kernel_errors.LLMError(
                    str(e),
                    provider=self.name,
                    status_code=status,
                    retryable=False,
                ) from e
            except asyncio.TimeoutError as e:
                raise kernel_errors.LLMTimeoutError(
                    f"Request timed out after {effective_timeout}s",
                    provider=self.name,
                    retryable=True,
                ) from e
            except kernel_errors.LLMError:
                raise  # Already translated, don't double-wrap
            except Exception as e:
                raise kernel_errors.LLMError(
                    str(e) or f"{type(e).__name__}: (no message)",
                    provider=self.name,
                    retryable=True,
                ) from e

        async def _on_retry(attempt: int, delay: float, error: kernel_errors.LLMError):
            """Callback invoked before each retry sleep."""
            if self.coordinator and hasattr(self.coordinator, "hooks"):
                await self.coordinator.hooks.emit(
                    PROVIDER_RETRY,
                    {
                        "provider": self.name,
                        "attempt": attempt,
                        "max_retries": self._retry_config.max_retries,
                        "delay": delay,
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                    },
                )

        try:
            response = await retry_with_backoff(
                _do_complete,
                self._retry_config,
                on_retry=_on_retry,
            )

            elapsed_ms = int((time.time() - start_time) * 1000)

            logger.info(
                "[PROVIDER] Received response from %s API (status=%s)",
                self.api_label,
                getattr(response, "status", "unknown"),
            )

            # RAW level: Complete response object from OpenAI API (if debug AND raw_debug enabled)
            if (
                self.coordinator
                and hasattr(self.coordinator, "hooks")
                and self.debug
                and self.raw_debug
            ):
                await self.coordinator.hooks.emit(
                    "llm:response:raw",
                    {
                        "lvl": "DEBUG",
                        "provider": self.name,
                        "response": response.model_dump(),  # Pydantic model → dict (complete untruncated)
                    },
                )

            # Handle background mode polling for long-running requests (deep research)
            # Background responses start in queued/in_progress state and need polling until completion
            if background_mode and hasattr(response, "status"):
                poll_count = 0
                response_id = getattr(response, "id", None)

                while response.status in BACKGROUND_POLLING_STATUSES:
                    poll_count += 1
                    current_status = response.status

                    # Check timeout
                    elapsed_total = time.time() - start_time
                    if elapsed_total >= effective_timeout:
                        logger.warning(
                            f"[PROVIDER] Background request timed out after {elapsed_total:.1f}s "
                            f"(status={current_status}, polls={poll_count})"
                        )
                        break

                    # Emit status update event
                    if self.coordinator and hasattr(self.coordinator, "hooks"):
                        await self.coordinator.hooks.emit(
                            "provider:background_status",
                            {
                                "provider": self.name,
                                "response_id": response_id,
                                "status": current_status,
                                "poll_count": poll_count,
                                "elapsed_ms": int((time.time() - start_time) * 1000),
                            },
                        )

                    logger.info(
                        f"[PROVIDER] Background request status: {current_status} "
                        f"(poll {poll_count}, waiting {poll_interval}s)"
                    )

                    # Wait before next poll
                    await asyncio.sleep(poll_interval)

                    # Poll for updated status
                    try:
                        response = await self.client.responses.retrieve(response_id)
                    except Exception as poll_error:
                        logger.error(
                            f"[PROVIDER] Failed to poll background request: {poll_error}"
                        )
                        break

                elapsed_ms = int((time.time() - start_time) * 1000)
                logger.info(
                    f"[PROVIDER] Background request completed: status={response.status}, "
                    f"polls={poll_count}, elapsed={elapsed_ms}ms"
                )

                # Check for failed/cancelled status
                if response.status == BACKGROUND_STATUS_FAILED:
                    error_msg = f"Background request failed after {poll_count} polls"
                    if hasattr(response, "error") and response.error:
                        error_msg = f"{error_msg}: {response.error}"
                    raise RuntimeError(error_msg)

            # Handle incomplete responses via auto-continuation
            # OpenAI Responses API may return status="incomplete" with reason like "max_output_tokens"
            # We automatically continue until complete to provide seamless experience
            accumulated_output = (
                list(response.output) if hasattr(response, "output") else []
            )
            final_response = response
            continuation_count = 0

            while (
                hasattr(final_response, "status")
                and final_response.status == "incomplete"
                and continuation_count < MAX_CONTINUATION_ATTEMPTS
            ):
                continuation_count += 1

                # Extract incomplete reason for logging
                incomplete_reason = "unknown"
                if hasattr(final_response, "incomplete_details"):
                    details = final_response.incomplete_details
                    if isinstance(details, dict):
                        incomplete_reason = details.get("reason", "unknown")
                    elif hasattr(details, "reason"):
                        incomplete_reason = details.reason

                logger.info(
                    f"[PROVIDER] Response incomplete (reason: {incomplete_reason}), "
                    f"auto-continuing with previous_response_id={final_response.id} "
                    f"(continuation {continuation_count}/{MAX_CONTINUATION_ATTEMPTS})"
                )

                # Emit continuation event for observability
                if self.coordinator and hasattr(self.coordinator, "hooks"):
                    await self.coordinator.hooks.emit(
                        "provider:incomplete_continuation",
                        {
                            "provider": self.name,
                            "response_id": final_response.id,
                            "reason": incomplete_reason,
                            "continuation_number": continuation_count,
                            "max_attempts": MAX_CONTINUATION_ATTEMPTS,
                        },
                    )

                # Build continuation params using input-based pattern (stateless-compatible)
                # Instead of previous_response_id (requires store:true), we include the
                # accumulated output in the input to preserve context
                continuation_input = self._build_continuation_input(
                    input_messages, accumulated_output
                )

                continue_params = {
                    "model": params["model"],
                    "input": continuation_input,
                }

                # Inherit important params if they were set
                if "instructions" in params:
                    continue_params["instructions"] = params["instructions"]
                if "max_output_tokens" in params:
                    continue_params["max_output_tokens"] = params["max_output_tokens"]
                if "temperature" in params:
                    continue_params["temperature"] = params["temperature"]
                if "reasoning" in params:
                    continue_params["reasoning"] = params["reasoning"]
                if "include" in params:
                    continue_params["include"] = params["include"]
                if "tools" in params:
                    continue_params["tools"] = params["tools"]
                    continue_params["tool_choice"] = params.get("tool_choice", "auto")
                    continue_params["parallel_tool_calls"] = params.get(
                        "parallel_tool_calls", True
                    )
                if "store" in params:
                    continue_params["store"] = params["store"]

                # Make continuation call
                try:
                    continue_start = time.time()
                    final_response = await asyncio.wait_for(
                        self.client.responses.create(**continue_params),
                        timeout=self.timeout,
                    )
                    continue_elapsed = int((time.time() - continue_start) * 1000)
                    elapsed_ms += continue_elapsed

                    # Accumulate output from continuation
                    if hasattr(final_response, "output"):
                        accumulated_output.extend(final_response.output)

                    # Emit raw debug for continuation if enabled
                    if (
                        self.coordinator
                        and hasattr(self.coordinator, "hooks")
                        and self.debug
                        and self.raw_debug
                    ):
                        await self.coordinator.hooks.emit(
                            "llm:response:raw",
                            {
                                "lvl": "DEBUG",
                                "provider": self.name,
                                "response": final_response.model_dump(),
                                "continuation": continuation_count,
                            },
                        )

                except Exception as e:
                    logger.error(
                        f"[PROVIDER] Continuation call {continuation_count} failed: {e}. "
                        f"Returning partial response from {continuation_count} continuation(s)"
                    )
                    break  # Return what we have so far

            # Log completion summary
            if continuation_count > 0:
                final_status = getattr(final_response, "status", "unknown")
                logger.info(
                    f"[PROVIDER] Completed after {continuation_count} continuation(s), "
                    f"final status: {final_status}, total time: {elapsed_ms}ms"
                )

            # Use the final response and accumulated output for conversion
            response = final_response

            # Extract usage counts
            usage_obj = response.usage if hasattr(response, "usage") else None
            usage_counts = {"input": 0, "output": 0, "total": 0}
            if usage_obj:
                if hasattr(usage_obj, "input_tokens"):
                    usage_counts["input"] = usage_obj.input_tokens
                if hasattr(usage_obj, "output_tokens"):
                    usage_counts["output"] = usage_obj.output_tokens
                usage_counts["total"] = usage_counts["input"] + usage_counts["output"]

            # Emit llm:response event
            if self.coordinator and hasattr(self.coordinator, "hooks"):
                # INFO level: Summary only
                await self.coordinator.hooks.emit(
                    "llm:response",
                    {
                        "provider": self.name,
                        "model": params["model"],
                        "usage": {
                            "input": usage_counts["input"],
                            "output": usage_counts["output"],
                        },
                        "status": "ok",
                        "duration_ms": elapsed_ms,
                        "continuation_count": continuation_count
                        if continuation_count > 0
                        else None,
                    },
                )

                # DEBUG level: Full response with truncated values (if debug enabled)
                if self.debug:
                    response_dict = response.model_dump()  # Pydantic model → dict
                    await self.coordinator.hooks.emit(
                        "llm:response:debug",
                        {
                            "lvl": "DEBUG",
                            "provider": self.name,
                            "response": self._truncate_values(response_dict),
                            "status": "ok",
                            "duration_ms": elapsed_ms,
                            "continuation_count": continuation_count
                            if continuation_count > 0
                            else None,
                        },
                    )

            # Convert to ChatResponse with accumulated output
            # If there were continuations, use the accumulated output; otherwise use response.output directly
            if continuation_count > 0:
                # Use new helper for accumulated output
                return convert_response_with_accumulated_output(
                    response, accumulated_output, continuation_count, OpenAIChatResponse
                )
            # Use existing conversion for normal (non-continued) responses
            return self._convert_to_chat_response(response)

        except kernel_errors.LLMError as e:
            # Phase 2: Kernel error types — emit llm:response error event, then propagate
            elapsed_ms = int((time.time() - start_time) * 1000)
            error_msg = str(e) or f"{type(e).__name__}: (no message)"
            logger.error("[PROVIDER] %s API error: %s", self.api_label, error_msg)

            if self.coordinator and hasattr(self.coordinator, "hooks"):
                await self.coordinator.hooks.emit(
                    "llm:response",
                    {
                        "status": "error",
                        "duration_ms": elapsed_ms,
                        "error": error_msg,
                        "provider": self.name,
                        "model": params["model"],
                    },
                )
            raise

        except Exception as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            # Ensure error message is never empty
            error_msg = str(e) or f"{type(e).__name__}: (no message)"
            logger.error("[PROVIDER] %s API error: %s", self.api_label, error_msg)

            # Emit error event
            if self.coordinator and hasattr(self.coordinator, "hooks"):
                await self.coordinator.hooks.emit(
                    "llm:response",
                    {
                        "status": "error",
                        "duration_ms": elapsed_ms,
                        "error": error_msg,
                        "provider": self.name,
                        "model": params["model"],
                    },
                )
            # Re-raise with meaningful message if original was empty
            if not str(e):
                raise type(e)(error_msg) from e
            raise

    def _convert_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert messages to OpenAI Responses API format.

        Handles:
        - User messages: Simple text content
        - Assistant messages: Reconstructs with tool calls if present
        - Tool messages: Converts to appropriate format

        Args:
            messages: List of message dicts from ChatRequest

        Returns:
            List of OpenAI-formatted message objects per Responses API spec
        """
        openai_messages = []
        i = 0

        while i < len(messages):
            msg = messages[i]
            role = msg.get("role")
            content = msg.get("content", "")

            # Skip system messages (handled via instructions parameter)
            if role == "system":
                i += 1
                continue

            # Handle tool result messages - use native function_call_output format
            if role == "tool":
                while i < len(messages) and messages[i].get("role") == "tool":
                    tool_msg = messages[i]
                    tool_call_id = tool_msg.get("tool_call_id")
                    tool_content = tool_msg.get("content", "")
                    tool_name = tool_msg.get("tool_name", "unknown")

                    if tool_call_id:
                        output_str = (
                            tool_content
                            if isinstance(tool_content, str)
                            else json.dumps(tool_content)
                        )
                        # Use apply_patch_call_output for native apply_patch calls
                        if tool_call_id in self._native_call_ids:
                            # Determine status: "failed" if content signals error, else "completed"
                            _patch_status = "completed"
                            if (
                                isinstance(tool_content, dict)
                                and tool_content.get("success") is False
                            ):
                                _patch_status = "failed"
                            elif isinstance(tool_content, str):
                                try:
                                    _parsed = json.loads(tool_content)
                                    if (
                                        isinstance(_parsed, dict)
                                        and _parsed.get("success") is False
                                    ):
                                        _patch_status = "failed"
                                except (json.JSONDecodeError, TypeError):
                                    # Not JSON — infer status from output format.
                                    # apply_patch success = git-style status lines
                                    # ("M file.py", "A new.py", "D old.py", "R a -> b").
                                    # Any other non-empty string is an error message.
                                    _first = (
                                        tool_content.split("\n", 1)[0]
                                        if tool_content
                                        else ""
                                    )
                                    if _first and _first[:2] not in (
                                        "M ",
                                        "A ",
                                        "D ",
                                        "R ",
                                    ):
                                        _patch_status = "failed"

                            openai_messages.append(
                                {
                                    "type": "apply_patch_call_output",
                                    "call_id": tool_call_id,
                                    "output": output_str,
                                    "status": _patch_status,
                                }
                            )
                        else:
                            # Standard function_call_output format
                            # Per OpenAI Responses API spec (see ai_context/openai-api-guide.txt)
                            openai_messages.append(
                                {
                                    "type": "function_call_output",
                                    "call_id": tool_call_id,
                                    "output": output_str,
                                }
                            )
                    else:
                        # Fallback for messages without tool_call_id (legacy/compacted messages)
                        logger.warning(
                            f"Tool result missing tool_call_id for '{tool_name}', using text fallback. "
                            "This may reduce model accuracy for multi-tool scenarios."
                        )
                        openai_messages.append(
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "input_text",
                                        "text": f"[Tool: {tool_name}]\n{tool_content}",
                                    }
                                ],
                            }
                        )
                    i += 1
                continue

            # Handle assistant messages
            if role == "assistant":
                assistant_content = []
                reasoning_items_to_add = []  # Top-level reasoning items (not in message content)
                function_call_items = []  # function_call items to add as top-level
                metadata = msg.get("metadata", {})

                # Handle tool_calls field (from context storage, Anthropic-style)
                tool_calls_field = msg.get("tool_calls", [])
                for tc in tool_calls_field:
                    tc_id = tc.get("id") or tc.get("tool_call_id", "")
                    tc_name = tc.get("name", "")
                    tc_args = tc.get("arguments") or tc.get("input", {})
                    if isinstance(tc_args, str):
                        tc_args_str = tc_args
                    else:
                        tc_args_str = json.dumps(tc_args) if tc_args else "{}"
                    if tc_id and tc_name:
                        function_call_items.append(
                            {
                                "type": "function_call",
                                "call_id": tc_id,
                                "name": tc_name,
                                "arguments": tc_args_str,
                            }
                        )

                # Handle structured content (list of blocks)
                if isinstance(content, list):
                    for block in content:
                        # Handle dict blocks (from context storage)
                        if isinstance(block, dict):
                            block_type = block.get("type")
                            if block_type == "text":
                                assistant_content.append(
                                    {
                                        "type": "output_text",
                                        "text": block.get("text", ""),
                                    }
                                )
                            elif block_type == "tool_call":
                                # Convert tool_call block to function_call item
                                tc_id = block.get("id", "")
                                tc_name = block.get("name", "")
                                tc_input = block.get("input", {})
                                if isinstance(tc_input, str):
                                    tc_args_str = tc_input
                                else:
                                    tc_args_str = (
                                        json.dumps(tc_input) if tc_input else "{}"
                                    )
                                if tc_id and tc_name:
                                    function_call_items.append(
                                        {
                                            "type": "function_call",
                                            "call_id": tc_id,
                                            "name": tc_name,
                                            "arguments": tc_args_str,
                                        }
                                    )
                            elif block_type == "thinking":
                                # Extract reasoning state for top-level insertion
                                # Reasoning items must be top-level in input, not in message content!
                                block_content = block.get("content")
                                if block_content and len(block_content) >= 2:
                                    encrypted_content = block_content[0]
                                    reasoning_id = block_content[1]
                                    if reasoning_id:
                                        reasoning_item = {
                                            "type": "reasoning",
                                            "id": reasoning_id,
                                        }
                                        if encrypted_content:
                                            reasoning_item["encrypted_content"] = (
                                                encrypted_content
                                            )
                                        # Always include summary (required by OpenAI API).
                                        # Use thinking text when available, empty list otherwise.
                                        thinking_text = block.get("thinking")
                                        reasoning_item["summary"] = (
                                            [
                                                {
                                                    "type": "summary_text",
                                                    "text": thinking_text,
                                                }
                                            ]
                                            if thinking_text
                                            else []
                                        )
                                        reasoning_items_to_add.append(reasoning_item)
                        elif hasattr(block, "type"):
                            # Handle ContentBlock objects (TextBlock, ThinkingBlock, ToolCallBlock, etc.)
                            if block.type == "text":
                                assistant_content.append(
                                    {"type": "output_text", "text": block.text}
                                )
                            elif block.type == "tool_call":
                                # Convert ToolCallBlock to function_call item
                                tc_id = getattr(block, "id", "")
                                tc_name = getattr(block, "name", "")
                                tc_input = getattr(block, "input", {})
                                if isinstance(tc_input, str):
                                    tc_args_str = tc_input
                                else:
                                    tc_args_str = (
                                        json.dumps(tc_input) if tc_input else "{}"
                                    )
                                if tc_id and tc_name:
                                    function_call_items.append(
                                        {
                                            "type": "function_call",
                                            "call_id": tc_id,
                                            "name": tc_name,
                                            "arguments": tc_args_str,
                                        }
                                    )
                            elif (
                                block.type == "thinking"
                                and hasattr(block, "content")
                                and block.content
                                and len(block.content) >= 2
                            ):
                                # Extract reasoning state for top-level insertion
                                # Reasoning items must be top-level in input, not in message content!
                                encrypted_content = block.content[0]
                                reasoning_id = block.content[1]

                                if (
                                    reasoning_id
                                ):  # Only include if we have a reasoning ID
                                    reasoning_item = {
                                        "type": "reasoning",
                                        "id": reasoning_id,
                                    }

                                    # Add encrypted content if available
                                    if encrypted_content:
                                        reasoning_item["encrypted_content"] = (
                                            encrypted_content
                                        )

                                    # Always include summary (required by OpenAI API).
                                    # Use thinking text when available, empty list otherwise.
                                    thinking_text = (
                                        getattr(block, "thinking", None)
                                        if hasattr(block, "thinking")
                                        else None
                                    )
                                    reasoning_item["summary"] = (
                                        [
                                            {
                                                "type": "summary_text",
                                                "text": thinking_text,
                                            }
                                        ]
                                        if thinking_text
                                        else []
                                    )

                                    reasoning_items_to_add.append(reasoning_item)

                # Handle simple string content
                elif isinstance(content, str) and content:
                    assistant_content.append({"type": "output_text", "text": content})

                # Defensive: strip orphaned reasoning items that have no encrypted_content.
                # These occur when the model reasoned but include=[reasoning.encrypted_content]
                # was not requested — the reasoning ID exists but can't be sent back, causing 404s.
                if metadata and metadata.get(METADATA_REASONING_ITEMS):
                    has_usable_reasoning = any(
                        isinstance(item, dict)
                        and item.get("type") == "reasoning"
                        and item.get("encrypted_content")
                        for item in reasoning_items_to_add
                    )
                    if not has_usable_reasoning:
                        logger.warning(
                            "[PROVIDER] Reasoning IDs in metadata but encrypted_content unavailable. "
                            "Stripping orphaned reasoning references to prevent API errors. "
                            "Ensure include=[reasoning.encrypted_content] is requested for store=false."
                        )
                        # Strip orphaned reasoning items that would cause 404 errors
                        reasoning_items_to_add.clear()

                # Add reasoning items as TOP-LEVEL entries (before assistant message)
                # Per OpenAI Responses API: reasoning items must be top-level, not in message content
                for reasoning_item in reasoning_items_to_add:
                    openai_messages.append(reasoning_item)

                # Only add assistant message if there's content
                if assistant_content:
                    openai_messages.append(
                        {"role": "assistant", "content": assistant_content}
                    )

                # Add function_call items as TOP-LEVEL entries (after assistant message)
                # Per OpenAI Responses API: function_call items are separate from message content
                for fc_item in function_call_items:
                    openai_messages.append(fc_item)

                i += 1

            # Handle developer messages as XML-wrapped user messages
            elif role == "developer":
                wrapped = f"<context_file>\n{content}\n</context_file>"
                openai_messages.append(
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": wrapped}],
                    }
                )
                i += 1

            # Handle user messages
            elif role == "user":
                # Handle structured content (list of blocks including text and images)
                if isinstance(content, list):
                    content_items = []
                    for block in content:
                        if isinstance(block, dict):
                            block_type = block.get("type")
                            if block_type == "text":
                                content_items.append(
                                    {
                                        "type": "input_text",
                                        "text": block.get("text", ""),
                                    }
                                )
                            elif block_type == "image":
                                # Convert ImageBlock to OpenAI Responses API input_image format
                                source = block.get("source", {})
                                if source.get("type") == "base64":
                                    # OpenAI uses data URI format: data:image/jpeg;base64,{data}
                                    media_type = source.get("media_type", "image/jpeg")
                                    data = source.get("data", "")
                                    content_items.append(
                                        {
                                            "type": "input_image",
                                            "image_url": f"data:{media_type};base64,{data}",
                                        }
                                    )
                                else:
                                    logger.warning(
                                        f"Unsupported image source type: {source.get('type')}"
                                    )

                    if content_items:
                        openai_messages.append(
                            {"role": "user", "content": content_items}
                        )
                else:
                    # Simple string content
                    openai_messages.append(
                        {
                            "role": "user",
                            "content": [{"type": "input_text", "text": content}],
                        }
                    )
                i += 1
            else:
                # Unknown role - skip
                logger.warning(f"Unknown message role: {role}")
                i += 1

        return openai_messages

    def _convert_tools_from_request(self, tools: list) -> list[dict[str, Any]]:
        """Convert ToolSpec objects from ChatRequest to OpenAI format.

        Handles both user-defined function tools and native OpenAI-hosted tools
        (web_search_preview, file_search, code_interpreter).

        Native tools are passed through directly when specified as dicts with
        a recognized 'type' field. User-defined tools are converted to function
        tool format.

        Args:
            tools: List of ToolSpec objects or native tool dicts

        Returns:
            List of OpenAI-formatted tool definitions
        """
        openai_tools = []

        # Lazy detection of native apply_patch engine via coordinator capability.
        # Once detected, the flag persists — no repeated lookups.
        if not self._apply_patch_native:
            engine = self.coordinator.get_capability("apply_patch.engine")
            if engine == "native":
                self._apply_patch_native = True

        for tool in tools:
            # Check if this is a native OpenAI tool (dict with recognized type)
            if isinstance(tool, dict):
                tool_type = tool.get("type", "")
                if tool_type in NATIVE_TOOL_TYPES:
                    # Pass through native tools directly (web_search_preview, file_search, code_interpreter)
                    openai_tools.append(tool)
                    continue
                # Fall through to handle as function tool if type is "function" or unrecognized

            # Handle ToolSpec objects (user-defined function tools)
            if hasattr(tool, "name"):
                # Special handling for apply_patch with native engine
                if tool.name == "apply_patch" and self._apply_patch_native:
                    openai_tools.append({"type": "apply_patch"})
                    continue

                openai_tools.append(
                    {
                        "type": "function",
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": tool.parameters,
                    }
                )
            elif isinstance(tool, dict) and "name" in tool:
                # Handle dict-format function tool
                openai_tools.append(
                    {
                        "type": "function",
                        "name": tool.get("name", ""),
                        "description": tool.get("description", ""),
                        "parameters": tool.get("parameters", {}),
                    }
                )

        return openai_tools

    def _convert_to_chat_response(self, response: Any) -> ChatResponse:
        """Convert OpenAI response to ChatResponse format.

        Args:
            response: OpenAI API response

        Returns:
            ChatResponse with content blocks
        """
        from amplifier_core.message_models import (
            ReasoningBlock as ResponseReasoningBlock,
        )
        from amplifier_core.message_models import TextBlock
        from amplifier_core.message_models import ThinkingBlock
        from amplifier_core.message_models import ToolCall
        from amplifier_core.message_models import ToolCallBlock
        from amplifier_core.message_models import Usage

        content_blocks = []
        tool_calls = []
        event_blocks: list[TextContent | ThinkingContent | ToolCallContent] = []
        text_accumulator: list[str] = []
        reasoning_item_ids: list[str] = []  # Track reasoning IDs for metadata

        # Parse output blocks
        for block in response.output:
            # Handle both SDK objects and dictionaries
            if hasattr(block, "type"):
                block_type = block.type

                if block_type == "message":
                    # Extract text from message content
                    block_content = getattr(block, "content", [])
                    if isinstance(block_content, list):
                        for content_item in block_content:
                            if (
                                hasattr(content_item, "type")
                                and content_item.type == "output_text"
                            ):
                                text = getattr(content_item, "text", "")
                                content_blocks.append(TextBlock(text=text))
                                text_accumulator.append(text)
                                event_blocks.append(
                                    TextContent(
                                        text=text,
                                        raw=getattr(content_item, "raw", None),
                                    )
                                )
                    elif isinstance(block_content, str):
                        content_blocks.append(TextBlock(text=block_content))
                        text_accumulator.append(block_content)
                        event_blocks.append(TextContent(text=block_content))

                elif block_type == "reasoning":
                    # Extract reasoning ID and encrypted content for state preservation
                    reasoning_id = getattr(block, "id", None)
                    encrypted_content = getattr(block, "encrypted_content", None)

                    # Track reasoning item ID for metadata (backward compat)
                    if reasoning_id:
                        reasoning_item_ids.append(reasoning_id)

                    # Extract reasoning summary if available
                    reasoning_summary = getattr(block, "summary", None) or getattr(
                        block, "text", None
                    )

                    # Use helper to extract reasoning text
                    reasoning_text = extract_reasoning_text(reasoning_summary)

                    # Fallback to original logic if helper didn't find text
                    if reasoning_text is None and isinstance(reasoning_summary, list):
                        # Extract text from list of summary objects (dict or Pydantic models)
                        texts = []
                        for item in reasoning_summary:
                            if isinstance(item, dict):
                                texts.append(item.get("text", ""))
                            elif hasattr(item, "text"):
                                texts.append(getattr(item, "text", ""))
                            elif isinstance(item, str):
                                texts.append(item)
                        reasoning_text = "\n".join(filter(None, texts))
                    elif isinstance(reasoning_summary, str):
                        reasoning_text = reasoning_summary
                    elif isinstance(reasoning_summary, dict):
                        reasoning_text = reasoning_summary.get(
                            "text", str(reasoning_summary)
                        )
                    elif hasattr(reasoning_summary, "text"):
                        reasoning_text = getattr(
                            reasoning_summary, "text", str(reasoning_summary)
                        )

                    # Create thinking block if there's reasoning text OR encrypted state to preserve
                    if reasoning_text or encrypted_content:
                        # Store reasoning state in content field for re-insertion
                        # content[0] = encrypted_content (for full reasoning continuity)
                        # content[1] = reasoning_id (rs_* ID for OpenAI)
                        thinking_block = ThinkingBlock(
                            thinking=reasoning_text
                            or "",  # May be empty when only encrypted_content exists
                            signature=None,
                            visibility="internal",
                            content=[encrypted_content, reasoning_id],
                        )
                        logger.info(
                            f"[PROVIDER] Created ThinkingBlock: id={reasoning_id}, "
                            f"has_encrypted={encrypted_content is not None}, "
                            f"enc_len={len(encrypted_content) if encrypted_content else 0}"
                        )
                        content_blocks.append(thinking_block)
                        event_blocks.append(ThinkingContent(text=reasoning_text or ""))
                        # NOTE: Do NOT add reasoning to text_accumulator - it's internal process, not response content

                elif block_type in {"tool_call", "function_call"}:
                    tool_id = getattr(block, "id", "") or getattr(block, "call_id", "")
                    tool_name = getattr(block, "name", "")
                    tool_input = getattr(block, "input", None)
                    if tool_input is None and hasattr(block, "arguments"):
                        tool_input = block.arguments
                    if isinstance(tool_input, str):
                        try:
                            tool_input = json.loads(tool_input)
                        except json.JSONDecodeError:
                            logger.debug(
                                "Failed to decode tool call arguments: %s", tool_input
                            )
                    if tool_input is None:
                        tool_input = {}
                    # Ensure tool_input is dict after json.loads or default
                    if not isinstance(tool_input, dict):
                        tool_input = {}
                    content_blocks.append(
                        ToolCallBlock(id=tool_id, name=tool_name, input=tool_input)
                    )
                    tool_calls.append(
                        ToolCall(id=tool_id, name=tool_name, arguments=tool_input)
                    )

                elif block_type == "apply_patch_call":
                    call_id = getattr(block, "call_id", "")
                    operation = block.operation
                    args = {
                        "type": getattr(operation, "type", ""),
                        "path": getattr(operation, "path", ""),
                        "diff": getattr(operation, "diff", ""),
                    }
                    content_blocks.append(
                        ToolCallBlock(id=call_id, name="apply_patch", input=args)
                    )
                    tool_calls.append(
                        ToolCall(id=call_id, name="apply_patch", arguments=args)
                    )
                    # Track for round-trip output format
                    self._native_call_ids.add(call_id)

            else:
                # Dictionary format
                block_type = block.get("type")

                if block_type == "message":
                    block_content = block.get("content", [])
                    if isinstance(block_content, list):
                        for content_item in block_content:
                            if content_item.get("type") == "output_text":
                                text = content_item.get("text", "")
                                content_blocks.append(TextBlock(text=text))
                                text_accumulator.append(text)
                                event_blocks.append(
                                    TextContent(text=text, raw=content_item)
                                )
                    elif isinstance(block_content, str):
                        content_blocks.append(TextBlock(text=block_content))
                        text_accumulator.append(block_content)
                        event_blocks.append(TextContent(text=block_content, raw=block))

                elif block_type == "reasoning":
                    # Extract reasoning ID and encrypted content for state preservation
                    reasoning_id = block.get("id")
                    encrypted_content = block.get("encrypted_content")

                    # Track reasoning item ID for metadata (backward compat)
                    if reasoning_id:
                        reasoning_item_ids.append(reasoning_id)

                    # Extract reasoning summary if available
                    reasoning_summary = block.get("summary") or block.get("text")

                    # Use helper to extract reasoning text
                    reasoning_text = extract_reasoning_text(reasoning_summary)

                    # Fallback to original logic if helper didn't find text
                    if reasoning_text is None and isinstance(reasoning_summary, list):
                        # Extract text from list of summary objects (dict or Pydantic models)
                        texts = []
                        for item in reasoning_summary:
                            if isinstance(item, dict):
                                texts.append(item.get("text", ""))
                            elif hasattr(item, "text"):
                                texts.append(getattr(item, "text", ""))
                            elif isinstance(item, str):
                                texts.append(item)
                        reasoning_text = "\n".join(filter(None, texts))
                    elif isinstance(reasoning_summary, str):
                        reasoning_text = reasoning_summary
                    elif isinstance(reasoning_summary, dict):
                        reasoning_text = reasoning_summary.get(
                            "text", str(reasoning_summary)
                        )
                    elif hasattr(reasoning_summary, "text"):
                        reasoning_text = getattr(
                            reasoning_summary, "text", str(reasoning_summary)
                        )

                    # Create thinking block if there's reasoning text OR encrypted state to preserve
                    if reasoning_text or encrypted_content:
                        # Store reasoning state in content field for re-insertion
                        # content[0] = encrypted_content (for full reasoning continuity)
                        # content[1] = reasoning_id (rs_* ID for OpenAI)
                        thinking_block = ThinkingBlock(
                            thinking=reasoning_text
                            or "",  # May be empty when only encrypted_content exists
                            signature=None,
                            visibility="internal",
                            content=[encrypted_content, reasoning_id],
                        )
                        logger.info(
                            f"[PROVIDER] Created ThinkingBlock: id={reasoning_id}, "
                            f"has_encrypted={encrypted_content is not None}, "
                            f"enc_len={len(encrypted_content) if encrypted_content else 0}"
                        )
                        content_blocks.append(thinking_block)
                        event_blocks.append(ThinkingContent(text=reasoning_text or ""))
                        # NOTE: Do NOT add reasoning to text_accumulator - it's internal process, not response content

                elif block_type in {"tool_call", "function_call"}:
                    tool_id = block.get("id") or block.get("call_id", "")
                    tool_name = block.get("name", "")
                    tool_input = block.get("input")
                    if tool_input is None:
                        tool_input = block.get("arguments", {})
                    if isinstance(tool_input, str):
                        try:
                            tool_input = json.loads(tool_input)
                        except json.JSONDecodeError:
                            logger.debug(
                                "Failed to decode tool call arguments: %s", tool_input
                            )
                    if tool_input is None:
                        tool_input = {}
                    # Ensure tool_input is dict after json.loads or default
                    if not isinstance(tool_input, dict):
                        tool_input = {}
                    content_blocks.append(
                        ToolCallBlock(id=tool_id, name=tool_name, input=tool_input)
                    )
                    tool_calls.append(
                        ToolCall(id=tool_id, name=tool_name, arguments=tool_input)
                    )
                    event_blocks.append(
                        ToolCallContent(
                            id=tool_id, name=tool_name, arguments=tool_input, raw=block
                        )
                    )

                elif block_type == "apply_patch_call":
                    call_id = block.get("call_id", "")
                    operation = block.get("operation", {})
                    args = {
                        "type": operation.get("type", ""),
                        "path": operation.get("path", ""),
                        "diff": operation.get("diff", ""),
                    }
                    content_blocks.append(
                        ToolCallBlock(id=call_id, name="apply_patch", input=args)
                    )
                    tool_calls.append(
                        ToolCall(id=call_id, name="apply_patch", arguments=args)
                    )
                    self._native_call_ids.add(call_id)

        # Extract usage counts
        usage_obj = response.usage if hasattr(response, "usage") else None
        usage_counts = {"input": 0, "output": 0, "total": 0}
        if usage_obj:
            if hasattr(usage_obj, "input_tokens"):
                usage_counts["input"] = usage_obj.input_tokens
            if hasattr(usage_obj, "output_tokens"):
                usage_counts["output"] = usage_obj.output_tokens
            usage_counts["total"] = usage_counts["input"] + usage_counts["output"]

        # Phase 2: Extract reasoning_tokens from output_tokens_details
        reasoning_tokens = None
        if usage_obj and hasattr(usage_obj, "output_tokens_details"):
            details = usage_obj.output_tokens_details
            if details and hasattr(details, "reasoning_tokens"):
                reasoning_tokens = details.reasoning_tokens

        # Extract cache_read_tokens from input_tokens_details
        cache_read_tokens = None
        if usage_obj and hasattr(usage_obj, "input_tokens_details"):
            details = usage_obj.input_tokens_details
            if details and hasattr(details, "cached_tokens"):
                cache_read_tokens = details.cached_tokens  # 0 is a valid measurement

        usage = Usage(
            input_tokens=usage_counts["input"],
            output_tokens=usage_counts["output"],
            total_tokens=usage_counts["total"],
            reasoning_tokens=reasoning_tokens,
            cache_read_tokens=cache_read_tokens,
        )

        combined_text = "\n\n".join(text_accumulator).strip()

        # Per OpenAI docs: "response.output_text is the safest way to retrieve the final answer"
        # Extract it directly from the response if available
        raw_output_text = getattr(response, "output_text", None)

        # Build metadata with provider-specific state
        metadata = {}

        # Response ID (for next turn's previous_response_id)
        if hasattr(response, "id"):
            metadata[METADATA_RESPONSE_ID] = response.id

        # Status (completed/incomplete)
        if hasattr(response, "status"):
            metadata[METADATA_STATUS] = response.status

            # If incomplete, record the reason
            if response.status == "incomplete":
                incomplete_details = getattr(response, "incomplete_details", None)
                if incomplete_details:
                    if isinstance(incomplete_details, dict):
                        metadata[METADATA_INCOMPLETE_REASON] = incomplete_details.get(
                            "reason"
                        )
                    elif hasattr(incomplete_details, "reason"):
                        metadata[METADATA_INCOMPLETE_REASON] = incomplete_details.reason

        # Reasoning item IDs (for explicit passing if needed)
        if reasoning_item_ids:
            metadata[METADATA_REASONING_ITEMS] = reasoning_item_ids

        # DEBUG: Log what we're returning
        logger.info(
            f"[PROVIDER] Returning ChatResponse with {len(content_blocks)} content blocks"
        )
        for i, block in enumerate(content_blocks):
            block_type = block.type if hasattr(block, "type") else "unknown"
            has_content = hasattr(block, "content") and block.content is not None
            logger.info(
                f"[PROVIDER]   Block {i}: type={block_type}, has_content_field={has_content}"
            )

        chat_response = OpenAIChatResponse(
            content=content_blocks,
            tool_calls=tool_calls if tool_calls else None,
            usage=usage,
            finish_reason=getattr(response, "finish_reason", None),
            content_blocks=event_blocks if event_blocks else None,
            text=combined_text or None,
            output_text=raw_output_text,  # Per OpenAI docs: safest way to get final answer
            metadata=metadata if metadata else None,
        )

        return chat_response
