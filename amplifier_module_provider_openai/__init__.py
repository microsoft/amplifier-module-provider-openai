"""
OpenAI provider module for Amplifier.
Integrates with OpenAI's Responses API.
"""

__all__ = ["mount", "OpenAIProvider"]

import asyncio
import json
import logging
import os
import time
from typing import Any
from typing import cast

from amplifier_core import ModuleCoordinator
from amplifier_core.content_models import TextContent
from amplifier_core.content_models import ThinkingContent
from amplifier_core.content_models import ToolCallContent
from amplifier_core.message_models import ChatRequest
from amplifier_core.message_models import ChatResponse
from amplifier_core.message_models import ToolCall
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class OpenAIChatResponse(ChatResponse):
    """ChatResponse with additional fields for streaming UI compatibility."""

    content_blocks: list[TextContent | ThinkingContent | ToolCallContent] | None = None
    text: str | None = None


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
        """Initialize OpenAI provider with Responses API client."""
        if client is None:
            if api_key is None:
                raise ValueError("api_key or client must be provided")
            self.client = AsyncOpenAI(api_key=api_key)
        else:
            self.client = client
            if api_key is None:
                api_key = "injected-client"
        self.config = config or {}
        self.coordinator = coordinator

        # Configuration with sensible defaults
        self.default_model = self.config.get("default_model", "gpt-5-codex")
        self.max_tokens = self.config.get("max_tokens", 4096)
        self.temperature = self.config.get("temperature", None)  # None = not sent (some models don't support it)
        self.reasoning = self.config.get("reasoning", None)  # None = not sent (minimal|low|medium|high)
        self.enable_state = self.config.get("enable_state", False)
        self.debug = self.config.get("debug", False)  # Enable full request/response logging
        self.raw_debug = self.config.get("raw_debug", False)  # Enable ultra-verbose raw API I/O logging
        self.timeout = self.config.get("timeout", 300.0)  # API timeout in seconds (default 5 minutes)

        # Provider priority for selection (lower = higher priority)
        self.priority = self.config.get("priority", 100)

    def _find_missing_tool_results(self, messages: list) -> list[tuple[str, str, dict]]:
        """Find tool calls without matching results.

        Scans conversation for assistant tool calls and validates each has
        a corresponding tool result message. Returns missing pairs.

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
            elif msg.role == "tool" and hasattr(msg, "tool_call_id") and msg.tool_call_id:
                tool_results.add(msg.tool_call_id)

        return [(call_id, name, args) for call_id, (name, args) in tool_calls.items() if call_id not in tool_results]

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

            # Inject synthetic results
            for call_id, tool_name, _ in missing:
                synthetic = self._create_synthetic_result(call_id, tool_name)
                request.messages.append(synthetic)

            # Emit observability event
            if self.coordinator and hasattr(self.coordinator, "hooks"):
                await self.coordinator.hooks.emit(
                    "provider:tool_sequence_repaired",
                    {
                        "provider": self.name,
                        "repair_count": len(missing),
                        "repairs": [
                            {"tool_call_id": call_id, "tool_name": tool_name} for call_id, tool_name, _ in missing
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

    async def _complete_chat_request(self, request: ChatRequest, **kwargs) -> ChatResponse:
        """Handle ChatRequest format with developer message conversion.

        Args:
            request: ChatRequest with messages
            **kwargs: Additional parameters

        Returns:
            ChatResponse with content blocks
        """
        logger.info(f"[PROVIDER] Received ChatRequest with {len(request.messages)} messages")
        logger.info(f"[PROVIDER] Message roles: {[m.role for m in request.messages]}")

        message_list = list(request.messages)

        # Separate messages by role
        system_msgs = [m for m in message_list if m.role == "system"]
        developer_msgs = [m for m in message_list if m.role == "developer"]
        conversation = [m for m in message_list if m.role in ("user", "assistant", "tool")]

        logger.info(
            f"[PROVIDER] Separated: {len(system_msgs)} system, {len(developer_msgs)} developer, {len(conversation)} conversation"
        )

        # Combine system messages as instructions
        instructions = (
            "\n\n".join(m.content if isinstance(m.content, str) else "" for m in system_msgs) if system_msgs else None
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

        # Prepare request parameters per Responses API spec
        params = {
            "model": kwargs.get("model", self.default_model),
            "input": input_messages,  # Array of message objects, not text string
        }

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

        reasoning_effort = kwargs.get("reasoning", getattr(request, "reasoning", None)) or self.reasoning
        if reasoning_effort:
            params["reasoning"] = {"effort": reasoning_effort}
            # Request reasoning content if available
            params["include"] = kwargs.get("include", ["reasoning.encrypted_content"])

        # Add tools if provided
        if request.tools:
            params["tools"] = self._convert_tools_from_request(request.tools)
            # Add tool-related parameters per Responses API spec
            params["tool_choice"] = kwargs.get("tool_choice", "auto")
            params["parallel_tool_calls"] = kwargs.get("parallel_tool_calls", True)

        # Add store parameter (required for some providers like Azure)
        params["store"] = kwargs.get("store", False)

        logger.info(
            f"[PROVIDER] {self.api_label} API call - model: {params['model']}, has_instructions: {bool(instructions)}, tools: {len(request.tools) if request.tools else 0}"
        )

        thinking_enabled = bool(kwargs.get("extended_thinking"))
        thinking_budget = None
        if thinking_enabled:
            if "reasoning" not in params:
                params["reasoning"] = {
                    "effort": kwargs.get("reasoning_effort") or self.config.get("reasoning_effort", "high")
                }

            budget_tokens = kwargs.get("thinking_budget_tokens") or self.config.get("thinking_budget_tokens") or 0
            buffer_tokens = kwargs.get("thinking_budget_buffer") or self.config.get("thinking_budget_buffer", 1024)

            if budget_tokens:
                thinking_budget = budget_tokens
                target_tokens = budget_tokens + buffer_tokens
                if params.get("max_output_tokens"):
                    params["max_output_tokens"] = max(params["max_output_tokens"], target_tokens)
                else:
                    params["max_output_tokens"] = target_tokens

            logger.info(
                "[PROVIDER] Extended thinking enabled (effort=%s, budget=%s, buffer=%s)",
                params["reasoning"]["effort"],
                thinking_budget or "default",
                buffer_tokens,
            )

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
                },
            )

            # DEBUG level: Full request payload (if debug enabled)
            if self.debug:
                await self.coordinator.hooks.emit(
                    "llm:request:debug",
                    {
                        "lvl": "DEBUG",
                        "provider": self.name,
                        "request": {
                            "model": params["model"],
                            "input": input_messages,  # Message array, not text
                            "instructions": instructions,
                            "max_output_tokens": params.get("max_output_tokens"),
                            "temperature": params.get("temperature"),
                            "reasoning": params.get("reasoning"),
                            "thinking_enabled": thinking_enabled,
                            "tools": params.get("tools"),
                            "tool_choice": params.get("tool_choice"),
                        },
                    },
                )

        if self.coordinator and hasattr(self.coordinator, "hooks") and self.debug and self.raw_debug:
            await self.coordinator.hooks.emit(
                "llm:request:raw",
                {
                    "lvl": "DEBUG",
                    "provider": self.name,
                    "params": params,
                },
            )

        start_time = time.time()

        # Call provider API
        try:
            response = await asyncio.wait_for(self.client.responses.create(**params), timeout=self.timeout)
            elapsed_ms = int((time.time() - start_time) * 1000)

            logger.info("[PROVIDER] Received response from %s API", self.api_label)

            if self.coordinator and hasattr(self.coordinator, "hooks") and self.debug and self.raw_debug:
                await self.coordinator.hooks.emit(
                    "llm:response:raw",
                    {
                        "lvl": "DEBUG",
                        "provider": self.name,
                        "response": response,
                    },
                )

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
                        "usage": {"input": usage_counts["input"], "output": usage_counts["output"]},
                        "status": "ok",
                        "duration_ms": elapsed_ms,
                    },
                )

                # DEBUG level: Full response (if debug enabled)
                if self.debug:
                    content_preview = str(response.output)[:500] if response.output else ""
                    await self.coordinator.hooks.emit(
                        "llm:response:debug",
                        {
                            "lvl": "DEBUG",
                            "provider": self.name,
                            "response": {
                                "content_preview": content_preview,
                            },
                            "status": "ok",
                            "duration_ms": elapsed_ms,
                        },
                    )

            # Convert to ChatResponse
            return self._convert_to_chat_response(response)

        except Exception as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.error("[PROVIDER] %s API error: %s", self.api_label, e)

            # Emit error event
            if self.coordinator and hasattr(self.coordinator, "hooks"):
                await self.coordinator.hooks.emit(
                    "llm:response",
                    {
                        "status": "error",
                        "duration_ms": elapsed_ms,
                        "error": str(e),
                        "provider": self.name,
                        "model": params["model"],
                    },
                )
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

            # Handle tool result messages
            if role == "tool":
                # For OpenAI Responses API, convert tool results to text
                # API doesn't support tool_result content type - use input_text
                tool_results_parts = []
                while i < len(messages) and messages[i].get("role") == "tool":
                    tool_msg = messages[i]
                    tool_name = tool_msg.get("tool_name", "unknown")
                    tool_content = tool_msg.get("content", "")

                    # Format as text for API
                    tool_results_parts.append(f"[Tool: {tool_name}]\n{tool_content}")
                    i += 1

                # Add as user message with combined tool results as text
                if tool_results_parts:
                    combined_text = "\n\n".join(tool_results_parts)
                    openai_messages.append({"role": "user", "content": [{"type": "input_text", "text": combined_text}]})
                continue

            # Handle assistant messages
            if role == "assistant":
                # For Responses API, only include text content in input
                # Tool call details are inferred from tool definitions and results
                if content:
                    openai_messages.append({"role": "assistant", "content": [{"type": "output_text", "text": content}]})
                # Skip assistant messages with no text content (tool-only messages)
                i += 1

            # Handle developer messages as XML-wrapped user messages
            elif role == "developer":
                wrapped = f"<context_file>\n{content}\n</context_file>"
                openai_messages.append({"role": "user", "content": [{"type": "input_text", "text": wrapped}]})
                i += 1

            # Handle user messages
            elif role == "user":
                openai_messages.append(
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": content}] if isinstance(content, str) else content,
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

        Args:
            tools: List of ToolSpec objects

        Returns:
            List of OpenAI-formatted tool definitions
        """
        openai_tools = []
        for tool in tools:
            openai_tools.append(
                {
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.parameters,
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
        from amplifier_core.message_models import ReasoningBlock as ResponseReasoningBlock
        from amplifier_core.message_models import TextBlock
        from amplifier_core.message_models import ThinkingBlock
        from amplifier_core.message_models import ToolCall
        from amplifier_core.message_models import ToolCallBlock
        from amplifier_core.message_models import Usage

        content_blocks = []
        tool_calls = []
        event_blocks: list[TextContent | ThinkingContent | ToolCallContent] = []
        text_accumulator: list[str] = []

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
                            if hasattr(content_item, "type") and content_item.type == "output_text":
                                text = getattr(content_item, "text", "")
                                content_blocks.append(TextBlock(text=text))
                                text_accumulator.append(text)
                                event_blocks.append(TextContent(text=text, raw=getattr(content_item, "raw", None)))
                    elif isinstance(block_content, str):
                        content_blocks.append(TextBlock(text=block_content))
                        text_accumulator.append(block_content)
                        event_blocks.append(TextContent(text=block_content))

                elif block_type == "reasoning":
                    # Extract reasoning summary if available
                    reasoning_summary = getattr(block, "summary", None) or getattr(block, "text", None)
                    # Only create thinking block if there's actual content
                    if reasoning_summary:
                        content_blocks.append(
                            ThinkingBlock(thinking=reasoning_summary, signature=None, visibility="internal")
                        )
                        event_blocks.append(ThinkingContent(text=reasoning_summary))
                        text_accumulator.append(reasoning_summary)

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
                            logger.debug("Failed to decode tool call arguments: %s", tool_input)
                    if tool_input is None:
                        tool_input = {}
                    # Ensure tool_input is dict after json.loads or default
                    if not isinstance(tool_input, dict):
                        tool_input = {}
                    content_blocks.append(ToolCallBlock(id=tool_id, name=tool_name, input=tool_input))
                    tool_calls.append(ToolCall(id=tool_id, name=tool_name, arguments=tool_input))
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
                                event_blocks.append(TextContent(text=text, raw=content_item))
                    elif isinstance(block_content, str):
                        content_blocks.append(TextBlock(text=block_content))
                        text_accumulator.append(block_content)
                        event_blocks.append(TextContent(text=block_content, raw=block))

                elif block_type == "reasoning":
                    # Extract reasoning summary if available
                    reasoning_summary = getattr(block, "summary", None) or getattr(block, "text", None)
                    # Only create thinking block if there's actual content
                    if reasoning_summary:
                        content_blocks.append(
                            ThinkingBlock(thinking=reasoning_summary, signature=None, visibility="internal")
                        )
                        event_blocks.append(ThinkingContent(text=reasoning_summary))
                        text_accumulator.append(reasoning_summary)

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
                            logger.debug("Failed to decode tool call arguments: %s", tool_input)
                    if tool_input is None:
                        tool_input = {}
                    # Ensure tool_input is dict after json.loads or default
                    if not isinstance(tool_input, dict):
                        tool_input = {}
                    content_blocks.append(ToolCallBlock(id=tool_id, name=tool_name, input=tool_input))
                    tool_calls.append(ToolCall(id=tool_id, name=tool_name, arguments=tool_input))
                    event_blocks.append(ToolCallContent(id=tool_id, name=tool_name, arguments=tool_input, raw=block))

        # Extract usage counts
        usage_obj = response.usage if hasattr(response, "usage") else None
        usage_counts = {"input": 0, "output": 0, "total": 0}
        if usage_obj:
            if hasattr(usage_obj, "input_tokens"):
                usage_counts["input"] = usage_obj.input_tokens
            if hasattr(usage_obj, "output_tokens"):
                usage_counts["output"] = usage_obj.output_tokens
            usage_counts["total"] = usage_counts["input"] + usage_counts["output"]

        usage = Usage(
            input_tokens=usage_counts["input"],
            output_tokens=usage_counts["output"],
            total_tokens=usage_counts["total"],
        )

        combined_text = "\n\n".join(text_accumulator).strip()

        chat_response = OpenAIChatResponse(
            content=content_blocks,
            tool_calls=tool_calls if tool_calls else None,
            usage=usage,
            finish_reason=getattr(response, "stop_reason", None),
            content_blocks=event_blocks if event_blocks else None,
            text=combined_text or None,
        )

        return chat_response
