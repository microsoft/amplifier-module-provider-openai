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

    async def complete(self, request: ChatRequest, **kwargs) -> ChatResponse:
        """Generate completion using Responses API.

        Args:
            request: Typed chat request with messages, tools, config
            **kwargs: Provider-specific options (override request fields)

        Returns:
            ChatResponse with content blocks, tool calls, usage
        """
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
        conversation = [m for m in message_list if m.role in ("user", "assistant")]

        logger.info(
            f"[PROVIDER] Separated: {len(system_msgs)} system, {len(developer_msgs)} developer, {len(conversation)} conversation"
        )

        # Combine system messages as instructions
        instructions = (
            "\n\n".join(m.content if isinstance(m.content, str) else "" for m in system_msgs) if system_msgs else None
        )

        # Convert developer messages to XML-wrapped format
        developer_input = []
        for i, dev_msg in enumerate(developer_msgs):
            content = dev_msg.content if isinstance(dev_msg.content, str) else ""
            logger.info(f"[PROVIDER] Converting developer message {i + 1}/{len(developer_msgs)}: length={len(content)}")
            wrapped = f"<context_file>\n{content}\n</context_file>"
            developer_input.append(f"USER: {wrapped}")

        # Convert conversation messages to text format
        conversation_parts = []
        for m in conversation:
            role_label = m.role.upper()
            if isinstance(m.content, str):
                conversation_parts.append(f"{role_label}: {m.content}")
            elif isinstance(m.content, list):
                # Extract text from content blocks
                text_parts = []
                for block in m.content:
                    if block.type == "text":
                        text_parts.append(block.text)
                if text_parts:
                    conversation_parts.append(f"{role_label}: {' '.join(text_parts)}")

        # Combine: developer context THEN conversation
        input_parts = []
        if developer_input:
            input_parts.extend(developer_input)
        if conversation_parts:
            input_parts.extend(conversation_parts)

        input_text = "\n\n".join(input_parts)
        logger.info(f"[PROVIDER] Final input length: {len(input_text)}")

        # Prepare request parameters
        params = {
            "model": kwargs.get("model", self.default_model),
            "input": input_text,
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

        # Add tools if provided
        if request.tools:
            params["tools"] = self._convert_tools_from_request(request.tools)

        logger.info(
            f"[PROVIDER] {self.api_label} API call - model: {params['model']}, has_instructions: {bool(instructions)}"
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
                            "input": input_text,
                            "instructions": instructions,
                            "max_output_tokens": params.get("max_output_tokens"),
                            "temperature": params.get("temperature"),
                            "reasoning": params.get("reasoning"),
                            "thinking_enabled": thinking_enabled,
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
                    # Simplified reasoning handling - just create a placeholder
                    placeholder = "[Reasoning content from o-series model]"
                    content_blocks.append(ThinkingBlock(thinking=placeholder, signature=None, visibility="internal"))
                    event_blocks.append(ThinkingContent(text=placeholder))
                    text_accumulator.append(placeholder)

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
                    # Simplified reasoning handling - just create a placeholder
                    placeholder = "[Reasoning content from o-series model]"
                    content_blocks.append(ThinkingBlock(thinking=placeholder, signature=None, visibility="internal"))
                    event_blocks.append(ThinkingContent(text=placeholder))
                    text_accumulator.append(placeholder)

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
