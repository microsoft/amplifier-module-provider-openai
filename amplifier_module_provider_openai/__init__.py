"""
OpenAI provider module for Amplifier.
Integrates with OpenAI's Responses API.
"""

import asyncio
import logging
import os
import time
from typing import Any
from typing import Optional

from amplifier_core import ModuleCoordinator
from amplifier_core import ProviderResponse
from amplifier_core import ToolCall
from amplifier_core.content_models import TextContent
from amplifier_core.content_models import ThinkingContent
from amplifier_core.content_models import ToolCallContent
from amplifier_core.message_models import ChatRequest
from amplifier_core.message_models import ChatResponse
from amplifier_core.message_models import Message
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


async def mount(coordinator: ModuleCoordinator, config: dict[str, Any] | None = None):
    """Mount the OpenAI provider."""
    config = config or {}

    # Get API key from config or environment
    api_key = config.get("api_key") or os.environ.get("OPENAI_API_KEY")

    if not api_key:
        logger.warning("No API key found for OpenAI provider")
        return None

    provider = OpenAIProvider(api_key, config, coordinator)
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

    def __init__(
        self, api_key: str, config: dict[str, Any] | None = None, coordinator: ModuleCoordinator | None = None
    ):
        """Initialize OpenAI provider with Responses API client."""
        self.client = AsyncOpenAI(api_key=api_key)
        self.config = config or {}
        self.coordinator = coordinator

        # Configuration with sensible defaults
        self.default_model = self.config.get("default_model", "gpt-5-codex")
        self.max_tokens = self.config.get("max_tokens", 4096)
        self.temperature = self.config.get("temperature", None)  # None = not sent (some models don't support it)
        self.reasoning = self.config.get("reasoning", None)  # None = not sent (minimal|low|medium|high)
        self.enable_state = self.config.get("enable_state", False)
        self.debug = self.config.get("debug", False)  # Enable full request/response logging
        self.timeout = self.config.get("timeout", 300.0)  # API timeout in seconds (default 5 minutes)

        # Provider priority for selection (lower = higher priority)
        self.priority = self.config.get("priority", 100)

    async def complete(self, messages: list[dict[str, Any]] | ChatRequest, **kwargs) -> ProviderResponse | ChatResponse:
        """Generate completion using Responses API.

        Args:
            messages: Conversation history (list of dicts or ChatRequest)
            **kwargs: Additional parameters

        Returns:
            Provider response or ChatResponse
        """
        # Handle ChatRequest format
        if isinstance(messages, ChatRequest):
            return await self._complete_chat_request(messages, **kwargs)

        # 1. Extract system instructions and convert messages to input
        instructions, remaining_messages = self._extract_system_instructions(messages)
        input_text = self._convert_messages_to_input(remaining_messages)

        # 2. Prepare request parameters
        params = {
            "model": kwargs.get("model", self.default_model),
            "input": input_text,
        }

        # Add instructions if present
        if instructions:
            params["instructions"] = instructions

        # Add max output tokens
        if max_tokens := kwargs.get("max_tokens", self.max_tokens):
            params["max_output_tokens"] = max_tokens

        # Add temperature
        if temperature := kwargs.get("temperature", self.temperature):
            params["temperature"] = temperature

        # Add reasoning control
        if reasoning := kwargs.get("reasoning", self.reasoning):
            params["reasoning"] = {"effort": reasoning}

        # Add tools if provided
        if "tools" in kwargs and kwargs["tools"]:
            params["tools"] = self._convert_tools(kwargs["tools"])

        # Add JSON schema if requested
        if json_schema := kwargs.get("json_schema"):
            params["text"] = {"format": {"type": "json_schema", "json_schema": json_schema}}

        # Handle stateful conversations if enabled
        if self.enable_state:
            params["store"] = True
            if previous_id := kwargs.get("previous_response_id"):
                params["previous_response_id"] = previous_id

        # Emit llm:request event if coordinator is available
        if self.coordinator and hasattr(self.coordinator, "hooks"):
            # INFO level: Summary only
            await self.coordinator.hooks.emit(
                "llm:request",
                {
                    "data": {
                        "provider": "openai",
                        "model": params["model"],
                        "message_count": len(remaining_messages),
                        "reasoning_enabled": params.get("reasoning") is not None,
                    }
                },
            )

            # DEBUG level: Full request payload (if debug enabled)
            if self.debug:
                await self.coordinator.hooks.emit(
                    "llm:request:debug",
                    {
                        "lvl": "DEBUG",
                        "data": {
                            "provider": "openai",
                            "request": {
                                "model": params["model"],
                                "input": input_text,
                                "instructions": instructions,
                                "max_output_tokens": params.get("max_output_tokens"),
                                "temperature": params.get("temperature"),
                                "reasoning": params.get("reasoning"),
                            },
                        },
                    },
                )

        start_time = time.time()
        try:
            # 3. Call Responses API with timeout
            try:
                # Add timeout to prevent hanging (30 seconds)
                response = await asyncio.wait_for(self.client.responses.create(**params), timeout=self.timeout)
                elapsed_ms = int((time.time() - start_time) * 1000)
            except TimeoutError:
                logger.error(f"OpenAI Responses API timed out after 30s. Input: {input_text[:200]}...")
                # Emit error response event
                if self.coordinator and hasattr(self.coordinator, "hooks"):
                    await self.coordinator.hooks.emit(
                        "llm:response",
                        {
                            "data": {
                                "provider": "openai",
                                "model": params["model"],
                            },
                            "status": "error",
                            "duration_ms": int((time.time() - start_time) * 1000),
                            "error": f"Timeout after {self.timeout} seconds",
                        },
                    )
                raise TimeoutError(f"OpenAI API request timed out after {self.timeout} seconds")

            # 4. Parse response output
            content, tool_calls, content_blocks = self._parse_response_output(response.output)

            # Check if we have reasoning/thinking blocks and emit events
            has_reasoning = False
            if self.coordinator and hasattr(self.coordinator, "hooks"):
                for block in content_blocks or []:
                    if isinstance(block, ThinkingContent):
                        has_reasoning = True
                        # Emit thinking:final event for reasoning blocks
                        await self.coordinator.hooks.emit("thinking:final", {"text": block.text})

            # Emit llm:response success event
            if self.coordinator and hasattr(self.coordinator, "hooks"):
                # INFO level: Summary only
                await self.coordinator.hooks.emit(
                    "llm:response",
                    {
                        "data": {
                            "provider": "openai",
                            "model": params["model"],
                            "usage": {
                                "input": getattr(response.usage, "prompt_tokens", 0)
                                if hasattr(response, "usage")
                                else 0,
                                "output": getattr(response.usage, "completion_tokens", 0)
                                if hasattr(response, "usage")
                                else 0,
                            },
                            "has_reasoning": has_reasoning,
                        },
                        "status": "ok",
                        "duration_ms": elapsed_ms,
                    },
                )

                # DEBUG level: Full response (if debug enabled)
                if self.debug:
                    await self.coordinator.hooks.emit(
                        "llm:response:debug",
                        {
                            "lvl": "DEBUG",
                            "data": {
                                "provider": "openai",
                                "response": {
                                    "content": content[:500] + "..." if len(content) > 500 else content,
                                    "tool_calls": [{"tool": tc.tool, "id": tc.id} for tc in tool_calls]
                                    if tool_calls
                                    else [],
                                },
                            },
                            "status": "ok",
                            "duration_ms": elapsed_ms,
                        },
                    )

            # 5. Return standardized response
            return ProviderResponse(
                content=content,
                raw=response,
                usage={
                    "input": getattr(response.usage, "prompt_tokens", 0) if hasattr(response, "usage") else 0,
                    "output": getattr(response.usage, "completion_tokens", 0) if hasattr(response, "usage") else 0,
                    "total": getattr(response.usage, "total_tokens", 0) if hasattr(response, "usage") else 0,
                },
                tool_calls=tool_calls if tool_calls else None,
                content_blocks=content_blocks if content_blocks else None,
            )

        except Exception as e:
            logger.error(f"OpenAI Responses API error: {e}")

            # Emit llm:response event with error
            if self.coordinator and hasattr(self.coordinator, "hooks"):
                await self.coordinator.hooks.emit(
                    "llm:response",
                    {
                        "data": {
                            "provider": "openai",
                            "model": params.get("model", self.default_model),
                        },
                        "status": "error",
                        "duration_ms": int((time.time() - start_time) * 1000),
                        "error": str(e),
                    },
                )

            raise

    def parse_tool_calls(self, response: ProviderResponse) -> list[ToolCall]:
        """Parse tool calls from provider response."""
        return response.tool_calls or []

    def _extract_system_instructions(self, messages: list[dict[str, Any]]) -> tuple[str | None, list[dict[str, Any]]]:
        """Extract system messages as instructions."""
        system_messages = [m for m in messages if m.get("role") == "system"]
        other_messages = [m for m in messages if m.get("role") != "system"]

        instructions = None
        if system_messages:
            instructions = "\n\n".join([m.get("content", "") for m in system_messages])

        return instructions, other_messages

    def _convert_messages_to_input(self, messages: list[dict[str, Any]]) -> str:
        """Convert message array to single input string."""
        formatted = []

        for msg in messages:
            role = msg.get("role", "").upper()
            content = msg.get("content", "")

            # Handle developer messages - wrap in XML
            if role == "DEVELOPER":
                wrapped = f"<context_file>\n{content}\n</context_file>"
                formatted.append(f"USER: {wrapped}")
            # Handle tool messages
            elif role == "TOOL":
                # Include tool results in the input
                tool_id = msg.get("tool_call_id", "unknown")
                formatted.append(f"TOOL RESULT [{tool_id}]: {content}")
            elif role == "ASSISTANT" and msg.get("tool_calls"):
                # Include assistant's tool calls
                tool_call_desc = ", ".join([tc.get("tool", "") for tc in msg["tool_calls"]])
                if content:
                    formatted.append(f"{role}: {content} [Called tools: {tool_call_desc}]")
                else:
                    formatted.append(f"{role}: [Called tools: {tool_call_desc}]")
            else:
                # Regular messages
                formatted.append(f"{role}: {content}")

        return "\n\n".join(formatted)

    def _parse_response_output(self, output: list[Any]) -> tuple[str, list[ToolCall], list[Any]]:
        """Parse output blocks into content, tool calls, and content_blocks.

        Note: output can be either SDK objects or dictionaries depending on the response.
        """
        content_parts = []
        tool_calls = []
        content_blocks = []

        for block in output:
            # Handle both SDK objects and dictionaries
            if hasattr(block, "type"):
                # SDK object (like ResponseReasoningItem, ResponseMessageItem, etc.)
                block_type = block.type

                if block_type == "message":
                    # Extract text from message content
                    block_content = getattr(block, "content", [])
                    if isinstance(block_content, list):
                        for content_item in block_content:
                            if hasattr(content_item, "type") and content_item.type == "output_text":
                                text = getattr(content_item, "text", "")
                                content_parts.append(text)
                                content_blocks.append(TextContent(text=text, raw=content_item))
                            elif hasattr(content_item, "get") and content_item.get("type") == "output_text":
                                text = content_item.get("text", "")
                                content_parts.append(text)
                                content_blocks.append(TextContent(text=text, raw=content_item))
                    elif isinstance(block_content, str):
                        content_parts.append(block_content)
                        content_blocks.append(TextContent(text=block_content, raw=block))

                elif block_type == "reasoning":
                    # Extract reasoning as ThinkingContent
                    reasoning_text = getattr(block, "text", "")
                    if reasoning_text:
                        content_blocks.append(ThinkingContent(text=reasoning_text, raw=block))

                elif block_type == "tool_call":
                    # Native tool call from Responses API
                    tool_calls.append(
                        ToolCall(
                            tool=getattr(block, "name", ""),
                            arguments=getattr(block, "input", {}),
                            id=getattr(block, "id", ""),
                        )
                    )
                    content_blocks.append(
                        ToolCallContent(
                            id=getattr(block, "id", ""),
                            name=getattr(block, "name", ""),
                            arguments=getattr(block, "input", {}),
                            raw=block,
                        )
                    )
            else:
                # Dictionary format
                block_type = block.get("type")

                if block_type == "message":
                    # Extract text from message content
                    block_content = block.get("content", [])
                    if isinstance(block_content, list):
                        for content_item in block_content:
                            if content_item.get("type") == "output_text":
                                text = content_item.get("text", "")
                                content_parts.append(text)
                                content_blocks.append(TextContent(text=text, raw=content_item))
                    elif isinstance(block_content, str):
                        content_parts.append(block_content)
                        content_blocks.append(TextContent(text=block_content, raw=block))

                elif block_type == "reasoning":
                    # Extract reasoning as ThinkingContent
                    reasoning_text = block.get("text", "")
                    if reasoning_text:
                        content_blocks.append(ThinkingContent(text=reasoning_text, raw=block))

                elif block_type == "tool_call":
                    # Native tool call from Responses API
                    tool_calls.append(
                        ToolCall(tool=block.get("name", ""), arguments=block.get("input", {}), id=block.get("id", ""))
                    )
                    content_blocks.append(
                        ToolCallContent(
                            id=block.get("id", ""),
                            name=block.get("name", ""),
                            arguments=block.get("input", {}),
                            raw=block,
                        )
                    )

        content = "\n\n".join(content_parts) if content_parts else ""
        return content, tool_calls, content_blocks

    def _convert_tools(self, tools: list[Any]) -> list[dict[str, Any]]:
        """Convert tools to Responses API format."""
        responses_tools = []

        for tool in tools:
            # Get schema from tool if available
            input_schema = getattr(tool, "input_schema", {"type": "object", "properties": {}, "required": []})

            responses_tools.append(
                {"type": "function", "name": tool.name, "description": tool.description, "parameters": input_schema}
            )

        return responses_tools

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

        # Separate messages by role
        system_msgs = [m for m in request.messages if m.role == "system"]
        developer_msgs = [m for m in request.messages if m.role == "developer"]
        conversation = [m for m in request.messages if m.role in ("user", "assistant")]

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

        # Convert conversation messages
        conversation_dicts = [m.model_dump() for m in conversation]
        conversation_input = self._convert_messages_to_input(conversation_dicts)

        # Combine: developer context THEN conversation
        input_parts = []
        if developer_input:
            input_parts.extend(developer_input)
        if conversation_input:
            input_parts.append(conversation_input)

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

        # Add tools if provided
        if request.tools:
            params["tools"] = self._convert_tools_from_request(request.tools)

        logger.info(f"[PROVIDER] OpenAI API call - model: {params['model']}, has_instructions: {bool(instructions)}")

        # Emit llm:request event
        if self.coordinator and hasattr(self.coordinator, "hooks"):
            # INFO level: Summary only
            await self.coordinator.hooks.emit(
                "llm:request",
                {
                    "data": {
                        "provider": "openai",
                        "model": params["model"],
                        "message_count": len(request.messages),
                        "has_instructions": bool(instructions),
                    }
                },
            )

            # DEBUG level: Full request payload (if debug enabled)
            if self.debug:
                await self.coordinator.hooks.emit(
                    "llm:request:debug",
                    {
                        "lvl": "DEBUG",
                        "data": {
                            "provider": "openai",
                            "request": {
                                "model": params["model"],
                                "input": input_text,
                                "instructions": instructions,
                                "max_output_tokens": params.get("max_output_tokens"),
                                "temperature": params.get("temperature"),
                            },
                        },
                    },
                )

        start_time = time.time()

        # Call OpenAI API
        try:
            response = await asyncio.wait_for(self.client.responses.create(**params), timeout=30.0)
            elapsed_ms = int((time.time() - start_time) * 1000)

            logger.info("[PROVIDER] Received response from OpenAI API")

            # Emit llm:response event
            if self.coordinator and hasattr(self.coordinator, "hooks"):
                # INFO level: Summary only
                await self.coordinator.hooks.emit(
                    "llm:response",
                    {
                        "data": {
                            "provider": "openai",
                            "model": params["model"],
                            "usage": {
                                "input": getattr(response.usage, "prompt_tokens", 0)
                                if hasattr(response, "usage")
                                else 0,
                                "output": getattr(response.usage, "completion_tokens", 0)
                                if hasattr(response, "usage")
                                else 0,
                            },
                        },
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
                            "data": {
                                "provider": "openai",
                                "response": {
                                    "content_preview": content_preview,
                                },
                            },
                            "status": "ok",
                            "duration_ms": elapsed_ms,
                        },
                    )

            # Convert to ChatResponse
            return self._convert_to_chat_response(response)

        except Exception as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.error(f"[PROVIDER] OpenAI API error: {e}")

            # Emit error event
            if self.coordinator and hasattr(self.coordinator, "hooks"):
                await self.coordinator.hooks.emit(
                    "llm:response",
                    {
                        "data": {
                            "provider": "openai",
                            "model": params["model"],
                        },
                        "status": "error",
                        "duration_ms": elapsed_ms,
                        "error": str(e),
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
        from amplifier_core.message_models import TextBlock
        from amplifier_core.message_models import ThinkingBlock
        from amplifier_core.message_models import ToolCall
        from amplifier_core.message_models import ToolCallBlock
        from amplifier_core.message_models import Usage

        content_blocks = []
        tool_calls = []

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
                    elif isinstance(block_content, str):
                        content_blocks.append(TextBlock(text=block_content))

                elif block_type == "reasoning":
                    # Extract reasoning as ThinkingBlock
                    reasoning_text = getattr(block, "text", "")
                    if reasoning_text:
                        content_blocks.append(ThinkingBlock(thinking=reasoning_text, signature=None))

                elif block_type == "tool_call":
                    # Native tool call from Responses API
                    tool_id = getattr(block, "id", "")
                    tool_name = getattr(block, "name", "")
                    tool_input = getattr(block, "input", {})
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
                    elif isinstance(block_content, str):
                        content_blocks.append(TextBlock(text=block_content))

                elif block_type == "reasoning":
                    reasoning_text = block.get("text", "")
                    if reasoning_text:
                        content_blocks.append(ThinkingBlock(thinking=reasoning_text, signature=None))

                elif block_type == "tool_call":
                    tool_id = block.get("id", "")
                    tool_name = block.get("name", "")
                    tool_input = block.get("input", {})
                    content_blocks.append(ToolCallBlock(id=tool_id, name=tool_name, input=tool_input))
                    tool_calls.append(ToolCall(id=tool_id, name=tool_name, arguments=tool_input))

        usage = Usage(
            input_tokens=getattr(response.usage, "prompt_tokens", 0) if hasattr(response, "usage") else 0,
            output_tokens=getattr(response.usage, "completion_tokens", 0) if hasattr(response, "usage") else 0,
            total_tokens=getattr(response.usage, "total_tokens", 0) if hasattr(response, "usage") else 0,
        )

        return ChatResponse(
            content=content_blocks,
            tool_calls=tool_calls if tool_calls else None,
            usage=usage,
            finish_reason=getattr(response, "stop_reason", None),
        )
