"""
OpenAI provider module for Amplifier.
Integrates with OpenAI's Responses API.
"""

import asyncio
import logging
import os
from typing import Any

from amplifier_core import ModuleCoordinator
from amplifier_core import ProviderResponse
from amplifier_core import ToolCall
from amplifier_core.content_models import TextContent
from amplifier_core.content_models import ThinkingContent
from amplifier_core.content_models import ToolCallContent
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

    provider = OpenAIProvider(api_key, config)
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

    def __init__(self, api_key: str, config: dict[str, Any] | None = None):
        """Initialize OpenAI provider with Responses API client."""
        self.client = AsyncOpenAI(api_key=api_key)
        self.config = config or {}

        # Configuration with sensible defaults
        self.default_model = self.config.get("default_model", "gpt-5-codex")
        self.max_tokens = self.config.get("max_tokens", 4096)
        self.temperature = self.config.get("temperature", None)  # None = not sent (some models don't support it)
        self.reasoning = self.config.get("reasoning", None)  # None = not sent (minimal|low|medium|high)
        self.enable_state = self.config.get("enable_state", False)

        # Provider priority for selection (lower = higher priority)
        self.priority = self.config.get("priority", 100)

    async def complete(self, messages: list[dict[str, Any]], **kwargs) -> ProviderResponse:
        """Generate completion using Responses API."""

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

        try:
            # 3. Call Responses API with timeout
            try:
                # Add timeout to prevent hanging (30 seconds)
                response = await asyncio.wait_for(self.client.responses.create(**params), timeout=30.0)
            except TimeoutError:
                logger.error(f"OpenAI Responses API timed out after 30s. Input: {input_text[:200]}...")
                raise TimeoutError("OpenAI API request timed out after 30 seconds")

            # 4. Parse response output
            content, tool_calls, content_blocks = self._parse_response_output(response.output)

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

            # Handle tool messages
            if role == "TOOL":
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
