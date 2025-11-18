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
from amplifier_core import ProviderResponse
from amplifier_core import ToolCall
from amplifier_core.content_models import TextContent
from amplifier_core.content_models import ThinkingContent
from amplifier_core.content_models import ToolCallContent
from amplifier_core.message_models import ChatRequest
from amplifier_core.message_models import ChatResponse
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

        messages, repair_count, repairs_made = self._sanitize_incomplete_tool_sequences(messages)
        if repair_count and self.coordinator and hasattr(self.coordinator, "hooks"):
            # Emit observability event for repair
            await self.coordinator.hooks.emit(
                "provider:tool_sequence_repaired",
                {
                    "provider": self.name,
                    "repair_count": repair_count,
                    "repairs": repairs_made,
                },
            )

        try:
            self._validate_tool_message_sequence(messages)
        except ValueError as exc:
            await self._emit_validation_error("tool_call_sequence", str(exc), len(messages))
            raise RuntimeError(
                f"Cannot send messages to {self.api_label} Responses API: {exc}\n\n"
                f"This usually indicates missing or out-of-order tool responses."
            ) from exc

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
        reasoning_effort = kwargs.get("reasoning", self.reasoning)
        if reasoning_effort:
            params["reasoning"] = {"effort": reasoning_effort}

        thinking_enabled = bool(kwargs.get("extended_thinking"))
        thinking_budget = None
        if thinking_enabled:
            # Default reasoning effort to high if not explicitly set
            if not params.get("reasoning"):
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
                "Extended thinking enabled for %s Responses API (effort=%s, budget=%s, buffer=%s)",
                self.api_label,
                params["reasoning"]["effort"],
                thinking_budget or "default",
                buffer_tokens,
            )
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
                    "provider": self.name,
                    "model": params["model"],
                    "message_count": len(remaining_messages),
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

        # RAW DEBUG: Complete request params sent to API (ultra-verbose)
        if self.coordinator and hasattr(self.coordinator, "hooks") and self.debug and self.raw_debug:
            await self.coordinator.hooks.emit(
                "llm:request:raw",
                {
                    "lvl": "DEBUG",
                    "provider": self.name,
                    "params": params,  # Complete params dict as-is
                },
            )

        start_time = time.time()
        try:
            # 3. Call Responses API with timeout
            try:
                # Add timeout to prevent hanging (30 seconds)
                response = await asyncio.wait_for(self.client.responses.create(**params), timeout=self.timeout)

                # RAW DEBUG: Complete raw response from API (ultra-verbose)
                if self.coordinator and hasattr(self.coordinator, "hooks") and self.debug and self.raw_debug:
                    await self.coordinator.hooks.emit(
                        "llm:response:raw",
                        {
                            "lvl": "DEBUG",
                            "provider": self.name,
                            "response": response,  # Complete response object as-is
                        },
                    )

                elapsed_ms = int((time.time() - start_time) * 1000)
            except TimeoutError:
                logger.error(
                    "%s Responses API timed out after 30s. Input: %s...",
                    self.api_label,
                    input_text[:200],
                )
                # Emit error response event
                if self.coordinator and hasattr(self.coordinator, "hooks"):
                    await self.coordinator.hooks.emit(
                        "llm:response",
                        {
                            "provider": self.name,
                            "model": params["model"],
                            "status": "error",
                            "duration_ms": int((time.time() - start_time) * 1000),
                            "error": f"Timeout after {self.timeout} seconds",
                        },
                    )
                raise TimeoutError(f"{self.api_label} API request timed out after {self.timeout} seconds")

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

            usage_counts = self._extract_usage_counts(response.usage if hasattr(response, "usage") else None)

            # Emit llm:response success event
            if self.coordinator and hasattr(self.coordinator, "hooks"):
                # INFO level: Summary only
                await self.coordinator.hooks.emit(
                    "llm:response",
                    {
                        "provider": self.name,
                        "model": params["model"],
                        "usage": {"input": usage_counts["input"], "output": usage_counts["output"]},
                        "has_reasoning": has_reasoning,
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
                            "provider": self.name,
                            "response": {
                                "content": content[:500] + "..." if len(content) > 500 else content,
                                "tool_calls": [{"tool": tc.tool, "id": tc.id} for tc in tool_calls]
                                if tool_calls
                                else [],
                            },
                            "status": "ok",
                            "duration_ms": elapsed_ms,
                        },
                    )

            # 5. Return standardized response
            return ProviderResponse(
                content=content,
                raw=response,
                usage=usage_counts,
                tool_calls=tool_calls if tool_calls else None,
                content_blocks=content_blocks if content_blocks else None,
            )

        except Exception as e:
            logger.error("%s Responses API error: %s", self.api_label, e)

            # Emit llm:response event with error
            if self.coordinator and hasattr(self.coordinator, "hooks"):
                await self.coordinator.hooks.emit(
                    "llm:response",
                    {
                        "status": "error",
                        "duration_ms": int((time.time() - start_time) * 1000),
                        "error": str(e),
                        "provider": self.name,
                        "model": params.get("model", self.default_model),
                    },
                )

            raise

    def parse_tool_calls(self, response: ProviderResponse) -> list[ToolCall]:
        """Parse tool calls from provider response."""
        return response.tool_calls or []

    def _sanitize_incomplete_tool_sequences(self, messages: list[Any]) -> tuple[list[Any], int, list[dict]]:
        """Repair incomplete tool call sequences by injecting synthetic tool results.

        When assistant messages have tool_calls without matching tool results, inject
        synthetic results that make the failure visible to the LLM while allowing the
        conversation to continue (graceful degradation).

        Returns:
            (repaired_messages, count_of_repairs_made, list_of_repairs)
        """
        if not messages:
            return messages, 0, []

        repaired = list(messages)
        repair_count = 0
        repairs_made = []
        i = 0

        while i < len(repaired):
            message = repaired[i]
            role = self._normalize_role(self._get_message_attr(message, "role"))

            if role == "assistant":
                tool_calls = self._get_message_attr(message, "tool_calls")
                if tool_calls:
                    # Extract tool_call IDs
                    expected_ids = {
                        self._get_message_attr(tc, "id") for tc in tool_calls if self._get_message_attr(tc, "id")
                    }

                    # Scan forward for matching tool results
                    found_ids = set()
                    j = i + 1
                    while j < len(repaired):
                        next_message = repaired[j]
                        next_role = self._normalize_role(self._get_message_attr(next_message, "role"))
                        if next_role == "tool":
                            result_id = self._extract_tool_result_id(next_message)
                            if result_id and result_id in expected_ids:
                                found_ids.add(result_id)
                            j += 1
                        else:
                            break

                    # Find missing tool results
                    missing_ids = expected_ids - found_ids
                    if missing_ids:
                        # Inject synthetic tool results for each missing ID
                        insert_position = i + 1
                        for tool_call in tool_calls:
                            tc_id = self._get_message_attr(tool_call, "id")
                            if tc_id in missing_ids:
                                synthetic_result = self._create_synthetic_tool_result(tool_call)
                                repaired.insert(insert_position, synthetic_result)
                                insert_position += 1
                                repair_count += 1

                                # Track repair for observability
                                repairs_made.append(
                                    {
                                        "tool_call_id": str(tc_id),
                                        "tool_name": self._extract_tool_name(tool_call),
                                        "message_index": i,
                                    }
                                )

                        logger.warning(
                            "Injected %d synthetic tool result(s) for %s API (missing IDs: %s)",
                            len(missing_ids),
                            self.api_label,
                            sorted(missing_ids),
                        )

            i += 1

        return repaired, repair_count, repairs_made

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

    def _validate_tool_message_sequence(self, messages: list[Any]) -> None:
        """Ensure assistant tool calls are paired with subsequent tool results."""
        if not messages:
            return

        pending: set[str] = set()

        for index, message in enumerate(messages):
            role = self._normalize_role(self._get_message_attr(message, "role"))

            if role == "assistant":
                tool_call_ids = self._extract_tool_call_ids(message)
                if pending:
                    raise ValueError(
                        f"Assistant message at index {index} encountered before resolving tool results "
                        f"for outstanding tool calls {sorted(pending)}."
                    )
                pending = tool_call_ids

            elif role == "tool":
                call_id = self._extract_tool_result_id(message)
                if not pending:
                    raise ValueError(f"Tool result message at index {index} found without any preceding tool calls.")
                if not call_id:
                    raise ValueError(f"Tool result message at index {index} is missing tool_call_id.")
                if call_id not in pending:
                    raise ValueError(
                        f"Tool result message at index {index} references unknown tool_call_id '{call_id}'."
                    )
                pending.remove(call_id)

            else:
                if pending:
                    raise ValueError(
                        f"Expected tool result messages for outstanding tool calls {sorted(pending)}, "
                        f"but encountered role '{role or 'unknown'}' at index {index}."
                    )

        if pending:
            raise ValueError(f"Conversation ended with unresolved tool calls {sorted(pending)}.")

    def _normalize_role(self, role: Any) -> str | None:
        if isinstance(role, str):
            return role.lower()
        return role

    def _get_message_attr(self, message: Any, key: str) -> Any:
        if isinstance(message, dict):
            return message.get(key)
        return getattr(message, key, None)

    def _extract_tool_call_ids(self, message: Any) -> set[str]:
        result: set[str] = set()
        tool_calls = self._get_message_attr(message, "tool_calls")
        if not tool_calls:
            return result

        for call in tool_calls:
            call_id = self._get_message_attr(call, "id") or self._get_message_attr(call, "tool_call_id")
            if not call_id:
                raise ValueError("Assistant tool call is missing an id.")
            result.add(str(call_id))

        return result

    def _extract_tool_result_id(self, message: Any) -> str | None:
        tool_call_id = self._get_message_attr(message, "tool_call_id")
        if tool_call_id is None:
            return None
        return str(tool_call_id)

    def _extract_tool_name(self, tool_call: Any) -> str:
        """Extract tool name from a tool_call object."""
        function = self._get_message_attr(tool_call, "function")
        if function:
            name = self._get_message_attr(function, "name")
            if name:
                return str(name)
        # Fallback: try direct 'tool' or 'name' attribute
        tool = self._get_message_attr(tool_call, "tool")
        if tool:
            return str(tool)
        name = self._get_message_attr(tool_call, "name")
        if name:
            return str(name)
        return "unknown"

    def _safe_stringify(self, value: Any) -> str:
        """Safely convert value to string for error messages."""
        if value is None:
            return "null"
        if isinstance(value, str):
            return value[:500] + ("..." if len(value) > 500 else "")
        if isinstance(value, dict):
            try:
                json_str = json.dumps(value, ensure_ascii=False)
                return json_str[:500] + ("..." if len(json_str) > 500 else "")
            except Exception:
                return str(value)[:500]
        return str(value)[:500]

    def _create_synthetic_tool_result(self, tool_call: Any) -> dict[str, Any]:
        """Create synthetic tool result for missing tool call result.

        Makes the failure visible to the LLM while maintaining conversation validity.
        This is graceful degradation - the session continues but the error is observable.

        Args:
            tool_call: The tool call that's missing its result

        Returns:
            Synthetic tool result message in OpenAI format
        """
        tool_call_id = self._get_message_attr(tool_call, "id")
        tool_name = self._extract_tool_name(tool_call)

        # Extract arguments
        function = self._get_message_attr(tool_call, "function")
        if function:
            arguments = self._get_message_attr(function, "arguments")
        else:
            arguments = self._get_message_attr(tool_call, "arguments")

        return {
            "role": "tool",
            "tool_call_id": str(tool_call_id) if tool_call_id else "unknown",
            "content": (
                f"[SYSTEM ERROR: Tool result missing from conversation history]\n\n"
                f"Tool: {tool_name}\n"
                f"Arguments: {self._safe_stringify(arguments)}\n\n"
                f"This likely indicates:\n"
                f"- Context compaction dropped this result\n"
                f"- Parsing error in message history\n"
                f"- State corruption during session\n\n"
                f"The tool may have executed successfully, but the result was lost.\n"
                f"Please acknowledge this error and ask the user to retry if needed."
            ),
        }

    async def _emit_validation_error(self, validation: str, error: str, message_count: int) -> None:
        if self.coordinator and hasattr(self.coordinator, "hooks"):
            await self.coordinator.hooks.emit(
                "provider:validation_error",
                {
                    "provider": self.name,
                    "validation": validation,
                    "error": error,
                    "message_count": message_count,
                },
            )

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
                block_type = getattr(block, "type", "")

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
                    # Extract reasoning as ThinkingContent (flattened text)
                    reasoning_text = self._extract_reasoning_text(block)
                    if reasoning_text:
                        content_blocks.append(ThinkingContent(text=reasoning_text, raw=block))
                    else:
                        # Reasoning block exists but content not available (encrypted/hidden)
                        # Create placeholder to show reasoning occurred
                        content_blocks.append(
                            ThinkingContent(text="[Internal reasoning encrypted or hidden by provider]", raw=block)
                        )

                elif block_type in {"tool_call", "function_call"}:
                    arguments = getattr(block, "input", None)
                    if arguments is None and hasattr(block, "arguments"):
                        arguments = getattr(block, "arguments", None)
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except json.JSONDecodeError:
                            logger.debug("Failed to decode tool call arguments: %s", arguments)
                            arguments = {}
                    if arguments is None or not isinstance(arguments, dict):
                        arguments = {}

                    call_id = getattr(block, "id", "") or getattr(block, "call_id", "")
                    tool_name = getattr(block, "name", "")
                    if not arguments:
                        logger.debug("Skipping tool call '%s' with empty arguments", tool_name)
                        continue

                    tool_calls.append(
                        ToolCall(
                            tool=tool_name,
                            arguments=arguments,
                            id=call_id,
                        )
                    )
                    content_blocks.append(
                        ToolCallContent(
                            id=call_id,
                            name=tool_name,
                            arguments=arguments,
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
                    reasoning_text = self._extract_reasoning_text(block)
                    if reasoning_text:
                        content_blocks.append(ThinkingContent(text=reasoning_text, raw=block))
                    else:
                        # Reasoning block exists but content not available (encrypted/hidden)
                        content_blocks.append(
                            ThinkingContent(text="[Internal reasoning encrypted or hidden by provider]", raw=block)
                        )

                elif block_type in {"tool_call", "function_call"}:
                    arguments = block.get("input")
                    if arguments is None:
                        arguments = block.get("arguments", {})

                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except json.JSONDecodeError:
                            logger.debug("Failed to decode tool call arguments: %s", arguments[:500])
                            arguments = {}
                    if arguments is None or not isinstance(arguments, dict):
                        arguments = {}

                    call_id = block.get("id") or block.get("call_id", "")
                    tool_name = block.get("name", "")

                    if not arguments:
                        logger.debug("Skipping tool call '%s' with empty arguments", tool_name)
                        continue

                    tool_calls.append(ToolCall(tool=tool_name, arguments=arguments, id=call_id))
                    content_blocks.append(
                        ToolCallContent(
                            id=call_id,
                            name=tool_name,
                            arguments=arguments,
                            raw=block,
                        )
                    )

        content = "\n\n".join(content_parts) if content_parts else ""
        return content, tool_calls, content_blocks

    def _extract_reasoning_text(self, block: Any) -> str:
        """Flatten reasoning content into a readable text representation."""
        fragments: list[str] = []

        text_attr = getattr(block, "text", None)
        if isinstance(text_attr, str) and text_attr:
            fragments.append(text_attr)

        if hasattr(block, "content"):
            fragments.extend(self._flatten_reasoning_items(cast(Any, block).content))
        elif isinstance(block, dict) and "content" in block:
            fragments.extend(self._flatten_reasoning_items(block.get("content")))

        if not fragments and isinstance(block, dict):
            text_value = block.get("text")
            if isinstance(text_value, str) and text_value:
                fragments.append(text_value)

        return "\n".join(fragment for fragment in fragments if fragment)

    def _flatten_reasoning_items(self, items: Any) -> list[str]:
        """Extract text fragments from structured reasoning content."""
        fragments: list[str] = []
        if not items:
            return fragments

        if isinstance(items, list):
            iterable = items
        else:
            iterable = [items]

        for item in iterable:
            if item is None:
                continue
            text_val = getattr(item, "text", None)
            if isinstance(text_val, str) and text_val:
                fragments.append(text_val)
                continue
            if isinstance(item, dict):
                dict_text = item.get("text")
                if isinstance(dict_text, str) and dict_text:
                    fragments.append(dict_text)
                    continue
                nested = item.get("content")
                if nested:
                    fragments.extend(self._flatten_reasoning_items(nested))

        return fragments

    def _flatten_reasoning_summary(self, items: list[Any]) -> str:
        """Flatten summary items into a single string."""
        fragments = []
        for item in items or []:
            if isinstance(item, dict):
                text_val = item.get("text")
                if isinstance(text_val, str) and text_val:
                    fragments.append(text_val)
            else:
                text_val = getattr(item, "text", None)
                if isinstance(text_val, str) and text_val:
                    fragments.append(text_val)
        return "\n".join(fragments)

    def _collect_reasoning_metadata(self, block: Any) -> tuple[list[Any], list[Any], Any]:
        """Collect structured reasoning fields for ChatResponse conversion."""
        raw_content = None
        raw_summary = None
        visibility = None

        if hasattr(block, "content"):
            raw_content = cast(Any, block).content
        elif isinstance(block, dict):
            raw_content = block.get("content")

        if hasattr(block, "summary"):
            raw_summary = cast(Any, block).summary
        elif isinstance(block, dict):
            raw_summary = block.get("summary")

        if hasattr(block, "visibility"):
            visibility = cast(Any, block).visibility
        elif isinstance(block, dict):
            visibility = block.get("visibility")

        def _ensure_list(value: Any) -> list[Any]:
            if not value:
                return []
            if isinstance(value, list):
                return value
            if isinstance(value, tuple):
                return list(value)
            return [value]

        content_items = _ensure_list(raw_content)
        summary_items = _ensure_list(raw_summary)

        return content_items, summary_items, visibility

    def _extract_usage_counts(self, usage: Any | None) -> dict[str, int]:
        """Normalize usage metrics from OpenAI Responses API."""
        if not usage:
            return {"input": 0, "output": 0, "total": 0}

        input_tokens = getattr(usage, "input_tokens", None)
        if input_tokens is None:
            input_tokens = getattr(usage, "prompt_tokens", 0)

        output_tokens = getattr(usage, "output_tokens", None)
        if output_tokens is None:
            output_tokens = getattr(usage, "completion_tokens", 0)

        total_tokens = getattr(usage, "total_tokens", None)
        if total_tokens is None:
            total_tokens = input_tokens + output_tokens

        # Guard against None values
        input_tokens = input_tokens or 0
        output_tokens = output_tokens or 0
        total_tokens = total_tokens or input_tokens + output_tokens

        return {"input": int(input_tokens), "output": int(output_tokens), "total": int(total_tokens)}

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

        message_list = list(request.messages)
        message_list, repair_count, repairs_made = self._sanitize_incomplete_tool_sequences(message_list)
        if repair_count and self.coordinator and hasattr(self.coordinator, "hooks"):
            # Emit observability event for repair
            await self.coordinator.hooks.emit(
                "provider:tool_sequence_repaired",
                {
                    "provider": self.name,
                    "repair_count": repair_count,
                    "repairs": repairs_made,
                },
            )

        try:
            self._validate_tool_message_sequence(message_list)
        except ValueError as exc:
            await self._emit_validation_error("tool_call_sequence", str(exc), len(message_list))
            raise RuntimeError(
                f"Cannot send ChatRequest to {self.api_label} Responses API: "
                f"{exc}\n\nPlease inspect the conversation history for tool call/results ordering issues."
            ) from exc

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

            usage_counts = self._extract_usage_counts(response.usage if hasattr(response, "usage") else None)

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
                    content_items, summary_items, visibility = self._collect_reasoning_metadata(block)
                    if content_items or summary_items:
                        content_blocks.append(
                            ResponseReasoningBlock(content=content_items, summary=summary_items, visibility=visibility)
                        )
                        flattened = self._extract_reasoning_text(block) or self._flatten_reasoning_summary(
                            summary_items
                        )
                        if flattened:
                            text_accumulator.append(flattened)
                            event_blocks.append(ThinkingContent(text=flattened))
                    else:
                        reasoning_text = self._extract_reasoning_text(block)
                        if reasoning_text:
                            content_blocks.append(
                                ThinkingBlock(thinking=reasoning_text, signature=None, visibility=visibility)
                            )
                            event_blocks.append(ThinkingContent(text=reasoning_text))
                            text_accumulator.append(reasoning_text)
                        else:
                            # Reasoning block exists but no extractable content
                            placeholder = "[Internal reasoning encrypted or hidden by provider]"
                            content_blocks.append(
                                ThinkingBlock(thinking=placeholder, signature=None, visibility=visibility)
                            )
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
                    content_items, summary_items, visibility = self._collect_reasoning_metadata(block)
                    if content_items or summary_items:
                        content_blocks.append(
                            ResponseReasoningBlock(content=content_items, summary=summary_items, visibility=visibility)
                        )
                        flattened = self._extract_reasoning_text(block) or self._flatten_reasoning_summary(
                            summary_items
                        )
                        if flattened:
                            text_accumulator.append(flattened)
                            event_blocks.append(ThinkingContent(text=flattened))
                    else:
                        reasoning_text = self._extract_reasoning_text(block)
                        if reasoning_text:
                            content_blocks.append(
                                ThinkingBlock(thinking=reasoning_text, signature=None, visibility=visibility)
                            )
                            event_blocks.append(ThinkingContent(text=reasoning_text))
                            text_accumulator.append(reasoning_text)
                        else:
                            # Reasoning block exists but no extractable content
                            placeholder = "[Internal reasoning encrypted or hidden by provider]"
                            content_blocks.append(
                                ThinkingBlock(thinking=placeholder, signature=None, visibility=visibility)
                            )
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

        usage_counts = self._extract_usage_counts(response.usage if hasattr(response, "usage") else None)
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
