"""
Codex provider module for Amplifier.
Integrates with the Codex CLI in non-interactive mode (codex exec).
"""

__all__ = [
    "mount",
    "CodexProvider",
    "SessionManager",
    "SessionMetadata",
    "SessionState",
]

# Amplifier module metadata
__amplifier_module_type__ = "provider"

import asyncio
import json
import logging
import re
import shutil
import time
import uuid
from typing import Any

from amplifier_core import ModelInfo
from amplifier_core import ModuleCoordinator
from amplifier_core import ProviderInfo
from amplifier_core import TextContent
from amplifier_core import ThinkingContent
from amplifier_core import ToolCallContent
from amplifier_core.message_models import ChatRequest
from amplifier_core.message_models import ChatResponse
from amplifier_core.message_models import Message
from amplifier_core.message_models import TextBlock
from amplifier_core.message_models import ToolCall
from amplifier_core.message_models import ToolCallBlock
from amplifier_core.message_models import Usage

from .sessions import SessionManager
from .sessions import SessionMetadata
from .sessions import SessionState

logger = logging.getLogger(__name__)


class CodexChatResponse(ChatResponse):
    """ChatResponse with additional fields for streaming UI compatibility."""

    content_blocks: list[TextContent | ThinkingContent | ToolCallContent] | None = None
    text: str | None = None


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

METADATA_SESSION_ID = "codex:session_id"
METADATA_DURATION_MS = "codex:duration_ms"

DEFAULT_MODEL = "gpt-5.2-codex"
DEFAULT_TIMEOUT = 300.0
DEFAULT_MAX_TOKENS = 64000

# Model specifications (Codex CLI recommended list)
MODELS = {
    "gpt-5.2-codex": {
        "id": "gpt-5.2-codex",
        "display_name": "GPT-5.2-Codex",
        "context_window": 200000,
        "max_output_tokens": 64000,
        "capabilities": ["tools", "streaming"],
    },
    "gpt-5.1-codex-mini": {
        "id": "gpt-5.1-codex-mini",
        "display_name": "GPT-5.1-Codex Mini",
        "context_window": 200000,
        "max_output_tokens": 64000,
        "capabilities": ["tools", "streaming", "fast"],
    },
    "gpt-5.1-codex-max": {
        "id": "gpt-5.1-codex-max",
        "display_name": "GPT-5.1-Codex Max",
        "context_window": 200000,
        "max_output_tokens": 64000,
        "capabilities": ["tools", "streaming"],
    },
    "gpt-5.1-codex": {
        "id": "gpt-5.1-codex",
        "display_name": "GPT-5.1-Codex",
        "context_window": 200000,
        "max_output_tokens": 64000,
        "capabilities": ["tools", "streaming"],
    },
    "gpt-5-codex": {
        "id": "gpt-5-codex",
        "display_name": "GPT-5-Codex",
        "context_window": 200000,
        "max_output_tokens": 64000,
        "capabilities": ["tools", "streaming"],
    },
    "gpt-5-codex-mini": {
        "id": "gpt-5-codex-mini",
        "display_name": "GPT-5-Codex Mini",
        "context_window": 200000,
        "max_output_tokens": 64000,
        "capabilities": ["tools", "streaming", "fast"],
    },
}


# -----------------------------------------------------------------------------
# Mount function
# -----------------------------------------------------------------------------


async def mount(
    coordinator: ModuleCoordinator, config: dict[str, Any] | None = None
) -> None:
    """Mount the Codex provider using Codex CLI.

    Args:
        coordinator: The module coordinator to mount to.
        config: Optional configuration dictionary.
    """
    config = config or {}

    cli_path = shutil.which("codex")
    if not cli_path:
        logger.warning("Codex CLI not found. Install with: npm i -g @openai/codex")
        return None

    provider = CodexProvider(config=config, coordinator=coordinator)
    await coordinator.mount("providers", provider, name="codex")
    logger.info("Mounted CodexProvider (Codex CLI - non-interactive)")
    return None


# -----------------------------------------------------------------------------
# Provider Implementation
# -----------------------------------------------------------------------------


class CodexProvider:
    """Codex CLI integration for Amplifier (non-interactive mode)."""

    name = "codex"
    api_label = "Codex CLI"

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        coordinator: ModuleCoordinator | None = None,
    ):
        """Initialize the Codex provider."""
        self.config = config or {}
        self.coordinator = coordinator

        # Configuration
        self.default_model = self.config.get("default_model", DEFAULT_MODEL)
        self.timeout = self.config.get("timeout", DEFAULT_TIMEOUT)
        self.debug = self.config.get("debug", False)
        self.max_tokens = self.config.get("max_tokens", DEFAULT_MAX_TOKENS)

        # Codex CLI flags
        self.profile = self.config.get("profile")
        self.sandbox = self.config.get("sandbox")
        self.skip_git_repo_check = self.config.get("skip_git_repo_check", True)
        self.full_auto = self.config.get("full_auto", False)

        # Track repaired tool call IDs to prevent infinite detection loops
        self._repaired_tool_ids: set[str] = set()

        # Session persistence for prompt caching across restarts
        self._session_manager = SessionManager(
            session_dir=self.config.get("session_dir"),
        )

        amplifier_session_id = self._get_amplifier_session_id()
        self._session_state = self._session_manager.get_or_create_session(
            session_id=amplifier_session_id,
            name=self.config.get("session_name", "amplifier-codex"),
        )

        # Valid tool names for filtering invalid tool calls
        self._valid_tool_names: set[str] = set()
        # Filtered tool calls fed back to Codex as unavailable
        self._filtered_tool_calls: list[dict[str, Any]] = []

    def _get_amplifier_session_id(self) -> str | None:
        """Get the Amplifier session ID from the coordinator."""
        if not self.coordinator:
            return None

        if hasattr(self.coordinator, "session"):
            session = getattr(self.coordinator, "session", None)
            if session and hasattr(session, "id"):
                return str(session.id)

        if hasattr(self.coordinator, "config"):
            config = getattr(self.coordinator, "config", {})
            if isinstance(config, dict) and "session_id" in config:
                return str(config["session_id"])

        return None

    def _get_codex_session_id(self) -> str | None:
        """Get the Codex CLI session ID for resumption."""
        return self._session_state.metadata.codex_session_id

    def _save_session(self) -> None:
        """Save the current session state to disk."""
        self._session_manager.save_session(self._session_state)
        if self.debug:
            efficiency = self._session_state.get_cache_efficiency()
            logger.debug(
                "[PROVIDER] Session saved: %s, cache efficiency: %.1f%%",
                self._session_state.metadata.session_id,
                efficiency * 100.0,
            )

    def get_info(self) -> ProviderInfo:
        """Return provider information."""
        return ProviderInfo(
            id="codex",
            display_name="Codex CLI",
            credential_env_vars=[],  # CLI handles auth via login/session
            capabilities=["streaming", "tools"],
            defaults={
                "model": self.default_model,
                "max_tokens": self.max_tokens,
                "timeout": self.timeout,
            },
            config_fields=[],
        )

    async def list_models(self) -> list[ModelInfo]:
        """List available Codex models (static list)."""
        return [
            ModelInfo(
                id=spec["id"],
                display_name=spec["display_name"],
                context_window=spec["context_window"],
                max_output_tokens=spec["max_output_tokens"],
                capabilities=spec["capabilities"],
                defaults={},
            )
            for spec in MODELS.values()
        ]

    def parse_tool_calls(self, response: ChatResponse) -> list[ToolCall]:
        """Parse tool calls from response and filter empty arguments."""
        if not response.tool_calls:
            return []

        valid_calls = []
        for tc in response.tool_calls:
            if not tc.arguments:
                logger.debug("Filtering out tool '%s' with empty arguments", tc.name)
                continue
            valid_calls.append(tc)

        return valid_calls

    # -------------------------------------------------------------------------
    # Tool result validation and repair
    # -------------------------------------------------------------------------

    def _find_missing_tool_results(
        self, messages: list[Message]
    ) -> list[tuple[int, str, str, dict]]:
        """Find tool calls without matching results."""
        tool_calls = {}
        tool_results = set()

        for idx, msg in enumerate(messages):
            if msg.role == "assistant" and isinstance(msg.content, list):
                for block in msg.content:
                    if hasattr(block, "type") and block.type == "tool_call":
                        tool_calls[block.id] = (idx, block.name, block.input)

            elif (
                msg.role == "tool" and hasattr(msg, "tool_call_id") and msg.tool_call_id
            ):
                tool_results.add(msg.tool_call_id)

        return [
            (msg_idx, call_id, name, args)
            for call_id, (msg_idx, name, args) in tool_calls.items()
            if call_id not in tool_results and call_id not in self._repaired_tool_ids
        ]

    def _create_synthetic_result(self, call_id: str, tool_name: str) -> Message:
        """Create synthetic error result for missing tool response."""
        return Message(
            role="tool",
            content=(
                "[SYSTEM ERROR: Tool result missing from conversation history]\n\n"
                f"Tool: {tool_name}\n"
                f"Call ID: {call_id}\n\n"
                "This indicates the tool result was lost after execution.\n"
                "Likely causes: context compaction bug, message parsing error, or state corruption.\n\n"
                "The tool may have executed successfully, but the result was lost.\n"
                "Please acknowledge this error and offer to retry the operation."
            ),
            tool_call_id=call_id,
            name=tool_name,
        )

    # -------------------------------------------------------------------------
    # Main completion method
    # -------------------------------------------------------------------------

    async def complete(self, request: ChatRequest, **kwargs: Any) -> ChatResponse:
        """Execute a completion request via Codex CLI (codex exec)."""
        self._valid_tool_names = set()
        if request.tools:
            for tool in request.tools:
                if hasattr(tool, "name"):
                    self._valid_tool_names.add(tool.name)
                elif isinstance(tool, dict) and "name" in tool:
                    self._valid_tool_names.add(tool["name"])

        previous_filtered_calls = self._filtered_tool_calls.copy()
        self._filtered_tool_calls = []

        missing = self._find_missing_tool_results(request.messages)
        if missing:
            logger.warning(
                "[PROVIDER] Codex: Detected %d missing tool result(s). Injecting synthetic errors.",
                len(missing),
            )

            from collections import defaultdict

            by_msg_idx: dict[int, list[tuple[str, str]]] = defaultdict(list)
            for msg_idx, call_id, tool_name, _ in missing:
                by_msg_idx[msg_idx].append((call_id, tool_name))

            for msg_idx in sorted(by_msg_idx.keys(), reverse=True):
                synthetics = []
                for call_id, tool_name in by_msg_idx[msg_idx]:
                    synthetics.append(self._create_synthetic_result(call_id, tool_name))
                    self._repaired_tool_ids.add(call_id)

                insert_pos = msg_idx + 1
                for i, synthetic in enumerate(synthetics):
                    request.messages.insert(insert_pos + i, synthetic)

            if self.coordinator and hasattr(self.coordinator, "hooks"):
                await self.coordinator.hooks.emit(
                    "provider:tool_sequence_repaired",
                    {
                        "provider": self.name,
                        "repair_count": len(missing),
                        "repairs": [
                            {"tool_call_id": call_id, "tool_name": tool_name}
                            for _, call_id, tool_name, _ in missing
                        ],
                    },
                )

        for tool_call in previous_filtered_calls:
            request.messages.append(
                Message(
                    role="tool",
                    content=(
                        "[SYSTEM NOTICE: Tool call rejected]\n\n"
                        f"Tool: {tool_call['name']}\n"
                        "is not available in the current context and cannot be called.\n"
                        "Please acknowledge this error and offer to retry the operation."
                    ),
                    tool_call_id=tool_call["id"],
                    name=tool_call["name"],
                )
            )

        start_time = time.time()

        cli_path = shutil.which("codex")
        if not cli_path:
            raise RuntimeError(
                "Codex CLI not found. Install with: npm i -g @openai/codex"
            )

        model = (
            kwargs.get("model") or getattr(request, "model", None) or self.default_model
        )

        request_metadata = getattr(request, "metadata", None) or {}
        existing_session_id = (
            request_metadata.get(METADATA_SESSION_ID) or self._get_codex_session_id()
        )
        resuming = existing_session_id is not None

        system_prompt, user_prompt = self._convert_messages(
            request.messages, request.tools, resuming=resuming
        )

        cmd = self._build_command(
            cli_path=cli_path,
            model=model,
            session_id=existing_session_id,
        )

        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
        else:
            full_prompt = user_prompt

        if self.debug:
            logger.debug("[PROVIDER] Command: %s ...", " ".join(cmd[:10]))
            logger.debug("[PROVIDER] System prompt length: %d", len(system_prompt))

        await self._emit_event(
            "llm:request",
            {
                "provider": self.name,
                "model": model,
                "messages_count": len(request.messages),
                "tools_count": len(request.tools) if request.tools else 0,
                "resume_session": existing_session_id is not None,
            },
        )

        response_data = await self._execute_cli(cmd, full_prompt)

        response_session_id = response_data.get("metadata", {}).get(METADATA_SESSION_ID)
        if response_session_id:
            self._session_state.set_codex_session_id(response_session_id)
            if self.debug:
                logger.debug(
                    "[PROVIDER] Stored Codex session ID for resumption: %s",
                    response_session_id,
                )

        duration = time.time() - start_time
        chat_response = self._build_response(response_data, duration)

        usage_data = response_data.get("usage", {})
        self._session_state.update_usage(
            input_tokens=usage_data.get("input_tokens", 0),
            output_tokens=usage_data.get("output_tokens", 0),
            cache_read=usage_data.get("cache_read_input_tokens", 0),
            cache_creation=usage_data.get("cache_creation_input_tokens", 0),
            duration_ms=int(duration * 1000),
        )
        self._save_session()

        await self._emit_event(
            "llm:response",
            {
                "provider": self.name,
                "model": model,
                "status": "ok",
                "duration_ms": int(duration * 1000),
                "usage": {
                    "input": chat_response.usage.input_tokens
                    if chat_response.usage
                    else 0,
                    "output": chat_response.usage.output_tokens
                    if chat_response.usage
                    else 0,
                },
                "has_tool_calls": bool(chat_response.tool_calls),
                "tool_calls_count": len(chat_response.tool_calls)
                if chat_response.tool_calls
                else 0,
            },
        )

        return chat_response

    # -------------------------------------------------------------------------
    # Message conversion
    # -------------------------------------------------------------------------

    def _convert_messages(
        self,
        messages: list[Message],
        tools: list[Any] | None,
        resuming: bool = False,
    ) -> tuple[str, str]:
        """Convert Amplifier messages to Codex CLI prompt format."""
        system_parts = []
        conversation_parts = []
        tool_schema = ""

        if resuming:
            messages = self._get_current_turn_messages(messages)
        else:
            if tools:
                tool_definitions = self._convert_tools(tools)
                tool_schema = self._build_tool_schema(tool_definitions)

        for msg in messages:
            role = msg.role
            content = self._extract_content(msg)

            if role == "system":
                if not resuming:
                    system_parts.append(f"<system-reminder>{content}</system-reminder>")

            elif role == "user":
                if content.strip().startswith("<system-reminder"):
                    conversation_parts.append(content)
                else:
                    conversation_parts.append(f"<user>{content}</user>")

            elif role == "assistant":
                assistant_content = self._format_assistant_message(msg)
                conversation_parts.append(f"<assistant>{assistant_content}</assistant>")

            elif role == "tool":
                tool_result = self._format_tool_result(msg)
                conversation_parts.append(f"{tool_result}")

            elif role == "developer":
                wrapped = f"<context_file>\n{content}\n</context_file>"
                conversation_parts.append(f"{wrapped}")

        if tool_schema:
            system_parts.append(tool_schema)
        system_prompt = "\n\n".join(system_parts) if system_parts else ""
        user_prompt = "\n\n".join(conversation_parts) if conversation_parts else ""

        if len(messages) == 1 and messages[0].role == "user":
            user_prompt = self._extract_content(messages[0])

        return system_prompt, user_prompt

    def _get_current_turn_messages(self, messages: list[Message]) -> list[Message]:
        """Get only messages from the current turn (after last assistant response)."""
        last_assistant_idx = -1
        for i, msg in enumerate(messages):
            if msg.role == "assistant":
                last_assistant_idx = i

        if last_assistant_idx == -1:
            return messages

        return messages[last_assistant_idx + 1 :]

    def _extract_content(self, msg: Message) -> str:
        """Extract text content from a message."""
        content = msg.content

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            text_parts = []
            for block in content:
                if hasattr(block, "text"):
                    text_parts.append(block.text)
                elif isinstance(block, dict) and "text" in block:
                    text_parts.append(block["text"])
            return "\n".join(text_parts)

        return str(content) if content else ""

    def _format_assistant_message(self, msg: Message) -> str:
        """Format an assistant message, including any tool calls."""
        parts = []

        content = msg.content
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if hasattr(block, "type"):
                    if block.type == "text":
                        parts.append(block.text)
                    elif block.type in ("tool_use", "tool_call"):
                        tool_call_str = json.dumps(
                            {
                                "tool": block.name,
                                "id": block.id,
                                "input": getattr(block, "input", {}),
                            }
                        )
                        parts.append(f"<tool_use>{tool_call_str}</tool_use>")
                elif isinstance(block, dict):
                    if block.get("type") == "text":
                        parts.append(block.get("text", ""))
                    elif block.get("type") in ("tool_use", "tool_call"):
                        tool_call_str = json.dumps(
                            {
                                "tool": block.get("name"),
                                "id": block.get("id"),
                                "input": block.get("input", {}),
                            }
                        )
                        parts.append(f"<tool_use>{tool_call_str}</tool_use>")

        return "\n".join(parts)

    def _format_tool_result(self, msg: Message) -> str:
        """Format a tool result message."""
        tool_call_id = getattr(msg, "tool_call_id", None)
        tool_name = getattr(msg, "name", "unknown")
        content = self._extract_content(msg)
        is_error = getattr(msg, "is_error", False)

        result = {
            "tool_call_id": tool_call_id,
            "tool": tool_name,
            "result": content,
            "is_error": is_error,
        }

        return f"<tool_result>{json.dumps(result)}</tool_result>"

    # -------------------------------------------------------------------------
    # Tool conversion
    # -------------------------------------------------------------------------

    def _convert_tools(self, tools: list[Any]) -> list[dict[str, Any]]:
        """Convert Amplifier tool specs to Codex prompt format."""
        tool_definitions = []

        for tool in tools:
            if hasattr(tool, "name"):
                tool_def = {
                    "name": tool.name,
                    "description": getattr(tool, "description", ""),
                    "input_schema": getattr(tool, "parameters", {}),
                }
            elif isinstance(tool, dict):
                tool_def = {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "input_schema": tool.get(
                        "parameters", tool.get("input_schema", {})
                    ),
                }
            else:
                continue

            tool_definitions.append(tool_def)

        return tool_definitions

    def _build_tool_schema(self, tools: list[dict[str, Any]]) -> str:
        """Build tool schema for system prompt."""
        if not tools:
            return ""

        tools_json = json.dumps(tools, indent=2)
        tool_use_example = json.dumps(
            {
                "tool": "tool_name",
                "id": "unique_id",
                "input": {"param1": "value1"},
            }
        )

        return (
            '<system-reminder source="tools-context">\n'
            "Available tools:\n"
            "<tools>\n"
            f"{tools_json}\n"
            "</tools>\n\n"
            "To call a tool, use this format:\n"
            "<tool_use>\n"
            f"{tool_use_example}\n"
            "</tool_use>\n\n"
            'Generate a unique ID for each call (e.g., "call_1", "call_2").\n'
            "Tool results will be provided in <tool_result> blocks.\n"
            "</system-reminder>"
        )

    # -------------------------------------------------------------------------
    # CLI execution
    # -------------------------------------------------------------------------

    def _build_command(
        self,
        cli_path: str,
        model: str,
        session_id: str | None,
    ) -> list[str]:
        """Build the Codex CLI command."""
        cmd: list[str] = [cli_path, "exec", "--json", "--model", model]

        if self.profile:
            cmd.extend(["--profile", str(self.profile)])
        if self.sandbox:
            cmd.extend(["--sandbox", str(self.sandbox)])
        if self.full_auto:
            cmd.append("--full-auto")
        if self.skip_git_repo_check:
            cmd.append("--skip-git-repo-check")

        if session_id:
            cmd = [cli_path, "exec", "resume", session_id, "--json", "--model", model]
            if self.profile:
                cmd.extend(["--profile", str(self.profile)])
            if self.sandbox:
                cmd.extend(["--sandbox", str(self.sandbox)])
            if self.full_auto:
                cmd.append("--full-auto")
            if self.skip_git_repo_check:
                cmd.append("--skip-git-repo-check")

        cmd.append("-")
        return cmd

    async def _execute_cli(self, cmd: list[str], prompt: str) -> dict[str, Any]:
        """Execute Codex CLI and parse JSONL output."""
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        assert proc.stdin is not None
        proc.stdin.write(prompt.encode("utf-8"))
        await proc.stdin.drain()
        proc.stdin.close()
        await proc.stdin.wait_closed()

        response_text = ""
        usage_data: dict[str, Any] = {}
        metadata: dict[str, Any] = {}
        tool_calls: list[dict[str, Any]] = []
        last_assistant_text = ""

        assert proc.stdout is not None

        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            line_str = line.decode("utf-8").strip()
            if not line_str:
                continue

            try:
                event_data = json.loads(line_str)
            except json.JSONDecodeError:
                if self.debug:
                    logger.warning(
                        "[PROVIDER] Failed to parse JSONL: %s", line_str[:100]
                    )
                continue

            event_type = event_data.get("type")

            if event_type == "thread.started":
                session_id = event_data.get("thread_id") or event_data.get("session_id")
                if session_id:
                    metadata[METADATA_SESSION_ID] = session_id

            elif event_type == "turn.completed":
                raw_usage = event_data.get("usage", {}) or {}
                usage_data = {
                    "input_tokens": raw_usage.get("input_tokens", 0),
                    "output_tokens": raw_usage.get("output_tokens", 0),
                    "cache_read_input_tokens": raw_usage.get(
                        "cached_input_tokens",
                        raw_usage.get("cache_read_input_tokens", 0),
                    ),
                }

            elif event_type == "turn.failed":
                error_message = event_data.get("error") or event_data.get("message")
                raise RuntimeError(f"Codex CLI failed turn: {error_message}")

            elif event_type == "error":
                error_message = event_data.get("message") or event_data.get("error")
                raise RuntimeError(f"Codex CLI error: {error_message}")

            elif isinstance(event_type, str) and event_type.startswith("item."):
                item = event_data.get("item", event_data)
                item_text, item_tool_calls = self._parse_item(item)
                if item_text:
                    last_assistant_text = item_text
                    response_text = (
                        f"{response_text}\n{item_text}".strip()
                        if response_text
                        else item_text
                    )
                if item_tool_calls:
                    tool_calls.extend(item_tool_calls)

        await proc.wait()

        if proc.returncode != 0:
            stderr_data = await proc.stderr.read() if proc.stderr else b""
            error_msg = stderr_data.decode("utf-8").strip()
            if error_msg:
                raise RuntimeError(
                    f"Codex CLI failed (exit {proc.returncode}): {error_msg}"
                )
            raise RuntimeError(
                "Codex CLI failed. Subscription limits or auth may be invalid."
            )

        if not response_text and last_assistant_text:
            response_text = last_assistant_text

        # De-duplicate tool calls by ID across all events
        if tool_calls:
            seen_ids: set[str] = set()
            deduped_calls: list[dict[str, Any]] = []
            for tc in tool_calls:
                call_id = tc.get("id")
                if call_id and call_id in seen_ids:
                    continue
                if call_id:
                    seen_ids.add(call_id)
                deduped_calls.append(tc)
            tool_calls = deduped_calls

        return {
            "text": response_text,
            "tool_calls": tool_calls,
            "usage": usage_data,
            "metadata": metadata,
        }

    def _parse_item(self, item: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
        """Parse a JSONL item event for text and tool calls."""
        tool_calls: list[dict[str, Any]] = []
        text_parts: list[str] = []

        item_type = item.get("type") or item.get("item_type")
        if item_type in {"message", "assistant_message", "agent_message", "assistant"}:
            text_parts.extend(self._extract_text_from_item(item))
            for text in self._extract_tool_calls_from_text("\n".join(text_parts)):
                tool_calls.append(text)

        if item_type in {"tool_call", "tool_use", "tool", "mcp_tool_call"}:
            tool_call = self._normalize_tool_call(item)
            if tool_call:
                tool_calls.append(tool_call)

        # Handle nested message shapes
        message = item.get("message")
        if isinstance(message, dict):
            text_parts.extend(self._extract_text_from_item(message))
            for text in self._extract_tool_calls_from_text("\n".join(text_parts)):
                tool_calls.append(text)

        # De-duplicate tool calls by id
        if tool_calls:
            seen = set()
            deduped = []
            for tc in tool_calls:
                call_id = tc.get("id")
                if call_id and call_id in seen:
                    continue
                if call_id:
                    seen.add(call_id)
                deduped.append(tc)
            tool_calls = deduped

        return ("\n".join(text_parts).strip(), tool_calls)

    def _extract_text_from_item(self, item: dict[str, Any]) -> list[str]:
        """Extract plain text from a JSONL item/message."""
        texts: list[str] = []
        if "text" in item and isinstance(item["text"], str):
            texts.append(item["text"])

        content = item.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    block_type = block.get("type")
                    if block_type in {"text", "output_text"} and "text" in block:
                        texts.append(block.get("text", ""))
        elif isinstance(content, str):
            texts.append(content)

        return [t for t in texts if t]

    def _normalize_tool_call(self, item: dict[str, Any]) -> dict[str, Any] | None:
        """Normalize tool call fields from JSONL item."""
        name = item.get("name") or item.get("tool") or item.get("tool_name")
        arguments = item.get("arguments") or item.get("input") or item.get("params")
        call_id = item.get("id") or item.get("tool_call_id") or item.get("call_id")

        if not name:
            return None

        if call_id is None:
            call_id = f"call_{uuid.uuid4().hex[:8]}"

        if arguments is None:
            arguments = {}

        return {"id": call_id, "name": name, "arguments": arguments}

    # -------------------------------------------------------------------------
    # Response building
    # -------------------------------------------------------------------------

    def _extract_tool_calls_from_text(self, text: str) -> list[dict[str, Any]]:
        """Extract tool calls from <tool_use>...</tool_use> blocks."""
        tool_calls = []
        pattern = r"<tool_use>\s*(.*?)\s*</tool_use>"
        matches = re.findall(pattern, re.sub(r"```[\s\S]*?```", "", text), re.DOTALL)

        for match in matches:
            stripped = match.strip()
            if not stripped.startswith("{"):
                continue
            try:
                tool_data = json.loads(match)
                tool_call = {
                    "id": tool_data.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                    "name": tool_data.get("tool", tool_data.get("name", "")),
                    "arguments": tool_data.get("input", tool_data.get("arguments", {})),
                }

                if (
                    self._valid_tool_names
                    and tool_call["name"] not in self._valid_tool_names
                ):
                    self._filtered_tool_calls.append(tool_call)
                    continue

                tool_calls.append(tool_call)
            except json.JSONDecodeError:
                continue

        return tool_calls

    def _clean_response_text(self, text: str) -> str:
        """Remove tool_use blocks from response text."""
        cleaned = re.sub(r"<tool_use>.*?</tool_use>", "", text, flags=re.DOTALL)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    def _build_response(
        self, response_data: dict[str, Any], duration: float
    ) -> CodexChatResponse:
        """Build a CodexChatResponse from parsed response data."""
        raw_text = response_data.get("text", "")
        tool_call_dicts = response_data.get("tool_calls", [])
        usage_data = response_data.get("usage", {})
        metadata = response_data.get("metadata", {})

        clean_text = self._clean_response_text(raw_text)

        content_blocks: list[Any] = []
        event_blocks: list[TextContent | ThinkingContent | ToolCallContent] = []

        if clean_text:
            content_blocks.append(TextBlock(text=clean_text))
            event_blocks.append(TextContent(text=clean_text))

        for tc in tool_call_dicts:
            content_blocks.append(
                ToolCallBlock(
                    id=tc["id"],
                    name=tc["name"],
                    input=tc["arguments"],
                )
            )
            event_blocks.append(
                ToolCallContent(
                    id=tc["id"],
                    name=tc["name"],
                    arguments=tc["arguments"],
                )
            )

        tool_calls: list[ToolCall] | None = None
        if tool_call_dicts:
            tool_calls = [
                ToolCall(
                    id=tc["id"],
                    name=tc["name"],
                    arguments=tc["arguments"],
                )
                for tc in tool_call_dicts
            ]

        input_tokens = usage_data.get("input_tokens", 0)
        output_tokens = usage_data.get("output_tokens", 0)
        total_input = input_tokens

        usage = Usage(
            input_tokens=total_input,
            output_tokens=output_tokens,
            total_tokens=total_input + output_tokens,
        )

        finish_reason = "tool_use" if tool_calls else "end_turn"

        metadata[METADATA_DURATION_MS] = int(duration * 1000)

        return CodexChatResponse(
            content=content_blocks,
            tool_calls=tool_calls,
            usage=usage,
            finish_reason=finish_reason,
            metadata=metadata,
            content_blocks=event_blocks if event_blocks else None,
            text=clean_text or None,
        )

    # -------------------------------------------------------------------------
    # Event emission
    # -------------------------------------------------------------------------

    async def _emit_event(self, event: str, data: dict[str, Any]) -> None:
        """Emit an event through the coordinator's hooks if available."""
        if self.coordinator and hasattr(self.coordinator, "hooks"):
            await self.coordinator.hooks.emit(event, data)
