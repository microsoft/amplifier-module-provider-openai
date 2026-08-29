"""Response handling for OpenAI Responses API.

This module handles conversion of OpenAI API responses to Amplifier's ChatResponse format,
including reasoning extraction, incomplete response continuation, and reasoning state preservation.

Following the "bricks and studs" philosophy, this is a self-contained module that can be
regenerated independently of the main provider code.
"""

import json
import logging
from typing import Any

from amplifier_core import TextContent
from amplifier_core import ThinkingContent
from amplifier_core import ToolCallContent
from amplifier_core import llm_errors as kernel_errors
from amplifier_core.message_models import TextBlock
from amplifier_core.message_models import ThinkingBlock
from amplifier_core.message_models import ToolCall
from amplifier_core.message_models import ToolCallBlock
from amplifier_core.message_models import Usage

from ._constants import METADATA_CONTINUATION_COUNT
from ._constants import METADATA_INCOMPLETE_REASON
from ._constants import METADATA_REASONING_ITEMS
from ._constants import METADATA_RESPONSE_ID
from ._constants import METADATA_STATUS

logger = logging.getLogger(__name__)


class FunctionCallTruncationError(kernel_errors.LLMError):
    """A function_call was truncated (max_output_tokens mid-arguments) and cannot be executed.

    Raised instead of surfacing the call with empty/garbage arguments. A
    truncated function_call cannot be resumed by the Responses API
    continuation mechanism — each continuation restarts and truncates again
    (observed live: 5 fruitless continuations, then stitched `{}`-argument
    calls keyed by item id that 400'd the session). Failing loud here is the
    contract: no `{}`-argument calls, no item-id-keyed dispatch, no silent
    degradation.
    """

    def __init__(
        self, message: str, *, provider: str | None = None, model: str | None = None
    ) -> None:
        super().__init__(message, provider=provider, model=model, retryable=False)


def _function_call_fields(block: Any) -> tuple[str | None, str | None, str, Any, Any]:
    """Extract (call_id, item_id, name, arguments, status) from SDK object or dict."""
    if isinstance(block, dict):
        call_id = block.get("call_id")
        item_id = block.get("id")
        name = block.get("name", "")
        arguments = block.get("input")
        if arguments is None:
            arguments = block.get("arguments")
        status = block.get("status")
    else:
        call_id = getattr(block, "call_id", None)
        item_id = getattr(block, "id", None)
        name = getattr(block, "name", "")
        arguments = getattr(block, "input", None)
        if arguments is None:
            arguments = getattr(block, "arguments", None)
        status = getattr(block, "status", None)
    return call_id, item_id, name, arguments, status


def describe_incomplete_function_calls(output_items: list[Any]) -> list[dict[str, Any]]:
    """Non-raising detection of function_call items that must never be executed.

    A call is unexecutable when its status is "incomplete" (truncated by
    max_output_tokens) or its arguments are a non-empty string that does not
    parse as JSON (the truncation artifact). Returns one summary dict per
    offending item so the caller can decide retry-vs-fail.
    """
    problems: list[dict[str, Any]] = []
    for block in output_items:
        block_type = (
            block.get("type")
            if isinstance(block, dict)
            else getattr(block, "type", None)
        )
        if block_type not in {"tool_call", "function_call"}:
            continue
        call_id, item_id, name, arguments, status = _function_call_fields(block)
        reason = None
        if status == "incomplete":
            reason = "status_incomplete"
        elif isinstance(arguments, str) and arguments.strip():
            try:
                json.loads(arguments)
            except json.JSONDecodeError:
                reason = "arguments_unparseable"
        if reason:
            problems.append(
                {
                    "reason": reason,
                    "call_id": call_id,
                    "item_id": item_id,
                    "name": name,
                    "arguments_len": len(arguments)
                    if isinstance(arguments, str)
                    else None,
                }
            )
    return problems


def parse_function_call_block(block: Any) -> tuple[str, str, dict[str, Any]]:
    """Parse a function_call output item into (tool_id, tool_name, arguments).

    Invariants (the P4 contract):
    - tool_id is the pairing key: `call_id` PREFERRED over the item `id`.
      Tool outputs must pair by call_id (`call_…`); dispatching by the
      Responses-API item id (`fc_…`) makes outputs unpairable and 400s the
      next request.
    - An incomplete (truncated) call raises FunctionCallTruncationError —
      it is NEVER surfaced as an executable call with `{}` arguments.
      Truncated-but-unlabeled calls never reach this function: the
      response-level retry loop (gated on response.status == "incomplete")
      intercepts them via describe_incomplete_function_calls before
      conversion.
    - A NON-incomplete call whose arguments do not parse as JSON is a known
      model failure mode, not a truncation symptom. It is coerced to `{}`
      with a loud warning so the tool-error feedback loop can drive
      recovery — raising here would kill the session on a COMPLETED
      response (FunctionCallTruncationError is non-retryable and caught
      nowhere upstream).
    - An empty/absent arguments payload is a legitimate no-argument call.
    """
    call_id, item_id, name, arguments, status = _function_call_fields(block)

    if status == "incomplete":
        raise FunctionCallTruncationError(
            f"function_call '{name}' (call_id={call_id or item_id}) has status "
            "'incomplete' — its arguments were truncated by max_output_tokens "
            "and it cannot be executed"
        )

    tool_input: Any = arguments
    if isinstance(tool_input, str):
        if not tool_input.strip():
            tool_input = {}
        else:
            try:
                tool_input = json.loads(tool_input)
            except json.JSONDecodeError:
                logger.warning(
                    "[PROVIDER] function_call '%s' (call_id=%s, status=%s) "
                    "carries unparseable JSON arguments (%d chars) on a "
                    "non-incomplete call — coercing to {} so the tool error "
                    "surfaces to the model instead of killing the session. "
                    "Argument prefix: %.120s",
                    name,
                    call_id or item_id,
                    status,
                    len(arguments),
                    arguments,
                )
                tool_input = {}
    if tool_input is None:
        tool_input = {}
    if not isinstance(tool_input, dict):
        tool_input = {}

    return call_id or item_id or "", name, tool_input


def merge_discarded_usage(usage: Usage, discarded_usage_objs: list[Any]) -> Usage:
    """Fold token counts from discarded attempts into a reported Usage.

    The truncation-retry policy discards a truncated attempt's output and
    retries with a raised budget. The discarded attempt was still BILLED —
    a full input pass plus up to the previous output budget (including
    reasoning tokens) — so its usage must be reported, not silently dropped.

    Each discarded attempt's input is NORMALIZED to the kernel Usage
    contract before it is added, exactly as the final response's input is
    normalized by the two conversion paths: OpenAI's raw
    ``usage.input_tokens`` already CONTAINS ``cache_write_tokens`` as a
    SUBSET, whereas the contract defines ``input_tokens`` as "fresh +
    cache_read, cache_write NOT included" so that every consumer can
    compute ``total_input = input_tokens + cache_write_tokens``. Adding a
    discarded attempt's RAW input to the already-normalized ``usage``
    would let that formula double-count the discarded attempt's cache
    write (measured: a 4,000-token write reported a 24,000 gross where the
    truth was 20,000). Subtract it out per attempt, then contribute the
    write to ``cache_write_tokens`` exactly once.

    Args:
        usage: The Usage built from the final (kept) response, already
            normalized to the kernel contract.
        discarded_usage_objs: SDK usage objects from discarded attempts
            (entries may be None when a response carried no usage).

    Returns:
        A new Usage with the discarded attempts' tokens added in, still
        satisfying the contract (gross input reconstructs as
        ``input_tokens + cache_write_tokens``).
    """
    add_input = 0
    add_output = 0
    add_reasoning = 0
    add_cache_read = 0
    add_cache_write = 0
    saw_reasoning = False
    saw_cache_read = False
    saw_cache_write = False

    for usage_obj in discarded_usage_objs:
        if usage_obj is None:
            continue
        raw_input = getattr(usage_obj, "input_tokens", 0) or 0
        add_output += getattr(usage_obj, "output_tokens", 0) or 0
        output_details = getattr(usage_obj, "output_tokens_details", None)
        if output_details is not None:
            reasoning = getattr(output_details, "reasoning_tokens", None)
            if reasoning is not None:
                saw_reasoning = True
                add_reasoning += reasoning
        obj_cache_write = 0
        input_details = getattr(usage_obj, "input_tokens_details", None)
        if input_details is not None:
            cached = getattr(input_details, "cached_tokens", None)
            if cached is not None:
                saw_cache_read = True
                add_cache_read += cached
            cache_write = getattr(input_details, "cache_write_tokens", None)
            if cache_write is not None:
                saw_cache_write = True
                obj_cache_write = cache_write or 0
                add_cache_write += obj_cache_write
        # Normalize this attempt's input to the kernel contract before
        # adding it to the already-normalized `usage`: the raw vendor total
        # contains cache_write as a SUBSET, and the write is contributed
        # separately via `add_cache_write` above. max(0, ...) mirrors the
        # guard the conversion paths apply to a malformed payload.
        add_input += max(0, raw_input - obj_cache_write)

    new_input = usage.input_tokens + add_input
    new_output = usage.output_tokens + add_output
    updates: dict[str, Any] = {
        "input_tokens": new_input,
        "output_tokens": new_output,
        "total_tokens": new_input + new_output,
    }
    if usage.reasoning_tokens is not None or saw_reasoning:
        updates["reasoning_tokens"] = (usage.reasoning_tokens or 0) + add_reasoning
    if usage.cache_read_tokens is not None or saw_cache_read:
        updates["cache_read_tokens"] = (usage.cache_read_tokens or 0) + add_cache_read
    if usage.cache_write_tokens is not None or saw_cache_write:
        updates["cache_write_tokens"] = (
            usage.cache_write_tokens or 0
        ) + add_cache_write
    return usage.model_copy(update=updates)


def extract_reasoning_text(reasoning_summary: Any) -> str | None:
    """Extract reasoning text from various summary formats.

    OpenAI returns reasoning summaries in different formats depending on the response.
    This handles all known formats and extracts the text content.

    Args:
        reasoning_summary: The summary field from a reasoning block

    Returns:
        Extracted text or None if no text found
    """
    reasoning_text = None

    if isinstance(reasoning_summary, list):
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
        reasoning_text = reasoning_summary.get("text", str(reasoning_summary))
    elif hasattr(reasoning_summary, "text"):
        reasoning_text = getattr(reasoning_summary, "text", str(reasoning_summary))

    return reasoning_text if reasoning_text else None


def convert_response_with_accumulated_output(
    final_response: Any,
    accumulated_output: list[Any],
    continuation_count: int,
    chat_response_class: type,
) -> Any:
    """Convert OpenAI response with accumulated output to ChatResponse.

    This handles responses that may have been continued multiple times due to
    incomplete status. All output from all continuations is accumulated and
    merged into a single ChatResponse.

    Args:
        final_response: The final (completed) response object from OpenAI
        accumulated_output: All output items from all continuation calls
        continuation_count: Number of continuations made (0 if no continuations)
        chat_response_class: The ChatResponse class to instantiate (allows OpenAIChatResponse)

    Returns:
        ChatResponse with all accumulated content and metadata
    """
    content_blocks = []
    tool_calls = []
    event_blocks: list[TextContent | ThinkingContent | ToolCallContent] = []
    text_accumulator: list[str] = []
    reasoning_item_ids: list[str] = []

    # Process ALL accumulated output items
    for block in accumulated_output:
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
                                    text=text, raw=getattr(content_item, "raw", None)
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

                # Extract reasoning summary
                reasoning_summary = getattr(block, "summary", None) or getattr(
                    block, "text", None
                )
                reasoning_text = extract_reasoning_text(reasoning_summary)

                # Create thinking block if there's reasoning text OR encrypted state to preserve
                if reasoning_text or encrypted_content:
                    # Named dict, NOT a positional list -- see Change B in
                    # stateless-reset-fix-spec.md: amplifier_foundation's
                    # sanitize_for_json drops None from lists, which silently
                    # collapsed [None, "rs_abc"] -> ["rs_abc"] and made every
                    # block captured without ciphertext permanently
                    # unreplayable after resume. A dict survives the same
                    # key-dropping without losing what each surviving value
                    # MEANS. See _decode_reasoning_state in __init__.py for
                    # the back-compat reader (this file has no reader of its
                    # own -- reasoning replay always goes through
                    # amplifier_module_provider_openai._convert_messages).
                    content_blocks.append(
                        ThinkingBlock(
                            thinking=reasoning_text
                            or "",  # May be empty when only encrypted_content exists
                            signature=None,
                            visibility="internal",
                            content=[
                                {
                                    "encrypted_content": encrypted_content,
                                    "id": reasoning_id,
                                    "summary": reasoning_text or None,
                                }
                            ],
                        )
                    )
                    event_blocks.append(ThinkingContent(text=reasoning_text or ""))
                    # NOTE: Do NOT add reasoning to text_accumulator - it's internal process, not response content

            elif block_type in {"tool_call", "function_call"}:
                # P4: call_id-keyed. Incomplete (truncated) calls raise
                # FunctionCallTruncationError; a non-incomplete call with
                # unparseable arguments coerces to {} with a loud warning
                # (survivable model failure, not truncation).
                tool_id, tool_name, tool_input = parse_function_call_block(block)
                content_blocks.append(
                    ToolCallBlock(id=tool_id, name=tool_name, input=tool_input)
                )
                tool_calls.append(
                    ToolCall(id=tool_id, name=tool_name, arguments=tool_input)
                )
                event_blocks.append(
                    ToolCallContent(id=tool_id, name=tool_name, arguments=tool_input)
                )

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

                # Extract reasoning summary
                reasoning_summary = block.get("summary") or block.get("text")
                reasoning_text = extract_reasoning_text(reasoning_summary)

                # Create thinking block if there's reasoning text OR encrypted state to preserve
                if reasoning_text or encrypted_content:
                    # Named dict, NOT a positional list -- see Change B in
                    # stateless-reset-fix-spec.md (same rationale as the
                    # SDK-object branch above).
                    content_blocks.append(
                        ThinkingBlock(
                            thinking=reasoning_text
                            or "",  # May be empty when only encrypted_content exists
                            signature=None,
                            visibility="internal",
                            content=[
                                {
                                    "encrypted_content": encrypted_content,
                                    "id": reasoning_id,
                                    "summary": reasoning_text or None,
                                }
                            ],
                        )
                    )
                    event_blocks.append(ThinkingContent(text=reasoning_text or ""))
                    # NOTE: Do NOT add reasoning to text_accumulator - it's internal process, not response content

            elif block_type in {"tool_call", "function_call"}:
                # P4: call_id-keyed (see SDK-object branch above for the
                # incomplete-vs-unparseable handling).
                tool_id, tool_name, tool_input = parse_function_call_block(block)
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

    # Extract usage from final response.
    #
    # OpenAI's usage.input_tokens (Responses API) is the RAW vendor gross
    # total: fresh + cache_read + cache_write ALL COMBINED (cache_write is a
    # SUBSET of it -- see _cost.py's `fresh_input = prompt_tokens - cached
    # - cache_write` derivation, confirmed against live gpt-5.6-sol usage).
    # This differs from Anthropic, where cache_write (cache_creation) is
    # reported as a genuinely DISJOINT bucket on top of input_tokens.
    #
    # The kernel Usage contract (amplifier_core CONTRACTS.md) normalizes
    # input_tokens to "gross total: fresh + cache_read combined, cache_write
    # NOT included" -- so every consumer can safely compute
    # `total_input = input_tokens + cache_write_tokens` regardless of
    # provider. Emitting the raw OpenAI total here (which already contains
    # cache_write) would let that formula double-count the write tokens.
    # Subtract cache_write_tokens out of the raw total before it becomes the
    # public Usage.input_tokens value.
    usage_obj = final_response.usage if hasattr(final_response, "usage") else None
    usage_counts = {"input": 0, "output": 0, "total": 0}
    _raw_input_tokens = 0
    if usage_obj:
        if hasattr(usage_obj, "input_tokens"):
            _raw_input_tokens = usage_obj.input_tokens
        if hasattr(usage_obj, "output_tokens"):
            usage_counts["output"] = usage_obj.output_tokens

    # Phase 2: Extract reasoning_tokens from output_tokens_details
    reasoning_tokens = None
    if usage_obj and hasattr(usage_obj, "output_tokens_details"):
        details = usage_obj.output_tokens_details
        if details and hasattr(details, "reasoning_tokens"):
            reasoning_tokens = details.reasoning_tokens

    # Extract cache_read_tokens (and, for GPT-5.6, cache_write_tokens) from
    # input_tokens_details. Field verified live on gpt-5.6-sol (2026-07-14):
    # usage.input_tokens_details.{cached_tokens, cache_write_tokens}.
    cache_read_tokens = None
    cache_write_tokens = None
    if usage_obj and hasattr(usage_obj, "input_tokens_details"):
        details = usage_obj.input_tokens_details
        if details and hasattr(details, "cached_tokens"):
            cache_read_tokens = details.cached_tokens  # 0 is a valid measurement
        if details and hasattr(details, "cache_write_tokens"):
            cache_write_tokens = details.cache_write_tokens  # GPT-5.6+; 0 valid

    # Normalize: cache_write is a SUBSET of the raw OpenAI total, so remove it
    # to get the "fresh + cache_read" gross the kernel contract expects.
    # max(0, ...) guards against a malformed/unexpected API payload where
    # cache_write_tokens would otherwise exceed input_tokens.
    usage_counts["input"] = max(0, _raw_input_tokens - (cache_write_tokens or 0))
    usage_counts["total"] = usage_counts["input"] + usage_counts["output"]

    usage = Usage(
        input_tokens=usage_counts["input"],
        output_tokens=usage_counts["output"],
        total_tokens=usage_counts["total"],
        reasoning_tokens=reasoning_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
    )

    # Build metadata with provider-specific state
    metadata = {}

    # Response ID (for next turn's previous_response_id)
    if hasattr(final_response, "id"):
        metadata[METADATA_RESPONSE_ID] = final_response.id

    # Status (should be "completed" after continuations, or "incomplete" if we gave up)
    if hasattr(final_response, "status"):
        metadata[METADATA_STATUS] = final_response.status

        # If still incomplete after all attempts, record the reason
        if final_response.status == "incomplete":
            incomplete_details = getattr(final_response, "incomplete_details", None)
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

    # Continuation count (for debugging/metrics)
    if continuation_count > 0:
        metadata[METADATA_CONTINUATION_COUNT] = continuation_count

    combined_text = "\n\n".join(text_accumulator).strip()

    chat_response = chat_response_class(
        content=content_blocks,
        tool_calls=tool_calls if tool_calls else None,
        usage=usage,
        finish_reason=getattr(final_response, "finish_reason", None),
        content_blocks=event_blocks if event_blocks else None,
        text=combined_text or None,
        metadata=metadata if metadata else None,
    )

    return chat_response
