"""
OpenAI provider module for Amplifier.
Integrates with OpenAI's Responses API.
"""

__all__ = ["mount", "OpenAIProvider"]

# Amplifier module metadata
__amplifier_module_type__ = "provider"

import asyncio
import inspect
import json
import logging
import os
import time
import uuid
from collections import defaultdict
from decimal import Decimal
from typing import Any

import openai
from pydantic import ValidationError

from amplifier_core import ConfigField
from amplifier_core import ModelInfo
from amplifier_core import ModuleCoordinator
from amplifier_core import ProviderInfo
from amplifier_core import TextContent
from amplifier_core import ThinkingContent
from amplifier_core import ToolCallContent
from amplifier_core import llm_errors as kernel_errors
from amplifier_core.events import PROVIDER_RETRY
from amplifier_core.models import HookResult
from amplifier_core.utils import redact_secrets
from amplifier_core.message_models import ChatRequest
from amplifier_core.message_models import ChatResponse
from amplifier_core.message_models import ToolCall
from amplifier_core.utils.retry import RetryConfig, retry_with_backoff
from openai import AsyncOpenAI

from ._constants import BACKGROUND_POLLING_STATUSES
from ._constants import BACKGROUND_STATUS_FAILED
from ._constants import DEFAULT_BACKGROUND_TIMEOUT
from ._constants import DEFAULT_MAX_TOKENS
from ._constants import DEFAULT_MODEL
from ._constants import DEFAULT_POLL_INTERVAL
from ._constants import DEFAULT_REASONING_SUMMARY
from ._constants import DEFAULT_TIMEOUT
from ._constants import DEFAULT_PROMPT_CACHE_RETENTION
from ._constants import DEFAULT_TRUNCATION
from ._constants import DEEP_RESEARCH_MODELS
from ._constants import MAX_CONTINUATION_ATTEMPTS
from ._constants import METADATA_INCOMPLETE_REASON
from ._constants import METADATA_REASONING_ITEMS
from ._constants import METADATA_RESPONSE_ID
from ._constants import METADATA_STATUS
from ._constants import NATIVE_TOOL_TYPES
from ._constants import RESPONSE_CHAIN_INVALIDATED
from ._constants import RESPONSE_NOT_FOUND_ERROR_CODES
from ._response_handling import FunctionCallTruncationError
from ._response_handling import convert_response_with_accumulated_output
from ._response_handling import describe_incomplete_function_calls
from ._response_handling import extract_reasoning_text
from ._response_handling import parse_function_call_block
from ._capabilities import get_capabilities
from ._cost import compute_cost

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Process-wide concurrency gate
# ---------------------------------------------------------------------------
# Shared across ALL OpenAIProvider instances in this process (including
# parent + delegated child sessions). Prevents blast patterns that trigger
# Cloudflare bot detection when many sessions delegate simultaneously.
# Created lazily on the first API call; keyed by event loop so that tests
# using asyncio.run() get fresh semaphores rather than inheriting stale state.

_process_semaphore: asyncio.Semaphore | None = None
_process_semaphore_loop: Any = None  # asyncio.AbstractEventLoop
_process_semaphore_max: int = 0
_active_requests: int = 0  # currently holding semaphore (executing)
_waiting_requests: int = 0  # waiting to acquire semaphore


async def _get_process_semaphore(max_concurrent: int) -> asyncio.Semaphore | None:
    """Get or create the process-wide concurrency semaphore.

    Returns ``None`` when ``max_concurrent <= 0`` (semaphore disabled).
    Recreates the semaphore when called from a different event loop so that
    unit tests using ``asyncio.run()`` always get a fresh, valid semaphore.
    """
    global _process_semaphore, _process_semaphore_loop, _process_semaphore_max
    if max_concurrent <= 0:
        return None
    current_loop = asyncio.get_running_loop()
    if (
        _process_semaphore is None
        or _process_semaphore_loop is not current_loop
        or _process_semaphore_max != max_concurrent
    ):
        _process_semaphore = asyncio.Semaphore(max_concurrent)
        _process_semaphore_loop = current_loop
        _process_semaphore_max = max_concurrent
    return _process_semaphore


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

    _totals: dict = {"cost_usd": None, "has_data": False}

    def _add_cost(cost) -> None:
        if cost is not None:
            _totals["cost_usd"] = (_totals["cost_usd"] or Decimal("0")) + cost
            _totals["has_data"] = True

    # Get API key from config or environment
    api_key = config.get("api_key") or os.environ.get("OPENAI_API_KEY")

    if not api_key:
        logger.warning("No API key found for OpenAI provider")
        return None

    provider = OpenAIProvider(
        api_key=api_key, config=config, coordinator=coordinator, add_cost=_add_cost
    )
    await coordinator.mount("providers", provider, name="openai")

    # Break the OpenAI Responses API response chain whenever the context is
    # compacted, so the provider stops rebuilding the pre-compaction
    # server-side context via previous_response_id (which drives unbounded
    # input-token growth -> context_length_exceeded). The default context
    # module emits the literal "context:compaction"; other context managers use
    # the kernel's "context:pre_compact"/"context:post_compact". Subscribe to
    # all three so the fix survives a swapped context module.
    async def _on_compaction(event: str, data: dict[str, Any]) -> HookResult:
        provider._reset_chain_on_next_request = True
        logger.info(
            "[PROVIDER] Compaction event '%s' received; breaking OpenAI "
            "response chain on next request.",
            event,
        )
        return HookResult()

    if hasattr(coordinator, "hooks") and coordinator.hooks is not None:
        for _compaction_event in (
            "context:compaction",
            "context:pre_compact",
            "context:post_compact",
        ):
            try:
                coordinator.hooks.on(_compaction_event, _on_compaction)
            except Exception as sub_err:  # pragma: no cover - defensive
                logger.warning(
                    "[PROVIDER] Could not subscribe to '%s' for compaction "
                    "chain reset: %s",
                    _compaction_event,
                    sub_err,
                )

    coordinator.register_contributor(
        "session.cost",
        "provider-openai",
        lambda: (
            {
                "cost_usd": str(_totals["cost_usd"])
                if _totals["cost_usd"] is not None
                else None
            }
            if _totals["has_data"]
            else None
        ),
    )
    logger.info("Mounted OpenAIProvider (Responses API)")

    # Return cleanup function
    async def cleanup():
        await provider.close()

    return cleanup


# gpt-5.5-pro accepts only {medium, high, xhigh} (verified live API 2026-04-24).
# Catching disallowed values pre-flight gives callers a clear error instead of
# an opaque API HTTP 400.
_GPT_5_5_PRO_ALLOWED_EFFORTS = frozenset({"medium", "high", "xhigh"})


def _validate_gpt_5_5_pro_effort(model_id: str, reasoning_param: Any) -> None:
    """Reject gpt-5.5-pro requests whose reasoning.effort is not in the allowed set.

    Runs once per request inside _build_params(). No-op for any model that does
    not start with 'gpt-5.5-pro' (so dated snapshots like
    'gpt-5.5-pro-2026-04-23' are also covered).

    Raises:
        kernel_errors.InvalidRequestError: if the resolved effort is set to a
            value the live API would reject.
    """
    if not model_id.startswith("gpt-5.5-pro"):
        return
    if reasoning_param is None:
        return
    if isinstance(reasoning_param, dict):
        effort = reasoning_param.get("effort")
    else:
        effort = reasoning_param
    if effort is None or effort in _GPT_5_5_PRO_ALLOWED_EFFORTS:
        return
    raise kernel_errors.InvalidRequestError(
        f"Model {model_id!r} requires reasoning.effort in "
        f"{{'medium', 'high', 'xhigh'}}; got {effort!r}. "
        f"gpt-5.5-pro rejects 'minimal', 'none', and 'low' "
        f"(verified against live API 2026-04-24). "
        f"Set reasoning.effort to one of the allowed values "
        f"or omit it to use the model default."
    )


# Full vocabulary of reasoning.effort values any curated model accepts.
# "minimal".."xhigh" per OpenAI docs; "max" added by gpt-5.6 (see README).
# Used for mount-time validation of the canonical `reasoning_effort` config
# key: a value outside this set can never succeed at request time, so it
# fails loud at mount instead of 400ing mid-session.
_KNOWN_REASONING_EFFORTS = frozenset(
    {"none", "minimal", "low", "medium", "high", "xhigh", "max"}
)


def _resolve_config_reasoning_effort(value: Any, model_id: str) -> str | None:
    """Validate/normalize the canonical `reasoning_effort` config key at mount.

    Returns the normalized effort string, or None when the key should not
    inject a reasoning param:
      - value is None / "" (key absent or blank)
      - value is "none" — the provisioned ConfigField default, meaning "use
        the provider/model default behavior". This deliberately does NOT emit
        reasoning={"effort": "none"}: absence must not start injecting a
        value (OpenAI's own default-medium guidance), and existing installs
        carry reasoning_effort="none" from the provisioning UI.

    Raises:
        ValueError: when the value is not a recognized effort, or when the
            default model's accepted set is known and excludes it
            (gpt-5.5-pro accepts only {medium, high, xhigh}). Failing here
            surfaces at mount — loud and immediate — instead of as an API
            HTTP 400 mid-session.
    """
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if not normalized or normalized == "none":
        return None
    if normalized not in _KNOWN_REASONING_EFFORTS:
        raise ValueError(
            f"Invalid config 'reasoning_effort'={value!r} for provider-openai. "
            f"Valid values: {', '.join(sorted(_KNOWN_REASONING_EFFORTS))}. "
            f"Fix the provider config (settings.yaml / bundle config block)."
        )
    # Per-model accepted sets, where known. gpt-5.5-pro is the only model
    # with a verified restricted set today (live API 2026-04-24).
    if model_id.startswith("gpt-5.5-pro") and (
        normalized not in _GPT_5_5_PRO_ALLOWED_EFFORTS
    ):
        raise ValueError(
            f"Config 'reasoning_effort'={value!r} is not accepted by model "
            f"{model_id!r}: gpt-5.5-pro requires one of "
            f"{sorted(_GPT_5_5_PRO_ALLOWED_EFFORTS)} "
            f"(verified against live API 2026-04-24)."
        )
    return normalized


# reasoning.mode selects the reasoning strategy. Verified live against gpt-5.6-sol
# 2026-07-14: mode in {"standard", "pro"} ("pro" = deeper internal reasoning, one
# final answer). Only forwarded when the caller sets it; models that do not support
# a mode (pre-5.6) reject it loudly at the API, which is the desired fail-loud.
_REASONING_MODE_ALLOWED = frozenset({"standard", "pro"})


def _validate_reasoning_mode(reasoning_param: Any) -> None:
    """Reject a reasoning.mode value the API would not accept.

    Validates the value (not model support) when present -- a clear pre-flight
    error instead of an opaque HTTP 400. No-op unless reasoning_param is a dict
    carrying a non-None 'mode'.
    """
    if not isinstance(reasoning_param, dict):
        return
    mode = reasoning_param.get("mode")
    if mode is None or mode in _REASONING_MODE_ALLOWED:
        return
    raise kernel_errors.InvalidRequestError(
        f"reasoning.mode must be one of {{'standard', 'pro'}}; got {mode!r}. "
        f"'pro' requires a GPT-5.6 model (verified against live API 2026-07-14)."
    )


# prompt_cache_options.mode is a stable enum; ttl currently accepts only "30m" but
# is left to the API to validate (it is the more volatile field). Verified live
# against gpt-5.6-sol 2026-07-14. Note: prompt_cache_options COEXISTS with
# prompt_cache_retention -- it does not replace it.
_PROMPT_CACHE_OPTIONS_MODES = frozenset({"implicit", "explicit"})

_CONTEXT_OVERFLOW_ERROR_CODES = frozenset({"context_length_exceeded"})

# Substring fallbacks. Kept for older Chat Completions wording and for
# errors that arrive without a machine-readable code. "exceeds the context
# window" is the current Responses API phrasing, which none of the legacy
# markers matched.
_CONTEXT_OVERFLOW_MESSAGE_MARKERS = (
    "context length",
    "too many tokens",
    "maximum context",
    "exceeds the context window",
)


def _extract_error_fields(body: object) -> tuple[str | None, str | None]:
    """Return (code, type) from an OpenAI error body, tolerating shapes."""
    if not isinstance(body, dict):
        return None, None
    err = body.get("error")
    if isinstance(err, dict):
        return err.get("code"), err.get("type")
    return body.get("code"), body.get("type")


def _is_context_overflow(err_code: str | None, raw_msg: str) -> bool:
    """True when an OpenAI error denotes context-window overflow.

    Code first -- stable, and mirrors the existing 404 chain-invalidation
    precedent in this file (``err_code in RESPONSE_NOT_FOUND_ERROR_CODES``).
    Message substrings are a fallback only.
    """
    if err_code is not None and err_code in _CONTEXT_OVERFLOW_ERROR_CODES:
        return True
    return any(m in raw_msg for m in _CONTEXT_OVERFLOW_MESSAGE_MARKERS)


def _validate_prompt_cache_options(options: Any) -> None:
    """Validate the prompt_cache_options object shape/mode enum pre-flight."""
    if not isinstance(options, dict):
        raise kernel_errors.InvalidRequestError(
            f"prompt_cache_options must be an object with 'mode'/'ttl'; "
            f"got {type(options).__name__}."
        )
    mode = options.get("mode")
    if mode is not None and mode not in _PROMPT_CACHE_OPTIONS_MODES:
        raise kernel_errors.InvalidRequestError(
            f"prompt_cache_options.mode must be one of "
            f"{{'implicit', 'explicit'}}; got {mode!r}."
        )


def _drop_unsupported_in_memory_retention(
    model_id: str, retention: str | None
) -> str | None:
    """Return *retention* unless it would be rejected by the model.

    OpenAI's gpt-5.5 family (and any future model with extended-only retention)
    does NOT support `prompt_cache_retention="in_memory"` — the API returns a
    400. Rather than pass a value we know will fail, we drop the field and log
    a warning. The capability flag (set in `_capabilities.py`) is the single
    source of truth for which models support which retention values.

    Returns None when the value should be dropped. Otherwise returns *retention*
    unchanged. Models with capability flag True (the default) always pass through.

    Note: We deliberately do NOT validate `retention="24h"` against any
    model-supported list. That list (`gpt-5.5`, `gpt-5.4`, ..., `gpt-4.1`) is
    mutable and OpenAI's responsibility to enforce. Our job here is only to
    block the one combination we KNOW will hard-error.
    """
    if retention != "in_memory":
        return retention
    caps = get_capabilities(model_id)
    if caps.supports_in_memory_retention:
        return retention
    logger.warning(
        "[PROVIDER] Dropping prompt_cache_retention='in_memory' for model %r: "
        "model only supports '24h' retention. Omit the field or pass '24h' "
        "to silence this warning.",
        model_id,
    )
    return None


def _drop_unsupported_24h_retention(model_id: str, retention: str | None) -> str | None:
    """Return *retention* unless `"24h"` would be rejected by the model.

    Mirror of `_drop_unsupported_in_memory_retention`. The capability flag
    `supports_24h_retention` is True for all current model families
    (smoke-tested against gpt-4o, gpt-5.x, o-series — all accept "24h"
    despite not being formally enumerated in the cookbook list). Future
    families that prove to reject "24h" can flip the flag False on their
    branch in `_capabilities.py`.

    Returns None when the value should be dropped. Otherwise returns
    *retention* unchanged.
    """
    if retention != "24h":
        return retention
    caps = get_capabilities(model_id)
    if caps.supports_24h_retention:
        return retention
    logger.warning(
        "[PROVIDER] Dropping prompt_cache_retention='24h' for model %r: "
        "model rejects 24h retention. Omit the field or pass 'in_memory' "
        "to silence this warning.",
        model_id,
    )
    return None


def _computer_action_to_dict(action: Any) -> dict[str, Any]:
    """Normalize one `computer_call` action entry to a plain dict.

    Actions arrive either as plain dicts (dict-format response replay) or as
    SDK objects (openai-python pydantic models on the live parsed-response
    path). Handle both without depending on a specific action class, since
    the action union has many variants (click, move, keypress, scroll,
    drag, type, wait, screenshot, ...).
    """
    if isinstance(action, dict):
        return action
    if hasattr(action, "model_dump"):
        return action.model_dump()
    if hasattr(action, "__dict__"):
        return {k: v for k, v in vars(action).items() if not k.startswith("_")}
    return {"value": action}


def _extract_computer_actions(block: Any) -> list[dict[str, Any]]:
    """Extract the actions batch from a `computer_call` response item.

    Live Responses API traffic (captured against gpt-5.6) returns a
    **batched** `actions` array on `computer_call` items, not a singular
    `action` field -- confirmed via `openai-turn0.json`/`openai-turn1.json`
    fixtures, each `{"...", "actions": [...], "call_id": "..."}`. Support a
    singular `action` field defensively too, in case an older/different
    wire shape is ever encountered, rather than silently dropping the call.
    """
    if isinstance(block, dict):
        actions = block.get("actions")
        if actions is None:
            single = block.get("action")
            actions = [single] if single is not None else []
    else:
        actions = getattr(block, "actions", None)
        if actions is None:
            single = getattr(block, "action", None)
            actions = [single] if single is not None else []
    return [_computer_action_to_dict(a) for a in actions]


def _extract_computer_screenshot_data_url(tool_content: Any) -> str:
    """Build the `image_url` data URI OpenAI's `computer_call_output` expects.

    Accepts either:
    - a plain base64 PNG string (already-encoded image data), or
    - a list of content blocks containing an `ImageBlock`-shaped dict
      (`{"type": "image", "source": {"type": "base64", "media_type", "data"}}`),
      mirroring the existing `role == "user"` image conversion (`input_image`,
      above).

    Raises ValueError if neither shape is present. A `computer_call_output`
    with no image is not a valid response to a `computer_call` -- per the
    "fail loud, never silently degrade" requirement, this must surface as an
    error rather than be sent as a malformed/empty request.
    """
    if isinstance(tool_content, str) and tool_content:
        return f"data:image/png;base64,{tool_content}"

    if isinstance(tool_content, list):
        for block in tool_content:
            if isinstance(block, dict) and block.get("type") == "image":
                source = block.get("source", {})
                if source.get("type") == "base64" and source.get("data"):
                    media_type = source.get("media_type", "image/png")
                    return f"data:{media_type};base64,{source['data']}"

    raise ValueError(
        "computer_call tool result did not contain image data; expected a "
        "base64 PNG string or an image content block "
        f"(got: {type(tool_content).__name__})"
    )


async def _maybe_await(value: Any) -> Any:
    """Await *value* if it's awaitable, otherwise return it as-is.

    `client.responses.with_raw_response.create(**params)`'s return type
    varies by openai SDK internals: the installed SDK (2.8.1) returns a
    `LegacyAPIResponse` whose `.parse()` is synchronous (verified live --
    calling it either returns the parsed model directly or raises
    `pydantic.ValidationError` immediately, it is never a coroutine). Other
    SDK versions may return the newer `APIResponse`, whose `.parse()` *is*
    a coroutine. This normalizes both shapes for `_create_response` and
    `_read_raw_json_body` without depending on which one is live.
    """
    if inspect.isawaitable(value):
        return await value
    return value


async def _read_raw_json_body(raw_response: Any) -> Any:
    """Decode the JSON body of a raw SDK response, sync or async API.

    Prefers a `.json()` method if the response object has one (the newer
    `APIResponse` shape). The installed SDK's `LegacyAPIResponse` has no
    `.json()` at all (verified live) -- only a synchronous `.content`
    (bytes) property, so that's the fallback.
    """
    json_method = getattr(raw_response, "json", None)
    if callable(json_method):
        return await _maybe_await(json_method())
    return json.loads(raw_response.content)


def _params_declare_computer_tool(params: dict[str, Any]) -> bool:
    """True if this request's `tools` list declares the native `computer` tool.

    Scopes the raw-JSON fallback in `OpenAIProvider._create_response` to
    exactly the request shape that can trigger it. Every other request
    keeps calling `client.responses.create()` directly and unchanged.
    """
    tools = params.get("tools") or []
    return any(isinstance(t, dict) and t.get("type") == "computer" for t in tools)


class _RawResponseObject:
    """Minimal read-only view over a raw Responses-API JSON body.

    Exists only for the fallback path in `OpenAIProvider._create_response`:
    when the installed openai SDK's typed `Response` model rejects a real
    `computer_call` payload, callers throughout `_complete_chat_request`
    and `_convert_to_chat_response` still need `response.output`,
    `response.usage`, `response.status`, etc. to behave like the SDK's own
    parsed objects, because that code reads the response via
    `getattr`/`hasattr`, not `dict.get`.

    Only *nested dict* attributes (`usage`, `incomplete_details`, ...) are
    wrapped. List-valued attributes (`output`) are left as plain lists of
    plain dicts on purpose: `_convert_to_chat_response`'s per-block parsing
    already branches on `hasattr(block, "type")` vs. dict `block.get(...)`
    for every block type it handles -- this wrapper piggybacks on that
    existing dual-path convention rather than inventing a new one.
    """

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def __getattr__(self, name: str) -> Any:
        try:
            value = self._data[name]
        except KeyError:
            raise AttributeError(name) from None
        return _RawResponseObject(value) if isinstance(value, dict) else value

    def get(self, key: str, default: Any = None) -> Any:
        value = self._data.get(key, default)
        return _RawResponseObject(value) if isinstance(value, dict) else value


def _build_assistant_message_item(
    content_parts: list[dict[str, Any]],
    message_id: str | None = None,
) -> dict[str, Any]:
    """Serialize assistant content as a spec-compliant Responses API message item.

    Emits the canonical ``ResponseOutputMessage`` shape used when assistant history
    is replayed in the Responses API ``input`` array. This is the single form every
    tested backend accepts -- verified on the wire against the OpenAI Responses API,
    llama.cpp's llama-server, and vLLM 0.19:

    - ``type: "message"`` is REQUIRED by llama-server: its input-item dispatch keys
      on the literal ``type`` field, so a role-only item 400s with
      "Cannot determine type of 'item'".
    - ``id`` and ``status`` are REQUIRED by vLLM: input items validate against the
      openai SDK's ``ResponseOutputMessageParam``, which marks both required, so a
      role-only message raises ``pydantic.ValidationError``.
    - ``annotations: []`` on each ``output_text`` part mirrors OpenAI's own output
      items and is accepted by every backend.

    Real OpenAI is permissive and accepts looser forms, but this canonical form is
    the intersection all backends accept. Ref: OpenAI Responses API -- a "message"
    is a discriminated Item type alongside function_call / function_call_output /
    reasoning.

    Args:
        content_parts: Assistant output parts, each ``{"type": "output_text",
            "text": ...}``. ``annotations`` is filled in if absent.
        message_id: Preserved message id when available; a fresh ``msg_<hex>`` is
            synthesized otherwise (replayed-history ids need only be valid strings,
            not server-issued references).

    Returns:
        One Responses API assistant message item.
    """
    normalized: list[dict[str, Any]] = []
    for part in content_parts:
        if isinstance(part, dict):
            normalized.append(
                {
                    "type": part.get("type", "output_text"),
                    "text": part.get("text", ""),
                    "annotations": part.get("annotations", []),
                }
            )
        else:
            normalized.append(
                {"type": "output_text", "text": str(part), "annotations": []}
            )
    return {
        "type": "message",
        "id": message_id or f"msg_{uuid.uuid4().hex}",
        "role": "assistant",
        "status": "completed",
        "content": normalized,
    }


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
        add_cost=None,
    ):
        """Initialize OpenAI provider with Responses API client.

        The SDK client is created lazily on first use, allowing get_info()
        to work without valid credentials.
        """
        self._api_key = api_key
        self._client: AsyncOpenAI | None = client  # Lazy init if None
        self.config = config or {}
        self.coordinator = coordinator

        # One-shot flag set by the compaction hook handlers registered in
        # mount(). When the context manager compacts (or a swapped context
        # module fires its pre/post-compaction event), the local transcript
        # shrinks but the most recent assistant message may still carry a
        # pre-compaction openai:response_id. Chaining from it via
        # previous_response_id makes OpenAI rebuild the full pre-compaction
        # server-side context, so input tokens climb without bound until
        # context_length_exceeded. This flag tells the next request to break
        # the chain so it restarts from the compacted view.
        self._reset_chain_on_next_request = False

        # Configuration with sensible defaults (from _constants.py - single source of truth)
        self.base_url = self.config.get(
            "base_url", None
        )  # Optional custom endpoint (None = OpenAI default)
        self.default_model = self.config.get("default_model", DEFAULT_MODEL)
        # P5: no fixed 4096 default. None = derive per request from the
        # model's capability max_output_tokens (mirrors provider-anthropic,
        # which defaults to its capability table). An explicit config value
        # still wins. DEFAULT_MAX_TOKENS survives only as the documented
        # fallback when capability data is absent (see _resolve_max_tokens).
        self.max_tokens = self.config.get("max_tokens")
        self.temperature = self.config.get(
            "temperature", None
        )  # None = not sent (some models don't support it)
        self.reasoning = self.config.get(
            "reasoning", None
        )  # None = not sent (none|low|medium|high|xhigh)
        # Canonical effort config key: "reasoning_effort" (matches the
        # kernel's portable request.reasoning_effort field). Validated at
        # mount — unknown values raise here instead of 400ing mid-session.
        # Normalized to None when absent/blank/"none" (the provisioned
        # ConfigField default) so absence never injects a reasoning param.
        self.reasoning_effort = _resolve_config_reasoning_effort(
            self.config.get("reasoning_effort"), self.default_model
        )
        if self.reasoning_effort is not None and self.reasoning is not None:
            logger.warning(
                "[PROVIDER] Both 'reasoning_effort' and 'reasoning' are set "
                "in config; 'reasoning_effort'=%r (canonical) wins and "
                "'reasoning'=%r is ignored.",
                self.reasoning_effort,
                self.reasoning,
            )
        # Loudness guard against silently-inert config: warn about
        # effort-family keys this provider does NOT consume.
        for _inert_key in ("effort",):
            if _inert_key in self.config:
                logger.warning(
                    "[PROVIDER] Config key '%s' is not consumed by "
                    "provider-openai and has no effect. Accepted effort keys: "
                    "'reasoning_effort' (canonical), 'reasoning' (legacy).",
                    _inert_key,
                )
        self.reasoning_summary = self.config.get(
            "reasoning_summary", DEFAULT_REASONING_SUMMARY
        )
        # `truncation` defaults to None (omit the field) for cache stability.
        # When the model context fills, OpenAI now returns an explicit error
        # instead of silently rewriting the prefix. Pass
        # `config={"truncation": "auto"}` to opt into the legacy auto-drop
        # behavior.
        self.truncation = self.config.get("truncation", DEFAULT_TRUNCATION)
        self.enable_state = self.config.get("enable_state", False)
        self.raw = self.config.get("raw", False)  # Include raw payload in events
        self.timeout = self.config.get("timeout", DEFAULT_TIMEOUT)
        self.filtered = self.config.get(
            "filtered", True
        )  # Filter to curated model list by default

        # Prompt-caching hint parameters (Responses API top-level fields).
        # All three default to None = "don't send the field, use OpenAI's
        # model default". Per-call kwargs override the config default — same
        # pattern as `truncation`.
        #
        # prompt_cache_key: stable per-conversation (or per-tenant+system-prompt)
        #   identifier. OpenAI shards by hash of first ~256 input tokens; setting
        #   a stable key keeps subsequent requests routed to the same machine,
        #   maximizing cache hit rate. Use this instead of the `user` field for
        #   cache-routing per OpenAI's July 2025 guidance (the `user` field is
        #   retained on the API but is no longer the recommended cache signal).
        # prompt_cache_retention: "in_memory" (5–10 min) or "24h" (extended
        #   GPU-local KV storage). Defaults to "24h" so caching is stable
        #   across all models, including gpt-5.4 and below where OpenAI's
        #   server-side default is the much-shorter "in_memory". Models that
        #   reject "24h" (capability flag False) get the field dropped with a
        #   warning. Models that reject "in_memory" (gpt-5.5+) get the same
        #   treatment via the existing helper. Pass None explicitly to fall
        #   back to OpenAI's per-model default.
        # safety_identifier: abuse-tracking signal — the request-side counterpart
        #   to `prompt_cache_key`. Per-end-user value (not per-deployment), which
        #   is why it is intentionally NOT exposed via ConfigField; set it via
        #   per-call kwargs.
        # Empty strings (e.g. from UI form defaults) are coerced to None so we
        # don't send empty-string fields to OpenAI.
        self.prompt_cache_key: str | None = self.config.get("prompt_cache_key") or None
        # prompt_cache_retention defaults to "24h" so non-gpt-5.5 models get
        # extended GPU-local KV storage instead of OpenAI's per-model
        # in_memory default (5–10 min). Use `dict.get(..., DEFAULT)` so an
        # explicit `None` in config is preserved as "let OpenAI pick the
        # model default" rather than overridden. Empty string is coerced to
        # None to match the established UI-form pattern.
        _retention = self.config.get(
            "prompt_cache_retention", DEFAULT_PROMPT_CACHE_RETENTION
        )
        self.prompt_cache_retention: str | None = _retention if _retention else None
        # prompt_cache_options (GPT-5.6): explicit prompt-cache control that COEXISTS
        # with prompt_cache_retention. Shape {"mode": "implicit"|"explicit", "ttl":
        # "30m"}; default None = do not send. Verified live 2026-07-14.
        self.prompt_cache_options: dict | None = (
            self.config.get("prompt_cache_options") or None
        )
        self.safety_identifier: str | None = (
            self.config.get("safety_identifier") or None
        )

        # Response chaining for reasoning models — the Responses API's
        # `previous_response_id` mechanism. Tri-state:
        #   "auto" (default) — on iff get_capabilities(model).supports_reasoning
        #   True             — force on regardless of model
        #   False            — force off (ZDR / privacy / regulated-industry opt-out)
        #
        # When active, three things happen on each call to a reasoning model:
        #   (a) params["store"] = True
        #   (b) params["previous_response_id"] = <id from last assistant.metadata>
        #       (when a prior response_id is available; first turn has none)
        #   (c) `include=["reasoning.encrypted_content"]` is NOT requested,
        #       and ThinkingBlocks are NOT re-inserted into the input array
        #       (server holds the reasoning state under previous_response_id;
        #       sending encrypted blobs inline busts the cache prefix).
        #
        # Per OpenAI Cookbook 201 §4.5: Responses API + chaining gives 40–80%
        # better cache utilization on reasoning workloads vs. stateless mode.
        #
        # Interaction with `enable_state`:
        #   - `enable_state` remains the broad server-state switch (e.g. for
        #     callers that need response retrieval/inspection regardless of
        #     reasoning).
        #   - `enable_response_chaining` is the cache-driven knob for
        #     reasoning models specifically.
        #   - When chaining resolves to True, `store=True` is forced (chaining
        #     requires it). Otherwise `enable_state` decides `store`.
        raw_chain = self.config.get("enable_response_chaining", "auto")
        # Coerce: accept "auto" | True | False | None | "" → normalize to "auto"|True|False
        if raw_chain in (None, "", "auto"):
            self.enable_response_chaining: str | bool = "auto"
        else:
            self.enable_response_chaining = bool(raw_chain)

        # Deep research / background mode configuration
        self.poll_interval = self.config.get("poll_interval", DEFAULT_POLL_INTERVAL)
        self.background_timeout = self.config.get(
            "background_timeout", DEFAULT_BACKGROUND_TIMEOUT
        )

        # Provider priority for selection (lower = higher priority)
        self.priority = self.config.get("priority", 100)

        # Long context flag — when False (default), GPT-5.4 reports 272K context
        # (the pricing threshold) instead of the full 1,050K window, keeping costs
        # predictable.  Set to True to advertise the full context window.
        self.enable_long_context = self.config.get("enable_long_context", False)

        # Streaming flag — when True (default), uses client.responses.stream() with
        # chunked HTTP transport to prevent timeouts on large context requests.
        # This is NOT progressive token streaming to the user; it collects the complete
        # response before returning, matching what the Anthropic provider does.
        # Set to False to use the blocking create() path (useful for tests / compat).
        self.use_streaming = self.config.get("use_streaming", True)

        # Retry configuration — delegates to shared retry_with_backoff() from amplifier-core.
        self._retry_config = RetryConfig(
            max_retries=int(self.config.get("max_retries", 5)),
            initial_delay=float(self.config.get("min_retry_delay", 1.0)),
            max_delay=float(self.config.get("max_retry_delay", 60.0)),
            jitter=bool(self.config.get("retry_jitter", True)),
        )

        # Track tool call IDs that have been repaired with synthetic results.
        # This prevents infinite loops when the same missing tool results are
        # detected repeatedly across LLM iterations (since synthetic results
        # are injected into request.messages but not persisted to message store).
        self._repaired_tool_ids: set[str] = set()

        # Apply patch native mode detection — set during tool conversion
        self._apply_patch_native = False
        self._native_call_ids: set[str] = set()
        # Maps call_id -> native tool type ("computer", etc.) for call_ids in
        # _native_call_ids. Only populated for native types that need a
        # different result envelope than apply_patch's default
        # (apply_patch_call_output); absence of an entry means "apply_patch",
        # preserving existing behavior/tests untouched.
        self._native_call_types: dict[str, str] = {}
        self._add_cost = add_cost or (lambda cost: None)

        # Process-wide concurrency gate.
        # Limits how many API calls this process has in-flight simultaneously,
        # shared across ALL provider instances (parent + delegated child sessions).
        # This prevents blast patterns (e.g. parallel: true recipes spawning 20+
        # concurrent calls) that trigger Cloudflare bot-detection on api.openai.com.
        # Set to 0 to disable the semaphore entirely.
        self._max_concurrent_requests = int(
            self.config.get("max_concurrent_requests", 5)
        )

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

    @staticmethod
    def _is_cloudflare_challenge(error: openai.APIStatusError) -> bool:
        """Detect Cloudflare bot-management challenge responses.

        Cloudflare interposes HTML challenge pages (HTTP 403) that look nothing
        like real API errors.  Signals:

        1. The body did not parse as a JSON object/array. (When the SDK
           cannot parse the body as JSON it stores the RAW TEXT in
           ``error.body`` -- a str, NOT None; a parsed error is a dict/list.)
        2. The Content-Type is text/html (not application/json).
        3. The raw response text contains Cloudflare markers.

        Any combination of (1 + 2) or (1 + 3) is sufficient.  If the SDK
        successfully parsed a JSON body, this is a real API error regardless
        of other signals.
        """
        # Only a PARSED JSON body (dict/list) means a genuine, structured
        # API error. When the SDK cannot parse the body as JSON it stores the
        # RAW TEXT in ``error.body`` -- a str, NOT None -- so a "body is not
        # None" guard bails on exactly the HTML challenge pages this exists to
        # catch. Fall through for a str (or absent) body; bail only on parsed
        # JSON.
        body = getattr(error, "body", None)
        if isinstance(body, (dict, list)):
            return False

        # Inspect the raw HTTP response for HTML / Cloudflare signals
        response = getattr(error, "response", None)
        if response is None:
            return False

        content_type = getattr(response, "headers", {}).get("content-type", "").lower()
        if "text/html" in content_type:
            return True

        # Fallback: scan response text for Cloudflare markers (case-insensitive)
        text = (getattr(response, "text", "") or "").lower()
        cf_markers = (
            "just a moment",
            "cf-browser-verification",
            "cloudflare",
            "checking if the site connection is secure",
        )
        return any(marker in text for marker in cf_markers)

    def get_info(self) -> ProviderInfo:
        """Get provider metadata."""
        caps = get_capabilities(self.default_model)
        if self.enable_long_context and caps.long_context_pricing_threshold:
            reported_context = caps.context_window  # 1,050,000 for GPT-5.4
        else:
            reported_context = (
                caps.long_context_pricing_threshold or caps.context_window
            )
        return ProviderInfo(
            id="openai",
            display_name="OpenAI",
            credential_env_vars=["OPENAI_API_KEY"],
            capabilities=["streaming", "tools", "reasoning", "batch", "json_mode"],
            defaults={
                "model": self.default_model,
                "max_tokens": 16384,
                "temperature": None,
                "timeout": 600.0,
                "context_window": reported_context,
                "max_output_tokens": caps.max_output_tokens,
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
                    choices=["none", "low", "medium", "high", "xhigh", "max"],
                    default="none",
                    required=False,
                    requires_model=True,  # Shown after model selection
                ),
                ConfigField(
                    id="enable_long_context",
                    display_name="Enable long context",
                    field_type="boolean",
                    prompt="Enable long context (>272K tokens, 2x input / 1.5x output pricing)",
                    required=False,
                    default="false",
                ),
                ConfigField(
                    id="prompt_cache_key",
                    display_name="Prompt cache key",
                    field_type="text",
                    prompt=(
                        "Stable identifier for OpenAI prompt-cache routing "
                        "(e.g. conversation ID or tenant+system-prompt-version)"
                    ),
                    required=False,
                    default="",
                ),
                ConfigField(
                    id="prompt_cache_retention",
                    display_name="Prompt cache retention",
                    field_type="choice",
                    prompt=(
                        "Cache retention window. Leave unset to use the model "
                        "default (recommended)."
                    ),
                    choices=["in_memory", "24h"],
                    required=False,
                    # default=None (not "") because "" is not a member of
                    # `choices`; UI renderers that validate `default in choices`
                    # would reject it. None signals "leave unset" cleanly.
                    default=None,
                ),
                ConfigField(
                    id="enable_response_chaining",
                    display_name="Enable response chaining",
                    field_type="choice",
                    prompt=(
                        "Response chaining for reasoning models via previous_response_id. "
                        '"auto" (default) enables chaining for reasoning-capable models '
                        'only. Set to "false" to disable for ZDR / regulated-industry '
                        "deployments that cannot retain server-side state."
                    ),
                    choices=["auto", "true", "false"],
                    required=False,
                    default="auto",
                ),
                # NOTE: `safety_identifier` is intentionally NOT exposed as a
                # ConfigField. It is a per-end-user signal, not a per-deployment
                # one — surfacing it in the UI invites operators to set a single
                # global value, which defeats its abuse-tracking purpose. The
                # provider still accepts it via per-call kwargs and (for tests
                # and unusual deployments) via the config dict.
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

            caps = get_capabilities(model_id)
            capabilities = list(caps.capability_tags)
            if self.enable_long_context and caps.long_context_pricing_threshold:
                reported_context = caps.context_window
            else:
                reported_context = (
                    caps.long_context_pricing_threshold or caps.context_window
                )
            max_output_tokens = caps.max_output_tokens
            if is_deep_research:
                defaults = {"max_tokens": 32768, "background": True}
            else:
                defaults = {"max_tokens": 16384, "reasoning_effort": "none"}

            models.append(
                ModelInfo(
                    id=model_id,
                    display_name=display_name,
                    context_window=reported_context,
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
            "gpt-5.5": "GPT 5.5",
            "gpt-5.5-pro": "GPT 5.5 Pro",
            "gpt-5.4": "GPT 5.4",
            "gpt-5.4-pro": "GPT 5.4 Pro",
            "gpt-5.3-codex": "GPT-5.3 codex",
            "gpt-5.2": "GPT 5.2",
            "gpt-5.2-pro": "GPT 5.2 Pro",
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
        """Check if the model supports reasoning via capabilities lookup.

        Returns False for empty/unknown model names.
        """
        if not model_name:
            return False
        caps = get_capabilities(model_name)
        return caps.supports_reasoning

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

        # If we extracted any assistant content, add as a spec-compliant message item
        if assistant_content:
            continuation_input.append(
                _build_assistant_message_item(assistant_content)
            )

        return continuation_input

    async def _enforce_chain_output_pairing(
        self,
        delta_input: list[dict[str, Any]],
        chained_msg: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Chain-aware pairing invariant for the response-chaining delta path.

        When a request chains via previous_response_id, the function_call
        items live SERVER-SIDE in the chained response; the delta input
        carries only the outputs. The server requires every call in the
        chained response to have a function_call_output in this input,
        paired BY call_id — an unpaired call is a non-retryable 400
        ("No tool output found for function call ...") that kills the
        session. The generic wire backstop in _convert_messages cannot
        protect this case (no function_call items in the input to check).

        Enforced here, against the LOCAL record of the chained turn (the
        assistant message's tool_calls):
        1. every chained tool call id must have a matching output — an
           error output is synthesized for any orphan;
        2. any fc_-prefixed output id (a Responses-API ITEM id, not a
           call id) is a bug upstream: it can pair with nothing
           server-side, so it is dropped with a loud warning — observed
           live killing a session even for a COMPLETED, successfully
           executed tool call.
        """
        expected_ids: list[str] = []
        for tc in chained_msg.get("tool_calls") or []:
            if isinstance(tc, dict):
                tc_id = tc.get("id") or tc.get("tool_call_id")
                if tc_id:
                    expected_ids.append(str(tc_id))
        content = chained_msg.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_call":
                    b_id = block.get("id")
                    if b_id and str(b_id) not in expected_ids:
                        expected_ids.append(str(b_id))

        provided_ids: set[str] = set()
        anomalous_items: list[dict[str, Any]] = []
        kept: list[dict[str, Any]] = []
        for item in delta_input:
            if isinstance(item, dict) and item.get("type") == "function_call_output":
                call_id = str(item.get("call_id") or "")
                if call_id.startswith("fc_"):
                    anomalous_items.append(item)
                    logger.warning(
                        "[PROVIDER] Chain pairing: dropping function_call_output "
                        "keyed by ITEM id %s (output len %d) — fc_ ids cannot "
                        "pair with any server-side call; this indicates an "
                        "upstream dispatch-keying bug.",
                        call_id,
                        len(str(item.get("output") or "")),
                    )
                    continue
                provided_ids.add(call_id)
            kept.append(item)

        missing_ids = [cid for cid in expected_ids if cid not in provided_ids]
        for cid in missing_ids:
            logger.warning(
                "[PROVIDER] Chain pairing: synthesizing error output for "
                "orphaned chained call %s (expected=%s, provided=%s).",
                cid,
                expected_ids,
                sorted(provided_ids),
            )
            kept.append(
                {
                    "type": "function_call_output",
                    "call_id": cid,
                    "output": (
                        "[error] Tool execution result missing for this call "
                        "(chained-turn pairing repair). The result was lost or "
                        "mis-keyed; re-issue the tool call if still needed."
                    ),
                }
            )

        if (
            (missing_ids or anomalous_items)
            and self.coordinator
            and hasattr(self.coordinator, "hooks")
        ):
            await self.coordinator.hooks.emit(
                "provider:chain_pairing_repaired",
                {
                    "provider": self.name,
                    "expected_call_ids": expected_ids,
                    "provided_call_ids": sorted(provided_ids),
                    "synthesized_for": missing_ids,
                    "dropped_item_id_outputs": [
                        str(i.get("call_id")) for i in anomalous_items
                    ],
                },
            )
        return kept

    def _find_missing_tool_results(
        self, messages: list
    ) -> list[tuple[int, str, str, dict]]:
        """Find tool calls without matching results.

        Scans conversation for assistant tool calls and validates each has
        a corresponding tool result message. Returns missing tuples including
        the index of the assistant message containing each tool_use block.

        Excludes tool call IDs that have already been repaired with synthetic
        results to prevent infinite detection loops.

        Returns:
            List of (msg_idx, call_id, tool_name, tool_arguments) tuples for unpaired calls
        """
        tool_calls: dict[
            str, tuple[int, str, dict]
        ] = {}  # {call_id: (msg_idx, name, args)}
        tool_results: set[str] = set()  # {call_id}

        for idx, msg in enumerate(messages):
            # Check assistant messages for ToolCallBlock in content
            if msg.role == "assistant" and isinstance(msg.content, list):
                for block in msg.content:
                    if hasattr(block, "type") and block.type == "tool_call":
                        tool_calls[block.id] = (idx, block.name, block.input)

            # Check tool messages for tool_call_id
            elif (
                msg.role == "tool" and hasattr(msg, "tool_call_id") and msg.tool_call_id
            ):
                tool_results.add(msg.tool_call_id)

        # Exclude IDs that have already been repaired to prevent infinite loops
        return [
            (msg_idx, call_id, name, args)
            for call_id, (msg_idx, name, args) in tool_calls.items()
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
                f"Tool IDs: {[call_id for _, call_id, _, _ in missing]}"
            )

            # Group missing calls by the assistant message index that contains them.
            # Insert synthetics right after each assistant message (not at the end),
            # so ordering requirements are satisfied even when user messages follow.
            by_msg_idx: dict[int, list[tuple[str, str]]] = defaultdict(list)
            for msg_idx, call_id, tool_name, _ in missing:
                by_msg_idx[msg_idx].append((call_id, tool_name))

            synthetic_assistant_count = 0

            # Process in REVERSE index order so earlier insertions don't shift later indices
            for msg_idx in sorted(by_msg_idx.keys(), reverse=True):
                synthetics = []
                for call_id, tool_name in by_msg_idx[msg_idx]:
                    synthetics.append(self._create_synthetic_result(call_id, tool_name))
                    # Track this ID so we don't detect it as missing again in future iterations
                    self._repaired_tool_ids.add(call_id)

                insert_pos = msg_idx + 1
                for i, synthetic in enumerate(synthetics):
                    request.messages.insert(insert_pos + i, synthetic)

                # FM3: If a real user message follows the inserted synthetics, also insert
                # a synthetic assistant response to close the interrupted turn.
                next_pos = insert_pos + len(synthetics)
                if next_pos < len(request.messages):
                    next_msg = request.messages[next_pos]
                    is_real_user = (
                        next_msg.role == "user"
                        and not getattr(next_msg, "tool_call_id", None)
                        and not (
                            isinstance(next_msg.content, str)
                            and next_msg.content.strip().startswith("<system-reminder>")
                        )
                    )
                    if is_real_user:
                        from amplifier_core.message_models import Message

                        synthetic_assistant = Message(
                            role="assistant",
                            content=(
                                "The previous tool calls were interrupted due to a session error. "
                                "This was automatically repaired."
                            ),
                        )
                        request.messages.insert(next_pos, synthetic_assistant)
                        synthetic_assistant_count += 1

            # Emit observability event
            if self.coordinator and hasattr(self.coordinator, "hooks"):
                event_data: dict[str, Any] = {
                    "provider": self.name,
                    "repair_count": len(missing),
                    "repairs": [
                        {"tool_call_id": call_id, "tool_name": tool_name}
                        for _, call_id, tool_name, _ in missing
                    ],
                }
                if synthetic_assistant_count > 0:
                    event_data["synthetic_assistant_count"] = synthetic_assistant_count
                await self.coordinator.hooks.emit(
                    "provider:tool_sequence_repaired",
                    event_data,
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

    def _should_chain_responses(self, model_id: str, kwargs: dict[str, Any]) -> bool:
        """Resolve whether response chaining is active for *this* call.

        Precedence (highest first):
          1. kwargs["enable_response_chaining"]  (per-call override)
          2. self.enable_response_chaining       (config)
          3. "auto" → caps.supports_reasoning
        """
        override = kwargs.get("enable_response_chaining", self.enable_response_chaining)
        if override is True:
            return True
        if override is False:
            return False
        # "auto"
        return get_capabilities(model_id).supports_reasoning

    async def _create_response(self, params: dict[str, Any]) -> Any:
        """Call `client.responses.create(**params)`.

        For computer-use requests, parses from the raw JSON body instead of
        the SDK's typed model.

        Why: OpenAI's GA `computer_call` response omits `pending_safety_checks`
        entirely. Every openai-python release checked -- 2.8.1 (installed) and
        2.52.0 (latest) -- declares it a required field with no default on
        `ResponseComputerToolCall`, so `client.responses.create()`'s automatic
        parsing raises `pydantic.ValidationError` on a real live response,
        independent of anything this provider does (see PR #58 for the version
        evidence). `provider-anthropic` already uses `with_raw_response` on its
        non-streaming path (for headers); this follows the same precedent to
        read the body before the SDK's typed model gets a chance to reject it.

        Scoped to requests that declare the `computer` tool via
        `_params_declare_computer_tool`: every other request keeps calling
        `client.responses.create()` directly, unchanged, so this fallback
        cannot mask an unrelated parsing regression.

        `with_raw_response.create()` returns a `LegacyAPIResponse` on the
        installed SDK (2.8.1) whose `.parse()` is synchronous and has no
        `.json()` method at all -- verified live, not assumed from the type
        stubs (which describe the newer async `APIResponse`, not what this
        SDK version actually returns for this resource). Both `.parse()` and
        the raw-body read are handled for either shape via `_maybe_await` /
        `_read_raw_json_body` so this keeps working if the SDK's
        with_raw_response implementation changes.

        Fails loud: if the raw body itself doesn't parse as the expected
        shape, a `RuntimeError` is raised with the original `ValidationError`
        preserved as its cause -- never a silently empty/partial response.
        """
        if not _params_declare_computer_tool(params):
            return await self.client.responses.create(**params)

        raw_response = await self.client.responses.with_raw_response.create(**params)
        try:
            return await _maybe_await(raw_response.parse())
        except ValidationError as e:
            logger.warning(
                "[PROVIDER] %s: typed Response model rejected a computer-use "
                "response (%s). Falling back to the raw JSON body -- see "
                "PR #58 for the pending_safety_checks SDK defect this works "
                "around.",
                self.api_label,
                e,
            )
            try:
                body = await _read_raw_json_body(raw_response)
            except (json.JSONDecodeError, AttributeError) as json_error:
                raise RuntimeError(
                    "computer-use response fallback failed: the raw JSON "
                    "body could not be decoded either (typed-parse error: "
                    f"{e}; json-decode error: {json_error})"
                ) from json_error
            if not isinstance(body, dict) or "output" not in body:
                raise RuntimeError(
                    "computer-use response fallback failed: raw JSON body "
                    f"is missing the expected 'output' field (got: "
                    f"{type(body).__name__})"
                ) from e
            return _RawResponseObject(body)

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

        # Decide chaining BEFORE message conversion so we can suppress
        # encrypted-content re-insertion when chain_active is True.
        model_name = kwargs.get("model", self.default_model)
        chain_active = self._should_chain_responses(model_name, kwargs)

        # Convert to OpenAI Responses API message format
        input_messages = self._convert_messages(
            all_messages_for_conversion,
            skip_reasoning_reinsertion=chain_active,
        )
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

        # If a compaction fired since the last request, break the chain now.
        # The stored previous_response_id points at the pre-compaction
        # server-side context; chaining from it would rebuild the un-compacted
        # transcript server-side and grow input tokens without bound. Dropping
        # it forces a fresh prefix built from the compacted local transcript
        # (which is still sent in full via `input`). One-shot: consume and
        # clear so subsequent turns chain normally from the new post-compaction
        # response.
        if self._reset_chain_on_next_request:
            if previous_response_id is not None:
                logger.info(
                    "[PROVIDER] Breaking response chain after compaction "
                    "(dropping previous_response_id=%s).",
                    previous_response_id,
                )
                previous_response_id = None
            self._reset_chain_on_next_request = False

        # Prepare request parameters per Responses API spec
        params = {
            "model": model_name,
            "input": input_messages,  # Array of message objects, not text string
        }

        # Check for background mode (used for deep research and long-running requests)
        # Background mode requires store=True per OpenAI API requirements
        background_mode = kwargs.get("background", False)

        # Auto-enable background mode for deep research models
        if model_name in DEEP_RESEARCH_MODELS or model_name.startswith(
            ("o3-deep-research", "o4-mini-deep-research")
        ):
            # Deep research models should use background mode by default
            background_mode = kwargs.get("background", True)
            logger.info(
                f"[PROVIDER] Deep research model detected: {model_name}, background={background_mode}"
            )

        # ---- Response chaining resolution (PR-B) -----------------------------
        # Three knobs interact:
        #   - background_mode            → forces store=True
        #   - enable_response_chaining   → for reasoning models, forces store=True
        #                                  AND attaches previous_response_id
        #   - enable_state (legacy)      → broad server-state opt-in
        # Per-call kwargs override config for both.
        # (chain_active was already resolved above before _convert_messages)

        store_enabled = kwargs.get("store", self.enable_state)
        if background_mode:
            store_enabled = True
            logger.info("[PROVIDER] Background mode enabled, forcing store=True")
        if chain_active:
            store_enabled = True  # chaining requires server-side state
        params["store"] = store_enabled

        # Attach previous_response_id when:
        #   - we found one in the last assistant's metadata, AND
        #   - either chaining is active for this model, OR
        #     legacy enable_state is on (preserves existing behavior).
        # Track on params for downstream invalidation-retry logic.
        if previous_response_id and (chain_active or store_enabled):
            params["previous_response_id"] = previous_response_id
            logger.debug(
                "[PROVIDER] Using previous_response_id=%s (chain_active=%s, store=%s)",
                previous_response_id,
                chain_active,
                store_enabled,
            )
            # previous_response_id already carries the full prior
            # request+response as server-side state. Re-sending the entire
            # local conversation in `input` double-counts every prior token
            # server-side. Send ONLY the delta: developer context for this turn
            # plus the conversation messages added AFTER the chained assistant
            # turn.
            chain_idx = None
            for i in range(len(conversation) - 1, -1, -1):
                cm = conversation[i]
                cm_dict = cm.model_dump() if hasattr(cm, "model_dump") else cm
                if (
                    getattr(cm, "role", None) == "assistant"
                    and isinstance(cm_dict, dict)
                    and cm_dict.get("metadata")
                    and cm_dict["metadata"].get(METADATA_RESPONSE_ID)
                    == previous_response_id
                ):
                    chain_idx = i
                    break
            if chain_idx is not None:
                delta_for_conversion = [dev.model_dump() for dev in developer_msgs] + [
                    cm.model_dump() for cm in conversation[chain_idx + 1 :]
                ]
                params["input"] = self._convert_messages(
                    delta_for_conversion,
                    skip_reasoning_reinsertion=chain_active,
                )
                # Pathway #2 (chain-aware pairing invariant): the server-side
                # response referenced by previous_response_id carries the
                # function_call items; the delta input carries only outputs.
                # The generic wire backstop in _convert_messages cannot see
                # server-side calls, so enforce here: every tool call issued
                # by the assistant turn being chained from MUST have a
                # function_call_output paired BY call_id in this delta, or
                # the API 400s ("No tool output found for function call ...")
                # and kills the session. Observed live: a COMPLETED,
                # successfully-executed tool call still 400'd because its
                # output was keyed by the fc_ item id instead of call_id.
                chained_msg = conversation[chain_idx]
                chained_dict = (
                    chained_msg.model_dump()
                    if hasattr(chained_msg, "model_dump")
                    else chained_msg
                )
                params["input"] = await self._enforce_chain_output_pairing(
                    params["input"],
                    chained_dict if isinstance(chained_dict, dict) else {},
                )
                logger.info(
                    "[PROVIDER] Response chaining active: trimmed input to delta "
                    "(%d local messages -> %d API messages after previous_response_id)",
                    len(delta_for_conversion),
                    len(params["input"]),
                )
        elif previous_response_id:
            logger.debug(
                "[PROVIDER] Skipping previous_response_id (chain_active=False, store=False). "
                "Relying on explicit reasoning re-insertion from metadata/content."
            )

        if instructions:
            params["instructions"] = instructions

        if request.max_output_tokens:
            params["max_output_tokens"] = request.max_output_tokens
        else:
            # P5: default the output budget to the MODEL's capability limit
            # instead of a fixed 4096 (mirrors provider-anthropic). A 4096
            # cap silently truncated large tool calls (a >4K-token write_file
            # can never complete) while the Anthropic provider ran at its
            # model cap — both a live session-killer (P4 trigger) and a
            # cross-provider parity confound. Explicit request/kwargs/config
            # values still win; DEFAULT_MAX_TOKENS is only the fallback when
            # capability data is absent.
            max_tokens = kwargs.get("max_tokens", self.max_tokens)
            if max_tokens is None:
                caps = get_capabilities(params["model"])
                max_tokens = caps.max_output_tokens or DEFAULT_MAX_TOKENS
            params["max_output_tokens"] = max_tokens

        if request.temperature is not None:
            params["temperature"] = request.temperature
        elif temperature := kwargs.get("temperature", self.temperature):
            params["temperature"] = temperature

        # Phase 2: Reasoning parameter precedence chain
        # kwargs["reasoning"] > kwargs["reasoning_effort"] > request.reasoning_effort
        #   > config "reasoning_effort" (canonical) > config "reasoning" (legacy) > None
        #
        # An explicit kwargs["reasoning"]/request.reasoning dict is a deliberate
        # provider-specific override and is forwarded ungated below -- the caller
        # owns the consequences. Every other path here builds `reasoning` from a
        # portable effort field (kwargs["reasoning_effort"], request.reasoning_effort,
        # or config "reasoning_effort") and is capability-gated: a model that can't
        # reason gets a loud no-op instead of a mid-session API 400.
        reasoning_param = kwargs.get("reasoning", getattr(request, "reasoning", None))
        if reasoning_param is None:
            effort_hint = kwargs.get("reasoning_effort") or request.reasoning_effort
            if effort_hint:
                if get_capabilities(model_name).supports_reasoning:
                    reasoning_param = {
                        "effort": effort_hint,
                        "summary": self.reasoning_summary,
                    }
                else:
                    logger.warning(
                        "[PROVIDER] Ignoring 'reasoning_effort'=%r: "
                        "model %s does not support reasoning.",
                        effort_hint,
                        model_name,
                    )
        if reasoning_param is None and self.reasoning_effort is not None:
            # Canonical config key (validated/normalized at mount; "none" and
            # absence resolve to None so this path never fires for them).
            if get_capabilities(model_name).supports_reasoning:
                reasoning_param = {
                    "effort": self.reasoning_effort,
                    "summary": self.reasoning_summary,
                }
            else:
                logger.warning(
                    "[PROVIDER] Ignoring config 'reasoning_effort'=%r: "
                    "model %s does not support reasoning.",
                    self.reasoning_effort,
                    model_name,
                )
        if reasoning_param is None:
            reasoning_param = self.reasoning
        _validate_gpt_5_5_pro_effort(model_name, reasoning_param)
        _validate_reasoning_mode(reasoning_param)
        if reasoning_param:
            # Handle both dict format ({"effort": "low", "summary": "auto"}) and string format ("low")
            if isinstance(reasoning_param, dict):
                # Dict format: use as-is, but apply defaults for missing keys
                params["reasoning"] = {
                    "effort": reasoning_param.get("effort", "medium"),
                    "summary": reasoning_param.get("summary", self.reasoning_summary),
                }
                # reasoning.mode: "pro" (GPT-5.6) enables extended internal reasoning.
                # Only forwarded when the caller sets it, so pre-5.6 models are
                # unaffected; verified live 2026-07-14 (mode in {standard, pro}).
                _reasoning_mode = reasoning_param.get("mode")
                if _reasoning_mode is not None:
                    params["reasoning"]["mode"] = _reasoning_mode
            else:
                # String format: use as effort level with default summary
                params["reasoning"] = {
                    "effort": reasoning_param,
                    "summary": self.reasoning_summary,  # Verbosity: auto|concise|detailed
                }
            logger.info(f"[PROVIDER] Setting reasoning: {params['reasoning']}")

        # Request encrypted_content when model supports reasoning (regardless of effort level).
        # Reasoning-capable models CAN produce reasoning tokens even with effort=none.
        # Without include=[reasoning.encrypted_content], reasoning token content is lost
        # when store=false (Amplifier's default), causing orphaned reasoning references.
        # Exception: explicit effort="none" suppresses include (caller opted out of reasoning).
        #
        # When chaining is active, server holds reasoning state under
        # previous_response_id. Re-inserting encrypted_content inline would
        # (a) be redundant and (b) actively hurt the cache prefix because the
        # ciphertext changes per call. So skip encrypted-content include when
        # chaining is on; only request it on stateless/non-reasoning paths.
        if not store_enabled and not chain_active:
            caps = get_capabilities(model_name)
            active_effort: str | None = None
            if "reasoning" in params:
                r = params["reasoning"]
                active_effort = r.get("effort") if isinstance(r, dict) else r
            # Explicit effort (including "none") overrides the capability-based default.
            # If the caller explicitly opts out of reasoning, respect that choice.
            if active_effort is not None:
                model_will_reason = active_effort != "none"
            else:
                model_will_reason = caps.supports_reasoning
            if model_will_reason:
                params["include"] = kwargs.get(
                    "include", ["reasoning.encrypted_content"]
                )
                logger.debug(
                    "[PROVIDER] Requesting encrypted_content (stateless path, model will reason: %s, effort=%s)",
                    model_name,
                    active_effort or caps.default_reasoning_effort,
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
            params["tools"] = self._convert_tools_from_request(tools_list, model_name)
            # Add tool-related parameters per Responses API spec
            params["tool_choice"] = kwargs.get("tool_choice", "auto")
            params["parallel_tool_calls"] = kwargs.get("parallel_tool_calls", True)
            # max_tool_calls limits how many tool calls the model can make
            # Important for deep research to prevent excessive searching that consumes token budget
            if max_tool_calls := kwargs.get("max_tool_calls"):
                params["max_tool_calls"] = max_tool_calls

        # Truncation parameter — default None (omit) for cache-prefix
        # stability; per-call kwarg overrides the config default.
        truncation = kwargs.get("truncation", self.truncation)
        if truncation:
            params["truncation"] = truncation

        # Prompt-caching hint parameters (Responses API top-level fields).
        # Per-call kwargs override the config default; None / "" means "do not
        # send". The trailing `or None` mirrors the empty-string coercion in
        # __init__() so that a caller passing `prompt_cache_key=""` (e.g. from
        # a UI form) is treated the same as omitting the field.
        prompt_cache_key = kwargs.get("prompt_cache_key", self.prompt_cache_key) or None
        if prompt_cache_key is not None:
            params["prompt_cache_key"] = prompt_cache_key

        prompt_cache_retention = (
            kwargs.get("prompt_cache_retention", self.prompt_cache_retention) or None
        )
        # Drop retention values the model is known to reject. Each helper is
        # a no-op unless its target value is set AND the capability flag is
        # False. Today only `supports_in_memory_retention=False` (gpt-5.5)
        # actually fires; `supports_24h_retention=False` is reserved for
        # future families that prove to reject "24h".
        prompt_cache_retention = _drop_unsupported_in_memory_retention(
            model_name, prompt_cache_retention
        )
        prompt_cache_retention = _drop_unsupported_24h_retention(
            model_name, prompt_cache_retention
        )
        if prompt_cache_retention is not None:
            params["prompt_cache_retention"] = prompt_cache_retention

        # prompt_cache_options (GPT-5.6): explicit prompt-cache control that COEXISTS
        # with prompt_cache_retention (verified live 2026-07-14 -- both are echoed
        # together; it is NOT a replacement). Forwarded verbatim after a mode-enum
        # pre-flight check; the API validates ttl (currently only "30m").
        prompt_cache_options = (
            kwargs.get("prompt_cache_options", self.prompt_cache_options) or None
        )
        if prompt_cache_options is not None:
            _validate_prompt_cache_options(prompt_cache_options)
            params["prompt_cache_options"] = prompt_cache_options

        safety_identifier = (
            kwargs.get("safety_identifier", self.safety_identifier) or None
        )
        if safety_identifier is not None:
            params["safety_identifier"] = safety_identifier

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
        # Only applies to models with a non-None default_reasoning_effort (o-series, gpt-5.2
        # and below). GPT-5.4+ has default_reasoning_effort=None — it doesn't reason by
        # default, so no reasoning param should be sent unless explicitly requested.
        if self._model_may_reason(model_name) and "reasoning" not in params:
            caps_for_auto = get_capabilities(model_name)
            if caps_for_auto.default_reasoning_effort is not None:
                params["reasoning"] = {"summary": "auto"}

        # Emit llm:request event
        if self.coordinator and hasattr(self.coordinator, "hooks"):
            request_payload: dict[str, Any] = {
                "provider": self.name,
                "model": params["model"],
                "message_count": len(message_list),
                "has_instructions": bool(instructions),
                "reasoning_enabled": params.get("reasoning") is not None,
                "thinking_enabled": thinking_enabled,
                "thinking_budget": thinking_budget,
                "background_mode": background_mode,
            }
            if self.raw:
                request_payload["raw"] = redact_secrets(params)
            await self.coordinator.hooks.emit("llm:request", request_payload)

        start_time = time.time()

        # Use appropriate timeout for background mode (deep research can take minutes)
        effective_timeout = self.background_timeout if background_mode else self.timeout
        poll_interval = kwargs.get("poll_interval", self.poll_interval)

        # Call provider API with shared retry_with_backoff from amplifier-core.
        # Error translation happens inside _do_complete() so that retry_with_backoff
        # sees LLMError (and checks retryable) rather than raw SDK exceptions.

        # Mutable container for rate-limit headers captured inside _do_complete.
        # Using a list-of-one so the nonlocal assignment works across retries.
        captured_rate_limit_info: dict[str, Any] = {}

        # Per-request streaming override (does NOT mutate self.use_streaming).
        # Callers like session-namer pass metadata={"stream": False} to force
        # the blocking create() path and suppress llm:stream_* events.
        _meta = getattr(request, "metadata", None)
        _use_streaming = self.use_streaming
        if isinstance(_meta, dict) and _meta.get("stream") is False:
            _use_streaming = False

        # Background mode (deep-research polling) does not produce token events;
        # disable the stream-event loop for that path.
        supports_streaming = not background_mode

        async def _do_complete():
            """Single API call attempt with SDK → kernel error translation."""
            nonlocal captured_rate_limit_info

            async def _handle_context_overflow(e: Exception, error_msg: str):
                """Break an active response chain, then raise ContextLengthError.

                Shared by the 400 path and the streaming APIError path, which
                surface the same underlying condition.
                """
                # An overflow while a response chain is active almost always
                # means previous_response_id is holding a large pre-compaction
                # server-side context. Break the chain once and retry with the
                # full (compacted) local transcript so the request is bounded
                # by the local view. This is the self-heal that also covers the
                # resume path, where a fresh process re-lifts a stale on-disk
                # openai:response_id before any compaction event has fired.
                if "previous_response_id" in params:
                    overflow_id = params.pop("previous_response_id")
                    # Chain is gone -> the server holds no prior context, so
                    # restore the full converted history (mirrors the 404
                    # invalidation path). Retrying with only the delta would
                    # silently drop the entire prior conversation.
                    params["input"] = input_messages
                    logger.warning(
                        "[PROVIDER] context_length_exceeded with active "
                        "response chain (previous_response_id=%s). Breaking "
                        "chain and retrying with full compacted input.",
                        overflow_id,
                    )
                    if self.coordinator and hasattr(self.coordinator, "hooks"):
                        await self.coordinator.hooks.emit(
                            RESPONSE_CHAIN_INVALIDATED,
                            {
                                "provider": self.name,
                                "model": params.get("model"),
                                "invalidated_id": overflow_id,
                                "error_code": "context_length_exceeded",
                            },
                        )
                    # Retry once. previous_response_id is gone, so a second
                    # overflow cannot re-enter this branch — it falls through
                    # to the ContextLengthError below, which is non-retryable
                    # and so fails fast instead of burning max_retries on a
                    # request that cannot succeed. Raising the typed error does
                    # not trigger recovery: compaction is driven by the context
                    # manager's own token threshold at request-build time, not
                    # by provider errors, and nothing catches
                    # ContextLengthError to retry. It is consumed for
                    # presentation -- the CLI renders a "Context Length
                    # Exceeded" panel with an actionable tip rather than a
                    # generic error.
                    return await _do_complete()
                raise kernel_errors.ContextLengthError(
                    error_msg,
                    provider=self.name,
                    status_code=400,
                ) from e

            try:
                if _use_streaming:
                    # Streaming path — chunked HTTP transport prevents timeouts on
                    # large context requests. The event loop emits contract events
                    # (llm:stream_*); get_final_response() collects the complete
                    # response afterwards so callers see no difference in return value.
                    request_id = str(uuid.uuid4())
                    seq: dict[int, int] = {}  # block_index → next seq number
                    block_types: dict[int, str] = {}  # block_index → contract type
                    partial_emitted = False
                    # Terminal response captured off the stream events. The SDK's
                    # get_final_response() only accepts a `response.completed` event,
                    # so a legitimate non-completed terminal (`response.incomplete`
                    # from max_output_tokens / content filtering) makes it raise
                    # "Didn't receive a `response.completed` event". We capture the
                    # response here so we can recover it. See amplifier-support#339.
                    final_response = None
                    hooks_available = bool(
                        self.coordinator and hasattr(self.coordinator, "hooks")
                    )
                    # Emit events only when coordinator is present AND this is not
                    # a background-mode (deep-research) polling call.
                    emit_stream_events = hooks_available and supports_streaming

                    try:
                        async with asyncio.timeout(effective_timeout):
                            async with self.client.responses.stream(**params) as stream:
                                if emit_stream_events:
                                    async for event in stream:
                                        et = event.type

                                        # Capture the terminal response as we stream so a
                                        # non-`completed` terminal (`response.incomplete`)
                                        # can be recovered below. This runs on the
                                        # event-emitting path (the standard streaming
                                        # case); `response.failed` is deliberately not
                                        # captured so genuine failures still raise.
                                        if et in (
                                            "response.completed",
                                            "response.incomplete",
                                        ):
                                            final_response = getattr(
                                                event, "response", None
                                            )

                                        if et == "response.output_item.added":
                                            idx = event.output_index
                                            item_type = getattr(
                                                event.item, "type", None
                                            )
                                            block_type = {
                                                "message": "text",
                                                "reasoning": "thinking",
                                                "function_call": "tool_use",
                                            }.get(item_type, "text")
                                            block_types[idx] = block_type
                                            seq[idx] = 0
                                            payload: dict[str, Any] = {
                                                "request_id": request_id,
                                                "block_index": idx,
                                                "block_type": block_type,
                                            }
                                            if block_type == "tool_use":
                                                name = getattr(event.item, "name", None)
                                                if name:
                                                    payload["name"] = name
                                            await self.coordinator.hooks.emit(
                                                "llm:stream_block_start", payload
                                            )

                                        elif et == "response.output_text.delta":
                                            text = event.delta
                                            if text:
                                                idx = event.output_index
                                                await self.coordinator.hooks.emit(
                                                    "llm:stream_block_delta",
                                                    {
                                                        "request_id": request_id,
                                                        "block_index": idx,
                                                        "block_type": block_types.get(
                                                            idx, "text"
                                                        ),
                                                        "sequence": seq.get(idx, 0),
                                                        "text": text,
                                                    },
                                                )
                                                seq[idx] = seq.get(idx, 0) + 1
                                                partial_emitted = True

                                        elif et in (
                                            "response.reasoning_summary_text.delta",
                                            "response.reasoning_text.delta",
                                        ):
                                            text = event.delta
                                            if text:
                                                idx = event.output_index
                                                await self.coordinator.hooks.emit(
                                                    "llm:stream_block_delta",
                                                    {
                                                        "request_id": request_id,
                                                        "block_index": idx,
                                                        "block_type": block_types.get(
                                                            idx, "thinking"
                                                        ),
                                                        "sequence": seq.get(idx, 0),
                                                        "text": text,
                                                    },
                                                )
                                                seq[idx] = seq.get(idx, 0) + 1
                                                partial_emitted = True

                                        elif et == "response.output_item.done":
                                            idx = event.output_index
                                            if idx in block_types:
                                                await self.coordinator.hooks.emit(
                                                    "llm:stream_block_end",
                                                    {
                                                        "request_id": request_id,
                                                        "block_index": idx,
                                                        "block_type": block_types[idx],
                                                    },
                                                )

                                try:
                                    response = await stream.get_final_response()
                                except RuntimeError as e:
                                    # The SDK raises this when the stream ends without a
                                    # `response.completed` event — e.g. a `response.incomplete`
                                    # terminal from hitting max_output_tokens or content
                                    # filtering. Recover the response we captured off the
                                    # terminal event instead of failing the whole request.
                                    # See amplifier-support#339.
                                    if (
                                        "response.completed" not in str(e)
                                        or final_response is None
                                    ):
                                        raise
                                    response = final_response
                                    logger.warning(
                                        "[PROVIDER] %s recovered a non-`completed` "
                                        "streaming response (status=%s, model=%s): "
                                        "returning the partial response instead of "
                                        "failing. This usually means max_output_tokens "
                                        "was hit or content filtering stopped "
                                        "generation. See amplifier-support#339.",
                                        self.api_label,
                                        getattr(final_response, "status", "unknown"),
                                        params.get("model", "unknown"),
                                    )
                                # Extract rate limit headers from the underlying HTTP response.
                                # The OpenAI SDK stores it as stream._response (httpx.Response).
                                raw_http = getattr(stream, "_response", None)
                                headers = getattr(raw_http, "headers", None)
                                captured_rate_limit_info = (
                                    self._extract_rate_limit_headers(headers)
                                )
                                return response
                    except Exception as e:
                        # If a partial stream was already emitted, signal abort to
                        # consumers before re-raising for normal error translation.
                        if partial_emitted and hooks_available:
                            await self.coordinator.hooks.emit(
                                "llm:stream_aborted",
                                {
                                    "request_id": request_id,
                                    "error": {
                                        "type": type(e).__name__,
                                        "msg": str(e),
                                    },
                                },
                            )
                        raise
                else:
                    # Non-streaming path — preserved for tests and backward compat.
                    return await asyncio.wait_for(
                        self._create_response(params),
                        timeout=effective_timeout,
                    )
            except openai.RateLimitError as e:
                retry_after = None
                if hasattr(e, "response") and e.response is not None:
                    # Standard header (seconds)
                    ra_header = e.response.headers.get("retry-after")
                    if ra_header:
                        try:
                            retry_after = float(ra_header)
                        except (ValueError, TypeError):
                            pass
                    # Azure-specific fallback (milliseconds, divide by 1000)
                    # Azure OpenAI returns x-ms-retry-after-ms instead of
                    # (or in addition to) the standard retry-after header.
                    if retry_after is None:
                        ms_header = e.response.headers.get("x-ms-retry-after-ms")
                        if ms_header:
                            try:
                                retry_after = float(ms_header) / 1000.0
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
                body = getattr(e, "body", None)
                error_msg = json.dumps(body) if body is not None else str(e)
                raise kernel_errors.RateLimitError(
                    error_msg,
                    provider=self.name,
                    status_code=429,
                    retryable=retryable,
                    retry_after=retry_after,
                ) from e
            except openai.AuthenticationError as e:
                body = getattr(e, "body", None)
                error_msg = json.dumps(body) if body is not None else str(e)
                raise kernel_errors.AuthenticationError(
                    error_msg,
                    provider=self.name,
                    status_code=getattr(e, "status_code", 401),
                ) from e
            except openai.BadRequestError as e:
                raw_msg = str(e).lower()
                body = getattr(e, "body", None)
                error_msg = json.dumps(body) if body is not None else str(e)
                err_code, _ = _extract_error_fields(body)
                if _is_context_overflow(err_code, raw_msg):
                    return await _handle_context_overflow(e, error_msg)
                elif (
                    "content filter" in raw_msg
                    or "safety" in raw_msg
                    or "blocked" in raw_msg
                ):
                    raise kernel_errors.ContentFilterError(
                        error_msg,
                        provider=self.name,
                        status_code=400,
                    ) from e
                else:
                    raise kernel_errors.InvalidRequestError(
                        error_msg,
                        provider=self.name,
                        status_code=400,
                    ) from e
            except openai.APIStatusError as e:
                status = getattr(e, "status_code", 500)
                body = getattr(e, "body", None)
                error_msg = json.dumps(body) if body is not None else str(e)
                if status == 403:
                    if self._is_cloudflare_challenge(e):
                        logger.warning(
                            "[PROVIDER] Cloudflare challenge detected (HTTP 403 "
                            "with HTML body). Treating as transient — will retry."
                        )
                        raise kernel_errors.ProviderUnavailableError(
                            "Cloudflare bot challenge (transient 403 with HTML body). "
                            "This typically resolves on retry.",
                            provider=self.name,
                            status_code=403,
                            retryable=True,
                        ) from e
                    raise kernel_errors.AccessDeniedError(
                        error_msg,
                        provider=self.name,
                        status_code=403,
                    ) from e
                if status == 404:
                    # Specific case: previous_response_id is unknown/expired.
                    # Retry once without the field (graceful chain rebuild).
                    err_code = None
                    if isinstance(body, dict):
                        err_code = (body.get("error") or {}).get("code")
                    raw_msg_404 = str(e).lower()
                    is_chain_invalidation = (
                        err_code in RESPONSE_NOT_FOUND_ERROR_CODES
                        or "previous_response_id" in raw_msg_404
                    )
                    if is_chain_invalidation and "previous_response_id" in params:
                        invalidated_id = params.pop("previous_response_id")
                        # When chaining was active we trimmed params["input"]
                        # down to the post-chain delta. With the chain now
                        # invalidated the server holds no prior context, so
                        # restore the full converted history. OpenAI's
                        # documented recovery is previous_response_id=null +
                        # full input; retrying with only the delta would
                        # silently drop the entire prior conversation.
                        params["input"] = input_messages
                        logger.warning(
                            "[PROVIDER] previous_response_id=%s invalidated by server "
                            "(code=%s). Retrying without chain.",
                            invalidated_id,
                            err_code,
                        )
                        if self.coordinator and hasattr(self.coordinator, "hooks"):
                            await self.coordinator.hooks.emit(
                                RESPONSE_CHAIN_INVALIDATED,
                                {
                                    "provider": self.name,
                                    "model": params.get("model"),
                                    "invalidated_id": invalidated_id,
                                    "error_code": err_code,
                                },
                            )
                        # Recurse once. Do NOT loop indefinitely: by removing
                        # the field, the next call cannot hit this branch again.
                        # NOTE: when chain_active was True we also suppressed
                        # encrypted-content re-insertion in _convert_messages.
                        # That decision is locked into input_messages already;
                        # the retry uses the same input. The server will treat
                        # this as a fresh prefix — equivalent to today's
                        # stateless reasoning behavior. Reasoning state for
                        # *this* turn is lost; next turn re-chains via the
                        # response_id we get from this retry.
                        return await _do_complete()
                    # Fall through to existing 404 handling
                    raise kernel_errors.NotFoundError(
                        error_msg,
                        provider=self.name,
                        status_code=404,
                    ) from e
                if status >= 500:
                    raise kernel_errors.ProviderUnavailableError(
                        error_msg,
                        provider=self.name,
                        status_code=status,
                        retryable=True,
                    ) from e
                raise kernel_errors.LLMError(
                    error_msg,
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
            except openai.APIError as e:
                # Streaming failures never reach the typed branches above. OpenAI
                # returns HTTP 200, opens the SSE stream, then delivers the failure as
                # an SSE "error" event; the SDK re-raises it as a bare openai.APIError,
                # which is the PARENT of APIStatusError and carries no status_code.
                # Before this branch existed such errors fell through to the generic
                # `except Exception` handler, which hardcodes retryable=True -- so a
                # deterministic 400 (context overflow, invalid params) was retried
                # max_retries times before surfacing.
                if isinstance(e, (openai.APIConnectionError, openai.APITimeoutError)):
                    # Transport-level failures are genuinely transient.
                    raise kernel_errors.LLMError(
                        str(e) or f"{type(e).__name__}: (no message)",
                        provider=self.name,
                        retryable=True,
                    ) from e
                body = getattr(e, "body", None)
                error_msg = (
                    json.dumps(body)
                    if body is not None
                    else (str(e) or f"{type(e).__name__}: (no message)")
                )
                err_code, err_type = _extract_error_fields(body)
                raw_msg = str(e).lower()
                if _is_context_overflow(err_code, raw_msg):
                    return await _handle_context_overflow(e, error_msg)
                if err_type == "invalid_request_error":
                    # Deterministic client error surfaced mid-stream -- retrying
                    # replays the identical request and fails identically.
                    #
                    # `type` is the only safe discriminator here. Do NOT widen this
                    # to "any error carrying a `code`": the codes documented for this
                    # API (openai.types.responses.ResponseError) are led by
                    # "server_error" and "rate_limit_exceeded", which are precisely
                    # the transient failures retries exist for. Classifying on the
                    # presence of a code would make them permanent.
                    raise kernel_errors.InvalidRequestError(
                        error_msg,
                        provider=self.name,
                        status_code=400,
                    ) from e
                # Unclassifiable -- preserve the prior conservative default.
                raise kernel_errors.LLMError(
                    error_msg,
                    provider=self.name,
                    retryable=True,
                ) from e
            except Exception as e:
                body = getattr(e, "body", None)
                if body is not None:
                    error_msg = json.dumps(body)
                else:
                    error_msg = str(e) or f"{type(e).__name__}: (no message)"
                raise kernel_errors.LLMError(
                    error_msg,
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

        async def _do_complete_guarded():
            """Semaphore-gated wrapper around _do_complete with concurrency logging.

            Acquires the process-wide concurrency semaphore before each API call
            attempt so that at most ``max_concurrent_requests`` calls are in-flight
            simultaneously across all provider instances in this process.

            This is the function passed to retry_with_backoff so that:
            - the semaphore is *released* between retry attempts (during backoff sleep)
            - each fresh attempt must re-acquire before hitting the network
            """
            global _active_requests, _waiting_requests
            sem = await _get_process_semaphore(self._max_concurrent_requests)
            if sem is not None:
                _waiting_requests += 1
                async with sem:
                    _waiting_requests -= 1
                    _active_requests += 1
                    try:
                        if self.coordinator and hasattr(self.coordinator, "hooks"):
                            await self.coordinator.hooks.emit(
                                "provider:concurrency",
                                {
                                    "provider": self.name,
                                    "model": params["model"],
                                    "active_requests": _active_requests,
                                    "waiting_requests": _waiting_requests,
                                    "max_concurrent": self._max_concurrent_requests,
                                    "process_id": os.getpid(),
                                },
                            )
                        return await _do_complete()
                    finally:
                        _active_requests -= 1
            else:
                # Semaphore disabled (max_concurrent_requests=0) — still log
                _active_requests += 1
                try:
                    if self.coordinator and hasattr(self.coordinator, "hooks"):
                        await self.coordinator.hooks.emit(
                            "provider:concurrency",
                            {
                                "provider": self.name,
                                "model": params["model"],
                                "active_requests": _active_requests,
                                "waiting_requests": _waiting_requests,
                                "max_concurrent": 0,
                                "process_id": os.getpid(),
                            },
                        )
                    return await _do_complete()
                finally:
                    _active_requests -= 1

        try:
            response = await retry_with_backoff(
                _do_complete_guarded,
                self._retry_config,
                on_retry=_on_retry,
            )

            elapsed_ms = int((time.time() - start_time) * 1000)

            logger.info(
                "[PROVIDER] Received response from %s API (status=%s)",
                self.api_label,
                getattr(response, "status", "unknown"),
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
            truncation_retry_done = False

            while (
                hasattr(final_response, "status")
                and final_response.status == "incomplete"
                and continuation_count < MAX_CONTINUATION_ATTEMPTS
            ):
                # P4: a response that ended INSIDE a function_call cannot be
                # resumed by continuation — arguments are not resumable, so
                # each continuation restarts and truncates again (observed
                # live: 5 fruitless continuations, then `{}`-argument calls
                # that 400'd the session). Policy: discard the truncated
                # output and RETRY the original request once with the output
                # budget raised to the model's capability cap. If we are
                # already at the cap (or the retry also truncates), fail
                # loud with a structured error naming the truncation.
                incomplete_calls = describe_incomplete_function_calls(
                    list(getattr(final_response, "output", []) or [])
                )
                if incomplete_calls:
                    caps = get_capabilities(params["model"])
                    cap_tokens = caps.max_output_tokens or DEFAULT_MAX_TOKENS
                    current_budget = params.get("max_output_tokens") or 0
                    call_desc = ", ".join(
                        f"{c['name']}(call_id={c['call_id'] or c['item_id']}, {c['reason']})"
                        for c in incomplete_calls
                    )
                    if not truncation_retry_done and current_budget < cap_tokens:
                        truncation_retry_done = True
                        logger.warning(
                            "[PROVIDER] Response truncated mid-function_call (%s). "
                            "Discarding truncated output and retrying with "
                            "max_output_tokens raised %s -> %s (model cap).",
                            call_desc,
                            current_budget,
                            cap_tokens,
                        )
                        if self.coordinator and hasattr(self.coordinator, "hooks"):
                            await self.coordinator.hooks.emit(
                                "provider:truncated_function_call_retry",
                                {
                                    "provider": self.name,
                                    "model": params["model"],
                                    "incomplete_calls": incomplete_calls,
                                    "previous_max_output_tokens": current_budget,
                                    "retry_max_output_tokens": cap_tokens,
                                },
                            )
                        params["max_output_tokens"] = cap_tokens
                        retry_start = time.time()
                        final_response = await asyncio.wait_for(
                            self._create_response(params),
                            timeout=self.timeout,
                        )
                        elapsed_ms += int((time.time() - retry_start) * 1000)
                        # Nothing was executed from the truncated attempt;
                        # replace the accumulated output wholesale.
                        accumulated_output = (
                            list(final_response.output)
                            if hasattr(final_response, "output")
                            else []
                        )
                        continue
                    raise FunctionCallTruncationError(
                        f"Response truncated mid-function_call and cannot be "
                        f"recovered: {call_desc}. Output budget "
                        f"{current_budget} tokens"
                        + (
                            " (already at the model's max_output_tokens cap"
                            + (
                                "; a raised-budget retry was already attempted"
                                if truncation_retry_done
                                else ""
                            )
                            + ")"
                        )
                        + ". The tool call's arguments exceeded the budget; "
                        "refusing to surface a truncated call as executable.",
                        provider=self.name,
                        model=params["model"],
                    )

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
                # Note: continue_params intentionally does NOT inherit
                # previous_response_id. The incomplete-continuation path uses
                # _build_continuation_input() to carry context forward
                # explicitly. Mixing previous_response_id with a rebuilt input
                # array would double-count tokens.
                if "store" in params:
                    continue_params["store"] = params["store"]
                if "prompt_cache_key" in params:
                    continue_params["prompt_cache_key"] = params["prompt_cache_key"]
                if "prompt_cache_retention" in params:
                    continue_params["prompt_cache_retention"] = params[
                        "prompt_cache_retention"
                    ]
                if "safety_identifier" in params:
                    continue_params["safety_identifier"] = params["safety_identifier"]
                if "prompt_cache_options" in params:
                    continue_params["prompt_cache_options"] = params[
                        "prompt_cache_options"
                    ]

                # Make continuation call
                try:
                    continue_start = time.time()
                    final_response = await asyncio.wait_for(
                        self._create_response(continue_params),
                        timeout=self.timeout,
                    )
                    continue_elapsed = int((time.time() - continue_start) * 1000)
                    elapsed_ms += continue_elapsed

                    # Accumulate output from continuation
                    if hasattr(final_response, "output"):
                        accumulated_output.extend(final_response.output)

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

            # Convert to ChatResponse FIRST (before emitting llm:response)
            # so event usage fields come from the canonical ChatResponse
            if continuation_count > 0:
                # Use new helper for accumulated output
                chat_response = convert_response_with_accumulated_output(
                    response, accumulated_output, continuation_count, OpenAIChatResponse
                )
            else:
                # Use existing conversion for normal (non-continued) responses
                chat_response = self._convert_to_chat_response(response)

            # Emit llm:response event using canonical usage fields from chat_response
            if self.coordinator and hasattr(self.coordinator, "hooks"):
                event_usage: dict[str, Any] = {}
                if chat_response.usage:
                    event_usage["input_tokens"] = chat_response.usage.input_tokens
                    event_usage["output_tokens"] = chat_response.usage.output_tokens
                    if chat_response.usage.cache_read_tokens is not None:
                        event_usage["cache_read_tokens"] = (
                            chat_response.usage.cache_read_tokens
                        )
                    # cache_write_tokens is REQUIRED for a consumer to reconstruct
                    # gross input. Usage.input_tokens is normalized to
                    # "fresh + cache_read" (cache_write subtracted out -- see
                    # _convert_to_chat_response), so gross input is only
                    # recoverable as input_tokens + cache_write_tokens. Omitting
                    # it here silently understated a cold turn's input to just
                    # the fresh remainder (measured: a real 45,320-token cache
                    # write serialized away to nothing, leaving "3"). Emitted
                    # whenever the provider measured it -- None only for
                    # pre-5.6 models, which never write cache and for which
                    # input_tokens is already the full gross.
                    if chat_response.usage.cache_write_tokens is not None:
                        event_usage["cache_write_tokens"] = (
                            chat_response.usage.cache_write_tokens
                        )
                    _cost_usd = getattr(chat_response.usage, "cost_usd", None)
                    event_usage["cost_usd"] = (
                        str(_cost_usd) if _cost_usd is not None else None
                    )
                response_event: dict[str, Any] = {
                    "provider": self.name,
                    "model": params["model"],
                    "usage": event_usage,
                    "status": "ok",
                    "duration_ms": elapsed_ms,
                    "continuation_count": continuation_count
                    if continuation_count > 0
                    else None,
                }
                if self.raw:
                    response_event["raw"] = redact_secrets(response.model_dump())
                if captured_rate_limit_info:
                    response_event["rate_limits"] = captured_rate_limit_info
                await self.coordinator.hooks.emit("llm:response", response_event)

            return chat_response

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

    def _extract_rate_limit_headers(self, headers: Any) -> dict[str, Any]:
        """Extract rate limit information from OpenAI response headers.

        OpenAI returns rate limit headers on every response:
        - x-ratelimit-limit-requests / x-ratelimit-remaining-requests / x-ratelimit-reset-requests
        - x-ratelimit-limit-tokens  / x-ratelimit-remaining-tokens  / x-ratelimit-reset-tokens

        Args:
            headers: Response headers (dict-like object, or None)

        Returns:
            Dict with parsed rate limit values, or empty dict if unavailable.
        """
        if not headers:
            return {}

        def get_int(key: str) -> int | None:
            val = headers.get(key)
            if val is not None:
                try:
                    return int(val)
                except (ValueError, TypeError):
                    pass
            return None

        def get_str(key: str) -> str | None:
            val = headers.get(key)
            if val is not None and val != "":
                return str(val)
            return None

        info: dict[str, Any] = {}

        requests_limit = get_int("x-ratelimit-limit-requests")
        requests_remaining = get_int("x-ratelimit-remaining-requests")
        requests_reset = get_str("x-ratelimit-reset-requests")
        if requests_limit is not None:
            info["requests_limit"] = requests_limit
        if requests_remaining is not None:
            info["requests_remaining"] = requests_remaining
        if requests_reset is not None:
            info["requests_reset"] = requests_reset

        tokens_limit = get_int("x-ratelimit-limit-tokens")
        tokens_remaining = get_int("x-ratelimit-remaining-tokens")
        tokens_reset = get_str("x-ratelimit-reset-tokens")
        if tokens_limit is not None:
            info["tokens_limit"] = tokens_limit
        if tokens_remaining is not None:
            info["tokens_remaining"] = tokens_remaining
        if tokens_reset is not None:
            info["tokens_reset"] = tokens_reset

        return info

    def _convert_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        skip_reasoning_reinsertion: bool = False,
    ) -> list[dict[str, Any]]:
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
                        # Use computer_call_output (with an image envelope, not
                        # a stringified blob) for native computer calls. This
                        # must be checked before the generic native-call branch
                        # below, since computer call_ids are also present in
                        # _native_call_ids but need a different result shape.
                        if self._native_call_types.get(tool_call_id) == "computer":
                            image_url = _extract_computer_screenshot_data_url(
                                tool_content
                            )
                            openai_messages.append(
                                {
                                    "type": "computer_call_output",
                                    "call_id": tool_call_id,
                                    "output": {
                                        "type": "computer_screenshot",
                                        "image_url": image_url,
                                        "detail": "original",
                                    },
                                }
                            )
                        # Use apply_patch_call_output for native apply_patch calls
                        elif tool_call_id in self._native_call_ids:
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
                                    try:
                                        tc_input = json.loads(tc_input)
                                    except (json.JSONDecodeError, TypeError):
                                        tc_input = {}
                                if not isinstance(tc_input, dict):
                                    tc_input = {}
                                if tc_id and tc_name:
                                    # Detect historical native apply_patch_call by operation shape:
                                    # native calls store {"type": <op>, "path": ..., "diff": ...}
                                    # where <op> is one of the known operation types.
                                    _native_op_types = {
                                        "update_file",
                                        "create_file",
                                        "delete_file",
                                        "rename_file",
                                    }
                                    if (
                                        tc_name == "apply_patch"
                                        and tc_input.get("type") in _native_op_types
                                    ):
                                        # Restore as native apply_patch_call so the output
                                        # is also replayed with the correct type.
                                        self._native_call_ids.add(tc_id)
                                        function_call_items.append(
                                            {
                                                "type": "apply_patch_call",
                                                "call_id": tc_id,
                                                "operation": {
                                                    k: v
                                                    for k, v in tc_input.items()
                                                    if not (
                                                        k == "diff"
                                                        and tc_input.get("type")
                                                        not in (
                                                            "create_file",
                                                            "update_file",
                                                        )
                                                    )
                                                },
                                                "status": "completed",
                                            }
                                        )
                                    elif tc_name == "computer" and isinstance(
                                        tc_input.get("actions"), list
                                    ):
                                        # Detect historical native computer_call by input
                                        # shape: native calls store {"actions": [...]}
                                        # (see _extract_computer_actions). Restore as
                                        # native computer_call so the result is replayed
                                        # with computer_call_output, not function_call_output.
                                        self._native_call_ids.add(tc_id)
                                        self._native_call_types[tc_id] = "computer"
                                        function_call_items.append(
                                            {
                                                "type": "computer_call",
                                                "call_id": tc_id,
                                                "actions": tc_input.get("actions", []),
                                                "status": "completed",
                                            }
                                        )
                                    else:
                                        tc_args_str = (
                                            json.dumps(tc_input) if tc_input else "{}"
                                        )
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
                                        if skip_reasoning_reinsertion:
                                            # Chaining is active: server holds reasoning state under
                                            # previous_response_id. Don't re-emit encrypted_content
                                            # (would bust the cache prefix) and don't emit bare rs_*
                                            # IDs (server already knows them).
                                            pass
                                        else:
                                            reasoning_items_to_add.append(
                                                reasoning_item
                                            )
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
                                    try:
                                        tc_input = json.loads(tc_input)
                                    except (json.JSONDecodeError, TypeError):
                                        tc_input = {}
                                if not isinstance(tc_input, dict):
                                    tc_input = {}
                                if tc_id and tc_name:
                                    # Detect historical native apply_patch_call by operation shape:
                                    # native calls store {"type": <op>, "path": ..., "diff": ...}
                                    # where <op> is one of the known operation types.
                                    _native_op_types = {
                                        "update_file",
                                        "create_file",
                                        "delete_file",
                                        "rename_file",
                                    }
                                    if tc_name == "computer" and isinstance(
                                        tc_input.get("actions"), list
                                    ):
                                        # Detect historical native computer_call by input
                                        # shape: native calls store {"actions": [...]}
                                        # (see _extract_computer_actions). Restore as
                                        # native computer_call so the result is replayed
                                        # with computer_call_output, not
                                        # function_call_output.
                                        self._native_call_ids.add(tc_id)
                                        self._native_call_types[tc_id] = "computer"
                                        function_call_items.append(
                                            {
                                                "type": "computer_call",
                                                "call_id": tc_id,
                                                "actions": tc_input.get("actions", []),
                                                "status": "completed",
                                            }
                                        )
                                    elif (
                                        tc_name == "apply_patch"
                                        and tc_input.get("type") in _native_op_types
                                    ):
                                        # Restore as native apply_patch_call so the output
                                        # is also replayed with the correct type.
                                        self._native_call_ids.add(tc_id)
                                        function_call_items.append(
                                            {
                                                "type": "apply_patch_call",
                                                "call_id": tc_id,
                                                "operation": {
                                                    k: v
                                                    for k, v in tc_input.items()
                                                    if not (
                                                        k == "diff"
                                                        and tc_input.get("type")
                                                        not in (
                                                            "create_file",
                                                            "update_file",
                                                        )
                                                    )
                                                },
                                                "status": "completed",
                                            }
                                        )
                                    else:
                                        tc_args_str = (
                                            json.dumps(tc_input) if tc_input else "{}"
                                        )
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

                                    if skip_reasoning_reinsertion:
                                        # Chaining is active: server holds reasoning state under
                                        # previous_response_id. Don't re-emit encrypted_content
                                        # (would bust the cache prefix) and don't emit bare rs_*
                                        # IDs (server already knows them).
                                        pass
                                    else:
                                        reasoning_items_to_add.append(reasoning_item)

                # Handle simple string content
                elif isinstance(content, str) and content:
                    assistant_content.append({"type": "output_text", "text": content})

                # Defensive: strip orphaned reasoning items that have no encrypted_content.
                # These occur when the model reasoned but include=[reasoning.encrypted_content]
                # was not requested — the reasoning ID exists but can't be sent back, causing 404s.
                #
                # When chaining is active (skip_reasoning_reinsertion=True), the server holds
                # reasoning state under previous_response_id; we deliberately did not collect
                # encrypted_content into reasoning_items_to_add. So the "metadata has reasoning
                # IDs but list is empty" condition is the expected steady state, not an orphan.
                # Skip the orphan check to avoid spurious warnings and unnecessary clears.
                if (
                    not skip_reasoning_reinsertion
                    and metadata
                    and metadata.get(METADATA_REASONING_ITEMS)
                ):
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
                        _build_assistant_message_item(assistant_content)
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

        # P4 wire-path invariant: every function_call item replayed into the
        # input MUST have a function_call_output paired by call_id, or the
        # API rejects the whole request (400 "No tool output found for
        # function call <call_id>") and kills the session. The message-level
        # repair (_find_missing_tool_results) is the primary net; this is the
        # last-resort backstop at the wire format itself — if an orphan
        # slipped through, synthesize an error output rather than letting the
        # request 400.
        output_call_ids = {
            item.get("call_id")
            for item in openai_messages
            if isinstance(item, dict) and item.get("type") == "function_call_output"
        }
        repaired: list[dict[str, Any]] = []
        for item in openai_messages:
            repaired.append(item)
            if (
                isinstance(item, dict)
                and item.get("type") == "function_call"
                and item.get("call_id")
                and item["call_id"] not in output_call_ids
            ):
                logger.warning(
                    "[PROVIDER] Orphaned function_call %s (%s) reached the wire "
                    "path with no paired output; synthesizing an error output "
                    "to keep the request valid.",
                    item["call_id"],
                    item.get("name", "?"),
                )
                repaired.append(
                    {
                        "type": "function_call_output",
                        "call_id": item["call_id"],
                        "output": (
                            "[error] Tool execution result missing (the call was "
                            "interrupted). Synthesized to preserve conversation "
                            "validity — re-issue the tool call if still needed."
                        ),
                    }
                )
        return repaired

    def _convert_tools_from_request(
        self, tools: list, model_name: str | None = None
    ) -> list[dict[str, Any]]:
        """Convert ToolSpec objects from ChatRequest to OpenAI format.

        Handles both user-defined function tools and native OpenAI-hosted tools
        (web_search_preview, file_search, code_interpreter).

        Native tools are passed through directly when specified as dicts with
        a recognized 'type' field. User-defined tools are converted to function
        tool format.

        Args:
            tools: List of ToolSpec objects or native tool dicts
            model_name: Model actually being used for this request. Required to
                decide whether the native `{"type": "apply_patch"}` tool shape is
                safe to send (not every model supports it — see
                ModelCapabilities.supports_native_apply_patch). Falls back to
                self.default_model when omitted (e.g. existing call sites/tests).

        Returns:
            List of OpenAI-formatted tool definitions
        """
        openai_tools = []
        resolved_model = model_name or self.default_model

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
                # Native `computer` tool carried via ToolSpec extras. ToolSpec
                # is declared `extra="allow"` (amplifier_core.message_models),
                # so a tool that exposes `native_tool_spec = {"type": "computer"}`
                # (see amplifier-module-loop-streaming's native-tool-spec
                # mechanism) rides a `type="computer"` attribute onto the
                # resulting ToolSpec instance untouched. Detect it the same
                # way: an extra attribute, not a hardcoded name check.
                #
                # Unlike apply_patch, `computer` must be emitted completely
                # bare -- no name/description/parameters. Live Responses API
                # traffic confirms the tool accepts *zero* declaration fields:
                # `{"type": "computer"}` -> 200; `display_width`,
                # `display_height`, `environment`, `display_width_px` (any of
                # them, alone) -> 400 "Unknown parameter". Forwarding
                # tool.description/tool.parameters here would break the
                # request outright, so this branch discards them by
                # construction rather than by omission.
                if getattr(tool, "type", None) == "computer":
                    openai_tools.append({"type": "computer"})
                    continue

                # Special handling for apply_patch with native engine — but only
                # for models confirmed to support OpenAI's native apply_patch tool
                # type. Sending {"type": "apply_patch"} to an unsupported model
                # (e.g. gpt-5-mini) causes the API to reject the request outright:
                # "Tool 'apply_patch' is not supported with gpt-5-mini". Fall back
                # to the already-supported function-tool shape in that case.
                if tool.name == "apply_patch" and self._apply_patch_native:
                    if get_capabilities(resolved_model).supports_native_apply_patch:
                        openai_tools.append({"type": "apply_patch"})
                        continue
                    logger.info(
                        "[PROVIDER] Model %s does not support native apply_patch; "
                        "falling back to function-tool mode for this request.",
                        resolved_model,
                    )

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
                    # P4: call_id-keyed; raises FunctionCallTruncationError on
                    # incomplete/unparseable calls instead of surfacing {}.
                    tool_id, tool_name, tool_input = parse_function_call_block(block)
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

                elif block_type == "computer_call":
                    call_id = getattr(block, "call_id", "")
                    args = {"actions": _extract_computer_actions(block)}
                    content_blocks.append(
                        ToolCallBlock(id=call_id, name="computer", input=args)
                    )
                    tool_calls.append(
                        ToolCall(id=call_id, name="computer", arguments=args)
                    )
                    # Track for round-trip output format (computer_call_output,
                    # not apply_patch_call_output -- see _native_call_types).
                    self._native_call_ids.add(call_id)
                    self._native_call_types[call_id] = "computer"

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
                    # P4: call_id-keyed; raises FunctionCallTruncationError on
                    # incomplete/unparseable calls instead of surfacing {}.
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

                elif block_type == "computer_call":
                    call_id = block.get("call_id", "")
                    args = {"actions": _extract_computer_actions(block)}
                    content_blocks.append(
                        ToolCallBlock(id=call_id, name="computer", input=args)
                    )
                    tool_calls.append(
                        ToolCall(id=call_id, name="computer", arguments=args)
                    )
                    self._native_call_ids.add(call_id)
                    self._native_call_types[call_id] = "computer"

        # Extract usage counts.
        #
        # OpenAI's usage.input_tokens (Responses API) is the RAW vendor gross
        # total: fresh + cache_read + cache_write ALL COMBINED (cache_write is
        # a SUBSET of it -- see the `fresh_input = prompt_tokens - cached -
        # cache_write` derivation in _cost.py, confirmed against live
        # gpt-5.6-sol usage). This differs from Anthropic, where cache_write
        # (cache_creation) is a genuinely DISJOINT bucket reported on top of
        # input_tokens.
        #
        # The kernel Usage contract (amplifier_core CONTRACTS.md) normalizes
        # input_tokens to "gross total: fresh + cache_read combined,
        # cache_write NOT included" -- every consumer (e.g. the streaming-UI
        # display) computes `total_input = input_tokens + cache_write_tokens`
        # assuming that shape. Emitting OpenAI's raw total unmodified (already
        # containing cache_write) makes that formula double-count the write
        # tokens (measured: 18,053 displayed vs. a true 9,028 gross input).
        #
        # `_raw_input_tokens` is kept SEPARATELY from the normalized
        # `usage_counts["input"]` because compute_cost() below needs the raw
        # vendor gross (its own `prompt_tokens` param already subtracts
        # cached/cache_write internally) -- normalizing input_tokens here must
        # not also perturb the cost computation.
        usage_obj = response.usage if hasattr(response, "usage") else None
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

        # Normalize: cache_write is a SUBSET of the raw OpenAI total, so
        # remove it to get the "fresh + cache_read" gross the kernel contract
        # expects. max(0, ...) guards against a malformed/unexpected API
        # payload where cache_write_tokens would otherwise exceed input_tokens.
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

        # M2: Stamp cost_usd onto Usage (zero-transformation passthrough from API fields).
        # prompt_tokens is the total including cached AND cache-write; both are subtracted
        # inside compute_cost to prevent double-charging. NOTE: this deliberately uses
        # `_raw_input_tokens` (the unnormalized vendor gross), NOT `usage_counts["input"]`
        # (which has cache_write subtracted out for the public Usage contract above) --
        # compute_cost's own internal subtraction expects the raw combined total.
        if usage_obj:
            _prompt_tokens = getattr(usage_obj, "prompt_tokens", _raw_input_tokens)
            _completion_tokens = getattr(
                usage_obj, "completion_tokens", usage_counts["output"]
            )
            _cached_tokens = cache_read_tokens or 0
            _cache_write_tokens = cache_write_tokens or 0
            cost = compute_cost(
                getattr(response, "model", ""),
                prompt_tokens=_prompt_tokens,
                completion_tokens=_completion_tokens,
                cached_tokens=_cached_tokens,
                cache_write_tokens=_cache_write_tokens,
            )
            if cost is not None:
                usage = usage.model_copy(update={"cost_usd": cost})
                self._add_cost(cost)

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

    async def close(self) -> None:
        """Close the underlying OpenAI client to prevent resource leaks."""
        if self._client is not None:
            await self._client.close()
            self._client = None
