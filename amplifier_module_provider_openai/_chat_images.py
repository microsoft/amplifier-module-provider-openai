"""Narrow canonical chat-image projection; never fetch or relocate pixels."""

import copy
import json
import re
from collections import Counter
from urllib.parse import urlsplit

from amplifier_core.llm_errors import InvalidRequestError

from ._capabilities import _EXACT_CAPABILITIES, get_capabilities

UNAVAILABLE = (
    "[Image unavailable: this source, position, model or endpoint is not qualified "
    "for image input. Supply it in a supported user message or correlated function "
    "result on an image-capable OpenAI Responses model.]"
)
OMITTED = "[image omitted from diagnostics]"


def known_vision(model):
    # Never use the legacy optimistic unknown-bucket supports_vision default.
    caps = _EXACT_CAPABILITIES.get(model)
    legacy = re.sub(r"-\d{4}-\d{2}-\d{2}$", "", model)
    return (caps is not None and caps.supports_vision) or legacy in {
        "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-4o", "gpt-4o-mini",
        "o1", "o1-pro", "o3", "o3-pro", "o4-mini", "gpt-5-mini", "gpt-5-nano",
    }


def image_url(block):
    source = block.get("source")
    if not isinstance(source, dict):
        return None
    if source.get("type") == "base64":
        mime, data = source.get("media_type"), source.get("data")
        if isinstance(mime, str) and mime.startswith("image/") and isinstance(data, str) and data:
            return f"data:{mime};base64,{data}"
    if source.get("type") == "url":
        url = source.get("url")
        if isinstance(url, str):
            try:
                parsed = urlsplit(url)
                if parsed.scheme in {"https", "http"} and parsed.hostname and not any(c.isspace() for c in url):
                    return url
            except ValueError:
                pass
    return None


def has_images(value):
    """Inspect canonical content/envelopes, not arbitrary tool JSON or text."""
    if isinstance(value, list):
        return any(
            isinstance(block, dict) and (
                block.get("type") == "image"
                or (block.get("type") == "tool_result" and has_images(block.get("output")))
            ) for block in value
        )
    if isinstance(value, dict):
        if value.get("type") == "image":
            return True
        if value.get("type") == "tool_result":
            return has_images(value.get("output"))
        content = value.get("content")
        return (
            isinstance(content, list)
            and all(isinstance(b, dict) and b.get("type") in {"text", "image"} for b in content)
            and has_images(content)
        )
    return False


def _error():
    return InvalidRequestError(
        "Image-bearing tool result requires one matching preceding call and an "
        "unambiguous tool_call_id. Original history is unchanged; repair the "
        "correlation before continuing.", provider="openai", retryable=False,
    )


def project_messages(
    messages, *, qualified, model, native_ids=(), strict_direct=False, ordinary_names=(),
):
    """Normalize known ToolResult output lists/content envelopes on a copy.

    Scalar envelope fields, including failure/status, remain accompanying text.
    Authoritative native replay has its own validators and is not projected here.
    """
    result, calls, results = [], {}, set()
    native_ids = set(native_ids)
    image_ids = set()
    result_counts = Counter()
    for message in messages:
        if message.get("role") == "tool" and message.get("tool_call_id"):
            result_counts[message["tool_call_id"]] += 1
        elif message.get("role") in {"user", "tool"} and isinstance(message.get("content"), list):
            for block in message["content"]:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    result_counts[block.get("tool_call_id")] += 1
    for original in messages:
        msg = copy.deepcopy(original)
        role, content = msg.get("role"), msg.get("content")
        metadata = msg.get("metadata") or {}
        opaque = (
            role == "assistant" and "openai:output_items" in metadata
        ) or (role == "tool" and msg.get("tool_call_id") in native_ids)
        nested_image = isinstance(content, list) and any(
            isinstance(b, dict) and b.get("type") == "tool_result" and has_images(b)
            for b in content
        )
        if role == "tool" and nested_image and len(content) != 1:
            raise _error()
        if role == "tool" and has_images(content) and not nested_image and not opaque and strict_direct:
            identity = msg.get("tool_call_id")
            if not identity or calls.get(identity) != 1 or identity in results or result_counts[identity] != 1:
                raise _error()
            image_ids.add(identity)
        if isinstance(content, list) and not opaque:
            projected = []
            for block in content:
                if not isinstance(block, dict):
                    projected.append(block)
                    continue
                kind = block.get("type")
                if kind == "tool_result" and has_images(block.get("output")):
                    identity = block.get("tool_call_id")
                    if role not in {"user", "tool"}:
                        projected.append({"type": "text", "text": UNAVAILABLE})
                        continue
                    if (
                        not identity or calls.get(identity) != 1 or identity in results
                        or result_counts[identity] != 1
                        or (msg.get("tool_call_id") and msg["tool_call_id"] != identity)
                    ):
                        raise _error()
                    if projected:
                        result.append({**msg, "content": projected})
                        projected = []
                    output = block["output"]
                    if isinstance(output, dict):
                        rest = {k: v for k, v in output.items() if k != "content"}
                        if media_values(rest) or not isinstance(output.get("content"), list):
                            raise _error()
                        output = [
                            *([{"type": "text", "text": json.dumps(rest)}] if rest else []),
                            *output["content"],
                        ]
                    if not isinstance(output, list) or any(
                        not isinstance(b, dict) or b.get("type") not in {"text", "image"}
                        for b in output
                    ):
                        raise _error()
                    status = {k: block[k] for k in ("status", "success", "is_error", "error") if k in block}
                    if status:
                        if media_values(status):
                            raise _error()
                        output = [{"type": "text", "text": json.dumps(status)}, *output]
                    result.append({
                        **msg, "role": "tool", "tool_call_id": identity,
                        "content": _project_blocks(output, qualified, model, tool=True),
                    })
                    results.add(identity)
                    image_ids.add(identity)
                else:
                    projected.extend(_project_blocks(
                        [block], qualified, model, tool=role == "tool", role=role,
                    ))
            if projected or not content:
                msg["content"] = projected
                result.append(msg)
        else:
            result.append(msg)
        if role == "tool" and has_images(content) and not nested_image and not msg.get("tool_call_id"):
            raise _error()
        if role == "tool" and not nested_image and msg.get("tool_call_id"):
            results.add(msg["tool_call_id"])
        if role != "assistant":
            continue
        call_items = metadata["openai:output_items"] if (
            role == "assistant" and "openai:output_items" in metadata
        ) else [
            *(msg.get("tool_calls") or []),
            *(content if isinstance(content, list) else []),
        ]
        for call in call_items:
            if isinstance(call, dict) and call.get("type") in {"computer_call", "apply_patch_call"}:
                native_ids.add(call.get("call_id"))
            if isinstance(call, dict) and call.get("type") in {None, "tool_call", "function_call"}:
                identity = call.get("call_id") or call.get("id") or call.get("tool_call_id")
                if identity:
                    # Equivalent Core tool_calls/content views denote one call.
                    def equivalent(block, call=call, identity=identity):
                        if not isinstance(block, dict) or block.get("type") != "tool_call":
                            return False
                        left = call.get("arguments") or call.get("input", {})
                        right = block.get("input", {})
                        try:
                            left = json.loads(left) if isinstance(left, str) else left
                            right = json.loads(right) if isinstance(right, str) else right
                        except ValueError:
                            return False
                        return (
                            block.get("id") == identity
                            and block.get("name") == call.get("name", call.get("tool", ""))
                            and left == right
                        )
                    if call in (msg.get("tool_calls") or []) and any(
                        equivalent(b) for b in (content if isinstance(content, list) else [])
                    ):
                        continue
                    calls[identity] = calls.get(identity, 0) + 1
                    if call.get("name") in ordinary_names or call.get("type") == "function_call":
                        native_ids.discard(identity)
                        continue
                    arguments = call.get("input") or {}
                    if isinstance(arguments, dict) and (
                        (call.get("name") == "computer" and isinstance(arguments.get("actions"), list))
                        or (call.get("name") == "apply_patch" and arguments.get("type") in {
                            "create_file", "update_file", "delete_file", "rename_file",
                        })
                    ):
                        native_ids.add(identity)
    if any(calls.get(identity) != 1 for identity in image_ids):
        raise _error()
    return result


def _project_blocks(blocks, qualified, model, *, tool=False, role="user"):
    out = []
    for block in blocks:
        if block.get("type") != "image":
            out.append(block)
        elif role not in {"user", "tool"} or not qualified or not known_vision(model):
            marker = (
                "[Image omitted: selected model does not support vision.]"
                if tool and not get_capabilities(model).supports_vision else UNAVAILABLE
            )
            out.append({"type": "text", "text": marker})
        elif image_url(block) is None:
            # Keep the established direct-tool validation errors payload-free.
            out.append(block if tool else {"type": "text", "text": UNAVAILABLE})
        else:
            out.append(block)
    return out


def instruction_text(content):
    if isinstance(content, str):
        return content
    return "\n".join(
        block.get("text", "") if block.get("type") == "text" else UNAVAILABLE
        for block in content if isinstance(block, dict) and block.get("type") in {"text", "image"}
    )


def media_values(value):
    """Exact known media values for echo removal; no global retained registry."""
    found = set()
    if isinstance(value, list):
        for item in value:
            found.update(media_values(item))
    elif isinstance(value, dict):
        if value.get("type") == "image":
            source = value.get("source") or {}
            source = source if isinstance(source, dict) else {}
            for key in ("data", "url"):
                if isinstance(source.get(key), str) and source[key]:
                    found.add(source[key])
        if value.get("type") in {"input_image", "computer_screenshot"}:
            url = value.get("image_url")
            if isinstance(url, str) and url:
                found.add(url)
                if url.startswith("data:") and "," in url:
                    found.add(url.split(",", 1)[1])
        for item in value.values():
            found.update(media_values(item))
    return found


def diagnostic_copy(value, echoes=()):
    """Copy before the first subscriber; never apply to SDK/replay payloads.

    Known image envelopes plus exact submitted URL/base64 echoes are omitted.
    This cannot recognize transformed media hidden in arbitrary prose.
    """
    if isinstance(value, dict):
        if value.get("type") in {"image", "input_image", "computer_screenshot"}:
            return {"type": value["type"], "media": OMITTED}
        return {k: diagnostic_copy(v, echoes) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [diagnostic_copy(v, echoes) for v in value]
    if isinstance(value, str):
        encoded = {
            encoded for secret in echoes
            for encoded in (secret, json.dumps(secret)[1:-1], json.dumps(secret, ensure_ascii=False)[1:-1])
        }
        for secret in sorted(encoded, key=len, reverse=True):
            value = value.replace(secret, OMITTED)
    return value


def sanitize_error(error, echoes):
    """Remove media echoes, without changing error policy or clean cause chains."""
    cursor, texts, seen = error, [], set()
    while cursor is not None and id(cursor) not in seen:
        seen.add(id(cursor))
        texts.append(str(cursor))
        cursor = cursor.__cause__ or cursor.__context__
    if not any(diagnostic_copy(text, echoes) != text for text in texts):
        return
    error.args = tuple(diagnostic_copy(list(error.args), echoes))
    # Some public SDK exceptions render .message/.body rather than .args.
    for name in ("message", "body", "response_details"):
        if hasattr(error, name):
            setattr(error, name, diagnostic_copy(getattr(error, name), echoes))
    error.__cause__ = None
    error.__context__ = None
    error.__suppress_context__ = True