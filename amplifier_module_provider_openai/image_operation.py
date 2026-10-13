"""Experimental instance-bound image operation, separate from chat completion."""
from __future__ import annotations

import base64
import hashlib
import json
import re
import uuid
from copy import deepcopy
from io import BytesIO

import httpx

PROTOCOL = "amplifier.image-operation.v1"
MAX_BYTES = 8 * 1024 * 1024
MODELS = {"gpt-image-1", "gpt-image-1-mini", "gpt-image-1.5", "gpt-image-2",
          "gpt-image-2.5-flare", "gpt-image-2.5-sunburst"}
CONTROLLERS = {"gpt-4.1", "gpt-4.1-mini", "gpt-4o", "gpt-5", "gpt-5-mini",
               "gpt-5.1", "gpt-5.2", "gpt-5.4", "gpt-5.5", "gpt-5.6-sol",
               "gpt-5.6-terra", "gpt-5.6-luna", "gpt-6-astra", "gpt-6-sol",
               "gpt-6-luna", "gpt-6.1-sol"}

class _BoundHTTP(httpx.AsyncClient):
    """Borrow the selected HTTP client without changing or closing it."""
    def __init__(self, original):
        super().__init__(follow_redirects=False)
        self.original = original

    def build_request(self, *args, **kwargs):
        return self.original.build_request(*args, **kwargs)

    async def send(self, request, **kwargs):
        kwargs["follow_redirects"] = False
        return await self.original.send(request, **kwargs)


def unavailable(reason):
    return {"protocol": PROTOCOL, "supported": False, "reason_code": reason,
            "safe_message": "This connection cannot prepare the requested image operation."}


def capabilities(provider):
    supported = provider._uses_standard_openai_endpoint() and isinstance(provider._api_key, str) and bool(provider._api_key)
    if provider._client is not None:
        from urllib.parse import urlparse
        url = urlparse(str(provider._client.base_url))
        supported = (supported and isinstance(provider._client.api_key, str)
                     and getattr(provider._client, "_api_key_provider", None) is None
                     and not any((url.username, url.password, url.query, url.fragment)))
    settings = provider.config.get("image_generation", {})
    if not isinstance(settings, dict) or ("enabled" in settings and settings["enabled"] is not True):
        supported = False
    return {"protocol": PROTOCOL, "supported": supported,
            "operations": ["generate", "edit"] if supported else [],
            "output_formats": ["image/png"], "entitlement": "unverified",
            "transports": ["responses", "images"] if supported else [],
            "automatic_retries": False}


def validate(request):
    if not isinstance(request, dict) or set(request) != {
        "protocol", "request_id", "operation", "prompt", "inputs", "selection", "output"
    } or request["protocol"] != PROTOCOL:
        raise ValueError("Unsupported image operation request.")
    if not isinstance(request["request_id"], str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", request["request_id"]):
        raise ValueError("Invalid image request identity.")
    if not isinstance(request["prompt"], str) or not 1 <= len(request["prompt"].strip()) <= 32000:
        raise ValueError("Invalid image prompt.")
    inputs = request["inputs"]
    if not isinstance(inputs, list) or len(inputs) > 4 or request["operation"] not in {"generate", "edit"} or (request["operation"] == "edit") != bool(inputs):
        raise ValueError("Generate takes no images; edit takes one to four images.")
    compact = deepcopy({k: v for k, v in request.items() if k != "inputs"})
    compact["inputs"] = []
    for index, image in enumerate(inputs):
        if not isinstance(image, dict) or set(image) != {"purpose", "mime_type", "data", "sha256"}:
            raise ValueError("Invalid image input fields.")
        data = image["data"]
        if not isinstance(data, bytes) or not 0 < len(data) <= MAX_BYTES or hashlib.sha256(data).hexdigest() != image["sha256"]:
            raise ValueError("Invalid image input identity or size.")
        if image["purpose"] != ("target" if index == 0 else "reference"):
            raise ValueError("Invalid image input order.")
        raster(data, image["mime_type"])
        compact["inputs"].append({k: v for k, v in image.items() if k != "data"})
    selection = request["selection"]
    if not isinstance(selection, dict) or set(selection) != {"model", "config"} or not isinstance(selection["config"], dict):
        raise ValueError("Invalid image selection.")
    config = selection["config"]
    # Ordinary role preferences carry chat effort; image execution never forwards
    # arbitrary wire, account or transport-client overrides from those fields.
    if set(config) - {"image_generation", "reasoning_effort", "max_output_tokens"}:
        raise ValueError("Unsupported image preference override.")
    image_config = config.get("image_generation", {})
    if not isinstance(image_config, dict) or set(image_config) - {"model", "transport"}:
        raise ValueError("Unsupported image execution override.")
    output = request["output"]
    if not isinstance(output, dict) or set(output) != {"count", "mime_type", "size", "quality", "background"}:
        raise ValueError("Invalid image output requirements.")
    return hashlib.sha256(json.dumps(compact, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def raster(data, mime_type="image/png", size="auto"):
    from PIL import Image
    if not isinstance(data, bytes) or not 0 < len(data) <= MAX_BYTES:
        raise ValueError("Invalid image output size.")
    formats = {"image/png": "PNG", "image/jpeg": "JPEG", "image/webp": "WEBP"}
    with Image.open(BytesIO(data)) as image:
        if image.format != formats.get(mime_type) or getattr(image, "n_frames", 1) != 1:
            raise ValueError("Image bytes do not match the declared format.")
        if not 0 < image.width <= 4096 or not 0 < image.height <= 4096:
            raise ValueError("Image dimensions exceed the artifact limit.")
        if size != "auto" and size != f"{image.width}x{image.height}":
            raise ValueError("Image does not meet the required dimensions.")
        if mime_type == "image/png" and image.mode not in {"RGB", "RGBA"}:
            raise ValueError("An RGB/RGBA PNG is required.")
        image.verify()
    with Image.open(BytesIO(data)) as image:
        image.load()
    return data


def binding(provider):
    client = provider.client
    # Kept privately, never serialized into plans or diagnostics.
    return (client, provider._api_key, getattr(client, "api_key", None),
            str(client.base_url), getattr(client, "organization", None),
            getattr(client, "project", None),
            # SDK omission sentinels are not transmitted and don't retain
            # equality under deepcopy. Snapshot effective scalar headers only.
            {k: v for k, v in client.default_headers.items() if isinstance(v, str)},
            getattr(client, "_client", None),
            dict(client._client.headers), client._client.auth,
            provider.timeout,
            deepcopy(provider.config.get("image_generation")))


async def prepare(provider, request):
    request_hash = validate(request)
    if not capabilities(provider)["supported"]:
        return unavailable("unsupported_connection")
    config = request["selection"]["config"].get("image_generation", {})
    model = config.get("model")
    transport = config.get("transport", "responses")
    controller = request["selection"]["model"]
    output = request["output"]
    if transport not in {"responses", "images"} or model not in MODELS:
        return unavailable("unsupported_model_or_transport")
    if transport == "responses" and controller not in CONTROLLERS:
        return unavailable("unsupported_controller")
    if output["count"] != 1 or type(output["count"]) is not int or output["mime_type"] != "image/png":
        return unavailable("unsupported_output")
    if output["size"] not in {"auto", "1024x1024", "1536x1024", "1024x1536"} or output["quality"] not in {"auto", "low", "medium", "high"} or output["background"] not in {"auto", "opaque", "transparent"}:
        return unavailable("unsupported_controls")
    client = provider.client
    if not provider._uses_standard_openai_endpoint():
        return unavailable("unsupported_connection")
    endpoint = client.responses if transport == "responses" else client.images
    method = "create" if transport == "responses" else ("edit" if request["operation"] == "edit" else "generate")
    if not callable(getattr(endpoint, method, None)):
        return unavailable("unsupported_sdk")
    # No model-list requirement: exact supported IDs can be used on accounts
    # whose catalog is incomplete. Entitlement remains unverified, not assumed.
    plan = {"protocol": PROTOCOL, "id": uuid.uuid4().hex, "request_hash": request_hash,
            "transport": transport, "model": model,
            "controller": controller if transport == "responses" else None,
            "output": deepcopy(output)}
    plans = getattr(provider, "_prepared_image_operations", None)
    if plans is None:
        plans = provider._prepared_image_operations = {}
    if len(plans) >= 64:
        raise ValueError("Too many unconsumed image preparations.")
    plans[plan["id"]] = (deepcopy(plan), binding(provider))
    return {"protocol": PROTOCOL, "supported": True, "plan": plan}


def _dump(value):
    return value.model_dump(mode="json", warnings="error") if callable(getattr(value, "model_dump", None)) else value


async def generate(provider, request, *, plan):
    request_hash = validate(request)
    plans = getattr(provider, "_prepared_image_operations", {})
    saved = plans.get(plan.get("id")) if isinstance(plan, dict) else None
    if saved is None or saved[0] != plan or request_hash != plan["request_hash"] or saved[1] != binding(provider) or not capabilities(provider)["supported"]:
        raise ValueError("The image preparation is invalid, consumed or bound to a changed connection.")
    # Consume before the first await. Even an interrupted invocation cannot reuse
    # this provider plan; durable cross-process identity remains the caller's job.
    del plans[plan["id"]]
    plan = deepcopy(saved[0])
    try:
        return await _dispatch(provider, request, plan)
    except Exception:  # noqa: BLE001 - safe unknown-outcome boundary, never retry
        # SDK errors may echo request bodies. Cancellation still propagates;
        # every other post-admission failure is safe, non-replayable evidence.
        return {"protocol": PROTOCOL, "status": "unknown", "outputs": [],
                "reason_code": "generation_outcome_unknown", "dispatch_state": "unknown"}


async def _dispatch(provider, request, plan):
    async with _BoundHTTP(provider.client._client) as transport:
        client = provider.client.with_options(max_retries=0, timeout=provider.timeout, http_client=transport)
        return await _dispatch_with_client(client, request, plan)


async def _dispatch_with_client(client, request, plan):
    options = {k: plan["output"][k] for k in ("size", "quality", "background")}
    if plan["transport"] == "responses":
        content = [{"type": "input_text", "text": request["prompt"]}]
        for image in request["inputs"]:
            content.append({"type": "input_image", "image_url": "data:" + image["mime_type"] + ";base64," + base64.b64encode(image["data"]).decode()})
        response = await client.responses.create(
            model=plan["controller"], input=[{"role": "user", "content": content}],
            tools=[{"type": "image_generation", "model": plan["model"], "action": request["operation"], "output_format": "png", **options}],
            tool_choice={"type": "image_generation"}, max_tool_calls=1,
            parallel_tool_calls=False, store=False, stream=False,
        )
    else:
        params = {"model": plan["model"], "prompt": request["prompt"], "n": 1, "output_format": "png", **options}
        if request["operation"] == "edit":
            params["image"] = [(f"input-{i}.png", image["data"], image["mime_type"]) for i, image in enumerate(request["inputs"])]
            response = await client.images.edit(**params)
        else:
            response = await client.images.generate(**params)
    return _parse_response(response, plan)


def _parse_response(response, plan):
    """Retain bounded, independently validated evidence even on parse failure."""
    responses = plan["transport"] == "responses"
    reason = "incomplete_response" if responses and getattr(response, "status", None) != "completed" else None
    outputs, calls = [], []

    def invalid():
        nonlocal reason
        reason = reason or "generation_outcome_unknown"

    def identifier(value):
        if value is None or isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.:/-]{1,512}", value):
            return value
        invalid()
        return None

    def append(output):
        if len(outputs) >= 16:
            raise ValueError("Oversized image operation output.")
        if output["kind"] == "text":
            text = output["text"]
            if not isinstance(text, str) or len(text) + sum(len(o.get("text", "")) for o in outputs) > 32000:
                raise ValueError("Invalid or oversized image operation text.")
        outputs.append(output)

    upstream = {"request_id": identifier(getattr(response, "_request_id", None))}
    models = [{"purpose": "image", "requested": plan["model"], "resolved": None}]
    if responses:
        upstream.update(response_id=identifier(getattr(response, "id", None)), image_call_id=None)
        models.insert(0, {"purpose": "controller", "requested": plan["controller"],
                          "resolved": identifier(getattr(response, "model", None))})
    items = getattr(response, "output" if responses else "data", None)
    if not isinstance(items, list):
        invalid()
        items = []
    for item in items:
        try:
            kind = item.type if responses else "image_generation_call"
            if kind == "image_generation_call":
                if responses:
                    calls.append(item)
                    if getattr(item, "status", None) != "completed":
                        raise ValueError("Incomplete image output.")
                encoded = item.result if responses else item.b64_json
                if not isinstance(encoded, str) or len(encoded) > 12 * 1024 * 1024:
                    raise ValueError("Missing or oversized base64 image output.")
                append({"kind": "image", "mime_type": "image/png",
                        "data": raster(base64.b64decode(encoded, validate=True), size=plan["output"]["size"])})
            elif kind == "message":
                if getattr(item, "status", None) != "completed" or not isinstance(item.content, list):
                    raise ValueError("Incomplete or malformed message output.")
                for part in item.content:
                    if part.type != "output_text":
                        raise ValueError("Unexpected image operation text.")
                    append({"kind": "text", "text": part.text})
            elif kind != "reasoning":
                raise ValueError("Unexpected image operation output.")
        except Exception:  # noqa: BLE001 - keep earlier validated outputs, never exception text
            invalid()
    if responses and len(calls) == 1:
        upstream["image_call_id"] = identifier(getattr(calls[0], "id", None))
        models[1]["resolved"] = identifier(getattr(calls[0], "model", None))
    try:
        usage = _dump(getattr(response, "usage", None))
        if usage is not None and (
            not isinstance(usage, dict) or len(json.dumps(usage, allow_nan=False)) > 32000
            or any(usage.get(k) is not None and (type(usage[k]) is not int or usage[k] < 0)
                   for k in ("input_tokens", "output_tokens", "total_tokens"))
        ):
            raise ValueError("Malformed image operation usage.")
    except Exception:  # noqa: BLE001 - metadata failure must not discard paid images
        invalid()
        usage = None
    count = sum(o["kind"] == "image" for o in outputs)
    normalized = {k: usage[k] for k in ("input_tokens", "output_tokens", "total_tokens")
                  if isinstance(usage, dict) and type(usage.get(k)) is int and usage[k] >= 0}
    result = {"protocol": PROTOCOL, "status": "unknown" if reason else "completed" if count == 1 else "failed",
              "outputs": outputs, "models": models, "upstream": upstream,
              "usage": {"normalized": normalized or None, "provider": usage},
              "effective_options": plan["output"], "warnings": [], "continuation": None}
    if reason:
        result.update(reason_code=reason, dispatch_state="unknown")
    return result
