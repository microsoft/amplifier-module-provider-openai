"""Experimental instance-bound image operation, separate from chat completion."""
from __future__ import annotations

import base64
from copy import deepcopy
import hashlib
from io import BytesIO
import json
import re
import uuid

PROTOCOL = "amplifier.image-operation.v1"
MAX_BYTES = 8 * 1024 * 1024
MODELS = {"gpt-image-1", "gpt-image-1-mini", "gpt-image-1.5", "gpt-image-2",
          "gpt-image-2.5-flare", "gpt-image-2.5-sunburst"}
CONTROLLERS = {"gpt-4.1", "gpt-4.1-mini", "gpt-4o", "gpt-5", "gpt-5-mini",
               "gpt-5.1", "gpt-5.2", "gpt-5.4", "gpt-5.5", "gpt-5.6-sol",
               "gpt-5.6-terra", "gpt-5.6-luna", "gpt-6-astra", "gpt-6-sol",
               "gpt-6-luna", "gpt-6.1-sol"}


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
            getattr(client, "_client", None), provider.timeout,
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
    return value.model_dump(mode="json") if callable(getattr(value, "model_dump", None)) else value


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
    except Exception:
        # SDK errors may echo request bodies. Cancellation still propagates;
        # every other post-admission failure is safe, non-replayable evidence.
        return {"protocol": PROTOCOL, "status": "unknown", "outputs": [],
                "reason_code": "generation_outcome_unknown", "dispatch_state": "unknown"}


async def _dispatch(provider, request, plan):
    client = provider.client.with_options(max_retries=0, timeout=provider.timeout)
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
        if response.status != "completed":
            return {"protocol": PROTOCOL, "status": "unknown", "outputs": [], "reason_code": "incomplete_response"}
        outputs, calls = [], []
        for item in response.output or []:
            if item.type == "image_generation_call":
                if getattr(item, "status", None) != "completed" or not isinstance(item.result, str) or len(item.result) > 12 * 1024 * 1024:
                    raise ValueError("Incomplete or oversized image output.")
                outputs.append({"kind": "image", "mime_type": "image/png", "data": raster(base64.b64decode(item.result, validate=True), size=options["size"])})
                calls.append(item)
            elif item.type == "message":
                for part in item.content:
                    if part.type == "output_text":
                        outputs.append({"kind": "text", "text": part.text})
        observed = getattr(response, "model", None)
        upstream = {"request_id": getattr(response, "_request_id", None), "response_id": response.id,
                    "image_call_id": calls[0].id if len(calls) == 1 else None}
        models = [{"purpose": "controller", "requested": plan["controller"], "resolved": observed},
                  {"purpose": "image", "requested": plan["model"], "resolved": getattr(calls[0], "model", None) if len(calls) == 1 else None}]
        usage = _dump(response.usage)
    else:
        params = {"model": plan["model"], "prompt": request["prompt"], "n": 1, "output_format": "png", **options}
        if request["operation"] == "edit":
            params["image"] = [(f"input-{i}.png", image["data"], image["mime_type"]) for i, image in enumerate(request["inputs"])]
            response = await client.images.edit(**params)
        else:
            response = await client.images.generate(**params)
        outputs = []
        for item in response.data or []:
            if not isinstance(item.b64_json, str) or len(item.b64_json) > 12 * 1024 * 1024:
                raise ValueError("Missing or oversized base64 image output.")
            outputs.append({"kind": "image", "mime_type": "image/png", "data": raster(base64.b64decode(item.b64_json, validate=True), size=options["size"])})
        models = [{"purpose": "image", "requested": plan["model"], "resolved": None}]
        upstream = {"request_id": getattr(response, "_request_id", None)}
        usage = _dump(getattr(response, "usage", None))
    if len(outputs) > 16 or sum(len(o.get("text", "")) for o in outputs) > 32000:
        raise ValueError("Oversized image operation output.")
    count = sum(o["kind"] == "image" for o in outputs)
    normalized = {k: usage[k] for k in ("input_tokens", "output_tokens", "total_tokens")
                  if isinstance(usage, dict) and type(usage.get(k)) is int and usage[k] >= 0}
    return {"protocol": PROTOCOL, "status": "completed" if count == 1 else "failed",
            "outputs": outputs, "models": models, "upstream": upstream,
            "usage": {"normalized": normalized or None, "provider": usage},
            "effective_options": plan["output"], "warnings": [], "continuation": None}
