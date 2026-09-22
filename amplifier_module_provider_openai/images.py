"""Optional Images API backend. No tool, filesystem, or application policy."""

from __future__ import annotations

import base64


class OpenAIImageBackend:
    """Duck-typed image backend, selected explicitly by the consuming host/tool.

    Registration describes configured wire support, never account entitlement.
    The ordinary provider credential and endpoint own this API request. Images
    do not inherit a chat model's name, reasoning effort or native transport.
    """

    def __init__(self, provider, config):
        self.provider = provider
        self.model = config.get("model")
        self.timeout = float(config.get("timeout", 180))
        if not 1 <= self.timeout <= 600:
            raise ValueError(
                "image_generation.timeout must be between 1 and 600 seconds"
            )

    def describe(self):
        configured = isinstance(self.model, str) and bool(self.model.strip())
        return {
            "provider": "openai",
            "model": self.model if configured else None,
            "configured": configured,
            "missing": [] if configured else ["image_model"],
            "operations": ["generate", "edit"],
            "outputFormats": ["png"],
            "entitlement": "unverified",
            "automaticRetries": False,
        }

    async def generate(self, *, action, prompt, images, size, quality, background):
        if not self.describe()["configured"]:
            raise ValueError("Configure image_generation.model explicitly.")
        if action not in {"generate", "edit"} or (action == "edit") != bool(images):
            raise ValueError(
                "Generation takes no image inputs; edit needs image inputs."
            )
        client = self.provider.client.with_options(timeout=self.timeout, max_retries=0)
        params = {
            "model": self.model,
            "prompt": prompt,
            "n": 1,
            "size": size,
            "quality": quality,
            "background": background,
            "output_format": "png",
        }
        if action == "edit":
            result = await client.images.edit(
                **params,
                image=[
                    (image["name"], image["data"], image["mimeType"])
                    for image in images
                ],
            )
        else:
            result = await client.images.generate(**params)
        if len(result.data or []) != 1 or not result.data[0].b64_json:
            raise ValueError("Images API did not return exactly one base64 image.")
        # Bound decoded bytes before allocating them; URLs are never fetched.
        if len(result.data[0].b64_json) > 12 * 1024 * 1024:
            raise ValueError("Images API output exceeds the bounded artifact limit.")
        return {
            "data": base64.b64decode(result.data[0].b64_json, validate=True),
            "request_id": getattr(result, "_request_id", None),
            "usage": result.usage.model_dump() if result.usage is not None else None,
        }


def register_image_backend(coordinator, provider, config):
    settings = config.get("image_generation") or {}
    if settings.get("enabled") is not True:
        return lambda: None
    identity = settings.get("id", "openai")
    if not isinstance(identity, str) or not identity or len(identity) > 100:
        raise ValueError("image_generation.id must be a nonempty backend identifier")
    backends = dict(coordinator.get_capability("image.backends") or {})
    if identity in backends:
        raise ValueError("An image backend with this id is already registered")
    backend = OpenAIImageBackend(provider, settings)
    backends[identity] = backend
    coordinator.register_capability("image.backends", backends)

    def cleanup():
        current = dict(coordinator.get_capability("image.backends") or {})
        if current.get(identity) is backend:
            del current[identity]
            coordinator.register_capability("image.backends", current)

    return cleanup
