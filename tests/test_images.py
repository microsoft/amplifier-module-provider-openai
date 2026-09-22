import base64
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from amplifier_module_provider_openai.images import (
    OpenAIImageBackend,
    register_image_backend,
)


def fixture(config=None):
    result = SimpleNamespace(
        data=[SimpleNamespace(b64_json=base64.b64encode(b"png bytes").decode())],
        usage=None,
        _request_id="req-image",
    )
    client = SimpleNamespace(
        images=SimpleNamespace(
            generate=AsyncMock(return_value=result), edit=AsyncMock(return_value=result)
        )
    )
    provider = SimpleNamespace(client=Mock())
    provider.client.with_options.return_value = client
    return (
        OpenAIImageBackend(provider, config or {"model": "configured-image-model"}),
        provider,
        client,
    )


@pytest.mark.asyncio
async def test_images_uses_explicit_model_one_image_no_retries_and_real_edit_bytes():
    backend, provider, client = fixture()
    args = {
        "prompt": "make one",
        "size": "1024x1024",
        "quality": "low",
        "background": "opaque",
    }
    result = await backend.generate(action="generate", images=[], **args)
    assert result["data"] == b"png bytes"
    assert result["request_id"] == "req-image" and result["usage"] is None
    provider.client.with_options.assert_called_once_with(timeout=180, max_retries=0)
    client.images.generate.assert_awaited_once_with(
        model="configured-image-model", n=1, output_format="png", **args
    )
    await backend.generate(
        action="edit",
        images=[{"name": "source.png", "data": b"source", "mimeType": "image/png"}],
        **args,
    )
    client.images.edit.assert_awaited_once_with(
        model="configured-image-model",
        n=1,
        output_format="png",
        image=[("source.png", b"source", "image/png")],
        **args,
    )


@pytest.mark.asyncio
async def test_missing_model_and_url_only_result_are_not_capability_success():
    backend, _, client = fixture({"enabled": True})
    assert backend.describe()["configured"] is False
    args = {
        "action": "generate",
        "prompt": "make one",
        "images": [],
        "size": "1024x1024",
        "quality": "low",
        "background": "opaque",
    }
    with pytest.raises(ValueError, match="explicitly"):
        await backend.generate(**args)
    client.images.generate.assert_not_called()
    backend.model = "configured-image-model"
    client.images.generate.return_value.data[0].b64_json = None
    with pytest.raises(ValueError, match="base64"):
        await backend.generate(**args)


def test_registration_is_opt_in_explicit_and_cleanup_does_not_remove_other_backend():
    capabilities = {}
    coordinator = SimpleNamespace(
        get_capability=capabilities.get, register_capability=capabilities.__setitem__
    )
    provider = SimpleNamespace()
    register_image_backend(coordinator, provider, {})()
    assert not capabilities
    cleanup = register_image_backend(
        coordinator,
        provider,
        {"image_generation": {"enabled": True, "id": "chosen", "model": "configured"}},
    )
    assert (
        capabilities["image.backends"]["chosen"].describe()["entitlement"]
        == "unverified"
    )
    with pytest.raises(ValueError, match="already"):
        register_image_backend(
            coordinator,
            provider,
            {"image_generation": {"enabled": True, "id": "chosen"}},
        )
    capabilities["image.backends"]["other"] = object()
    cleanup()
    assert set(capabilities["image.backends"]) == {"other"}
