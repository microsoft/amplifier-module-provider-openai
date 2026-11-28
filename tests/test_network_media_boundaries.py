"""Keep media and paid image paths outside added host recovery."""
import importlib.util
from pathlib import Path

import httpx
import openai
import pytest
from amplifier_core import llm_errors as errors
from amplifier_core.message_models import ChatRequest, ImageBlock, Message

from .test_network_errors import (  # noqa: F401 - pytest fixture
    failing_provider,
    local_only_sockets,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("case", ["dns", "sdk_timeout"])
async def test_media_input_does_not_gain_host_recovery(streaming, case):
    image = ChatRequest(messages=[Message(role="user", content=[ImageBlock(source={
        "type": "base64", "media_type": "image/png", "data": "bG9jYWwtcHJpdmF0ZS1pbWFnZQ==",
    })])])
    async with failing_provider(case, streaming) as p:
        # Direct classifier coverage, not a vendor media-route qualification.
        with pytest.raises(errors.LLMError) as caught:
            await p._complete_chat_request(image)
        assert type(caught.value) is errors.LLMError
        assert caught.value.retryable is True  # Old chat-media policy, not paid generation.
        assert "bG9jYWw" not in str(caught.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["images", "responses"])
async def test_paid_image_operation_remains_one_unknown_effect(transport):
    from amplifier_module_provider_openai import OpenAIProvider

    spec = importlib.util.spec_from_file_location("image_fixtures", Path(__file__).with_name("test_image_operation.py"))
    fixtures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixtures)
    sent = []
    def fail(req):
        sent.append(req)
        raise httpx.ConnectError("local loss", request=req)

    async with openai.AsyncOpenAI(api_key="not-a-real-key", base_url="http://127.0.0.1:9/v1",
                                 http_client=httpx.AsyncClient(transport=httpx.MockTransport(fail))) as client:
        p = OpenAIProvider(api_key="not-a-real-key", client=client, config={"base_url": "http://127.0.0.1:9/v1"})
        # Synthetic eligibility only: the operation uses an explicit local SDK
        # and fake transport. Do not claim loopback is a qualified vendor route.
        p._uses_standard_openai_endpoint = lambda: True
        item = fixtures.request(transport=transport)
        plan = (await p.prepare_image_generation(item))["plan"]
        result = await p.generate_image(item, plan=plan)
        assert result["status"] == "unknown"
        with pytest.raises(ValueError):
            await p.generate_image(item, plan=plan)
        assert len(sent) == 1
