"""Real SDK terminal evidence must not mask the original validation refusal.

The local error-accounting fixture exercises the same compute seam used by
integrated terminal billing. Parent integration tests own that complete loop.
All transport is offline MockTransport, including streaming.
"""

import json

import httpx
import pytest
from amplifier_core.llm_errors import InvalidRequestError
from amplifier_core.message_models import ChatRequest, Message, ToolSpec
from openai import AsyncOpenAI

from amplifier_module_provider_openai import OpenAIProvider


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("model_fields", [
    {}, {"model": None}, {"model": False}, {"model": True}, {"model": 0},
    {"model": 55}, {"model": 1.5}, {"model": []}, {"model": ["gpt-5.5"]},
    {"model": {}}, {"model": {"id": "gpt-5.5"}},
])
async def test_invalid_sdk_terminal_keeps_refusal_when_error_accounting_runs(
    streaming, model_fields,
):
    payload = {
        "id": "resp_invalid_model", "object": "response", "created_at": 1,
        "status": "completed", "parallel_tool_calls": True,
        "tool_choice": "auto", "tools": [{"type": "computer"}],
        "output": [{
            "type": "computer_call", "id": "cu_fixture", "call_id": "call_fixture",
            "status": "completed", "actions": [{"type": "screenshot"}],
        }],
        "usage": {
            "input_tokens": 10, "output_tokens": 2, "total_tokens": 12,
            "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
        **model_fields,
    }
    sent = []

    def handler(request):
        body = json.loads(request.content)
        sent.append(body)
        if body.get("stream"):
            events = "".join(
                "data: " + json.dumps({
                    "type": kind, "sequence_number": index, "response": payload,
                }) + "\n\n"
                for index, kind in enumerate(("response.created", "response.completed"))
            )
            return httpx.Response(
                200, text=events, headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(200, json=payload)

    async with AsyncOpenAI(
        api_key="offline", max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    ) as client:
        provider = OpenAIProvider(client=client, config={"max_retries": 0})
        declaration = ToolSpec(
            name="desk", description="Observe only", type="function",
            parameters={
                "type": "object", "required": ["action"],
                "properties": {"action": {"type": "string"}},
                "additionalProperties": False,
            },
            computer_use={
                "version": 1, "operations": ["screenshot"],
                "image_space": {"width": 1280, "height": 720, "unit": "pixel"},
                "unavailable_operations": {},
            },
        )
        params, state, _ = provider._assemble_initial_responses_params(ChatRequest(
            model="gpt-5.5", messages=[Message(role="user", content="Observe")],
            tools=[declaration],
        ))
        if streaming:
            async with client.responses.stream(**params) as stream:
                async for _event in stream:
                    pass
                response = await stream.get_final_response()
        else:
            response = await client.responses.create(**params)

        original_error = None
        cost = "not_accounted"
        with pytest.raises(InvalidRequestError) as caught:
            try:
                provider._convert_to_chat_response(
                    response, computer_context=state["neutral_computer_context"],
                )
            except InvalidRequestError as error:
                original_error = error
                cost = provider._compute_attempt_cost(response)
                raise
        assert caught.value is original_error
        assert caught.value.code == "computer_use_response_invalid"
        assert caught.value.retryable is False
        assert cost is None
        assert len(sent) == 1  # Validation/accounting does not retry generation.
        assert sent[0]["model"] == "gpt-5.5"