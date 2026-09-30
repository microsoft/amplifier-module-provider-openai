"""Shared synthetic Responses-API fixtures for GPT-6 (Astra/Sol/Luna) tests.

Not a test module itself (leading underscore keeps pytest from collecting
it as one) -- just fixture builders shared by the error-contract and
streaming-contract test files so both exercise the same fake response
shape instead of drifting apart.
"""

from __future__ import annotations

from types import SimpleNamespace


def make_completed_response(
    *,
    model: str = "gpt-6-sol",
    status: str = "completed",
    service_tier: str | None = "default",
    input_tokens: int = 100,
    output_tokens: int = 10,
    text: str = "Hi",
) -> SimpleNamespace:
    """A minimal completed non-streaming Responses-API object.

    Matches the shape `_convert_to_chat_response` expects: `.output` is a
    list of message/reasoning/function_call items, `.usage` carries the
    nested `input_tokens_details`/`output_tokens_details` OpenAI uses for
    cache and reasoning token accounting.
    """
    return SimpleNamespace(
        id="resp_gpt6",
        model=model,
        status=status,
        service_tier=service_tier,
        output=[
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text=text)],
            )
        ],
        incomplete_details=None,
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_tokens_details=SimpleNamespace(cached_tokens=0, cache_write_tokens=0),
        ),
    )
