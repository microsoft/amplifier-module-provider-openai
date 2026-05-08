"""Live smoke test for PR-B: response chaining for reasoning models.

Empirically verifies that chaining ON delivers better cache utilization than
chaining OFF for reasoning workloads. Requires OPENAI_API_KEY in env.

Skip in CI; run manually before merge. Capture the output in the PR description.
Pass criteria (per spec §5.3):
  1. Turn 2 with chaining ON returns cache_read_tokens > 0.
  2. Turn 2 with chaining ON has cache_read_tokens >= chaining OFF.
  3. No errors or warnings in stderr.
"""

import asyncio
import os

from amplifier_core.message_models import ChatRequest, Message

from amplifier_module_provider_openai import OpenAIProvider

MODEL = "gpt-5.5"  # reasoning model with supports_reasoning=True


async def run_two_turn_chain(provider, prompt_a, prompt_b):
    # Turn 1
    req1 = ChatRequest(messages=[Message(role="user", content=prompt_a)])
    resp1 = await provider.complete(req1, model=MODEL)
    usage1 = resp1.usage
    print(
        f"Turn 1: id={resp1.metadata.get('openai:response_id') if resp1.metadata else 'N/A'}, "
        f"input_tokens={getattr(usage1, 'input_tokens', 'N/A')}, "
        f"cache_read={getattr(usage1, 'cache_read_tokens', 'N/A')}"
    )

    # Turn 2 — include resp1 in the message list so the metadata flows
    resp1_text = ""
    for block in resp1.content or []:
        if hasattr(block, "text") and block.text:
            resp1_text += block.text
    req2 = ChatRequest(
        messages=[
            Message(role="user", content=prompt_a),
            Message(
                role="assistant",
                content=resp1_text or "...",
                metadata=resp1.metadata or {},
            ),
            Message(role="user", content=prompt_b),
        ]
    )
    resp2 = await provider.complete(req2, model=MODEL)
    usage2 = resp2.usage
    print(
        f"Turn 2: id={resp2.metadata.get('openai:response_id') if resp2.metadata else 'N/A'}, "
        f"input_tokens={getattr(usage2, 'input_tokens', 'N/A')}, "
        f"cache_read={getattr(usage2, 'cache_read_tokens', 'N/A')}"
    )

    return resp1, resp2


async def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY not set — cannot run live smoke test")

    # Prompt large enough (>=1024 tokens) for OpenAI to create a cache entry
    big_context = "Background: " + ("foo bar baz " * 200)
    prompt_a = f"{big_context}\n\nQuestion: What is 2+2?"
    prompt_b = "And what about 3+3?"

    # --- Run 1: chaining ON (default for gpt-5.5) ---
    print("=== chaining ON (default) ===")
    provider_on = OpenAIProvider(api_key=api_key, config={})
    _, resp2_on = await run_two_turn_chain(provider_on, prompt_a, prompt_b)

    # --- Run 2: chaining OFF (ZDR opt-out) ---
    print("\n=== chaining OFF (opt-out) ===")
    provider_off = OpenAIProvider(
        api_key=api_key,
        config={"enable_response_chaining": False},
    )
    _, resp2_off = await run_two_turn_chain(provider_off, prompt_a, prompt_b)

    # --- Assertions ---
    cache_on = getattr(resp2_on.usage, "cache_read_tokens", 0) or 0
    cache_off = getattr(resp2_off.usage, "cache_read_tokens", 0) or 0

    print(
        f"\ncache_read ON  = {cache_on}\n"
        f"cache_read OFF = {cache_off}\n"
        f"delta = {cache_on - cache_off}"
    )

    assert cache_on > 0, (
        f"Turn 2 with chaining must show cache_read_tokens > 0; got {cache_on}"
    )
    assert cache_on >= cache_off, (
        "Chaining should not REDUCE cache_read_tokens vs. stateless. "
        f"ON={cache_on}, OFF={cache_off}"
    )

    print("\n✅ All assertions passed.")


if __name__ == "__main__":
    asyncio.run(main())
