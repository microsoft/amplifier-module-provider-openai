# Amplifier OpenAI Provider Module

GPT model integration for Amplifier via OpenAI's Responses API.

## Prerequisites

- **Python 3.11+**
- **[UV](https://github.com/astral-sh/uv)** - Fast Python package manager

### Installing UV

```bash
# macOS/Linux/WSL
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## Purpose

Provides access to OpenAI's GPT-5 and GPT-4 models as an LLM provider for Amplifier using the Responses API for enhanced capabilities.

## Contract

**Module Type:** Provider
**Mount Point:** `providers`
**Entry Point:** `amplifier_module_provider_openai:mount`

## Supported Models

- `gpt-5.6-sol` / `gpt-5.6-terra` / `gpt-5.6-luna` - GPT-5.6 tiers (flagship / balanced / cost-efficient); alias `gpt-5.6` → `gpt-5.6-sol`. **`gpt-5.6-sol` is the default.** Adds `reasoning.effort="max"`, `reasoning.mode="pro"`, and `prompt_cache_options`. Note: gpt-5.6 bills cache-write tokens at 1.25× input (automatic on prompts >1024 tokens) and rejects `in_memory` retention (auto-dropped to 24h).
- `gpt-5.5` - Prior-generation GPT-5 model
- `gpt-5.4` - Balanced GPT-5 model
- `gpt-5-mini` - Smaller, faster GPT-5
- `gpt-5-nano` - Smallest GPT-5 variant

## Configuration

The wizard collects four fields (`api_key`, `base_url`, `reasoning_effort`,
`enable_long_context`) plus the app's own model picker. Every other setting is
configured directly in `settings.yaml` / the bundle config block — see the
[settings-key reference](#settings-key-reference) below for the full list.

```toml
[[providers]]
module = "provider-openai"
name = "openai"
config = {
    default_model = "gpt-5.6-sol",
    reasoning_effort = "low",              # none|minimal|low|medium|high|xhigh|max
    max_output_tokens = null,              # null = the model's capability max
    prompt_cache_retention = "24h",        # "24h" | "in_memory" | null
    # ...see the settings-key reference table for every key
}
```

> Note: `safety_identifier` is intentionally NOT a deployment config field. It
> is a per-end-user signal (abuse tracking) and must be set per-call via
> `kwargs`.

### Settings-key reference

Every config key the provider reads. `OpenAI param` is the Responses API
parameter the key maps to, or **Amplifier-only** when it has no direct API
counterpart. `Wizard?` marks the four keys the app-cli wizard prompts for.

| Key | OpenAI param | What it does | Cost impact | Wizard? |
| --- | --- | --- | --- | --- |
| `api_key` | (auth) | OpenAI API key. Resolved from `OPENAI_API_KEY` if unset. | — | ✅ |
| `base_url` | (client) | Custom endpoint. `null` = OpenAI default. | — | ✅ |
| `default_model` | `model` | Model id used when a request doesn't pin one. | — | (picker) |
| `reasoning_effort` | `reasoning.effort` | Session-default reasoning effort (canonical key). `"none"`/unset sends nothing. | Higher effort = more reasoning tokens, slower, costlier. | ✅ |
| `enable_long_context` | **Amplifier-only** | Changes the *reported* context window (see [Long context](#long-context)). Does not map to an API param. | **≈2× on gpt-5.6 when input exceeds 272K** — whole-request re-rating. | ✅ |
| `max_output_tokens` | `max_output_tokens` | Output-token budget. `null` = the model capability's max. **Config key is `max_output_tokens`; the per-call kwarg is still `max_tokens`.** | Caps output length. | |
| `reasoning` | `reasoning` | LEGACY effort alias. Use for the dict form (`{effort=..., mode="pro", context=...}`). `reasoning_effort` wins if both set. | See `reasoning_effort`. | |
| `reasoning_summary` | `reasoning.summary` | Reasoning verbosity: `auto`\|`concise`\|`detailed`. | `detailed` uses more output tokens. | |
| `truncation` | `truncation` | `null` (default) omits the field; API errors on overflow. `"auto"` drops oldest messages (busts cache). | `"auto"` lowers cache hit rate. | |
| `raw` | **Amplifier-only** | When `true`, includes the full (redacted) request payload in `llm:request` events. | — | |
| `timeout` | (client) | Per-request timeout seconds. | — | |
| `hide_dated_models` | **Amplifier-only** | Hides dated snapshot ids (`gpt-5.6-2026-07-09`) from `list_models`. | — | |
| `prompt_cache_key` | `prompt_cache_key` | Stable cache-routing identifier. **Settings-only** (no ConfigField). | Improves cache hit rate. | |
| `prompt_cache_retention` | `prompt_cache_retention` | `"24h"` \| `"in_memory"` \| `null`. gpt-5.5/5.6 reject `in_memory` (auto-dropped to 24h). **Settings-only now.** | `"24h"` stabilizes cache lifetime. | |
| `prompt_cache_options` | `prompt_cache_options` | `{mode, ttl}`. **`mode: "explicit"` is dropped at mount** (see [Prompt caching](#prompt-caching)); `ttl` passes through. | `explicit` w/ no breakpoints would disable caching (~10×). | |
| `reasoning_context` | `reasoning.context` | `auto`\|`current_turn`\|`all_turns`. First-class key; composes with `reasoning_effort` (the legacy `reasoning` dict does not). | `current_turn` trims rendered reasoning on long loops. | |
| `safety_identifier` | `safety_identifier` | Per-end-user abuse-tracking signal. **kwargs-only in practice**; settable via config for tests. | — | |
| `text_verbosity` | `text.verbosity` | GPT-5.6 response-length control: `low`\|`medium`\|`high`. **Settings-only now** (ConfigField removed). | — | |
| `reasoning_replay_scope` | **Amplifier-only** | Bounds inline reasoning replay: `turn` (default) \| `all` \| `none`. | `"all"` grows the payload without bound (~1,200 chars/blob). | |
| `poll_interval` | (background) | Seconds between background-mode status polls. | — | |
| `background_timeout` | (background) | Timeout seconds for background (deep-research) requests. | — | |
| `priority` | **Amplifier-only** | Provider selection priority (lower = higher). | — | |
| `use_streaming` | (transport) | Chunked HTTP transport (default `true`). Not progressive UI streaming. | — | |
| `max_retries` / `min_retry_delay` / `max_retry_delay` / `retry_jitter` | (retry) | Shared retry-with-backoff configuration. | — | |
| `max_concurrent_requests` | **Amplifier-only** | Process-wide in-flight concurrency gate (default 5; 0 disables). | — | |
| `extra_request_params` | **Amplifier-only (escape hatch)** | Arbitrary Responses API params, merged LAST, user wins. Own the consequences. Round-tripped by app-cli config tooling. | Depends on what you set. | |

**Deprecated aliases** (still work, warn once, will be removed):

| Old key | Use instead |
| --- | --- |
| `max_tokens` | `max_output_tokens` |
| `filtered` | `hide_dated_models` |

**Removed keys** (each emits a targeted migration warning naming what to do instead):

| Removed key | Migration |
| --- | --- |
| `enable_response_chaining` | Removed — the provider is always stateless now (see [Conversation state](#conversation-state)). |
| `enable_state` | Removed — `store` is managed automatically (false, except background mode which requires true); use `extra_request_params` to force it. |
| `enable_reasoning_context` | Removed — `reasoning.context` is now forwarded whenever you supply it. Set the first-class key `reasoning_context = "current_turn"` (composes with `reasoning_effort`), or put it in the legacy `reasoning` dict, e.g. `reasoning = {effort = "high", context = "current_turn"}`. |
| `thinking_budget_tokens` | Removed — `extended_thinking` still forces high reasoning effort, but no longer adjusts `max_output_tokens`. Set `max_output_tokens` directly. |
| `thinking_budget_buffer` | Removed — see `thinking_budget_tokens`. |

### Unrecognized config keys

At construction, the provider warns once (with a `did you mean`-style
suggestion when a close match exists) about any config key it does not
recognize. The check is silent on every key documented above, on the
deprecated aliases and removed-but-recognized keys (which get their own
targeted warnings), and on `api_key` / `id` / `module` / `source` / `priority`
(infrastructure fields an app or kernel may place alongside a provider's
config).

**Extending the recognized set for a subclass.** A provider module that
*subclasses* `OpenAIProvider` and passes its own config straight through
(e.g. `provider-azure-openai`) can declare its own additional keys so they
don't trip this warning:

```python
from amplifier_module_provider_openai import OpenAIProvider


class MyProvider(OpenAIProvider):
    EXTRA_KNOWN_CONFIG_KEYS = frozenset({"my_custom_key", "another_key"})
```

### Reasoning Effort

The `reasoning_effort` config key (canonical — it matches the kernel's portable
`request.reasoning_effort` field) sets a session-level default reasoning effort
applied to **every** request. The legacy `reasoning` key remains a working
alias; when both are set, `reasoning_effort` wins (a warning is logged).

Precedence (highest wins):

1. `kwargs["reasoning"]` — the full dict form, per call
2. `kwargs["reasoning_effort"]` — effort string, per call
3. `request.reasoning_effort` — the kernel's portable per-request field
4. `config["reasoning_effort"]` — canonical session default (this key)
5. `config["reasoning"]` — legacy session default
6. Nothing sent — the model's own default applies

Notes:

- **`"none"` and unset both send no reasoning parameter.**
- **Values are validated at mount**, not at request time. An unrecognized
  effort, or one the default model rejects (`gpt-5.5-pro` accepts only
  `medium`, `high`, `xhigh`), raises immediately instead of surfacing as an
  HTTP 400 mid-session.
- **Non-reasoning models are skipped with a warning**, not an error.
- Use the legacy `reasoning` key when you need the dict form to also set
  `reasoning.mode` (`{effort = "high", mode = "pro"}`) or `reasoning.context`
  (GPT-5.6 persisted reasoning) — both are forwarded ungated for an explicit
  `reasoning` dict; the caller owns the consequences.
- **For `reasoning.context`, prefer the first-class `reasoning_context` key.**
  The legacy dict is outranked by `reasoning_effort`, so an operator setting
  both had their `context` silently dropped. `reasoning_context` composes with
  whichever path built the reasoning object, and an explicit `context` inside a
  caller-supplied `reasoning` dict still wins. Measured on this provider's own
  stateless manual-replay path (`store=false`, reasoning items replayed inline):
  with no `context` field the API's effective mode is `all_turns`; an explicit
  `current_turn` is honored and echoed back (t8p, gpt-5.6-terra, 2026-09-02).
  It has no effect on a request that sends no reasoning parameter at all — the
  provider logs a warning rather than inventing one.

## Conversation state

**The provider is always stateless.** Every request carries the full converted
input and `store: false`. There is no chaining flag and no chaining code path —
`previous_response_id` is never sent.

- **The one exception:** background mode (deep research) forces `store: true`
  per-request, internally, because the Responses API requires the response to
  be retrievable for polling.
- **Encrypted reasoning replay.** `include: ["reasoning.encrypted_content"]` is
  requested whenever the model will reason; reasoning items are replayed inline,
  bounded by `reasoning_replay_scope` (default `"turn"` — assistant turns since
  the last non-ephemeral user message). See
  [Reasoning state preservation](#reasoning-state-preservation).
- **ZDR posture.** With `store: false` on every non-background request and no
  `previous_response_id` anywhere, the ZDR opt-out is now the *default and only*
  behaviour — the `enable_response_chaining = false` incantation older versions
  prescribed is obsolete and unnecessary. Operators who *want* server-side
  retention must opt in explicitly via `extra_request_params = { store = true }`.

### `reasoning_replay_scope` — bounded stateless reasoning replay

The provider re-inserts prior `ThinkingBlock` state inline on every request.
Unbounded, this grows linearly with conversation length (encrypted reasoning
blobs measured ~1,200 chars each, over 50% of the payload by turn 4 in live
probing). This key bounds how far back replay reaches:

| Value | Behavior |
| --- | --- |
| `"turn"` (default) | Replay reasoning only for assistant turns since the last non-ephemeral user message — the in-flight tool loop, per OpenAI's "single turn spans multiple API calls" guidance. Flat cost, independent of conversation length. |
| `"all"` | Replay every turn's reasoning. Unbounded growth. Escape hatch. |
| `"none"` | No inline reasoning replay. |

An unrecognized value falls back to `"turn"` with a warning.

## Prompt Caching

The provider exposes OpenAI's prompt-caching hint parameters. Defaults:
`prompt_cache_retention = "24h"` (extended GPU-local KV storage on every
supported model), `prompt_cache_key` unset, `truncation = null` (the field is
omitted so the cached prefix is never silently rewritten on overflow).

See also: [OpenAI Cookbook — Prompt Caching 201](https://cookbook.openai.com/examples/prompt_caching_201).

### `prompt_cache_key` — cache-routing identifier

OpenAI shards Responses API traffic by hashing the first ~256 input tokens. A
stable `prompt_cache_key` keeps a logical conversation pinned to one machine
regardless of small prefix drift, and is the recommended cache signal as of
OpenAI's July 2025 guidance.

| Deployment shape | Recommended key |
| --- | --- |
| Single-user agent loop (typical Amplifier) | conversation/session ID |
| Multi-tenant with shared system prompt | `f"{tenant_id}:{system_prompt_version}"` |
| Low-volume single-session | leave unset; prefix-hash routing is sufficient |

### `prompt_cache_retention` — TTL hint

| Value | Meaning |
| --- | --- |
| `"24h"` | Extended GPU-local KV storage. Provider default. |
| `"in_memory"` | 5–10 min in-process cache. Rejected by gpt-5.5/5.6 (auto-dropped to `"24h"` with a warning). |
| `null` | Field omitted; OpenAI picks the per-model default. |

### `prompt_cache_options` — explicit-mode dropped at mount

`prompt_cache_options` is `{mode, ttl}`. **`mode: "explicit"` is rejected at
mount** and downgraded to implicit with a one-time warning (the `ttl` key
passes through unchanged): this provider ships no `prompt_cache_breakpoint`
mechanism anywhere, and explicit mode with zero breakpoints disables prompt
caching **entirely** — no reads, no writes — turning a ~95% cache-read workload
into 100% full-price input (~10× regression, live-probed 2026-08-28).

> Residual gap, by design: a caller passing
> `prompt_cache_options={"mode": "explicit"}` via **per-call kwargs** bypasses
> mount validation and reaches the wire. This is consistent with the provider's
> stance on explicit caller overrides — the caller owns the consequences.

### `extra_request_params`

The documented escape hatch for Responses API parameters this provider does not
model (including `store`). It is a dict, **settings-only** (never a
`ConfigField`), merged into the request params **last** — after every
provider-computed key — so it overrides anything the provider set, deliberately.

- **User wins, loudly.** Any provider-computed key it clobbers is named in a
  one-time warning per key per provider instance. You own the consequences: an
  unknown or malformed parameter surfaces as an API 400, not a provider bug.
- **Applies to every request**, including the incomplete-continuation request.
- **The documented way to force `store: true`**:
  `extra_request_params = { store = true }`.
- Round-tripped by app-cli config tooling (see app-cli #286).

```toml
config = { extra_request_params = { store = true, seed = 42 } }
```

## Long context

`enable_long_context` (default off) controls the **reported** context window;
it does not map to an API parameter.

- The threshold is measured on **INPUT tokens only**, at **272,000**.
- The boundary is **strict**: exactly 272,000 is short-context; `> 272,000` is
  long.
- On **gpt-5.6**, exceeding it re-rates the **ENTIRE request** — input, output,
  cached, and cache-write tokens — at long rates. **Whole-request, not
  marginal-on-the-overage.**
- **Which models actually have the tier:** `gpt-5.6-sol` / `-terra` / `-luna`
  are the only models with modelled long rates. `gpt-5.4` and variants carry a
  272K threshold but have **no long rates modelled**, so the flag only changes
  the *reported* window for them. **`gpt-5.5` has no threshold at all** — the
  flag is a no-op there.
- **What the flag does:** with it off (default), `get_info`/`list_models`
  report the 272K threshold as the context window, so unpinned sessions compact
  against the standard-priced window. With it on, they report the full measured
  window (900,000 for 5.6 — empirically measured, not the ~1.05M marketing
  number).

The `enable_long_context` ConfigField is gated (`show_when`) to gpt-5.6-family
models — the only models where the flag carries a cost consequence.

## Debugging (`raw`)

Set `raw: true` to include the full, redacted request payload in the
`llm:request` event this provider emits. This is the only debug toggle the
module reads; there is no separate `debug` / `raw_debug` event tier.

```yaml
providers:
  - module: provider-openai
    config:
      raw: true
      default_model: gpt-5.6-sol
```

## Environment Variables

```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Features

### Reasoning Summary Levels

`reasoning_summary` controls the verbosity of reasoning blocks:

- **`auto`** — Model decides appropriate detail level
- **`concise`** — Brief reasoning summaries (faster, fewer tokens)
- **`detailed`** — Verbose reasoning output (default here)

### Tool Calling

The provider detects OpenAI Responses API `function_call` / `tool_call` blocks
automatically, decodes JSON arguments, and returns standard `ToolCall` objects
to Amplifier. No extra configuration is required.

### Incomplete Response Auto-Continuation

When OpenAI returns `status: "incomplete"` (e.g. `max_output_tokens` reached),
the provider automatically continues generation until the response is complete
(up to `MAX_CONTINUATION_ATTEMPTS`, default 5), then returns a single merged
`ChatResponse`.

The continuation is **input-based and stateless**: the accumulated output so
far is appended to the request's own input array (as an `incomplete`-stamped
assistant message) and re-sent — there is no `previous_response_id`. Each
continuation carries the same inherited params as the primary request,
including `extra_request_params`.

Each continuation emits a `provider:incomplete_continuation` event.

> **Known limitation (pre-existing, unchanged):** `_build_continuation_input`
> carries forward only accumulated `output_text` — reasoning items and
> tool/function calls from the truncated output are not replayed into the
> continuation. The mid-`function_call` truncation case is handled earlier and
> more aggressively by the truncation-retry policy (discard + retry once at the
> model's max output budget).

### Reasoning State Preservation

The provider preserves reasoning state across conversation **steps** (each API
call within a turn, e.g. a tool loop) via **explicit, inline reasoning
re-insertion** — the only mechanism, since the provider is stateless-only.

1. **Requests encrypted content** — every reasoning-capable request includes
   `include=["reasoning.encrypted_content"]` (unconditional — live measurement
   found it cache-neutral).
2. **Stores complete reasoning state** — encrypted content and reasoning id are
   stored in `ThinkingBlock.content`.
3. **Re-inserts reasoning items** — reasoning blocks are converted back to
   OpenAI top-level `reasoning` items on subsequent requests, bounded by
   `reasoning_replay_scope`.

**`ThinkingBlock.content` encoding**: reasoning state is stored as a
single-element list containing a named dict —
`content: [{"encrypted_content": ..., "id": "rs_*", "summary": ...}]`. A named
dict survives Amplifier's transcript sanitizer (which drops `None` values)
without losing the *meaning* of what remains. The provider still reads two
legacy on-disk shapes (a 2-element positional list, and a 1-element collapsed
list detected by the `rs_*` id prefix); a legacy 1-element block whose id
cannot be paired with ciphertext is unrecoverable and is dropped with a warning.

### Context overflow

The provider is stateless: the request's `input` already carries the full local
transcript, so there is nothing a retry could shrink. A `context_length_exceeded`
400 (or the equivalent streaming error) raises `ContextLengthError`
**immediately** — no retry. Compaction is the context manager's job, driven by
its own token threshold at request-build time.

### Metadata Keys

The provider populates `ChatResponse.metadata` with OpenAI-specific state:

| Key | Type | Description |
| --- | --- | --- |
| `openai:response_id` | `str` | Response id (captured for support/debug correlation; never read back into a later request). |
| `openai:status` | `str` | `"completed"` or `"incomplete"`. |
| `openai:incomplete_reason` | `str` | `"max_output_tokens"` or `"content_filter"`. |
| `openai:reasoning_items` | `list[str]` | Reasoning item ids (`rs_*`) for state preservation. |
| `openai:continuation_count` | `int` | Number of auto-continuations performed (if > 0). |

All keys use the `openai:` prefix to prevent collisions with other providers.

### Graceful Error Recovery

If tool results are missing from conversation history (context-management bugs,
parsing errors, state corruption), the provider detects the unpaired tool calls
and injects synthetic `[SYSTEM ERROR: Tool result missing]` results so the API
accepts the request and the session continues, rather than crashing.

Repairs emit a `provider:tool_sequence_repaired` event carrying `provider`,
`repair_count`, and `repairs` (a list of `{tool_call_id, tool_name}`), with
`repair_site: "message_level"`. `repair_count` counts synthesized results and
always equals `len(repairs)`. A wire-level backstop in `_convert_messages`
provides the same protection at the request-format boundary for the full
`_PAIRED_OUTPUT_ITEM_TYPES` vocabulary (function / apply_patch / computer
outputs).

## Dependencies

- `amplifier-core>=1.0.0`
- `openai>=1.0.0`

## Contributing

> [!NOTE]
> This project is not currently accepting external contributions, but we're actively working toward opening this up. We value community input and look forward to collaborating in the future. For now, feel free to fork and experiment!

Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit [Contributor License Agreements](https://cla.opensource.microsoft.com).

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
