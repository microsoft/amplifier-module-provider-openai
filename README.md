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

Provides access to OpenAI's GPT-6, GPT-5, and GPT-4 models as an LLM provider for Amplifier using the Responses API for enhanced capabilities.

## Contract

**Module Type:** Provider
**Mount Point:** `providers`
**Entry Point:** `amplifier_module_provider_openai:mount`

## Supported Models

- `gpt-6-astra` - GPT 6 Astra. Reports a 272,000-token input budget by default, or 922,000 with long context enabled (within its 1,050,000-token native total window). Supports a 128,000-token output limit, reasoning, vision, streaming, and native `apply_patch` and `computer` tools.
- `gpt-5.6-sol` / `gpt-5.6-terra` / `gpt-5.6-luna` - GPT-5.6 tiers (flagship / balanced / cost-efficient); alias `gpt-5.6` → `gpt-5.6-sol`. **`gpt-5.6-sol` is the default.** Adds `reasoning.effort="max"`, `reasoning.mode="pro"`, and `prompt_cache_options`. Note: gpt-5.6 bills cache-write tokens at 1.25× input (automatic on prompts >1024 tokens) and rejects `in_memory` retention (auto-dropped to 24h).
- `gpt-5.5` - Prior-generation GPT-5 model
- `gpt-5.4` - Balanced GPT-5 model
- `gpt-5-mini` - Smaller, faster GPT-5: 400,000-token total context, 272,000-token maximum input, and 128,000-token maximum output. Preflight retains its 4,096-token safety reserve; reducing the output cap does not raise the input ceiling.
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
| `reasoning_effort` | `reasoning.effort` | Session-default reasoning effort (canonical key). `"none"`/unset sends nothing. Astra accepts only `low`, `medium`, `high`, `xhigh`, or `max` when sent. | Higher effort = more reasoning tokens, slower, costlier. | ✅ |
| `enable_long_context` | **Amplifier-only** | Changes the *reported* context window (see [Long context](#long-context)). Does not map to an API param. | **≈2× on gpt-5.6/Astra when input exceeds 272K** — whole-request re-rating. | ✅ |
| `max_output_tokens` | `max_output_tokens` | Output-token budget. `null` = the model capability's max. **Config key is `max_output_tokens`; the per-call kwarg is still `max_tokens`.** | Caps output length. | |
| `reasoning` | `reasoning` | LEGACY effort alias. Use for the dict form (`{effort=..., mode="pro", context=...}`). `reasoning_effort` wins if both set. | See `reasoning_effort`. | |
| `reasoning_summary` | `reasoning.summary` | Reasoning verbosity: `auto`\|`concise`\|`detailed`. | `detailed` uses more output tokens. | |
| `truncation` | `truncation` | `null` (default) omits the field; API errors on overflow. `"auto"` drops oldest messages (busts cache). | `"auto"` lowers cache hit rate. | |
| `raw` | **Amplifier-only** | When `true`, includes the full (redacted) request payload in `llm:request` events. | — | |
| `timeout` | (client) | Per-request timeout seconds. | — | |
| `hide_dated_models` | **Amplifier-only** | Hides dated snapshot ids (`gpt-5.6-2026-07-09`) from `list_models`. | — | |
| `prompt_cache_key` | `prompt_cache_key` | Stable cache-routing identifier. **Settings-only** (no ConfigField). | Improves cache hit rate. | |
| `prompt_cache_retention` | `prompt_cache_retention` | `"24h"` \| `"in_memory"` \| `null`. Astra removes this legacy field and warns once; use `prompt_cache_options.ttl: "30m"`. gpt-5.5/5.6 reject `in_memory` (auto-dropped to 24h). **Settings-only now.** | `"24h"` stabilizes cache lifetime where supported. | |
| `prompt_cache_options` | `prompt_cache_options` | `{mode, ttl}`. **`mode: "explicit"` is dropped at mount** (see [Prompt caching](#prompt-caching)); Astra accepts only `ttl: "30m"`. | `explicit` w/ no breakpoints would disable caching (~10×). | |
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
| `extra_request_params` | **Amplifier-only (escape hatch)** | Responses API params override provider defaults; Astra's final compatibility checks still apply. Round-tripped by app-cli config tooling. | Depends on what you set. | |
| `tool_search` | `tools` (shape) | Mapping: `mode` (`off` default \| `namespaced`), optional `namespaces` table, optional `always_loaded`. See [Deferred tool loading](#deferred-tool-loading-tool_searchmode). | Non-default rebuilds the prompt cache once, and the model must *search* for a deferred tool. | |

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
`prompt_cache_retention = "24h"` (extended GPU-local KV storage where
supported), `prompt_cache_key` unset, `truncation = null` (the field is
omitted so the cached prefix is never silently rewritten on overflow).

For Astra, the legacy retention field is always omitted; use
`prompt_cache_options.ttl: "30m"` instead.

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

### `prompt_cache_options` — tool-result boundaries alongside implicit caching

For `gpt-5.6-luna`, `gpt-5.6-terra` and their hyphenated variants, the provider
adds `prompt_cache_breakpoint: {mode: explicit}` to each eligible function
result. A string output becomes an `input_text` block with identical text;
structured outputs receive the marker on their last string-valued `input_text`
block. Existing markers and non-text blocks are preserved. Instructions, roles,
reasoning and native tool outputs are not rewritten.

Implicit caching, cache keys and TTL defaults are unchanged. Configured
`mode: "explicit"` is still removed at mount with a warning, preserving `ttl`:
requests without eligible results may have no boundary, and explicit-only
caching without a boundary disables caching entirely. Per-call explicit
options remain intentional caller overrides. Boundaries permit reuse; they do
not guarantee cache hits after compaction or other prompt changes.

### `extra_request_params`

The documented escape hatch for Responses API parameters this provider does not
model (including `store`). It is a dict, **settings-only** (never a
`ConfigField`), merged into the request params **last** — after every
provider-computed key — so it overrides anything the provider set, deliberately.

**Astra exception:** final compatibility checks run after this merge. Unsupported
sampling, log-probability, reasoning-effort, and cache-TTL values fail before the
SDK call; legacy cache retention is removed.

**Luna/Terra cache exception:** after the final merge, automatic tool-result
boundaries also apply to caller-supplied `input`, using the effective `model`.
This can change a string output's wire representation to a block list while
preserving its text. Caller-owned objects and existing markers are not mutated.

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

### Function-tool strictness

Responses normalizes an omitted function-tool `strict` field to strict mode.
To keep ordinary tool schemas best-effort (including schemas with optional
properties), this provider sends `strict: false` by default. A `ToolSpec`
carrying a boolean `strict` extra preserves that explicit raw Responses API
opt-in/out; the provider does not alter schemas or invent core fields.

## Deferred tool loading (`tool_search.mode`)

**Off by default.** `tool_search.mode: off` (the default, and what you get when
the key is absent) adds no namespace or deferred-loading behavior beyond the
standard current tool conversion, including its deliberate `strict: false`
function-tool default. Native tools remain unchanged. The namespaced branch is
only reachable when the flag is set explicitly.
`tests/test_tool_search_namespaces.py` pins the default `tools` array by value
*and* by sha256 over its serialized bytes.

`tool_search.mode: namespaced` groups the tool roster into namespaces with
`defer_loading: true` and adds `{"type": "tool_search"}`, so the model sees only
namespace names and descriptions up front and pulls in the full definitions when
it needs them. Discovered tools are appended at the **end** of the context
window, which is why this shrinks the pinned head without rewriting it.

```yaml
config:
  tool_search:
    mode: namespaced
    # optional -- defaults to the shipped table
    namespaces:
      - name: files
        description: Read, write, edit, search and list files in the workspace.
        members: [apply_patch, edit_file, glob, grep, read_file, write_file]
      - name: shell
        description: Run shell commands and manage a task checklist.
        members: [bash, todo]
    always_loaded: [bash, todo]
```

Measured on `gpt-5.6-terra` against a 14-tool, 18,458-token head
(`tool_choice: "none"`, so the block cost is exact):

| mode | tool-block tokens | head saving |
|---|---:|---:|
| `off` | 8,677 | — |
| `namespaced` (default hedge) | 1,925 | **6,752 (36.6% of head)** |
| `namespaced`, `always_loaded: []` | 1,133 | 7,544 (40.9%) |

### Things to know before turning it on

- **OpenAI only, by construction.** Everything happens below the `ChatRequest`
  seam. No other provider can observe the feature existing — which matters,
  because editing the tools array is a full cache rebuild on Anthropic.
- **Turning it on costs one cold cache rebuild**, since the tools block changes.
  One-time, not recurring.
- **The model can substitute a visible tool for a deferred one.** Measured 1 in
  20 turns: asked to read a file, the model reached for always-loaded `bash`
  instead of searching for `read_file`. Deferring `bash` too removed it in 10/10
  retries, at the cost of a search round-trip on nearly every session. **This is
  the named, unpriced quality hazard that had this feature reverted once
  (`#83`); it is what `cal`'s Phase 2 exists to measure.**
- **Loading granularity is set by the MODEL's own search arguments, not by a
  server rule** (`cal`, n=13, 0 counter-examples): `{paths:[ns]}` returns the
  whole namespace (6/6), `{paths:[ns], query:term}` returns only matches (7/7).
  Which shape the model emits is prompt-driven and the provider cannot control
  it, so expected loaded tokens is a **band [2,002 , 3,006]**, not a point.
- **`web`, `browser`, `python` and `computer` are reserved** namespace names and
  return HTTP 400. The table validator refuses them at mount.
- **`tool_choice` is forced to `"auto"`** in this mode: forcing a tool the model
  has not discovered yet has no defined behaviour.
- A tool registered **mid-session** rides a developer-role `additional_tools`
  input item at the tail rather than being spliced into the cached tools block.

### Round-tripping the hosted items is not optional

The API returns `tool_search_call` / `tool_search_output` items describing which
tools it loaded. They are captured onto `ChatResponse.metadata` under
`openai:tool_search_items` and replayed verbatim into `input` on the next
request. Drop them and every discovered tool ceases to exist for the model **and**
the cache breaks forward — silently, with no error. Capture and replay are
deliberately **not** gated on `tool_search.mode`, so a resumed session cannot
lose its loaded set because a flag was off in the new process.

## Long context

`enable_long_context` (default off) controls the **reported** context window;
it does not map to an API parameter.

- The threshold is measured on **INPUT tokens only**, at **272,000**.
- The boundary is **strict**: exactly 272,000 is short-context; `> 272,000` is
  long.
- On **gpt-5.6 and gpt-6-astra**, exceeding it re-rates the **ENTIRE request** — input, output,
  cached, and cache-write tokens — at long rates. **Whole-request, not
  marginal-on-the-overage.**
- **Which models actually have the tier:** `gpt-5.6-sol` / `-terra` / `-luna`
  and `gpt-6-astra` have modelled long rates. `gpt-5.4` and variants carry a
  272K threshold but have **no long rates modelled**, so the flag only changes
  the *reported* window for them. **`gpt-5.5` has no threshold at all** — the
  flag is a no-op there.
- **What the flag does:** with it off (default), `get_info`/`list_models`
  report the 272K threshold as the context window, so unpinned sessions compact
  against the standard-priced window. With it on, they report the full measured
  input budget (900,000 for 5.6, empirically measured; 922,000 for Astra,
  reserving 128,000 output tokens from its 1,050,000-token total window).

The `enable_long_context` ConfigField is gated (`show_when`) to gpt-5.6-family
models and the exact `gpt-6-astra` ID, where the flag carries a cost consequence.

## GPT-6 Astra

`gpt-6-astra` is supported only by its documented exact model ID; this module
does not infer support or pricing for hypothetical snapshots. It sends no
default `reasoning.effort`. The configured `"none"` selector remains an
Amplifier omission sentinel; an outgoing `none`, `minimal`, or unknown effort
is rejected before the SDK call. Unsupported outgoing `temperature`, `top_p`,
`logprobs`, `top_logprobs`, and `include: ["message.output_text.logprobs"]` are likewise
rejected after `extra_request_params` has performed its final merge.

Prompt caching uses `prompt_cache_options.ttl: "30m"` (the only documented TTL).
The legacy `prompt_cache_retention` field is removed from every Astra wire
payload, including continuation calls. The existing explicit-cache-mode safety
guard remains: this provider does not create cache breakpoints, so explicit
mode is dropped rather than disabling caching.

### Context and token estimates

The native total context window is 1,050,000 tokens and maximum output is
128,000 tokens. `ModelCapabilities.context_window` is the safe input and
compaction budget: 272,000 by default to avoid long-context pricing, or
922,000 with `enable_long_context: true`. The boundary is strict: exactly
272,000 input tokens uses short pricing; 272,001 re-rates the whole request.

Rates are USD per million tokens in fresh input / cached input / cache writes /
output order. Standard rates are `$10 / $1 / $12.50 / $50`; long-context rates
are `$20 / $2 / $25 / $75`. Static Batch and Flex rates are half the
corresponding Standard rates; Fast rates are double. This provider does not
implement Batch transport.

Runtime Astra accounting uses the actual response `service_tier`: `default` is
Standard, `flex` is half, and `priority` or `fast` is double. A Fast request
may return `default` after a downgrade, which is charged at Standard instead.
Missing or unpriced tiers report no cost rather than a fabricated estimate.
Reasoning tokens are already included in output tokens; cache reads and writes
are each subtracted once from gross input. These are token-only estimates:
hosted-tool fees and regional-processing uplifts are excluded.

The Responses API supports more Astra features than this adapter orchestrates.
It does not add WebSocket transport or mid-turn steering, async tool-call
orchestration, multi-agent orchestration, or compaction orchestration.

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

#### Tool-result images

Ordinary function tools may return an ordered list of canonical text and
base64-image blocks. The provider sends these directly as `input_text` and
`input_image` items in the matching `function_call_output`; known
non-vision models retain the text and receive an explicit image-omitted notice.

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

Before each SDK dispatch, the provider also applies a local, serialized-payload
estimate. A fresh provider starts conservatively at one estimated token per UTF-8
payload byte, so an initial long request or resumed session can have less usable
capacity than a provider with matched response usage. Successful Responses API
usage calibrates a model-local byte rate; it does not retain prompts and does not
turn the bootstrap estimate into a permanent high-water limit. This is an
operational guard, not an authoritative native-token tokenizer or a guarantee of
the service's context limit.

`request_budget` returns a concrete budget while that estimate is within its
allowance. For an uncalibrated model whose serialized-byte bootstrap exceeds the
allowance, it returns `None`: the local estimate cannot establish either a fit or
an overflow. A compatible context-management loop must accept `None`, and a
producer must be released only after its consumer has accepted that result. The
provider validates the final payload, then sends that cold request for
authoritative API validation and warns once per model. The API can still reject
the actual request for context overflow; this behavior never discards protected
input to make the request fit.

The byte estimate applies only to text payloads. Typed image/file content and
computer screenshots require the native count below: encoded bytes, URLs, and
file identifiers cannot establish their model token cost. If counting is
unavailable or fails, their preflight budget is `None` and the unchanged request
is sent for API validation, with a scalar-only warning once per model. Model,
output-reservation, and serialization checks still apply. Multimodal response
usage remains in actual usage accounting but never calibrates the text byte
estimate. Data URLs or image-shaped JSON inside ordinary text, tool arguments,
or tool schemas remain subject to the text guard.

### Native Responses input counts

For a direct `OpenAIProvider` on the effective standard
`https://api.openai.com/v1` route, `request_budget` uses the SDK's
`responses.input_tokens.count` operation for a native input measurement. Its
result is reported as:

```python
measurement = {
    "kind": "provider_count",
    "source": "official.operation",
    "input_tokens": C,
}
```

`C` is input-only. It is compared with the provider's input allowance, which
already reserves the selected output cap and the safety reserve; neither is
subtracted from `C`. The counter receives the finalized countable Responses
projection: model, instructions, input/history (including native replay), tools,
tool choice, parallel-tool setting, reasoning, text format, and truncation.
Create-only response settings are not sent to the counter.

The `request_budget:provider_count` capability means that `request_budget`
returns an awaitable native decision. Consumers of that capability must await
the result and handle `None`: an individual count can fail, be malformed, or
be unprojectable after the capability was advertised. The instance capability
remains advertised in those cases.

Native counting is unavailable when the installed SDK lacks the stable helper,
its version is missing, malformed, unsupported, or pre-release, or the
provider is a subclass/custom/Azure/proxy route. Those legacy unsupported
routes retain their established synchronous estimate-based `dict` or `None`
`request_budget` result; they do not advertise
`request_budget:provider_count`.

There is no count cache. A Context/Loop measured preflight calls the counter once
for its frozen candidate. Separately, generation takes one final pre-event count
for its first SDK create/stream dispatch and refreshes that count immediately
before every physical retry. Therefore a Context/Loop flow with `N` physical
generation attempts makes `N + 1` count requests; a direct generation with `N`
attempts makes `N`. The preflight does not mutate provider state or emit completion
events. A successful final count overrides a stale serialized-byte calibration; an
over-allowance count blocks generation before its SDK create/stream request.

### Metadata Keys

The provider populates `ChatResponse.metadata` with OpenAI-specific state:

| Key | Type | Description |
| --- | --- | --- |
| `openai:response_id` | `str` | Response id (captured for support/debug correlation; never read back into a later request). |
| `openai:status` | `str` | `"completed"` or `"incomplete"`. |
| `openai:incomplete_reason` | `str` | `"max_output_tokens"` or `"content_filter"`. |
| `openai:reasoning_items` | `list[str]` | Reasoning item ids (`rs_*`) for state preservation. |
| `openai:continuation_count` | `int` | Number of auto-continuations performed (if > 0). |
| `openai:tool_search_items` | `list[dict]` | Hosted `tool_search_call` / `tool_search_output` items, replayed verbatim into `input` on the next request. Dropping them makes every discovered tool cease to exist for the model **and** breaks the cache forward. |

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

### Optional provider-owned native transport

Hosts can opt into the `native` extra and the session-local
`NativeResponsesProvider` adapter for supported Responses WebSocket steering and
explicit opaque compaction. Ordinary provider mounting remains unchanged.
See [native transport contract and limits](docs/native-responses.md).

### Continuing after a native computer screenshot failure

A failed computer result remains a non-retryable local protocol error in its current turn (`computer_result_not_image`). After a later non-ephemeral user message, the provider builds a request-only projection: the failed native call/result pair becomes bounded, explicitly untrusted textual evidence carrying the original call identity, result kind and digest. Canonical messages are unchanged. Valid screenshot pairs and unrelated tools remain intact.

This allows a user to discuss the failure without manufacturing a screenshot or replaying the prior action. It does not grant approval, resume a computer tool, clear a durable tool/provider halt, or rewrite the original result. Hosts must mark injected reminders/observations ephemeral; they are not new user input. The native transport uses the normalized view to start a new lineage once, retaining its existing refusal to move pending steering into a rewritten context. Real user instructions still require ordinary tool authority and any separate safety-halt resolution.
