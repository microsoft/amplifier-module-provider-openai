## Responses API Coding Guide

This guide is a practical reference for formatting a `/v1/responses` request that captures instructions, reasoning preferences, and tool definitions, and for validating the streamed results and token usage coming back from the API. Use it to craft your own agent requests or to sanity-check what a client should send.

### 1. Building the Request Envelope

A well-formed Responses API request typically includes:

- `model` – the model slug (e.g., `gpt-5.1-codex`, `o4-mini`).
- `instructions` – global guidance (AGENTS.md, developer instructions, etc.).
- `input` – the conversation history, expressed as an array of message objects.
- `tools` – a list of available tool definitions, each with JSON Schema metadata.
- `tool_choice` – usually `"auto"`, unless you must force a specific tool.
- `parallel_tool_calls` – `true` when the model and workflow support concurrent tool invocations.
- `reasoning` & `include` – opt into reasoning summaries and encrypted traces.
- `stream`, `store`, `prompt_cache_key` – delivery and caching controls.

Example request assembled from `temp.json`, simplified for readability:

```json
{
  "model": "gpt-5.1-codex",
  "instructions": "# AGENTS.md instructions...\n<INSTRUCTIONS> ...",
  "input": [
    {
      "role": "system",
      "content": [{ "text": "# AGENTS.md instructions..." }]
    },
    {
      "role": "user",
      "content": [
        {
          "text": "Can you figure out where the calls to the OpenAI API are made?"
        }
      ]
    }
  ],
  "tools": [
    {
      "type": "function",
      "name": "shell",
      "description": "Execute a shell command",
      "parameters": {
        "type": "object",
        "properties": {
          "command": { "type": "string" }
        },
        "required": ["command"]
      }
    }
    // ... other tools (apply_patch, MCP tools, etc.)
  ],
  "tool_choice": "auto",
  "parallel_tool_calls": true,
  "reasoning": {
    "effort": "medium",
    "summary": "concise"
  },
  "include": ["reasoning.encrypted_content"],
  "store": false,
  "stream": true,
  "prompt_cache_key": "conv_e564...",
  "prompt_metadata": null,
  "text": null
}
```

**Key tips:**

- Keep `instructions` concise but comprehensive; the Responses API treats it as the system prompt for every turn.
- `input` should include every prior tool call scaffold so the model can continue seamlessly.
- If you require outputs in a structured schema, include it in the `tools` array or pass an `output_schema` field (not shown above).

### 2. Tool Definitions and Calls

- Each tool entry declares a name, description, and JSON schema for the arguments.
- The model emits tool calls via streamed events (`response.output_item.*`), mirroring the schema you provide.
- Use `parallel_tool_calls: true` only when your runtime can execute multiple tools concurrently; otherwise set it to `false`.
- To force a particular tool, set `tool_choice` to an object like `{ "type": "tool", "name": "apply_patch" }`.

### 3. Reasoning Options

- The `reasoning` object lets you hint at reasoning effort levels (`minimal`, `low`, `medium`, `high`) and summary verbosity (`concise`, `detailed`, etc.).
- Adding `"reasoning.encrypted_content"` to `include` instructs the API to stream encrypted reasoning deltas (`response.reasoning_*` SSE events).
- Reasoning output arrives alongside normal assistant text; decrypt it with the utility provided in the Responses API docs if you need the raw traces.

### 4. Streaming & Delivery Settings

- `stream: true` enables server-sent events so you can react to tool calls and partial text in real time. If you only need the final response, set it to `false`.
- `store` controls whether the API persists the response for retrieval; some providers (e.g., Azure) require `true`, while OpenAI defaults to `false`.
- `prompt_cache_key` is an optional identifier you can set to reuse cached context (handy for incremental runs with the same session history).

### 5. Token Usage and Context Management

- Each streamed session ends with a `response.completed` event that contains a `usage` object: `{ input_tokens, output_tokens, total_tokens, ... }`. Use it to update your own accounting or rate-limit dashboards.
- Track `total_tokens` across turns; if it approaches the model’s context window, summarize older history (compact) before issuing the next request. A simple strategy is:
  1. Define a compaction threshold (e.g., 90% of context size).
  2. When a turn exceeds it, run a summarization pass that collapses the oldest segments into a short narrative, then rebuild the next request’s `input` using that summary plus the newest items.
  3. If summarizing still doesn’t free enough room, prompt the user to start a fresh session rather than risking `context_length_exceeded`.

### 6. Putting It All Together

When designing your own agent flow with the Responses API:

1. Collect all system/developer instructions into `instructions`.
2. Serialize the entire conversation so far into `input`, including tool-call scaffolding.
3. Declare every tool the agent may call in `tools` and set `tool_choice` appropriately.
4. Configure `reasoning`, `include`, `parallel_tool_calls`, `store`, and `stream` based on your runtime’s needs.
5. Monitor token usage via `response.completed` events and trigger compaction before you hit the context ceiling.
6. Use a lightweight proxy or middleware to capture raw payloads when debugging serialization issues.

With those pieces in place, you can produce stable, reproducible Responses API requests that support rich reasoning and automated tool orchestration—independent of any specific client implementation.
