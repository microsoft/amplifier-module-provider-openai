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

- `gpt-5.1-codex` - GPT-5 optimized for code (default)
- `gpt-5.1` - Latest GPT-5 model
- `gpt-5-mini` - Smaller, faster GPT-5
- `gpt-5-nano` - Smallest GPT-5 variant

## Configuration

```toml
[[providers]]
module = "provider-openai"
name = "openai"
config = {
    base_url = null,                 # Optional custom endpoint (null = OpenAI default)
    default_model = "gpt-5.1-codex",
    max_tokens = 4096,
    temperature = 0.7,
    reasoning = "low",              # Reasoning effort: minimal|low|medium|high
    reasoning_summary = "detailed",  # Reasoning verbosity: auto|concise|detailed
    truncation = "auto",            # Automatic context management (default: "auto")
    enable_state = false,
    debug = false,      # Enable standard debug events
    raw_debug = false   # Enable ultra-verbose raw API I/O logging
}
```

### Debug Configuration

**Standard Debug** (`debug: true`):

- Emits `llm:request:debug` and `llm:response:debug` events
- Contains request/response summaries with message counts, model info, usage stats
- Moderate log volume, suitable for development

**Raw Debug** (`debug: true, raw_debug: true`):

- Emits `llm:request:raw` and `llm:response:raw` events
- Contains complete, unmodified request params and response objects
- Extreme log volume, use only for deep provider integration debugging
- Captures the exact data sent to/from OpenAI API before any processing

**Example**:

```yaml
providers:
  - module: provider-openai
    config:
      debug: true # Enable debug events
      raw_debug: true # Enable raw API I/O capture
      default_model: gpt-5.1-codex
```

## Environment Variables

```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Usage

```python
# In amplifier configuration
[provider]
name = "openai"
model = "gpt-5.1-codex"
```

## Features

### Responses API Capabilities

- **Reasoning Control** - Adjust reasoning effort (minimal, low, medium, high)
- **Reasoning Summary Verbosity** - Control detail level of reasoning output (auto, concise, detailed)
- **Extended Thinking Toggle** - Enables high-effort reasoning with automatic token budgeting
- **Explicit Reasoning Preservation** - Re-inserts reasoning items (with encrypted content) into conversation for robust multi-turn reasoning
- **Automatic Context Management** - Optional truncation parameter for automatic conversation history management
- **Stateful Conversations** - Optional conversation persistence
- **Native Tools** - Built-in web search, image generation, code interpreter
- **Structured Output** - JSON schema-based output formatting
- **Function Calling** - Custom tool use support
- **Token Counting** - Usage tracking and management

### Reasoning Summary Levels

The `reasoning_summary` config controls the verbosity of reasoning blocks in the model's response:

- **`auto`** (default if not specified) - Model decides appropriate detail level
- **`concise`** - Brief reasoning summaries (faster, fewer tokens)
- **`detailed`** - Verbose reasoning output similar to Anthropic's extended thinking blocks

**Example comparison:**

```yaml
# Concise reasoning (brief summaries)
providers:
  - module: provider-openai
    config:
      reasoning: "medium"
      reasoning_summary: "concise"

# Detailed reasoning (verbose like Anthropic's thinking blocks)
providers:
  - module: provider-openai
    config:
      reasoning: "high"
      reasoning_summary: "detailed"
```

**Note:** Detailed reasoning consumes more output tokens but provides deeper insight into the model's thought process, useful for complex problem-solving and debugging.

### Tool Calling

The provider detects OpenAI Responses API `function_call` / `tool_call`
blocks automatically, decodes JSON arguments, and returns standard
`ToolCall` objects to Amplifier. No extra configuration is required—tools
declared in your config or profiles execute as soon as the model requests
them.

### Incomplete Response Auto-Continuation

The provider automatically handles incomplete responses from the OpenAI Responses API:

**The Problem**: OpenAI may return `status: "incomplete"` when generation is cut off due to:

- `max_output_tokens` limit reached
- Content filter triggered
- Other API constraints

**The Solution**: The provider automatically continues generation using `previous_response_id` until the response is complete:

1. **Transparent continuation** - Makes follow-up calls automatically (up to 5 attempts)
2. **Output accumulation** - Merges reasoning items and messages from all continuations
3. **Single response** - Returns complete ChatResponse to orchestrator
4. **Full observability** - Emits `provider:incomplete_continuation` events for each continuation

**Example flow**:

```python
# User request triggers large response
response = await provider.complete(request)

# Provider internally (if incomplete):
# 1. Initial call returns status="incomplete", reason="max_output_tokens"
# 2. Continuation 1: Uses previous_response_id, gets more output
# 3. Continuation 2: Uses previous_response_id, gets final output
# 4. Returns merged response with all content

# Orchestrator receives complete response, unaware of continuations
```

**Configuration**: Set maximum continuation attempts (default: 5):

```python
# In _constants.py
MAX_CONTINUATION_ATTEMPTS = 5  # Prevents infinite loops
```

**Observability**: Monitor via events in session logs:

```json
{
  "event": "provider:incomplete_continuation",
  "provider": "openai",
  "response_id": "resp_abc123",
  "reason": "max_output_tokens",
  "continuation_number": 1,
  "max_attempts": 5
}
```

### Reasoning State Preservation

The provider preserves reasoning state across conversation **steps** for improved multi-turn performance:

**The Problem**: Reasoning models (o3, o4, gpt-5.1) produce internal reasoning traces (rs\_\* IDs) that improve subsequent responses by ~3-5% when preserved. This is especially critical when tool calls are involved.

**Important Distinction**:

- **Turn**: A user prompt → (possibly multiple API calls) → final assistant response
- **Step**: Each individual API call within a turn (tool call loops = multiple steps per turn)
- **Reasoning items must be preserved across STEPS, not just TURNS**

**The Solution**: The provider uses **explicit reasoning re-insertion** for robust step-by-step reasoning:

1. **Requests encrypted content** - API call includes `include=["reasoning.encrypted_content"]`
2. **Stores complete reasoning state** - Both encrypted content and reasoning ID stored in `ThinkingBlock.content` field
3. **Re-inserts reasoning items** - Explicitly converts reasoning blocks back to OpenAI format in subsequent turns
4. **Maintains metadata** - Also tracks reasoning IDs in metadata for backward compatibility

**How it works** (tool call example showing step-by-step preservation):

```python
# Step 1: User asks question requiring tool
response_1 = await provider.complete(request)
# response_1.output contains:
#   - reasoning item: rs_abc123 (with encrypted_content)
#   - tool_call: get_weather(latitude=48.8566, longitude=2.3522)
#
# Provider stores ThinkingBlock with:
#   - thinking: "reasoning summary text"
#   - content: [encrypted_content, "rs_abc123"]  # Full reasoning state
#   - metadata: {"openai:reasoning_items": ["rs_abc123"], ...}

# Orchestrator executes tool, adds result to context
# (Note: This is still within the SAME TURN, just a different STEP)

# Step 2: Provider called again with tool result (SAME TURN!)
response_2 = await provider.complete(request_with_tool_result)
# Provider reconstructs reasoning item from previous step:
# {
#   "type": "reasoning",
#   "id": "rs_abc123",
#   "encrypted_content": "...",  # From ThinkingBlock.content[0]
#   "summary": [{"type": "summary_text", "text": "..."}]
# }
# OpenAI receives: [user_msg, reasoning_item, tool_call, tool_result]
# Model uses preserved reasoning from step 1 to generate final answer
```

**Key insight from OpenAI docs**: "While this is another API call, we consider this as a single turn in the conversation." Reasoning must be preserved across steps (API calls) within the same turn, especially when tools are involved.

**Benefits**:

- **More robust** - Explicit re-insertion doesn't rely on server-side state
- **Stateless compatible** - Works with `store: false` configuration
- **Better multi-turn performance** - ~5% improvement per OpenAI benchmarks
- **Critical for tool calling** - Recommended by OpenAI for reasoning models with tools
- **Follows OpenAI docs** - Implements "context += response.output" pattern

### Automatic Context Management (Truncation)

The provider supports automatic conversation history management via the `truncation` parameter:

**The Problem**: Long conversations can exceed context limits, requiring manual truncation or compaction.

**The Solution**: OpenAI's `truncation: "auto"` parameter automatically drops older messages when approaching context limits.

**Configuration**:

```yaml
providers:
  - module: provider-openai
    config:
      truncation: "auto"  # Enables automatic context management (default)
      # OR
      truncation: null    # Disables automatic truncation (manual control)
```

**How it works**:

- OpenAI automatically removes oldest messages when context limit approached
- FIFO (first-in, first-out) - most recent messages preserved
- Transparent to application - no errors or warnings
- Works with all conversation types (reasoning, tools, multi-turn)

**Trade-offs**:

- ✅ **Simplicity** - No manual context management needed
- ✅ **Reliability** - Never hits context limit errors
- ❌ **Control** - Can't specify which messages to drop
- ❌ **Predictability** - Drop timing depends on token counts

**When to use**:

- **Auto truncation** - For user-facing applications where simplicity matters
- **Manual control** - For debugging, analysis, or when specific messages must be preserved

**Default**: `truncation: "auto"` (enabled by default for ease of use)

### Metadata Keys

The provider populates `ChatResponse.metadata` with OpenAI-specific state:

| Key                         | Type        | Description                                                       |
| --------------------------- | ----------- | ----------------------------------------------------------------- |
| `openai:response_id`        | `str`       | Response ID for continuation and reasoning preservation           |
| `openai:status`             | `str`       | Response status: `"completed"` or `"incomplete"`                  |
| `openai:incomplete_reason`  | `str`       | Reason if incomplete: `"max_output_tokens"` or `"content_filter"` |
| `openai:reasoning_items`    | `list[str]` | Reasoning item IDs (rs\_\*) for state preservation                |
| `openai:continuation_count` | `int`       | Number of auto-continuations performed (if > 0)                   |

**Example metadata**:

```python
{
    "openai:response_id": "resp_05fb664e4d9dca6a016920b9b1153c819487f88da867114925",
    "openai:status": "completed",
    "openai:reasoning_items": ["rs_05fb664e4d9dca6a016920b9b1daac81949b7ea950bddef95a"],
    "openai:continuation_count": 2
}
```

**Namespacing**: All keys use `openai:` prefix to prevent collisions with other providers (per kernel philosophy).

### Graceful Error Recovery

The provider implements graceful degradation for incomplete tool call sequences:

**The Problem**: If tool results are missing from conversation history (due to context compaction bugs, parsing errors, or state corruption), the OpenAI API rejects the entire request, breaking the user's session.

**The Solution**: The provider automatically detects missing tool results and injects synthetic results that:

1. **Make the failure visible** - LLM sees `[SYSTEM ERROR: Tool result missing]` message
2. **Maintain conversation validity** - API accepts the request, session continues
3. **Enable recovery** - LLM can acknowledge the error and ask user to retry
4. **Provide observability** - Emits `provider:tool_sequence_repaired` event with details

**Example**:

```python
# Broken conversation history (missing tool result)
messages = [
    {"role": "assistant", "tool_calls": [{"id": "call_123", "function": {"name": "get_weather", ...}}]},
    # MISSING: {"role": "tool", "tool_call_id": "call_123", "content": "..."}
    {"role": "user", "content": "Thanks"}
]

# Provider injects synthetic result:
{
    "role": "tool",
    "tool_call_id": "call_123",
    "content": "[SYSTEM ERROR: Tool result missing from conversation history]\n\nTool: get_weather\n..."
}

# LLM responds: "I notice the weather tool failed. Let me try again..."
# Session continues instead of crashing
```

**Observability**: Repairs are logged as warnings and emit `provider:tool_sequence_repaired` events for monitoring.

**Philosophy**: This is **graceful degradation** following kernel philosophy - errors in other modules (context management) don't crash the provider or kill the user's session.

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
Any use of third-party trademarks or logos are subject to those third-party's policies.
