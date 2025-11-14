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

- `gpt-5-codex` - GPT-5 optimized for code (default)
- `gpt-5` - Latest GPT-5 model
- `gpt-5-mini` - Smaller, faster GPT-5
- `gpt-5-codex` - Code-optimized GPT-5
- `gpt-5-nano` - Smallest GPT-5 variant

## Configuration

```toml
[[providers]]
module = "provider-openai"
name = "openai"
config = {
    default_model = "gpt-5-codex",
    max_tokens = 4096,
    temperature = 0.7,
    reasoning = "low",
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
      debug: true      # Enable debug events
      raw_debug: true  # Enable raw API I/O capture
      default_model: gpt-5-codex
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
model = "gpt-5-codex"
```

## Features

### Responses API Capabilities

- **Reasoning Control** - Adjust reasoning effort (minimal, low, medium, high)
- **Extended Thinking Toggle** - Enables high-effort reasoning with automatic token budgeting
- **Stateful Conversations** - Optional conversation persistence
- **Native Tools** - Built-in web search, image generation, code interpreter
- **Structured Output** - JSON schema-based output formatting
- **Function Calling** - Custom tool use support
- **Token Counting** - Usage tracking and management

### Tool Calling

The provider detects OpenAI Responses API `function_call` / `tool_call`
blocks automatically, decodes JSON arguments, and returns standard
`ToolCall` objects to Amplifier. No extra configuration is required—tools
declared in your config or profiles execute as soon as the model requests
them.

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
