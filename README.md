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
    enable_state = false
}
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
- **Stateful Conversations** - Optional conversation persistence
- **Native Tools** - Built-in web search, image generation, code interpreter
- **Structured Output** - JSON schema-based output formatting
- **Function Calling** - Custom tool use support
- **Token Counting** - Usage tracking and management

### Tool Calling

The provider now detects OpenAI Responses API `function_call` / `tool_call`
blocks automatically, decodes JSON arguments, and returns standard
`ToolCall` objects to Amplifier. No extra configuration is required—tools
declared in your config or profiles will execute as soon as the model requests
them.

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
