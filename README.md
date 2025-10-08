# Amplifier OpenAI Provider Module

GPT model integration for Amplifier via OpenAI API.

## Purpose

Provides access to OpenAI's GPT models (GPT-4, GPT-3.5, etc.) as an LLM provider for Amplifier.

## Contract

**Module Type:** Provider
**Mount Point:** `providers`
**Entry Point:** `amplifier_mod_provider_openai:mount`

## Supported Models

- `gpt-4-turbo` - Latest GPT-4 Turbo
- `gpt-4` - GPT-4
- `gpt-3.5-turbo` - GPT-3.5 Turbo

## Configuration

```toml
[[providers]]
module = "provider-openai"
name = "openai"
config = {
    model = "gpt-4-turbo",
    max_tokens = 4096,
    temperature = 0.7
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
model = "gpt-4-turbo"
```

## Features

- Streaming support
- Function calling (tool use)
- Vision capabilities (on supported models)
- JSON mode
- Token counting and management

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
