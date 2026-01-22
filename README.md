# Amplifier Codex Provider Module

Codex CLI integration for Amplifier using `codex exec --json` (subscription-based).

## Purpose

Provides a Codex-backed LLM provider for Amplifier without API keys by calling the
Codex CLI in non-interactive mode and translating JSONL output into Amplifier responses.

## Contract

**Module Type:** Provider
**Mount Point:** `providers`
**Entry Point:** `amplifier_module_provider_codex:mount`

## Install

```bash
amplifier module add provider-codex --source git+https://github.com/shohei81/amplifier-module-provider-codex@main
```

## Prerequisites

- **Python 3.11+**
- **Codex CLI** installed and authenticated
- **Node.js** (for the CLI install)

```bash
npm i -g @openai/codex
codex login
```

## Supported Models

- `gpt-5.2-codex` (default)
- `gpt-5.1-codex`
- `gpt-5.1-codex-mini`
- `gpt-5.1-codex-max`
- `gpt-5-codex`
- `gpt-5-codex-mini`

## Configuration

```toml
[[providers]]
module = "provider-codex"
name = "codex"
config = {
    default_model = "gpt-5.2-codex",
    timeout = 300,
    skip_git_repo_check = true,
    profile = null,     # Optional Codex CLI profile name
    sandbox = null,     # Optional CLI sandbox mode
    full_auto = false   # Allow Codex to run commands and edit files
}
```

## Notes

- Uses Codex CLI sessions for caching when available (`~/.amplifier-codex/sessions`).
- Tool calls are emitted as `<tool_use>...</tool_use>` blocks and parsed from JSONL.
- Codex CLI runs in read-only mode by default; set `sandbox` or `full_auto` only if intended.
