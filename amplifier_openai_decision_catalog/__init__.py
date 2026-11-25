"""Decision backend metadata, discoverable without importing the provider SDK."""


def catalog():
    from amplifier_decisions.config import validate

    return {
        "schema_version": 2,
        "provider_module": "provider-openai",
        "validate": validate,
        "backends": [{
            "kind": "openai-decisions",
            "label": "OpenAI Decisions",
            "default_model": "gpt-6-luna",
            "models": ["gpt-6-luna"],
            "configuration_key": "decisions",
        }],
    }
