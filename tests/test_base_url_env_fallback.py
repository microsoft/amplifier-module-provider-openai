"""Tests for base_url environment variable fallback.

These tests verify backward compatibility when adding OPENAI_BASE_URL
environment variable support for ollama launch integration.

Critical invariant: config["base_url"] MUST take precedence over env var.
"""

from amplifier_module_provider_openai import OpenAIProvider


class TestBackwardCompatibility:
    """Tests that prove existing behavior is preserved."""

    def test_config_base_url_used_when_provided(self, monkeypatch):
        """Config base_url must be used - this is existing user behavior."""
        # Ensure env var is NOT set (clean environment)
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

        config = {"base_url": "https://my-proxy.example.com"}
        provider = OpenAIProvider("test-key", config=config)

        assert provider.base_url == "https://my-proxy.example.com"

    def test_config_takes_precedence_over_env_var(self, monkeypatch):
        """Config MUST win when both config and env var are set.

        This is CRITICAL for backward compatibility - users with explicit
        config should not be affected by system-wide env vars.
        """
        monkeypatch.setenv("OPENAI_BASE_URL", "https://from-env.example.com")

        config = {"base_url": "https://from-config.example.com"}
        provider = OpenAIProvider("test-key", config=config)

        assert provider.base_url == "https://from-config.example.com"

    def test_none_base_url_when_neither_set(self, monkeypatch):
        """When neither config nor env var set, base_url should be None.

        This preserves default SDK behavior (uses api.openai.com).
        """
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

        provider = OpenAIProvider("test-key", config={})

        assert provider.base_url is None


class TestEnvVarFallback:
    """Tests for new environment variable fallback feature."""

    def test_env_var_used_when_config_missing(self, monkeypatch):
        """Env var should be used when config key is not present."""
        monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1")

        provider = OpenAIProvider("test-key", config={})

        assert provider.base_url == "http://localhost:11434/v1"

    def test_env_var_used_when_config_is_none(self, monkeypatch):
        """Env var should be used when config explicitly sets None."""
        monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1")

        config = {"base_url": None}
        provider = OpenAIProvider("test-key", config=config)

        assert provider.base_url == "http://localhost:11434/v1"


class TestBaseUrlConfigField:
    """Tests for base_url ConfigField declaration."""

    def test_base_url_config_field_declared(self):
        """Test that base_url ConfigField is properly declared in get_info()."""
        provider = OpenAIProvider("test-api-key", config={})
        info = provider.get_info()

        base_url_field = next(
            (f for f in info.config_fields if f.id == "base_url"),
            None,
        )

        assert base_url_field is not None, "base_url ConfigField should be declared"
        assert base_url_field.display_name == "API Base URL"
        assert base_url_field.field_type == "text"
        assert base_url_field.required is False

    def test_base_url_config_field_has_env_var(self):
        """Test that base_url ConfigField declares OPENAI_BASE_URL env var."""
        provider = OpenAIProvider("test-api-key", config={})
        info = provider.get_info()

        base_url_field = next(
            (f for f in info.config_fields if f.id == "base_url"),
            None,
        )

        assert base_url_field is not None
        assert base_url_field.env_var == "OPENAI_BASE_URL"

    def test_base_url_config_field_has_default(self):
        """Test that base_url ConfigField has default value."""
        provider = OpenAIProvider("test-api-key", config={})
        info = provider.get_info()

        base_url_field = next(
            (f for f in info.config_fields if f.id == "base_url"),
            None,
        )

        assert base_url_field is not None
        assert base_url_field.default == "https://api.openai.com/v1"
