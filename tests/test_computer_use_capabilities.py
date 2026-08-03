"""Tests for `supports_native_computer_use` on `ModelCapabilities`.

Mirrors the style of `TestGetCapabilitiesApplyPatchDefaultTrueRule` in
test_apply_patch_integration.py, but covers the OPPOSITE distribution:
computer-use support is narrow and version+size-gated (minor >= 4 and not a
"-nano" tier), not default-True-with-named-exclusions like apply_patch.

Every value asserted here is backed by a live call against
https://api.openai.com/v1/responses with a bare `{"type": "computer"}` tool
declaration (see PR description for the full probe transcript).
"""

from __future__ import annotations


class TestGetCapabilitiesNativeComputerUseRule:
    """Direct tests on `_capabilities.get_capabilities()` for the
    `supports_native_computer_use` flag \u2014 empirically derived from live
    Responses API calls (2026-08-03).

    Philosophy: default to False (assume NOT supported); only mark True for
    the family/version/size combination we have direct empirical proof
    accepts the native tool. This is the inverse posture from
    `supports_native_apply_patch`, because the live evidence runs the
    opposite direction: most models reject `computer`, only gpt-5.4+
    non-nano variants accept it.
    """

    # --- SUPPORTED: gpt-5.4, gpt-5.5, gpt-5.6 generations, any non-nano size ---

    def test_gpt_5_4_supported(self) -> None:
        from amplifier_module_provider_openai._capabilities import get_capabilities

        assert get_capabilities("gpt-5.4").supports_native_computer_use is True

    def test_gpt_5_4_mini_supported(self) -> None:
        """Confirms size does NOT exclude mini within a supported generation
        (only nano is excluded) \u2014 live-confirmed 200 response."""
        from amplifier_module_provider_openai._capabilities import get_capabilities

        assert get_capabilities("gpt-5.4-mini").supports_native_computer_use is True

    def test_gpt_5_4_pro_supported(self) -> None:
        from amplifier_module_provider_openai._capabilities import get_capabilities

        assert get_capabilities("gpt-5.4-pro").supports_native_computer_use is True

    def test_gpt_5_4_nano_not_supported(self) -> None:
        """The load-bearing counter-example versus apply_patch: nano is
        confirmed NOT supported for computer despite minor == 4, unlike
        apply_patch where gpt-5.4-nano IS supported. Size gates this tool;
        it did not gate apply_patch."""
        from amplifier_module_provider_openai._capabilities import get_capabilities

        assert get_capabilities("gpt-5.4-nano").supports_native_computer_use is False

    def test_gpt_5_5_supported(self) -> None:
        from amplifier_module_provider_openai._capabilities import get_capabilities

        assert get_capabilities("gpt-5.5").supports_native_computer_use is True

    def test_gpt_5_5_pro_supported(self) -> None:
        from amplifier_module_provider_openai._capabilities import get_capabilities

        assert get_capabilities("gpt-5.5-pro").supports_native_computer_use is True

    def test_gpt_5_6_supported(self) -> None:
        from amplifier_module_provider_openai._capabilities import get_capabilities

        assert get_capabilities("gpt-5.6").supports_native_computer_use is True

    # --- NOT supported: earlier gpt-5 minor versions (minor < 4) ---

    def test_gpt_5_bare_not_supported(self) -> None:
        from amplifier_module_provider_openai._capabilities import get_capabilities

        assert get_capabilities("gpt-5").supports_native_computer_use is False

    def test_gpt_5_1_not_supported(self) -> None:
        from amplifier_module_provider_openai._capabilities import get_capabilities

        assert get_capabilities("gpt-5.1").supports_native_computer_use is False

    def test_gpt_5_2_not_supported(self) -> None:
        from amplifier_module_provider_openai._capabilities import get_capabilities

        assert get_capabilities("gpt-5.2").supports_native_computer_use is False

    def test_gpt_5_3_chat_latest_not_supported(self) -> None:
        """gpt-5.3-chat-latest is confirmed SUPPORTED for apply_patch (the
        named-exclusions rule flips it True there) but NOT supported for
        computer \u2014 the two flags diverge on this exact model, proving they
        must be independent fields rather than derived from one another."""
        from amplifier_module_provider_openai._capabilities import get_capabilities

        assert (
            get_capabilities("gpt-5.3-chat-latest").supports_native_computer_use
            is False
        )

    # --- NOT supported: gpt-5.0 generation "mini" family branch ---

    def test_gpt_5_mini_bare_not_supported(self) -> None:
        from amplifier_module_provider_openai._capabilities import get_capabilities

        assert get_capabilities("gpt-5-mini").supports_native_computer_use is False

    # --- NOT supported: gpt-4.x ---

    def test_gpt_4o_not_supported(self) -> None:
        from amplifier_module_provider_openai._capabilities import get_capabilities

        assert get_capabilities("gpt-4o").supports_native_computer_use is False

    def test_gpt_4_1_not_supported(self) -> None:
        from amplifier_module_provider_openai._capabilities import get_capabilities

        assert get_capabilities("gpt-4.1").supports_native_computer_use is False

    # --- NOT supported: o-series ---

    def test_o1_not_supported(self) -> None:
        from amplifier_module_provider_openai._capabilities import get_capabilities

        assert get_capabilities("o1").supports_native_computer_use is False

    def test_o3_mini_not_supported(self) -> None:
        from amplifier_module_provider_openai._capabilities import get_capabilities

        assert get_capabilities("o3-mini").supports_native_computer_use is False

    # --- NOT supported: deep-research (dataclass default, not live-probed:
    # --- the model id is retired/unreachable; deep-research shares the
    # --- narrow-tool-surface profile documented for apply_patch) ---

    def test_deep_research_not_supported(self) -> None:
        from amplifier_module_provider_openai._capabilities import get_capabilities

        assert (
            get_capabilities("o3-deep-research").supports_native_computer_use is False
        )

    # --- "unknown" bucket defaults False (opposite of apply_patch's True) ---

    def test_novel_unrecognized_model_defaults_false(self) -> None:
        """A genuinely novel/untested model id defaults False \u2014 the accepted
        trade-off for this flag: computer-use support is the exception, not
        the rule, so an unrecognized model_id should NOT be assumed to
        accept a tool that carries UI/desktop-control implications."""
        from amplifier_module_provider_openai._capabilities import get_capabilities

        caps = get_capabilities("gpt-6-hypothetical")
        assert caps.family == "unknown"
        assert caps.supports_native_computer_use is False

    def test_default_dataclass_supports_native_computer_use_is_false(self) -> None:
        """The bare dataclass default (no family-specific override at all)
        is False \u2014 opposite baseline from supports_native_apply_patch."""
        from amplifier_module_provider_openai._capabilities import ModelCapabilities

        assert ModelCapabilities(family="test").supports_native_computer_use is False

    # --- Independence from supports_native_apply_patch on the same model ---

    def test_flags_are_independent_on_gpt_5_4_nano(self) -> None:
        """gpt-5.4-nano is the clearest proof the two flags are independent:
        apply_patch is confirmed SUPPORTED, computer is confirmed NOT
        supported, on the identical model id."""
        from amplifier_module_provider_openai._capabilities import get_capabilities

        caps = get_capabilities("gpt-5.4-nano")
        assert caps.supports_native_apply_patch is True
        assert caps.supports_native_computer_use is False
