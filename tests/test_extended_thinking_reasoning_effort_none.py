"""Regression: extended_thinking must never send a literal 'none' effort
that came from the config omission sentinel.

`reasoning_effort: "none"` in provider config is the provisioning-UI
omission sentinel: `_resolve_config_reasoning_effort()` normalizes it to
`self.reasoning_effort = None` ("use the model default", i.e. don't inject a
reasoning param at all on the normal path). But `extended_thinking` always
injects a reasoning param, and before this fix it read the RAW config value
(`self.config.get("reasoning_effort", "high")`) instead of the normalized
`self.reasoning_effort` attribute -- so config `reasoning_effort: "none"` +
`extended_thinking=True` sent the literal string "none" to the API. That
breaks gpt-6-astra outright (it does not accept `effort="none"`) and
silently disables reasoning on gpt-6-sol/-luna (which accept "none" as a
real, valid, reasoning-off value) instead of the intended "use extended
thinking -> high effort".

A REQUEST-level `reasoning_effort` kwarg (including an explicit literal
"none") is a different thing entirely -- an explicit caller instruction --
and must still be sent exactly as given.
"""

from __future__ import annotations

from amplifier_core.message_models import ChatRequest, Message

from amplifier_module_provider_openai import OpenAIProvider


def _provider(model: str, **config: object) -> OpenAIProvider:
    return OpenAIProvider(
        api_key="[REDACTED:SECRET]",
        config={"default_model": model, **config},
    )


def _request() -> ChatRequest:
    return ChatRequest(messages=[Message(role="user", content="hi")])


def _assemble_reasoning(model: str, cfg: dict, **kwargs) -> dict | None:
    provider = _provider(model, **cfg)
    params, _, _ = provider._assemble_initial_responses_params(
        _request(), **kwargs
    )
    return params.get("reasoning")


# ---------------------------------------------------------------------------
# Config sentinel + extended_thinking -> falls back to high, per model family
# ---------------------------------------------------------------------------


def test_astra_config_effort_none_extended_thinking_falls_back_to_high():
    """The primary regression: gpt-6-astra rejects a literal 'none' effort
    outright, so before the fix this raised InvalidRequestError."""
    reasoning = _assemble_reasoning(
        "gpt-6-astra", {"reasoning_effort": "none"}, extended_thinking=True
    )
    assert reasoning is not None
    assert reasoning["effort"] == "high"


def test_sol_config_effort_none_extended_thinking_falls_back_to_high():
    """gpt-6-sol/-luna DO accept a literal 'none' -- but that would silently
    turn reasoning back OFF, defeating extended_thinking's whole purpose."""
    reasoning = _assemble_reasoning(
        "gpt-6-sol", {"reasoning_effort": "none"}, extended_thinking=True
    )
    assert reasoning is not None
    assert reasoning["effort"] == "high"


def test_gpt_5_6_config_effort_none_extended_thinking_falls_back_to_high():
    reasoning = _assemble_reasoning(
        "gpt-5.6-terra", {"reasoning_effort": "none"}, extended_thinking=True
    )
    assert reasoning is not None
    assert reasoning["effort"] == "high"


def test_no_config_effort_extended_thinking_still_defaults_to_high():
    """Baseline: no config effort at all -- still "high" (unchanged)."""
    reasoning = _assemble_reasoning("gpt-6-sol", {}, extended_thinking=True)
    assert reasoning is not None
    assert reasoning["effort"] == "high"


def test_config_effort_medium_extended_thinking_is_preserved():
    """A real (non-sentinel) config effort is untouched by this fix."""
    reasoning = _assemble_reasoning(
        "gpt-6-sol", {"reasoning_effort": "medium"}, extended_thinking=True
    )
    assert reasoning is not None
    assert reasoning["effort"] == "medium"


# ---------------------------------------------------------------------------
# Request-level literal "none" always wins and is sent as given
# ---------------------------------------------------------------------------


def test_request_level_literal_none_kwarg_is_sent_as_given():
    """kwargs['reasoning_effort'] == 'none' is an explicit caller
    instruction, not the config sentinel -- it must remain literal even
    with extended_thinking=True and even on a model that otherwise defaults
    to 'high'."""
    reasoning = _assemble_reasoning(
        "gpt-6-sol",
        {},
        extended_thinking=True,
        reasoning_effort="none",
    )
    assert reasoning is not None
    assert reasoning["effort"] == "none"


def test_request_level_literal_none_kwarg_wins_over_config_high():
    reasoning = _assemble_reasoning(
        "gpt-6-sol",
        {"reasoning_effort": "high"},
        extended_thinking=True,
        reasoning_effort="none",
    )
    assert reasoning is not None
    assert reasoning["effort"] == "none"
