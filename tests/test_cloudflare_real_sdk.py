"""Regression: Cloudflare detection must fire on a REAL SDK error object.

`_is_cloudflare_challenge` guards on ``error.body``. Hand-built fixtures set
``body=None`` -- but the real OpenAI SDK stores the RAW TEXT in ``error.body``
(a str, not None) when it cannot parse the body as JSON. A "body is not None"
guard therefore bailed on exactly the challenge pages this detector exists to
catch.

This builds the error via ``client._make_status_error_from_response(response)``
-- the SDK's own path -- so it fails if the body-is-None premise returns.
"""

import httpx
import openai

from amplifier_module_provider_openai import OpenAIProvider


def _sdk_error(status: int, content_type: str, body: bytes) -> openai.APIStatusError:
    client = openai.OpenAI(api_key="x")
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    response = httpx.Response(
        status, headers={"content-type": content_type}, content=body, request=request
    )
    return client._make_status_error_from_response(response)


def test_real_html_error_body_is_str_not_none():
    err = _sdk_error(403, "text/html", b"<html>Just a moment...</html>")
    assert isinstance(err.body, str)


def test_real_json_error_body_is_dict():
    err = _sdk_error(400, "application/json", b'{"error":{"message":"bad"}}')
    assert isinstance(err.body, dict)


def test_real_html_challenge_is_detected():
    err = _sdk_error(
        403, "text/html", b"<html><title>Just a moment...</title>Cloudflare</html>"
    )
    assert OpenAIProvider._is_cloudflare_challenge(err) is True


def test_real_json_error_is_not_a_challenge():
    err = _sdk_error(403, "application/json", b'{"error":{"message":"forbidden"}}')
    assert OpenAIProvider._is_cloudflare_challenge(err) is False
