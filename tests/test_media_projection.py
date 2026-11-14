"""Private media admission through real SDK HTTP transport, never inference."""

import base64
import copy
import hashlib
import io
import json
from types import SimpleNamespace

import httpx
import pytest
from amplifier_core.llm_errors import InvalidRequestError, LLMError
from amplifier_core.message_models import (
    ChatRequest,
    ImageBlock,
    Message,
    TextBlock,
    ToolCallBlock,
)
from openai import AsyncOpenAI
from PIL import Image

from amplifier_module_provider_openai import OpenAIProvider, _media


def pixels(image_format="PNG"):
    out = io.BytesIO()
    Image.new("RGB", (3, 2), (29, 101, 173)).save(out, format=image_format)
    return out.getvalue()


def request(role="user", image_format="PNG"):
    data = base64.b64encode(pixels(image_format)).decode()
    parts = [
        TextBlock(text="before"),
        ImageBlock(
            source={
                "type": "base64",
                "media_type": "image/png" if image_format == "PNG" else "image/jpeg",
                "data": data,
            }
        ),
        TextBlock(text="after"),
    ]
    if role == "tool":
        messages = [
            Message(role="user", content="read"),
            Message(
                role="assistant",
                content=[ToolCallBlock(id="c1", name="read_file", input={})],
            ),
            Message(role="tool", tool_call_id="c1", content=parts),
        ]
    else:
        messages = [Message(role="user", content=parts)]
    return ChatRequest(messages=messages, max_output_tokens=64)


class Cap:
    version = 1

    def __init__(self, request):
        self.request, self.valid, self.calls = request, True, []

    def validate(self, request, *, purpose, request_options):
        assert request is self.request
        self.calls.append((purpose, copy.deepcopy(request_options)))
        if not self.valid:
            raise ValueError("expired")
        return {"valid": True}


def response(model):
    return {
        "id": "resp_fixture",
        "object": "response",
        "created_at": 0,
        "model": model,
        "status": "completed",
        "output": [
            {
                "type": "message",
                "id": "msg_fixture",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "OK", "annotations": []}],
            }
        ],
        "usage": {"input_tokens": 30, "output_tokens": 1, "total_tokens": 31},
    }


def provider(
    req,
    model="gpt-6.1-sol",
    *,
    stream=False,
    handler=None,
    provider_type=OpenAIProvider,
):
    cap = Cap(req)
    calls, events = [], []

    async def emit(name, data):
        events.append((name, copy.deepcopy(data)))

    def receive(r):
        body = json.loads(r.content)
        calls.append((r.url.path, body))
        if handler:
            custom = handler(r, cap, calls)
            if custom is not None:
                return custom
        if r.url.path.endswith("/input_tokens"):
            return httpx.Response(
                200, json={"object": "response.input_tokens", "input_tokens": 30}
            )
        if r.url.path.endswith("/compact"):
            return httpx.Response(
                200,
                json={
                    "id": "cmp_fixture",
                    "object": "response.compaction",
                    "created_at": 0,
                    "output": [
                        {
                            "type": "compaction",
                            "id": "cmp_item",
                            "encrypted_content": "opaque",
                        }
                    ],
                    "usage": {
                        "input_tokens": 30,
                        "output_tokens": 2,
                        "total_tokens": 32,
                    },
                },
            )
        reply = response(model)
        if stream:
            events_wire = [
                {"type": "response.created", "response": {**reply, "output": []}},
                {"type": "response.completed", "response": reply},
            ]
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                text="".join("data: " + json.dumps(v) + "\n\n" for v in events_wire),
            )
        return httpx.Response(200, json=reply)

    client = AsyncOpenAI(
        api_key="offline",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(receive)),
    )
    p = provider_type(
        client=client,
        coordinator=SimpleNamespace(
            get_capability=lambda _: cap, hooks=SimpleNamespace(emit=emit)
        ),
        config={
            "default_model": model,
            "raw": True,
            "use_streaming": stream,
            "max_retries": 1,
            "min_retry_delay": 0,
            "max_retry_delay": 0,
        },
    )
    return p, cap, calls, events


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model,stream",
    [
        (model, stream)
        for model in sorted(_media.VISION_MODELS)
        for stream in [False, True]
        if not stream or model not in _media.NONSTREAM_MODELS
    ],
)
@pytest.mark.parametrize("role", ["user", "tool"])
@pytest.mark.parametrize("image_format", ["PNG", "JPEG"])
async def test_model_wire_count_send_original_pixels(model, role, stream, image_format):
    req = request(role, image_format)
    canonical = req.model_dump_json()
    p, cap, calls, events = provider(req, model, stream=stream)
    try:
        budget = await p.request_budget(req, context_estimate=100)
        assert budget["measurement"]["kind"] == "provider_count"
        assert budget["context_window_tokens"] == _media.MEDIA_LIMITS[model][0]
        await p.complete(req)
        assert len(calls) == 3
        assert calls[0][1]["input"] == calls[1][1]["input"] == calls[2][1]["input"]
        assert all(body["model"] == model for _, body in calls)
        final = calls[-1][1]["input"][-1]
        if role == "tool":
            assert final["call_id"] == "c1"
            parts = final["output"]
        else:
            parts = final["content"]
        assert [v["type"] for v in parts] == ["input_text", "input_image", "input_text"]
        data = parts[1]["image_url"].split(",", 1)[1]
        assert (
            hashlib.sha256(base64.b64decode(data, validate=True)).digest()
            == hashlib.sha256(pixels(image_format)).digest()
        )
        assert data not in json.dumps(events)
        assert "not retained" in json.dumps(events)
        assert req.model_dump_json() == canonical
        assert {purpose for purpose, _ in cap.calls} == {"count", "transport"}
    finally:
        await p.close()


@pytest.mark.asyncio
async def test_revoke_after_count_no_generation():
    req = request()
    p, cap, calls, _ = provider(req)
    try:
        await p.request_budget(req, context_estimate=100)
        cap.valid = False
        with pytest.raises(InvalidRequestError):
            await p.complete(req)
        assert len(calls) == 1
    finally:
        await p.close()


@pytest.mark.asyncio
async def test_changed_options_after_fit_refuses_before_second_count():
    req = request()
    p, _, calls, _ = provider(req)
    try:
        await p.request_budget(req, context_estimate=100)
        with pytest.raises(InvalidRequestError):
            await p.complete(req, reasoning_effort="low")
        assert len(calls) == 1
    finally:
        await p.close()


@pytest.mark.asyncio
async def test_unknown_transport_not_replayed():
    req = request()

    def fail(r, cap, calls):
        if not r.url.path.endswith("/input_tokens"):
            raise httpx.ReadTimeout("lost response")

    p, _, calls, _ = provider(req, handler=fail)
    try:
        with pytest.raises(LLMError) as caught:
            await p.complete(req)
        assert not caught.value.retryable
        assert sum(not path.endswith("/input_tokens") for path, _ in calls) == 1
    finally:
        await p.close()


@pytest.mark.asyncio
async def test_compaction_same_private_image_input():
    req = request("tool")
    p, _, calls, _ = provider(req)
    try:
        result = await p.compact_context(req)
        assert result["kind"] == "native"
        assert len(calls) == 2
        assert calls[0][1]["input"] == calls[1][1]["input"]
        assert calls[1][0].endswith("/compact")
    finally:
        await p.close()


@pytest.mark.asyncio
async def test_known_rejection_rechecks_revocation_before_retry():
    req = request()

    def reject(r, cap, calls):
        if not r.url.path.endswith("/input_tokens"):
            cap.valid = False
            return httpx.Response(503, json={"error": {"message": "known rejection"}})

    p, _, calls, _ = provider(req, handler=reject)
    try:
        with pytest.raises(InvalidRequestError):
            await p.complete(req)
        assert sum(not path.endswith("/input_tokens") for path, _ in calls) == 1
    finally:
        await p.close()


@pytest.mark.asyncio
async def test_request_hook_cannot_mutate_canonical_after_fit():
    req = request()
    p, _, calls, _ = provider(req)

    async def mutate(name, data):
        if name == "llm:request":
            req.messages[0].content[0].text = "changed after planning"

    p.coordinator.hooks.emit = mutate
    try:
        with pytest.raises(InvalidRequestError):
            await p.complete(req)
        # Exact read-only counting precedes llm:request; no generation may
        # observe the subsequently changed canonical object.
        assert all(path.endswith("/input_tokens") for path, _ in calls)
    finally:
        await p.close()


def test_known_wire_diagnostic_copy_does_not_change_opaque_provenance():
    encoded = base64.b64encode(pixels()).decode()
    source = {
        "input": [
            {
                "type": "computer_call_output",
                "call_id": "c1",
                "output": {
                    "type": "computer_screenshot",
                    "image_url": "data:image/png;base64," + encoded,
                },
            }
        ],
        "metadata": {"opaque": "must survive"},
        "arguments": {"image_url": "opaque"},
    }
    original = copy.deepcopy(source)
    safe = _media.diagnostic_copy(source)
    assert source == original and encoded not in json.dumps(safe)
    assert (
        safe["metadata"] == source["metadata"]
        and safe["arguments"] == source["arguments"]
    )


@pytest.mark.asyncio
async def test_account_changed_after_count_refuses_before_generation():
    req = request()
    p, _, calls, _ = provider(req)
    try:
        await p.request_budget(req, context_estimate=100)
        p.client.api_key = "changed-account"
        with pytest.raises(InvalidRequestError):
            await p.complete(req)
        assert len(calls) == 1
    finally:
        await p.close()


@pytest.mark.asyncio
async def test_nested_request_cannot_borrow_outer_private_witness():
    req = request()
    other = req.model_copy(deep=True)
    p, _, calls, _ = provider(req)
    refused = []

    async def observe(name, data):
        if name == "llm:request":
            with pytest.raises(InvalidRequestError):
                await p.complete(other)
            refused.append(True)

    p.coordinator.hooks.emit = observe
    try:
        await p.complete(req)
        assert refused == [True]
        assert sum(not path.endswith("/input_tokens") for path, _ in calls) == 1
    finally:
        await p.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["api_key", "client"])
async def test_subclass_actual_client_slot_bound_before_transport(mutation):
    class SeparateClientProvider(OpenAIProvider):
        @property
        def client(self):
            return getattr(self, "_alternate", None) or super().client

    req = request()
    p, _, calls, _ = provider(req, provider_type=SeparateClientProvider)
    p._alternate, p._client = p._client, None
    original = p.client

    async def mutate(name, data):
        if name == "llm:request":
            if mutation == "client":
                p._alternate = original.with_options()
            else:
                original.api_key = "changed-account"

    p.coordinator.hooks.emit = mutate
    try:
        with pytest.raises(InvalidRequestError):
            await p.complete(req)
        assert not calls
    finally:
        await original.close()


def test_user_only_support_does_not_authorize_tool_position():
    p = OpenAIProvider(api_key="offline")
    p.get_media_request_support = lambda **_: {"v": 1, "positions": ["user"]}
    assert not p._media_tool_output_supported("gpt-6.1-sol")


@pytest.mark.asyncio
async def test_unexpected_response_image_union_never_warns_pixels():
    import warnings

    encoded = base64.b64encode(pixels()).decode()

    def reply(r, cap, calls):
        if not r.url.path.endswith("/input_tokens"):
            body = response("gpt-6.1-sol")
            body["output"] = [
                {
                    "type": "function_call_output",
                    "call_id": "c1",
                    "output": [
                        {
                            "type": "input_image",
                            "image_url": "data:image/png;base64," + encoded,
                        }
                    ],
                }
            ]
            return httpx.Response(200, json=body)

    req = request()
    p, _, _, events = provider(req, handler=reply)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            await p.complete(req)
        assert not [
            w
            for w in caught
            if "input_image" in str(w.message) or "image/png" in str(w.message)
        ]
        assert encoded not in json.dumps(events)
    finally:
        await p.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["count", "send"])
@pytest.mark.parametrize("auth_kind", ["callback", "runtime", "workload"])
async def test_unqualified_sdk_async_auth_refuses_before_private_bytes(
    operation, auth_kind
):
    req = request()
    p, _, calls, _ = provider(req)
    invoked = []

    async def credential():
        invoked.append(True)
        req.messages[0].content[0].text = "changed inside SDK auth"
        return "different-account"

    if auth_kind == "callback":
        p.client._api_key_provider = credential
    elif auth_kind == "runtime":
        p.client._provider_runtime = SimpleNamespace()
    else:
        p.client._workload_identity_auth = SimpleNamespace()
    try:
        with pytest.raises(InvalidRequestError):
            if operation == "count":
                await p.request_budget(req, context_estimate=100)
            else:
                await p.complete(req)
        assert not calls and not invoked
    finally:
        # These synthetic auth stubs must not participate in SDK cleanup.
        p.client._provider_runtime = None
        p.client._workload_identity_auth = None
        await p.close()


@pytest.mark.asyncio
async def test_same_object_nested_options_cannot_borrow_outer_fit():
    req = request()
    p, _, calls, _ = provider(req)
    refused = []

    async def observe(name, data):
        if name == "llm:request":
            with pytest.raises(InvalidRequestError):
                await p.complete(req, reasoning_effort="low")
            refused.append(True)

    p.coordinator.hooks.emit = observe
    try:
        await p.complete(req)
        assert refused == [True]
        assert sum(not path.endswith("/input_tokens") for path, _ in calls) == 1
    finally:
        await p.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["api_key", "model"])
async def test_synchronous_estimated_fit_remembers_wire_without_faking_count(mutation):
    class EstimatedProvider(OpenAIProvider):
        def _provider_count_available(self):
            return False

    req = ChatRequest(
        messages=[Message(role="user", content="unsupported media warning text")]
    )
    p, _, calls, _ = provider(req, provider_type=EstimatedProvider)
    try:
        result = p.request_budget(req, context_estimate=100)
        assert isinstance(result, dict) and "measurement" not in result
        if mutation == "model":
            p.extra_request_params["model"] = "gpt-6-luna"
        else:
            p.client.api_key = "different-account"
        with pytest.raises(InvalidRequestError):
            await p.complete(req)
        assert not calls
    finally:
        await p.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("nested", [False, True])
async def test_native_received_error_trace_omits_exact_input_echo(nested):
    from tests.test_native_transport import make

    req = request()
    canonical = req.model_dump_json()
    encoded = base64.b64encode(pixels()).decode()
    url = "data:image/png;base64," + encoded
    p, owner = make()
    p.owner = None
    owner.messages = [m.model_dump() for m in req.messages]
    cap = Cap(req)
    p.coordinator = SimpleNamespace(get_capability=lambda _: cap)
    observed = []
    p.trace = lambda name, data: observed.append((name, copy.deepcopy(data)))
    p.trace_context = lambda data: observed.append(("context", copy.deepcopy(data)))
    error = {"message": "invalid image " + url}
    if nested:
        error["content"] = [{"type": "input_image", "image_url": url}]
    p.socket.feed({"type": "error", "error": error})
    socket = p.socket
    try:
        with pytest.raises(LLMError) as caught:
            await p.complete(req)
        assert not caught.value.retryable and len(socket.sent) == 1
        assert encoded not in json.dumps(observed)
        assert req.model_dump_json() == canonical
    finally:
        await p.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method",
    ["_prepare_url", "_build_headers", "http_auth", "resource_post", "raw_create"],
)
async def test_late_dispatch_override_refused_before_http(method):
    req = request()
    p, _, calls, _ = provider(req)
    if method == "http_auth":
        p.client._client._build_request_auth = lambda *args, **kwargs: None
    elif method == "resource_post":
        p.client.responses._post = lambda *args, **kwargs: None
    elif method == "raw_create":
        p.client.responses.with_raw_response.create = lambda *args, **kwargs: None
    else:
        setattr(p.client, method, lambda *args, **kwargs: None)
    try:
        with pytest.raises(InvalidRequestError):
            await p.complete(req)
        assert not calls
    finally:
        await p.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_key", [None, 42, lambda: "not-static"])
async def test_actual_api_key_must_be_static_string(bad_key):
    req = request()
    p, _, calls, _ = provider(req)
    p.client.api_key = bad_key
    try:
        with pytest.raises(InvalidRequestError):
            await p.complete(req)
        assert not calls
    finally:
        await p.close()


@pytest.mark.asyncio
async def test_absent_sdk_optional_compact_does_not_break_create(monkeypatch):
    req = request()
    p, _, _, _ = provider(req)
    raw = p.client.responses.with_raw_response
    del raw.compact
    methods = dict(_media.RESOURCE_DISPATCH[type(p.client.responses)])
    methods.pop("compact")
    monkeypatch.setitem(_media.RESOURCE_DISPATCH, type(p.client.responses), methods)
    try:
        assert _media.standard_dispatch(p.client)
        raw.compact = lambda **_: None
        assert not _media.standard_dispatch(p.client)
    finally:
        await p.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["create", "compact"])
async def test_wraps_interposer_cannot_borrow_raw_sdk_method_identity(method):
    from functools import wraps

    req = request()
    p, _, calls, _ = provider(req)
    raw = p.client.responses.with_raw_response
    original = getattr(raw, method)
    observed = []

    @wraps(original)
    async def interpose(*args, **kwargs):
        observed.append(kwargs)
        return await original(*args, **kwargs)

    setattr(raw, method, interpose)
    try:
        with pytest.raises(InvalidRequestError):
            await p.complete(req)
        assert not calls and not observed
    finally:
        await p.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["create", "compact"])
@pytest.mark.parametrize("spoof", ["attributes", "comparison"])
async def test_sdk_factory_wrapper_requires_actual_bound_method_closure(method, spoof):
    from openai._legacy_response import async_to_raw_response_wrapper

    req = request()
    p, _, calls, _ = provider(req)
    raw = p.client.responses.with_raw_response
    actual = getattr(raw, method).__wrapped__
    observed = []

    async def spy(*args, **kwargs):
        observed.append(kwargs)

    if spoof == "attributes":
        # A plain function can carry these attributes without being a method.
        spy.__self__, spy.__func__ = actual.__self__, actual.__func__
        replacement = async_to_raw_response_wrapper(spy)
    else:
        class EqualSpy:
            def __eq__(self, other):
                return True

            def __ne__(self, other):
                return False

            async def __call__(self, *args, **kwargs):
                observed.append(kwargs)

        replacement = async_to_raw_response_wrapper(EqualSpy())
        replacement.__wrapped__ = actual
    setattr(raw, method, replacement)
    try:
        with pytest.raises(InvalidRequestError):
            await p.complete(req)
        assert not calls and not observed
    finally:
        await p.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["gpt-5-mini", "gpt-5.4-2026-03-05", "chat-latest"])
async def test_documented_catalogue_additions_are_not_unavailable(model):
    req = request()
    p, _, _, _ = provider(req, model)
    try:
        assert p.get_media_request_support(model=model)["positions"] == ["user", "tool"]
    finally:
        await p.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["gpt-4.1-nano", "o1", "o1-pro", "o4-mini"])
async def test_documented_shutdown_is_not_current_positive_support(model):
    req = request()
    p, _, _, _ = provider(req, model)
    try:
        assert p.get_media_request_support(model=model)["positions"] == []
    finally:
        await p.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["o3-pro", "gpt-5.5-pro"])
async def test_documented_nonstream_route_refuses_before_private_count(model):
    req = request()
    p, _, calls, _ = provider(req, model, stream=True)
    try:
        with pytest.raises(InvalidRequestError, match="non.streaming"):
            await p.request_budget(req, context_estimate=100)
        assert not calls
    finally:
        await p.close()


def test_literal_catalogue_matches_retained_vendor_facts():
    from pathlib import Path

    evidence = json.loads(
        (Path(__file__).parents[1] / "contracts/openai-media-evidence.v1.json").read_text()
    )
    rows = {row["model"]: row for row in evidence["models"]}
    assert len(_media.VISION_MODELS) == 54
    assert _media.VISION_MODELS == {
        key for key, row in rows.items() if row["status"] != "shutdown"
    }
    assert _media.SHUTDOWN_MODELS == {
        key for key, row in rows.items() if row["status"] == "shutdown"
    }
    assert _media.NONSTREAM_MODELS == {
        key for key, row in rows.items() if not row["streaming"]
    }
    for key in _media.VISION_MODELS:
        row = rows[key]
        assert row["positions"] == ["user", "tool"]
        assert row["mime_types"] == ["image/png", "image/jpeg"]
        assert _media.MEDIA_LIMITS[key] == (
            row["context_window"], row["max_input_tokens"], row["max_output_tokens"]
        )
    assert not {"gpt-5.1-mini", "gpt-6-astra-law", "gpt-5.5-cyber-preview"} & _media.VISION_MODELS


@pytest.mark.asyncio
@pytest.mark.parametrize("model", sorted(_media.NONSTREAM_MODELS))
async def test_explicit_nonstream_metadata_uses_http_create_without_route_change(model):
    req = request()
    req.metadata = {"stream": False}
    p, _, calls, _ = provider(req, model)
    # Provider helper creates JSON responses; turn the preference on only.
    p.use_streaming = True
    try:
        await p.request_budget(req, context_estimate=100)
        await p.complete(req)
        assert len(calls) == 3 and all(body["model"] == model for _, body in calls)
        assert not calls[-1][1].get("stream")
    finally:
        await p.close()
