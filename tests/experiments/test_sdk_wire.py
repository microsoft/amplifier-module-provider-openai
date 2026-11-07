"""Real installed SDK, explicit MockTransport: never contacts an API endpoint."""

import importlib.util
import json
import unittest

from examples.openai_advanced.hosted_agents import BETA, ModelEvidence, build_request
from examples.openai_advanced.programmatic import read_only_tools

HAS_SDK = importlib.util.find_spec("openai") is not None
HAS_HTTPX = importlib.util.find_spec("httpx") is not None


@unittest.skipUnless(HAS_SDK and HAS_HTTPX, "SDK/httpx absent; local compatibility unknown")
class SDKWireTests(unittest.TestCase):
    def client(self, output):
        import httpx
        from openai import OpenAI

        self.requests = []

        def handle(request):
            self.requests.append(request)
            return httpx.Response(
                200,
                json={
                    "id": "resp_explicit_mock", "object": "response", "created_at": 0,
                    "status": "completed", "model": "explicit-mock-model", "output": output,
                },
            )

        # No configured credentials, network sockets, env-based provider lookup,
        # raw transport fallback or production client are used here.
        return OpenAI(
            api_key="offline-placeholder-not-a-secret",
            base_url="https://offline.invalid/v1",
            max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(handle)),
        )

    def test_ptc_metadata_and_order_survive_real_sdk_mock_wire(self):
        output = [
            {"type": "reasoning", "id": "r1", "summary": [],
             "encrypted_content": "opaque_reasoning"},
            {"type": "program", "id": "p1", "call_id": "call_p",
             "code": "text('mock')", "fingerprint": "opaque_fingerprint"},
            {"type": "function_call", "id": "f1", "call_id": "call_f",
             "name": "get_inventory", "arguments": '{"sku":"sku_123"}',
             "caller": {"type": "program", "caller_id": "call_p"}},
        ]
        with self.client(output) as client:
            response = client.responses.create(
                model="explicit-mock-model", input=[], tools=read_only_tools(), store=False,
            )
        wire = json.loads(self.requests[0].content)
        self.assertEqual(wire["tools"], read_only_tools())
        items = [i.model_dump(exclude_none=True) for i in response.output]
        for expected, item in zip(output, items, strict=True):
            for field, value in expected.items():
                self.assertEqual(item[field], value)
        self.assertEqual(self.requests[0].url.host, "offline.invalid")

    def test_beta_negotiation_and_attribution_survive_real_sdk_mock_wire(self):
        output = [
            {
                "type": "multi_agent_call", "id": "mac_123", "call_id": "call_spawn",
                "action": "spawn_agent", "arguments": '{"task_name":"reviewer"}',
                "agent": {"agent_name": "/root"},
            },
            {
                "type": "agent_message", "id": "amsg_123",
                "author": "/root/reviewer", "recipient": "/root",
                "agent": {"agent_name": "/root"},
                "content": [{"type": "encrypted_content", "encrypted_content": "opaque"}],
            },
            {
                "type": "message", "id": "m1", "role": "assistant", "status": "completed",
                "phase": "final_answer", "agent": {"agent_name": "/root"},
                "content": [{"type": "output_text", "text": "mock final",
                             "annotations": [], "logprobs": []}],
            },
        ]
        request, _ = build_request(
            ModelEvidence("explicit-mock-model", True, "explicit mock capability"),
            [], transport="http",
        )
        with self.client(output) as client:
            if not callable(getattr(getattr(client.beta, "responses", None), "create", None)):
                self.skipTest("Installed SDK lacks beta.responses.create; no raw fallback")
            response = client.beta.responses.create(**request)
        self.assertEqual(self.requests[0].headers["OpenAI-Beta"], BETA)
        wire = json.loads(self.requests[0].content)
        self.assertEqual(wire["multi_agent"], {"enabled": True, "max_concurrent_subagents": 3})
        # betas is SDK negotiation, not a fabricated request-body field.
        self.assertNotIn("betas", wire)
        items = [i.model_dump(exclude_none=True) for i in response.output]
        for expected, item in zip(output, items, strict=True):
            for field, value in expected.items():
                self.assertEqual(item[field], value)


if __name__ == "__main__":
    unittest.main()