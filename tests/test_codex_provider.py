import asyncio
import json

from amplifier_core.message_models import ChatRequest
from amplifier_core.message_models import Message
from amplifier_core.message_models import ToolCallBlock

from amplifier_module_provider_codex import CodexProvider


class FakeHooks:
    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    async def emit(self, name: str, payload: dict) -> None:
        self.events.append((name, payload))


class FakeCoordinator:
    def __init__(self):
        self.hooks = FakeHooks()


class FakeStdin:
    def __init__(self):
        self.buffer = b""

    def write(self, data: bytes) -> None:
        self.buffer += data

    async def drain(self) -> None:
        return None

    def close(self) -> None:
        return None

    async def wait_closed(self) -> None:
        return None


class FakeStream:
    def __init__(self, lines: list[bytes]):
        self._lines = lines
        self._idx = 0

    async def readline(self) -> bytes:
        if self._idx >= len(self._lines):
            return b""
        line = self._lines[self._idx]
        self._idx += 1
        return line

    async def read(self) -> bytes:
        return b""


class FakeProcess:
    def __init__(self, lines: list[dict], returncode: int = 0):
        self.stdin = FakeStdin()
        encoded = [json.dumps(line).encode("utf-8") + b"\n" for line in lines]
        self.stdout = FakeStream(encoded)
        self.stderr = FakeStream([])
        self.returncode = returncode

    async def wait(self) -> int:
        return self.returncode


def _make_subprocess_stub(lines: list[dict]):
    async def _stub(*_args, **_kwargs):
        return FakeProcess(lines)

    return _stub


def test_codex_basic_response(monkeypatch):
    provider = CodexProvider(config={"skip_git_repo_check": True})

    monkeypatch.setattr("shutil.which", lambda _cmd: "/usr/bin/codex")
    lines = [
        {"type": "thread.started", "thread_id": "thread_123"},
        {
            "type": "item.completed",
            "item": {
                "type": "message",
                "content": [{"type": "output_text", "text": "Hello Codex"}],
            },
        },
        {"type": "turn.completed", "usage": {"input_tokens": 10, "output_tokens": 5}},
    ]
    monkeypatch.setattr(
        asyncio, "create_subprocess_exec", _make_subprocess_stub(lines)
    )

    request = ChatRequest(messages=[Message(role="user", content="Hi")])
    response = asyncio.run(provider.complete(request))

    assert response.text == "Hello Codex"
    assert response.metadata.get("codex:session_id") == "thread_123"
    assert response.usage.input_tokens == 10
    assert response.usage.output_tokens == 5


def test_codex_tool_call_from_item(monkeypatch):
    provider = CodexProvider(config={"skip_git_repo_check": True})

    monkeypatch.setattr("shutil.which", lambda _cmd: "/usr/bin/codex")
    lines = [
        {
            "type": "item.completed",
            "item": {
                "type": "tool_call",
                "id": "call_1",
                "name": "search",
                "arguments": {"q": "test"},
            },
        },
        {"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 1}},
    ]
    monkeypatch.setattr(
        asyncio, "create_subprocess_exec", _make_subprocess_stub(lines)
    )

    request = ChatRequest(
        messages=[Message(role="user", content="Hi")],
        tools=[{"name": "search", "description": "", "parameters": {}}],
    )
    response = asyncio.run(provider.complete(request))

    assert response.tool_calls
    assert response.tool_calls[0].name == "search"
    assert response.tool_calls[0].arguments == {"q": "test"}


def test_codex_tool_call_from_text_filters_invalid():
    provider = CodexProvider(config={})
    provider._valid_tool_names = {"allowed"}

    text = (
        "Here is a tool call:\n"
        "<tool_use>{\"tool\":\"invalid\",\"id\":\"call_1\",\"input\":{}}</tool_use>"
    )
    calls = provider._extract_tool_calls_from_text(text)

    assert calls == []
    assert provider._filtered_tool_calls
    assert provider._filtered_tool_calls[0]["name"] == "invalid"


def test_codex_repairs_missing_tool_results(monkeypatch):
    provider = CodexProvider(config={"skip_git_repo_check": True})
    fake_coordinator = FakeCoordinator()
    provider.coordinator = fake_coordinator

    monkeypatch.setattr("shutil.which", lambda _cmd: "/usr/bin/codex")
    lines = [
        {
            "type": "item.completed",
            "item": {
                "type": "message",
                "content": [{"type": "output_text", "text": "Done"}],
            },
        },
        {"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 1}},
    ]
    monkeypatch.setattr(
        asyncio, "create_subprocess_exec", _make_subprocess_stub(lines)
    )

    messages = [
        Message(
            role="assistant",
            content=[ToolCallBlock(id="call_1", name="do", input={"x": 1})],
        ),
        Message(role="user", content="continue"),
    ]
    request = ChatRequest(messages=messages)

    asyncio.run(provider.complete(request))

    repair_events = [
        e for e in fake_coordinator.hooks.events if e[0] == "provider:tool_sequence_repaired"
    ]
    assert len(repair_events) == 1
    assert repair_events[0][1]["repair_count"] == 1
