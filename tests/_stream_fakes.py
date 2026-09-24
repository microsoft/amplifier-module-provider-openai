"""Shared fake Responses-API stream doubles, used by multiple test modules.

These reproduce the minimal on-the-wire shape needed to exercise
`OpenAIProvider`'s streaming SSE-event handling without a real network call:
an async context manager wrapping an async iterator of SSE-like events, plus
`get_final_response()` (called when the stream is exhausted without ever
having produced a terminal `response.completed`/`response.failed` handled
inline).

Kept in one place because getting a copy wrong silently stops a real
regression from failing -- see `test_gpt_6_astra.py` and
`test_misalignment_policy_stream.py` for representative use.
"""

from __future__ import annotations

from types import SimpleNamespace


class StreamContext:
    """Async context manager wrapping a fake stream, mirroring the SDK's
    `client.responses.stream(...)` return value."""

    def __init__(self, stream: object) -> None:
        self._stream = stream

    async def __aenter__(self) -> object:
        return self._stream

    async def __aexit__(self, *args: object) -> None:
        return None


class TerminalFailedStream:
    """A stream whose only terminal event is `response.failed`, with NO
    `response.completed` event -- exactly what a mid-stream terminal failure
    (e.g. a misalignment-monitoring policy stop) looks like on the wire."""

    def __init__(self, failed_response: SimpleNamespace) -> None:
        self._failed_response = failed_response
        self._sent = False
        self._response = SimpleNamespace(headers={})

    def __aiter__(self) -> TerminalFailedStream:
        return self

    async def __anext__(self) -> SimpleNamespace:
        if self._sent:
            raise StopAsyncIteration
        self._sent = True
        return SimpleNamespace(type="response.failed", response=self._failed_response)

    async def get_final_response(self) -> None:
        raise RuntimeError("Didn't receive a `response.completed` event.")


class CompletedStream:
    """A stream that succeeds normally: a single `response.completed` event."""

    def __init__(self, response: SimpleNamespace) -> None:
        self._response = SimpleNamespace(headers={})
        self._final_response = response
        self._sent = False

    def __aiter__(self) -> CompletedStream:
        return self

    async def __anext__(self) -> SimpleNamespace:
        if self._sent:
            raise StopAsyncIteration
        self._sent = True
        return SimpleNamespace(type="response.completed", response=self._final_response)

    async def get_final_response(self) -> SimpleNamespace:
        return self._final_response
