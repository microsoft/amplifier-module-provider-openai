"""Offline WebSocket multiplex experiment; NOT a production/native transport.

Protocol reference supplied for this experiment (fetched 2026-11-06):
https://developers.openai.com/api/docs/guides/websocket-mode.md

One mock connection carries independent owned streams, each with at most one
active generation. This does not change the actual native one-generation owner.
Models are opaque caller input: capability/availability is UNKNOWN here.

Only response.create is emitted, with store=False. Budgets bound local request
admissions, NOT provider tokens, money, or remote execution. Stop fences local
admission; it cannot cancel remote effects. Unknown outcomes require the owning
caller to reconcile outside this experiment; there is no reconnect/retry API.
Workdirs are identity boundaries only: no directories or tools are opened.
Custody/provenance are caller-supplied experiment data, not verified host grants
or an OS security boundary. The trusted connection owner drives pump/receive.
"""

from __future__ import annotations

import json
import re
import time
from collections import deque
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class Rejected(ValueError):
    """Admission failed without emitting a frame."""


@dataclass(frozen=True)
class Custody:
    owner: str
    token: str


@dataclass
class Request:
    request_id: str
    frame: dict[str, Any]
    custody: Custody
    provenance: tuple[str, ...]
    workdir: str
    fork_source: str | None = None
    state: str = "queued"
    response_id: str | None = None
    reason: str | None = None
    events: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Stream:
    custody: Custody
    provenance: tuple[str, ...]
    workdir: str
    request_limit: int
    admitted: int = 0
    stopped: bool = False
    revoked: bool = False
    reconciliation_required: bool = False
    active: str | None = None
    queue: deque[str] = field(default_factory=deque)
    requests: dict[str, Request] = field(default_factory=dict)
    errors: list[dict[str, Any]] = field(default_factory=list)
    # Forks retain their source until the *target* emits response.in_progress.
    fork_waiters: set[tuple[str, str]] = field(default_factory=set)


class MockSocket:
    """JSON round-tripping, single-threaded socket mock; never opens a socket."""

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self.incoming: deque[dict[str, Any] | None] = deque()
        self.fail_after_send = False

    def send(self, frame: dict[str, Any]) -> None:
        self.sent.append(json.loads(json.dumps(frame)))
        if self.fail_after_send:
            self.fail_after_send = False
            raise OSError("mock send acknowledgement lost")

    def feed(self, *events: dict[str, Any] | None) -> None:
        """None represents socket closure, not an empty receive buffer."""
        self.incoming.extend(deepcopy(events))

    def recv(self) -> dict[str, Any] | None:
        return self.incoming.popleft()


class Multiplex:
    """Explicit admission -> pump -> receive reducer, with no effect replay.

    All API calls are serialized by the caller; this is not thread-safe.
    Snapshot methods return copies so callers cannot mutate custody or queues.
    A reserved default stream counts toward the 32 distinct stream identities.
    Terminal/local-error frames settle only their stream. Connection failures
    invalidate every cached response and leave all sent active requests unknown.
    """

    def __init__(
        self, socket: MockSocket, *, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.socket = socket
        self.connected = True
        self.connection_reason: str | None = None
        self._clock = clock
        self._opened = clock()
        self._streams: dict[str, Stream] = {}
        self._seen = {"default"}
        self._cache: dict[str, str] = {}
        self._response_owners: dict[str, tuple[str, str]] = {}
        self._unowned_errors: list[dict[str, Any]] = []

    @staticmethod
    def _validate_id(stream_id: str) -> None:
        if not isinstance(stream_id, str) or not re.fullmatch(
            r"[A-Za-z0-9_.-]{1,256}", stream_id
        ):
            raise Rejected("invalid stream_id")

    def _live(self) -> None:
        if self.connected and self._clock() - self._opened >= 3600:
            self.disconnect("60-minute connection lifetime reached")
        if not self.connected:
            raise Rejected("connection closed; owner reconciliation required")

    def register(
        self,
        stream_id: str,
        custody: Custody,
        *,
        workdir: str,
        provenance: tuple[str, ...],
        request_limit: int,
    ) -> None:
        self._live()
        self._validate_id(stream_id)
        if stream_id in self._streams:
            raise Rejected("stream already owned")
        if stream_id not in self._seen and len(self._seen) >= 32:
            raise Rejected("32 distinct streams including default exhausted")
        if (
            not isinstance(custody, Custody)
            or not isinstance(custody.owner, str)
            or not custody.owner
            or not isinstance(custody.token, str)
            or not custody.token
            or not isinstance(provenance, tuple)
            or not provenance
            or any(not isinstance(item, str) or not item for item in provenance)
        ):
            raise Rejected("explicit custody and commission provenance required")
        if type(request_limit) is not int or request_limit < 1:
            raise Rejected("request_limit must be a positive integer")
        path = Path(workdir)
        if not path.is_absolute():
            raise Rejected("workdir must be absolute")
        path = path.resolve()
        for peer in self._streams.values():
            other = Path(peer.workdir)
            if path == other or path in other.parents or other in path.parents:
                raise Rejected("owned workdirs must not overlap")
        self._streams[stream_id] = Stream(custody, provenance, str(path), request_limit)
        self._seen.add(stream_id)

    def _owned(self, stream_id: str, custody: Custody) -> Stream:
        self._validate_id(stream_id)
        stream = self._streams.get(stream_id)
        if stream is None or stream.custody != custody or stream.revoked:
            raise Rejected("stream custody mismatch or revoked")
        return stream

    def submit(
        self,
        stream_id: str,
        custody: Custody,
        *,
        request_id: str,
        model: str,
        input: list[Any],
        previous_response_id: str | None = None,
    ) -> None:
        """Queue a new request; optional continuation must be this lane's cache."""
        self._enqueue(
            stream_id, custody, request_id, model, input, previous_response_id
        )

    def fork(
        self,
        source_id: str,
        source_custody: Custody,
        target_id: str,
        target_custody: Custody,
        *,
        previous_response_id: str,
        request_id: str,
        model: str,
        input: list[Any],
    ) -> None:
        """Explicitly authorized cache sharing, never custody/workdir sharing."""
        self._live()
        source = self._owned(source_id, source_custody)
        if source_id == target_id or source.active is not None:
            raise Rejected("fork needs a different stream and settled source")
        if source.stopped or source.reconciliation_required:
            raise Rejected("source fenced")
        if (
            not isinstance(previous_response_id, str)
            or self._cache.get(previous_response_id) != source_id
        ):
            raise Rejected("completed response absent from this connection cache")
        # A source waits for its fork targets to start. Adding source -> target
        # must not close a dependency cycle, or pump cannot start either fork.
        pending = [target_id]
        visited = set()
        while pending:
            current = pending.pop()
            if current == source_id:
                raise Rejected("cyclic fork dependency")
            if current in visited:
                continue
            visited.add(current)
            stream = self._streams.get(current)
            if stream is not None:
                pending.extend(target for target, _ in stream.fork_waiters)
        self._enqueue(
            target_id,
            target_custody,
            request_id,
            model,
            input,
            previous_response_id,
            fork_source=source_id,
        )
        source.fork_waiters.add((target_id, request_id))

    def _enqueue(
        self,
        stream_id: str,
        custody: Custody,
        request_id: str,
        model: str,
        input: list[Any],
        previous_response_id: str | None,
        *,
        fork_source: str | None = None,
    ) -> None:
        self._live()
        stream = self._owned(stream_id, custody)
        if stream.stopped or stream.reconciliation_required:
            raise Rejected("stream fenced; owner reconciliation required")
        if not isinstance(request_id, str) or not request_id:
            raise Rejected("explicit local request_id required")
        if request_id in stream.requests:
            raise Rejected("request already admitted; replay forbidden")
        if stream.admitted >= stream.request_limit:
            raise Rejected("stream admission budget exhausted")
        if not isinstance(model, str) or not model or not isinstance(input, list):
            raise Rejected("opaque model and input list required")
        if previous_response_id is not None and (
            not isinstance(previous_response_id, str)
            or self._cache.get(previous_response_id) != (fork_source or stream_id)
        ):
            raise Rejected("previous response absent from owned connection cache")
        frame = {
            "type": "response.create",
            "stream_id": stream_id,
            "model": model,
            "store": False,
            "input": input,
        }
        if previous_response_id is not None:
            frame["previous_response_id"] = previous_response_id
        # Validate and snapshot before charging admission or changing queues.
        try:
            frame = json.loads(json.dumps(frame, allow_nan=False))
        except (TypeError, ValueError) as exc:
            raise Rejected("request is not JSON data") from exc
        stream.requests[request_id] = Request(
            request_id, frame, custody, stream.provenance, stream.workdir, fork_source
        )
        stream.queue.append(request_id)
        stream.admitted += 1

    def pump(self) -> None:
        """Send at most one request per idle lane; never resend a sent request."""
        self._live()
        for stream_id, stream in self._streams.items():
            if (
                stream.stopped
                or stream.reconciliation_required
                or stream.active is not None
                or stream.fork_waiters
                or not stream.queue
            ):
                continue
            request = stream.requests[stream.queue[0]]
            previous = request.frame.get("previous_response_id")
            if previous is not None and self._cache.get(previous) != (
                request.fork_source or stream_id
            ):
                # A queued continuation is not permission to invent/replay input.
                stream.reconciliation_required = True
                request.reason = "queued previous response evicted; owner reconcile"
                continue
            stream.queue.popleft()
            stream.active = request.request_id
            request.state = "active"
            # Advancing a source may evict its prior store:false response.
            self._cache = {
                key: owner for key, owner in self._cache.items() if owner != stream_id
            }
            try:
                self.socket.send(request.frame)
            except OSError as exc:
                self.disconnect(str(exc))
                return

    def receive(self) -> None:
        """Reduce one mock inbound event. Empty buffers raise IndexError."""
        self._live()
        event = self.socket.recv()
        if event is None:
            self.disconnect("socket closed")
            return
        if not isinstance(event, dict) or not isinstance(event.get("type"), str):
            self.disconnect("malformed event")
            return
        kind = event["type"]
        if kind == "error" and "stream_id" not in event:
            self.disconnect("connection error: " + str(event.get("error")))
            return
        stream_id = event.get("stream_id")
        if not isinstance(stream_id, str):
            self.disconnect("event has no registered stream_id")
            return
        if stream_id not in self._streams:
            if kind == "error":
                # No owned lane matches: retain evidence, never blame a peer.
                self._unowned_errors.append(deepcopy(event))
            else:
                self.disconnect("event has no registered stream_id")
            return
        stream = self._streams[stream_id]
        if kind == "error":
            stream.errors.append(deepcopy(event))
            if stream.active is None:
                # No sent request can be safely settled; fence only this lane.
                stream.reconciliation_required = True
                return
        if stream.active is None:
            self.disconnect("event cannot be correlated to an active generation")
            return
        request = stream.requests[stream.active]
        response = {} if kind == "error" else event.get("response", {})
        if not isinstance(response, dict):
            self.disconnect("malformed response")
            return
        response_id = (
            None if kind == "error" else response.get("id", event.get("response_id"))
        )
        lifecycle_events = {
            "response.created",
            "response.in_progress",
            "response.completed",
            "response.failed",
            "response.incomplete",
        }
        if kind in lifecycle_events and response_id is None:
            self.disconnect("response lifecycle event missing identity")
            return
        if response_id is not None:
            if (
                not isinstance(response_id, str)
                or not response_id
                or (
                    request.response_id is not None
                    and request.response_id != response_id
                )
                or (
                    response_id in self._response_owners
                    and self._response_owners[response_id]
                    != (stream_id, request.request_id)
                )
            ):
                self.disconnect("response identity mismatch")
                return
            request.response_id = response_id
            self._response_owners[response_id] = (stream_id, request.request_id)
        request.events.append(deepcopy(event))
        # Stop/revocation cannot be undone by a late success frame.
        if request.state == "unknown":
            return
        if kind == "response.in_progress" and request.fork_source is not None:
            self._streams[request.fork_source].fork_waiters.discard(
                (stream_id, request.request_id)
            )
        terminal = {
            "response.completed": "completed",
            "response.failed": "failed",
            "response.incomplete": "incomplete",
            "error": "error",
        }
        if kind not in terminal:
            return
        request.state = terminal[kind]
        request.reason = str(event.get("error")) if kind == "error" else None
        stream.active = None
        if (
            request.fork_source is not None
            and (stream_id, request.request_id)
            in self._streams[request.fork_source].fork_waiters
        ):
            # Failure or an out-of-order completion is NOT the required guard.
            stream.reconciliation_required = True
            self._streams[request.fork_source].reconciliation_required = True
            request.reason = "fork never reached response.in_progress; owner reconcile"
        if request.state == "completed" and not stream.reconciliation_required:
            self._cache[request.response_id] = stream_id

    def stop(self, stream_id: str, custody: Custody) -> None:
        """Owner-local Stop: no remote cancellation or claim of effect rollback."""
        stream = self._owned(stream_id, custody)
        stream.stopped = True
        if stream.active is not None:
            request = stream.requests[stream.active]
            request.state = "unknown"
            request.reason = "Stop after send; owner reconcile remote outcome"
            stream.reconciliation_required = True
            if (
                request.fork_source is not None
                and (stream_id, request.request_id)
                in self._streams[request.fork_source].fork_waiters
            ):
                self._streams[request.fork_source].reconciliation_required = True
        while stream.queue:
            request = stream.requests[stream.queue.popleft()]
            request.state = "not_sent"
            request.reason = "Stop before send"
            if request.fork_source is not None:
                # Deliberately keep the source fenced until owner reconciliation.
                self._streams[request.fork_source].reconciliation_required = True

    def revoke(self, stream_id: str, custody: Custody) -> None:
        """Fence stale custody without granting or transferring a new owner."""
        self.stop(stream_id, custody)
        self._streams[stream_id].revoked = True

    def disconnect(self, reason: str) -> None:
        """Loss affects all active lanes; unsent work stays definitively unsent."""
        if not self.connected:
            return
        self.connected = False
        self.connection_reason = reason
        self._cache.clear()
        self._response_owners.clear()
        for stream in self._streams.values():
            if stream.active is not None:
                request = stream.requests[stream.active]
                request.state = "unknown"
                request.reason = reason + "; owner reconciliation required"
                stream.reconciliation_required = True
            while stream.queue:
                request = stream.requests[stream.queue.popleft()]
                request.state = "not_sent"
                request.reason = reason
            stream.fork_waiters.clear()

    def snapshot(self, stream_id: str, custody: Custody) -> Stream:
        return deepcopy(self._owned(stream_id, custody))

    def cached(self, stream_id: str, custody: Custody) -> tuple[str, ...]:
        self._owned(stream_id, custody)
        return tuple(key for key, owner in self._cache.items() if owner == stream_id)

    def unowned_errors(self) -> list[dict[str, Any]]:
        """Connection-owner audit of errors that could not name an owned lane."""
        return deepcopy(self._unowned_errors)
