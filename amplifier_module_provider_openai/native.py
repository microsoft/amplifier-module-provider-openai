"""Optional provider-owned Responses WebSocket transport.

Hosts inject the live owner protocol and optional private diagnostic callbacks.
No host, loop, CLI or app package is imported here. Ordinary provider behavior is
unchanged unless a host explicitly wraps this instance and supplies an owner.
"""

import asyncio
import contextvars
import copy
import inspect
import json
import time
import uuid
from collections import Counter
from urllib.parse import urlparse

import websockets
from amplifier_core.llm_errors import LLMError
from amplifier_core.message_models import ToolCall
from websockets.protocol import State

from . import OpenAIProvider, _RawResponseObject
from .native_checkpoint import NativeCheckpointMixin

NATIVE_REQUEST = contextvars.ContextVar("openai_native_request", default=None)
NATIVE_BUDGET = contextvars.ContextVar("openai_native_budget", default=None)


def key(message):
    return json.dumps(message, sort_keys=True, default=str)


def add_usage(total, usage):
    for name, value in usage.items():
        if isinstance(value, dict):
            add_usage(total.setdefault(name, {}), value)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            total[name] = total.get(name, 0) + value


class NativeResponsesProvider(NativeCheckpointMixin, OpenAIProvider):
    native_bundle_live = True

    @classmethod
    def wrap(
        cls, original, *, owner_getter, trace=None, trace_context=None, lifecycle=None
    ):
        if "complete" in original.__dict__:
            raise ValueError(
                "Install native transport before instance-level provider wrappers"
            )
        provider = cls.__new__(cls)
        # A session-local extension preserves the mounted driver's credentials,
        # configuration, native serializers, cost contributor and lazy client.
        provider.__dict__ = original.__dict__.copy()
        provider._observer_failure = None
        provider._close_failure = None
        provider.lifecycle = lifecycle
        provider.owner_getter = owner_getter
        provider.trace = trace or (lambda *args: None)
        provider.trace_context = trace_context or (lambda *args: None)
        provider._request_messages = []
        provider._checkpoint = None
        provider._checkpoint_status = "empty"
        provider._native_busy = False
        provider.socket = None
        provider.epoch = str(uuid.uuid4())
        provider.previous_response_id = None
        provider.response_id = None
        provider.owner = None
        provider.inflight = None
        provider.awaiting_successor = False
        provider.seen = Counter()
        provider.last_context = Counter()
        provider.async_calls = {}
        provider.delivered_results = set()
        provider.pending_parent = False
        provider.full_messages = []
        provider.pending_steer = None
        provider.native_calls = {}
        provider.socket_opened_at = None
        provider.request_uncertain = False
        return provider

    async def _emit(self, kind, **data):
        if self.lifecycle is not None:
            await self.lifecycle(kind, data)
        if self.owner is not None:
            await self.owner.runtime.emit(kind, **data)

    def _prepare_native_messages(self, messages):
        return messages

    def _validate_native_items(self, items):
        return items

    @staticmethod
    def _steer_identity(event):
        identity = (event.get("steer") or {}).get("id")
        if not isinstance(identity, str) or not identity:
            raise RuntimeError("Missing steering identity")
        return identity

    def _match_steer(self, event):
        received = self._steer_identity(event)
        if self.inflight.get("accepted") and self.inflight.get("steer_id") != received:
            raise RuntimeError("Mismatched steering identity")

    async def _uncertain(self, reason):
        if not self.request_uncertain:
            return
        pending = self.inflight or self.pending_steer
        try:
            await self._emit(
                "native.outcome_unknown",
                reason=reason,
                input_id=pending["command"].id if pending else None,
                response_id=self.response_id or self.previous_response_id,
                accepted=bool(pending and pending.get("accepted")),
                execution_replayed=False,
            )
        except (Exception, asyncio.CancelledError) as exc:  # noqa: BLE001 - preserve the original non-retryable failure
            # A broken receipt sink must never escape as a generic retryable
            # transport failure. The prior submitting/running receipt remains
            # nonterminal and the host reconciles it to unknown after restart.
            self._observer_failure = type(exc).__name__
        finally:
            if self.socket:
                try:
                    await self.socket.close()
                except (Exception, asyncio.CancelledError) as exc:  # noqa: BLE001 - cleanup cannot enable replay
                    self._close_failure = type(exc).__name__
                self.socket = None

    async def request_budget(self, request, **kwargs):
        owner = self.owner_getter()
        options = self._merge_request_options(
            kwargs.get("request_options"),
            {k: v for k, v in kwargs.items() if k != "request_options"},
        )
        canonical = None
        if (
            owner is not None
            and self._checkpoint
            and options.get("model", self.default_model) == "gpt-6-astra"
            and options.get("extended_thinking") is not False
            and (request.metadata or {}).get("stream") is not False
        ):
            canonical = await owner.context.get_messages()
        token = NATIVE_BUDGET.set((self, canonical) if canonical is not None else None)
        try:
            result = super().request_budget(request, **kwargs)
            return await result if inspect.isawaitable(result) else result
        finally:
            NATIVE_BUDGET.reset(token)

    def _assemble_initial_responses_params(self, request, **kwargs):
        params, state, logs = super()._assemble_initial_responses_params(
            request, **kwargs
        )
        if NATIVE_REQUEST.get() is self:
            # Retain the canonical wire view only in private planning state. The
            # public hook/budget payload is the actual full compact window.
            state = {**state, "native_full_params": copy.deepcopy(params)}
            params = self._plan_checkpoint(params, self.full_messages)
        else:
            budget = NATIVE_BUDGET.get()
            if budget is not None and budget[0] is self:
                params = self._plan_checkpoint(params, budget[1])
        return params, state, logs

    def _commit_initial_assembly_state(self, state):
        super()._commit_initial_assembly_state(state)
        if NATIVE_REQUEST.get() is self:
            self._native_full_params = state.get("native_full_params")

    async def complete(self, request, **kwargs):
        kwargs = self._merge_request_options(
            kwargs.pop("request_options", None), kwargs
        )
        owner = self.owner_getter()
        if (
            owner is None
            or (request.metadata or {}).get("stream") is False
            or kwargs.get("extended_thinking") is False
            or kwargs.get("model", self.default_model) != "gpt-6-astra"
        ):
            token = NATIVE_REQUEST.set(None)
            try:
                return await super().complete(request, **kwargs)
            finally:
                NATIVE_REQUEST.reset(token)
        if self.owner is not None or self._native_busy:
            raise RuntimeError("Native provider already owns a generation")
        self.owner = owner
        self._native_full_params = None
        self._native_request_attempted = False
        self._request_messages = [
            m.model_dump()
            for m in request.messages
            if m.role in {"developer", "user", "assistant", "tool"}
        ]
        token = NATIVE_REQUEST.set(self)
        request = request.model_copy(
            update={"metadata": {**(request.metadata or {}), "stream": False}}
        )
        try:
            await self._renew_idle_connection()
            # context-simple compacts the request view, not the saved history.
            # Keep original call identities/results outside that lossy view.
            getter = getattr(owner.context, "get_messages", None)
            self.full_messages = await getter() if getter else []
            for message in [
                *self.full_messages,
                *(m.model_dump() for m in request.messages),
            ]:
                for call in (message.get("metadata") or {}).get(
                    "converge_native_async_calls", []
                ):
                    self.async_calls.setdefault(call["call_id"], call)
            response = await super().complete(request, **kwargs)
            self._last_native_request = (request, copy.deepcopy(kwargs))
            calls = [
                self.async_calls[call.id]
                for call in response.tool_calls or []
                if call.id in self.async_calls
            ]
            response.metadata = {
                **(response.metadata or {}),
                "converge_live_epoch": self.epoch,
                "converge_native_async_calls": calls,
            }
            return response
        finally:
            NATIVE_REQUEST.reset(token)
            self.owner = None
            self.response_id = None
            self.full_messages = []

    def _find_missing_tool_results(self, messages):
        missing = super()._find_missing_tool_results(messages)
        if NATIVE_REQUEST.get() is self:
            # Native async calls are intentionally unpaired while running.
            remaining = [entry for entry in missing if entry[1] not in self.async_calls]
            if remaining:
                raise LLMError(
                    "Native tool results are missing; explicit recovery is required",
                    provider=self.name,
                    retryable=False,
                )
            return []
        return missing

    def _convert_messages(self, messages, **kwargs):
        messages = self._prepare_native_messages(messages)
        if NATIVE_REQUEST.get() is not self:
            return self._validate_native_items(
                super()._convert_messages(messages, **kwargs)
            )
        for message in messages:
            for call in (message.get("metadata") or {}).get(
                "converge_native_async_calls", []
            ):
                self.async_calls.setdefault(call["call_id"], call)
        # Compare the normalized request view. The presence of an old failed or
        # superseded screenshot alone must not reset every subsequent request
        # (or let a utility call reset an active manager's lineage).
        persistent = Counter(
            key(m)
            for m in messages
            if m.get("role") != "tool"
            and not (m.get("metadata") or {}).get("ephemeral")
        )
        if self.last_context - persistent:
            # A rewritten/compacted window must become a new lineage, so dropped
            # instructions do not linger on the server. Do not discard pending
            # tool identities or accepted steering as a side effect of compaction.
            if self.pending_parent:
                raise LLMError(
                    "Pending steering cannot move to a rewritten context or new connection; explicit recovery is required",
                    provider=self.name,
                    retryable=False,
                )
            self.previous_response_id = None
            self.seen.clear()
            self.epoch = str(uuid.uuid4())
        self.last_context = persistent
        selected, occurrences = [], Counter()
        for message in messages:
            metadata = message.get("metadata") or {}
            if (
                self.previous_response_id
                and metadata.get("converge_live_epoch") == self.epoch
            ):
                continue
            call_id = message.get("tool_call_id")
            if message.get("role") == "tool" and call_id in self.async_calls:
                continue  # Reconcile from the original job/full history below.
            identity = key(message)
            occurrences[identity] += 1
            if (
                not self.previous_response_id
                or metadata.get("ephemeral")
                or occurrences[identity] > self.seen[identity]
            ):
                selected.append(message)
        self.seen |= occurrences
        # Run the full serializer once to retain native call/namespace maps even
        # when their assistant messages are already in the server's lineage.
        super()._convert_messages(messages, **kwargs)
        items = super()._convert_messages(selected, **kwargs)
        outstanding = {
            identity
            for identity in self.async_calls
            if (job := self.owner.native_job(identity)) is not None
            and (not job.get("restored") or job.get("recovered"))
            and identity not in self.delivered_results
        }
        results = {
            m.get("tool_call_id"): m.get("content")
            for m in [*messages, *self.full_messages]
            if m.get("role") == "tool"
        }

        def append_call(target, identity):
            if not self.previous_response_id:
                target.append(copy.deepcopy(self.async_calls[identity]))
            elif identity in self.delivered_results:
                return
            job = self.owner.native_job(identity)
            result = job.get("result") if job is not None else results.get(identity)
            if result is not None:
                target.append(
                    {
                        "type": "function_call_output",
                        "call_id": identity,
                        "output": result,
                    }
                )
                self.delivered_results.add(identity)

        # Restore the native call and its actual result IN PLACE. Moving old
        # calls behind the newest user message makes historical delegations look
        # newly issued. Discard compacted receipts and synchronous repair's
        # fabricated outputs; an unfinished async call stays unpaired.
        reconciled, visible = [], set()
        for item in items:
            identity = item.get("call_id")
            if identity in self.async_calls and item.get("type") in {
                "function_call",
                "function_call_output",
            }:
                if item["type"] == "function_call" and identity not in visible:
                    append_call(reconciled, identity)
                    visible.add(identity)
            else:
                reconciled.append(item)
        # Jobs omitted by compaction still need their original identities on a
        # fresh lineage, before the current direction. A continuation sends only
        # newly available real results after its new input.
        pending_history = []
        for identity in self.async_calls:
            if identity in outstanding - visible:
                append_call(
                    reconciled if self.previous_response_id else pending_history,
                    identity,
                )
        return self._validate_native_items(pending_history + reconciled)

    async def _connect(self):
        if self.socket is not None:
            return
        client = self.client
        base = str(client.base_url).rstrip("/")
        parsed = urlparse(base)
        if (
            parsed.scheme != "https"
            or parsed.netloc != "api.openai.com"
            or parsed.path.rstrip("/") != "/v1"
            or parsed.query
            or parsed.fragment
        ):
            raise RuntimeError(
                "Native Astra requires the configured official OpenAI endpoint"
            )
        headers = {
            name: value
            for name, value in {**client.auth_headers, **client.default_headers}.items()
            if isinstance(value, str)
        }
        headers.pop("Content-Type", None)
        self.socket = await websockets.connect(
            "wss://" + parsed.netloc + parsed.path + "/responses",
            additional_headers=headers,
            open_timeout=20,
            close_timeout=5,
            max_size=16 * 1024 * 1024,
        )
        self.socket_opened_at = time.monotonic()

    async def _renew_idle_connection(self):
        # Called before serialization of a NEW request. Never retry a send/recv:
        # if the previous generation ended ambiguously, its tools may have run.
        if self.request_uncertain:
            raise LLMError(
                "Native response outcome is uncertain; no automatic replay",
                provider=self.name,
                retryable=False,
            )
        if self.socket is None:
            return
        closed = getattr(self.socket, "state", State.OPEN) != State.OPEN
        aged = (
            self.socket_opened_at is not None
            and time.monotonic() - self.socket_opened_at >= 55 * 60
        )
        if not closed and not aged:
            return
        if self.pending_parent or self.inflight or self.awaiting_successor:
            if closed:
                self.request_uncertain = True
                await self._uncertain("pending_connection_closed")
                raise LLMError(
                    "Native connection closed with pending steering; no automatic replay",
                    provider=self.name,
                    retryable=False,
                )
            return  # Pending successors belong to this socket until resolved.
        await self.socket.close()
        self.socket = None
        self.socket_opened_at = None
        # The configured driver uses store=false. The socket's response cache
        # cannot be resumed on another socket. Serialize the current full window
        # and real outstanding call identities/results, as after compaction.
        self.previous_response_id = None
        self.seen.clear()
        self.epoch = str(uuid.uuid4())
        await self._emit(
            "native.connection.renewed",
            reason="closed" if closed else "age",
            context="current_window",
            execution_replayed=False,
        )

    async def steer_live(self, command):
        if (
            not self.owner
            or not self.response_id
            or self.inflight
            or self.awaiting_successor
        ):
            return False
        self.inflight = {
            "command": command,
            "accepted": False,
            "parent": self.response_id,
        }
        try:
            await self.socket.send(
                json.dumps(
                    {
                        "type": "response.steer",
                        "previous_response_id": self.response_id,
                        "input": self.owner._text(command),
                    }
                )
            )
        except BaseException:
            await self._uncertain("steer_send")
            raise
        await self._emit(
            "steering.sent", input_id=command.id, response_id=self.response_id
        )
        return True

    async def _create_response(self, params, *, native_input_tokens=None):
        if NATIVE_REQUEST.get() is not self:
            return await super()._create_response(
                params, native_input_tokens=native_input_tokens
            )
        try:
            if getattr(self, "_native_request_attempted", False):
                raise LLMError(
                    "Native transport requires an explicit next provider request; no internal replay",
                    provider=self.name,
                    retryable=False,
                )
            # Planning uses the full request without advancing connection lineage.
            # Commit native deltas only once, immediately before the actual send.
            full_params = copy.deepcopy(
                getattr(self, "_native_full_params", None) or params
            )
            ordinary = super()._convert_messages(
                self._prepare_native_messages(self._request_messages)
            )
            if ordinary != full_params.get("input"):
                raise LLMError(
                    "Native transport cannot preserve this input assembly extension; use ordinary transport",
                    provider=self.name,
                    retryable=False,
                )
            params = {**params, "input": self._convert_messages(self._request_messages)}
            params = self._apply_checkpoint(params, full_params)
            if native_input_tokens is None:
                await self._guard_assembled_params_with_provider_count(params)
            else:
                self._guard_assembled_params(
                    params, native_input_tokens=native_input_tokens
                )
            self._native_request_attempted = True
            return await self._native_response(params)
        except asyncio.CancelledError:
            await self._uncertain("cancelled")
            raise
        except Exception as exc:
            # The conventional driver's generic exception handler retries.
            # Native calls may already have dispatched jobs, so explicitly use
            # the kernel's non-retryable contract even for transport failures.
            await self._uncertain(type(exc).__name__)
            if isinstance(exc, LLMError) and not self.request_uncertain:
                raise
            raise LLMError(
                "Native Responses stopped; outcome unknown, no automatic replay"
                if self.request_uncertain
                else "Native request rejected before send; no automatic replay",
                provider=self.name,
                retryable=False,
            ) from exc

    async def _native_response(self, params):
        await self._connect()
        payload = copy.deepcopy(params)
        if payload.get("background") or payload.get("stream"):
            raise RuntimeError(
                "Native bundle transport does not support background polling or SSE parameters"
            )
        payload.pop("stream", None)
        # SDK transport-only options are not Responses wire fields.
        for field in ("timeout", "extra_headers", "extra_query"):
            if field in payload:
                raise RuntimeError(
                    "Native transport cannot preserve this per-request transport override"
                )
        extra_body = payload.pop("extra_body", {})
        payload.update(extra_body)
        if payload.get("conversation") or payload.get("context_management"):
            raise ValueError(
                "Native steering cannot combine conversation state or automatic compaction"
            )
        if self.previous_response_id:
            payload["previous_response_id"] = self.previous_response_id
        if self.pending_parent:
            for required in self.pending_steer["required_input"]:
                identity = required.get("call_id")
                if not identity or not any(
                    m.get("role") == "tool" and m.get("tool_call_id") == identity
                    for m in self.full_messages
                ):
                    raise ValueError(
                        "Pending native steering requires the saved original tool result"
                    )
                if not any(
                    i.get("call_id") == identity
                    and i.get("type") == required.get("type")
                    for i in payload.get("input", [])
                ):
                    raise ValueError(
                        "Pending native steering continuation is missing a required input"
                    )
        async_names = set()

        def mark(tools):
            for tool in tools:
                if tool.get("type") == "namespace":
                    mark(tool.get("tools", []))
                elif (
                    self.owner.config.get("background_delegate")
                    and tool.get("type") == "function"
                    and tool.get("name") == "delegate"
                ):
                    tool["async"] = True
                    async_names.add("delegate")

        mark(payload.get("tools", []))
        self.trace_context(payload)
        self.trace("native.send", {"type": "response.create", **payload})
        self.request_uncertain = True
        await self._emit("native.submitting", execution_replayed=False)
        await self.socket.send(json.dumps({"type": "response.create", **payload}))
        await self._emit(
            "native.request",
            continuation=bool(self.previous_response_id),
            input_items=len(payload.get("input", [])),
            async_tools=sorted(async_names),
        )
        output, output_ids, totals, terminal = [], set(), {}, None
        output_bytes = 0
        event_count = 0
        resumed_steer = self.pending_steer if self.pending_parent else None
        self.pending_parent = False
        self.pending_steer = None
        async with asyncio.timeout(max(1, self.timeout - 2)):
            while True:
                event_count += 1
                if event_count > 100_000:
                    raise RuntimeError("Native event bound exceeded")
                event = json.loads(await self.socket.recv())
                self.trace("native.receive", event)
                kind = event.get("type")
                if kind == "response.created":
                    terminal = None
                    self.response_id = event["response"]["id"]
                    if self.awaiting_successor or resumed_steer:
                        pending = self.inflight or resumed_steer
                        if self.response_id == pending.get("parent"):
                            raise RuntimeError(
                                "Steering successor must have a new response identity"
                            )
                        await self._emit(
                            "steering.applied",
                            input_id=pending["command"].id,
                            response_id=self.response_id,
                            steer_id=pending.get("steer_id"),
                        )
                        self.awaiting_successor = False
                        self.inflight = None
                        resumed_steer = None
                    await self._emit(
                        "native.response.created", response_id=self.response_id
                    )
                elif kind == "response.output_item.done":
                    item = event["item"]
                    if item.get("type") == "function_call" and item.get("async"):
                        if item.get("name") not in async_names:
                            raise RuntimeError("Unexpected native async tool")
                        call_id = item["call_id"]
                        if call_id not in self.async_calls:
                            if len(self.async_calls) >= 1024:
                                raise RuntimeError(
                                    "Native asynchronous call bound exceeded"
                                )
                            self.async_calls[call_id] = item
                            await self.owner.start_native_job(
                                ToolCall(
                                    id=call_id,
                                    name=item["name"],
                                    arguments=json.loads(item["arguments"]),
                                )
                            )
                elif kind == "response.steer.accepted":
                    if not self.inflight or self.inflight["accepted"]:
                        raise RuntimeError(
                            "Unmatched or duplicate steering acknowledgment"
                        )
                    self.inflight["accepted"] = True
                    self.inflight["steer_id"] = self._steer_identity(event)
                    self.awaiting_successor = True
                    command = self.inflight["command"]
                    # Record accepted direction once. The server owns delivery;
                    # the next ordinary request must not submit it a second time.
                    await self.owner.context.add_message(
                        {
                            "role": "user",
                            "content": self.owner._text(command),
                            "metadata": {
                                "converge_live_epoch": self.epoch,
                                "live_input_id": command.id,
                            },
                        }
                    )
                    await self._emit(
                        "steering.accepted",
                        input_id=command.id,
                        response_id=self.inflight["parent"],
                        steer_id=event.get("steer", {}).get("id"),
                    )
                elif kind == "response.steer.failed":
                    if not self.inflight:
                        raise RuntimeError("Unmatched steering failure")
                    self._match_steer(event)
                    command = self.inflight["command"]
                    await self._emit("steering.failed", input_id=command.id)
                    # Once acceptance is in canonical history, automatic
                    # boundary reinsertion would duplicate the same user input.
                    # Retain that original evidence and require explicit recovery.
                    if self.inflight["accepted"]:
                        raise LLMError(
                            "Accepted steering was not applied; explicit input recovery is required",
                            provider=self.name,
                            retryable=False,
                        )
                    # A definite failure before acceptance has no canonical
                    # entry yet. Only this case can enter the boundary queue.
                    self.owner.pending.append(command)
                    self.owner.steer("[loop-live inbox wake]")
                    self.inflight, self.awaiting_successor = None, False
                    if terminal:
                        break
                elif kind == "response.steer.pending":
                    if not self.inflight or not self.inflight["accepted"]:
                        raise RuntimeError("Unmatched pending steering")
                    self._match_steer(event)
                    self.awaiting_successor = False
                    self.pending_parent = True
                    self.pending_steer = {
                        **self.inflight,
                        "required_input": event.get("required_input", []),
                    }
                    await self._emit(
                        "steering.pending",
                        input_id=self.inflight["command"].id,
                        response_id=self.previous_response_id,
                    )
                    self.inflight = None
                    if terminal:
                        break
                elif kind in {"response.completed", "response.incomplete"}:
                    terminal = event["response"]
                    self.previous_response_id = terminal["id"]
                    self.response_id = None
                    if self.owner.config.get("latest_response_only"):
                        output.clear()
                        output_ids.clear()
                    output_bytes += len(json.dumps(terminal.get("output", [])).encode())
                    if output_bytes > 32 * 1024 * 1024:
                        raise RuntimeError("Native output bound exceeded")
                    for item in terminal.get("output", []):
                        if item.get("call_id"):
                            if len(self.native_calls) >= 10_000:
                                raise RuntimeError("Native call history bound exceeded")
                            self.native_calls[item["call_id"]] = item
                        identity = item.get("id") or key(item)
                        if identity not in output_ids:
                            output_ids.add(identity)
                            output.append(item)
                    usage = terminal.get("usage") or {}
                    add_usage(totals, usage)
                    if (
                        kind == "response.incomplete"
                        and (terminal.get("incomplete_details") or {}).get("reason")
                        != "steered"
                    ):
                        raise RuntimeError(
                            "Native response incomplete: "
                            + str(
                                (terminal.get("incomplete_details") or {}).get("reason")
                            )
                        )
                    if not self.inflight and not self.awaiting_successor:
                        break
                elif kind in {"response.failed", "error"}:
                    # Deliberately non-retryable: a job may already have run.
                    code = (
                        event.get("error")
                        or (event.get("response") or {}).get("error")
                        or {}
                    ).get("code", kind)
                    await self._emit("native.error", code=code)
                    raise RuntimeError(
                        "Native Responses failure; no automatic execution replay: "
                        + str(code)
                    )
        terminal = {**terminal, "output": output, "usage": totals}
        await self._emit(
            "native.completed",
            response_id=terminal["id"],
            pending_steering=self.pending_parent,
        )
        self.request_uncertain = False
        return _RawResponseObject(terminal)

    async def close_live(self):
        if self.request_uncertain or self.pending_parent:
            self.request_uncertain = True
            await self._uncertain("connection_closed")
        if self.socket:
            await self.socket.close()
            self.socket = None
        await super().close()
