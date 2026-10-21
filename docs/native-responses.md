# Optional native Responses transport

`amplifier_module_provider_openai.native.NativeResponsesProvider` extracts the
previous CLI-owned transport into this provider. The default `OpenAIProvider`
and its module mount are unchanged. A host explicitly installs the `native`
extra and wraps a session-local driver **before** instance-level observation or
permission wrappers:

```python
native = NativeResponsesProvider.wrap(driver, owner_getter=current_owner,
    lifecycle=save_private_receipt)
```

The current native steering capability is the selected `gpt-6-astra` model at
`https://api.openai.com/v1`. Other selected models, utility requests explicitly
marked `stream: false`, and requests without a live owner use the ordinary
provider. Native WebSocket calls preserve the provider's request construction,
response parsing, usage and hooks. They do not choose a model, execute tools
directly, or import a CLI, loop, or host package. Streaming SSE, background
polling, conversation-bound requests, automatic compaction, transport overrides,
and unrecognized input-assembly extensions are rejected before sending. A host
must advertise these limits instead of enabling native by default.

The injected owner supplies `context.get_messages/add_message`, `runtime.emit`,
`config`, `_text(command)`, `pending`, `steer(wake)`, `native_job(call_id)`, and
`start_native_job(Core ToolCall)`. The latter must retain the host's existing
hooks and approval path. Only explicitly enabled native asynchronous delegate
calls use it. The CLI continues to own its image budgeting and computer-result
policy through `_prepare_native_messages` and `_validate_native_items` hooks.
Optional private `trace` and `trace_context` callbacks retain CLI diagnostics;
public receipts never contain input/output or hidden reasoning.

`steer_live(command)` accepts one identified input while a response is active.
Acceptance is recorded once but is not application. `steering.applied` is emitted
only when a different successor `response.created` arrives. Pending steering
requires matching actual saved tool outputs on the same socket and original
parent; it never fabricates or reruns a result. A definite failure before
acceptance can enter the boundary queue. Accepted-then-failed input is retained
in canonical history and stops for explicit recovery instead of being inserted
a second time. Incorrect/duplicate receipt identities fail closed.

The host's async lifecycle sink records `native.submitting` **before** the send,
then identity-only response/steering/completion observations. A failed initial
receipt prevents sending. Hosts bind session/generation themselves; callbacks
must not infer authority from provider payloads. A disconnect, timeout, bound
exceeded, or cancelled await after sending is non-retryable and reports an
unknown outcome. Closing a socket is not proof of remote cancellation. No
request, accepted steering input, or tool is automatically replayed. Idle closed
or aged connections can start a fresh lineage using the current context and
actual tool receipts; unresolved pending/unknown work cannot reconnect that way.

Frames are bounded to 16 MiB, cumulative response output to 32 MiB, receive
events to 100,000 per request, asynchronous call identities to 1,024 and native
call history to 10,000. Configured request timeout still applies. Reaching a
bound is a visible unknown/recovery boundary, never silently truncated success.

## Optional immutable outbound admission

`wrap(..., prepare_send=prepare)` adds a private, fail-closed boundary for both
`response.create` and `response.steer`. Omitting it preserves existing callers.
It does not enable native transport, change provider selection, or authorize tools.

```python
from amplifier_module_provider_openai.native import NativeSendAdmission

async def prepare(kind, *, input_id):
    # Capture the host's current attempt, not an identity from provider content.
    attempt = current_attempt()
    await prepare_host_validation(attempt)

    def admit(envelope):
        # Synchronous final currentness check and private observation.
        if not still_current(attempt):
            return False
        observe_private_candidate(attempt, envelope)
        return True

    return NativeSendAdmission(attempt_id=attempt.id, admit=admit)
```

The preparation callable may be synchronous or awaitable. It receives the wire
kind and `input_id=None` for create, or the local `command.id` for steer. It must
return `NativeSendAdmission` with a nonempty host-supplied string `attempt_id`
and a synchronous `admit` callable. The provider does not generate, persist,
deduplicate, or establish authority for these host identities.

For create, existing `trace_context`, `trace("native.send", ...)`, and awaited
`native.submitting` lifecycle/runtime calls still run first. Preparation runs
after checkpoint/delta construction. The provider then serializes the final
envelope and passes a frozen `NativeOutboundEnvelope` to `admit`. Its fields are
`attempt_id`, `kind`, `input_id`, and `payload`: the exact JSON string to be
passed to `socket.send`, not a mutable dictionary or the full preflight request.
Parsing it produces a separate copy. The payload is omitted from the record's
repr; hosts must not place private content in ordinary logs. Existing diagnostic
callbacks remain private and mutable, not send proofs.

Only literal `True` admits; false, missing/malformed admission, exceptions, or an
awaitable returned by `admit` prevent this send. Cancellation during preparation
also prevents it. Steering reserves a separate preparation slot and
rechecks the owner, socket and parent afterward; a completed/replaced parent
cannot receive stale steering. A refused unsent steer clears only its own slot,
does not enter the boundary queue, and does not close an independently running
create. A candidate observation is not a sent or remotely accepted receipt.
An unsent create is known unsent, but its delta/checkpoint bookkeeping has already
advanced. That adapter refuses further completions with an explicit host-recovery
error; it does not silently omit the unsent input or reset lineage to retry.
The host must retain canonical history and its original attempt disposition.
Reconstructing an adapter is a deliberate host recovery decision, not permission
to replay. Await `close_live()` on the original adapter **before** reconstructing
it; a failed close remains a cleanup failure, not successful resource disposal.
No-callback callers keep their existing submitting/unknown semantics.

### Persisting known-unsent outcomes

Hosts opting into `prepare_send` must also consume the following lifecycle
contract. It is not installed automatically in existing CLI/app consumers:
ignoring the terminal event leaves their earlier submitting receipt unresolved.
Bind the injected lifecycle sink to the original host attempt, owner and budget
before calling `complete`/`steer_live`; do not select a new attempt from mutable
global state after an await.

For an opted-in create **or steer**, `native.submitting` carries a fresh local
`send_id`, `operation` (`response.create` or `response.steer`), captured `epoch`,
`response_id` (previous response for create; parent for steer), `input_id`
(`None` for create; local command ID for steer), `attempt_id=None`, and
`execution_replayed=False`. This ID correlates lifecycle records; it is not a
server idempotency key, new host attempt, budget or authorization.

If initial persistence, preparation, final admission or the pre-send identity
check fails/cancels, `native.not_sent` carries those same identities plus a fixed
`reason` (`rejected` or `cancelled`). `attempt_id` is the host-supplied admission
ID if validated preparation returned one; otherwise it remains `None`. Resolve
that `None` using the original host binding and `send_id`, never a new attempt.
There is no callback exception, prompt, tool payload or response content in the
receipt. It means this operation never entered `socket.send`; it does not
dispose of resources, complete an active parent, or authorize replay.

The provider snapshots the lifecycle and owner-runtime sinks before submission
and passes separate copies of the receipt to each. Terminal delivery tries each
captured sink once, with a one-second **cooperative** timeout per sink, without
changing the original exception/cancellation or send uncertainty. A failed,
cancelled or timed-out sink sets the existing `native_status().observerFailed`;
there is no hidden retry. Non-yielding or cancellation-suppressing host callbacks
cannot be forcibly bounded by asyncio. Hosts must keep sinks cooperative.

Durable consumer rules (exercised by the synthetic disk/restart fixture):

1. Persist `submitting`, keyed by `send_id`, under the original host attempt,
   owner/budget and captured connection epoch. A steer has its own record.
2. On matching `native.not_sent`, persist terminal `not_sent` for **only** that
   send. Validate the operation, parent, epoch and local input identity; retain
   the original attempt and budget. Never terminate its parent or erase another
   send's unknown outcome.
3. On restart, retain persisted `not_sent`. A submitting record with no persisted
   terminal remains `unknown`, including a crash or receipt-sink failure.
   An in-memory flag, caught local exception or successful socket close cannot
   substitute for the terminal record.

Once send is entered, cancellation/failure follows the existing unknown outcome
path, **never** `native.not_sent`. A rejected/cancelled unsent steer can be
recorded while the parent is still running, completes, or becomes unknown; its
terminal does not reclassify that parent. Downstream hosts must implement this
distinction before opting in. These tests do not claim installed host adoption.

There is no intervening await between synchronous final admission and
selecting the immutable argument handed to `socket.send`. This is a **local
admission/argument-selection point**, not proof of remote receipt, network
atomicity, writer exclusion through delivery, or model comprehension. A send can
still fail or be cancelled after observation; existing unknown-outcome and
no-replacement/no-replay rules apply. Neither observation nor refusal authorizes
a replacement attempt. Hosts must reconcile the original attempt, not retry
just to obtain an observation.

For early tool provenance, the existing synchronous `trace("native.receive",
event)` runs before async tool dispatch. Hosts can associate each
`response.created` and each subsequent `response.output_item.done` (original
call ID/name/arguments) with that response in wire event order. Steering
acceptance is not application: correlate local input IDs with the existing
accepted/applied identities and the distinct successor response. One completion
can aggregate tools from several responses; its final response ID is **not** the
origin of every tool. Validate missing/conflicting identities in the host before
execution. No provenance is inferred from `previous_response_id` or opaque
checkpoint content, and this API supplies no canonical store or Work policy.

## Explicit opaque compaction

`native_compact(canonical=..., identity=...)` requires an idle, settled transport
and a prior successful native request to bind its actual model/tool settings.
It calls the SDK's raw `/responses/compact` endpoint once with SDK retries off.
Compaction preserves the last successful request's exact system instructions,
including factory or ephemeral host policy absent from canonical history.
Changed future instructions invalidate the derived checkpoint. Its result
exposes only finite declared usage counters for host accounting; omitted usage
remains unknown. All returned JSON output items, including opaque encrypted content and retained
items, are copied unchanged. There is no synthetic summary and no automatic
compaction while steering. The next fresh request uses the **entire** returned
window plus only the new suffix, without `previous_response_id`. Request budgets
and the initial send guard measure that same opaque window through a pure
planner; fitting does not advance connection lineage. Each top-level completion
can issue one native request. A driver-internal continuation requires an explicit
next provider call instead of silently replaying the request.

`native_export_checkpoint()` is a **private host persistence API**, never model
or UI content. `native_restore_checkpoint(record, canonical=..., identity=...)`
verifies its format, record digest, configuration digest, exact canonical prefix
count/hash, model and provider instance. A new suffix is allowed. Actual request
assembly must also match the exact original wire prefix and request settings.
With turn-scoped reasoning, a later real user message normally removes prior
reasoning items from that wire view. The planner allows that deletion only after
reconstructing and verifying the saved original wire digest, then comparing every
remaining covered item exactly. It uses the resulting shorter prefix boundary,
preserving the opaque checkpoint and sending only the new suffix. Text, tool
results, authority, instructions and configuration receive no such relaxation.
An unrecognized serialization change, rewriting, request fitting, or changed
tools discards derived state visibly.
Original history is untouched. This format is distinct from context-managed
summaries and cannot be restored through their summary capability. A host must
save originals first, bind ownership, use private atomic persistence, and mark
nonterminal receipts unknown after restart without replay.

## Evidence and scope

Protocol fixtures exercise real provider assembly/parsing, accepted vs applied
identity, pending continuation, interruption/reconnect, bounded ownership and
opaque JSON preservation/restore. Separately authorized synthetic live checks on
2026-09-20 used the configured `gpt-6-astra` / `xhigh` account, OpenAI SDK 3.5.0,
and a 1,024 output-token cap without changing saved defaults. They established
real streaming steering and, in a separate retained fixture, exact checkpoint
restoration and a completed continuation. The earlier failed continuity check is
preserved alongside the fix evidence. These checks do not establish arbitrary
account/model support or recovery of an unknown remote action. See
[bounded live acceptance evidence](native-acceptance-2026-09-20.md).

Protocol checked against current official documentation on 2026-09-20:
[steering](https://developers.openai.com/api/docs/guides/steering),
[WebSocket mode](https://developers.openai.com/api/docs/guides/websocket-mode), and
[compaction](https://developers.openai.com/api/docs/guides/compaction).
