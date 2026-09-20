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

## Explicit opaque compaction

`native_compact(canonical=..., identity=...)` requires an idle, settled transport
and a prior successful native request to bind its actual model/tool settings.
It calls the SDK's raw `/responses/compact` endpoint once with SDK retries off.
All returned JSON output items, including opaque encrypted content and retained
items, are copied unchanged. There is no synthetic summary and no automatic
compaction while steering. The next fresh request uses the **entire** returned
window plus only the new suffix, without `previous_response_id`.

`native_export_checkpoint()` is a **private host persistence API**, never model
or UI content. `native_restore_checkpoint(record, canonical=..., identity=...)`
verifies its format, record digest, configuration digest, exact canonical prefix
count/hash, model and provider instance. A new suffix is allowed. Actual request
assembly must also match the exact original wire prefix and request settings;
rewriting, request fitting, or changed tools discards derived state visibly.
Original history is untouched. This format is distinct from context-managed
summaries and cannot be restored through their summary capability. A host must
save originals first, bind ownership, use private atomic persistence, and mark
nonterminal receipts unknown after restart without replay.

## Evidence and scope

Protocol fixtures exercise real provider assembly/parsing, accepted vs applied
identity, pending continuation, interruption/reconnect, bounded ownership and
opaque JSON preservation/restore. They do not prove live provider availability.
The selected live acceptance configuration was Terra/high; native steering was
not enabled or live-tested on a different model.

Protocol checked against current official documentation on 2026-09-20:
[steering](https://developers.openai.com/api/docs/guides/steering),
[WebSocket mode](https://developers.openai.com/api/docs/guides/websocket-mode), and
[compaction](https://developers.openai.com/api/docs/guides/compaction).
