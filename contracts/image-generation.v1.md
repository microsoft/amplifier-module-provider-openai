# OpenAI image generation — v1 (DRAFT 2026-10-13)

Proposed provider-owned boundary; qualification remains separate from this text.
The existing `images.py` implements a direct Images backend, not the complete optional interface below.
This document proposes observable behavior; it does not certify implementation, entitlement, or live API acceptance.

## Who builds against this

Callers preparing and executing an image operation on an already selected provider instance, and maintainers qualifying that instance's native transport.
The provider owns authentication, wire translation and parsing, not account selection, permission policy, durable receipts or file delivery.

## What it is

An optional plain-data exchange identified by `amplifier.image-operation.v1`, independent of Core's required provider protocol.
This contract owns native transport, not the shared exchange schema; interoperability records pin the exact caller schema revision without importing a peer module.
Synthetic lifecycle, not a complete schema or executable example:

```text
caller -> provider.get_image_generation_capabilities(): protocol, qualified transports
caller -> await provider.prepare_image_generation(image_intent): frozen credential-free plan
caller -> await provider.generate_image(image_intent, plan=plan): final bytes + provenance
```

Good contract example: a changed image-model binding is refused before any generation request.
Bad contract example: "works with every OpenAI-compatible service"; compatibility branding cannot establish this boundary.

## The promises

1. **Capability is optional and instance-qualified.** The complete method set above and recognized protocol version are required; unknown versions or partial implementations are unsupported.
   Capability inspection is local: no login, catalog request or paid probe. Preparation may make bounded non-generating catalog reads through this instance's connection.
   SDK support, effective endpoint, authentication mode and selected models qualify each transport; catalog visibility does not prove generation entitlement.
   Proxies, custom endpoints and subclasses do not inherit standard OpenAI support without separate qualification. Core and ordinary chat-model defaults remain unchanged.

2. **Preparation freezes selection, not permission.** The serializable plan binds the exact instance, endpoint identity, transport, controller/image model roles, operation, input digests and effective controls without credentials.
   Requests carry intent and ordered bounded image bytes, not credentials, arbitrary paths, URLs or unrestricted wire overrides. Unsupported controls fail before dispatch.
   Caller-selected models are preserved; any permitted selector is resolved before admission. Execution revalidates the request and binding against the plan without selecting or ranking replacements.
   A prepared plan neither grants permission nor proves a spending limit; changed bindings, inputs or controls require new preparation and admission.

3. **Responses is the default new image transport, with two distinct models.** The top-level model is the selected controller; the native `image_generation` tool uses the separately selected image model.
   The request forces that native tool with `tool_choice`, includes no unrelated tools, and sets `max_tool_calls=1`, `store=False` and SDK `max_retries=0`.
   Image execution does not enter ordinary completion's retries or automatic continuations. Controller reasoning effort is not image quality.
   Unsupported controller/image pairs or conflicting overrides are refused, not relaxed; no image-only model is substituted into the controller slot.

4. **Edits are stateless, and the direct Images route remains available.** Each edit carries the original target and references as ordered bytes; no `previous_response_id` or retained server conversation is required.
   Existing explicit direct-Images routes and image-model pins retain their transport and model choice; they are not silently migrated to Responses.
   Direct generation/editing also disables SDK retries and validates its own model-specific controls. It is never a fallback after an uncertain Responses request.
   The provider does not read/write files, fetch output URLs or mutate its ordinary chat configuration to perform either operation.

5. **Only validated final output can satisfy the request.** Completed `image_generation_call` output is decoded within admitted byte/pixel limits and returned with accompanying final text in order.
   Format, actual geometry and cardinality are checked; refusal, incomplete output, invalid bytes, missing images or unexpected image counts cannot become a completed one-image result.
   Available revised prompts and call identifiers remain distinguishable from the caller's prompt. Recoverable unexpected outputs remain evidence, within limits, not permission to generate again.
   Original image bytes are preserved when possible; any required conversion is explicit in provenance, never an unreported repair of strict output requirements.

6. **One invocation makes at most one generation dispatch.** There is no automatic retry, continuation, account/model substitution or alternate-transport replay, including after cancellation and server errors.
   A timeout or cancellation after possible dispatch is `unknown` unless authoritative terminal evidence establishes another outcome; pre-dispatch refusal remains distinguishable.
   Errors expose safe codes and known dispatch state, not credentials or raw request payloads. Caller-admitted deadlines and cancellation are honored without inheriting chat retry policy.
   Local request identity is not upstream idempotency, exactly-once service execution, or a guarantee against a caller submitting another invocation.

7. **Provenance and usage retain uncertainty.** Results distinguish requested selectors, prepared model IDs and provider-observed versions for controller and image roles, plus available request/response/image-call IDs.
   An unreported observed version stays unknown rather than being filled with the requested alias. Raw vendor usage and normalized usage remain separately identifiable.
   Controller/image usage is not counted twice; unavailable splits, image cost or usage remain unknown, not zero and not fabricated from chat pricing.
   Diagnostics and plans contain no secrets; the provider returns evidence for caller accounting rather than claiming that an unenforced budget was enforced.

## Not in v1

- Streaming previews, batches, background jobs and retained multi-turn state require distinct cancellation, retention and paid-effect qualification before promotion.
- Automatic transport/model fallback and cross-invocation deduplication require an explicit execution/identity contract; compatibility labels alone cannot promote endpoint support.
- Provider-owned ranking policy, filesystem delivery and visual-quality guarantees are outside this boundary; any proposed expansion needs its own observable promises and evidence.

## How the kit checks it

**PROPOSED checks — NOT run.** Provider-local fixtures would inspect SDK requests and public results without live credentials; live checks require a separately authorized bounded spend envelope.

| Promise | Positive check proposed | Negative check proposed |
| --- | --- | --- |
| 1 | Qualified instance exposes all three methods/version without network activity. | Partial interface, unknown version, proxy or subclass without qualification cannot advertise standard support. |
| 2 | Matching prepared request dispatches with the frozen binding and controls. | Changed bytes, endpoint or model is refused with zero generation calls. |
| 3 | Captured Responses request has distinct models, forced tool, one-call cap, storage off and retries zero. | Image model in controller slot or override restoring retries/storage is refused before dispatch. |
| 4 | Explicit legacy generate/edit stays on Images; stateless edit carries ordered bytes. | A failed Responses call causes no Images dispatch or server-state chaining. |
| 5 | Valid final image/text, revised prompt and identifiers survive parsing in order. | Incomplete, malformed or extra image output is not reported as one-image success and causes no replay. |
| 6 | One admitted invocation produces one physical generation request. | Post-send timeout/cancellation remains unknown with zero replacement requests. |
| 7 | Separately reported controller/image usage and observed versions survive normalization. | Missing usage/version remains unknown; overlapping usage is not summed twice. |

Freeze qualification additionally needs an implemented kit, supported SDK/version matrix, authorized live generate/edit evidence and explicit review; none is supplied by this draft.
Public API reference: [OpenAI image generation](https://developers.openai.com/api/docs/guides/image-generation).

## Open questions

- Which minimum SDK version and controller/image combinations pass the complete request, parsing and no-retry checks?
- Which safe refusal codes and binding-change representation will callers use without exposing credentials?
- What usage fields are independently reported versus overlapping totals on each qualified Responses/Images model combination?

## Changelog

| Date | Change | Evidence |
| --- | --- | --- |
| 2026-10-13 | Initial provider image promises. | Existing direct-Images implementation and documented Responses image API; no runtime qualification claimed. |
