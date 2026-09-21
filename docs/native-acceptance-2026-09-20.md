# Bounded native transport acceptance, 2026-09-20

These separately authorized synthetic checks used the selected configured
`gpt-6-astra` / `xhigh` account, official OpenAI endpoint, SDK 3.5.0 and
websockets 16.1.1. Saved defaults were unchanged. Every generation had a 1,024
output-token cap and no SDK retries. No desktop, microphone, real files, or
external tool effects were involved. The reports omit credentials, private
canonical reasoning and opaque checkpoint contents.

## Real steering

[The first report](evidence/native-provider-live-smoke.json) records one initial
generation plus its accepted steering successor and one explicit compact call.
The initial response became incomplete because it was steered; the successor
completed and used the corrected tag. Twelve public text deltas were observed.
One pure local synthetic delegate result was produced, but the two-generation
cap stopped before it was sent back. The idle reconnect handshake and checkpoint
creation succeeded. That run did **not** establish tool-result consumption or a
generated continuation after reconnect, and its private state was not retained.

## Preserved checkpoint failure and fix

[A separate reconstructed fixture](evidence/native-continuity-live-report.json)
used the known public synthetic call/result pair as explicitly labeled input,
with no executable tools. Its first generation acknowledged `FINAL-B 17`. A
single compact request created an opaque checkpoint, which was saved privately
with its exact canonical prefix and then restored into a new provider instance.

The next generation returned the right answer but failed the intended contract:
the new user turn removed prior reasoning from the ordinary wire view, so the
strict saved wire-prefix check discarded the checkpoint and sent full history.
The historical result was resent as input; no tool executed. This failed report
has not been replaced with the later success.

Source `0db11fa32c86f820458917a4d5679f8548cb508a` repairs that comparison. It
requires the unchanged canonical prefix, configuration and request binding,
reconstructs the original wire prefix and verifies its saved exact digest, then
allows only the normal expiry of old reasoning items. Every remaining covered
item must match exactly. It does not relax text, tool-result or policy checks.
Tests cover the actual transport send, pure budgeting, unchanged state and
rejection of altered canonical history, tool results, text, instructions,
inserted authority and ephemeral messages pretending to be user turns.

[Offline validation](evidence/native-continuity-offline-report.json) used that
same retained checkpoint with a client that raises on any access. All eight
checks passed without network calls. The retained private state hash remained
`48afcfec1e0ff4e593ae2274bf2d50648d59e406a75e8a6efa53456cc14289d4`.

## One continuation from the exact retained state

[The separately authorized final request](evidence/native-continuity-resume-report.json)
restored that original checkpoint into a fresh provider and issued exactly one
generation: no new compact request, tools or retries. Its actual input was the
entire original opaque output, unchanged, followed only by the new user message.
No historical tool result was resent. The response completed with `FINAL-B 17`;
all eleven acceptance checks passed, including unchanged canonical prefix,
checkpoint bytes and saved configuration.

Response identity:
`resp_01d0128c2c529602016ab076c25df487d0b78dbe200b8c13bc`.
Declared usage was 178 input, 39 output and 217 total tokens; these are provider
usage counters, not a billing estimate. The full report records ordered request,
created and completed identities. The private checkpoint remains outside Git.

This establishes actual checkpoint consumption after creating a fresh transport
from saved state. It is a reconstructed continuity fixture, not recovery of the
discarded first steering run. It does not authorize replay after an uncertain
remote outcome or prove the async delegate result from that first run was
consumed. Account availability and other models retain their documented limits.
