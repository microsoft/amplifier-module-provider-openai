# Native computer screenshot validation

Native computer call outputs require image evidence. A tool safety halt, permission denial, text result, or malformed screenshot must never be labelled as base64 PNG merely because its tool content is a string.

The provider validates the native result before constructing `computer_call_output`. It accepts a raw base64 PNG string, a base64 image data URL, or exactly one base64 image block. It bounds encoded/decoded size at 20 MiB of image bytes, checks supported MIME/container agreement (PNG/JPEG/GIF/WebP), and limits dimensions to 40 million pixels. PNG validation also checks chunk boundaries, CRCs, and a bounded complete compressed pixel stream. Other formats receive bounded container/header checks, not a general-purpose full image decoder. Original encoded image bytes and MIME are preserved.

Results without valid image evidence raise a local `InvalidRequestError`, with `retryable=False`, `code=computer_result_not_image`, the tool call ID, and a safe result-kind classification. Arbitrary tool text and image data are omitted from the diagnostic. The original tool result remains unchanged in history. No provider request with fabricated pixels is sent, and the provider never repeats the computer action or clears its safety state.

This is a protocol correctness boundary. It does not introduce automatic recovery from a native-call safety halt. Continuing a transcript whose native call lacks an image needs an explicit, separately supported recovery path; it must not infer that interrupted external work was undone or safe to replay.
