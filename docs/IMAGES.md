# Optional Images API backend

The provider can register an image backend independently of its chat model. It
is disabled by default. Configuration belongs to the provider's existing account
and endpoint, using the same ordinary API credential:

```yaml
image_generation:
  enabled: true
  id: images
  model: YOUR_CHOSEN_IMAGE_MODEL
  timeout: 180
```

No model is selected by default. `get_info()` and chat model capabilities remain
unchanged: vision input support never implies Images API access. Account/project
entitlement and billing are validated only by the actual API request. Subscription
service credentials from other applications are not an Images API credential.

When enabled, the provider registers its explicit `id` in coordinator capability
`image.backends`. Consumers call `describe()` for configuration readiness and
`generate(action, prompt, images, size, quality, background)` for one image.
`images` contains already-read bytes, names and MIME types. Generation calls the
Images API generation endpoint; edits send those exact inputs to its edit endpoint.
It returns PNG bytes, provider request ID and provider-reported usage when present.
The backend does not own file access, tool authorization, artifacts or UI behavior.

Automatic SDK retries are disabled for these paid operations. The consumer must
persist its effect receipt before requesting work and reconcile uncertain results
instead of silently retrying. No image URLs are fetched, and there is no fallback
to chat generation, a different model, or a different account. Multiple configured
instances need distinct image backend IDs; duplicate IDs fail clearly.

The public API contract follows the official
[image generation guide](https://developers.openai.com/api/docs/guides/image-generation)
and [image edit reference](https://developers.openai.com/api/reference/resources/images/methods/edit).
Current support is one PNG per call with ordinary size, quality and background
options. Additional model-specific controls require separate qualification.
