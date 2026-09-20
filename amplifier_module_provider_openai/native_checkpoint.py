"""Opaque provider checkpoints, separate from canonical history and summaries."""

import copy
import hashlib
import json

from amplifier_core.message_models import Message

from . import _read_raw_json_body

FORMAT = "openai.responses.compact.v1"
MAX_CHECKPOINT_BYTES = 16 * 1024 * 1024


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def binding(params):
    return digest(
        {
            k: v
            for k, v in params.items()
            if k
            not in {
                "input",
                "stream",
                "store",
                "max_output_tokens",
                "previous_response_id",
            }
        }
    )


class NativeCheckpointMixin:
    def native_status(self):
        return {
            "supported": True,
            "transport": "responses_websocket",
            "model": getattr(self, "_last_native_request", (None, {}))[1].get(
                "model", self.default_model
            ),
            "observerFailed": bool(self._observer_failure),
            "active": self.owner is not None,
            "outcome": "running"
            if self.owner is not None
            else "unknown"
            if self.request_uncertain
            else "pending"
            if self.pending_parent
            else "settled",
            "checkpoint": self._checkpoint_status,
            "format": FORMAT,
            "compactAvailable": hasattr(self, "_last_native_request")
            and not self.owner
            and not self.request_uncertain
            and not self.pending_parent,
            "automaticCompaction": False,
        }

    def _config_digest(self):
        # Only a digest crosses this boundary; secrets never enter receipts.
        return digest(self.config)

    def native_export_checkpoint(self):
        return copy.deepcopy(self._checkpoint)

    def _discard_checkpoint(self, reason):
        self._checkpoint = None
        self._checkpoint_status = "discarded:" + reason
        # A server-side cache must not retain rejected derived state.
        self.previous_response_id = None
        self.seen.clear()

    def native_restore_checkpoint(self, record, *, canonical, identity):
        if self.owner or self.request_uncertain or self.pending_parent:
            raise ValueError("Cannot restore while native work is active or uncertain")
        try:
            if len(json.dumps(record).encode()) > MAX_CHECKPOINT_BYTES:
                raise ValueError("size")
            if record["format"] != FORMAT or record["identity"] != identity:
                raise ValueError("identity")
            if record["configDigest"] != self._config_digest():
                raise ValueError("configuration")
            count = record["sourceCount"]
            if not isinstance(count, int) or count < 0 or count > len(canonical):
                raise ValueError("prefix")
            if digest(canonical[:count]) != record["sourceRevision"]:
                raise ValueError("prefix")
            if (
                digest({k: v for k, v in record.items() if k != "recordDigest"})
                != record["recordDigest"]
            ):
                raise ValueError("digest")
            if not isinstance(record["output"], list) or not any(
                isinstance(i, dict)
                and i.get("type") == "compaction"
                and isinstance(i.get("encrypted_content"), str)
                and i["encrypted_content"]
                for i in record["output"]
            ):
                raise ValueError("opaque_output")
        except (KeyError, ValueError, TypeError):
            self._discard_checkpoint("incompatible")
            return self.native_status()
        self._checkpoint = copy.deepcopy(record)
        self._checkpoint_status = "restored"
        self.previous_response_id = None
        self.seen.clear()
        return self.native_status()

    async def native_compact(self, *, canonical, identity):
        if (
            self.owner
            or self._native_busy
            or self.request_uncertain
            or self.pending_parent
        ):
            raise ValueError(
                "Finish active work and resolve unknown outcomes before compacting"
            )
        if not hasattr(self, "_last_native_request"):
            raise ValueError(
                "A successful native request is required to bind its actual model and tool configuration"
            )
        self._native_busy = True
        try:
            request, kwargs = self._last_native_request
            request = request.model_copy(
                update={"messages": [Message(**m) for m in canonical]}
            )
            params, _, _ = self._assemble_initial_responses_params(request, **kwargs)
            if (
                params.get("conversation")
                or params.get("context_management")
                or (params.get("extra_body") or {}).get("context_management")
            ):
                raise ValueError(
                    "Automatic compaction and conversation-bound steering are unsupported"
                )
            api = getattr(
                getattr(self.client.responses, "with_raw_response", None),
                "compact",
                None,
            )
            if api is None:
                raise ValueError(
                    "Installed provider SDK does not expose Responses compaction"
                )
            # Disable SDK retries too: an unknown response is never silently
            # reissued. Compaction returns a derived window, never tool effects.
            client = self.client.with_options(max_retries=0)
            result = await client.responses.with_raw_response.compact(
                **{
                    k: copy.deepcopy(v)
                    for k, v in params.items()
                    if k in {"model", "input", "instructions"}
                }
            )
            data = await _read_raw_json_body(result)
            output = data.get("output") if isinstance(data, dict) else None
            if not isinstance(output, list) or not any(
                isinstance(i, dict)
                and i.get("type") == "compaction"
                and isinstance(i.get("encrypted_content"), str)
                and i["encrypted_content"]
                for i in output
            ):
                raise ValueError("Provider did not return an opaque compaction window")
            record = {
                "format": FORMAT,
                "identity": copy.deepcopy(identity),
                "configDigest": self._config_digest(),
                "sourceCount": len(canonical),
                "sourceRevision": digest(canonical),
                "wireCount": len(params["input"]),
                "wireRevision": digest(params["input"]),
                "requestBinding": binding(params),
                "output": copy.deepcopy(output),
            }
            record["recordDigest"] = digest(record)
            if len(json.dumps(record).encode()) > MAX_CHECKPOINT_BYTES:
                raise ValueError("Provider checkpoint exceeds private storage bound")
            if self.socket:
                await self.socket.close()
                self.socket = None
            self.previous_response_id = None
            self.seen.clear()
            self._checkpoint = record
            self._checkpoint_status = "created"
            return self.native_status()
        finally:
            self._native_busy = False

    def _apply_checkpoint(self, params, full_params):
        record = self._checkpoint
        if not record:
            return params
        count, wire_count = record["sourceCount"], record["wireCount"]
        full = full_params["input"]
        if (
            digest(self.full_messages[:count]) != record["sourceRevision"]
            or self._config_digest() != record["configDigest"]
            or binding(full_params) != record["requestBinding"]
            or digest(full[:wire_count]) != record["wireRevision"]
        ):
            self._discard_checkpoint("history_or_configuration_changed")
            return full_params
        if not self.previous_response_id:
            # Preserve EVERY returned item unchanged, followed only by the new
            # suffix. No inspection, reconstruction, or synthetic summary.
            params = {
                **params,
                "input": copy.deepcopy(record["output"]) + full[wire_count:],
            }
        return params
