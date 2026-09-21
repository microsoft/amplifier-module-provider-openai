"""Opaque provider checkpoints, separate from canonical history and summaries."""

import copy
import hashlib
import json
import math

from amplifier_core.message_models import Message

from . import OpenAIProvider, _read_raw_json_body

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
            # Factories and ephemeral host guidance supply system messages in
            # the actual request without adding them to canonical history.
            # Bind compaction to the last successful request's exact instructions;
            # changing those instructions on the next request invalidates it.
            systems = [m for m in request.messages if m.role == "system"]
            messages = [Message(**m) for m in canonical]
            if systems:
                messages = [*systems, *(m for m in messages if m.role != "system")]
            request = request.model_copy(update={"messages": messages})
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
            usage = data.get("usage") or {}
            # Only declared finite counters cross the public response boundary.
            # Omitted usage remains unknown, never an invented zero-cost call.
            usage = (
                {
                    key: value
                    for key, value in usage.items()
                    if key in {"input_tokens", "output_tokens", "total_tokens"}
                    and type(value) in (int, float)
                    and math.isfinite(value)
                    and value >= 0
                }
                if isinstance(usage, dict)
                else {}
            )
            return {**self.native_status(), "usage": usage}
        finally:
            self._native_busy = False

    def _checkpoint_prefix_length(self, params, canonical):
        """Locate the exact covered wire prefix, allowing normal reasoning expiry.

        A new user turn expires prior reasoning in the regular serializer. The
        canonical prefix is still identical, but its wire item count shrinks.
        Reconstruct the original wire view and verify its saved digest before
        permitting only that deletion; never relax text, tool or policy checks.
        """
        record = self._checkpoint
        if not record:
            return None
        count, wire_count = record["sourceCount"], record["wireCount"]
        if not (
            len(canonical) >= count
            and digest(canonical[:count]) == record["sourceRevision"]
            and self._config_digest() == record["configDigest"]
            and binding(params) == record["requestBinding"]
        ):
            return None
        if digest(params["input"][:wire_count]) == record["wireRevision"]:
            return wire_count
        if self.reasoning_replay_scope != "turn" or not any(
            message.get("role") == "user"
            and not (message.get("metadata") or {}).get("ephemeral")
            for message in canonical[count:]
        ):
            return None

        # Bypass native lineage/delta selection. This isolated base serializer
        # neither mutates live call attribution nor emits planning diagnostics.
        planner = copy.copy(self)
        planner._native_call_ids = set(self._native_call_ids)
        planner._native_call_types = dict(self._native_call_types)
        planner._assembly_log_records = []
        original = OpenAIProvider._convert_messages(planner, canonical[:count])
        if len(original) != wire_count or digest(original) != record["wireRevision"]:
            return None
        covered = [item for item in original if item.get("type") != "reasoning"]
        if len(covered) == wire_count:
            return None
        if digest(params["input"][: len(covered)]) != digest(covered):
            return None
        return len(covered)

    def _checkpoint_matches(self, params, canonical):
        return self._checkpoint_prefix_length(params, canonical) is not None

    def _plan_checkpoint(self, params, canonical):
        # Pure preflight: do not mutate provider lineage or discard state while
        # a fitter is exploring candidate request windows.
        prefix_length = self._checkpoint_prefix_length(params, canonical)
        if prefix_length is None:
            return params
        return {
            **params,
            "input": copy.deepcopy(self._checkpoint["output"])
            + params["input"][prefix_length:],
        }

    def _apply_checkpoint(self, params, full_params):
        if not self._checkpoint:
            return params
        if not self._checkpoint_matches(full_params, self.full_messages):
            self._discard_checkpoint("history_or_configuration_changed")
            return full_params
        if not self.previous_response_id:
            # Preserve EVERY returned item unchanged, followed only by the new
            # suffix. No inspection, reconstruction, or synthetic summary.
            params = self._plan_checkpoint(full_params, self.full_messages)
        return params
