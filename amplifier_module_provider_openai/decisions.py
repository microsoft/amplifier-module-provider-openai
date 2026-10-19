"""Optional Decisions capability, owned by the configured OpenAI provider.

Only enabled mounts import the small shared spike contract. Authentication,
account endpoint and transport come from the existing provider client. Chat
model selection and reasoning settings are deliberately separate from Decisions.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import math
import re
import time

from openai import APIError, APIStatusError


def decision_info():
    return {
        "schema_version": 1,
        "provider": "openai",
        "available": True,
        "requires_explicit_enable": True,
        "kinds": ["predicate", "choice", "score"],
        "images": True,
        "models": ["gpt-6-luna"],
        "configuration_key": "decisions",
    }


def probability(value):
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ValueError("invalid_probability")
    return float(value)


class OpenAIDecisionBackend:
    def __init__(self, provider, config, coordinator=None):
        try:
            from amplifier_decisions.contracts import Capabilities
        except ImportError as exc:
            raise ValueError(
                "The development Decisions capability requires the shared decision contract package"
            ) from exc
        model = config.get("model")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("Decisions requires an explicit model")
        self.provider, self.model, self.coordinator = provider, model, coordinator
        self.capabilities = Capabilities(images=True)
        self.closed = False

    def body(self, request):
        if request.allow_external_state is not True:
            raise ValueError("external_state_disabled")
        if (
            type(request.timeout_seconds) not in (int, float)
            or not 0.01 <= request.timeout_seconds <= 120
        ):
            raise ValueError("invalid_timeout")
        if not isinstance(request.state, (str, dict, list)):
            raise TypeError("invalid_state")
        caps = self.capabilities
        if not 1 <= len(request.questions) <= caps.max_questions:
            raise ValueError("question_limit")
        if len({q.name for q in request.questions}) != len(request.questions):
            raise ValueError("duplicate_questions")
        questions = []
        for question in request.questions:
            if not isinstance(question.name, str) or not re.fullmatch(
                r"[A-Za-z0-9_.-]{1,100}", question.name
            ):
                raise ValueError("invalid_name")
            if (
                question.kind not in caps.kinds
                or not isinstance(question.instructions, str)
                or not question.instructions.strip()
            ):
                raise ValueError("unsupported_question")
            row = {
                "name": question.name,
                "type": question.kind,
                "instructions": question.instructions,
            }
            if question.kind == "choice":
                if not 2 <= len(question.choices) <= caps.max_choices or any(
                    not isinstance(k, str) or not k or not isinstance(v, str)
                    for k, v in question.choices.items()
                ):
                    raise ValueError("invalid_choices")
                row["choices"] = [
                    {"value": k, "description": v} for k, v in question.choices.items()
                ]
            elif question.choices:
                raise ValueError("unexpected_choices")
            if question.kind == "score":
                if not 2 <= len(question.levels) <= caps.max_levels or any(
                    not isinstance(v, str) or not v for v in question.levels
                ):
                    raise ValueError("invalid_levels")
                row["levels"] = [
                    {"label": str(i), "description": v}
                    for i, v in enumerate(question.levels)
                ]
            elif question.levels:
                raise ValueError("unexpected_levels")
            questions.append(row)
        state = (
            request.state
            if isinstance(request.state, str)
            else json.dumps(request.state, ensure_ascii=False, allow_nan=False)
        )
        if (
            len(json.dumps([state, questions], allow_nan=False).encode())
            > caps.max_text_bytes
        ):
            raise ValueError("text_limit")
        if len(request.images) > caps.max_images:
            raise ValueError("image_limit")
        if request.images:
            from PIL import Image

            parts = [{"type": "input_text", "text": state}]
            total_bytes = 0
            for image in request.images:
                if (
                    image.media_type not in {"image/png", "image/jpeg"}
                    or not 0 < len(image.data) <= 16 * 1024**2
                ):
                    raise ValueError("invalid_image")
                total_bytes += len(image.data)
                if total_bytes > 24 * 1024**2:
                    raise ValueError("body_limit")
                with Image.open(io.BytesIO(image.data)) as opened:
                    if (
                        opened.format
                        != {"image/png": "PNG", "image/jpeg": "JPEG"}[image.media_type]
                    ):
                        raise ValueError("image_format")
                    opened.verify()
                data = base64.b64encode(image.data).decode("ascii")
                parts.append(
                    {
                        "type": "input_image",
                        "image_url": f"data:{image.media_type};base64,{data}",
                    }
                )
            state = [{"role": "user", "content": parts}]
        return {"model": self.model, "input": state, "questions": questions}

    def answer(self, question, row):
        from amplifier_decisions.contracts import Answer

        try:
            if row.get("type") == "refusal":
                return Answer(
                    question.name, question.kind, "refused", reason="provider_refusal"
                )
            if row.get("type") != question.kind:
                raise ValueError("answer_type")
            if question.kind == "predicate":
                return Answer(
                    question.name,
                    question.kind,
                    "answered",
                    probability(row["probability"]),
                )
            confidence = probability(row["confidence"])
            raw = row["probabilities"]
            if not isinstance(raw, list):
                raise TypeError("probability_shape")
            pairs = [
                (str(item["value"]), probability(item["probability"])) for item in raw
            ]
            probabilities = dict(pairs)
            keys = (
                set(question.choices)
                if question.kind == "choice"
                else {str(i) for i in range(len(question.levels))}
            )
            if (
                len(probabilities) != len(pairs)
                or set(probabilities) != keys
                or abs(sum(probabilities.values()) - 1) > 0.001
            ):
                raise ValueError("probability_distribution")
            if question.kind == "choice":
                value = row["choice"]
                if (
                    not isinstance(value, str)
                    or value not in keys
                    or probabilities[value] < max(probabilities.values()) - 0.000001
                ):
                    raise ValueError("invalid_choice")
            else:
                value = row["score"]
                if (
                    type(value) not in (int, float)
                    or not math.isfinite(value)
                    or not 0 <= value <= len(keys) - 1
                ):
                    raise ValueError("invalid_score")
                if (
                    abs(value - sum(int(k) * p for k, p in probabilities.items()))
                    > 0.001
                ):
                    raise ValueError("inconsistent_score")
            return Answer(
                question.name,
                question.kind,
                "answered",
                value,
                probabilities,
                confidence,
            )
        except (KeyError, TypeError, ValueError, AttributeError):
            return Answer(
                question.name, question.kind, "invalid", reason="invalid_answer"
            )

    async def decide(self, request):
        from amplifier_decisions.contracts import Answer, DecisionResult

        start = time.monotonic()

        def result(answers, payload=None, request_id=None):
            payload = payload or {}
            usage = payload.get("usage", {})
            usage = (
                {
                    k: v
                    for k, v in usage.items()
                    if k in {"input_tokens", "output_tokens", "total_tokens"}
                    and type(v) is int
                    and v >= 0
                }
                if isinstance(usage, dict)
                else {}
            )
            model = payload.get("model")
            return DecisionResult(
                "openai-decisions",
                self.model,
                model if isinstance(model, str) else None,
                answers,
                round((time.monotonic() - start) * 1000, 3),
                usage,
                request_id,
            )

        def failed(status, reason):
            return result(
                {
                    q.name: Answer(q.name, q.kind, status, reason=reason)
                    for q in request.questions
                }
            )

        if self.closed:
            return failed("unavailable", "closed")
        try:
            body = self.body(request)
        except (ValueError, TypeError, RecursionError, OSError, SyntaxError):
            return failed("invalid", "invalid_or_unsupported_request")
        if self.coordinator:
            check = await self.coordinator.hooks.emit(
                "decision:request",
                {
                    "provider": "openai",
                    "model": self.model,
                    "questions": len(request.questions),
                },
            )
            check = await self.coordinator.process_hook_result(
                check, "decision:request", "provider-openai"
            )
            if check.action == "deny":
                return failed("unavailable", "host_denied")
        try:
            async with asyncio.timeout(request.timeout_seconds):
                # with_options retains the provider's account, API key, base URL,
                # headers and transport. No second credential lookup/client pool.
                client = self.provider.client.with_options(
                    timeout=request.timeout_seconds, max_retries=0
                )
                response = await client.decisions.create(**body)
            payload = response.model_dump()
            rows = payload.get("answers")
            if not isinstance(rows, list) or any(
                not isinstance(row, dict) or not isinstance(row.get("name"), str)
                for row in rows
            ):
                return failed("invalid", "answer_envelope")
            by_name = {row["name"]: row for row in rows}
            if len(by_name) != len(rows) or set(by_name) - {
                q.name for q in request.questions
            }:
                return failed("invalid", "answer_envelope")
            value = result(
                {
                    q.name: self.answer(q, by_name.get(q.name))
                    for q in request.questions
                },
                payload,
                getattr(response, "_request_id", None),
            )
            if self.coordinator:
                await self.coordinator.hooks.emit(
                    "decision:response",
                    {
                        "provider": "openai",
                        "model": value.reported_model,
                        "usage": value.usage,
                        "request_id": value.request_id,
                        "elapsed_ms": value.elapsed_ms,
                    },
                )
            return value
        except TimeoutError:
            return failed("unavailable", "timeout")
        except APIStatusError as exc:
            return failed(
                "invalid" if exc.status_code in {400, 413, 415, 422} else "unavailable",
                f"http_{exc.status_code}",
            )
        except APIError:
            return failed("unavailable", "transport_error")
        except (ValueError, TypeError, AttributeError):
            return failed("invalid", "response_shape")

    async def close(self):
        # The provider owns the shared client's lifetime.
        self.closed = True


def register_decision_backend(coordinator, provider, config):
    row = config.get("decisions") or {}
    if row.get("enabled") is not True:
        return lambda: None
    identity = row.get("id", "openai-decisions")
    backends = dict(coordinator.get_capability("decision.backends") or {})
    if not isinstance(identity, str) or not identity or identity in backends:
        raise ValueError("Decision backend identifiers must be unique")
    backend = OpenAIDecisionBackend(provider, row, coordinator)
    backends[identity] = backend
    coordinator.register_capability("decision.backends", backends)
    if hasattr(coordinator, "register_contributor"):
        coordinator.register_contributor(
            "observability.events",
            "provider-openai-decisions",
            lambda: ["decision:request", "decision:response"],
        )

    def remove():
        backend.closed = True
        current = dict(coordinator.get_capability("decision.backends") or {})
        if current.get(identity) is backend:
            del current[identity]
            coordinator.register_capability("decision.backends", current)

    return remove
