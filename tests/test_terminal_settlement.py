"""Primary outcomes survive optional terminal delivery and usage settlement."""

import asyncio
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from amplifier_core.message_models import ChatRequest, Message

from amplifier_module_provider_openai import OpenAIProvider
from amplifier_module_provider_openai._generation_errors import (
    RequestOutcomeUnknownError,
)


def vendor_response(*, incomplete=False, usage=True):
    return SimpleNamespace(
        model="gpt-6-astra", service_tier="default", output=[],
        status="incomplete" if incomplete else "completed", id="fixture",
        incomplete_details=SimpleNamespace(reason="max_output_tokens"),
        usage=(SimpleNamespace(
            input_tokens=100, output_tokens=10,
            input_tokens_details=SimpleNamespace(cached_tokens=0, cache_write_tokens=0),
            output_tokens_details=SimpleNamespace(reasoning_tokens=2),
        ) if usage else None),
    )


def provider():
    p = OpenAIProvider(api_key="offline-placeholder", config={
        "default_model": "gpt-6-astra", "use_streaming": False,
        "max_concurrent_requests": 0, "max_retries": 1,
    })
    p._guard_assembled_params_with_provider_count = AsyncMock(return_value=100)
    p._add_cost = MagicMock()
    return p


def request():
    return ChatRequest(messages=[Message(role="user", content="fixture")])


@pytest.mark.asyncio
async def test_cancelled_continuation_retains_measured_usage():
    p = provider()
    accepted = asyncio.Event()
    original_cancellation = []

    async def create(**_):
        if not accepted.is_set():
            accepted.set()
            return vendor_response(incomplete=True)
        pending.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError as cancellation:
            original_cancellation.append(cancellation)
            raise

    pending = asyncio.Event()
    p.client.responses.create = AsyncMock(side_effect=create)
    events = []

    async def emit(name, data):
        events.append((name, data))

    p.coordinator = SimpleNamespace(hooks=SimpleNamespace(emit=emit))
    task = asyncio.create_task(p.complete(request()))
    await asyncio.wait_for(pending.wait(), 1)
    task.cancel("caller-stop")
    with pytest.raises(asyncio.CancelledError, match="caller-stop") as caught:
        await task
    assert task.cancelling() == 1
    assert original_cancellation == [caught.value]
    assert original_cancellation[0] is caught.value
    assert p.client.responses.create.await_count == 2
    usage = caught.value.usage
    assert usage["attempts"][0]["input_tokens"] == 100
    assert usage["attempts"][1]["input_tokens"] is None
    assert usage["input_tokens"] is usage["cost_usd"] is None
    assert usage["cost_known_subtotal_usd"] == "0.0015"
    assert usage["cost_complete"] is False
    p._add_cost.assert_called_once_with(Decimal("0.0015"))
    terminal = [data for name, data in events if name == "llm:response"]
    assert len(terminal) == 1 and terminal[0]["status"] == "cancelled"
    assert terminal[0]["usage"] == usage
    await p.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("initial_outcome", ["cancel", "error"])
async def test_new_caller_cancel_during_cleanup_takes_precedence(initial_outcome):
    from amplifier_module_provider_openai._terminal_settlement import _settle_optional

    entered = asyncio.Event()
    drained = asyncio.Event()
    primary = RequestOutcomeUnknownError(provider="openai")

    async def hook():
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            drained.set()

    async def call():
        try:
            if initial_outcome == "error":
                raise primary
            ready.set()
            await asyncio.Event().wait()
        except (Exception, asyncio.CancelledError):
            await _settle_optional(hook)
            raise

    ready = asyncio.Event()
    before = asyncio.all_tasks()
    task = asyncio.create_task(call())
    if initial_outcome == "cancel":
        await ready.wait()
        task.cancel("first-stop")
    await entered.wait()
    task.cancel("new-stop")
    with pytest.raises(asyncio.CancelledError, match="new-stop"):
        await task
    assert task.cancelling() == (2 if initial_outcome == "cancel" else 1)
    assert drained.is_set()
    assert not asyncio.all_tasks() - before


@pytest.mark.asyncio
async def test_cancel_after_normal_cost_commit_does_not_commit_twice():
    p = provider()
    p.client.responses.create = AsyncMock(return_value=vendor_response())
    terminal_entered = asyncio.Event()

    async def emit(name, data):
        if name == "llm:response" and data["status"] == "ok":
            terminal_entered.set()
            await asyncio.Event().wait()

    p.coordinator = SimpleNamespace(hooks=SimpleNamespace(emit=emit))
    task = asyncio.create_task(p.complete(request()))
    await terminal_entered.wait()
    task.cancel("caller-stop")
    with pytest.raises(asyncio.CancelledError, match="caller-stop") as caught:
        await task
    p._add_cost.assert_called_once_with(Decimal("0.0015"))
    assert caught.value.usage["cost_commit"] == "committed"
    assert caught.value.usage["cost_complete"] is True
    await p.close()


@pytest.mark.asyncio
async def test_new_cancel_during_usage_notification_retains_receipt():
    p = provider()
    pending, notifying = asyncio.Event(), asyncio.Event()

    async def create(**_):
        if p.client.responses.create.await_count == 1:
            return vendor_response(incomplete=True)
        pending.set()
        await asyncio.Event().wait()

    p.client.responses.create = AsyncMock(side_effect=create)

    async def emit(name, data):
        if name == "llm:response" and data["status"] == "cancelled":
            notifying.set()
            await asyncio.Event().wait()

    p.coordinator = SimpleNamespace(hooks=SimpleNamespace(emit=emit))
    task = asyncio.create_task(p.complete(request()))
    await pending.wait()
    task.cancel("first-stop")
    await notifying.wait()
    task.cancel("second-stop")
    with pytest.raises(asyncio.CancelledError, match="second-stop") as caught:
        await task
    assert task.cancelling() == 2
    assert caught.value.usage["attempts"][0]["input_tokens"] == 100
    assert caught.value.usage["attempts"][1]["input_tokens"] is None
    p._add_cost.assert_called_once_with(Decimal("0.0015"))
    await p.close()


@pytest.mark.asyncio
async def test_accounting_receipt_unavailable_does_not_replace_primary():
    p = provider()
    p.client.responses.create = AsyncMock(side_effect=RuntimeError("fixture-wire"))
    p._account_failed_responses = MagicMock(side_effect=ValueError("fixture-receipt"))
    with pytest.raises(RequestOutcomeUnknownError) as caught:
        await p.complete(request())
    assert caught.value.usage is None
    assert p.client.responses.create.await_count == 1
    await p.close()


@pytest.mark.asyncio
async def test_external_cancellation_from_accounting_still_propagates():
    p = provider()
    failed = vendor_response()

    def callback(_):
        asyncio.current_task().cancel("actual-caller")
        raise asyncio.CancelledError("actual-caller")

    p._add_cost = callback

    async def settle():
        p._account_failed_responses([failed])

    task = asyncio.create_task(settle())
    with pytest.raises(asyncio.CancelledError, match="actual-caller"):
        await task
    assert task.cancelling() == 1
    await p.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("secondary", ["calculator", "callback"])
@pytest.mark.parametrize("exception", [ValueError, asyncio.CancelledError])
async def test_accounting_exception_preserves_primary_provider_error(secondary, exception):
    p = provider()
    primary = RequestOutcomeUnknownError(provider=p.name)

    async def create(**_):
        if p.client.responses.create.await_count == 1:
            return vendor_response(incomplete=True)
        raise primary

    p.client.responses.create = AsyncMock(side_effect=create)
    if secondary == "calculator":
        p._compute_attempt_cost = MagicMock(side_effect=exception("private-pricing"))
    else:
        p._add_cost = MagicMock(side_effect=exception("private-callback"))
    events = []

    async def emit(name, data):
        events.append((name, data))

    p.coordinator = SimpleNamespace(hooks=SimpleNamespace(emit=emit))
    with pytest.raises(RequestOutcomeUnknownError) as caught:
        await p.complete(request())
    assert caught.value is primary
    assert primary.request_outcome == "unknown" and primary.retryable is False
    assert p.client.responses.create.await_count == 2
    assert primary.usage["attempts"][0]["input_tokens"] == 100
    if secondary == "calculator":
        assert primary.usage["cost_known_subtotal_usd"] is None
        p._add_cost.assert_not_called()
    else:
        p._add_cost.assert_called_once()
        assert primary.usage["cost_commit"] == "unknown"
    terminal = [data for name, data in events if name == "llm:response"]
    assert len(terminal) == 1 and terminal[0]["status"] == "error"
    assert terminal[0]["usage"] == primary.usage
    assert "private-" not in repr(terminal)
    await p.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "error", "cancel"])
@pytest.mark.parametrize("behavior", ["raise", "internal_cancel", "block"])
async def test_terminal_progress_cannot_replace_primary(outcome, behavior):
    p = provider()
    p.use_streaming = True
    accepted = asyncio.Event()
    events = []
    flushes = 0
    primary = RequestOutcomeUnknownError(provider=p.name)

    async def emit(name, data):
        nonlocal flushes
        events.append((name, data))
        if name == "llm:progress" and data["observation"] == "response_activity":
            flushes += 1
            if flushes == 2:
                if behavior == "raise":
                    raise ValueError("private-hook")
                if behavior == "internal_cancel":
                    raise asyncio.CancelledError("hook-not-caller")
                await asyncio.Event().wait()

    class Stream:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

        def __aiter__(self):
            async def events():
                yield SimpleNamespace(type="response.created")
                yield SimpleNamespace(type="response.in_progress")
                accepted.set()
                if outcome == "cancel":
                    await asyncio.Event().wait()
                if outcome == "error":
                    raise primary
            return events()

        get_final_response = AsyncMock(return_value=vendor_response())

    p.client.responses.stream = MagicMock(return_value=Stream())
    p.coordinator = SimpleNamespace(hooks=SimpleNamespace(emit=emit))
    before = asyncio.all_tasks()
    task = asyncio.create_task(p.complete(request()))
    if outcome == "cancel":
        await asyncio.wait_for(accepted.wait(), 1)
        task.cancel("caller-stop")
        with pytest.raises(asyncio.CancelledError, match="caller-stop"):
            await asyncio.wait_for(task, 0.7)
        assert task.cancelling() == 1
    elif outcome == "error":
        with pytest.raises(RequestOutcomeUnknownError) as caught:
            await asyncio.wait_for(task, 0.7)
        assert caught.value is primary
    else:
        result = await asyncio.wait_for(task, 0.7)
        assert result.usage.input_tokens == 100
    assert flushes == 2
    assert p.client.responses.stream.call_count == 1
    assert not asyncio.all_tasks() - before
    await p.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("site", ["assembly", "count"])
async def test_local_failure_before_dispatch_is_original_type(site):
    p = provider()
    original = TypeError("fixture-local")
    if site == "assembly":
        p._assemble_initial_responses_params = MagicMock(side_effect=original)
    else:
        p._guard_assembled_params_with_provider_count = AsyncMock(side_effect=original)
    p.client.responses.create = AsyncMock()
    with pytest.raises(TypeError) as caught:
        await p.complete(request())
    assert caught.value is original
    assert not hasattr(original, "request_outcome")
    p.client.responses.create.assert_not_called()
    p._add_cost.assert_not_called()
    await p.close()


@pytest.mark.asyncio
async def test_concurrent_logical_cancellations_have_separate_usage():
    p = provider()
    calls = {}
    pending = {"one": asyncio.Event(), "two": asyncio.Event()}

    async def create(**params):
        key = str(params["input"][-1]["content"])
        label = "one" if "one" in key else "two"
        calls[label] = calls.get(label, 0) + 1
        if calls[label] == 1:
            result = vendor_response(incomplete=True)
            result.usage.input_tokens = 100 if label == "one" else 200
            return result
        pending[label].set()
        await asyncio.Event().wait()

    p.client.responses.create = AsyncMock(side_effect=create)
    tasks = {label: asyncio.create_task(p.complete(ChatRequest(
        messages=[Message(role="user", content=label)]))) for label in pending}
    await asyncio.wait_for(asyncio.gather(*(event.wait() for event in pending.values())), 1)
    for label, task in tasks.items():
        task.cancel(label)
    errors = await asyncio.gather(*tasks.values(), return_exceptions=True)
    assert all(isinstance(error, asyncio.CancelledError) for error in errors)
    # gather intentionally replaces cancelled exceptions; retrieve each task's
    # original exception to verify the receipt attached at the provider boundary.
    for label, task in tasks.items():
        with pytest.raises(asyncio.CancelledError) as caught:
            task.result()
        assert caught.value.usage["attempts"][0]["input_tokens"] == (100 if label == "one" else 200)
        assert caught.value.usage["attempts"][1]["input_tokens"] is None
    assert calls == {"one": 2, "two": 2}
    assert p._add_cost.call_count == 2
    await p.close()


@pytest.mark.asyncio
async def test_native_core_callback_cleanup_is_not_provider_owned():
    from contextvars import ContextVar

    from amplifier_core import HookResult, ModuleCoordinator

    from amplifier_module_provider_openai._terminal_settlement import _settle_optional

    coordinator = ModuleCoordinator()
    ready, entered, cleanup, release, finish, drained = (asyncio.Event() for _ in range(6))
    context = ContextVar("native_terminal_context", default=None)
    seen, original = [], []

    async def hook(*_):
        seen.append(context.get())
        entered.set()
        try:
            await release.wait()
        finally:
            cleanup.set()
            await finish.wait()
            seen.append(context.get())
            drained.set()
        return HookResult()

    coordinator.hooks.register("fixture:terminal", hook)

    async def caller():
        context.set("original-context")
        ready.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError as error:
            original.append(error)
            await _settle_optional(lambda: coordinator.hooks.emit("fixture:terminal", {}))
            raise

    task = asyncio.create_task(caller())
    try:
        await ready.wait()
        task.cancel("original-stop")
        with pytest.raises(asyncio.CancelledError, match="original-stop") as caught:
            await asyncio.wait_for(task, 2)
        assert caught.value is original[0] and task.cancelling() == 1
        assert entered.is_set() and seen == ["original-context"]
        assert not drained.is_set()
    finally:
        release.set()
        finish.set()
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        # The fixture releases its own callback, not a foreign task drain.
        if entered.is_set():
            await asyncio.wait_for(drained.wait(), 2)
    assert drained.is_set() and seen == ["original-context", "original-context"]
    assert cleanup.is_set()