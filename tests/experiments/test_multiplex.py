"""Offline unittest qualification for the isolated multiplex experiment."""

import unittest
from pathlib import Path

from examples.openai_advanced.multiplex import Custody, MockSocket, Multiplex, Rejected

MODEL = "opaque-model-capability-unknown"


def progress(stream, response):
    return {
        "type": "response.in_progress",
        "stream_id": stream,
        "response": {"id": response},
    }


def completed(stream, response):
    return {
        "type": "response.completed",
        "stream_id": stream,
        "response": {"id": response},
    }


class MultiplexTests(unittest.TestCase):
    def setUp(self):
        self.socket = MockSocket()
        self.mux = Multiplex(self.socket)
        self.a = Custody("peer-a", "custody-a")
        self.b = Custody("peer-b", "custody-b")
        self.register("a", self.a)
        self.register("b", self.b)

    def register(self, stream, custody, **kwargs):
        label = str(stream)
        options = {
            "workdir": str(Path("/isolated-multiplex-mock") / label),
            "provenance": ("commission:" + label, "source-message:" + label),
            "request_limit": 10,
        }
        options.update(kwargs)
        self.mux.register(stream, custody, **options)

    def submit(self, stream="a", custody=None, request="a1", **kwargs):
        self.mux.submit(
            stream,
            custody or (self.a if stream == "a" else self.b),
            request_id=request,
            model=MODEL,
            input=[{"role": "user", "content": request}],
            **kwargs,
        )

    def request(self, stream, request):
        custody = self.a if stream == "a" else self.b
        return self.mux.snapshot(stream, custody).requests[request]

    def receive(self, *events):
        self.socket.feed(*events)
        for _ in events:
            self.mux.receive()

    def finish_source(self):
        self.submit()
        self.mux.pump()
        self.receive(progress("a", "parent"), completed("a", "parent"))

    def fork(self, **kwargs):
        options = {
            "previous_response_id": "parent",
            "request_id": "b-fork",
            "model": MODEL,
            "input": [],
        }
        options.update(kwargs)
        self.mux.fork("a", self.a, "b", self.b, **options)

    def test_actual_interleaving_demux_and_same_stream_fifo(self):
        self.submit()
        self.submit(request="a2")
        self.submit("b", request="b1")
        self.mux.pump()
        self.assertEqual(
            [frame["input"][0]["content"] for frame in self.socket.sent], ["a1", "b1"]
        )
        self.receive(
            progress("b", "rb"),
            progress("a", "ra"),
            {
                "type": "response.output_text.delta",
                "stream_id": "a",
                "response_id": "ra",
                "delta": "A",
            },
            {
                "type": "response.output_text.delta",
                "stream_id": "b",
                "response_id": "rb",
                "delta": "B",
            },
            {
                "type": "response.output_text.delta",
                "stream_id": "a",
                "response_id": "ra",
                "delta": "A2",
            },
            completed("b", "rb"),
        )
        self.mux.pump()
        self.assertEqual(len(self.socket.sent), 2)  # a2 cannot overlap a1
        self.receive(completed("a", "ra"))
        self.mux.pump()
        self.assertEqual(self.socket.sent[2]["input"][0]["content"], "a2")
        self.assertEqual(
            [e["delta"] for e in self.request("a", "a1").events if "delta" in e],
            ["A", "A2"],
        )
        self.assertEqual(
            [e["delta"] for e in self.request("b", "b1").events if "delta" in e],
            ["B"],
        )
        self.receive(progress("a", "ra2"), completed("a", "ra2"))
        self.assertEqual(self.request("a", "a2").state, "completed")

    def test_mutual_fork_cycle_rejected_without_charging_or_poisoning_queue(self):
        self.finish_source()
        self.submit("b", request="b1")
        self.mux.pump()
        self.receive(progress("b", "b-parent"), completed("b", "b-parent"))
        self.fork()
        before = self.mux.snapshot("a", self.a)
        with self.assertRaisesRegex(Rejected, "cyclic fork"):
            self.mux.fork(
                "b", self.b, "a", self.a, previous_response_id="b-parent",
                request_id="a-fork", model=MODEL, input=[],
            )
        after = self.mux.snapshot("a", self.a)
        self.assertEqual(after.admitted, before.admitted)
        self.assertNotIn("a-fork", after.requests)
        self.mux.pump()
        self.assertEqual(self.request("b", "b-fork").state, "active")
        self.receive(progress("b", "fork-child"), completed("b", "fork-child"))
        self.assertFalse(self.mux.snapshot("a", self.a).fork_waiters)

    def test_only_documented_create_fields_and_no_real_model_claim(self):
        self.submit()
        self.mux.pump()
        self.assertEqual(
            self.socket.sent,
            [
                {
                    "type": "response.create",
                    "stream_id": "a",
                    "model": MODEL,
                    "store": False,
                    "input": [{"role": "user", "content": "a1"}],
                }
            ],
        )

    def test_local_error_preserves_peer_and_queue_fifo(self):
        self.submit()
        self.submit(request="a2")
        self.submit("b", request="b1")
        self.mux.pump()
        self.receive(
            progress("b", "rb"),
            {"type": "error", "stream_id": "a", "error": {"code": "mock_bad_input"}},
        )
        self.assertTrue(self.mux.connected)
        self.assertEqual(self.request("a", "a1").state, "error")
        self.assertEqual(self.request("b", "b1").state, "active")
        self.mux.pump()
        self.assertEqual(self.request("a", "a2").state, "active")
        self.receive(completed("b", "rb"))
        self.assertEqual(self.request("b", "b1").state, "completed")

    def test_local_error_for_idle_or_unowned_lane_never_blames_active_peer(self):
        self.submit("b", request="b1")
        self.mux.pump()
        idle_error = {"type": "error", "stream_id": "a", "error": "idle mock error"}
        unowned_error = {
            "type": "error",
            "stream_id": "not-owned",
            "error": "unowned mock error",
        }
        self.receive(idle_error, unowned_error)
        self.assertTrue(self.mux.connected)
        self.assertTrue(self.mux.snapshot("a", self.a).reconciliation_required)
        self.assertEqual(self.mux.snapshot("a", self.a).errors, [idle_error])
        self.assertEqual(self.mux.unowned_errors(), [unowned_error])
        self.assertEqual(self.request("b", "b1").state, "active")
        self.receive(progress("b", "rb"), completed("b", "rb"))
        self.assertEqual(self.request("b", "b1").state, "completed")

    def test_connection_error_and_close_make_all_active_unknown(self):
        for event in (None, {"type": "error", "error": {"code": "mock_connection"}}):
            with self.subTest(event=event):
                self.setUp()
                self.submit()
                self.submit(request="a2")
                self.submit("b", request="b1")
                self.mux.pump()
                self.receive(event)
                self.assertFalse(self.mux.connected)
                for stream, request in (("a", "a1"), ("b", "b1")):
                    self.assertEqual(self.request(stream, request).state, "unknown")
                    custody = self.a if stream == "a" else self.b
                    self.assertTrue(
                        self.mux.snapshot(stream, custody).reconciliation_required
                    )
                self.assertEqual(self.request("a", "a2").state, "not_sent")
                with self.assertRaises(Rejected):
                    self.mux.pump()
                with self.assertRaises(Rejected):
                    self.submit(request="fresh-id-is-not-recovery")
                self.assertEqual(len(self.socket.sent), 2)

    def test_send_acknowledgement_loss_is_not_replayed(self):
        self.submit()
        self.submit("b", request="b1")
        self.socket.fail_after_send = True
        self.mux.pump()
        self.assertEqual(self.request("a", "a1").state, "unknown")
        self.assertEqual(self.request("b", "b1").state, "not_sent")
        self.assertEqual(len(self.socket.sent), 1)
        with self.assertRaises(Rejected):
            self.mux.pump()
        self.assertEqual(len(self.socket.sent), 1)

    def test_invalid_stream_ids_never_admitted(self):
        for stream in ("", "bad/id", "a b", "é", "x" * 257, None, 3):
            with self.subTest(stream=stream), self.assertRaises(Rejected):
                self.register(stream, self.a, workdir="/invalid-id-mock")
        self.assertEqual(self.socket.sent, [])

    def test_valid_id_boundary_and_default_counts_toward_32(self):
        self.register("X_.-" + "x" * 252, self.a)
        # a, b, boundary id plus reserved default = 4.
        for number in range(28):
            self.register("lane-" + str(number), self.a)
        with self.assertRaises(Rejected):
            self.register("too-many", self.a)
        self.register("default", self.a)
        self.assertEqual(self.socket.sent, [])

    def test_default_stream_has_explicit_echoed_identity(self):
        self.register("default", self.a)
        self.mux.submit("default", self.a, request_id="d1", model=MODEL, input=[])
        self.mux.pump()
        self.receive(progress("default", "rd"), completed("default", "rd"))
        self.assertEqual(self.socket.sent[0]["stream_id"], "default")
        self.assertEqual(
            self.mux.snapshot("default", self.a).requests["d1"].state, "completed"
        )

    def test_wrong_owner_stale_custody_and_duplicate_registration(self):
        for custody in (self.b, Custody("peer-a", "stale")):
            with self.subTest(custody=custody):
                with self.assertRaises(Rejected):
                    self.submit(custody=custody)
                with self.assertRaises(Rejected):
                    self.mux.stop("a", custody)
                with self.assertRaises(Rejected):
                    self.mux.snapshot("a", custody)
        with self.assertRaises(Rejected):
            self.register("a", self.b, workdir="/other-mock-dir")
        self.mux.revoke("a", self.a)
        with self.assertRaises(Rejected):
            self.submit()
        self.submit("b", request="b1")
        self.mux.pump()
        self.assertEqual([f["stream_id"] for f in self.socket.sent], ["b"])

    def test_request_budget_is_per_stream_and_queues_count(self):
        self.register("limited", self.a, request_limit=1)
        self.mux.submit("limited", self.a, request_id="l1", model=MODEL, input=[])
        with self.assertRaises(Rejected):
            self.mux.submit("limited", self.a, request_id="l2", model=MODEL, input=[])
        self.submit("b", request="b1")
        self.mux.pump()
        self.assertEqual({f["stream_id"] for f in self.socket.sent}, {"limited", "b"})
        self.assertEqual(self.mux.snapshot("limited", self.a).admitted, 1)

    def test_stop_fences_only_owner_and_late_success_cannot_undo_unknown(self):
        self.submit()
        self.submit(request="a2")
        self.submit("b", request="b1")
        self.mux.pump()
        self.mux.stop("a", self.a)
        with self.assertRaises(Rejected):
            self.submit(request="a3")
        self.receive(
            progress("a", "ra"),
            completed("a", "ra"),
            progress("b", "rb"),
            completed("b", "rb"),
        )
        self.mux.pump()
        self.assertEqual(self.request("a", "a1").state, "unknown")
        self.assertEqual(self.request("a", "a2").state, "not_sent")
        self.assertEqual(len(self.request("a", "a1").events), 2)
        self.assertEqual(self.request("b", "b1").state, "completed")
        self.assertEqual(len(self.socket.sent), 2)

    def test_provenance_and_workdirs_remain_per_stream_not_wire_fields(self):
        self.submit()
        self.submit("b", request="b1")
        self.mux.pump()
        a = self.request("a", "a1")
        b = self.request("b", "b1")
        self.assertNotEqual(a.workdir, b.workdir)
        self.assertNotEqual(a.provenance, b.provenance)
        self.assertNotEqual(a.custody, b.custody)
        self.assertEqual(a.provenance, ("commission:a", "source-message:a"))
        for frame in self.socket.sent:
            self.assertFalse({"custody", "workdir", "provenance"} & frame.keys())
        snapshot = self.mux.snapshot("a", self.a)
        snapshot.request_limit = 1000
        snapshot.requests["a1"].frame["model"] = "mutated"
        self.assertEqual(self.mux.snapshot("a", self.a).request_limit, 10)
        self.assertEqual(self.request("a", "a1").frame["model"], MODEL)

    def test_shared_nested_or_relative_workdirs_rejected(self):
        root = self.mux.snapshot("a", self.a).workdir
        for workdir in (root, root + "/tools", str(Path(root).parent), "relative"):
            with self.subTest(workdir=workdir), self.assertRaises(Rejected):
                self.register("overlap", self.b, workdir=workdir)

    def test_missing_provenance_and_invalid_budget_rejected(self):
        for options in (
            {"provenance": ()},
            {"provenance": ["not-immutable"]},
            {"request_limit": 0},
            {"request_limit": True},
        ):
            with self.subTest(options=options), self.assertRaises(Rejected):
                self.register("bad", self.a, **options)

    def test_fork_waits_for_target_in_progress_not_created(self):
        self.finish_source()
        self.fork()
        self.submit(request="a2")
        self.mux.pump()
        self.assertEqual([f["stream_id"] for f in self.socket.sent], ["a", "b"])
        self.assertEqual(self.socket.sent[-1]["previous_response_id"], "parent")
        self.receive(
            {"type": "response.created", "stream_id": "b", "response": {"id": "child"}}
        )
        self.mux.pump()
        self.assertEqual(self.request("a", "a2").state, "queued")
        self.receive(progress("b", "child"))
        self.mux.pump()
        self.assertEqual(self.request("a", "a2").state, "active")
        self.assertEqual(self.request("b", "b-fork").state, "active")
        self.assertEqual(self.mux.cached("a", self.a), ())
        self.assertEqual(
            self.request("b", "b-fork").provenance, ("commission:b", "source-message:b")
        )

    def test_fork_requires_source_custody_and_completed_cached_parent(self):
        self.submit()
        self.mux.pump()
        with self.assertRaises(Rejected):
            self.fork()
        self.receive(progress("a", "parent"), completed("a", "parent"))
        with self.assertRaises(Rejected):
            self.mux.fork(
                "a",
                self.b,
                "b",
                self.b,
                previous_response_id="parent",
                request_id="bad",
                model=MODEL,
                input=[],
            )
        with self.assertRaises(Rejected):
            self.submit("b", request="steal", previous_response_id="parent")
        self.submit(request="a2")
        self.mux.pump()  # source advances, evicting parent
        self.receive(progress("a", "new"), completed("a", "new"))
        with self.assertRaises(Rejected):
            self.fork()

    def test_fork_error_before_guard_does_not_unlock_source(self):
        self.finish_source()
        self.fork()
        self.submit(request="a2")
        self.mux.pump()
        self.receive({"type": "error", "stream_id": "b", "error": "mock failure"})
        self.mux.pump()
        self.assertEqual(self.request("a", "a2").state, "queued")
        self.assertTrue(self.mux.snapshot("a", self.a).reconciliation_required)
        self.assertTrue(self.mux.snapshot("b", self.b).reconciliation_required)
        self.assertEqual(len(self.socket.sent), 2)

    def test_queued_fork_barrier_survives_other_target_request(self):
        self.finish_source()
        self.submit("b", request="b1")
        self.mux.pump()
        self.fork()
        self.submit(request="a2")
        self.mux.pump()
        self.receive(progress("b", "rb"), completed("b", "rb"))
        self.mux.pump()
        self.assertEqual(self.request("b", "b-fork").state, "active")
        self.assertEqual(self.request("a", "a2").state, "queued")
        self.receive(progress("b", "forked"))
        self.mux.pump()
        self.assertEqual(self.request("a", "a2").state, "active")

    def test_stop_fork_before_guard_keeps_source_fenced(self):
        for sent in (False, True):
            with self.subTest(sent=sent):
                self.setUp()
                self.finish_source()
                self.fork()
                self.submit(request="a2")
                if sent:
                    self.mux.pump()
                self.mux.stop("b", self.b)
                if sent:
                    self.receive(progress("b", "child"), completed("b", "child"))
                self.mux.pump()
                self.assertTrue(self.mux.snapshot("a", self.a).reconciliation_required)
                self.assertEqual(self.request("a", "a2").state, "queued")
                self.assertEqual(
                    self.request("b", "b-fork").state,
                    "unknown" if sent else "not_sent",
                )

    def test_all_fork_waiters_must_reach_guard(self):
        self.finish_source()
        self.fork()
        self.fork(request_id="b-fork2")
        self.submit(request="a2")
        self.mux.pump()
        self.receive(progress("b", "child1"), completed("b", "child1"))
        self.mux.pump()
        self.assertEqual(self.request("a", "a2").state, "queued")
        self.assertEqual(self.request("b", "b-fork2").state, "active")
        self.receive(progress("b", "child2"))
        self.mux.pump()
        self.assertEqual(self.request("a", "a2").state, "active")

    def test_fork_completion_without_progress_is_not_a_guard(self):
        self.finish_source()
        self.fork()
        self.submit(request="a2")
        self.mux.pump()
        self.receive(completed("b", "child"))
        self.mux.pump()
        self.assertTrue(self.mux.snapshot("a", self.a).reconciliation_required)
        self.assertEqual(self.request("a", "a2").state, "queued")
        self.assertEqual(len(self.socket.sent), 2)

    def test_owned_continuation_and_stale_queued_cache_reference(self):
        self.finish_source()
        self.submit(request="a2", previous_response_id="parent")
        self.submit(request="a3", previous_response_id="parent")
        self.mux.pump()
        self.assertEqual(self.socket.sent[-1]["previous_response_id"], "parent")
        self.receive(progress("a", "ra2"), completed("a", "ra2"))
        self.mux.pump()
        self.assertEqual(self.request("a", "a3").state, "queued")
        self.assertTrue(self.mux.snapshot("a", self.a).reconciliation_required)
        self.assertIn("evicted", self.request("a", "a3").reason)
        self.assertEqual(len(self.socket.sent), 2)

    def test_close_loses_all_caches_and_cannot_reuse_on_fresh_connection(self):
        self.finish_source()
        self.submit("b", request="b1")
        self.mux.pump()
        self.receive(progress("b", "rb"), completed("b", "rb"))
        self.assertEqual(self.mux.cached("a", self.a), ("parent",))
        self.assertEqual(self.mux.cached("b", self.b), ("rb",))
        self.receive(None)
        self.assertEqual(self.mux.cached("a", self.a), ())
        self.assertEqual(self.mux.cached("b", self.b), ())
        self.mux = Multiplex(MockSocket())
        self.register("a", self.a)
        self.register("b", self.b)
        with self.assertRaises(Rejected):
            self.fork()
        self.assertEqual(self.mux.socket.sent, [])

    def test_duplicate_request_is_not_replay_and_input_is_snapshotted(self):
        data = [{"role": "user", "content": "original"}]
        self.mux.submit("a", self.a, request_id="r", model=MODEL, input=data)
        data[0]["content"] = "changed"
        with self.assertRaises(Rejected):
            self.mux.submit("a", self.a, request_id="r", model=MODEL, input=data)
        self.mux.pump()
        self.assertEqual(self.socket.sent[0]["input"][0]["content"], "original")
        self.receive(progress("a", "ra"), completed("a", "ra"))
        with self.assertRaises(Rejected):
            self.mux.submit("a", self.a, request_id="r", model=MODEL, input=data)
        self.assertEqual(len(self.socket.sent), 1)

    def test_same_local_request_id_is_scoped_to_owned_stream(self):
        self.submit(request="same")
        self.submit("b", request="same")
        self.mux.pump()
        self.receive(
            progress("a", "ra"),
            progress("b", "rb"),
            completed("b", "rb"),
            completed("a", "ra"),
        )
        self.assertEqual(self.request("a", "same").state, "completed")
        self.assertEqual(self.request("b", "same").state, "completed")

    def test_invalid_json_does_not_charge_budget_or_poison_queue(self):
        for data in ([object()], [float("nan")]):
            with self.subTest(data=data), self.assertRaises(Rejected):
                self.mux.submit("a", self.a, request_id="bad", model=MODEL, input=data)
        self.assertEqual(self.mux.snapshot("a", self.a).admitted, 0)
        self.assertFalse(self.mux.snapshot("a", self.a).queue)
        self.assertEqual(self.socket.sent, [])

    def test_cross_stream_response_id_collision_is_not_demuxed_as_peer_success(self):
        self.submit()
        self.submit("b", request="b1")
        self.mux.pump()
        self.receive(progress("a", "collision"), progress("b", "collision"))
        self.assertFalse(self.mux.connected)
        self.assertEqual(self.request("a", "a1").state, "unknown")
        self.assertEqual(self.request("b", "b1").state, "unknown")

    def test_malformed_or_wrong_response_identity_fences_connection(self):
        for event in (
            {"type": "response.completed"},
            {"type": "response.completed", "stream_id": "unowned"},
            {"type": "response.completed", "stream_id": "a"},
            completed("a", "wrong-response"),
            {"type": "error", "stream_id": None},
            {"type": "response.in_progress", "stream_id": "a"},
        ):
            with self.subTest(event=event):
                self.setUp()
                self.submit()
                self.submit("b", request="b1")
                self.mux.pump()
                self.receive(progress("a", "ra"), event)
                self.assertFalse(self.mux.connected)
                self.assertEqual(self.request("a", "a1").state, "unknown")
                self.assertEqual(self.request("b", "b1").state, "unknown")

    def test_failed_and_incomplete_responses_are_stream_local(self):
        for terminal in ("response.failed", "response.incomplete"):
            with self.subTest(terminal=terminal):
                self.setUp()
                self.submit()
                self.submit("b", request="b1")
                self.mux.pump()
                self.receive(
                    {"type": terminal, "stream_id": "a", "response": {"id": "ra"}}
                )
                self.assertTrue(self.mux.connected)
                self.assertEqual(self.request("a", "a1").state, terminal.split(".")[1])
                self.assertEqual(self.request("b", "b1").state, "active")

    def test_connection_lifetime_fences_without_reconnect(self):
        now = [0.0]
        self.mux = Multiplex(self.socket, clock=lambda: now[0])
        self.register("a", self.a)
        self.submit()
        self.mux.pump()
        now[0] = 3600.0
        with self.assertRaises(Rejected):
            self.mux.pump()
        self.assertEqual(self.request("a", "a1").state, "unknown")
        self.assertEqual(len(self.socket.sent), 1)


if __name__ == "__main__":
    unittest.main()
