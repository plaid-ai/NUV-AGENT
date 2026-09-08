from __future__ import annotations

import asyncio
import contextlib
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from nuvion_app.inference.command_inbox import DurableCommandInbox
from nuvion_app.inference.command_observation import DurableCommandObservationOutbox
from nuvion_app.inference.command_runtime import FleetCommandRuntime
from nuvion_app.inference.effect_reconciler import (
    FleetEffectCoordinator,
    ReconcilerRegistry,
)
from nuvion_app.inference.reconcile_store import DurableReconcileStore
from nuvion_app.inference.settings_reconciler import (
    AtomicSettingsStore,
    SettingsReconciler,
)
from tests.inference.test_pipeline_durable_safety import pipeline
from tests.inference.test_settings_reconciler import (
    _command,
    _healthy_command_outbox,
    _healthy_event_outbox,
    _Runtime,
)


async def _forever(*_args):
    await asyncio.Future()


class PipelineOfflineEffectsTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config_path = self.root / "agent.env"
        self.config_path.write_text("NUVION_MODEL_POINTER=visualad/iq9075-htp-demo\n")
        self.settings = AtomicSettingsStore(self.config_path, self.root / "settings")
        self.inbox = DurableCommandInbox(self.root / "inbox.sqlite3")
        self.observations = DurableCommandObservationOutbox(self.inbox)
        self.clock = [100.0]
        self.store = DurableReconcileStore(
            self.inbox,
            observation_outbox=self.observations,
            monotonic_clock=lambda: self.clock[0],
        )
        self.desired = {
            "pointer": "visualad/iq9075-htp-demo",
            "digest": "sha256:" + "a" * 64,
        }
        self.command = _command(
            1, activation="RESTART", sections={"model": self.desired}
        )
        self.inbox.accept(self.command)
        self.inbox.transition(self.command.command_id, "IN_PROGRESS")
        self.inbox.run_transactional_effect(
            self.command.command_id,
            lambda connection: self.store.stage_verified(self.command, connection),
        )
        self.restart = mock.Mock(return_value=True)
        before = self._runtime("before", _Runtime())
        before.effect_coordinator.run_once()
        self.assertEqual(self.restart.call_count, 1)
        self.restart.reset_mock()
        self.patches = contextlib.ExitStack()
        self.addCleanup(self.patches.close)
        for name, value in {
            "signaling_loop": None,
            "outbound_queue": None,
            "initialize_durable_event_outbox": mock.Mock(),
            "refresh_updater_runtime_telemetry": mock.AsyncMock(),
            "updater_telemetry_refresh_sender": _forever,
            "FLEET_EFFECT_RECONCILE_INTERVAL_SEC": 0.25,
            "log": mock.Mock(),
        }.items():
            self.patches.enter_context(mock.patch.object(pipeline, name, value))
        self.client_task = None

    async def asyncTearDown(self):
        await self._stop_client()

    async def _stop_client(self):
        if self.client_task is not None:
            self.client_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.client_task
            self.client_task = None

    def _runtime(self, process_id, adapter):
        registry = ReconcilerRegistry()
        registry.register(
            SettingsReconciler(
                store=self.settings,
                runtime=adapter,
                process_instance_id=process_id,
                startup_clock=lambda: self.clock[0],
                event_outbox_health_provider=_healthy_event_outbox,
                command_outbox_health_provider=_healthy_command_outbox,
            )
        )
        coordinator = FleetEffectCoordinator(
            inbox=self.inbox,
            store=self.store,
            registry=registry,
            process_instance_id=process_id,
            restart_requester=self.restart,
        )
        return FleetCommandRuntime(
            inbox=self.inbox,
            processor=types.SimpleNamespace(),
            http_client=types.SimpleNamespace(),
            ack_sender=mock.Mock(return_value=False),
            effect_coordinator=coordinator,
            reconcile_store=self.store,
            observation_outbox=self.observations,
        )

    async def _start_offline(self, runtime):
        self.patches.enter_context(mock.patch.object(pipeline, "login", _forever))
        self.patches.enter_context(
            mock.patch.object(
                pipeline, "initialize_fleet_command_runtime", return_value=runtime
            )
        )
        self.client_task = asyncio.create_task(pipeline.signaling_client_main())
        await self._wait_for(
            lambda: (
                self.store.get_job(self.command.command_id).checkpoint.get("nextAction")
                == "RETRY_EFFECT"
            )
        )

    async def _wait_for(self, predicate):
        async with asyncio.timeout(2):
            while not predicate():
                await asyncio.sleep(0.01)

    async def test_offline_candidate_commits_after_fresh_model_proof(self):
        adapter = _Runtime()
        adapter.startup_pending = mock.Mock(return_value=True)
        runtime = self._runtime("candidate", adapter)
        await self._start_offline(runtime)
        self.assertEqual(self.inbox.get(self.command.command_id).status, "IN_PROGRESS")
        self.assertEqual(self.observations.pending(), [])
        adapter.state["model"] = self.desired
        adapter.startup_pending.return_value = False
        self.clock[0] += 120
        await self._wait_for(lambda: self.settings.marker()["phase"] == "COMMITTED")
        await self._wait_for(
            lambda: self.inbox.get(self.command.command_id).status == "SUCCEEDED"
        )
        await self._wait_for(lambda: runtime.ack_sender.call_count == 1)
        self.assertEqual(len(self.observations.pending()), 1)
        self.assertEqual(
            self.inbox.get(self.command.command_id).sequence, self.command.sequence
        )
        self.restart.assert_not_called()
        # Publication was unavailable, but terminal ACK/state stay durable.
        self.assertEqual(runtime.ack_sender.call_count, 1)

    async def _assert_offline_rollback(self, *, timeout):
        adapter = _Runtime()
        adapter.startup_pending = mock.Mock(return_value=True)
        adapter.healthy = False
        runtime = self._runtime("candidate", adapter)
        await self._start_offline(runtime)
        if timeout:
            self.clock[0] += 600
        else:
            adapter.startup_pending.return_value = False
            self.clock[0] += 120
        await self._wait_for(
            lambda: self.settings.marker()["phase"] == "ROLLBACK_STAGED"
        )
        await self._wait_for(lambda: self.restart.call_count == 1)
        self.assertEqual(self.observations.pending(), [])
        self.assertTrue(self.settings.lkg_is_active())
        await self._stop_client()
        # Simulate the supervisor's LKG process, still with no login/transport.
        lkg = _Runtime()
        lkg.startup_pending = mock.Mock(return_value=True)
        self.restart.reset_mock()
        runtime = self._runtime("lkg", lkg)
        await self._start_offline(runtime)
        lkg.startup_pending.return_value = False
        self.clock[0] += 120
        await self._wait_for(
            lambda: self.inbox.get(self.command.command_id).status == "ROLLED_BACK"
        )
        await self._wait_for(lambda: runtime.ack_sender.call_count == 1)
        self.assertEqual(self.settings.marker()["phase"], "ROLLED_BACK")
        self.assertEqual(self.observations.pending(), [])
        self.assertEqual(runtime.ack_sender.call_args.args[1]["status"], "ROLLED_BACK")
        runtime.replay_recent_acks()
        self.assertEqual(runtime.ack_sender.call_args.args[1]["status"], "ROLLED_BACK")
        self.restart.assert_not_called()

    async def test_offline_inference_failure_restores_lkg(self):
        await self._assert_offline_rollback(timeout=False)

    async def test_offline_600_second_timeout_restores_lkg(self):
        await self._assert_offline_rollback(timeout=True)

    async def test_transport_reconnects_do_not_replace_local_worker_and_shutdown_cancels_it(
        self,
    ):
        started = []
        stopped = []
        connections = []
        twice_disconnected = asyncio.Event()

        async def worker(_runtime):
            started.append(asyncio.current_task())
            try:
                await asyncio.Future()
            finally:
                stopped.append(asyncio.current_task())

        class Socket:
            def __init__(self):
                self.responses = iter(("o", 'a["CONNECTED\\n\\n\\u0000"]'))

            async def recv(self):
                return next(self.responses)

            async def send(self, _payload):
                return None

            def __aiter__(self):
                return self

            async def __anext__(self):
                await asyncio.sleep(0)
                raise ConnectionError("test transport disconnected")

        @contextlib.asynccontextmanager
        async def connect(_url):
            connections.append(1)
            try:
                yield Socket()
            finally:
                if len(connections) == 2:
                    twice_disconnected.set()

        real_sleep = asyncio.sleep

        async def sleep(delay):
            if delay == 10:
                if len(connections) >= 2:
                    await asyncio.Future()
                await real_sleep(0)
            else:
                await real_sleep(delay)

        runtime = types.SimpleNamespace(
            reconcile_effects=mock.AsyncMock(return_value=0),
            on_connected=mock.AsyncMock(return_value=0),
        )
        replacements = {
            "initialize_fleet_command_runtime": mock.Mock(return_value=runtime),
            "login": mock.AsyncMock(return_value="test-only"),
            "set_auth_token": mock.Mock(),
            "_reset_agent_ws_state": mock.Mock(),
            "_reset_webrtc_signaling_transport": mock.Mock(),
            "_set_update_commit_signaling_ready": mock.Mock(),
            "fleet_effect_reconcile_sender": worker,
            "outbound_sender": _forever,
            "durable_event_replay_sender": _forever,
            "stomp_heartbeat_sender": _forever,
            "device_state_heartbeat_sender": _forever,
            "fleet_command_poll_sender": _forever,
            "webrtc_stats_sender": _forever,
            "fleet_observation_sender": _forever,
            "REQUIRED_AGENT_SUBSCRIPTIONS": (),
            "CONNECTIVITY_ENABLED": False,
        }
        for name, value in replacements.items():
            self.patches.enter_context(mock.patch.object(pipeline, name, value))
        self.patches.enter_context(
            mock.patch.object(pipeline.websockets, "connect", connect)
        )
        self.patches.enter_context(
            mock.patch.object(
                pipeline.stomper, "unpack_frame", return_value={"headers": {}}
            )
        )
        self.patches.enter_context(mock.patch.object(pipeline.asyncio, "sleep", sleep))
        self.client_task = asyncio.create_task(pipeline.signaling_client_main())
        await asyncio.wait_for(twice_disconnected.wait(), timeout=2)
        await real_sleep(0)
        self.assertEqual(len(started), 1)
        self.assertEqual(stopped, [])
        await self._stop_client()
        self.assertEqual(stopped, started)


if __name__ == "__main__":
    unittest.main()
