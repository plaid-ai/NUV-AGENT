"""Blocking network probes must not stall camera/control websocket I/O."""
import asyncio
import threading
import unittest
from unittest import mock

# Reuse the native-GStreamer-independent pipeline import setup.
from tests.inference.test_pipeline_durable_safety import pipeline


class ConnectivityLatencyTest(unittest.IsolatedAsyncioTestCase):
    async def test_blocking_probe_does_not_block_signaling_loop(self):
        entered = threading.Event()
        release = threading.Event()
        reporter = mock.Mock()
        def collect():
            entered.set()
            release.wait(timeout=0.4)
            return {'quality': 'GOOD'}
        reporter.collect_sample_payload.side_effect = collect
        reporter.build_transition_payload.return_value = None
        # A callback on the signaling event loop must run before the probe can
        # finish, even though the probe is genuinely blocked on a thread event.
        observed = []
        with mock.patch.object(pipeline, 'get_device_state_coordinator'), \
                mock.patch.object(pipeline, 'fleet_command_runtime', None):
            task = asyncio.create_task(pipeline.device_connectivity_sender(reporter))
            def check():
                observed.append(entered.is_set() and not release.is_set()
                                and reporter.build_transition_payload.call_count == 0)
                release.set()
            asyncio.get_running_loop().call_later(0.05, check)
            try:
                await asyncio.sleep(0.1)
            finally:
                release.set()
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        self.assertEqual(observed, [True])

    async def test_empty_sample_does_not_start_second_synchronous_probe(self):
        reporter = mock.Mock()
        reporter.collect_sample_payload.return_value = None
        reporter.build_transition_payload.return_value = None
        with mock.patch.object(pipeline, 'get_device_state_coordinator'), \
                mock.patch.object(pipeline, 'fleet_command_runtime', None):
            task = asyncio.create_task(pipeline.device_connectivity_sender(reporter))
            try:
                await asyncio.sleep(0.05)
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        reporter.collect_sample_payload.assert_called_once()
        reporter.build_transition_payload.assert_not_called()
