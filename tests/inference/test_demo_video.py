import threading
import types
import unittest
from unittest.mock import Mock
from nuvion_app.inference.demo_video import DemoVideoBridge

class DemoVideoBridgeTest(unittest.TestCase):
    def bridge(self):
        bridge = DemoVideoBridge.__new__(DemoVideoBridge)
        bridge.stop = threading.Event()
        bridge.decoder = Mock()
        bridge.sink = Mock()
        bridge.appsrc = Mock()
        bridge.on_loop = Mock()
        bridge.gst = types.SimpleNamespace(
            MSECOND=1000000, CLOCK_TIME_NONE=-1,
            Format=types.SimpleNamespace(TIME=3),
            SeekFlags=types.SimpleNamespace(FLUSH=1, KEY_UNIT=2),
            State=types.SimpleNamespace(NULL=0),
            FlowReturn=types.SimpleNamespace(OK=0),
        )
        return bridge

    def test_file_pts_never_reaches_live_appsrc(self):
        b = self.bridge()
        buffer = types.SimpleNamespace(pts=900, dts=800)
        sample = Mock()
        sample.get_buffer.return_value.copy_deep.return_value = buffer
        b.sink.emit.return_value = sample
        def push(*args):
            b.stop.set()
            return 0
        b.appsrc.emit.side_effect = push
        b._pump()
        self.assertEqual((buffer.pts, buffer.dts), (-1, -1))
        b.appsrc.emit.assert_called_once_with('push-buffer', buffer)

    def test_eos_rewinds_only_private_decoder(self):
        b = self.bridge()
        b.sink.emit.return_value = None
        b.sink.get_property.return_value = True
        def seek(*args):
            b.stop.set()
            return True
        b.decoder.seek_simple.side_effect = seek
        b._pump()
        b.decoder.seek_simple.assert_called_once_with(3, 3, 0)
        b.on_loop.assert_called_once()
        b.appsrc.emit.assert_not_called()
