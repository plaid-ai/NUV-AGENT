"""Loop a private decoder without sending EOS/seek/clock resets to WebRTC."""
from pathlib import Path
import logging
import threading

log = logging.getLogger(__name__)

class DemoVideoBridge:
    def __init__(self, path, appsrc, gst, width, height, fps, on_loop=None):
        self.gst, self.appsrc, self.on_loop = gst, appsrc, on_loop
        uri = Path(path).resolve().as_uri()
        self.decoder = gst.parse_launch(
            f'uridecodebin uri="{uri}" caps="video/x-raw" ! videoconvert ! '
            f'videoscale add-borders=true ! videorate ! '
            f'video/x-raw,format=RGB,width={width},height={height},pixel-aspect-ratio=1/1,framerate={fps}/1 ! '
            'appsink name=frames sync=true max-buffers=1 drop=true wait-on-eos=false'
        )
        self.sink = self.decoder.get_by_name('frames')
        self.stop = threading.Event()
        self.thread = None

    def start(self):
        if self.decoder.set_state(self.gst.State.PLAYING) == self.gst.StateChangeReturn.FAILURE:
            raise RuntimeError('Demo decoder failed to start')
        self.thread = threading.Thread(target=self._pump, name='demo-video', daemon=True)
        self.thread.start()

    def _pump(self):
        gst = self.gst
        try:
            while not self.stop.is_set():
                sample = self.sink.emit('try-pull-sample', 200 * gst.MSECOND)
                if sample is None:
                    if self.sink.get_property('eos'):
                        if self.on_loop:
                            self.on_loop()
                        if not self.decoder.seek_simple(gst.Format.TIME, gst.SeekFlags.FLUSH | gst.SeekFlags.KEY_UNIT, 0):
                            raise RuntimeError('Demo decoder rewind failed')
                        log.info('[DEMO] Local video rewound; live stream clock preserved')
                    else:
                        error = self.decoder.get_bus().pop_filtered(gst.MessageType.ERROR)
                        if error:
                            raise RuntimeError(str(error.parse_error()))
                    continue
                buffer = sample.get_buffer().copy_deep()
                # appsrc stamps the MAIN pipeline running time, never file PTS.
                buffer.pts = gst.CLOCK_TIME_NONE
                buffer.dts = gst.CLOCK_TIME_NONE
                result = self.appsrc.emit('push-buffer', buffer)
                if result != gst.FlowReturn.OK and not self.stop.is_set():
                    raise RuntimeError(f'Demo video push failed: {result}')
        except Exception:
            log.exception('[DEMO] Video bridge failed')
        finally:
            self.decoder.set_state(gst.State.NULL)

    def close(self):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=3)
        self.decoder.set_state(self.gst.State.NULL)
