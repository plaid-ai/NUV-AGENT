import tempfile
import unittest
from pathlib import Path
from nuvion_app.runtime.system_health import SystemHealthCollector


class SystemHealthTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.collector = SystemHealthCollector('jetson_orin_nano_dev', sys_root=self.root, boot_path=self.root/'boot')

    def write(self, path, value):
        p = self.root / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(value)

    def test_sensor_units_and_power_scope(self):
        self.write('class/thermal/thermal_zone0/type', 'cpu')
        self.write('class/thermal/thermal_zone0/temp', '45000')
        for name, value in {'name':'ina3221','in1_label':'VDD_IN','in1_input':'5000','curr1_input':'2000'}.items():
            self.write('class/hwmon/hwmon0/'+name, value)
        sensors = self.collector.collect()['sensors']
        power = next(x for x in sensors if x['kind'] == 'POWER')
        self.assertEqual(power['value'], 10)
        self.assertEqual(power['scope'], 'MODULE')
        self.assertEqual(sensors[0]['value'], 45)

    def test_missing_and_invalid_are_not_zero(self):
        self.write('class/thermal/thermal_zone0/type', 'cpu')
        self.write('class/thermal/thermal_zone0/temp', 'NaN')
        sensors = self.collector.collect()['sensors']
        self.assertIsNone(sensors[0]['value'])
        self.assertEqual(next(x for x in sensors if x['kind']=='POWER')['availability'], 'UNSUPPORTED')

    def test_pro_rail_power_not_whole_device(self):
        for name, value in {'name':'ina232','power1_input':'13600000'}.items():
            self.write('class/hwmon/hwmon0/'+name, value)
        power = next(x for x in self.collector.collect()['sensors'] if x['kind']=='POWER')
        self.assertEqual(power['value'], 13.6)
        self.assertEqual(power['scope'], 'RAIL')

    def test_stream_sequence_changes_without_changing_boot(self):
        first = self.collector.collect(); second = self.collector.collect()
        self.assertEqual(first['streamId'], second['streamId'])
        self.assertEqual(second['sequence'], first['sequence']+1)

    def test_reconnect_does_not_overlap_collection(self):
        self.collector._lock.acquire()
        try:
            self.assertIsNone(self.collector.collect())
        finally:
            self.collector._lock.release()
        self.assertEqual(self.collector.collect()['sequence'], 1)

    def test_thermal_warning_requires_duration_and_hysteresis(self):
        from unittest.mock import patch
        reading = dict(kind='TEMPERATURE', critical=90, value=91)
        with patch('nuvion_app.runtime.system_health.time.monotonic') as clock:
            for at in (0, 5, 10):
                clock.return_value=at
                self.assertEqual(self.collector._thermal_state([reading]), 'NORMAL')
            clock.return_value=15
            self.assertEqual(self.collector._thermal_state([reading]), 'WARNING')
            reading['value']=89
            clock.return_value=20
            self.assertEqual(self.collector._thermal_state([reading]), 'WARNING')
            reading['value']=86
            for at in (25,30,35,40):
                clock.return_value=at
                result=self.collector._thermal_state([reading])
            self.assertEqual(result,'NORMAL')
            self.assertEqual(self.collector._thermal_state([]),'UNKNOWN')

    def test_rpi_pmic_rails_are_not_total_input_power(self):
        from unittest.mock import patch
        from subprocess import CompletedProcess
        self.collector.profile='rpi5_deepx_dx_m1'
        with patch('nuvion_app.runtime.system_health.subprocess.run', side_effect=[
            CompletedProcess([],0,'VDD_CORE_A current(7)=2.0A\nVDD_CORE_V volt(7)=1.1V'),
            CompletedProcess([],0,'throttled=0x50004')]):
            sensors=self.collector.collect()['sensors']
        power=next(s for s in sensors if s['kind']=='POWER')
        self.assertAlmostEqual(power['value'],2.2)
        self.assertEqual(power['scope'],'RAIL')
        self.assertEqual(next(s for s in sensors if s['kind']=='THROTTLE')['value'],1)


class HealthSenderTest(unittest.IsolatedAsyncioTestCase):
    async def test_slow_collection_does_not_block_event_loop(self):
        import asyncio
        import time
        from unittest.mock import Mock
        from nuvion_app.runtime.system_health import system_health_sender
        collector=Mock()
        def collect():
            time.sleep(.1)
            return {'sequence':1}
        collector.collect.side_effect=collect
        sent=asyncio.Event()
        async def send(payload):
            sent.set()
        task=asyncio.create_task(system_health_sender(collector,send))
        try:
            ticks=0
            while not sent.is_set():
                ticks+=1
                await asyncio.sleep(.01)
            self.assertGreater(ticks,3)
        finally:
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
