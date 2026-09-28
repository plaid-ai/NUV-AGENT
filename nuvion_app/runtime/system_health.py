"""Read-only, bounded hardware telemetry. Never infer whole-device power."""
from __future__ import annotations

import asyncio
import hashlib
import math
import logging
import threading
import time
import re
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path


def _read(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except (OSError, TypeError, UnicodeError):
        return None


def _number(path: Path, divisor=1.0):
    try:
        value = float(_read(path)) / divisor
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


class SystemHealthCollector:
    def __init__(self, profile='unknown', *, sys_root=Path('/sys'), boot_path=Path('/proc/sys/kernel/random/boot_id')):
        self.root = Path(sys_root)
        self.profile = profile
        self.boot_id = _read(Path(boot_path)) or str(uuid.uuid4())
        self.stream_id = str(uuid.uuid4())
        self.sequence = 0
        self._lock = threading.Lock()
        self._warning = False
        self._transition_since = None
        self._last_sample = None

    def collect(self):
        # Reconnect cancellation cannot stop a worker already reading sysfs.
        # Skip rather than overlap/queue a second collection.
        if not self._lock.acquire(blocking=False):
            return None
        try:
            return self._collect()
        finally:
            self._lock.release()

    def _thermal_state(self, sensors):
        now = time.monotonic()
        readings = [s for s in sensors if s['kind'] == 'TEMPERATURE' and s['critical'] is not None
                    and s['value'] is not None]
        if not readings or (self._last_sample is not None and now-self._last_sample > 15):
            self._warning = False
            self._transition_since = None
        self._last_sample = now
        if not readings:
            return 'UNKNOWN'
        transition = (all(s['value'] < s['critical']-3 for s in readings) if self._warning
                      else any(s['value'] >= s['critical'] for s in readings))
        if transition:
            if self._transition_since is None:
                self._transition_since = now
            elif now-self._transition_since >= 15:
                self._warning = not self._warning
                self._transition_since = None
        else:
            self._transition_since = None
        return 'WARNING' if self._warning else 'NORMAL'

    def _collect(self):
        sensors = []
        def add(key, label, kind, value, scope='COMPONENT', critical=None, source='sysfs'):
            if len(sensors) >= 124:
                return
            if value is not None and (not math.isfinite(value) or value > 1_000_000 or (kind != 'TEMPERATURE' and value < 0)
                    or (kind == 'TEMPERATURE' and not -50 <= value <= 200)):
                value = None
            sensors.append(dict(sensorId=hashlib.sha256(key.encode()).hexdigest()[:24],
                                label=label[:128], kind=kind, value=value, scope=scope,
                                critical=critical, source=source,
                                availability='AVAILABLE' if value is not None else 'UNAVAILABLE'))

        for zone in sorted((self.root / 'class/thermal').glob('thermal_zone*'))[:128]:
            name = _read(zone / 'type') or 'thermal'
            limits = [_number(p.with_name(p.name.replace('_type', '_temp')), 1000)
                      for p in zone.glob('trip_point_*_type') if _read(p) == 'critical']
            limits = [x for x in limits if x is not None and 0 < x <= 200]
            add('thermal:' + name, name, 'TEMPERATURE', _number(zone / 'temp', 1000),
                critical=min(limits) if limits else None)

        for hw in sorted((self.root / 'class/hwmon').glob('hwmon*'))[:128]:
            name = _read(hw / 'name') or 'unknown'
            # hwmonN is enumeration-dependent. The physical parent + chip and
            # channel label stays stable when the enumeration changes.
            parent = re.sub(r'/hwmon/hwmon\d+$', '', str(hw.resolve()))
            base = name + ':' + parent
            for f in sorted(hw.glob('fan*_input')):
                add(base + f.name, name + ' ' + f.stem, 'FAN', _number(f), source=name)
            for f in sorted(hw.glob('power*_input')):
                label = _read(f.with_name(f.name.replace('_input', '_label'))) or name
                add(base + ':' + label, label, 'POWER', _number(f, 1_000_000), 'RAIL', source=name)
            # Jetson INA3221 exposes mV/mA instead of power*_input.
            if name == 'ina3221':
                for f in sorted(hw.glob('in*_label')):
                    label = _read(f) or ''
                    channel = f.name.removeprefix('in').removesuffix('_label')
                    if label not in {'VDD_IN', 'VDD_CPU_GPU_CV', 'VDD_SOC'}:
                        continue
                    voltage = _number(hw / f'in{channel}_input', 1000)
                    current = _number(hw / f'curr{channel}_input', 1000)
                    watts = voltage * current if voltage is not None and current is not None else None
                    scope = 'MODULE' if label == 'VDD_IN' and self.profile.startswith('jetson_') else 'RAIL'
                    add(base + ':' + label, label, 'POWER', watts, scope, source=name)

        if self.profile == 'rpi5_deepx_dx_m1':
            # PMIC rails omit parts of the device (DEEPX/peripherals). Keep
            # individual rails instead of presenting their sum as input power.
            try:
                result = subprocess.run(['vcgencmd', 'pmic_read_adc'], capture_output=True,
                                        text=True, timeout=1, check=False)
                values = {name: float(value) for name, value in re.findall(
                    r'([A-Za-z0-9_]+)\s+current\(\d+\)=([0-9.]+)A', result.stdout)}
                volts = {name: float(value) for name, value in re.findall(
                    r'([A-Za-z0-9_]+)\s+volt\(\d+\)=([0-9.]+)V', result.stdout)}
                for name, current in values.items():
                    rail = name.removesuffix('_A')
                    voltage = volts.get(rail + '_V')
                    if voltage is not None:
                        add('rpi-pmic:' + rail, rail, 'POWER', current * voltage, 'RAIL', source='rpi-pmic')
            except (OSError, subprocess.TimeoutExpired):
                pass
            try:
                result = subprocess.run(['vcgencmd', 'get_throttled'], capture_output=True,
                                        text=True, timeout=1, check=False)
                match = re.search(r'throttled=0x([0-9a-fA-F]+)', result.stdout)
                add('rpi:throttle', '현재 성능 제한', 'THROTTLE',
                    int(bool(int(match[1], 16) & 0x4)) if match else None, source='rpi-firmware')
            except (OSError, subprocess.TimeoutExpired):
                pass

        # Explicit unsupported readings survive into the UI; absent is not 0.
        for kind in ('TEMPERATURE', 'POWER', 'FAN', 'THROTTLE'):
            if not any(s['kind'] == kind for s in sensors):
                sensors.append(dict(sensorId='unsupported-' + kind, label=kind,
                                    kind=kind, value=None, scope='UNKNOWN', critical=None,
                                    source='sysfs', availability='UNSUPPORTED'))
        sensors = list({s['sensorId']: s for s in sensors}.values())
        self.sequence += 1
        return dict(schemaVersion=1, bootId=self.boot_id, streamId=self.stream_id,
                    sequence=self.sequence, profile=self.profile,
                    measuredAt=datetime.now(timezone.utc).isoformat(), thermalState=self._thermal_state(sensors), sensors=sensors)


async def system_health_sender(collector, send, interval=5.0):
    """At most one measurement in flight; no durable/reconnect backlog."""
    while True:
        started = time.monotonic()
        try:
            payload = await asyncio.to_thread(collector.collect)
            if payload is not None:
                await send(payload)
        except Exception as exc:
            # Health collection is optional; never terminate camera signaling.
            logging.getLogger(__name__).warning('Health sample skipped: %s', type(exc).__name__)
        await asyncio.sleep(max(0.1, max(5.0, interval) - (time.monotonic()-started)))
