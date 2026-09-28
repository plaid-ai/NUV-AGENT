# Device health telemetry v1

Read-only monitoring for NUVION (rpi5_deepx_dx_m1), Pro (ventuno_q), Ultra (jetson_orin_nx) and the Orin Nano Ultra prototype. `NUVION_SYSTEM_HEALTH_ENABLED=false` is the rollout default. Enabling it requires the BE device-health migration and receiver first.

The authenticated signaling connection sends `/app/device/health` every 5 seconds. Samples contain schemaVersion, bootId, streamId, sequence, profile, measuredAt, thermalState and at most 128 sensors. A sensor contains a stable ID, label, kind, value, scope, source, availability and an optional manufacturer critical temperature. Collection runs on a worker thread and never queues overlapping reads after reconnect. Missing/invalid values are null, never zero. No motor UART, GPIO writes or power-mode changes are performed.

- Temperature: thermal sysfs millidegrees converted to °C.
- Power: hwmon microW converted to W. Jetson INA3221 mV × mA converted to W; VDD_IN means **module power**, not complete appliance input. Other rails remain separate.
- Pro INA232: measured rail only; system-input coverage and shunt calibration still require board qualification.
- Raspberry Pi 5: vcgencmd PMIC rails and current throttle bit when available. Rail power excludes unmeasured DEEPX/peripherals. Basic-model physical validation remains pending.
- Fan/throttle: readable hwmon RPM and Raspberry Pi current throttle state; unsupported sensors are explicit. USB negotiated current is not a measured load.
- Thermal warning: a hardware critical threshold must remain exceeded for 15s; clear only after 15s below threshold minus 3°C. Measurement gaps reset the timer. No threshold means UNKNOWN, not healthy.

## Rollout and rollback

1. Apply `NUV-BE/deploy/sql/20260928_device_health.sql` and deploy its receiver. FE is additive but needs the new endpoints.
2. Deploy this Agent through the existing release mechanism, then set `NUVION_SYSTEM_HEALTH_ENABLED=true` in the managed runtime environment and restart under the normal service procedure.
3. Verify space-scoped latest measurements, measuredAt freshness (<20s), and 1-minute history. Verify camera controls and stop/deadman behavior continue to work.
4. Disable the flag to stop health transmission. The additive BE/FE endpoints can remain deployed; old data becomes visibly stale. This feature does not modify OTA signature or rollback policy.

## Evidence, 2026-09-28

Read-only collector runs on Ventuno Q and Orin Nano returned valid thermal and power values (51 and 14 sensor entries including unsupported placeholders). Example power: Pro rail 11.8W; Ultra VDD_IN module 13.4W. These are single samples, not calibrated whole-appliance power or performance comparisons. Native health/connectivity/camera/signaling regression subset: 42 tests passed. Raspberry Pi parser coverage uses fixtures; physical basic-model qualification is outstanding.
