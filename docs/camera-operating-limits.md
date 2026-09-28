# Ultra camera control

`CAMERA_POSITION_SET` schema 1 keeps LEFT/RIGHT/UP/DOWN as one acknowledged 11-tick (~0.97°) JOG. It additionally supports STATUS (read), STOP (hold), and LIMITS (persist operator bounds). Existing single-step clients remain compatible. The bridge firmware is unchanged; this interface does not issue RUN.

LIMITS requires exactly `limits: {panMin, panMax, tiltMin, tiltMax}` in integer encoder ticks, each 0..4095, minimum < maximum. Both present motor positions must fit the new range. The Agent stops/reads before saving atomically to `camera-limits.json` under its settings state directory. Invalid persisted data disables the NUV1 backend instead of silently discarding limits. Full encoder bounds are used only when no file exists.

The backend checks actual position and next goal for every JOG, including tracking. Tracking returns false at a boundary without terminating its worker. A manual command fails with an explicit limit error. The command expiry is checked before waiting for UART and again immediately before JOG. Movement acknowledgement waits for both measured positions to settle within two ticks of the goals, with a one-second settling timeout.

Successful reports contain `controlVersion: 2`, `positions: {pan, tilt}`, `limits`, `observedAt`, `protocol: NUV1`, and `health: FUNCTIONAL_HEALTHY`. Position is the measured encoder value, not the target. Conversion is ticks × 360 / 4096; these are absolute motor angles, without an installation zero calibration.

The UI repeats only after the previous command's exact ID/sequence is reported. Each movement has a three-second admission TTL. Release, blur, hidden document, error, or unmount ends repetition and follows the in-flight bounded step with STOP. This is stepped hold-to-move; it is not continuous velocity control or a hardware emergency stop.

Rollout: BE contract → Agent update → FE. A legacy Agent cannot supply controlVersion 2, so the new FE keeps motion disabled until a fresh STATUS succeeds. Validate physical direction, tracking limits, repeat/release behavior and restart persistence on the installed assembly after the update. Automated tests use a fake UART and do not establish physical calibration or mechanical clearance.
