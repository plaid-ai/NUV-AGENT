# OpenRB NUV1 smooth-v4

Canonical source for the existing NUV1-smooth-v3 bridge previously kept at
`hardware/motor-web/firmware/nuvion_motor_bridge` in the NUV workspace.

XL330 torque-off Present Position is signed continuous encoder data, including
4395 and negative values. For supported mode 3, zero homing offset and torque off,
v4 reports a single-turn `position` while retaining `raw_position`. It does not
write motor configuration for a STATUS query. Unsupported references set
`reference_valid=false` and cannot ARM. No EEPROM, Homing Offset, physical mount
zero, or movement limits are changed.

ARM validates both motors and hardware bounds, writes the current single-turn
goal **before** enabling torque, then checks the reset position and holds it.
Unexpected reference changes cause torque off. Active motor positions are never
wrapped. Existing 400ms RUN deadman and 2s control timeout remain unchanged.
Do not use modulo in Agent movement validation to bypass a legacy firmware error.

Build (OpenRB-150 core 0.2.1, Dynamixel2Arduino 0.8.1):

```
arduino-cli compile --fqbn OpenRB-150:samd:OpenRB-150 \
  --output-dir /tmp/nuv-openrb-build firmware/openrb/nuvion_motor_bridge
c++ -std=c++11 -Wall -Wextra -Werror firmware/openrb/position_reference_test.cpp -o /tmp/nuv-reference-test
/tmp/nuv-reference-test
c++ -std=c++11 -Wall -Wextra -Werror -I firmware/openrb/tests firmware/openrb/tests/arm_reference_test.cpp -o /tmp/nuv-arm-test
/tmp/nuv-arm-test
```

The host tests compile the real sketch against a simulated XL330 register bus:
4395/-1/8192, goal-before-enable ordering for both axes, unsupported mode/offset,
bounds rejection, and failed post-enable reference readback.

Deployment requires a USB data connection to OpenRB. GPIO Serial3 accepts NUV1
commands but is not a SAMD USB bootloader upload channel. Build and retain both
old and new binaries before USB bootloader entry; a 1200-baud touch erases the
application's first flash row on this board core. Never enter it without a
complete known-good rollback binary. Stop Agent/other port owners first. After
upload verify firmware, both raw/control positions, inactive torque, then a
bounded manual movement and STOP. Restore the prior binary if verification fails.
