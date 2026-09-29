#pragma once
#include <stdint.h>

// XL330 mode 3 resets its continuous Present Position on torque enable.
// Only an unpowered, zero-offset position-mode motor may be normalized.
// Keep the raw encoder separately; never wrap active motion or mode 5 data.
inline bool controlPosition(int32_t raw, int32_t mode, int32_t torque,
                            int32_t homingOffset, int32_t &position) {
  if (mode != 3 && mode != 5) return false;
  if (homingOffset != 0) return false;
  if (raw >= 0 && raw <= 4095) { position = raw; return true; }
  if (mode != 3 || torque != 0) return false;
  position = raw % 4096;
  if (position < 0) position += 4096;
  return true;
}
