#include <cassert>
#include <climits>
#include "nuvion_motor_bridge/position_reference.h"
int main() {
  int32_t position = -99;
  assert(controlPosition(4395, 3, 0, 0, position) && position == 299);
  assert(controlPosition(-1, 3, 0, 0, position) && position == 4095);
  assert(controlPosition(INT_MIN, 3, 0, 0, position) && position == 0);
  assert(controlPosition(INT_MAX, 3, 0, 0, position) && position == 4095);
  assert(controlPosition(4095, 3, 1, 0, position) && position == 4095);
  assert(controlPosition(0, 5, 1, 0, position) && position == 0);
  assert(!controlPosition(4395, 3, 1, 0, position));
  assert(!controlPosition(4395, 5, 0, 0, position));
  assert(!controlPosition(4395, 4, 0, 0, position));
  assert(!controlPosition(2000, 3, 0, 10, position));
}
