#include "util/hardware.h"

#include <sched.h>

namespace util {

int default_thread_count() {
  cpu_set_t set;
  CPU_ZERO(&set);
  sched_getaffinity(0, sizeof(set), &set);
  return CPU_COUNT(&set);
}

}  // namespace util
