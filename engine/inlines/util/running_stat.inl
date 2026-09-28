#include "util/running_stat.h"

namespace scribblez::util {

inline void RunningStat::push(double x) {
  ++n_;
  const double delta = x - mean_;
  mean_ += delta / n_;
  m2_ += delta * (x - mean_);
}

}  // namespace scribblez::util
