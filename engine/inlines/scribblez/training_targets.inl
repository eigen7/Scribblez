#pragma once

#include "scribblez/training_targets.h"

namespace scribblez {
namespace binlog {

template <typename... Ts>
void TargetList<Ts...>::encode_all(const GameLogView& v, float* out) {
  int off = 0;
  ((Ts::encode(v, out + off), off += detail::target_floats<Ts>()), ...);
}

}  // namespace binlog
}  // namespace scribblez
