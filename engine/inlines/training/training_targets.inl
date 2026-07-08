#pragma once

#include "training/training_targets.h"

namespace scribblez {

template <typename... Ts>
void TargetList<Ts...>::encode_all(const EncodeContext& v, float* out) {
  int off = 0;
  ((Ts::encode(v, out + off), off += detail::target_floats<Ts>()), ...);
}

}  // namespace scribblez
