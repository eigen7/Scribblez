#include "scribblez/label_encoder.h"

namespace scribblez {
namespace binlog {

void encode_labels(int score_active_final, int score_opp_final, float* out) {
  if (score_active_final > score_opp_final) {
    out[0] = 1.0f;
    out[1] = 0.0f;
    out[2] = 0.0f;
  } else if (score_active_final == score_opp_final) {
    out[0] = 0.0f;
    out[1] = 1.0f;
    out[2] = 0.0f;
  } else {
    out[0] = 0.0f;
    out[1] = 0.0f;
    out[2] = 1.0f;
  }
  out[3] = static_cast<float>(score_active_final - score_opp_final);
}

}  // namespace binlog
}  // namespace scribblez
