#include "nn/service_cache.h"

#include "nn/trt_eval_service.h"

namespace scribblez {
namespace nn {

// Out-of-line so destroying the owned services (and thus the complete
// TrtEvalService type) stays in this TensorRT-linked TU, not wherever a
// ServiceCache is held by value.
ServiceCache::ServiceCache() = default;
ServiceCache::~ServiceCache() = default;

PositionEvalService* ServiceCache::position_service(
  const NeuralNetParams<PositionEvaluationSpec>& params) {
  for (const PositionEntry& e : position_) {
    if (e.key == params) return e.service.get();
  }
  std::unique_ptr<PositionEvalService> service =
    make_loaded_service<PositionEvaluationSpec>(params);
  PositionEvalService* raw = service.get();
  position_.push_back({params, std::move(service)});
  return raw;
}

}  // namespace nn
}  // namespace scribblez
