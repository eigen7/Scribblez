#pragma once

#include "nn/eval_service.h"
#include "nn/neural_net.h"

#include <memory>
#include <vector>

namespace scribblez {
namespace nn {

// Owns the evaluation services shared across a run's worker threads, so N
// threads drive one engine -- and one TensorRT execution context, whose
// activation memory would otherwise be paid N times over -- per distinct model,
// instead of one apiece. A service is built lazily on first request and keyed
// on the full set of engine-determining params (NeuralNetParamsBase::operator==),
// so two seats with different models -- or the same model at a different
// precision, batch bound, or device -- stay distinct, while one seat's threads
// share.
//
// The returned pointers are non-owning: the cache owns every service's
// lifetime, so it MUST outlive every agent that borrows from it. GameEngine
// enforces this by declaring the cache before its agent vector, which is
// therefore destroyed first.
//
// Populated serially, during GameEngine construction; the cache itself is not
// synchronized. Evaluation through the returned services IS thread-safe -- that
// is the service's own contract (eval_service.h).
class ServiceCache {
 public:
  ServiceCache();
  ~ServiceCache();

  ServiceCache(const ServiceCache&) = delete;
  ServiceCache& operator=(const ServiceCache&) = delete;

  // The shared position-evaluation service for `params`, building and loading
  // it on first request and returning the same instance thereafter.
  PositionEvalService* position_service(const NeuralNetParams<PositionEvaluationSpec>& params);

 private:
  struct PositionEntry {
    NeuralNetParamsBase key;
    std::unique_ptr<PositionEvalService> service;
  };
  std::vector<PositionEntry> position_;
};

}  // namespace nn
}  // namespace scribblez
