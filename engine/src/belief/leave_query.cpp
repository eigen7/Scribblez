#include "belief/leave_query.h"

#include "util/work_pool.h"

namespace scribblez::belief {

std::vector<LeaveParams> predict_beliefs(std::span<const LeaveQuery> queries,
                                         const LeaveParamsFn& predict) {
  std::vector<size_t> modelled;
  for (size_t i = 0; i < queries.size(); ++i)
    if (queries[i].modelled()) modelled.push_back(i);
  std::vector<LeaveParams> out(queries.size());
  if (modelled.empty()) return out;
  std::vector<float> rows;
  rows.reserve(modelled.size() * queries[modelled[0]].input.size());
  for (size_t i : modelled)
    rows.insert(rows.end(), queries[i].input.begin(), queries[i].input.end());
  std::vector<LeaveParams> params(modelled.size());
  predict(rows.data(), int(modelled.size()), params.data());
  for (size_t j = 0; j < modelled.size(); ++j) out[modelled[j]] = params[j];
  return out;
}

std::shared_ptr<const LeaveSampler> belief_of(const LeaveQuery& q, const LeaveParams& params,
                                              const LeaveValueArrays& table) {
  if (!q.modelled()) return nullptr;
  return std::make_shared<const LeaveSampler>(params, q.length, q.leave_pool, table);
}

std::vector<std::shared_ptr<const LeaveSampler>> beliefs_of(std::span<const LeaveQuery> queries,
                                                            const LeaveParamsFn& predict,
                                                            const LeaveValueArrays& table) {
  const std::vector<LeaveParams> params = predict_beliefs(queries, predict);
  std::vector<std::shared_ptr<const LeaveSampler>> out(queries.size());
  util::parallel_for(queries.size(),
                     [&](size_t i) { out[i] = belief_of(queries[i], params[i], table); });
  return out;
}

}  // namespace scribblez::belief
