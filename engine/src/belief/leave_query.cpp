#include "belief/leave_query.h"

#include "util/work_pool.h"

namespace scribblez::belief {

std::vector<std::shared_ptr<const LeaveSampler>> beliefs_of(std::span<const LeaveQuery> queries,
                                                            const LeaveParamsFn& predict,
                                                            const LeaveValueArrays& table) {
  std::vector<size_t> modelled;
  std::vector<float> rows;
  for (size_t i = 0; i < queries.size(); ++i) {
    if (!queries[i].modelled()) continue;
    modelled.push_back(i);
    rows.insert(rows.end(), queries[i].input.begin(), queries[i].input.end());
  }
  std::vector<std::shared_ptr<const LeaveSampler>> out(queries.size());
  if (modelled.empty()) return out;
  std::vector<LeaveParams> params(modelled.size());
  predict(rows.data(), int(modelled.size()), params.data());
  util::parallel_for(modelled.size(), [&](size_t j) {
    const LeaveQuery& q = queries[modelled[j]];
    out[modelled[j]] =
      std::make_shared<const LeaveSampler>(params[j], q.length, q.leave_pool, table);
  });
  return out;
}

}  // namespace scribblez::belief
