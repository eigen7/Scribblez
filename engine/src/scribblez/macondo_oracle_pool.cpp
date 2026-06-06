#include "scribblez/macondo_oracle_pool.h"

#include <boost/program_options.hpp>

#include <stdexcept>

namespace scribblez {

MacondoOraclePool& MacondoOraclePool::instance() {
  static MacondoOraclePool pool;
  return pool;
}

void MacondoOraclePool::add_options(boost::program_options::options_description& desc) {
  namespace po = boost::program_options;
  desc.add_options()(
      "macondo",
      po::value<std::string>(&params_.binary_path)->default_value(params_.binary_path),
      "path to the macondo binary; used by HastyBot and (best-effort) by the "
      "human player to annotate the move list with equity");
}

MacondoOracle& MacondoOraclePool::get(int thread_id) {
  if (thread_id < 0) throw std::runtime_error("MacondoOraclePool::get: negative thread_id");
  std::lock_guard<std::mutex> lock(mutex_);
  if (static_cast<size_t>(thread_id) >= oracles_.size()) {
    oracles_.resize(static_cast<size_t>(thread_id) + 1);
  }
  auto& slot = oracles_[static_cast<size_t>(thread_id)];
  if (!slot) slot = std::make_unique<MacondoOracle>(params_);
  return *slot;
}

}  // namespace scribblez
