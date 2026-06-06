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

void MacondoOraclePool::set_params(const MacondoOracle::Params& params) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!oracles_.empty()) {
    throw std::runtime_error(
        "MacondoOraclePool::set_params called after an oracle was built");
  }
  params_ = params;
}

void MacondoOraclePool::set_lexicon(const std::string& lexicon) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!oracles_.empty()) {
    throw std::runtime_error(
        "MacondoOraclePool::set_lexicon called after an oracle was built");
  }
  params_.lexicon = lexicon;
}

MacondoOracle& MacondoOraclePool::get(int thread_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = oracles_.find(thread_id);
  if (it == oracles_.end()) {
    auto [inserted, _] =
        oracles_.emplace(thread_id, std::make_unique<MacondoOracle>(params_));
    it = inserted;
  }
  return *it->second;
}

}  // namespace scribblez
