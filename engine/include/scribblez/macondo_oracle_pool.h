#pragma once

#include "scribblez/macondo_oracle.h"

#include <memory>
#include <mutex>
#include <vector>

// Forward-declared so add_options() can register --macondo without dragging
// boost::program_options into every consumer of this header.
namespace boost::program_options { class options_description; }

namespace scribblez {

// Process-wide pool of MacondoOracle instances, indexed by GameRunner thread
// id (0..threads-1).
//
// Each game thread gets its own MacondoOracle on first access; two agents
// that share a thread id (i.e. the two seats of the same game) safely share
// the same oracle, since they take turns and never call evaluate()
// concurrently. Different thread ids get distinct subprocesses, which is
// what enables parallel HastyBot self-play.
//
// Agents should call get() once at construction time and cache the returned
// pointer, so the per-evaluate path is lock-free.
//
// Usage:
//   - call MacondoOraclePool::instance().add_options(desc) before parsing
//     argv (registers --macondo);
//   - agents call MacondoOraclePool::instance().get(thread_id_) in their
//     constructor and store the resulting MacondoOracle*.
class MacondoOraclePool {
 public:
  static MacondoOraclePool& instance();

  // Register --macondo on the given options_description. Mutates the stored
  // params in-place; takes effect for every subsequent get().
  void add_options(boost::program_options::options_description& desc);

  // Return the oracle for `thread_id`, lazily constructing it from the stored
  // params on first access. Thread-safe; grows the internal vector on demand.
  MacondoOracle& get(int thread_id);

 private:
  MacondoOraclePool() = default;

  std::mutex mutex_;
  MacondoOracle::Params params_;
  std::vector<std::unique_ptr<MacondoOracle>> oracles_;
};

}  // namespace scribblez
