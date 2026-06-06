#pragma once

#include "scribblez/macondo_oracle.h"

#include <memory>
#include <mutex>
#include <unordered_map>

// Forward-declared so add_options() can register --macondo without dragging
// boost::program_options into every consumer of this header.
namespace boost::program_options { class options_description; }

namespace scribblez {

// Process-wide pool of MacondoOracle instances, keyed by GameRunner thread id.
//
// Each game thread (0..threads-1) gets its own MacondoOracle on first access
// from that thread; two agents that share a thread id (i.e. the two seats of
// the same game) safely share the same oracle, since they take turns and
// never call evaluate() concurrently. Different thread ids get distinct
// subprocesses, which is what enables parallel HastyBot self-play.
//
// Usage:
//   - call MacondoOraclePool::set_params() at process startup (cheap; doesn't
//     spawn anything);
//   - call MacondoOraclePool::add_options() to register --macondo on the
//     top-level options_description;
//   - agents that need an oracle do
//       MacondoOraclePool::instance().get(thread_id_)
//     on demand; the subprocess is spawned on the first evaluate() call.
class MacondoOraclePool {
 public:
  static MacondoOraclePool& instance();

  // Register --macondo on the given options_description. Mutates the stored
  // params in-place; takes effect for every subsequent get(), and is rejected
  // if any oracle has already been built.
  void add_options(boost::program_options::options_description& desc);

  // Replace the params used for future oracles. Throws if any oracle has
  // already been built (we can't reconfigure a running subprocess).
  void set_params(const MacondoOracle::Params& params);

  // Override the lexicon used for future oracles, leaving binary_path
  // untouched. Convenience for play_game.cpp, which gets lexicon from
  // GameRunner::Params (not its own CLI option). Throws if any oracle has
  // already been built.
  void set_lexicon(const std::string& lexicon);

  // Return the oracle for `thread_id`, lazily constructing it from the stored
  // params on first access. Thread-safe.
  MacondoOracle& get(int thread_id);

 private:
  MacondoOraclePool() = default;

  std::mutex mutex_;
  MacondoOracle::Params params_;
  std::unordered_map<int, std::unique_ptr<MacondoOracle>> oracles_;
};

}  // namespace scribblez
