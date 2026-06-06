#pragma once

#include "scribblez/dictionary.h"

#include <memory>
#include <mutex>
#include <string>

namespace boost::program_options { class options_description; }

namespace scribblez {

// Process-wide source of truth for the lexicon (name + on-disk location) and
// the lazily-loaded Dictionary it resolves to. Consumed by GameRunner (for
// move generation) and by MacondoOracle (for the CGP `lex` clause), so both
// always agree without callers having to pass the name around.
//
// Usage:
//   - call Lexicon::instance().add_options(desc) before parsing argv;
//   - GameRunner calls Lexicon::instance().dict(), which triggers the load
//     on first call;
//   - MacondoOracle reads Lexicon::instance().name() when building its CGP.
class Lexicon {
 public:
  struct Params {
    // Lexicon name (e.g. "NWL23", "CSW24"). The .kwg is loaded from
    // <dir>/<name>.kwg.
    std::string name = "NWL23";

    // Directory holding the .kwg files. Defaults to the Docker mount layout
    // (/workspace/mount/lexica); rarely overridden.
    std::string dir = "/workspace/mount/lexica";
  };

  static Lexicon& instance();

  // Register --lexicon and --lexica-dir on the given options_description.
  // Mutates the stored params in-place; takes effect for every subsequent
  // dict() call. Rejected once the dictionary has been loaded.
  void add_options(boost::program_options::options_description& desc);

  // Replace the stored params. Throws if the dictionary has already been
  // loaded (we can't reload on the fly).
  void set_params(const Params& params);

  const std::string& name() const { return params_.name; }
  const std::string& dir() const { return params_.dir; }

  // Resolved path to the .kwg.
  std::string kwg_path() const { return params_.dir + "/" + params_.name + ".kwg"; }

  // Returns the loaded Dictionary, lazily reading <kwg_path()> on first call.
  // Throws on I/O failure.
  const Dictionary& dict();

 private:
  Lexicon() = default;

  std::mutex mutex_;
  Params params_;
  std::unique_ptr<Dictionary> dict_;
};

}  // namespace scribblez
