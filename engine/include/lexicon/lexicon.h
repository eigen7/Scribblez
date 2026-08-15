#pragma once

#include "lexicon/dictionary.h"

#include <memory>
#include <mutex>
#include <string>

namespace boost::program_options {
class options_description;
}

namespace scribblez {

// Process-wide source of truth for the lexicon (name + on-disk location) and
// the Dictionary it resolves to, so every consumer agrees without callers
// passing the name around. Register the options before parsing argv; the first
// dict() call loads.
class Lexicon {
 public:
  struct Params {
    std::string name = "NWL23";  // e.g. "NWL23", "CSW24"; loaded from <dir>/<name>.kwg
    std::string dir = "/workspace/mount/lexica";
  };

  static Lexicon& instance();

  // Registers --lexicon and --lexica-dir against the stored params. Rejected
  // once the dictionary has been loaded.
  void add_options(boost::program_options::options_description& desc);

  // Throws if the dictionary has already been loaded; there is no reloading.
  void set_params(const Params& params);

  const std::string& name() const { return params_.name; }
  const std::string& dir() const { return params_.dir; }

  std::string kwg_path() const { return params_.dir + "/" + params_.name + ".kwg"; }

  // Lazily reads kwg_path() on first call. Throws on I/O failure.
  const Dictionary& dict();

 private:
  Lexicon() = default;

  std::mutex mutex_;
  Params params_;
  std::unique_ptr<Dictionary> dict_;
};

// Lexicon::instance().dict(), but a failure is rethrown as util::CleanException
// with the error explained as an uninstalled lexicon, naming the path and the
// setup step that provides it. The way every command-line entry point loads the
// dictionary: a missing .kwg is a setup mistake, not an internal error, and
// deserves an actionable message rather than a raw I/O one.
const Dictionary& load_dictionary_or_throw();

}  // namespace scribblez
