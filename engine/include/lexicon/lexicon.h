#pragma once

#include "lexicon/dictionary.h"

#include <memory>
#include <mutex>
#include <string>

namespace boost::program_options {
class options_description;
}

namespace scribblez {

// Process-wide choice of lexicon and the Dictionary it loads, so consumers
// agree on it without passing a name around. Configure it (add_options() before
// parsing argv, or set_params()) before the first dict() call, which loads.
class Lexicon {
 public:
  struct Params {
    std::string name = "NWL23";  // loaded from <dir>/<name>.kwg
    std::string dir = "/workspace/mount/lexica";
  };

  static Lexicon& instance();

  // Registers --lexicon and --lexica-dir, which set the stored params when
  // notified. Like set_params(), notifying after dict() has loaded throws.
  void add_options(boost::program_options::options_description& desc);

  // Throws if dict() has already loaded; there is no reloading.
  void set_params(const Params& params);

  const std::string& name() const { return params_.name; }
  const std::string& dir() const { return params_.dir; }

  std::string kwg_path() const { return params_.dir + "/" + params_.name + ".kwg"; }

  // Loads kwg_path() on first call. Throws on I/O failure.
  const Dictionary& dict();

 private:
  Lexicon() = default;

  // The add_options() notifier for one field. Call without mutex_ held.
  void set_param(std::string Params::* field, const std::string& value);

  // Call with mutex_ held.
  void throw_if_loaded() const;

  std::mutex mutex_;
  Params params_;
  std::unique_ptr<Dictionary> dict_;
};

// Lexicon::instance().dict() for command-line entry points. A missing .kwg is
// a setup mistake rather than a bug, so failure is rethrown as a
// util::CleanException naming the path and the setup step that installs it.
const Dictionary& load_dictionary_or_throw();

}  // namespace scribblez
