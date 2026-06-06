#pragma once

#include "scribblez/board.h"
#include "scribblez/move.h"
#include "scribblez/rack.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

// Forward-declared so we can take a po::options_description by reference in
// Params::add_options() without dragging the heavyweight program_options
// headers into every consumer of macondo.h.
namespace boost::program_options { class options_description; }

namespace scribblez {

// A persistent `macondo` shell subprocess, shared by every consumer that needs
// Macondo's static-equity evaluator: HastyBot uses it to pick plays, and the
// human player uses it (best-effort) to annotate the cheat-mode move list with
// equity. Loaded once (lexicon/leaves) and driven a position at a time over
// its stdin/stdout.
//
// Usage: call set_params() at process startup (cheap; doesn't spawn anything).
// The first instance() call after that lazily builds the singleton from the
// stored params, and the first evaluate() call lazily spawns the subprocess.
// Consumers that never need Macondo never pay any cost.
class Macondo {
 public:
  // Configuration knobs. Owns whatever options Macondo needs from the user
  // (presently just the binary path). Default-constructs to sensible values
  // so callers that just want Macondo with its defaults can skip set_params.
  struct Params {
    // Path to the macondo binary. Defaults to the Docker-mount layout
    // (/workspace/mount/macondo/bin/shell, populated by setup_wizard.py);
    // override via --macondo for non-standard locations.
    std::string binary_path;

    // Lexicon name (e.g. "NWL23") used in the CGP `lex` clause; must match
    // a .kwg installed under macondo's data dir. Set by play_game from
    // GameRunner::Params::lexicon -- not its own CLI option.
    std::string lexicon = "NWL23";

    Params();

    // Register Params's user-facing options (presently just --macondo) on the
    // given options_description. Call this in main() before parsing the
    // command line, then pass the (mutated) Params to set_params().
    void add_options(boost::program_options::options_description& desc);
  };

  // Configure the singleton. May be called any number of times before the
  // first instance() call; the last set wins. After instance() has been
  // built, further set_params() calls throw (the subprocess can't be
  // reconfigured on the fly).
  static void set_params(const Params& params);

  // The singleton instance, lazily built from the most recent set_params()
  // (or from a default-constructed Params if set_params was never called).
  static Macondo& instance();

  // The results of one `gen` evaluation, matched back to the engine's
  // own legal_plays vector.
  struct EvalResult {
    // Macondo's static equity for each play, parallel to the input
    // legal_plays. nullopt for plays Macondo didn't return (Macondo only
    // emits the top N).
    std::vector<std::optional<double>> equities;
    // Index in legal_plays of Macondo's top-equity pick, or -1 if Macondo's
    // pick is not one of our legal_plays (e.g. it picked a pass we didn't
    // enumerate, or returned nothing).
    int best_index = -1;
  };

  // Spawns the subprocess on first call. Sends the position as a CGP `load`,
  // runs `gen N` (large enough to cover essentially any Scrabble position),
  // and parses the table back into per-legal-play equities. Throws on
  // subprocess failure.
  EvalResult evaluate(const Board& board, const Rack& my_rack, int my_score,
                      int opp_score, const std::vector<Move>& legal_plays);

  Macondo(const Macondo&) = delete;
  Macondo& operator=(const Macondo&) = delete;
  ~Macondo();

 private:
  Macondo(const std::string& binary, const std::string& lexicon);
  void ensure_started();

  struct Impl;                    // hides boost::process from the header
  std::unique_ptr<Impl> impl_;
  std::string binary_;
  std::string lexicon_;
};

}  // namespace scribblez
