#pragma once

#include "scribblez/board.h"
#include "scribblez/move.h"
#include "scribblez/rack.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace scribblez {

// A persistent `macondo` shell subprocess used as a static-equity oracle:
// HastyBot uses it to pick plays, and the human player uses it (best-effort)
// to annotate the cheat-mode move list with equity. Each instance owns one
// subprocess. NOT thread-safe -- each instance must be driven by at most one
// caller at a time; for parallel game threads use MacondoOraclePool to get
// per-thread instances.
class MacondoOracle {
 public:
  // Configuration knobs. Default-constructs to sensible values so callers
  // that just want Macondo with its defaults can skip configuring anything.
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
  };

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

  explicit MacondoOracle(const Params& params);
  ~MacondoOracle();

  MacondoOracle(const MacondoOracle&) = delete;
  MacondoOracle& operator=(const MacondoOracle&) = delete;

  // Spawns the subprocess on first call. Sends the position as a CGP `load`,
  // runs `gen N` (large enough to cover essentially any Scrabble position),
  // and parses the table back into per-legal-play equities. Throws on
  // subprocess failure.
  EvalResult evaluate(const Board& board, const Rack& my_rack, int my_score,
                      int opp_score, const std::vector<Move>& legal_plays);

 private:
  void ensure_started();

  struct Impl;                    // hides boost::process from the header
  std::unique_ptr<Impl> impl_;
  std::string binary_;
  std::string lexicon_;
};

}  // namespace scribblez
