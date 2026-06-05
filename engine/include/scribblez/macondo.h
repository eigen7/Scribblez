#pragma once

#include "scribblez/board.h"
#include "scribblez/move.h"
#include "scribblez/rack.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace scribblez {

// A persistent `macondo` shell subprocess, shared by every consumer that needs
// Macondo's static-equity evaluator: HastyBot uses it to pick plays, and the
// human player uses it (best-effort) to annotate the cheat-mode move list with
// equity. Loaded once (lexicon/leaves) and driven a position at a time over
// its stdin/stdout.
//
// Usage: call initialize() once at process startup with the path to the
// `macondo` binary, then access via instance(). The subprocess itself is
// spawned lazily on the first evaluate() call -- so configuring the path
// without ever evaluating is free.
class Macondo {
 public:
  // Configure the singleton with the path to the macondo binary. May be
  // called only once per process; subsequent calls throw.
  static void initialize(std::string binary_path);

  // True iff initialize() has been called.
  static bool initialized();

  // The singleton instance. Throws std::runtime_error if not initialized.
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
  explicit Macondo(std::string binary);
  void ensure_started();

  struct Impl;                    // hides boost::process from the header
  std::unique_ptr<Impl> impl_;
  std::string binary_;
};

}  // namespace scribblez
