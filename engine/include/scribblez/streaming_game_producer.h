#pragma once

// StreamingGameProducer runs unbounded HastyBot self-play across a thread pool,
// encoding each finished game's sampled position directly into a
// StreamingRowBuffer slot -- no disk, no .slog round-trip. It is the streaming
// counterpart to GameRunner: both drive a SelfPlayEngine, but this one loops
// forever (until stopped) and its sink writes rows into the ring buffer instead
// of files.
//
// Each worker thread owns its own GameSink (holding a PositionEncoder and a
// sampler RNG). The sink picks the sampled turn and encodes BEFORE claiming a
// ring-buffer row, so a game with no eligible sampled turn is dropped without
// ever claiming a slot row (which would otherwise stall that slot forever).

#include "scribblez/player_factory.h"
#include "scribblez/self_play_engine.h"
#include "scribblez/streaming_row_buffer.h"

#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>

namespace scribblez {
namespace binlog {

// Game-production counters (combined with RingStats by the trainer).
struct ProducerStats {
  int64_t games_played = 0;   // games whose sampled row was committed
  int64_t games_dropped = 0;  // games with no bag-nonempty (eligible) turn
};

class StreamingGameProducer {
 public:
  struct Params {
    bool post_move = true;
    bool apply_symmetry = true;
  };

  // Builds the agent pool (via SelfPlayEngine) and binds to `ring`. Does not
  // start producing until start() is called.
  StreamingGameProducer(const SelfPlayEngine::Params& engine_params,
                        const PlayerFactory::Params& player_params, const Params& params,
                        StreamingRowBuffer& ring);
  ~StreamingGameProducer();

  StreamingGameProducer(const StreamingGameProducer&) = delete;
  StreamingGameProducer& operator=(const StreamingGameProducer&) = delete;

  // Spawn one producer thread per agent pair. Idempotent.
  void start();

  // Stop the ring buffer and join all producer threads. Idempotent.
  void stop();

  ProducerStats stats() const;

 private:
  void worker_loop(int thread_idx);

  SelfPlayEngine engine_;
  Params params_;
  StreamingRowBuffer& ring_;

  std::vector<std::thread> workers_;
  std::atomic<uint64_t> next_game_{0};
  std::atomic<int64_t> games_played_{0};
  std::atomic<int64_t> games_dropped_{0};
  std::atomic<bool> stopping_{false};
  bool started_ = false;
};

}  // namespace binlog
}  // namespace scribblez
