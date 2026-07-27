#pragma once

// Runs unbounded HastyBot self-play across a thread pool, encoding each
// finished game's sampled position straight into a StreamingRowBuffer slot --
// no disk, no .slog round-trip. GameRunner's streaming counterpart: both drive
// a SelfPlayEngine, differing in the loop and the sink.
//
// Each worker thread owns its GameSink, which picks the sampled turn and
// encodes BEFORE claiming a ring-buffer row, so a game with no eligible turn is
// dropped without claiming one -- which would stall that slot forever.

#include "agent/player_factory.h"
#include "data/streaming_row_buffer.h"
#include "encoding/row_encoder.h"
#include "selfplay/self_play_engine.h"

#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>

namespace scribblez {
namespace binlog {

// Combined with RingStats by the trainer.
struct ProducerStats {
  int64_t games_played = 0;   // games whose sampled row was committed
  int64_t games_dropped = 0;  // games with no bag-nonempty (eligible) turn
};

class StreamingGameProducer {
 public:
  struct Params {
    // Selects the training task, and with it the row each worker encodes.
    RowEncoderFactory make_encoder;
    bool apply_symmetry = true;
  };

  // Builds the agent pool and binds to `ring`, producing nothing until start().
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
