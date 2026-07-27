#include "selfplay/streaming_game_producer.h"

#include "encoding/row_encoder.h"
#include "selfplay/game_sink.h"

#include <array>
#include <memory>
#include <random>
#include <utility>

namespace scribblez {
namespace binlog {

namespace {

// Per-worker sink: pick the sampled turn and encode the row BEFORE claiming a
// ring slot, so a game with no eligible turn is dropped without holding one.
// The task-specific work goes to a per-worker RowEncoder, leaving the sink
// task-agnostic; nothing here is shared across threads.
class RingBufferGameSink : public GameSink {
 public:
  RingBufferGameSink(StreamingRowBuffer& ring, std::unique_ptr<RowEncoder> encoder,
                     bool apply_symmetry, uint64_t seed, std::atomic<int64_t>* games_played,
                     std::atomic<int64_t>* games_dropped)
      : ring_(ring),
        encoder_(std::move(encoder)),
        apply_symmetry_(apply_symmetry),
        rng_(seed),
        games_played_(games_played),
        games_dropped_(games_dropped) {}

  void on_game(GameLogStorage&& storage, const std::array<int, 2>& /*seats*/) override {
    const GameLog view = storage.view();
    const int turn = encoder_->pick_turn(view, rng_);
    if (turn < 0) {
      games_dropped_->fetch_add(1, std::memory_order_relaxed);
      return;
    }
    const uint64_t r = ring_.claim_row();
    if (r == StreamingRowBuffer::kNoRow) return;  // buffer stopped
    const bool flip = apply_symmetry_ && (rng_() & 1ULL);
    encoder_->encode(view, turn, flip, ring_.row_dest(r));
    ring_.commit_row(r);
    games_played_->fetch_add(1, std::memory_order_relaxed);
  }

 private:
  StreamingRowBuffer& ring_;
  std::unique_ptr<RowEncoder> encoder_;
  bool apply_symmetry_;
  std::mt19937_64 rng_;
  std::atomic<int64_t>* games_played_;
  std::atomic<int64_t>* games_dropped_;
};

}  // namespace

StreamingGameProducer::StreamingGameProducer(const SelfPlayEngine::Params& engine_params,
                                             const PlayerFactory::Params& player_params,
                                             const Params& params, StreamingRowBuffer& ring)
    : engine_(engine_params, player_params), params_(params), ring_(ring) {}

StreamingGameProducer::~StreamingGameProducer() { stop(); }

void StreamingGameProducer::start() {
  if (started_) return;
  started_ = true;
  const int n = engine_.num_threads();
  workers_.reserve(static_cast<size_t>(n));
  for (int t = 0; t < n; ++t) {
    workers_.emplace_back([this, t]() { worker_loop(t); });
  }
}

void StreamingGameProducer::stop() {
  stopping_.store(true, std::memory_order_relaxed);
  ring_.stop();
  for (auto& w : workers_) {
    if (w.joinable()) w.join();
  }
  workers_.clear();
}

ProducerStats StreamingGameProducer::stats() const {
  ProducerStats s;
  s.games_played = games_played_.load(std::memory_order_relaxed);
  s.games_dropped = games_dropped_.load(std::memory_order_relaxed);
  return s;
}

void StreamingGameProducer::worker_loop(int thread_idx) {
  RingBufferGameSink sink(ring_, params_.make_encoder(), params_.apply_symmetry,
                          engine_.seed() + 0x9E3779B9ULL * static_cast<uint64_t>(thread_idx + 1),
                          &games_played_, &games_dropped_);
  while (!stopping_.load(std::memory_order_relaxed)) {
    const uint64_t game_idx = next_game_.fetch_add(1, std::memory_order_relaxed);
    const int seat0_player = static_cast<int>((engine_.seed() + game_idx) & 1ULL);
    const std::array<int, 2> seats = {seat0_player, 1 - seat0_player};
    engine_.play(thread_idx, seats, game_idx, sink);
  }
}

}  // namespace binlog
}  // namespace scribblez
