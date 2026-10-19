#include "data/pool_log.h"

#include "data/sidecar_io.h"
#include "util/assert.h"
#include "util/exception.h"

#include <algorithm>
#include <cstring>
#include <fstream>

namespace scribblez {

namespace {

// Copies `s` into a fixed-width field, NUL-padding the rest; too long throws.
template <size_t N>
void set_field(char (&field)[N], const std::string& s, const char* what) {
  if (s.size() > N) throw util::Exception("PoolWriter: {} '{}' exceeds {} bytes", what, s, N);
  std::memset(field, 0, N);
  std::memcpy(field, s.data(), s.size());
}

PoolFileHeader make_header(const PoolSources& s) {
  PoolFileHeader h{};
  h.magic = kPoolMagic;
  h.version = kPoolFormatVersion;
  h.flags = s.leave_model_hash.empty() ? 0 : kPoolFlagLeaveInference;
  h.horizon_plies = uint16_t(s.horizon_plies);
  h.rollouts = uint32_t(s.rollouts);
  h.win_table_format_version = s.win_table_format_version;
  h.win_table_generation = s.win_table_generation;
  set_field(h.win_table_hash, s.win_table_hash, "win table hash");
  set_field(h.win_table_tag, s.win_table_tag, "win table tag");
  set_field(h.leave_model_hash, s.leave_model_hash, "leave model hash");
  return h;
}

}  // namespace

PoolWriter::PoolWriter(const std::string& path, const PoolSources& sources) : path_(path) {
  const PoolFileHeader h = make_header(sources);
  append_bytes(&buffer_, &h, sizeof(h));
}

PoolWriter::~PoolWriter() {
  if (!closed_ && std::uncaught_exceptions() == uncaught_at_open_) close();
}

void PoolWriter::add_position(const PoolPosition& p) {
  RELEASE_ASSERT(!closed_);
  const size_t k = p.moves.size();
  RELEASE_ASSERT(p.equity_ranks.size() == k && p.labels.size() == k);
  const PoolPositionHeader h{p.at.game_idx, p.at.turn_idx, uint32_t(k), p.num_legal_moves};
  append_bytes(&buffer_, &h, sizeof(h));
  for (size_t c = 0; c < k; ++c) {
    const PoolLabel& l = p.labels[c];
    const PoolCandidate cand{p.moves[c], float(l.win_prob), float(l.paired_se),
                             float(l.q), p.equity_ranks[c], uint16_t(l.rollouts),
                             0};
    append_bytes(&buffer_, &cand, sizeof(cand));
  }
  ++num_positions_;
}

void PoolWriter::close() {
  RELEASE_ASSERT(!closed_);
  closed_ = true;
  reinterpret_cast<PoolFileHeader*>(buffer_.data())->num_positions = num_positions_;
  write_atomically(path_, buffer_);
}

PoolReader::PoolReader(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw util::Exception("PoolReader: cannot open {}", path);
  buffer_.resize(size_t(f.tellg()));
  f.seekg(0);
  f.read(buffer_.data(), std::streamsize(buffer_.size()));
  const std::string context = "PoolReader: " + path;
  size_t off = 0;
  take_bytes(buffer_, &off, sizeof(PoolFileHeader), context);
  if (header().magic != kPoolMagic) throw util::Exception("{}: bad magic", context);
  if (header().version != kPoolFormatVersion) {
    throw util::Exception("{}: format version {}, but this build reads {}", context,
                          header().version, kPoolFormatVersion);
  }
  for (uint32_t i = 0; i < header().num_positions; ++i) {
    Position pos{};
    pos.header = reinterpret_cast<const PoolPositionHeader*>(
      take_bytes(buffer_, &off, sizeof(PoolPositionHeader), context));
    pos.candidates = reinterpret_cast<const PoolCandidate*>(
      take_bytes(buffer_, &off, pos.header->num_candidates * sizeof(PoolCandidate), context));
    positions_.push_back(pos);
  }
}

}  // namespace scribblez
