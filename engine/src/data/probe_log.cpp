#include "data/probe_log.h"

#include "data/sidecar_io.h"
#include "util/assert.h"
#include "util/exception.h"

#include <algorithm>
#include <cstring>
#include <fstream>

namespace scribblez {
namespace {

ProbeRecord record_of(const Rollout& r, const RolloutTrace& t, int mover, int candidate,
                      int probe) {
  ProbeRecord rec{};
  rec.mover_rack = t.initial_racks[size_t(mover)];
  rec.opp_rack = t.initial_racks[size_t(1 - mover)];
  rec.p_win = float(r.p_win);
  rec.p_draw = float(r.p_draw);
  rec.p_loss = float(r.p_loss);
  rec.delta = float(r.delta);
  rec.delta_sq = float(r.delta_sq);
  rec.candidate = uint16_t(candidate);
  rec.probe = uint16_t(probe);
  rec.truncated = t.truncated ? 1 : 0;
  rec.num_turns = uint8_t(t.turns.size());
  return rec;
}

// The next `bytes` of `buf` at `*off`, advancing it.
const char* take(const std::vector<char>& buf, size_t* off, size_t bytes, const std::string& path) {
  if (*off + bytes > buf.size()) throw util::Exception("ProbeReader: truncated {}", path);
  const char* at = buf.data() + *off;
  *off += bytes;
  return at;
}

}  // namespace

ProbeWriter::ProbeWriter(const std::string& path, uint16_t flags,
                         const std::string& leaf_model_hash, const std::string& lexicon,
                         int horizon_plies, int probes)
    : path_(path), probes_(probes) {
  ProbeFileHeader hdr{};
  hdr.magic = kProbeMagic;
  hdr.version = kProbeVersion;
  hdr.flags = flags;
  hdr.num_positions = 0;  // patched in close()
  hdr.horizon_plies = uint16_t(horizon_plies);
  hdr.probes = uint16_t(probes);
  std::memcpy(hdr.leaf_model_hash, leaf_model_hash.data(),
              std::min(leaf_model_hash.size(), sizeof(hdr.leaf_model_hash)));
  std::memcpy(hdr.lexicon, lexicon.data(), std::min(lexicon.size(), sizeof(hdr.lexicon)));
  append_bytes(&buffer_, &hdr, sizeof(hdr));
}

ProbeWriter::~ProbeWriter() {
  if (!closed_) close();
}

void ProbeWriter::add_position(const ProbePosition& p) {
  RELEASE_ASSERT(!closed_);
  const size_t k = p.moves.size();
  RELEASE_ASSERT(p.equities.size() == k && p.equity_ranks.size() == k && p.strata.size() == k &&
                 p.rollouts.size() == k && p.traces.size() == k);
  std::vector<binlog::TurnBlob> turns;
  std::vector<ProbeRecord> records;
  for (size_t c = 0; c < k; ++c) {
    RELEASE_ASSERT(int(p.rollouts[c].size()) == probes_ && int(p.traces[c].size()) == probes_);
    for (int i = 0; i < probes_; ++i) {
      const RolloutTrace& t = p.traces[c][size_t(i)];
      records.push_back(record_of(p.rollouts[c][size_t(i)], t, p.mover, int(c), i));
      for (const TurnRecord& turn : t.turns) turns.push_back({turn.move, turn.drawn});
    }
  }
  ProbePositionHeader ph{};
  ph.game_index = p.at.game_idx;
  ph.turn_index = p.at.turn_idx;
  ph.base_seed = p.base_seed;
  ph.num_candidates = uint32_t(k);
  ph.num_legal_moves = p.num_legal_moves;
  ph.num_turns = uint32_t(turns.size());
  append_bytes(&buffer_, &ph, sizeof(ph));
  for (size_t c = 0; c < k; ++c) {
    ProbeCandidate pc{};
    pc.move = p.moves[c];
    pc.equity = p.equities[c];
    pc.equity_rank = p.equity_ranks[c];
    pc.stratum = p.strata[c];
    append_bytes(&buffer_, &pc, sizeof(pc));
  }
  append_bytes(&buffer_, records.data(), records.size() * sizeof(ProbeRecord));
  append_bytes(&buffer_, turns.data(), turns.size() * sizeof(binlog::TurnBlob));
  ++num_positions_;
}

void ProbeWriter::close() {
  RELEASE_ASSERT(!closed_);
  closed_ = true;
  reinterpret_cast<ProbeFileHeader*>(buffer_.data())->num_positions = num_positions_;
  write_atomically(path_, buffer_);
}

ProbeReader::ProbeReader(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw util::Exception("ProbeReader: cannot open {}", path);
  buffer_.resize(size_t(f.tellg()));
  f.seekg(0);
  f.read(buffer_.data(), std::streamsize(buffer_.size()));
  if (buffer_.size() < sizeof(ProbeFileHeader))
    throw util::Exception("ProbeReader: truncated header in {}", path);
  if (header().magic != kProbeMagic) throw util::Exception("ProbeReader: bad magic in {}", path);
  if (header().version != kProbeVersion) {
    throw util::Exception("ProbeReader: version mismatch in {} (file={} code={})", path,
                          header().version, kProbeVersion);
  }
  size_t off = sizeof(ProbeFileHeader);
  for (uint32_t i = 0; i < header().num_positions; ++i) {
    Position pos{};
    pos.header = reinterpret_cast<const ProbePositionHeader*>(
      take(buffer_, &off, sizeof(ProbePositionHeader), path));
    const size_t k = pos.header->num_candidates;
    pos.candidates = reinterpret_cast<const ProbeCandidate*>(
      take(buffer_, &off, k * sizeof(ProbeCandidate), path));
    pos.records = reinterpret_cast<const ProbeRecord*>(
      take(buffer_, &off, k * size_t(probes()) * sizeof(ProbeRecord), path));
    pos.turns = reinterpret_cast<const binlog::TurnBlob*>(
      take(buffer_, &off, size_t(pos.header->num_turns) * sizeof(binlog::TurnBlob), path));
    positions_.push_back(pos);
  }
}

}  // namespace scribblez
