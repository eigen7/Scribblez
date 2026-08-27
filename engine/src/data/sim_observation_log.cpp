#include "data/sim_observation_log.h"

#include "util/assert.h"
#include "util/exception.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <unistd.h>

namespace scribblez {

namespace {

void append_bytes(std::vector<char>* buffer, const void* data, size_t size) {
  const char* p = static_cast<const char*>(data);
  buffer->insert(buffer->end(), p, p + size);
}

// A fixed-width, NUL-padded model-hash header field read back as a string.
std::string header_hash_field(const char* field, size_t size) {
  return std::string(field, strnlen(field, size));
}

}  // namespace

SimObsWriter::SimObsWriter(const std::string& path, uint32_t flags,
                           const std::string& proposer_hash, const std::string& leaf_model_hash,
                           int horizon_plies)
    : path_(path) {
  RELEASE_ASSERT(leaf_model_hash.empty() == (horizon_plies == 0),
                 "value-truncated sims carry a leaf model hash and a horizon, "
                 "terminal sims neither");
  SimObsFileHeader hdr{};
  hdr.magic = kSimObsMagic;
  hdr.version = kSimObsVersion;
  hdr.horizon_plies = uint16_t(horizon_plies);
  hdr.num_positions = 0;  // patched in close()
  hdr.flags = flags;
  std::memcpy(hdr.proposer_hash, proposer_hash.data(),
              std::min(proposer_hash.size(), sizeof(hdr.proposer_hash)));
  std::memcpy(hdr.leaf_model_hash, leaf_model_hash.data(),
              std::min(leaf_model_hash.size(), sizeof(hdr.leaf_model_hash)));
  append_bytes(&buffer_, &hdr, sizeof(hdr));
}

SimObsWriter::~SimObsWriter() {
  if (!closed_) close();
}

void SimObsWriter::add_position(uint32_t game_index, uint32_t turn_index,
                                const std::vector<Move>& candidates,
                                const std::vector<SimObservation>& observations, uint32_t rollouts,
                                uint64_t base_seed, uint32_t num_legal_moves,
                                uint32_t position_flags) {
  RELEASE_ASSERT(!closed_);
  RELEASE_ASSERT(candidates.size() == observations.size());
  SimObsPositionHeader ph{};
  ph.game_index = game_index;
  ph.turn_index = turn_index;
  ph.num_candidates = candidates.size();
  ph.rollouts = rollouts;
  ph.base_seed = base_seed;
  ph.num_legal_moves = num_legal_moves;
  ph.flags = position_flags;
  append_bytes(&buffer_, &ph, sizeof(ph));
  for (size_t c = 0; c < candidates.size(); ++c) {
    SimObsRecord rec{};
    rec.move = candidates[c];
    rec.obs = observations[c];
    append_bytes(&buffer_, &rec, sizeof(rec));
  }
  ++num_positions_;
}

void SimObsWriter::close() {
  RELEASE_ASSERT(!closed_);
  closed_ = true;
  SimObsFileHeader* hdr = reinterpret_cast<SimObsFileHeader*>(buffer_.data());
  hdr->num_positions = num_positions_;
  // Temp-file + rename so the .sobs appears atomically: an interrupted run
  // never leaves a truncated file that a resume (which skips existing
  // sidecars) would silently keep.
  const std::string tmp = std::format("{}.tmp.{}", path_, ::getpid());
  {
    std::ofstream f(tmp, std::ios::binary);
    if (!f) throw util::Exception("SimObsWriter: cannot open {}", tmp);
    f.write(buffer_.data(), std::streamsize(buffer_.size()));
  }
  std::filesystem::rename(tmp, path_);
}

std::string SimObsReader::proposer_hash() const {
  return header_hash_field(header().proposer_hash, sizeof(header().proposer_hash));
}

std::string SimObsReader::leaf_model_hash() const {
  return header_hash_field(header().leaf_model_hash, sizeof(header().leaf_model_hash));
}

SimObsReader::SimObsReader(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw util::Exception("SimObsReader: cannot open {}", path);
  const std::streamsize size = f.tellg();
  f.seekg(0);
  buffer_.resize(size_t(size));
  f.read(buffer_.data(), size);

  if (buffer_.size() < sizeof(SimObsFileHeader)) {
    throw util::Exception("SimObsReader: truncated header in {}", path);
  }
  const SimObsFileHeader* hdr = reinterpret_cast<const SimObsFileHeader*>(buffer_.data());
  if (hdr->magic != kSimObsMagic) throw util::Exception("SimObsReader: bad magic in {}", path);
  if (hdr->version != kSimObsVersion) {
    throw util::Exception("SimObsReader: version mismatch in {} (file={} code={})", path,
                          hdr->version, kSimObsVersion);
  }

  size_t off = sizeof(SimObsFileHeader);
  positions_.reserve(hdr->num_positions);
  for (uint32_t p = 0; p < hdr->num_positions; ++p) {
    if (off + sizeof(SimObsPositionHeader) > buffer_.size()) {
      throw util::Exception("SimObsReader: truncated position header in {}", path);
    }
    const SimObsPositionHeader* ph =
      reinterpret_cast<const SimObsPositionHeader*>(buffer_.data() + off);
    off += sizeof(SimObsPositionHeader);
    const size_t records_bytes = size_t(ph->num_candidates) * sizeof(SimObsRecord);
    if (off + records_bytes > buffer_.size()) {
      throw util::Exception("SimObsReader: truncated records in {}", path);
    }
    positions_.push_back({ph, reinterpret_cast<const SimObsRecord*>(buffer_.data() + off)});
    off += records_bytes;
  }
}

}  // namespace scribblez
