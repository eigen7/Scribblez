#include "belief/leave_table.h"

#include "belief/leave_prior.h"
#include "data/sidecar_io.h"
#include "game/tile_counts.h"
#include "util/exception.h"
#include "util/hash.h"
#include "util/io.h"

#include <cstring>
#include <numeric>
#include <utility>

namespace scribblez::belief {

namespace {

LeaveTableRow make_row(const Rack& leave, float value) {
  LeaveTableRow row{};
  const TileCounts counts = leave.counts();
  for (int l = 0; l < TILE_KINDS; ++l) {
    row.counts[l] = uint8_t(counts.count(Tile::of(l)));
    if (row.counts[l] > 0) row.mask |= 1u << l;
  }
  row.value = value;
  return row;
}

std::span<const char> row_bytes(const std::vector<LeaveTableRow>& rows) {
  return {reinterpret_cast<const char*>(rows.data()), rows.size() * sizeof(LeaveTableRow)};
}

// Copies `s` into a fixed-width header field, NUL-padding the remainder.
template <size_t N>
void set_field(char (&field)[N], const std::string& s, const char* what) {
  if (s.size() > N) throw util::Exception("leave table {} '{}' exceeds {} bytes", what, s, N);
  std::memset(field, 0, N);
  std::memcpy(field, s.data(), s.size());
}

LeaveTableHeader read_header(const std::vector<char>& bytes, const std::string& path) {
  LeaveTableHeader h;
  if (bytes.size() < sizeof(h))
    throw util::Exception("{}: too short for a leave table header", path);
  std::memcpy(&h, bytes.data(), sizeof(h));
  if (h.magic != kLeaveTableMagic) throw util::Exception("{}: not a leave table (bad magic)", path);
  if (h.version != kLeaveTableFormatVersion) {
    throw util::Exception("{}: leave table format version {}, but this build reads version {}",
                          path, h.version, kLeaveTableFormatVersion);
  }
  return h;
}

}  // namespace

LeaveTable::LeaveTable(const LeaveTableHeader& header, std::vector<LeaveTableRow> rows)
    : header_(header), rows_(std::move(rows)) {
  for (int k = 1; k <= kLeaveTableMaxLength; ++k)
    offsets_[k] = offsets_[k - 1] + header_.row_counts[k - 1];
}

LeaveTable LeaveTable::build(const HastyEquity& equity, const std::string& lexicon) {
  LeaveTableHeader h{};
  h.magic = kLeaveTableMagic;
  h.version = kLeaveTableFormatVersion;
  set_field(h.lexicon, lexicon, "lexicon");
  set_field(h.klv_hash, equity.klv_hash(), "KLV hash");
  std::vector<LeaveTableRow> rows;
  for (int k = 1; k <= kLeaveTableMaxLength; ++k) {
    const std::vector<ScoredLeave> leaves = enumerate_leaves(TileCounts::full_distribution(), k);
    h.row_counts[k - 1] = uint32_t(leaves.size());
    for (const ScoredLeave& s : leaves)
      rows.push_back(make_row(s.leave, float(equity.leave_value(s.leave))));
  }
  set_field(h.content_id, util::content_hash(row_bytes(rows)), "content id");
  return LeaveTable(h, std::move(rows));
}

LeaveTable LeaveTable::load(const std::string& path) {
  const std::vector<char> bytes = util::read_file_bytes(path);
  const LeaveTableHeader h = read_header(bytes, path);
  const size_t num_rows = std::accumulate(h.row_counts.begin(), h.row_counts.end(), size_t(0));
  const size_t expected = sizeof(h) + num_rows * sizeof(LeaveTableRow);
  if (bytes.size() != expected) {
    throw util::Exception("{}: {} bytes, but its row counts imply {}", path, bytes.size(),
                          expected);
  }
  std::vector<LeaveTableRow> rows(num_rows);
  std::memcpy(rows.data(), bytes.data() + sizeof(h), num_rows * sizeof(LeaveTableRow));
  return LeaveTable(h, std::move(rows));
}

void LeaveTable::write(const std::string& path) const {
  std::vector<char> bytes;
  append_bytes(&bytes, &header_, sizeof(header_));
  const std::span<const char> payload = row_bytes(rows_);
  bytes.insert(bytes.end(), payload.begin(), payload.end());
  write_atomically(path, bytes);
}

std::string LeaveTable::lexicon() const {
  return {header_.lexicon, strnlen(header_.lexicon, kLeaveTableLexiconSize)};
}

std::span<const LeaveTableRow> LeaveTable::rows(int k) const {
  if (k < 1 || k > kLeaveTableMaxLength) throw util::Exception("no leave table for length {}", k);
  return std::span<const LeaveTableRow>(rows_).subspan(offsets_[k - 1],
                                                       offsets_[k] - offsets_[k - 1]);
}

}  // namespace scribblez::belief
