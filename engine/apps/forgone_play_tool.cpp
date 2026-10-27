// forgone_play_tool: writes the .forgone sidecar (data/forgone_play_log.h)
// beside each .slog, the forgone plays the leave inference model reads. The
// leave_inference workload's generators run it over each chunk they play,
// and its trainer over the held-out games.
//
//   forgone_play_tool --slog-dir chunk/ --threads 16
//   forgone_play_tool --slog-file a.slog --slog-file b.slog
//
// Every input gets a fresh sidecar, replacing any already there: the sidecar is
// a function of its .slog, which may have been regenerated under the same name.

#include "data/binary_log.h"
#include "data/forgone_play_log.h"
#include "data/slog_sampling.h"
#include "lexicon/dictionary.h"
#include "lexicon/lexicon.h"
#include "util/exception.h"
#include "util/io.h"
#include "util/misc.h"
#include "util/work_pool.h"

#include <boost/program_options.hpp>

#include <chrono>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace {

using namespace scribblez;

struct Options {
  std::string slog_dir;
  std::vector<std::string> slog_files;
  int threads = util::default_thread_count();
};

// The .slog at `path`, refusing anything that is not one this build reads.
std::vector<char> load_slog(const std::filesystem::path& path) {
  std::vector<char> bytes = util::read_file_bytes(path.string());
  const auto* hdr = reinterpret_cast<const binlog::FileHeader*>(bytes.data());
  if (bytes.size() < sizeof(binlog::FileHeader) || hdr->magic != binlog::kMagic ||
      hdr->version != binlog::kVersion) {
    throw util::CleanException("{}: not a version-{} .slog", path.string(), binlog::kVersion);
  }
  return bytes;
}

}  // namespace

int main(int argc, char** argv) {
  namespace po = boost::program_options;
  try {
    Options opt;
    po::options_description desc("forgone_play_tool options");
    desc.add_options()("help,h", "show this help and exit")(
      "slog-dir", po::value<std::string>(&opt.slog_dir),
      "directory of .slog files; each gets a .forgone sidecar")(
      "slog-file", po::value<std::vector<std::string>>(&opt.slog_files),
      "explicit .slog file to process (repeatable; overrides --slog-dir)")(
      "threads", po::value<int>(&opt.threads)->default_value(opt.threads), "parallel workers");
    Lexicon::instance().add_options(desc);
    util::parse_command_line(argc, argv, desc);
    util::set_worker_count(opt.threads);

    const Dictionary& dict = load_dictionary_or_throw();
    const auto t0 = std::chrono::steady_clock::now();
    uint64_t games = 0;
    for (const std::filesystem::path& path :
         binlog::resolve_slog_inputs(opt.slog_dir, opt.slog_files)) {
      const std::vector<char> slog = load_slog(path);
      write_forgone_sidecar(slog, dict, forgone_path_of(path.string()));
      games += reinterpret_cast<const binlog::FileHeader*>(slog.data())->num_games;
    }
    const double secs =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cerr << "forgone: " << games << " games in " << secs << " s, " << opt.threads
              << " threads\n";
    return 0;
  } catch (...) {
    return util::main_exit_code();
  }
}
