#include "util/misc.h"

#include "util/exception.h"

#include <boost/program_options.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <fstream>
#include <iostream>
#include <optional>
#include <sched.h>
#include <sstream>

namespace scribblez::util {

namespace {

// CPU count implied by a cgroup CPU quota (ceil(quota / period)), or
// std::nullopt when no quota is in force: cgroup v2's "max", cgroup v1's
// quota of -1, or neither version's files present (not running under Linux
// cgroups at all). v2 is tried first; v1 is only consulted when v2's file is
// absent, since Runpod hosts may run either.
std::optional<int> cgroup_quota_cpus() {
  if (std::ifstream v2("/sys/fs/cgroup/cpu.max"); v2) {
    std::string quota_str;
    long period = 0;
    v2 >> quota_str >> period;
    if (quota_str == "max") return std::nullopt;
    return static_cast<int>((std::stol(quota_str) + period - 1) / period);
  }

  std::ifstream quota_file("/sys/fs/cgroup/cpu/cpu.cfs_quota_us");
  std::ifstream period_file("/sys/fs/cgroup/cpu/cpu.cfs_period_us");
  if (!quota_file || !period_file) return std::nullopt;
  long quota = 0, period = 0;
  quota_file >> quota;
  period_file >> period;
  if (quota < 0) return std::nullopt;
  return static_cast<int>((quota + period - 1) / period);
}

}  // namespace

void parse_command_line(int argc, char** argv, boost::program_options::options_description& desc,
                        const std::string& help_epilog) {
  namespace po = boost::program_options;
  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
  } catch (const std::exception& e) {
    // program_options only streams; there is no to-string API for desc.
    std::ostringstream usage;
    usage << desc;
    throw CleanException("{}\n\n{}", e.what(), usage.str());
  }
  if (vm.count("help")) {
    std::cout << desc << "\n";
    if (!help_epilog.empty()) std::cout << help_epilog;
    throw CleanExit();
  }
}

int main_exit_code() {
  try {
    throw;
  } catch (const CleanExit&) {
    return 0;
  } catch (const CleanException& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  } catch (const std::exception& e) {
    std::cerr << "Unexpected error: " << e.what() << "\n";
    return 1;
  } catch (...) {
    std::cerr << "Unexpected error of unknown type\n";
    return 1;
  }
}

int default_thread_count() {
  cpu_set_t set;
  CPU_ZERO(&set);
  sched_getaffinity(0, sizeof(set), &set);
  int affinity = CPU_COUNT(&set);
  std::optional<int> quota_cpus = cgroup_quota_cpus();
  if (!quota_cpus) return affinity;
  return std::max(1, std::min(affinity, *quota_cpus));
}

uint64_t get_unique_id() {
  static std::atomic<uint64_t> last_id{0};
  while (true) {
    uint64_t ts = uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                             std::chrono::system_clock::now().time_since_epoch())
                             .count());
    uint64_t expected = last_id.load(std::memory_order_relaxed);
    if (ts <= expected) continue;  // clock hasn't advanced past last returned value; retry
    if (last_id.compare_exchange_weak(expected, ts, std::memory_order_acq_rel,
                                      std::memory_order_relaxed)) {
      return ts;
    }
    // Another thread updated last_id between our load and CAS; retry.
  }
}

}  // namespace scribblez::util
