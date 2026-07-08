#include "util/misc.h"

#include "util/exception.h"

#include <boost/program_options.hpp>

#include <atomic>
#include <chrono>
#include <iostream>
#include <sched.h>

namespace scribblez::util {

void parse_command_line(int argc, char** argv, boost::program_options::options_description& desc,
                        const std::string& help_epilog) {
  namespace po = boost::program_options;
  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n\n" << desc << "\n";
    throw Exception(e.what());
  }
  if (vm.count("help")) {
    std::cout << desc << "\n";
    if (!help_epilog.empty()) std::cout << help_epilog;
    throw CleanExit();
  }
}

int default_thread_count() {
  cpu_set_t set;
  CPU_ZERO(&set);
  sched_getaffinity(0, sizeof(set), &set);
  return CPU_COUNT(&set);
}

uint64_t get_unique_id() {
  static std::atomic<uint64_t> last_id{0};
  while (true) {
    uint64_t ts = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
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
