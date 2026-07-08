#pragma once

#include <cstdint>
#include <string>

namespace boost::program_options {
class options_description;
}

namespace scribblez::util {

// Parse argv against `desc`, handling --help and parse errors uniformly.
//
// On --help: prints `desc` (followed by `help_epilog`, if any) to stdout
// and throws CleanExit so main() can exit 0 without doing more work.
//
// On parse error: prints "Error: <what>\n\n<desc>\n" to stderr and throws
// scribblez::Exception (main() will exit non-zero without re-printing).
//
// On success: returns normally. The variables_map is fully `notify()`d, so
// every option's bound variable / notifier has already fired -- callers
// just read their populated Params structs and proceed.
void parse_command_line(int argc, char** argv, boost::program_options::options_description& desc,
                        const std::string& help_epilog = "");

// The number of logical processors available to this process (its CPU affinity
// mask, so taskset/cgroup cpusets are respected). This is the project-wide
// default for worker-thread counts: compute-bound thread pools (self-play
// games, Monte Carlo workers) default to using every available logical
// processor. The Python counterpart is scribblez.hardware.default_thread_count.
int default_thread_count();

// Returns a unique uint64_t nanosecond Unix timestamp. Thread-safe. If the
// current clock reading equals the last value returned, spins until the clock
// advances so that every returned value is strictly greater than the previous.
uint64_t get_unique_id();

}  // namespace scribblez::util
