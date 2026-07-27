#pragma once

#include <cstdint>
#include <string>

namespace boost::program_options {
class options_description;
}

namespace scribblez::util {

// Parse argv against `desc`, handling --help and parse errors uniformly: both
// print (--help to stdout, an error to stderr) and then throw, CleanExit or
// scribblez::Exception, so main() need not print again. On success the
// variables_map has been notify()'d, so callers just read their Params.
void parse_command_line(int argc, char** argv, boost::program_options::options_description& desc,
                        const std::string& help_epilog = "");

// Logical processors available to this process, from its CPU affinity mask, so
// taskset/cgroup cpusets are respected. The project-wide default for
// compute-bound worker pools; the Python counterpart is
// scribblez.hardware.default_thread_count.
int default_thread_count();

// A unique nanosecond Unix timestamp; thread-safe. Spins until the clock
// advances, so every value is strictly greater than the last.
uint64_t get_unique_id();

}  // namespace scribblez::util
