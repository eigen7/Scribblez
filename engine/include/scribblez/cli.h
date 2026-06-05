#pragma once

#include <string>

namespace boost::program_options { class options_description; }

namespace scribblez {

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
void parse_command_line(int argc, char** argv,
                        boost::program_options::options_description& desc,
                        const std::string& help_epilog = "");

}  // namespace scribblez
