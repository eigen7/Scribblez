#pragma once

#include <boost/program_options/options_description.hpp>

#include <string>

namespace scribblez {

// Render an agent's `--help` block for `play_game --help`: the human-readable
// `description` (already formatted with its own indentation and trailing
// newline) followed by the option list rendered from `opts`, or "(none)" when
// `opts` declares no options. Each agent builds one options_description that its
// from_spec() parses with and its options_help() renders here, so the parsed
// options and the documented options can never drift apart.
std::string agent_options_help(const std::string& description,
                               const boost::program_options::options_description& opts);

}  // namespace scribblez
