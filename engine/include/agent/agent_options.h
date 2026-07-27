#pragma once

#include <boost/program_options/options_description.hpp>

#include <string>

namespace scribblez {

// Render an agent's `--help` block: its description followed by its option
// list. An agent renders the same options_description it parses with, so the
// documented options cannot drift from the accepted ones.
std::string agent_options_help(const std::string& description,
                               const boost::program_options::options_description& opts);

}  // namespace scribblez
