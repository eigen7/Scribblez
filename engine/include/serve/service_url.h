#pragma once

#include <string>

namespace scribblez {

// The browser-facing URL for a gateway-routed dev service, read from the
// DEVENV_SERVICE_URL_<NAME> environment variable run_docker.py exports per
// devenv.toml [services] entry (see subtrees/devenv_utils/GATEWAY.md).
//
// Only while `port` matches `default_port`: a user-overridden port has no
// gateway route, the route being fixed to the default, so it falls back to
// "http://localhost:<port>".
//
// Only the URL shown to the user goes through here; internal readiness probes
// and proxy targets keep using localhost directly.
std::string service_url(const std::string& service, int port, int default_port);

}  // namespace scribblez
