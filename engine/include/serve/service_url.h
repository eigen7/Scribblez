#pragma once

#include <string>

namespace scribblez {

// The browser-facing URL for a gateway-routed dev service. run_docker.py
// exports one DEVENV_SERVICE_URL_<NAME> environment variable per devenv.toml
// [services] entry -- the URL at which the host browser reaches that service
// through the gateway (e.g. http://scribblez-web.localhost); see
// submodules/devenv_utils/GATEWAY.md.
//
// Returns that env URL, but only when `port` still matches `default_port`: a
// user-overridden port has no gateway route (the route is fixed to the
// default), so it falls back to "http://localhost:<port>". The env-var suffix
// is `service` uppercased with '-' mapped to '_'.
//
// Only the URL shown to the user (to open in a browser) goes through here;
// internal readiness probes and proxy targets keep using localhost directly.
std::string service_url(const std::string& service, int port, int default_port);

}  // namespace scribblez
