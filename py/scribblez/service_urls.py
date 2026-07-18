"""Browser-facing URLs for the project's gateway-routed dev services.

run_docker.py exports one DEVENV_SERVICE_URL_<NAME> variable per entry in
devenv.toml's [services] table -- the URL at which the host browser reaches
that service through the gateway, e.g.
DEVENV_SERVICE_URL_DASH=http://scribblez-dash.localhost (see
submodules/devenv_utils/GATEWAY.md). Tools that print a URL for the user to
open read it through service_url(); when the variable is absent (running
outside the dev container) the plain localhost form is the right fallback.
"""

import os


def _env_var(service: str) -> str:
    return "DEVENV_SERVICE_URL_" + service.upper().replace("-", "_")


def service_url(service: str, port: int, default_port: int) -> str:
    """The browser URL for `service`.

    Returns the gateway URL exported by run_docker.py, but only when `port`
    still matches `default_port`: a user-overridden port has no gateway route
    (the route is fixed to the default), so it falls back to
    http://localhost:<port>.
    """
    url = os.environ.get(_env_var(service))
    if url and port == default_port:
        return url
    return f"http://localhost:{port}"
