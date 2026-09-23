"""Browser-facing URLs for the project's gateway-routed dev services.

The dev container exports one DEVENV_SERVICE_URL_<NAME> variable per entry in
devenv.toml's [services] table, giving the URL at which the host browser
reaches that service through the gateway (subtrees/devenv_utils/GATEWAY.md).
Tools that print a URL for the user read it through service_url().
"""

import os


def _env_var(service: str) -> str:
    return "DEVENV_SERVICE_URL_" + service.upper().replace("-", "_")


def service_url(service: str, port: int, default_port: int) -> str:
    """The browser URL for `service`: the gateway URL when `port` is the
    default, else http://localhost:<port>, since the gateway only routes the
    default port."""
    url = os.environ.get(_env_var(service))
    if url and port == default_port:
        return url
    return f"http://localhost:{port}"
