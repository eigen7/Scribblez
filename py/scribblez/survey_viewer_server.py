"""Serving the sim-survey viewer (web ?tool=survey): its data at /api/survey
on a loopback API server, and the Vite dev server that the browser loads and
that proxies /api to it. Shared by the scripts that feed the viewer
(py/scripts/sim_survey_viewer.py, py/scripts/blind_spots_viewer.py), which use
the same gateway service and ports, so one replaces the other."""

import json
import os
import subprocess
import sys
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from urllib.parse import parse_qs, urlparse

from scribblez.dashboard.react_server import WEB_DIR, reclaim_port
from scribblez.service_urls import service_url

DEFAULT_API_PORT = 8091
DEFAULT_DEV_PORT = 5181


# A dynamic API route: the request's query parameters to a JSON-ready answer.
# Raising KeyError or ValueError on a bad query answers 400.
Route = Callable[[dict[str, list[str]]], dict]


def survey_handler(payload: bytes, routes: dict[str, Route]) -> type[BaseHTTPRequestHandler]:
    """A request handler answering GET /api/survey with `payload` and each of
    `routes`' paths with its answer."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/api/survey":
                self._send(payload)
            elif url.path in routes:
                try:
                    answer = routes[url.path](parse_qs(url.query))
                except (KeyError, ValueError, IndexError) as e:
                    self.send_error(400, str(e))
                    return
                self._send(json.dumps(answer).encode())
            else:
                self.send_error(404)

        def _send(self, body: bytes):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            pass  # one line per request is noise under the viewer's banner

    return Handler


def serve(
    data: dict, api_port: int, dev_port: int, banner: str, routes: dict[str, Route] | None = None
) -> int:
    """Serve `data` to the viewer, and `routes` beside it, until the dev server
    exits or Ctrl-C, printing `banner` beside the viewer's URL."""
    reclaim_port(api_port)
    reclaim_port(dev_port)
    # Loopback only: the browser reaches it through Vite's /api proxy.
    server = ThreadingHTTPServer(
        ("127.0.0.1", api_port), survey_handler(json.dumps(data).encode(), routes or {})
    )
    Thread(target=server.serve_forever, daemon=True).start()
    env = {
        **os.environ,
        "VITE_TOOL": "survey",
        "VITE_DEV_PORT": str(dev_port),
        "VITE_API_PORT": str(api_port),
    }
    vite = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=WEB_DIR,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    url = service_url("survey", dev_port, DEFAULT_DEV_PORT)
    print(f"\n{banner} Viewer: {url}\n", file=sys.stderr)
    try:
        vite.wait()
    except KeyboardInterrupt:
        pass
    finally:
        vite.terminate()
    return 0
