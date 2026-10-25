"""Serving the sim-survey viewer (web ?tool=survey): its data at /api/survey
on a loopback API server, and the Vite dev server that the browser loads and
that proxies /api to it. Shared by the scripts that feed the viewer
(py/scripts/sim_survey_viewer.py, py/scripts/blind_spots_viewer.py), which use
the same gateway service and ports, so one replaces the other."""

import json
import os
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

from scribblez.dashboard.react_server import WEB_DIR, reclaim_port
from scribblez.service_urls import service_url

DEFAULT_API_PORT = 8091
DEFAULT_DEV_PORT = 5181


def survey_handler(payload: bytes) -> type[BaseHTTPRequestHandler]:
    """A request handler answering GET /api/survey with `payload`."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != "/api/survey":
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format, *args):
            pass  # one line per request is noise under the viewer's banner

    return Handler


def serve(data: dict, api_port: int, dev_port: int, banner: str) -> int:
    """Serve `data` to the viewer until the dev server exits or Ctrl-C,
    printing `banner` beside the viewer's URL."""
    reclaim_port(api_port)
    reclaim_port(dev_port)
    # Loopback only: the browser reaches it through Vite's /api proxy.
    server = ThreadingHTTPServer(("127.0.0.1", api_port), survey_handler(json.dumps(data).encode()))
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
