"""Render sampled positions to web-styled PNGs via the headless web renderer.

Builds the web UI's GameState JSON for each position (from the C++ FFI) and
hands it to web/tools/render_positions.tsx, which screenshots the real React
board components with headless Chromium. Best-effort: if the Node/Playwright
toolchain isn't available, this warns and returns False rather than failing the
caller -- the .slog and ASCII dumps are produced regardless.
"""

import json
import subprocess
import tempfile
from pathlib import Path

from scribblez.ffi import dump_position_json, read_file_header

# py/scribblez/position_eval/eval/web_render.py -> repo root is five parents up.
REPO_ROOT = Path(__file__).resolve().parents[4]
WEB_DIR = REPO_ROOT / "web"


def render_position_images(slog_path: str | Path, out_dir: str | Path, post_move: bool = True):
    """Render each position of `slog_path` to <out_dir>/pos-NN.png. Returns success."""
    slog_path = Path(slog_path)
    out_dir = Path(out_dir)
    num_pos, _ = read_file_header(slog_path)
    positions = [
        {
            "name": f"pos-{k:02d}",
            "state": json.loads(dump_position_json(slog_path, k, post_move=post_move)),
        }
        for k in range(num_pos)
    ]
    out_dir.mkdir(parents=True, exist_ok=True)

    tmp = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump(positions, fh)
            tmp = fh.name
        subprocess.run(
            ["npx", "tsx", "tools/render_positions.tsx", tmp, str(out_dir)],
            cwd=WEB_DIR,
            check=True,
        )
    finally:
        if tmp:
            Path(tmp).unlink(missing_ok=True)
