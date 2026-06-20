#!/usr/bin/env python3
"""Automated policy-iteration runner: the whole loop, one command, one directory.

Each generation used to be generated and trained by hand (one tag per iteration,
manually wiring --init-from from the previous tag). This drives the full loop:

  gen 0 : HastyBot self-play          -> train gen-0 (from scratch)
  gen i : self-play with the gen-(i-1) model (top-K + temperature)
                                      -> train gen-i, warm-started from gen-(i-1)

Everything for a run lives under one tag directory:

  <mount>/tags/<run>/
    data/train/gen-<i>.slog      one file per generation
    data/test/gen-<i>.slog
    models/gen-<i>.onnx           only the final model per generation (not epochs)
    checkpoints/gen-<i>.pt
    pi_log.csv                    per-generation win-rate vs HastyBot
    _staging/                     transient per-gen training dirs (auto-removed)

Reuses the tested building blocks rather than reimplementing them: the play_game
binary for self-play, scripts/train.py (with --init-from for the warm start) for
training, and play_game again for the HastyBot benchmark. The loop is resumable
-- a generation whose data/model already exist is skipped -- so re-running
continues where it stopped.

Usage:
    python -m scripts.pi_run -t mypi -n 5 -g 100000
    # or: ./py/scripts/pi_run.py -t mypi -n 5 -g 100000
"""

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from scribblez.ffi import read_file_header, sample_slog
from scribblez.paths import TagPaths

PLAY_GAME = "/workspace/repo/target/engine/play_game"
TRAIN_SCRIPT = Path(__file__).resolve().parent / "train.py"
PY_ROOT = Path(__file__).resolve().parent.parent  # the `py/` dir (for child PYTHONPATH)
# play_game buffers this many games before flushing a file; chunking bounds peak
# memory on large generations, after which the chunks are merged into one file.
GEN_CHUNK = 50000
# play_game's end-of-run summary line: "... W/L/D vs <Opponent>: W / L / D".
WLD_RE = re.compile(r"vs \w+:\s*(\d+)\s*/\s*(\d+)\s*/\s*(\d+)")


class RunPaths:
    """Resolves the single-directory layout for one PI run."""

    def __init__(self, tag: str):
        self.tag = tag
        self.root = TagPaths(tag).root

    def train_file(self, gen: int) -> Path:
        return self.root / "data" / "train" / f"gen-{gen}.slog"

    def test_file(self, gen: int) -> Path:
        return self.root / "data" / "test" / f"gen-{gen}.slog"

    def onnx(self, gen: int) -> Path:
        return self.root / "models" / f"gen-{gen}.onnx"

    def ckpt(self, gen: int) -> Path:
        return self.root / "checkpoints" / f"gen-{gen}.pt"

    @property
    def log(self) -> Path:
        return self.root / "pi_log.csv"


def neural_spec(model: Path, top_k: int, temperature: float, precision: str) -> str:
    """A `--player` value for the neural top-K agent (mirrors generate_data)."""
    return (f"--type=neural --model={model} --top-k={top_k} "
            f"--temperature={temperature} --precision={precision}")


def run(cmd, capture: bool = False) -> subprocess.CompletedProcess:
    """Run a subprocess, echoing the command; raise on failure. train.py needs
    `py/` on PYTHONPATH, so the child env always carries it."""
    print("  $", " ".join(str(c) for c in cmd), flush=True)
    pythonpath = os.pathsep.join([str(PY_ROOT), os.environ.get("PYTHONPATH", "")])
    env = {**os.environ, "PYTHONPATH": pythonpath}
    r = subprocess.run(cmd, env=env, capture_output=capture, text=True)
    if r.returncode != 0:
        if capture:
            sys.stderr.write((r.stdout or "") + (r.stderr or ""))
        raise SystemExit(f"command failed (exit {r.returncode}): {cmd[0]}")
    return r


def generate_split(player_spec: str, num_games: int, threads: int, handicap_max: int,
                   dst_file: Path) -> None:
    """Run `num_games` self-play games and consolidate them into one .slog at
    `dst_file`. play_game writes GEN_CHUNK-sized files (bounding its in-memory
    buffer); the chunks are then merged on disk via the FFI, so the result is a
    single file per generation even for very large (e.g. 10^6-game) runs."""
    dst_file.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=str(dst_file.parent)) as tmp:
        run([PLAY_GAME,
             "--player", player_spec, "--player", player_spec,
             "--binary-log-dir", tmp,
             "--games-per-file", str(min(num_games, GEN_CHUNK)),
             "--games", str(num_games),
             "--threads", str(threads),
             "--random-handicap-max", str(handicap_max)])
        files = sorted(Path(tmp).glob("*.slog"))
        if not files:
            raise RuntimeError(f"play_game produced no .slog in {tmp}")
        if len(files) == 1:
            shutil.move(str(files[0]), str(dst_file))
            return
        picks = [(str(f), i) for f in files for i in range(read_file_header(f)[0])]
        sample_slog(dst_file, picks)


def generate_gen(rp: RunPaths, gen: int, args) -> None:
    if rp.train_file(gen).exists() and (args.test_ratio == 0 or rp.test_file(gen).exists()):
        print(f"[gen {gen}] data already present -- skipping generation")
        return
    if gen == 0:
        spec = "--type=hastybot"
        source = "HastyBot"
    else:
        prev = rp.onnx(gen - 1)
        if not prev.exists():
            raise SystemExit(f"[gen {gen}] missing previous model {prev}")
        spec = neural_spec(prev, args.top_k, args.temperature, args.precision)
        source = f"gen-{gen - 1} (top-{args.top_k}, T={args.temperature})"
    n_games = args.gen0_games if gen == 0 else args.num_games
    test_games = round(n_games * args.test_ratio)
    train_games = n_games - test_games
    print(f"[gen {gen}] generating {train_games} train / {test_games} test games via {source}")
    generate_split(spec, train_games, args.threads, args.random_handicap_max, rp.train_file(gen))
    if test_games > 0:
        generate_split(spec, test_games, args.threads, args.random_handicap_max, rp.test_file(gen))


def link_into(dst_dir: Path, src_file: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    link = dst_dir / src_file.name
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(src_file.resolve())


def last_epoch_file(d: Path, pattern: str) -> Path:
    files = sorted(d.glob(pattern))
    if not files:
        raise RuntimeError(f"no {pattern} under {d} (did training produce a checkpoint?)")
    return files[-1]  # epoch is zero-padded, so lexicographic == numeric order


def train_gen(rp: RunPaths, gen: int, args) -> None:
    if rp.onnx(gen).exists() and rp.ckpt(gen).exists():
        print(f"[gen {gen}] model already present -- skipping training")
        return

    # A transient training tag whose data dirs symlink this generation's file(s)
    # (the current gen, plus earlier ones when --train-window > 1). train.py reads
    # a whole directory, so this isolates exactly which generations it trains on.
    staging = f"{args.tag}/_staging/gen{gen}"
    sp = TagPaths(staging)
    shutil.rmtree(sp.root, ignore_errors=True)
    for g in range(max(0, gen - args.train_window + 1), gen + 1):
        link_into(sp.train_dir, rp.train_file(g))
    if rp.test_file(gen).exists():
        link_into(sp.test_dir, rp.test_file(gen))

    lr = args.lr0 if gen == 0 else args.lr
    cmd = [sys.executable, str(TRAIN_SCRIPT), "-t", staging,
           "--epochs", str(args.epochs), "--lr", str(lr),
           "--no-dashboard", "--no-probe", "--no-calibration"]
    if gen > 0:
        cmd += ["--init-from", str(rp.ckpt(gen - 1))]
    warm = f"warm-start from gen-{gen - 1}" if gen > 0 else "from scratch"
    print(f"[gen {gen}] training: epochs={args.epochs}, lr={lr}, {warm}, "
          f"window={args.train_window}")
    run(cmd)

    # Keep only the final epoch's artifacts, under the canonical per-gen names.
    rp.ckpt(gen).parent.mkdir(parents=True, exist_ok=True)
    rp.onnx(gen).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(last_epoch_file(sp.checkpoints_dir, "model_epoch_*.pt"), rp.ckpt(gen))
    shutil.copy2(last_epoch_file(sp.onnx_dir, "model_epoch_*.onnx"), rp.onnx(gen))
    shutil.rmtree(sp.root, ignore_errors=True)  # drop intermediate epochs / dashboard db
    print(f"[gen {gen}] saved {rp.onnx(gen).name} and {rp.ckpt(gen).name}")


LOG_FIELDS = ["gen", "hasty_w", "hasty_l", "hasty_d", "winrate_vs_hasty",
              "prev_w", "prev_l", "prev_d", "winrate_vs_prev"]


def append_log(rp: RunPaths, row: dict) -> None:
    new = not rp.log.exists()
    with open(rp.log, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS, restval="")
        if new:
            w.writeheader()
        w.writerow(row)


def play_match(spec_a: str, spec_b: str, games: int, threads: int):
    """Play `games` between two --player specs; return (wins, losses, draws) from
    the FIRST player's perspective, or None if the summary line can't be parsed."""
    r = run([PLAY_GAME, "--player", spec_a, "--player", spec_b,
             "--games", str(games), "--threads", str(threads)], capture=True)
    out = (r.stdout or "") + (r.stderr or "")
    m = WLD_RE.search(out)
    if not m:
        sys.stderr.write(out)
        print("WARNING: could not parse W/L/D from play_game output", file=sys.stderr)
        return None
    return tuple(int(x) for x in m.groups())


def win_pct(wld) -> float:
    w, l, _ = wld
    decisive = w + l
    return 100.0 * w / decisive if decisive else float("nan")


def benchmark_gen(rp: RunPaths, gen: int, args) -> None:
    if args.benchmark_games <= 0:
        return
    # Greedy (temperature 0) for evaluation -- the strength metric, not training.
    cur = neural_spec(rp.onnx(gen), args.top_k, 0.0, args.precision)
    row = {"gen": gen}

    print(f"[gen {gen}] benchmark vs HastyBot ({args.benchmark_games} games)")
    h = play_match(cur, "--type=hastybot", args.benchmark_games, args.threads)
    if h:
        row.update({"hasty_w": h[0], "hasty_l": h[1], "hasty_d": h[2],
                    "winrate_vs_hasty": f"{win_pct(h) / 100:.4f}"})
        print(f"[gen {gen}] vs HastyBot: W={h[0]} L={h[1]} D={h[2]}  win%={win_pct(h):.1f}")

    # New model vs the previous generation -- direct measure of per-step gain.
    if gen > 0:
        prev = neural_spec(rp.onnx(gen - 1), args.top_k, 0.0, args.precision)
        print(f"[gen {gen}] benchmark vs gen-{gen - 1} ({args.benchmark_games} games)")
        p = play_match(cur, prev, args.benchmark_games, args.threads)
        if p:
            row.update({"prev_w": p[0], "prev_l": p[1], "prev_d": p[2],
                        "winrate_vs_prev": f"{win_pct(p) / 100:.4f}"})
            print(f"[gen {gen}] vs gen-{gen - 1}: W={p[0]} L={p[1]} D={p[2]}  "
                  f"win%={win_pct(p):.1f}")

    append_log(rp, row)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("-t", "--tag", required=True,
                   help="Output tag: the whole run lives at <mount>/tags/<tag>/.")
    p.add_argument("-n", "--generations", type=int, default=5, help="Generations to run (0..n-1).")
    p.add_argument("-g", "--num-games", type=int, default=100000,
                   help="Games per neural generation (gen >= 1).")
    p.add_argument("--gen0-games", type=int, default=1000000,
                   help="Games for gen 0 (HastyBot self-play).")
    p.add_argument("--test-ratio", type=float, default=0.1, help="Held-out fraction per gen.")
    p.add_argument("-T", "--threads", type=int, default=8, help="Parallel game threads.")
    p.add_argument("--epochs", type=int, default=10, help="Training epochs per gen.")
    p.add_argument("--lr", type=float, default=2e-4, help="LR for warm-started gens (gen >= 1).")
    p.add_argument("--lr0", type=float, default=1e-3, help="LR for gen 0 (trained from scratch).")
    p.add_argument("--top-k", type=int, default=10, help="Neural agent candidate count.")
    p.add_argument("--temperature", type=float, default=3.0, help="Self-play sampling temperature.")
    p.add_argument("--precision", default="FP16", help="Neural agent TensorRT precision.")
    p.add_argument("--random-handicap-max", type=int, default=100, help="Per-game handicap max.")
    p.add_argument(
        "--train-window", type=int, default=1,
        help="Generations of data to train each model on (1 = current gen only; "
             ">1 = sliding window of the most recent gens).",
    )
    p.add_argument(
        "--benchmark-games", type=int, default=2000,
        help="Games vs HastyBot after each gen, logged to pi_log.csv (0 disables).",
    )
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    if not 0.0 <= args.test_ratio < 1.0:
        print("--test-ratio must be in [0, 1).", file=sys.stderr)
        return 2
    if args.train_window < 1:
        print("--train-window must be >= 1.", file=sys.stderr)
        return 2

    rp = RunPaths(args.tag)
    print(f"PI run '{args.tag}' -> {rp.root}  ({args.generations} generations)")
    for gen in range(args.generations):
        print(f"\n===== generation {gen} =====")
        generate_gen(rp, gen, args)
        train_gen(rp, gen, args)
        benchmark_gen(rp, gen, args)
    print(f"\nPI run complete. Win-rate trajectory: {rp.log}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
