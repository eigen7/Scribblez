#!/usr/bin/env python3
"""Automated policy-iteration runner: the whole loop, one command, one directory.

Each generation used to be generated and trained by hand (one tag per iteration,
manually wiring --init-from from the previous tag). This drives the full loop:

  gen 0 : HastyBot self-play (optionally temperature-sampled for exploration)
                                      -> train gen-0 (from scratch)
  gen i : self-play with the gen-(i-1) model (the value agent: all legal moves by
          default, or top-K by equity via --top-k) -> train gen-i, warm-started
          from gen-(i-1)

Each generation trains on a recency-weighted blend of all prior generations'
data (the current gen in full, older gens geometrically down-sampled), so the
model never forgets old positions while emphasizing recent self-play.

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
import random
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


def hasty_spec(args) -> str:
    """The gen-0 `--player` value: greedy HastyBot, or temperature-sampled
    HastyBot when --hasty-temperature > 0 (adds exploration to gen-0 data)."""
    if args.hasty_temperature > 0:
        return (f"--type=hastybot --temperature={args.hasty_temperature} "
                f"--top-k={args.hasty_top_k}")
    return "--type=hastybot"


def neural_spec(model: Path, args, temperature: float) -> str:
    """A `--player` value for the neural value agent at the given sampling
    temperature. --top-k selects the candidate set: 0 = every legal play (most
    diverse, slowest), K > 0 = the top-K by HastyBot equity (faster)."""
    return (f"--type=neural --model={model} --top-k={args.top_k} "
            f"--temperature={temperature} --precision={args.precision}")


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


def run_streaming(cmd) -> tuple[int, str]:
    """Run a subprocess, echoing its combined output to our stderr live while
    also capturing it. Lets play_game's periodic progress lines show in real
    time and still leaves the full output for the caller to parse (e.g. the
    benchmark's W/L/D summary). Returns (returncode, combined_output)."""
    print("  $", " ".join(str(c) for c in cmd), flush=True)
    pythonpath = os.pathsep.join([str(PY_ROOT), os.environ.get("PYTHONPATH", "")])
    env = {**os.environ, "PYTHONPATH": pythonpath}
    proc = subprocess.Popen(cmd, env=env, text=True, bufsize=1,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    chunks = []
    for line in proc.stdout:
        sys.stderr.write(line)
        sys.stderr.flush()
        chunks.append(line)
    proc.wait()
    return proc.returncode, "".join(chunks)


def generate_split(player_spec: str, num_games: int, threads: int, handicap_max: int,
                   dst_file: Path, sample_endgames: bool = False) -> None:
    """Run `num_games` self-play games and consolidate them into one .slog at
    `dst_file`. play_game writes GEN_CHUNK-sized files (bounding its in-memory
    buffer); the chunks are then merged on disk via the FFI, so the result is a
    single file per generation even for very large (e.g. 10^6-game) runs.

    When `sample_endgames` is set, play_game also draws training positions from
    endgame turns (bag empty); otherwise only pre-endgame turns are eligible."""
    dst_file.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=str(dst_file.parent)) as tmp:
        cmd = [PLAY_GAME,
               "--player", player_spec, "--player", player_spec,
               "--binary-log-dir", tmp,
               "--games-per-file", str(min(num_games, GEN_CHUNK)),
               "--games", str(num_games),
               "--threads", str(threads),
               "--random-handicap-max", str(handicap_max)]
        if sample_endgames:
            cmd.append("--sample-endgames")
        run(cmd)
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
        spec = hasty_spec(args)
        source = "HastyBot" + (f" (T={args.hasty_temperature}, top-{args.hasty_top_k})"
                               if args.hasty_temperature > 0 else "")
    else:
        prev = rp.onnx(gen - 1)
        if not prev.exists():
            raise SystemExit(f"[gen {gen}] missing previous model {prev}")
        spec = neural_spec(prev, args, args.temperature)
        agent = "all-moves" if args.top_k == 0 else f"top-{args.top_k}"
        source = f"gen-{gen - 1} ({agent}, T={args.temperature})"
    n_games = args.gen0_games if gen == 0 else args.num_games
    test_games = round(n_games * args.test_ratio)
    train_games = n_games - test_games
    print(f"[gen {gen}] generating {train_games} train / {test_games} test games via {source}")
    generate_split(spec, train_games, args.threads, args.random_handicap_max,
                   rp.train_file(gen), args.sample_endgames)
    if test_games > 0:
        generate_split(spec, test_games, args.threads, args.random_handicap_max,
                       rp.test_file(gen), args.sample_endgames)


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


def gens_in_window(gen: int, window: int) -> range:
    """The generations whose data feeds training at `gen`. window <= 0 means all
    of 0..gen; window > 0 caps the lookback to the most-recent `window` gens."""
    lo = 0 if window <= 0 else max(0, gen - window + 1)
    return range(lo, gen + 1)


def build_recency_weighted_train(rp: RunPaths, gen: int, args, dst_dir: Path) -> int:
    """Write one combined training .slog into `dst_dir`, blending every in-window
    generation with an exponential recency weight: generation g contributes a
    fraction decay^(gen-g) of its games (the current gen in full, older gens
    geometrically down-sampled). Every generation stays represented while recent
    self-play dominates the sample. Returns the total games written.

    Sampling is at the game level (one training position per game in a .slog), so
    this reuses the same sample_slog path that consolidates chunked generations.
    Deterministic per generation for resumable, reproducible runs."""
    rng = random.Random(0x5C12B1E2 ^ gen)
    picks: list[tuple[str, int]] = []
    summary: list[str] = []
    for g in gens_in_window(gen, args.train_window):
        src = rp.train_file(g)
        num_games, _ = read_file_header(src)
        weight = args.recency_decay ** (gen - g)
        keep = num_games if g == gen else max(0, min(round(weight * num_games), num_games))
        if keep <= 0:
            continue
        chosen = range(num_games) if keep >= num_games else rng.sample(range(num_games), keep)
        picks += [(str(src), i) for i in chosen]
        summary.append(f"gen-{g}:{keep}/{num_games}(w={weight:.3f})")
    if not picks:
        raise RuntimeError(f"[gen {gen}] recency window produced no training games")
    dst_dir.mkdir(parents=True, exist_ok=True)
    sample_slog(dst_dir / f"gen-{gen}-train.slog", picks)
    print(f"[gen {gen}] recency-weighted train set: {len(picks)} games  [{', '.join(summary)}]")
    return len(picks)


def train_gen(rp: RunPaths, gen: int, args) -> None:
    if rp.onnx(gen).exists() and rp.ckpt(gen).exists():
        print(f"[gen {gen}] model already present -- skipping training")
        return

    # A transient training tag. Its train dir holds one combined .slog blending
    # the in-window generations with recency weights; its test dir symlinks only
    # this generation's held-out file. train.py reads whole directories, so this
    # isolates exactly which data it trains and validates on.
    staging = f"{args.tag}/_staging/gen{gen}"
    sp = TagPaths(staging)
    shutil.rmtree(sp.root, ignore_errors=True)
    build_recency_weighted_train(rp, gen, args, sp.train_dir)
    if rp.test_file(gen).exists():
        link_into(sp.test_dir, rp.test_file(gen))

    lr = args.lr0 if gen == 0 else args.lr
    cmd = [sys.executable, str(TRAIN_SCRIPT), "-t", staging,
           "--epochs", str(args.epochs), "--lr", str(lr),
           "--no-dashboard", "--no-probe", "--no-calibration"]
    if gen > 0:
        cmd += ["--init-from", str(rp.ckpt(gen - 1))]
    warm = f"warm-start from gen-{gen - 1}" if gen > 0 else "from scratch"
    window = "all" if args.train_window <= 0 else args.train_window
    print(f"[gen {gen}] training: epochs={args.epochs}, lr={lr}, {warm}, "
          f"window={window}, recency-decay={args.recency_decay}")
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
    rc, out = run_streaming([PLAY_GAME, "--player", spec_a, "--player", spec_b,
                             "--games", str(games), "--threads", str(threads)])
    if rc != 0:
        raise SystemExit(f"command failed (exit {rc}): {PLAY_GAME}")
    m = WLD_RE.search(out)
    if not m:
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
    # Uses the same agent family as self-play so the benchmark reflects how the
    # trained policy actually plays.
    cur = neural_spec(rp.onnx(gen), args, 0.0)
    row = {"gen": gen}

    print(f"[gen {gen}] benchmark vs HastyBot ({args.benchmark_games} games)")
    h = play_match(cur, "--type=hastybot", args.benchmark_games, args.threads)
    if h:
        row.update({"hasty_w": h[0], "hasty_l": h[1], "hasty_d": h[2],
                    "winrate_vs_hasty": f"{win_pct(h) / 100:.4f}"})
        print(f"[gen {gen}] vs HastyBot: W={h[0]} L={h[1]} D={h[2]}  win%={win_pct(h):.1f}")

    # New model vs the previous generation -- direct measure of per-step gain.
    if gen > 0:
        prev = neural_spec(rp.onnx(gen - 1), args, 0.0)
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
    p.add_argument(
        "--top-k", type=int, default=0,
        help="Neural self-play candidate set for gen >= 1: 0 = every legal play "
             "(most diverse, slowest); K > 0 = top-K by HastyBot equity (faster).",
    )
    p.add_argument("--temperature", type=float, default=3.0, help="Self-play sampling temperature.")
    p.add_argument("--hasty-temperature", type=float, default=0.0,
                   help="Gen-0 HastyBot softmax temperature (0 = greedy; >0 explores).")
    p.add_argument("--hasty-top-k", type=int, default=10,
                   help="Gen-0 HastyBot candidate count when --hasty-temperature > 0.")
    p.add_argument("--sample-endgames", action="store_true",
                   help="Also draw training positions from endgame turns (bag empty); "
                        "by default only pre-endgame positions are sampled.")
    p.add_argument("--precision", default="FP16", help="Neural agent TensorRT precision.")
    p.add_argument("--random-handicap-max", type=int, default=100, help="Per-game handicap max.")
    p.add_argument(
        "--train-window", type=int, default=0,
        help="Max generations of past data to blend into each training set "
             "(0 = all generations; N = only the most-recent N).",
    )
    p.add_argument(
        "--recency-decay", type=float, default=0.5,
        help="Per-generation recency weight: generation g contributes a fraction "
             "decay^(gen-g) of its games (1.0 = uniform; smaller = favor recent).",
    )
    p.add_argument(
        "--benchmark-games", type=int, default=1000,
        help="Games vs HastyBot after each gen, logged to pi_log.csv (0 disables).",
    )
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    if not 0.0 <= args.test_ratio < 1.0:
        print("--test-ratio must be in [0, 1).", file=sys.stderr)
        return 2
    if args.train_window < 0:
        print("--train-window must be >= 0 (0 = all generations).", file=sys.stderr)
        return 2
    if not 0.0 < args.recency_decay <= 1.0:
        print("--recency-decay must be in (0, 1].", file=sys.stderr)
        return 2
    if args.top_k < 0:
        print("--top-k must be >= 0 (0 = all legal plays).", file=sys.stderr)
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
