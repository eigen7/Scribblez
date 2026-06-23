#!/usr/bin/env python3
"""Play a trained model against HastyBot (or another model) and report win rate.

This is the standalone "stage 3" of the research loop -- it does not generate or
train, it just pits an already-exported .onnx against an opponent and prints the
W/L/D tally and decisive win percentage.

Usage:
    # latest .onnx of a tag vs HastyBot:
    python -m scripts.match -t mytag

    # a specific epoch, more games, win-probability selection:
    python -m scripts.match -t mytag --epoch 8 --games 5000 --objective winprob

    # an explicit model file vs another model file:
    python -m scripts.match --model a.onnx --vs-model b.onnx --games 2000
"""

import argparse
import sys
from pathlib import Path

from scribblez.match import play_match, win_pct
from scribblez.paths import TagPaths


def latest_onnx(paths: TagPaths) -> Path:
    onnx = sorted(paths.onnx_dir.glob("model_epoch_*.onnx"))
    if not onnx:
        raise FileNotFoundError(f"No .onnx exports in {paths.onnx_dir}")
    return onnx[-1]  # epoch is zero-padded, so lexicographic == numeric order


def resolve_model(model: str, tag: str, epoch: int | None) -> Path:
    """Resolve a model .onnx from an explicit --model path, or from a tag's
    models/ dir (a specific --epoch, else the latest export)."""
    if model:
        path = Path(model)
        if not path.exists():
            raise FileNotFoundError(f"model not found: {path}")
        return path
    if not tag:
        raise SystemExit("specify the model via -t/--tag or --model")
    paths = TagPaths(tag)
    return paths.onnx_path(epoch) if epoch is not None else latest_onnx(paths)


def neural_spec(model: Path, args) -> str:
    return (f"--type=neural --model={model} --top-k={args.top_k} "
            f"--temperature={args.temperature} --precision={args.precision} "
            f"--objective={args.objective}")


def opponent_spec(args) -> tuple[str, str]:
    """The opponent --player spec and a human-readable name. HastyBot unless a
    second model is given via --vs-tag / --vs-model."""
    if args.vs_model or args.vs_tag:
        model = resolve_model(args.vs_model, args.vs_tag, args.vs_epoch)
        return neural_spec(model, args), f"model {model.name}"
    return "--type=hastybot", "HastyBot"


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("-t", "--tag", default="", help="Tag whose model to play (latest epoch).")
    p.add_argument("--epoch", type=int, default=None, help="Specific epoch (default: latest).")
    p.add_argument("--model", default="", help="Explicit .onnx path (overrides -t/--epoch).")
    p.add_argument("--vs-tag", default="", help="Opponent tag's model (default opponent: HastyBot).")
    p.add_argument("--vs-epoch", type=int, default=None, help="Opponent epoch (default: latest).")
    p.add_argument("--vs-model", default="", help="Explicit opponent .onnx path.")
    p.add_argument("-g", "--games", type=int, default=2000, help="Games to play.")
    p.add_argument("-T", "--threads", type=int, default=8, help="Parallel game threads.")
    p.add_argument("--top-k", type=int, default=10,
                   help="Candidate set: 0 = all legal plays, K > 0 = top-K by HastyBot equity.")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="Move-sampling temperature (0 = greedy; use 0 for strength evaluation).")
    p.add_argument("--objective", choices=("winprob", "scorediff"), default="winprob",
                   help="Selection head: winprob (P(win)+0.5*P(draw)) or scorediff (margin).")
    p.add_argument("--precision", default="FP16", help="Neural agent TensorRT precision.")
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    model = resolve_model(args.model, args.tag, args.epoch)
    spec = neural_spec(model, args)
    opp_spec, opp_name = opponent_spec(args)

    print(f"Match: {model}  vs  {opp_name}  ({args.games} games, "
          f"objective={args.objective}, top-k={args.top_k}, T={args.temperature})")
    wld = play_match(spec, opp_spec, args.games, args.threads)
    if wld is None:
        return 1
    w, l, d = wld
    print(f"\nResult (model's perspective):  W={w}  L={l}  D={d}")
    print(f"Decisive win rate: {win_pct(wld):.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
