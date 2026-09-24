"""The position-evaluation train role: an open-ended consume-and-train loop.

Runs as the position_eval workload's singleton `train` worker (SCZ_ROLE=train),
or via scripts/position_eval/train.py for headless debugging.

The trainer never generates games. The generation scheduler
(generational/scheduler.py) assembles generation directories from the
generator workers' output, and the trainer loops:

  1. wait until its cursor generation is complete on disk;
  2. train one epoch over a sliding window of the most recent complete
     generations, sampling turns_per_game turns per game;
  3. export, evaluate and checkpoint under that generation's index, evict
     generations that left the window, and publish the cursor
     (train_state.json), which the scheduler uses to pace the generators.

One epoch per generation bounds data reuse: each game is trained on `window`
times, once per generation it spends in the window. Epoch and generation are
the same clock; progress is otherwise measured in cumulative rows trained, the
dashboard's x-axis. A single rolling model.pt holds the resume state, so a
restarted worker continues where it stopped, and a stop loses at most the
current generation's epoch.

Where it runs. All I/O beyond local scratch goes through the worker's sink, so
the same trainer runs on the controller's machine or on a rented one: the sink
fetches generations before training on them and delivers each generation's
outputs afterwards (export, checkpoint, cursor, and last the record). The
trainer never writes dashboard.db; the dashboard ingests its records
(generational/records.py). On a fresh machine, the checkpoint, cursor and
window are restored through the sink, so the first epoch is the one a local
resume would have run. The optimizer arm (generational/optim.py) owns the
learning-rate policy; CPU thread counts are live dashboard controls, adopted
at the next generation.

Training-step performance. The forward runs under bf16 autocast, with the loss
computed on fp32-upcast outputs and fp32 weights and optimizer state. bf16 has
fp32's exponent range, so it cannot hit the overflow that rules out fp16
(docs/plans/fp16_safe_serving.md). Remaining fp32 matmuls use TF32. The
training forward is torch.compile'd; every other pass (BatchNorm
recalibration, evals, ONNX export, checkpointing) uses the eager module, so
state-dict keys and the exported graph are unaffected. Epochs drop the
trailing partial batch so the compiled graph sees one static shape and
compiles once. On the transformer trunk at batch 256 this is ~4x end to end
(0.2k -> 0.8k rows/s); the conv trunk gains ~2.6x, almost all from bf16.
"""

import functools
import itertools
import os
import queue
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path

import torch

from scribblez.dataset import SlogDataset
from scribblez.ffi import (
    get_input_shapes,
    session_input_arm,
    set_opp_leave_input,
)
from scribblez.generational import checkpoint, lifecycle
from scribblez.generational.checkpoint import GenerationalState
from scribblez.generational.controls import CpuController, default_controls, progress_line
from scribblez.generational.optim import build_optim_arm, build_optimizer
from scribblez.generational.records import TrainRecorder, read_controls
from scribblez.paths import TagPaths
from scribblez.position_eval import analysis as position_eval_analysis
from scribblez.position_eval.model import PositionEvalModel
from scribblez.position_eval.onnx_export import export_onnx
from scribblez.position_eval.train_loop import LossConfig, run_epoch
from scribblez.spatial_trunk import transformer_config
from scribblez.train_common import timed_print
from scribblez.workloads.base import WorkerContext
from scribblez.workloads.worker import WorkerStats, WorkerStopped

# Poll interval while waiting for the cursor generation to complete.
POLL_SECONDS = 5


def _publish_train_state(paths: TagPaths, state: GenerationalState):
    lifecycle.write_train_state(paths, asdict(state))


def _generation_ready(paths: TagPaths, sink, index: int) -> bool:
    """Whether generation `index` is complete locally, first trying to fetch
    it through `sink` if not. A published generation arrives manifest last, so
    a fetched one is complete."""
    gen_dir = paths.generation_dir(index)
    if lifecycle.is_complete(gen_dir):
        return True
    return sink.fetch_data_dir(f"generations/{gen_dir.name}", gen_dir) and lifecycle.is_complete(
        gen_dir
    )


def wait_for_generation(paths: TagPaths, index: int, sink):
    """Block until generation `index` is complete on disk. Time spent here
    shows in the Stats tab as generation being the bottleneck."""
    announced = False
    while not _generation_ready(paths, sink, index):
        if not announced:
            timed_print(f"waiting for generation {index} to complete ...")
            announced = True
        time.sleep(POLL_SECONDS)
    if announced:
        timed_print(f"generation {index} is complete")


def restore_from_sink(paths: TagPaths, sink):
    """On a machine with no local checkpoint, fetch the rolling checkpoint and
    cursor through the sink, if it has them."""
    if paths.rolling_checkpoint.exists():
        return
    if sink.fetch_file("checkpoints/model.pt", paths.rolling_checkpoint):
        sink.fetch_file("train_state.json", paths.train_state_path)
        timed_print(f"restored the rolling checkpoint through the {sink.kind} sink")


def ensure_window(paths: TagPaths, sink, cursor: int, window: int):
    """Fetch any missing generations of the window ending before `cursor`, so
    a restored trainer's first epoch matches a local resume. A generation that
    cannot be fetched just shortens the window."""
    for index in range(max(0, cursor - window), cursor):
        _generation_ready(paths, sink, index)


def _rows_left(params, state: GenerationalState) -> bool:
    return params.max_rows == 0 or state.rows_trained < params.max_rows


# How many generations' deliveries may queue behind the one in flight before
# submitting blocks training. Uploads falling this far behind signal a problem
# worth stalling for rather than letting the backlog grow.
MAX_PENDING_DELIVERIES = 2


class OutputDeliverer:
    """Delivers generations' outputs on a background thread.

    A remote trainer uploads ~150 MB per generation; this lets that overlap
    the next generation's training. A single thread keeps deliveries in
    submission order, which the record-last commit protocol relies on. A
    failed delivery stops the thread and is re-raised from the next `submit`,
    `collect` or `drain`, so it fails the runner instead of vanishing.
    """

    def __init__(self, max_pending: int = MAX_PENDING_DELIVERIES):
        self._pending: queue.Queue = queue.Queue(maxsize=max_pending)
        self._done: queue.Queue = queue.Queue()
        self._error: Exception | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, what: str, fn):
        """Queue `fn()` for the delivery thread; blocks while the queue is full."""
        while True:
            self._raise_failure()
            try:
                self._pending.put((what, fn), timeout=1.0)
                return
            except queue.Full:
                continue

    def collect(self) -> list[tuple[str, float]]:
        """(what, seconds) for every step finished since the last call."""
        out = []
        while True:
            try:
                out.append(self._done.get_nowait())
            except queue.Empty:
                break
        self._raise_failure()
        return out

    def drain(self):
        """Block until everything submitted has been delivered. Must be called
        before the runner exits, including on a stop."""
        self._pending.put(None)
        self._thread.join()
        self._raise_failure()

    def _raise_failure(self):
        if self._error is not None:
            raise self._error

    def _run(self):
        while True:
            item = self._pending.get()
            if item is None:
                return
            what, fn = item
            t0 = time.monotonic()
            try:
                fn()
            except Exception as e:  # noqa: BLE001 -- re-raised by submit()/drain(), not lost
                self._error = RuntimeError(f"delivering {what} failed: {e}")
                return
            self._done.put((what, time.monotonic() - t0))


def _snapshot(path: Path, gen: int) -> Path:
    """A hard link to `path`'s current version, for a delivery that may run
    after the trainer has rewritten `path`. Rewrites replace the file rather
    than modifying it, so the link keeps this version."""
    snap = path.with_name(f"{path.name}.gen{gen}")
    snap.unlink(missing_ok=True)
    os.link(path, snap)
    return snap


def _deliver_generation(
    sink, paths: TagPaths, gen: int, checkpoint_snap, state_snap, recorder, staged
):
    """One generation's deliveries: the export and its sidecars, the
    checkpoint, the cursor, and last the record, whose arrival means the rest
    is in place."""
    for sidecar in paths.onnx_sidecars:
        sink.deliver_output(sidecar, f"models/{sidecar.name}", keep=True)
    sink.deliver_output(paths.onnx_path(gen), f"models/{paths.onnx_path(gen).name}")
    sink.deliver_output(checkpoint_snap, "checkpoints/model.pt")
    sink.deliver_output(state_snap, "train_state.json")
    recorder.deliver_staged(staged)


def _checkpoint_and_eval(
    model, optimizer, recorder, paths, device, params, state, gen, result, elapsed, optim_arm, ctx
):
    """Evaluate, export ONNX, save the rolling checkpoint, publish the cursor,
    and queue the generation's deliveries. Returns the seconds spent.

    The record (metrics and eval, with rows trained as `positions`) is
    delivered last because it makes the generation visible on the dashboard,
    which may then ask for the export or rely on the checkpoint. The
    checkpoint and cursor are delivered as snapshots because the next
    generation rewrites them."""
    sys.stdout.write("\n")
    avg = result.losses
    lr_now = optim_arm.current
    ci = gen
    timed_print(
        f"[gen {gen}] rows={state.rows_trained} loss={avg['total']:.4f} "
        f"wld_acc={result.wld_acc:.4f} lr={lr_now:.2e} "
        f"grad_norm={result.grad_norm['grad_norm_mean']:.3f} "
        f"clipped={result.grad_norm['clip_frac']:.0%} {elapsed:.1f}s"
    )
    record = {
        "epoch": ci,
        "positions": state.rows_trained,
        "loss": avg["total"],
        "loss_wld": avg["wld"],
        "loss_score_diff": avg["score_diff"],
        "loss_score_diff_mean": avg["score_diff_mean"],
        "loss_score_diff_std": avg["score_diff_std"],
        "loss_opp_next_placement": avg["opp_next_placement"],
        "loss_self_next_placement": avg["self_next_placement"],
        "loss_opp_win_placement": avg["opp_win_placement"],
        "loss_self_win_placement": avg["self_win_placement"],
        "wld_acc": result.wld_acc,
        "lr": lr_now,
        **result.grad_norm,
        "elapsed_s": elapsed,
        # Arm-specific metrics, e.g. schedule-free's averaging weight, which
        # anneals in place of the learning rate.
        **optim_arm.metrics(),
    }
    # Model-vs-Monte-Carlo quality on the large eval set, recorded alongside
    # the training losses.
    t_eval = time.time()
    record.update(eval_position_eval_quality(model, ctx["position_eval_quality"], device))
    timed_print(
        f"  quality: win_mae={record['eval_win_mae']:.4f} "
        f"sd_mean_mae={record['eval_sd_mean_mae']:.1f}"
        + (
            f" place_l1 opp_next={record['eval_place_l1_opp_next']:.3f}"
            f" self_next={record['eval_place_l1_self_next']:.3f}"
            if "eval_place_l1_opp_next" in record
            else ""
        )
    )
    export_onnx(
        model,
        paths.onnx_path(ci),
        ctx["spatial_planes"],
        ctx["scalar_size"],
        opp_leave_input=params.face_up_leaves,
    )
    checkpoint.save(paths, model, optimizer, state, ctx["config"])
    _publish_train_state(paths, state)
    staged = recorder.stage_generation(ci, state.rows_trained, record)
    ctx["deliverer"].submit(
        f"generation {ci}",
        functools.partial(
            _deliver_generation,
            ctx["sink"],
            paths,
            ci,
            _snapshot(paths.rolling_checkpoint, ci),
            _snapshot(paths.train_state_path, ci),
            recorder,
            staged,
        ),
    )
    return time.time() - t_eval


# Batches the schedule-free arm recomputes BatchNorm statistics over before
# checkpointing (optim.ScheduleFreeArm.eval_mode), forward-only. Ten batches
# already match a full pass's statistics; 32 leaves margin. The WSD arm ignores
# them.
BN_RECALIBRATION_BATCHES = 32


def _recalibration_batches(ds, params, gen, device):
    """(spatial, scalar) input pairs on `device` for BatchNorm recalibration,
    drawn from `ds` under a seed distinct from the epoch's."""
    batches = ds.iter_batches(
        params.batch_size,
        seed=gen * 1000003 + 1,
        turns_per_game=params.turns_per_game,
        epoch_index=gen,
    )
    for batch in itertools.islice(batches, BN_RECALIBRATION_BATCHES):
        yield batch["input_spatial"].to(device), batch["input_scalar"].to(device)


def train_one_generation(
    model,
    train_model,
    optimizer,
    recorder,
    paths,
    device,
    params,
    state,
    loss_cfg,
    optim_arm,
    cpu,
    ctx,
):
    """Train one epoch over the window ending at the cursor generation, then
    checkpoint under that generation's index and advance the cursor.

    `train_model` is the compiled forward sharing `model`'s parameters. The
    optimizer arm swaps in the deployable weights (schedule-free's average)
    for the checkpoint step, so eval, export and saved state all see them."""
    gen = state.generation_index
    window = lifecycle.window_dirs(paths, gen, params.window)
    ds = SlogDataset(
        window, post_move=True, apply_symmetry=True, num_workers=cpu.dataloader_workers
    )
    timed_print(
        f"generation {gen}: window {[d.name for d in window]} "
        f"({ds.num_games} games, {ds.num_samples} eligible rows)"
    )
    # Seeding by generation gives each of a game's `window` passes a different
    # shuffle and different turns. drop_last keeps one static shape for the
    # compiled forward.
    batches = ds.iter_batches(
        params.batch_size,
        seed=gen * 1000003,
        turns_per_game=params.turns_per_game,
        epoch_index=gen,
        drop_last=True,
    )
    t0 = time.time()
    rows_before = state.rows_trained
    optim_arm.train_mode()
    result = run_epoch(
        train_model,
        optimizer,
        batches,
        device,
        loss_cfg,
        lr_fn=optim_arm.lr_fn,
        rows_trained=state.rows_trained,
        on_batch=functools.partial(progress_line, gen),
        grad_clip=params.grad_clip,
    )
    state.rows_trained = result.rows_trained
    state.generation_index = gen + 1
    elapsed = time.time() - t0
    optim_arm.eval_mode(model, _recalibration_batches(ds, params, gen, device))
    eval_seconds = _checkpoint_and_eval(
        model,
        optimizer,
        recorder,
        paths,
        device,
        params,
        state,
        gen,
        result,
        elapsed,
        optim_arm,
        ctx,
    )
    optim_arm.train_mode()
    if ctx["stats"] is not None:
        # The last finished delivery, usually the previous generation's.
        finished = ctx["deliverer"].collect()
        ctx["stats"].cycle_done(
            {
                "train_s": elapsed,
                "eval_s": eval_seconds,
                "upload_s": finished[-1][1] if finished else 0.0,
            },
            units=state.rows_trained - rows_before,
            nbytes=0,
        )


def run_generational_training(
    model, train_model, optimizer, recorder, paths, device, params, state, ctx
):
    """The wait->train->advance loop, from the resumed cursor onward."""
    loss_cfg = LossConfig.from_args(params)
    optim_arm = build_optim_arm(recorder, params, optimizer, state.rows_trained)
    cpu = CpuController(recorder, ctx["read_controls"])
    while _rows_left(params, state):
        cpu.refresh(state.rows_trained)
        wait_for_generation(paths, state.generation_index, ctx["sink"])
        train_one_generation(
            model,
            train_model,
            optimizer,
            recorder,
            paths,
            device,
            params,
            state,
            loss_cfg,
            optim_arm,
            cpu,
            ctx,
        )
        evicted = lifecycle.evict_beyond_window(paths, state.generation_index - 1, params.window)
        if evicted:
            timed_print(f"evicted generations {evicted} (window={params.window})")
    timed_print(f"Stopped at {state.rows_trained} rows (generation {state.generation_index}).")


# ---------------------------------------------------------------------------
# Per-checkpoint evaluation
# ---------------------------------------------------------------------------


def load_position_eval_quality(spatial_planes: int, face_up_leaves: bool) -> dict:
    """Load the large eval set's inputs and its Monte-Carlo ground truth for
    the run's information condition, evaluated at every checkpoint.

    A missing dataset or ground truth is fatal: otherwise a rented machine
    could train for hours before anyone noticed the eval curves were missing."""
    dataset = str(position_eval_analysis.LARGE_DATASET)
    names, inputs = position_eval_analysis.load_inputs(dataset, session_input_arm())
    assert names, f"no positions in {dataset}"
    gt = position_eval_analysis.load_ground_truth(dataset, names, face_up_leaves)
    texts, legal = position_eval_analysis.load_placement_frame(dataset)
    timed_print(f"position-evaluation quality eval: {len(names)} positions from {dataset}")
    return {
        "inputs": inputs,
        "spatial_planes": spatial_planes,
        "gt": gt,
        "texts": texts,
        "legal": legal,
    }


def eval_position_eval_quality(model, quality_eval: dict, device) -> dict:
    """Aggregate model-vs-Monte-Carlo metrics on the large eval set: value and
    score-differential errors, plus placement-plane errors when the ground
    truth has placement planes."""
    model.eval()
    preds = position_eval_analysis.predict(
        model, quality_eval["inputs"], quality_eval["spatial_planes"], device
    )
    record = position_eval_analysis.quality_metrics(preds, quality_eval["gt"])
    if quality_eval["gt"]["placement"] is not None:
        planes = position_eval_analysis.collapse_placement(
            preds["placement_logits"], quality_eval["texts"]
        )
        record.update(
            position_eval_analysis.placement_metrics(
                planes, quality_eval["gt"]["placement"], quality_eval["legal"]
            )
        )
    return record


# ---------------------------------------------------------------------------
# The runner
# ---------------------------------------------------------------------------


def run(ctx: WorkerContext) -> int:
    """The train-role entry point."""
    params = ctx.params
    # Must precede any engine call: the arm is baked into the FFI session, whose
    # input shapes the model, export and eval batches all follow. A
    # face-up-leaves run's games expose the opponent's leave, so its model
    # takes the opponent-leave input block.
    set_opp_leave_input(params.face_up_leaves)

    paths = ctx.tag_paths()
    paths.root.mkdir(parents=True, exist_ok=True)
    device = torch.device(os.environ.get("SCZ_DEVICE", "cuda"))
    print(f"Tag root: {paths.root}\nDevice: {device}")

    in_shapes = {s.name: s.dims for s in get_input_shapes()}
    spatial_planes = in_shapes["input_spatial"][0]
    scalar_size = in_shapes["input_scalar"][0]
    model = PositionEvalModel(
        spatial_planes=spatial_planes,
        scalar_size=scalar_size,
        num_blocks=params.num_blocks,
        trunk_channels=params.trunk_channels,
        use_film=params.use_film,
        transformer=transformer_config(asdict(params)),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters")
    optimizer = build_optimizer(model, params)
    # See "Training-step performance" in the module docstring.
    torch.set_float32_matmul_precision("high")
    train_model = torch.compile(model)

    recorder = TrainRecorder(ctx.sink)
    recorder.publish_run(
        ctx.tag,
        asdict(params),
        n_params,
        {
            "loss_wld": params.lambda_wld,
            "loss_score_diff": params.lambda_sd,
            "loss_opp_next_placement": params.lambda_next_placement,
            "loss_self_next_placement": params.lambda_next_placement,
            "loss_opp_win_placement": params.lambda_win_placement,
            "loss_self_win_placement": params.lambda_win_placement,
        },
        default_controls(),
    )

    run_ctx = {
        "config": asdict(params),
        "sink": ctx.sink,
        "read_controls": functools.partial(read_controls, ctx.sink),
        "spatial_planes": spatial_planes,
        "scalar_size": scalar_size,
        "position_eval_quality": load_position_eval_quality(spatial_planes, params.face_up_leaves),
        "stats": WorkerStats(ctx),
        "deliverer": OutputDeliverer(),
    }

    restore_from_sink(paths, ctx.sink)
    state = checkpoint.resume(paths, model, optimizer, device)
    ensure_window(paths, ctx.sink, state.generation_index, params.window)
    _publish_train_state(paths, state)
    try:
        run_generational_training(
            model, train_model, optimizer, recorder, paths, device, params, state, run_ctx
        )
    except (KeyboardInterrupt, WorkerStopped):
        timed_print("Stopped; last completed epoch is checkpointed.")
    finally:
        # Finish pending uploads; the stop grace period is long enough for a
        # generation's outputs.
        run_ctx["deliverer"].drain()
    return 0
