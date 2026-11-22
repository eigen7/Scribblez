"""The leaf eval model as a trainer drives it: built from a run's params,
trained one epoch over a window of generations, evaluated on game
directories beside two win tables, and exported for the engine. The
deep_bot_leaf_eval trainer drives one; the deep_bot trainer
(docs/plans/deep_bot_generations.md) drives it beside two other models.

The unit owns the model, its optimizer and the optimizer's arm. The trainer
owns the cursor, the checkpoint and the record.
"""

import functools
import time
from pathlib import Path

import torch

from scribblez.activation_peak import ActivationPeak
from scribblez.dataset import SlogDataset
from scribblez.deep_bot_leaf_eval.evaluation import evaluate
from scribblez.deep_bot_leaf_eval.model import (
    LAYOUT,
    LeafEvalModel,
    LeafEvalShape,
    LeafLossConfig,
)
from scribblez.deep_bot_leaf_eval.onnx_export import export_onnx
from scribblez.ffi import LeaveValueArrays
from scribblez.generational.controls import progress_line
from scribblez.generational.optim import build_optim_arm, build_optimizer
from scribblez.leave_inference import win_table
from scribblez.leave_inference.unit import FaceUpGuard
from scribblez.position_eval.train_loop import run_epoch
from scribblez.position_eval.trainer import recalibration_batches
from scribblez.train_common import timed_print

TASK = "deep_bot_leaf_eval"
EVAL_BATCH = 1024
FORMAT_KEY = "leaf_eval_format_version"
LEAVE_VALUES_KEY = "leave_values"


def model_shape(params) -> LeafEvalShape:
    return LeafEvalShape(
        trunk=params.trunk,
        join_planes=params.join_planes,
        width=params.width,
        depth=params.depth,
        num_heads=params.num_heads,
        normalize_scalar=params.normalize_scalar,
    )


def loss_config(params) -> LeafLossConfig:
    """position_eval's loss config, without its placement heads, plus the
    per-square and end-of-game heads'."""
    return LeafLossConfig(
        lambda_wld=params.lambda_wld,
        lambda_sd=params.lambda_sd,
        lambda_next_placement=0.0,
        lambda_win_placement=0.0,
        huber_delta_mean=params.huber_delta_mean,
        huber_delta_std=params.huber_delta_std,
        lambda_ownership=params.lambda_ownership,
        lambda_points=params.lambda_points,
        lambda_going_out=params.lambda_going_out,
        lambda_end_points=params.lambda_end_points,
        lambda_norm_input=params.lambda_norm_input,
    )


def loss_weights(params) -> dict[str, float]:
    """Each recorded loss's weight in the total, for the run record."""
    return {
        "loss_wld": params.lambda_wld,
        "loss_score_diff": params.lambda_sd,
        "loss_ownership": params.lambda_ownership,
        "loss_square_points": params.lambda_points,
        "loss_going_out": params.lambda_going_out,
        "loss_end_points": params.lambda_end_points,
        "loss_norm_input": params.lambda_norm_input,
    }


class LeafEvalUnit:
    def __init__(self, params, device):
        self.params = params
        self.device = device
        self.model = LeafEvalModel(model_shape(params)).to(device)
        self.optimizer = build_optimizer(self.model, params)
        torch.set_float32_matmul_precision("high")
        self.train_model = torch.compile(self.model)
        self.leave_values = LeaveValueArrays().content_hash
        self.loss_cfg = loss_config(params)
        self.optim_arm = None
        self._face_up_guard = FaceUpGuard()

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def versions(self) -> dict:
        """What a checkpoint of this model must match to be resumed: the row
        format and the leave values the rows read."""
        return {FORMAT_KEY: LAYOUT["format_version"], LEAVE_VALUES_KEY: self.leave_values}

    def start(self, recorder, rows_trained: int):
        """Take up the optimizer's schedule at `rows_trained`, once the
        checkpoint is loaded."""
        self.optim_arm = build_optim_arm(recorder, self.params, self.optimizer, rows_trained)

    def train(self, window: list[Path], gen: int, rows_trained: int, cpu):
        """One epoch over `window` as generation `gen`'s, the rows-clock at
        `rows_trained`. Returns the epoch's result
        (position_eval.train_loop.run_epoch) and its seconds, and leaves the
        model as it is evaluated and served (optim_arm.eval_mode) until
        resume_training()."""
        params = self.params
        self._face_up_guard.check(window)
        ds = SlogDataset(window, task=TASK, apply_symmetry=True, num_workers=cpu.dataloader_workers)
        timed_print(
            f"generation {gen}: window {[d.name for d in window]} "
            f"({ds.num_games} games, {ds.num_samples} rows)"
        )
        # drop_last keeps one static shape for the compiled forward.
        batches = ds.iter_batches(
            params.batch_size,
            seed=gen * 1000003,
            turns_per_game=params.turns_per_game,
            epoch_index=gen,
            drop_last=True,
        )
        t0 = time.time()
        self.optim_arm.train_mode()
        result = run_epoch(
            self.train_model,
            self.optimizer,
            batches,
            self.device,
            self.loss_cfg,
            lr_fn=self.optim_arm.lr_fn,
            rows_trained=rows_trained,
            on_batch=functools.partial(progress_line, gen),
            grad_clip=params.grad_clip,
        )
        elapsed = time.time() - t0
        self.optim_arm.eval_mode(self.model, recalibration_batches(ds, params, gen, self.device))
        return result, elapsed

    def resume_training(self):
        self.optim_arm.train_mode()

    def evaluate(self, root, cpu, fitted_table: Path | None = None) -> dict[str, float]:
        """The metrics on the games under `root`, a directory or several: the
        model's, `fitted_table`'s (a win table) when there is one, and
        Macondo's table's; and activation_peak, the largest activation the
        model reaches on them, against FP16's 65504 (activation_peak.py)."""
        ds = SlogDataset(root, task=TASK, apply_symmetry=False, num_workers=cpu.dataloader_workers)
        batches = ds.iter_batches(EVAL_BATCH, seed=0)
        with ActivationPeak(self.model) as peak:
            metrics = evaluate(
                self.model, batches, self.device, fitted_table, win_table.MACONDO_WIN_TABLE
            )
        return {**metrics, "activation_peak": peak.value()}

    def export(self, path: Path, provenance: dict):
        """Export the model to `path` for the engine (GPU rollouts' leaf)."""
        export_onnx(self.model, path, leave_values=self.leave_values, provenance=provenance)
