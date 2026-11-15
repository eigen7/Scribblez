"""The leave inference model as a trainer drives it: built from a run's
params, trained one epoch over a window of generations, evaluated on game
directories, and exported for the engine. The leave_inference trainer drives
one beside its win table; the deep_bot trainer
(docs/plans/deep_bot_generations.md) drives it beside two other models.

The unit owns the model, its optimizer and the optimizer's arm. The trainer
owns the cursor, the checkpoint and the record.
"""

import functools
import time
from pathlib import Path

import numpy as np
import torch

from scribblez.dataset import SlogDataset
from scribblez.ffi import LeaveValueArrays, format_layout, struct_dtype, task_input_shapes
from scribblez.generational.controls import progress_line
from scribblez.generational.optim import build_optim_arm, build_optimizer
from scribblez.leave_inference import loss
from scribblez.leave_inference.heads import HEAD_TILTED
from scribblez.leave_inference.model import LeaveInferenceModel
from scribblez.leave_inference.onnx_export import FORMAT_KEY, TABLE_KEY, export_onnx
from scribblez.leave_inference.train_loop import LAYOUT, evaluate, run_epoch
from scribblez.position_eval.trainer import recalibration_batches
from scribblez.train_common import timed_print

EVAL_BATCH = 1024
FACE_UP_FLAG = format_layout()["constants"]["slog"]["flag_face_up_leaves"]


def task(params) -> str:
    """The rows the run trains and evaluates on (ffi.TASK_CODES)."""
    return "public_leave_inference" if params.public else "leave_inference"


class FaceUpGuard:
    """Refuses face-up-leaves corpora: their leaves are public, so there is
    nothing to infer. Each file is checked once."""

    def __init__(self):
        self._checked: set[Path] = set()
        self._header = struct_dtype("SlogFileHeader")

    def check(self, dirs: list[Path]):
        for path in (p for d in dirs for p in d.glob("*.slog")):
            if path in self._checked:
                continue
            flags = int(np.fromfile(path, self._header, count=1)[0]["flags"])
            if flags & FACE_UP_FLAG:
                raise RuntimeError(
                    f"{path} is a face-up-leaves corpus; leave inference needs hidden leaves"
                )
            self._checked.add(path)


def eval_batches(root, params, cpu, hide_private: bool = False):
    """The rows of the games under `root`, a directory or several, in a fixed
    order, every observer's private history hidden or none."""
    ds = SlogDataset(
        root,
        task=task(params),
        post_move=False,
        apply_symmetry=False,
        hide_private_fraction=1.0 if hide_private else 0.0,
        num_workers=cpu.dataloader_workers,
    )
    return ds.iter_batches(EVAL_BATCH, seed=0)


def hidden_metrics(own: dict[str, float], hidden: dict[str, float]) -> dict[str, float]:
    """The NLL with every observer's private history hidden, as the model is
    served for an opponent, and its gap to the observer's own view over the
    same rows."""
    return {
        "eval_hidden_nll": hidden["eval_nll"],
        "eval_hidden_nll_vs_uniform": hidden["eval_nll_vs_uniform"],
        "eval_hidden_gap": hidden["eval_nll"] - own["eval_nll"],
    }


class LeaveInferenceUnit:
    def __init__(self, params, device):
        self.params = params
        self.device = device
        self.table = loss.TableTensors(LeaveValueArrays(), device)
        in_shapes = {s.name: s.dims for s in task_input_shapes(task(params))}
        self.spatial_planes = in_shapes["input_spatial"][0]
        self.scalar_size = in_shapes["input_scalar"][0]
        self.model = LeaveInferenceModel(
            spatial_planes=self.spatial_planes,
            scalar_size=self.scalar_size,
            trunk_channels=params.trunk_channels,
            num_blocks=params.num_blocks,
            head=params.head,
            normalize_scalar=params.normalize_scalar,
        ).to(device)
        self.optimizer = build_optimizer(self.model, params)
        # TF32 for the trunk; the tilted-prior pass pins its own precision.
        torch.set_float32_matmul_precision("high")
        self.train_model = torch.compile(self.model)
        self.optim_arm = None
        self._face_up_guard = FaceUpGuard()

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def versions(self) -> dict:
        """What a checkpoint of this model must match to be resumed: the row
        format and the leave values its table was built from."""
        return {FORMAT_KEY: LAYOUT["format_version"], TABLE_KEY: self.table.klv_hash}

    def start(self, recorder, rows_trained: int):
        """Take up the optimizer's schedule at `rows_trained`, once the
        checkpoint is loaded."""
        self.optim_arm = build_optim_arm(recorder, self.params, self.optimizer, rows_trained)

    def train(self, window: list[Path], gen: int, rows_trained: int, cpu):
        """One epoch over `window` as generation `gen`'s, the rows-clock at
        `rows_trained`. Returns the epoch's result (train_loop.run_epoch) and
        its seconds, and leaves the model as it is evaluated and served
        (optim_arm.eval_mode) until resume_training()."""
        params = self.params
        self._face_up_guard.check(window)
        ds = SlogDataset(
            window,
            task=task(params),
            post_move=False,
            apply_symmetry=True,
            hide_private_fraction=params.hide_private_fraction,
            num_workers=cpu.dataloader_workers,
        )
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
            self.table,
            batches,
            self.device,
            lr_fn=self.optim_arm.lr_fn,
            rows_trained=rows_trained,
            grad_clip=params.grad_clip,
            on_batch=functools.partial(progress_line, gen),
        )
        elapsed = time.time() - t0
        self.optim_arm.eval_mode(self.model, recalibration_batches(ds, params, gen, self.device))
        return result, elapsed

    def resume_training(self):
        self.optim_arm.train_mode()

    def evaluate(self, root, cpu) -> dict[str, float]:
        """The model's metrics on the games under `root`, from each
        observer's own view and with every private history hidden."""
        own = evaluate(self.model, self.table, eval_batches(root, self.params, cpu), self.device)
        hidden = evaluate(
            self.model, self.table, eval_batches(root, self.params, cpu, True), self.device
        )
        return own | hidden_metrics(own, hidden)

    def export(self, path: Path, provenance: dict) -> bool:
        """Export the model to `path` for the engine, if its head is served;
        whether it was."""
        if self.params.head != HEAD_TILTED:
            return False
        export_onnx(
            self.model,
            path,
            self.spatial_planes,
            self.scalar_size,
            leave_values=self.table.klv_hash,
            public=self.params.public,
            provenance=provenance,
        )
        return True
