"""The policy-set model's forms as a trainer drives them: built from a run's
params, trained one epoch over a window of generations' pools, evaluated on
pools beside the static-equity baseline, and the policy form exported for
PolicyBot. The policy_set trainer drives one; the deep_bot trainer
(docs/plans/deep_bot_generations.md) drives it beside two other models.

The unit owns the forms' models, their optimizer and the optimizer's arm. The
trainer owns the cursor, the checkpoint and the record.
"""

import functools
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from scribblez.ffi import format_layout, task_input_shapes
from scribblez.generational.controls import progress_line
from scribblez.generational.finite_guard import check_parameters
from scribblez.generational.optim import build_optim_arm, build_optimizer
from scribblez.move_set_eval.moves import move_encoding_version
from scribblez.policy_set import FORM_POLICY, dataset, forms, metrics, train_loop
from scribblez.policy_set.model import PolicySetEncoder, trunk_scalar
from scribblez.policy_set.onnx_export import export_onnx
from scribblez.train_common import timed_print

# The baseline's record key prefix; each form's records carry its own.
BASELINE = "equity_"
# The checkpoint-config entries a resume must match.
VERSION_KEYS = ("policy_set_format_version", "move_encoding_version", "input_encoding_version")
# Batches the schedule-free arm recomputes BatchNorm statistics over before
# evaluating and checkpointing (optim.ScheduleFreeArm.eval_mode).
BN_RECALIBRATION_BATCHES = 16


def versions() -> dict[str, int]:
    """This code's values of VERSION_KEYS."""
    constants = format_layout()["constants"]
    return {
        "policy_set_format_version": constants["policy_set"]["format_version"],
        "move_encoding_version": move_encoding_version(),
        "input_encoding_version": constants["input_encoding_version"],
    }


def build_encoder(params) -> PolicySetEncoder:
    shapes = {s.name: s.dims for s in task_input_shapes()}
    return PolicySetEncoder(
        shapes["input_spatial"][0],
        shapes["input_scalar"][0],
        params.trunk_channels,
        params.num_blocks,
        params.num_heads,
        params.normalize_scalar,
    )


def build_models(params) -> nn.ModuleDict:
    """The forms `params` selects (forms.selected), each on its own encoder."""
    return nn.ModuleDict(
        {
            name: forms.FORMS[name].build(build_encoder(params), params)
            for name in forms.selected(params)
        }
    )


def trunk_forward(models: nn.ModuleDict, spatial, scalar):
    """The passes that reach the models' BatchNorm layers: each form's trunk,
    on the trunk's scalar input (model.trunk_scalar)."""
    for model in models.values():
        model.encoder.trunk(spatial, scalar)


def recalibration_batches(batcher, items, params, gen, device):
    """(spatial, scalar) pairs for BatchNorm recalibration, drawn under a seed
    distinct from the epoch's."""
    order = dataset.batch_order(len(items), params.batch_size, seed=gen * 1000003 + 1)
    for idx in order[:BN_RECALIBRATION_BATCHES]:
        b = batcher.build([items[i] for i in idx])
        yield (
            b["input_spatial"].to(device),
            trunk_scalar(b["input_scalar"], b["opp_rack"]).to(device),
        )


def eval_record(models, batcher, held: list[dataset.PoolItem], params, device) -> dict:
    """The pools' statistics, the baseline's metrics, and each form's
    metrics."""
    if not held:
        return {}
    pool_list = [it.pool for it in held]
    return {
        **metrics.pool_stats(pool_list, "eval_"),
        **metrics.set_metrics(pool_list, metrics.equity_top, BASELINE),
        **train_loop.evaluate(models, batcher, held, device, params),
    }


@torch.no_grad()
def parameter_norm(model: nn.Module) -> float:
    """The global L2 norm of `model`'s parameters: a slow divergence shows as
    its growth well before anything goes non-finite."""
    return float(torch.stack([p.float().norm() for p in model.parameters()]).norm())


class PolicySetUnit:
    def __init__(self, params, device):
        self.params = params
        self.device = device
        self.models = build_models(params).to(device)
        self.optimizer = build_optimizer(self.models, params)
        self.batcher = dataset.PoolBatcher()
        self.optim_arm = None
        # The pools of the last epoch's window, whose statistics the record
        # carries.
        self.items: list[dataset.PoolItem] = []

    @property
    def model(self) -> nn.ModuleDict:
        """What the checkpoint saves: every form."""
        return self.models

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.models.parameters())

    def config(self) -> dict:
        """The params and versions an export and a checkpoint record."""
        return {**asdict(self.params), **versions()}

    def start(self, recorder, rows_trained: int):
        """Take up the optimizer's schedule at `rows_trained`, once the
        checkpoint is loaded."""
        self.optim_arm = build_optim_arm(recorder, self.params, self.optimizer, rows_trained)

    def train(self, window: list[Path], gen: int, rows_trained: int):
        """One epoch over `window`'s pools as generation `gen`'s, the
        rows-clock at `rows_trained`. Returns the epoch's result
        (train_loop.run_epoch) and its seconds, and leaves the models as they
        are evaluated and served (optim_arm.eval_mode) until
        resume_training()."""
        self.items = dataset.read_items(window)
        timed_print(
            f"generation {gen}: window {[d.name for d in window]} ({len(self.items)} pools)"
        )
        t0 = time.time()
        self.optim_arm.train_mode()
        result = train_loop.run_epoch(
            self.models,
            self.optimizer,
            self.batcher,
            self.items,
            self.device,
            self.params,
            seed=gen * 1000003,
            lr_fn=self.optim_arm.lr_fn,
            rows_trained=rows_trained,
            on_batch=functools.partial(progress_line, gen),
        )
        elapsed = time.time() - t0
        sys.stdout.write("\n")
        self.optim_arm.eval_mode(
            self.models,
            recalibration_batches(self.batcher, self.items, self.params, gen, self.device),
            trunk_forward,
        )
        return result, elapsed

    def resume_training(self):
        self.optim_arm.train_mode()

    def train_metrics(self, result) -> dict:
        """The last epoch's losses, and the gradient and parameter norms."""
        return {
            "loss": result.loss,
            **{f"loss_{key}": v for key, v in result.terms.items()},
            **{f"grad_norm_{name}": v for name, v in result.grad_norms.items()},
            **{f"param_norm_{name}": parameter_norm(m) for name, m in self.models.items()},
        }

    def window_stats(self) -> dict:
        """The statistics of the last epoch's pools."""
        return metrics.pool_stats([it.pool for it in self.items], "")

    def evaluate(self, held: list[dataset.PoolItem]) -> dict:
        """`held`'s statistics, the baseline's metrics on them, and each
        form's."""
        return eval_record(self.models, self.batcher, held, self.params, self.device)

    def check_finite(self, context: str):
        """Stop the worker fatally if a form's parameters went non-finite."""
        check_parameters(self.models.items(), context)

    def export(self, path: Path, provenance: dict) -> bool:
        """Export the policy form to `path` for PolicyBot, if the run trains
        one; whether it did."""
        if FORM_POLICY not in self.models:
            return False
        export_onnx(self.models[FORM_POLICY], path, self.config(), provenance)
        return True
