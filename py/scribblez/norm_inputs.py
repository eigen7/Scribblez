"""The restoring force on activation scales that the loss cannot see.

A normalization layer (LayerNorm, BatchNorm, RMSNorm) discards its input's
scale, so the training loss never opposes the growth of a tensor that feeds
one: a pre-norm transformer's residual stream into its final norm, a conv
block's activations into the next BatchNorm. Under the optimizer that scale
drifts upward without bound, and an FP16-served model eventually overflows
(docs/plans/fp16_safe_serving.md, "The leaf network"). A small penalty on the
mean square of every normalization input holds the scale down. The loss is
indifferent to the scale, so the penalty need not trade against the fit while
the scale stays small; shrinking one that has already run away is not free.
"""

import torch
import torch.nn as nn

from scribblez import transformer_tower

NORM_TYPES = (
    transformer_tower.RMSNorm,
    nn.LayerNorm,
    nn.RMSNorm,
    nn.GroupNorm,
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
)


class NormInputSquares:
    """Collects the mean square of every normalization layer's input in a
    model, between start() and take(): the model's forward brackets itself
    with them in training mode, so each training forward reports exactly its
    own norms' inputs (and works under torch.compile, which traces the hooks
    and the list into the graph)."""

    def __init__(self, model: nn.Module):
        self._squares: list[torch.Tensor] = []
        self._active = False
        for m in model.modules():
            if isinstance(m, NORM_TYPES):
                m.register_forward_pre_hook(self._record)

    def _record(self, module: nn.Module, args: tuple):
        if self._active:
            self._squares.append(args[0].float().square().mean())

    def start(self):
        self._squares = []
        self._active = True

    def take(self) -> torch.Tensor:
        """The mean, over the norms that ran since start(), of each one's mean
        squared input."""
        squares, self._squares = self._squares, []
        self._active = False
        return torch.stack(squares).mean()
