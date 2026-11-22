"""The largest activation a model reaches: how close it runs to the range of
the precision it is served in (FP16's largest finite value is 65504).

Measured over the inputs and outputs of every module in the model's forwards,
which covers a pre-norm residual stream (as the next norm's input) and an
MLP's hidden activations (as its output projection's input), the tensors
that grow when nothing in the loss opposes them (norm_inputs.py).
"""

import torch
import torch.nn as nn


def _tensors(value):
    """The floating-point tensors in a module's args or output."""
    if isinstance(value, torch.Tensor):
        if value.is_floating_point() and value.numel():
            yield value
    elif isinstance(value, (tuple, list)):
        for v in value:
            yield from _tensors(v)
    elif isinstance(value, dict):
        for v in value.values():
            yield from _tensors(v)


class ActivationPeak:
    """A context manager over `model`: while open, records the largest
    absolute value of any module's input or output in its forwards. The
    running maximum stays on the device until value()."""

    def __init__(self, model: nn.Module):
        self._model = model
        self._peak: torch.Tensor | None = None
        self._handles = []

    def __enter__(self):
        self._handles = [m.register_forward_hook(self._record) for m in self._model.modules()]
        return self

    def __exit__(self, *exc):
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def _record(self, module: nn.Module, args: tuple, output):
        for t in (*_tensors(args), *_tensors(output)):
            m = t.detach().abs().amax().float()
            self._peak = m if self._peak is None else torch.maximum(self._peak, m)

    def value(self) -> float:
        """The peak so far; 0 before any forward."""
        return 0.0 if self._peak is None else float(self._peak)
