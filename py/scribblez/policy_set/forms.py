"""The policy-set model's forms as the trainer drives them
(docs/plans/deep_bot_v0.md, "Training the forms together"): each builds its model on
its own encoder, computes its training loss terms on a batch, and builds sets.
The `forms` param picks one, "both" (slot and greedy) or "all".
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from scribblez.policy_set import (
    FORM_ALL,
    FORM_BOTH,
    FORM_GREEDY,
    FORM_POLICY,
    FORM_SLOT,
    greedy,
    policy,
    slot,
)
from scribblez.policy_set.model import PolicySetEncoder

Batch = dict[str, torch.Tensor]


@dataclass(frozen=True)
class Form:
    # (encoder, params) -> the form's model.
    build: Callable[[PolicySetEncoder, object], nn.Module]
    # (model, batch, params, rng) -> loss terms keyed by their record suffix;
    # the training loss is their sum.
    loss: Callable[[nn.Module, Batch, object, np.random.Generator], dict[str, torch.Tensor]]
    # (model, batch, n) -> each position's set of up to n, as candidate
    # indices within the position.
    sets: Callable[[nn.Module, Batch, int], list[list[int]]]
    # (model, batch) -> health scalars keyed by their record suffix, such as
    # the spread of its scores over a position's candidates.
    diagnose: Callable[[nn.Module, Batch], dict[str, torch.Tensor]]


def _slot_build(encoder: PolicySetEncoder, params) -> nn.Module:
    return slot.SlotModel(encoder, params.slots, params.slot_layers, params.num_heads)


def _slot_loss(model, batch: Batch, params, rng) -> dict[str, torch.Tensor]:
    _, by_n = slot.slot_loss(slot.slot_forward(model, batch)[0], batch)
    return {str(n): v for n, v in by_n.items()}


def _slot_sets(model, batch: Batch, n: int) -> list[list[int]]:
    logits, pad = slot.slot_forward(model, batch)
    return [picks[:n] for picks in slot.slot_picks(logits, pad)]


def _greedy_build(encoder: PolicySetEncoder, params) -> nn.Module:
    return greedy.GreedyModel(encoder, params.num_heads)


def _greedy_loss(model, batch: Batch, params, rng) -> dict[str, torch.Tensor]:
    position, chosen = greedy.sample_rows(
        batch["win_prob"].numpy(force=True), batch["move_equity"].numpy(force=True),
        batch["move_pos_id"].numpy(force=True), params.greedy_rows, rng,
    )  # fmt: skip
    rows = greedy.make_rows(position, chosen).to(batch["move_pos_id"].device)
    return {"gain": greedy.gain_loss(model, batch, rows)}


def _policy_build(encoder: PolicySetEncoder, params) -> nn.Module:
    return policy.PolicyModel(encoder)


def _policy_loss(model, batch: Batch, params, rng) -> dict[str, torch.Tensor]:
    return {"ce": policy.policy_loss(model, batch)}


FORMS = {
    FORM_SLOT: Form(_slot_build, _slot_loss, _slot_sets, slot.slot_diagnostics),
    FORM_GREEDY: Form(_greedy_build, _greedy_loss, greedy.greedy_picks, greedy.greedy_diagnostics),
    FORM_POLICY: Form(_policy_build, _policy_loss, policy.policy_picks, policy.policy_diagnostics),
}


def selected(forms: str) -> tuple[str, ...]:
    """The forms the `forms` param names."""
    if forms == FORM_BOTH:
        return (FORM_SLOT, FORM_GREEDY)
    if forms == FORM_ALL:
        return tuple(FORMS)
    return (forms,)
