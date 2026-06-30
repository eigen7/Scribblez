"""Frozen, compiled-lexicon modules the network can learn to use as a tool.

The premise: instead of forcing the network to grow an internal copy of the
lexicon, compile the lexicon into a module whose lexical weights are *frozen*
and *plug it in* as an input the network learns to query -- like handing a USB
drive to a computer. The compiled lexicon (a DAWG transition table, see
:mod:`lexicon_compiler`) is held in non-trainable buffers; the thin query/read
adapters around it are ordinary parameters. Those adapters *are* "the network
learning to operate the tool": gradients teach it what to ask and how to read
the answer, while the lexicon itself never changes.

A module is selected by name (``--lexicon-module``) from
:data:`LEXICON_MODULE_REGISTRY`. Each obeys one interface:

    forward(lane_feats:   (M, 15, C),    # the network's per-cell lane features
            lane_letters: (M, 15, 26))   # the known board tiles on the lane
        -> LexiconOutput(cell_residual: (M, 15, C) | None,
                         tokens:        (M, T, C)  | None)

``cell_residual`` is added to the 15 lane cells; ``tokens`` are prepended to the
lane's transformer sequence (alongside the rack tokens). Crucially, a module is
queried with the network's *learned* representation, never with the ground-truth
answer -- otherwise it would solve the task for the wrong reason and make the
held-out-word generalization test meaningless.

Modules run on every lane, rows and columns, with shared weights (transpose
sharing), matching the lane transformer: a word is a word along either axis.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from scribblez.max_move_per_lane.lexicon_compiler import (
    N_LETTERS,
    CompiledLexicon,
    compile_kwg,
)


@dataclass
class LexiconOutput:
    """What a lexicon module contributes to a lane's transformer sequence.

    cell_residual: ``(M, 15, C)`` added to the per-cell lane features, or None.
    tokens: ``(M, T, C)`` extra tokens prepended to the lane sequence, or None.
    cell_signals: ``(M, 15, S)`` the module's raw, interpretable per-cell lexical
        readout (e.g. accept / continuation / alive), or None. The network does
        not consume this -- it is exposed for tests and dashboard visualization.
    """

    cell_residual: torch.Tensor | None = None
    tokens: torch.Tensor | None = None
    cell_signals: torch.Tensor | None = None


class LexiconModule(nn.Module):
    """Base class for compiled-lexicon modules.

    Subclasses set :attr:`n_tokens` (how many tokens they prepend per lane, so
    the lane transformer can size its positional table) and implement forward.
    """

    n_tokens: int = 0


LEXICON_MODULE_REGISTRY: dict[str, Callable[..., LexiconModule]] = {}


def register_module(
    name: str,
) -> Callable[[Callable[..., LexiconModule]], Callable[..., LexiconModule]]:
    def deco(factory: Callable[..., LexiconModule]) -> Callable[..., LexiconModule]:
        LEXICON_MODULE_REGISTRY[name] = factory
        return factory

    return deco


def available_modules() -> list[str]:
    """All selectable ``--lexicon-module`` names, including the ``none`` no-op."""
    return ["none"] + sorted(LEXICON_MODULE_REGISTRY)


def parse_module_opts(items: list[str]) -> dict[str, object]:
    """Parse repeated ``KEY=VALUE`` CLI options into a kwargs dict.

    Values are coerced to int, then float, else left as a string."""
    opts: dict[str, object] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"lexicon option must be KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        opts[key.strip()] = _coerce(value.strip())
    return opts


def _coerce(value: str) -> object:
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    return value


def build_lexicon_module(
    name: str, *, channels: int, kwg_path: str, **opts: object
) -> LexiconModule | None:
    """Construct the named lexicon module, or None for ``"none"``.

    The lexicon is compiled from ``kwg_path``, which MUST be the same lexicon the
    self-play labels are computed from, or the frozen module and the training
    targets describe different dictionaries."""
    if name == "none":
        return None
    if name not in LEXICON_MODULE_REGISTRY:
        raise KeyError(f"unknown lexicon module {name!r}; choices: {available_modules()}")
    compiled = compile_kwg(kwg_path)
    return LEXICON_MODULE_REGISTRY[name](channels=channels, compiled=compiled, **opts)


@register_module("soft_traversal")
class SoftTraversalLexicon(LexiconModule):
    """A differentiable forward-DAWG walk along the lane, queried by the network.

    Design
    ------
    The frozen DAWG is a transition table; this module walks it left-to-right
    along a lane while the network supplies the letters. At each cell the letter
    fed in is the known board tile if the cell is occupied, else a *soft* query
    distribution the network emits from its own lane features. A learned restart
    gate lets a fresh word begin at any cell (words are not anchored at cell 0),
    so one left-to-right pass covers plays starting anywhere.

    The traversal state is a distribution over DAWG nodes, kept tractable as a
    sparse top-K set (the lexicon has tens of thousands of nodes; a dense state
    is infeasible). Each step reads out, differentiably:
      * accept   -- probability the soft prefix ending here is a complete word;
      * cont     -- per-letter legality: which letters the lexicon permits next
                    given the soft prefix (a soft, network-derived cross-check);
      * alive    -- how much of the soft letter mass was a legal continuation.
    These per-cell signals become a cell residual; their pooled summary becomes
    two lane tokens.

    Frozen vs. trainable
    --------------------
    The transition/accept/exists tables are buffers (frozen lexicon). Trainable:
    the query head (what letter to ask about), the restart gate, and the readout
    projections (how to turn lexical answers into features). Learning these is
    the network learning to use the tool.

    Tradeoffs / suitability
    -----------------------
    Faithful to "a differentiable automaton": exact transitions, true gradients
    through the soft letters, de-assemblable lexicon. Approximations: top-K
    truncation drops low-probability branches, and equal target states from
    different (state, letter) pairs are not merged before the top-K (a mild
    over-spread). Cost is O(lane_len * K * 26) per lane and is the heaviest of
    the planned modules.

    Toy vs. general: the left-to-right, single-anchor walk is tailored to the
    per-lane toy task -- it does not model a play threading bidirectionally
    through existing tiles, nor cross-words on the perpendicular axis (those are
    left to the shared transformer). The *soft* state is, however, exactly the
    primitive a future win-probability model wants (a soft prefix = genuine
    uncertainty over the opponent's tiles / the bag), so the mechanism is meant
    to transfer even though this particular anchoring is toy-specific.

    Options: ``topk`` (tracked states, default 16).
    """

    def __init__(self, *, channels: int, compiled: CompiledLexicon, topk: int = 16):
        super().__init__()
        self.n_tokens = 2
        self.channels = channels
        self.topk = int(topk)
        self.root = compiled.root
        self.dead = compiled.dead_state

        # Frozen lexicon: next state, acceptance, and letter-legality per state.
        exists = (compiled.next != compiled.dead_state) | compiled.accept
        self.register_buffer("next_tbl", torch.from_numpy(compiled.next.astype(np.int64)))
        self.register_buffer("accept_tbl", torch.from_numpy(compiled.accept.astype(np.float32)))
        self.register_buffer("exists_tbl", torch.from_numpy(exists.astype(np.float32)))

        # Trainable adapters (the network's "hands" on the tool).
        self.query = nn.Linear(channels, N_LETTERS)  # which letter to ask about
        self.restart = nn.Linear(channels, 1)  # may a new word start here?
        feat_dim = 1 + N_LETTERS + 1  # accept, cont(26), alive
        self.readout = nn.Linear(feat_dim, channels)
        self.token_proj = nn.Linear(2 * feat_dim, self.n_tokens * channels)

    def forward(self, lane_feats: torch.Tensor, lane_letters: torch.Tensor) -> LexiconOutput:
        m, length, _ = lane_feats.shape
        k = self.topk
        device = lane_feats.device

        # Letter fed at each cell: board tile if occupied, else the soft query.
        q = torch.softmax(self.query(lane_feats), dim=-1)  # (M, L, 26)
        occ = lane_letters.sum(-1, keepdim=True).clamp(max=1.0)  # (M, L, 1)
        letters_in = occ * lane_letters + (1.0 - occ) * q  # (M, L, 26)
        restart = torch.sigmoid(self.restart(lane_feats)).squeeze(-1)  # (M, L)

        # Sparse top-K state, initialized at the root with all the mass.
        val = lane_feats.new_zeros(m, k)
        val[:, 0] = 1.0
        idx = torch.full((m, k), self.dead, dtype=torch.long, device=device)
        idx[:, 0] = self.root
        root_col = torch.full((m, 1), self.root, dtype=torch.long, device=device)

        accept_steps, cont_steps, alive_steps = [], [], []
        for j in range(length):
            gate = restart[:, j : j + 1]  # (M, 1)
            cand_val = torch.cat([(1.0 - gate) * val, gate], dim=1)  # (M, K+1)
            cand_idx = torch.cat([idx, root_col], dim=1)  # (M, K+1)

            nxt = self.next_tbl[cand_idx]  # (M, K+1, 26)
            acc = self.accept_tbl[cand_idx]  # (M, K+1, 26)
            exists = self.exists_tbl[cand_idx]  # (M, K+1, 26)
            flow = cand_val.unsqueeze(-1) * letters_in[:, j].unsqueeze(1)  # (M, K+1, 26)

            cont_steps.append((cand_val.unsqueeze(-1) * exists).sum(1))  # (M, 26)
            accept_steps.append((flow * acc).sum(dim=(1, 2)))  # (M,)

            flat_idx = nxt.reshape(m, -1)
            flat_flow = flow.reshape(m, -1).masked_fill(flat_idx == self.dead, 0.0)
            top_flow, top_pos = flat_flow.topk(k, dim=1)
            idx = flat_idx.gather(1, top_pos)
            # Keep raw probability mass (no renormalization): the mass on a prefix
            # IS its soft probability, so accept reflects how likely the queried
            # word is and gradients reach the query head. The surviving mass
            # ("alive") doubles as a signal of how lexically legal the prefix is.
            val = top_flow
            alive_steps.append(val.sum(1))

        accept = torch.stack(accept_steps, dim=1).unsqueeze(-1)  # (M, L, 1)
        cont = torch.stack(cont_steps, dim=1)  # (M, L, 26)
        alive = torch.stack(alive_steps, dim=1).unsqueeze(-1)  # (M, L, 1)
        feat = torch.cat([accept, cont, alive], dim=-1)  # (M, L, feat_dim)

        cell_residual = self.readout(feat)  # (M, L, C)
        pooled = torch.cat([feat.mean(1), feat.amax(1)], dim=-1)  # (M, 2*feat_dim)
        tokens = self.token_proj(pooled).view(m, self.n_tokens, self.channels)
        return LexiconOutput(cell_residual=cell_residual, tokens=tokens, cell_signals=feat)
