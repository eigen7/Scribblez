"""Rack-best model: predict the longest word formable from a 7-tile rack.

The rack is fed as a length-7 one-hot sequence in **sorted** order (the anagram
tool relies on the sorted order; for the other tools it is just a canonical
order). A CLS transformer with an optional frozen lexicon tool drives an 8-way
head over the longest length (0..7; length 1 never occurs). Mirrors the
word-validity host with a classification head instead of a binary one.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scribblez.max_move_per_lane.lexicon_modules import LexiconModule

RACK_SIZE = 7
N_LETTERS = 26
N_CLASSES = RACK_SIZE + 1  # longest length 0..7


def encode_racks(racks: list[tuple]) -> torch.Tensor:
    """List of sorted letter tuples -> ``(N, 7)`` long letter indices (0..25)."""
    enc = np.zeros((len(racks), RACK_SIZE), dtype=np.int64)
    for i, rack in enumerate(racks):
        for j, ch in enumerate(rack):
            enc[i, j] = ord(ch) - ord("A")
    return torch.from_numpy(enc)


def onehot_racks(indices: torch.Tensor) -> torch.Tensor:
    """``(B, 7)`` indices -> ``(B, 7, 26)`` one-hot."""
    return F.one_hot(indices, N_LETTERS).float()


class RackBestModel(nn.Module):
    """CLS transformer over a sorted rack + optional frozen lexicon tool."""

    def __init__(
        self,
        channels: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        ffn_mult: int = 4,
        lexicon_module: LexiconModule | None = None,
        lane_ffn_mult: int | None = None,
    ):
        super().__init__()
        self.lexicon_module = lexicon_module
        n_lex = lexicon_module.n_tokens if lexicon_module is not None else 0
        self.n_prefix = 1 + n_lex  # CLS, then any lexicon-tool tokens

        self.embed = nn.Linear(N_LETTERS, channels)
        self.cls = nn.Parameter(torch.randn(1, 1, channels) * 0.02)
        self.pos = nn.Parameter(torch.randn(1, self.n_prefix + RACK_SIZE, channels) * 0.02)

        eff_ffn = ffn_mult if lane_ffn_mult is None else lane_ffn_mult
        layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=n_heads,
            dim_feedforward=max(1, eff_ffn * channels),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.head = nn.Sequential(
            nn.Linear(channels, channels), nn.GELU(), nn.Linear(channels, N_CLASSES)
        )

    def forward(self, rack_onehot: torch.Tensor) -> torch.Tensor:
        """rack_onehot: (B, 7, 26) sorted -> longest-length logits (B, 8)."""
        b = rack_onehot.size(0)
        feats = self.embed(rack_onehot)  # (B, 7, C)

        lex_tokens = None
        if self.lexicon_module is not None:
            out = self.lexicon_module(feats, rack_onehot)
            if out.cell_residual is not None:
                feats = feats + out.cell_residual
            lex_tokens = out.tokens

        cls = self.cls.expand(b, -1, -1)
        prefix = [cls] if lex_tokens is None else [cls, lex_tokens]
        x = torch.cat([*prefix, feats], dim=1)
        x = x + self.pos[:, : x.size(1)]
        x = self.encoder(x)
        return self.head(x[:, 0])  # CLS -> (B, 8)
