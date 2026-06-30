"""Word-validity model: is a given word lexicon-legal, or a phony?

The simplest tool-use probe. A word (2-15 letters) is classified valid/invalid.
The negatives are phonies generated to match the real lexicon's letter
statistics (see tools/generate_phony_lexicon.py), so surface features cannot
separate them -- a model can only succeed on held-out words by actually looking
them up. With a frozen compiled-lexicon tool (from the lexicon-module registry,
compiled from the *real* lexicon) plugged in, that lookup is available; without
one, the model can only memorize the training words and sits near chance on the
held-out split. The held-out accuracy is therefore the tool-use measurement.

The host is a small transformer over the padded word with a prepended CLS token
that drives the binary head. It mirrors the lane model: the lexicon tool feeds a
per-cell residual plus a couple of tokens, and `lane_ffn_mult` can shrink the
transformer's FFN ("replace" mode) so word knowledge must come from the tool.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scribblez.max_move_per_lane.lexicon_modules import LexiconModule

MAX_LEN = 15
N_LETTERS = 26


def encode_words(words: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode words to ``(indices (N, 15) long, lengths (N,) long)``. Letters are
    0..25; positions past a word's length are left at 0 (masked out by length)."""
    enc = np.zeros((len(words), MAX_LEN), dtype=np.int64)
    lengths = np.zeros(len(words), dtype=np.int64)
    for i, w in enumerate(words):
        lengths[i] = len(w)
        for j, ch in enumerate(w):
            enc[i, j] = ord(ch) - ord("A")
    return torch.from_numpy(enc), torch.from_numpy(lengths)


def onehot_batch(indices: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """``(B, 15) indices + (B,) lengths -> (B, 15, 26)`` one-hot, padding zeroed."""
    oh = F.one_hot(indices, N_LETTERS).float()
    valid = torch.arange(MAX_LEN, device=indices.device)[None, :] < lengths[:, None]
    return oh * valid[..., None]


class WordValidityModel(nn.Module):
    """CLS transformer over a padded word + optional frozen lexicon tool."""

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
        self.pos = nn.Parameter(torch.randn(1, self.n_prefix + MAX_LEN, channels) * 0.02)

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
        self.head = nn.Sequential(nn.Linear(channels, channels), nn.GELU(), nn.Linear(channels, 1))

    def forward(self, word_onehot: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """word_onehot: (B, 15, 26); lengths: (B,) -> validity logit (B,)."""
        b = word_onehot.size(0)
        feats = self.embed(word_onehot)  # (B, 15, C)

        lex_tokens = None
        if self.lexicon_module is not None:
            out = self.lexicon_module(feats, word_onehot)
            if out.cell_residual is not None:
                feats = feats + out.cell_residual
            lex_tokens = out.tokens

        cls = self.cls.expand(b, -1, -1)
        prefix = [cls] if lex_tokens is None else [cls, lex_tokens]
        x = torch.cat([*prefix, feats], dim=1)  # (B, n_prefix + 15, C)
        x = x + self.pos[:, : x.size(1)]

        pad = torch.arange(MAX_LEN, device=lengths.device)[None, :] >= lengths[:, None]
        mask = torch.cat(
            [torch.zeros(b, self.n_prefix, dtype=torch.bool, device=pad.device), pad], dim=1
        )
        x = self.encoder(x, src_key_padding_mask=mask)
        return self.head(x[:, 0]).squeeze(-1)  # CLS -> logit
