"""Word-validity model: is a word in the lexicon, or a phony?

The simplest lexical-tool probe. The negatives are phonies with the real
lexicon's letter statistics (tools/generate_phony_lexicon.py), so surface
features do not separate them: on held-out words a model succeeds only by
looking the word up. A compiled-lexicon tool built from the real lexicon makes
that lookup possible. Without one, the model can only memorize its training
words and should sit near chance on the held-out split, so held-out accuracy
measures tool use. Results are in docs/word_validity_experiments.md.

The model is a small transformer over the padded word, classifying from a
prepended CLS token. The tool plugs in the same way as in the max-move-per-lane
lane transformer: a per-cell residual plus prefix tokens, with `lane_ffn_mult`
available to shrink the FFN so word knowledge has to come from the tool.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scribblez.lexical_tool.modules import LexiconModule

MAX_LEN = 15
N_LETTERS = 26


def encode_words(words: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode words as (letter indices (N, 15), lengths (N,)). Positions past a
    word's end hold 0 and are masked by length downstream."""
    enc = np.zeros((len(words), MAX_LEN), dtype=np.int64)
    lengths = np.zeros(len(words), dtype=np.int64)
    for i, w in enumerate(words):
        lengths[i] = len(w)
        for j, ch in enumerate(w):
            enc[i, j] = ord(ch) - ord("A")
    return torch.from_numpy(enc), torch.from_numpy(lengths)


def onehot_batch(indices: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """One-hot letters (B, 15, 26), all-zero past each word's end."""
    oh = F.one_hot(indices, N_LETTERS).float()
    valid = torch.arange(MAX_LEN, device=indices.device)[None, :] < lengths[:, None]
    return oh * valid[..., None]


class WordValidityModel(nn.Module):
    """CLS-token transformer over a padded word, with an optional lexicon tool."""

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
        """(B, 15, 26) one-hot words and (B,) lengths -> (B,) validity logits."""
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
