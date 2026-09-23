"""Rack-best model: generate a longest word from a rack, letter by letter.

A decoder-only transformer reads the 7 rack tiles, then emits a word. At every
step the rack masks out letters it has run out of, and the frozen forward DAWG
masks out letters that leave no valid word prefix and allows END only after a
complete word. Every decode that reaches END is therefore a real word from the
rack, and the network only has to learn to make it as long as possible. With
the DAWG mask off, the network must learn the lexicon itself; comparing the two
is the tool-use experiment (docs/rack_best_experiments.md).

Tokens:
  input (28):   0..25 letters, 26 = BOS, 27 = PAD
  output (27):  0..25 letters, 26 = END
  targets pad with 27, the cross-entropy ignore index
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scribblez.lexical_tool.compiler import CompiledLexicon

RACK_SIZE = 7
MAX_WORD = 7
# Generated positions: input BOS + up to 7 letters, predicting up to 7 letters + END.
MAX_GEN = MAX_WORD + 1
N_LETTERS = 26
END = 26  # output
BOS = 26  # input
PAD = 27  # input, and target ignore index
N_OUT = 27


def encode_racks(racks: list[tuple]) -> torch.Tensor:
    enc = np.zeros((len(racks), RACK_SIZE), dtype=np.int64)
    for i, rack in enumerate(racks):
        for j, ch in enumerate(rack):
            enc[i, j] = ord(ch) - ord("A")
    return torch.from_numpy(enc)


def encode_targets(words: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    """Teacher-forcing tensors (gen_in, target): ``[BOS, l0, l1, ...]`` and
    ``[l0, l1, ..., END]``, both padded to MAX_GEN."""
    gen_in = np.full((len(words), MAX_GEN), PAD, dtype=np.int64)
    target = np.full((len(words), MAX_GEN), PAD, dtype=np.int64)
    for i, word in enumerate(words):
        letters = [ord(c) - ord("A") for c in word]
        seq_in = [BOS, *letters][:MAX_GEN]
        seq_out = [*letters, END][:MAX_GEN]
        gen_in[i, : len(seq_in)] = seq_in
        target[i, : len(seq_out)] = seq_out
    return torch.from_numpy(gen_in), torch.from_numpy(target)


class RackWordModel(nn.Module):
    """Decoder-only transformer with an optional forward-DAWG constraint."""

    def __init__(
        self,
        compiled: CompiledLexicon,
        channels: int = 128,
        n_layers: int = 3,
        n_heads: int = 4,
        ffn_mult: int = 4,
        use_dawg: bool = True,
    ):
        super().__init__()
        self.use_dawg = use_dawg
        self.root = compiled.root
        self.dead = compiled.dead_state
        self.register_buffer("dawg_next", torch.from_numpy(compiled.next.astype(np.int64)))
        self.register_buffer("dawg_accept", torch.from_numpy(compiled.accept.astype(np.float32)))

        self.embed = nn.Embedding(28, channels)  # letters, BOS, PAD
        self.type_emb = nn.Embedding(2, channels)  # 0 rack, 1 generated
        seq_len = RACK_SIZE + MAX_GEN
        self.pos = nn.Parameter(torch.randn(1, seq_len, channels) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=n_heads,
            dim_feedforward=ffn_mult * channels,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.head = nn.Linear(channels, N_OUT)
        self.register_buffer(
            "causal", torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
        )
        types = torch.tensor([0] * RACK_SIZE + [1] * MAX_GEN)
        self.register_buffer("types", types)

    def logits(self, rack: torch.Tensor, gen_in: torch.Tensor) -> torch.Tensor:
        """Unmasked logits at the generated positions: (B, MAX_GEN, 27)."""
        tok = torch.cat([rack, gen_in], dim=1)  # (B, 7+MAX_GEN)
        x = self.embed(tok) + self.type_emb(self.types) + self.pos[:, : tok.size(1)]
        h = self.encoder(x, mask=self.causal)
        return self.head(h[:, RACK_SIZE:])

    def _step_mask(self, node, is_word, rack_rem) -> torch.Tensor:
        """Allowed next symbols (B, 27), given the DAWG node, whether the prefix so
        far is a word, and the rack tiles remaining."""
        available = rack_rem > 0  # (B, 26)
        if self.use_dawg:
            # A transition exists if it leads onward OR completes a word (a leaf
            # word-end has next == dead but accept set).
            exists = (self.dawg_next[node] != self.dead) | (self.dawg_accept[node] > 0)
            letter_ok = available & exists
            end_ok = is_word > 0
        else:
            letter_ok = available
            end_ok = torch.ones(node.size(0), dtype=torch.bool, device=node.device)
        return torch.cat([letter_ok, end_ok[:, None]], dim=1)

    def teacher_masks(self, rack: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Allowed-symbol masks (B, MAX_GEN, 27) along the target's prefixes."""
        b, device = rack.size(0), rack.device
        node = torch.full((b,), self.root, dtype=torch.long, device=device)
        is_word = torch.zeros(b, device=device)
        rack_rem = F.one_hot(rack, N_LETTERS).sum(1).float()  # (B, 26)
        masks = []
        for p in range(MAX_GEN):
            masks.append(self._step_mask(node, is_word, rack_rem))
            sym = target[:, p]
            is_letter = sym < N_LETTERS
            letter = sym.clamp(max=N_LETTERS - 1)
            node_next = self.dawg_next[node, letter]
            word_next = self.dawg_accept[node, letter]
            node = torch.where(is_letter, node_next, node)
            is_word = torch.where(is_letter, word_next, is_word)
            rack_rem = rack_rem - F.one_hot(letter, N_LETTERS) * is_letter[:, None]
        return torch.stack(masks, dim=1)

    def forward(self, rack: torch.Tensor, gen_in: torch.Tensor, target: torch.Tensor):
        """Masked logits (B, MAX_GEN, 27) for teacher-forced training.

        Masks with a large finite value, not -inf. Positions past the word's END
        have every symbol masked, and an all -inf row makes log_softmax NaN even
        though the loss ignores that position."""
        logits = self.logits(rack, gen_in)
        masks = self.teacher_masks(rack, target)
        return logits.masked_fill(~masks, -1e9)

    @torch.no_grad()
    def greedy(self, rack: torch.Tensor) -> list[list[int]]:
        """Greedy constrained decode; returns each rack's word as letter indices."""
        b, device = rack.size(0), rack.device
        node = torch.full((b,), self.root, dtype=torch.long, device=device)
        is_word = torch.zeros(b, device=device)
        rack_rem = F.one_hot(rack, N_LETTERS).sum(1).float()
        gen_in = torch.full((b, MAX_GEN), PAD, dtype=torch.long, device=device)
        gen_in[:, 0] = BOS
        decoded = torch.full((b, MAX_GEN), -1, dtype=torch.long, device=device)
        done = torch.zeros(b, dtype=torch.bool, device=device)

        for p in range(MAX_GEN):
            mask = self._step_mask(node, is_word, rack_rem)
            step = self.logits(rack, gen_in)[:, p].masked_fill(~mask, float("-inf"))
            sym = step.argmax(-1)
            # At a dead-end prefix every symbol is masked and argmax returns an
            # arbitrary index, so also check that the pick was allowed.
            picked_ok = mask.gather(1, sym[:, None]).squeeze(1)
            take = (sym < N_LETTERS) & ~done & picked_ok
            letter = sym.clamp(max=N_LETTERS - 1)
            decoded[:, p] = torch.where(take, letter, torch.full_like(letter, -1))
            node_next = self.dawg_next[node, letter]  # read before advancing `node`
            word_next = self.dawg_accept[node, letter]
            node = torch.where(take, node_next, node)
            is_word = torch.where(take, word_next, is_word)
            rack_rem = rack_rem - F.one_hot(letter, N_LETTERS) * take[:, None]
            if p + 1 < MAX_GEN:
                gen_in[:, p + 1] = torch.where(take, letter, torch.full_like(letter, PAD))
            done = done | ~take
            if bool(done.all()):
                break

        words = []
        for row in decoded.tolist():
            words.append([x for x in row if x >= 0])
        return words
