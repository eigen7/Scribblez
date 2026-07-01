"""Tests for the rack-best toy: labels, the anagram module, and the host."""

import torch
import torch.nn.functional as F
from scribblez.max_move_per_lane.lexicon_compiler import CompiledLexicon
from scribblez.max_move_per_lane.lexicon_modules import AnagramLexicon, SoftTraversalLexicon
from scribblez.rack_best.data import longest_word_length
from scribblez.rack_best.model import RackBestModel, encode_racks, onehot_racks


def test_longest_word_length():
    wordbags = {2: {("A", "T")}, 3: {("A", "C", "T")}, 4: {("A", "C", "E", "R")}}
    assert longest_word_length(("A", "C", "T"), wordbags) == 3  # ACT (=CAT)
    assert longest_word_length(("A", "T", "X"), wordbags) == 2  # AT, no 3-word
    assert longest_word_length(("A", "C", "E", "R"), wordbags) == 4  # ACER (=CARE)
    assert longest_word_length(("X", "Y", "Z"), wordbags) == 0  # nothing


def _lex():
    return CompiledLexicon.from_words(["CAT", "AT", "CARE", "DOG", "EAT", "ATE"])


def test_anagram_module_finds_subset_lengths():
    mod = AnagramLexicon(channels=8, compiled=_lex())

    def achievable(letters):
        s = "".join(sorted(letters))
        oh = torch.zeros(1, len(s), 26)
        for i, ch in enumerate(s):
            oh[0, i, ord(ch) - ord("A")] = 1.0
        acc = mod(torch.zeros(1, len(s), 8), oh).cell_signals[0, 0]
        return {length for length in range(mod.n_bins) if acc[length] > 0}

    assert achievable("ACT") == {2, 3}  # AT, CAT
    assert achievable("CARE") == {4}  # CARE only (no T for AT/EAT)
    assert achievable("ATX") == {2}  # AT
    assert achievable("XYZ") == set()  # nothing

    # Gradient reaches the readout (the frozen tables/depths do not train).
    mod.zero_grad()
    oh = torch.zeros(1, 3, 26)
    for i, ch in enumerate("ACT"):
        oh[0, i, ord(ch) - ord("A")] = 1.0
    mod(torch.zeros(1, 3, 8), oh).tokens.sum().backward()
    assert mod.readout.weight.grad.abs().sum() > 0
    assert "depth" in dict(mod.named_buffers())


def test_host_forward_backward_each_module():
    racks = [("A", "C", "E", "I", "N", "R", "T"), ("A", "A", "B", "C", "D", "E", "F")]
    enc = encode_racks(racks)
    oh = onehot_racks(enc)
    labels = torch.tensor([6, 3])
    for mod in [
        None,
        AnagramLexicon(channels=16, compiled=_lex()),
        SoftTraversalLexicon(channels=16, compiled=_lex()),
    ]:
        m = RackBestModel(channels=16, n_layers=1, n_heads=2, lexicon_module=mod)
        logits = m(oh)
        assert logits.shape == (2, 8)
        F.cross_entropy(logits, labels).backward()


def test_lane_ffn_mult_shrinks_host():
    c = 16
    assert RackBestModel(channels=c, ffn_mult=4).encoder.layers[0].linear1.out_features == 4 * c
    assert (
        RackBestModel(channels=c, ffn_mult=4, lane_ffn_mult=1)
        .encoder.layers[0]
        .linear1.out_features
        == c
    )
