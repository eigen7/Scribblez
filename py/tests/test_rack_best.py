"""Tests for rack-best ordered word generation: labels, constraint, decoder."""

import torch
import torch.nn.functional as F
from scribblez.max_move_per_lane.lexicon_compiler import CompiledLexicon
from scribblez.rack_best.data import longest_words
from scribblez.rack_best.model import END, PAD, RackWordModel, encode_racks, encode_targets


def test_longest_words():
    index = {("A", "C", "T"): ["ACT", "CAT"], ("A", "T"): ["AT"], ("A", "C", "E", "R"): ["CARE"]}
    ml, words = longest_words(("A", "C", "T"), index)
    assert ml == 3 and words == ["ACT", "CAT"]  # both spellings of the longest anagram
    assert longest_words(("A", "T", "X"), index) == (2, ["AT"])
    assert longest_words(("X", "Y", "Z"), index) == (0, [])


def test_encode_targets():
    gen_in, target = encode_targets(["CAT"])
    # gen input starts with BOS then the letters; target is the letters then END.
    assert gen_in[0, :4].tolist() == [26, 2, 0, 19]  # BOS, C, A, T
    assert target[0, :4].tolist() == [2, 0, 19, END]  # C, A, T, END
    assert gen_in[0, 4:].tolist() == [PAD] * 4 and target[0, 4:].tolist() == [PAD] * 4


def _lex():
    return CompiledLexicon.from_words(["CARE", "CART", "CAT", "ACT", "RATE", "AT", "CAR"])


def test_teacher_masks_always_allow_the_target():
    # The constraint must never forbid a real word's own letters or its END.
    model = RackWordModel(_lex(), channels=16, n_layers=1, use_dawg=True)
    rack = encode_racks([("A", "C", "E", "R", "X", "Y", "Z")])
    gen_in, target = encode_targets(["CARE"])
    masks = model.teacher_masks(rack, target)  # (1, MAX_GEN, 27)
    for p in range(5):  # C, A, R, E, END
        sym = int(target[0, p])
        assert masks[0, p, sym], f"masked out target symbol {sym} at step {p}"


def test_greedy_recovers_valid_longest_after_overfit():
    lex = _lex()
    racks = [("A", "C", "E", "R", "T", "X", "Y"), ("A", "E", "R", "T", "B", "D", "F")]
    tgt_words = [
        longest_words(r, {tuple(sorted(w)): [w] for w in lex.words()})[1][0] for r in racks
    ]
    model = RackWordModel(lex, channels=32, n_layers=2, use_dawg=True)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    rack = encode_racks(racks)
    gen_in, target = encode_targets(tgt_words)
    for _ in range(150):
        logits = model(rack, gen_in, target)
        loss = F.cross_entropy(logits.reshape(-1, 27), target.reshape(-1), ignore_index=PAD)
        opt.zero_grad()
        loss.backward()
        opt.step()
    for r, want in zip(racks, tgt_words, strict=True):
        decoded = model.greedy(encode_racks([r]))[0]
        word = "".join(chr(ord("A") + x) for x in decoded)
        assert lex.contains(word)  # a real word
        assert len(word) == len(want)  # of maximal length


def test_forward_backward_with_and_without_dawg():
    lex = _lex()
    rack = encode_racks([("A", "C", "E", "R", "T", "X", "Y")])
    gen_in, target = encode_targets(["CARE"])
    for use_dawg in (True, False):
        model = RackWordModel(lex, channels=16, n_layers=1, use_dawg=use_dawg)
        logits = model(rack, gen_in, target)
        assert logits.shape == (1, gen_in.size(1), 27)
        F.cross_entropy(logits.reshape(-1, 27), target.reshape(-1), ignore_index=PAD).backward()
