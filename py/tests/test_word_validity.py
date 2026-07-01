"""Tests for the word-validity toy: encoding and the model with each tool."""

import torch
import torch.nn.functional as F
from scribblez.lexical_tool.compiler import CompiledLexicon
from scribblez.lexical_tool.modules import (
    KvMemoryLexicon,
    OracleCrosscheckLexicon,
    SoftTraversalLexicon,
)
from scribblez.word_validity.model import MAX_LEN, WordValidityModel, encode_words, onehot_batch


def _lex():
    return CompiledLexicon.from_words(["CAT", "CAR", "CARE", "AT", "DOG", "DOGS"])


def test_encode_and_onehot():
    enc, lengths = encode_words(["CAT", "AT"])
    assert enc.shape == (2, MAX_LEN)
    assert lengths.tolist() == [3, 2]
    assert enc[0, :3].tolist() == [2, 0, 19]  # C, A, T
    oh = onehot_batch(enc, lengths)
    assert oh.shape == (2, MAX_LEN, 26)
    assert oh[0].sum().item() == 3  # three letters set
    assert oh[1, 2:].sum().item() == 0  # padding zeroed for the length-2 word


def test_model_forward_backward_each_module():
    c = 16
    enc, lengths = encode_words(["CAT", "CARE", "AT", "DOG", "ZZ"])
    oh = onehot_batch(enc, lengths)
    labels = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0])
    modules = [
        None,
        SoftTraversalLexicon(channels=c, compiled=_lex()),
        OracleCrosscheckLexicon(channels=c, compiled=_lex()),
        KvMemoryLexicon(channels=c, compiled=_lex()),
    ]
    for mod in modules:
        m = WordValidityModel(channels=c, n_layers=1, n_heads=2, lexicon_module=mod)
        logit = m(oh, lengths)
        assert logit.shape == (5,)
        F.binary_cross_entropy_with_logits(logit, labels).backward()


def test_lane_ffn_mult_shrinks_host():
    c = 16
    full = WordValidityModel(channels=c, ffn_mult=4)
    shrunk = WordValidityModel(channels=c, ffn_mult=4, lane_ffn_mult=1)
    assert full.encoder.layers[0].linear1.out_features == 4 * c
    assert shrunk.encoder.layers[0].linear1.out_features == c
