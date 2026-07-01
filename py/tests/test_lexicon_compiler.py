"""Tests for compiling a KWG lexicon into compact DAWG transition tables.

The key guarantee is invertibility: the compiled tensors de-assemble back to the
exact word set, and that set agrees with an independent traversal of the same
file -- including via the GADDAG half, a wholly separate path."""

import os

import pytest
from scribblez.lexical_tool.compiler import (
    CompiledLexicon,
    RawKwg,
    compile_kwg,
    default_kwg_path,
)


def test_from_words_roundtrip():
    words = ["CAT", "CAR", "CARE", "CARES", "CATS", "AT", "DOG", "DOGS"]
    lex = CompiledLexicon.from_words(words)

    assert set(lex.words()) == set(words)
    for w in words:
        assert lex.contains(w), w
    for non in ["CA", "DO", "CARED", "ZEBRA", "", "cat"]:
        assert not lex.contains(non), non


def test_from_words_skips_invalid():
    lex = CompiledLexicon.from_words(["CAT", "CA7", "do-g", "AT"])
    assert set(lex.words()) == {"CAT", "AT"}


def test_dead_state_is_total():
    lex = CompiledLexicon.from_words(["AT"])
    # Every transition resolves: missing letters route to the absorbing DEAD row.
    assert lex.next.shape[1] == 26
    assert lex.next.min() >= 0
    assert lex.next.max() <= lex.dead_state
    assert lex.dead_state == lex.num_states - 1


@pytest.mark.skipif(not os.path.exists(default_kwg_path()), reason="lexicon unavailable")
def test_compile_real_kwg_roundtrip():
    path = default_kwg_path()
    lex = compile_kwg(path)
    raw = RawKwg.load(path)

    words = lex.words()
    assert len(words) > 100_000  # NWL23 is a large lexicon
    assert all(2 <= len(w) for w in words)

    # De-assembled words verify via the compact table AND via the independent
    # GADDAG path; a handful of non-words are rejected by both.
    sample = words[:: max(1, len(words) // 500)]
    for w in sample:
        assert lex.contains(w), w
        assert raw.contains_dawg(w), w
        assert raw.contains_gaddag(w), w
    for non in ["ZZZZ", "QQQQ", "ABCDEFG", "NOTAWORDHERE"]:
        assert not lex.contains(non)
        assert not raw.contains_gaddag(non)
