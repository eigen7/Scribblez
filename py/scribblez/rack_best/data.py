"""Dataset for the rack-best task: random racks labelled with a longest word.

Racks are 7 tiles drawn without blanks from a standard bag. When a rack has
several longest words, the training target is the alphabetically first; the
evaluation in scripts/rack_best/train.py accepts any of them. Racks that form
no word are dropped.
"""

import random
from itertools import combinations

import numpy as np

from scribblez.lexical_tool.compiler import compile_kwg

RACK_SIZE = 7

# Standard English tile distribution without the two blanks: 98 tiles.
STANDARD_BAG = {
    "A": 9,
    "B": 2,
    "C": 2,
    "D": 4,
    "E": 12,
    "F": 2,
    "G": 3,
    "H": 2,
    "I": 9,
    "J": 1,
    "K": 1,
    "L": 4,
    "M": 2,
    "N": 6,
    "O": 8,
    "P": 2,
    "Q": 1,
    "R": 6,
    "S": 4,
    "T": 6,
    "U": 4,
    "V": 2,
    "W": 2,
    "X": 1,
    "Y": 2,
    "Z": 1,
}


def bag_tiles(bag: dict = STANDARD_BAG) -> list[str]:
    return [letter for letter, count in bag.items() for _ in range(count)]


def build_anagram_index(real_kwg: str, min_len: int = 2, max_len: int = RACK_SIZE) -> dict:
    """Map each sorted-letter tuple to the words that are anagrams of it."""
    index: dict[tuple, list[str]] = {}
    for word in compile_kwg(real_kwg).words():
        if min_len <= len(word) <= max_len:
            index.setdefault(tuple(sorted(word)), []).append(word)
    return index


def longest_words(rack: tuple, index: dict, min_len: int = 2) -> tuple[int, list[str]]:
    """(longest length, all words of that length, sorted) formable from a sorted
    rack, or (0, []) if none."""
    for k in range(len(rack), min_len - 1, -1):
        words: set[str] = set()
        for combo in set(combinations(rack, k)):  # a sorted rack yields sorted combos
            words.update(index.get(combo, ()))
        if words:
            return k, sorted(words)
    return 0, []


def sample_racks(n_unique: int, rng: random.Random, rack_size: int = RACK_SIZE) -> list[tuple]:
    tiles = bag_tiles()
    seen, racks = set(), []
    while len(racks) < n_unique:
        rack = tuple(sorted(rng.sample(tiles, rack_size)))
        if rack not in seen:
            seen.add(rack)
            racks.append(rack)
    return racks


def make_dataset(real_kwg: str, n_unique: int, seed: int, rack_size: int = RACK_SIZE):
    """(racks, target words, longest lengths) for `n_unique` distinct racks,
    minus those that form no word."""
    rng = random.Random(seed)
    index = build_anagram_index(real_kwg, max_len=rack_size)
    racks, targets, max_lens = [], [], []
    for rack in sample_racks(n_unique, rng, rack_size):
        ml, words = longest_words(rack, index)
        if ml == 0:
            continue
        racks.append(rack)
        targets.append(words[0])
        max_lens.append(ml)
    return racks, targets, np.array(max_lens, dtype=np.int64)
