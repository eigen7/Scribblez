"""Racks and their longest-word labels for the rack-best toy.

A rack is 7 tiles drawn from the standard Scrabble bag (no blanks). The label is
the length of the longest word formable from any *subset* of the rack -- which
requires anagram search, not a single word lookup. Labels come from an exact
sub-bag check against the real lexicon's per-length word-bag sets.
"""

import random
from itertools import combinations

import numpy as np

from scribblez.max_move_per_lane.lexicon_compiler import compile_kwg

RACK_SIZE = 7

# Standard English Scrabble tile distribution, blanks excluded (98 letter tiles).
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


def build_wordbags(real_kwg: str, min_len: int = 2, max_len: int = RACK_SIZE) -> dict[int, set]:
    """length -> set of sorted-letter tuples of real words of that length."""
    bags: dict[int, set] = {}
    for word in compile_kwg(real_kwg).words():
        if min_len <= len(word) <= max_len:
            bags.setdefault(len(word), set()).add(tuple(sorted(word)))
    return bags


def longest_word_length(rack: tuple, wordbags: dict[int, set], min_len: int = 2) -> int:
    """Longest word formable from a subset of `rack` (a sorted letter tuple), or 0.

    Because the rack is sorted, each `combinations` subset is already sorted and
    matches the sorted-letter keys in `wordbags` directly."""
    for k in range(len(rack), min_len - 1, -1):
        bags = wordbags.get(k)
        if bags and any(combo in bags for combo in combinations(rack, k)):
            return k
    return 0


def sample_racks(n_unique: int, rng: random.Random, rack_size: int = RACK_SIZE) -> list[tuple]:
    """`n_unique` distinct sorted racks drawn (without replacement) from the bag."""
    tiles = bag_tiles()
    seen, racks = set(), []
    while len(racks) < n_unique:
        rack = tuple(sorted(rng.sample(tiles, rack_size)))
        if rack not in seen:
            seen.add(rack)
            racks.append(rack)
    return racks


def make_dataset(real_kwg: str, n_unique: int, seed: int, rack_size: int = RACK_SIZE):
    """`(racks, labels)` -- distinct sorted racks and their longest-word lengths."""
    rng = random.Random(seed)
    wordbags = build_wordbags(real_kwg, max_len=rack_size)
    racks = sample_racks(n_unique, rng, rack_size)
    labels = np.array([longest_word_length(r, wordbags) for r in racks], dtype=np.int64)
    return racks, labels
