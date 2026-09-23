"""Compile a KWG lexicon file into dense DAWG tables for the lexical-tool modules.

A KWG (Kurnia Word Graph) packs a forward DAWG and a GADDAG into one flat
little-endian ``uint32`` array; engine/include/lexicon/dictionary.h documents the
bit layout. This module extracts the DAWG half as a total transition table that
a network can traverse with tensor indexing:

    next[state, letter]   -> the state reached by appending ``letter``, or DEAD
    accept[state, letter] -> whether appending ``letter`` completes a word

States are the DAWG's arc lists reachable from the root, renumbered densely. A
single absorbing DEAD state (the last index) receives every missing or terminal
transition, so every ``(state, letter)`` pair resolves.

The encoding is lossless: :meth:`CompiledLexicon.words` recovers the full word
list from the tables. The lexical-tool experiments depend on this. Because the
module provably contains the whole lexicon, a model that fails to use it is
failing to learn, not missing information.

:class:`RawKwg` steps the on-disk arc lists directly. It exists so tests can
check the compact tables against an independent traversal of the same file.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

# KWG node bit layout (mirrors Dictionary in engine/include/lexicon/dictionary.h).
ARC_MASK = 0x003FFFFF
IS_END_BIT = 0x00400000
ACCEPTS_BIT = 0x00800000

N_LETTERS = 26  # A..Z. A lexicon has no blank: a blank stands for a letter when played.

# Default lexicon, matching the C++ Lexicon::Params defaults (lexicon/lexicon.h). A
# lexical-tool module must be compiled from the same .kwg that produced the self-play
# labels, or it will disagree with its own training targets.
DEFAULT_LEXICA_DIR = "/workspace/mount/lexica"
DEFAULT_LEXICON_NAME = "NWL23"


def default_kwg_path(name: str = DEFAULT_LEXICON_NAME) -> str:
    """Path to a lexicon's ``.kwg`` in the mounted lexica directory."""
    return f"{DEFAULT_LEXICA_DIR}/{name}.kwg"


@dataclass(frozen=True)
class CompiledLexicon:
    """Compact DAWG transition tables extracted from a KWG file.

    Attributes:
        next: ``(num_states, 26)`` int32 next state per letter; ``dead_state``
            where the transition is missing or ends the word.
        accept: ``(num_states, 26)`` bool; whether that letter completes a word.
        root: The DAWG root state.
        dead_state: The absorbing state, always ``num_states - 1``.
        source_hash: sha256 of the source ``.kwg`` bytes, or ``"from_words"``
            for a lexicon built by :meth:`from_words`.
    """

    next: np.ndarray
    accept: np.ndarray
    root: int
    dead_state: int
    source_hash: str

    @property
    def num_states(self) -> int:
        return int(self.next.shape[0])

    def contains(self, word: str) -> bool:
        """Whole-word membership, walking the compact tables."""
        if not word:
            return False
        state = self.root
        for i, ch in enumerate(word):
            letter = ord(ch) - ord("A")
            if not 0 <= letter < N_LETTERS:
                return False
            if i == len(word) - 1:
                return bool(self.accept[state, letter])
            state = int(self.next[state, letter])
            if state == self.dead_state:
                return False
        return False

    def words(self) -> list[str]:
        """Recover every word in the lexicon, in depth-first order."""
        # The walk reads every transition of every state. Convert the arrays to
        # lists once: per-element numpy indexing is far slower on a real lexicon.
        accept = self.accept.tolist()
        transitions = self.next.tolist()
        letters = [chr(ord("A") + letter) for letter in range(N_LETTERS)]

        out: list[str] = []
        stack = [(self.root, "")]
        while stack:
            state, prefix = stack.pop()
            acc_row = accept[state]
            next_row = transitions[state]
            for letter in range(N_LETTERS):
                ch = prefix + letters[letter]
                if acc_row[letter]:
                    out.append(ch)
                nxt = next_row[letter]
                if nxt != self.dead_state:
                    stack.append((nxt, ch))
        return out

    @classmethod
    def from_words(cls, words: list[str]) -> CompiledLexicon:
        """Build tables from a word list, for tests and held-out splits. Words are
        upper-cased; any containing a non-letter are skipped. The result is an
        unminimized trie."""
        children: list[dict[int, int]] = [{}]  # node 0 == root
        is_word: list[bool] = [False]
        for raw in words:
            w = raw.upper()
            if not w or any(not ("A" <= ch <= "Z") for ch in w):
                continue
            cur = 0
            for ch in w:
                letter = ord(ch) - ord("A")
                nxt = children[cur].get(letter)
                if nxt is None:
                    nxt = len(children)
                    children.append({})
                    is_word.append(False)
                    children[cur][letter] = nxt
                cur = nxt
            is_word[cur] = True

        num = len(children)
        dead = num
        next_tbl = np.full((num + 1, N_LETTERS), dead, dtype=np.int32)
        accept_tbl = np.zeros((num + 1, N_LETTERS), dtype=bool)
        for state, edges in enumerate(children):
            for letter, child in edges.items():
                next_tbl[state, letter] = child
                accept_tbl[state, letter] = is_word[child]
        return cls(next_tbl, accept_tbl, root=0, dead_state=dead, source_hash="from_words")


def _decode_nodes(raw: np.ndarray):
    """Decode the flat KWG array into per-node (tile, child, is_end, accepts)."""
    nodes = raw.astype(np.int64)
    tile = (nodes >> 24) & 0xFF
    child = nodes & ARC_MASK
    is_end = (nodes & IS_END_BIT) != 0
    accepts = (nodes & ACCEPTS_BIT) != 0
    return tile, child, is_end, accepts


def compile_kwg(path: str) -> CompiledLexicon:
    """Read a ``.kwg`` and extract its DAWG as compact transition tables.

    Vectorized over the node array, with no per-node Python loop, so a full
    lexicon such as NWL23 compiles in well under a second."""
    raw = np.fromfile(path, dtype="<u4")
    if raw.size < 2:
        raise ValueError(f"{path}: not a valid KWG (need at least 2 nodes)")
    source_hash = hashlib.sha256(raw.tobytes()).hexdigest()

    tile, child, _is_end, accepts = _decode_nodes(raw)
    n = raw.size
    dawg_root_head = int(child[0])  # node 0's arc_index is the DAWG root list.

    # Assign each arc to the arc list it belongs to. Arc lists are contiguous
    # runs, and every pointer (and each root) targets a list's first arc. So the
    # owner of arc i is the last pointed-to index at or before i.
    is_head = np.zeros(n, dtype=bool)
    nz = child > 0
    is_head[child[nz]] = True
    is_head[dawg_root_head] = True
    head_of = np.maximum.accumulate(np.where(is_head, np.arange(n), -1))

    # Each letter arc (tile 1..26) is a transition from its list's head, on
    # letter tile - 1, to its child list. Child 0 means the arc has no children.
    letter_arc = (tile >= 1) & (tile <= 26) & (head_of >= 0)
    arc_idx = np.nonzero(letter_arc)[0]
    src = head_of[arc_idx]
    letter = (tile[arc_idx] - 1).astype(np.int64)

    next_full = np.full((n, N_LETTERS), -1, dtype=np.int64)  # -1 == letter absent
    accept_full = np.zeros((n, N_LETTERS), dtype=bool)
    next_full[src, letter] = child[arc_idx]  # child may be 0 (terminal)
    accept_full[src, letter] = accepts[arc_idx]

    # Restrict to states reachable from the DAWG root over letter transitions.
    visited = np.zeros(n, dtype=bool)
    visited[dawg_root_head] = True
    frontier = np.array([dawg_root_head], dtype=np.int64)
    while frontier.size:
        nbrs = next_full[frontier].reshape(-1)
        nbrs = np.unique(nbrs[nbrs > 0])  # drop absent (-1) and childless (0)
        nbrs = nbrs[~visited[nbrs]]
        visited[nbrs] = True
        frontier = nbrs
    states = np.nonzero(visited)[0]

    # Compact: remap reachable heads to 0..S-1; everything else -> DEAD (== S).
    num_states = states.size
    dead = num_states
    remap = np.full(n, dead, dtype=np.int64)
    remap[states] = np.arange(num_states)

    sub_next = next_full[states]  # (S, 26): onward head, 0 terminal, -1 absent
    out_next = np.full((num_states + 1, N_LETTERS), dead, dtype=np.int32)
    has_child = sub_next > 0
    out_next[:num_states][has_child] = remap[sub_next[has_child]].astype(np.int32)
    out_accept = np.zeros((num_states + 1, N_LETTERS), dtype=bool)
    out_accept[:num_states] = accept_full[states]

    return CompiledLexicon(
        next=out_next,
        accept=out_accept,
        root=int(remap[dawg_root_head]),
        dead_state=dead,
        source_hash=source_hash,
    )


def write_kwg(compiled: CompiledLexicon, path: str):
    """Write the DAWG as a ``.kwg`` file; :func:`compile_kwg` reads back the same words.

    Used to package a generated lexicon, such as a phony one, as a KWG. The file
    has no GADDAG half (node 1's root slot is left empty), and the DAWG is not
    minimized. That is enough for :func:`compile_kwg` and the lexical-tool
    modules, but not for anything that walks the GADDAG, such as the engine's
    move generator.
    """
    dead = compiled.dead_state
    nxt, acc = compiled.next, compiled.accept
    has_arc = (nxt != dead) | acc  # (state, letter) pairs that are real arcs

    # The sorted arc letters of every state that has at least one outgoing arc.
    arcs = {s: np.nonzero(has_arc[s])[0] for s in range(dead) if has_arc[s].any()}

    # Lay each arc list in a contiguous block; indices 0 and 1 are header slots
    # (node 0's arc_index is the DAWG root list; node 1's GADDAG root is unused).
    start, cur = {}, 2
    for s in sorted(arcs):
        start[s] = cur
        cur += len(arcs[s])

    out = np.zeros(cur, dtype="<u4")
    out[0] = start[compiled.root] & ARC_MASK
    for s, letters in arcs.items():
        base = start[s]
        for i, letter in enumerate(letters):
            child = int(nxt[s, letter])
            word = ((int(letter) + 1) << 24) | (start.get(child, 0) & ARC_MASK)
            if acc[s, letter]:
                word |= ACCEPTS_BIT
            if i == len(letters) - 1:
                word |= IS_END_BIT
            out[base + i] = word
    out.tofile(path)


@dataclass
class RawKwg:
    """Walks the on-disk KWG node array the way C++ ``Dictionary::step_tile`` does.

    Tests use it to check :class:`CompiledLexicon` against an independent walk,
    including one through the GADDAG half, which shares no nodes with the DAWG.
    """

    nodes: np.ndarray  # int64 decoded raw array
    dawg_root: int
    gaddag_root: int

    @classmethod
    def load(cls, path: str) -> RawKwg:
        raw = np.fromfile(path, dtype="<u4").astype(np.int64)
        return cls(nodes=raw, dawg_root=int(raw[0] & ARC_MASK), gaddag_root=int(raw[1] & ARC_MASK))

    def _step(self, node: int, tile_value: int):
        """Scan the arc list at ``node`` for ``tile_value``; return (next, accepts, valid)."""
        i = node
        while True:
            a = int(self.nodes[i])
            if (a >> 24) == tile_value:
                return a & ARC_MASK, bool(a & ACCEPTS_BIT), True
            if a & IS_END_BIT:
                return 0, False, False
            i += 1

    def _walk(self, root: int, letters: str) -> bool:
        node = root
        for k, ch in enumerate(letters):
            nxt, acc, valid = self._step(node, ord(ch) - ord("A") + 1)
            if not valid:
                return False
            if k == len(letters) - 1:
                return acc
            if nxt == 0:
                return False
            node = nxt
        return False

    def contains_dawg(self, word: str) -> bool:
        """Membership via the forward DAWG."""
        return bool(word) and self._walk(self.dawg_root, word)

    def contains_gaddag(self, word: str) -> bool:
        """Membership via the GADDAG's fully reversed encoding of the word."""
        return bool(word) and self._walk(self.gaddag_root, word[::-1])
