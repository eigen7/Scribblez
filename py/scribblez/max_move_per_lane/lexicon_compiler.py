"""Compile a KWG lexicon file into frozen tensors for a neural lexicon module.

A KWG (Kurnia Word Graph) packs a forward DAWG and a GADDAG into one flat
little-endian ``uint32`` array (the bit layout is documented in
``engine/include/scribblez/dictionary.h``). This module reads that file directly
and extracts the DAWG half as a dense, compact transition table suitable for
differentiable traversal:

    next[state, letter]   -> the state reached by appending ``letter``, or DEAD
    accept[state, letter] -> whether appending ``letter`` completes a valid word

``state`` indexes the DAWG's arc-list heads, compacted to the nodes reachable
from the root (the GADDAG half and any unreachable nodes are dropped). A single
absorbing DEAD state (index ``num_states - 1``) collects every missing or
terminal transition, so the table is total: every ``(state, letter)`` resolves.

The extraction is invertible: :meth:`CompiledLexicon.words` walks the table back
out into the full word list ("de-assembly"). That round trip is what makes the
compiled tensors a faithful, recoverable encoding of the lexicon rather than a
lossy embedding -- it is the property the lexicon-tool experiment leans on (the
module *contains* the lexicon, so any failure to use it is the network's, not
missing information).

The compile is a vectorized numpy pass over the node array (no per-node Python
loop) and runs in well under a second for a ~1.2M-node lexicon such as NWL23.
:class:`RawKwg` keeps the original arc-list stepping for tests that want to
validate the compact table against an independent traversal of the same file.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

# KWG node bit layout (mirrors Dictionary in dictionary.h).
ARC_MASK = 0x003FFFFF
IS_END_BIT = 0x00400000
ACCEPTS_BIT = 0x00800000

N_LETTERS = 26  # A..Z. Blanks are not stored: a blank designates a played letter.

# Default lexicon, matching the C++ Lexicon defaults (see lexicon.h). The neural
# module MUST be compiled from the same .kwg the self-play labels use.
DEFAULT_LEXICA_DIR = "/workspace/mount/lexica"
DEFAULT_LEXICON_NAME = "NWL23"


def default_kwg_path(name: str = DEFAULT_LEXICON_NAME) -> str:
    """Path to a lexicon's ``.kwg`` under the standard mounted lexica directory."""
    return f"{DEFAULT_LEXICA_DIR}/{name}.kwg"


@dataclass(frozen=True)
class CompiledLexicon:
    """Compact DAWG transition tables extracted from a KWG file.

    Attributes:
        next: ``(num_states, 26)`` int32. Next state for each letter; the
            absorbing ``dead_state`` marks a missing or terminal transition.
        accept: ``(num_states, 26)`` bool. Whether that letter completes a word.
        root: The DAWG root state index.
        dead_state: The absorbing state index (``num_states - 1``).
        source_hash: sha256 of the source ``.kwg`` bytes (``"from_words"`` for
            hand-built lexicons).
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
        """Whole-word membership by walking the compact DAWG from the root."""
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
        """De-assemble: recover every word by depth-first walk of the table."""
        out: list[str] = []
        stack = [(self.root, "")]
        while stack:
            state, prefix = stack.pop()
            acc_row = self.accept[state]
            next_row = self.next[state]
            for letter in range(N_LETTERS):
                ch = prefix + chr(ord("A") + letter)
                if acc_row[letter]:
                    out.append(ch)
                nxt = int(next_row[letter])
                if nxt != self.dead_state:
                    stack.append((nxt, ch))
        return out

    @classmethod
    def from_words(cls, words: list[str]) -> CompiledLexicon:
        """Build compact tables from an explicit word list (for tests / held-out
        splits). Words are uppercase A..Z; others are skipped."""
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
    """Read a ``.kwg`` and extract its DAWG as compact transition tables."""
    raw = np.fromfile(path, dtype="<u4")
    if raw.size < 2:
        raise ValueError(f"{path}: not a valid KWG (need at least 2 nodes)")
    source_hash = hashlib.sha256(raw.tobytes()).hexdigest()

    tile, child, _is_end, accepts = _decode_nodes(raw)
    n = raw.size
    dawg_root_head = int(child[0])  # node 0's arc_index is the DAWG root list.

    # Every arc-list head is the target of some arc_index pointer (or a root).
    # head_of[i] = the most recent head index at or before i; arc lists are
    # contiguous runs, and KWG pointers only ever target a list's first arc, so
    # "last head <= i" assigns each arc to its owning list.
    is_head = np.zeros(n, dtype=bool)
    nz = child > 0
    is_head[child[nz]] = True
    is_head[dawg_root_head] = True
    head_of = np.maximum.accumulate(np.where(is_head, np.arange(n), -1))

    # Letter arcs (tile 1..26) become transitions FROM their list head ON
    # (tile - 1) TO child (0 == terminal, i.e. no onward list).
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
        nbrs = np.unique(nbrs[nbrs > 0])  # >0 is a real onward list head
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


@dataclass
class RawKwg:
    """The original KWG arc-list traversal, kept for independent validation.

    Steps the on-disk node array exactly as ``Dictionary::step_tile`` does in
    C++, so tests can confirm the compact :class:`CompiledLexicon` agrees with a
    from-scratch walk -- including via the GADDAG half, an entirely separate path
    through the same file.
    """

    nodes: np.ndarray  # int64 decoded raw array
    dawg_root: int
    gaddag_root: int

    @classmethod
    def load(cls, path: str) -> RawKwg:
        raw = np.fromfile(path, dtype="<u4").astype(np.int64)
        return cls(nodes=raw, dawg_root=int(raw[0] & ARC_MASK), gaddag_root=int(raw[1] & ARC_MASK))

    def _step(self, node: int, tile_value: int):
        """Scan a node's sibling arc list for ``tile_value`` -> (next, accepts, valid)."""
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
        """Membership via the GADDAG, using the fully-reversed encoding of the
        word -- a path through the GADDAG half that is independent of the DAWG."""
        return bool(word) and self._walk(self.gaddag_root, word[::-1])
