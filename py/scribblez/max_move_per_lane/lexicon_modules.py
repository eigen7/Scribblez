"""Frozen, compiled-lexicon modules the network can learn to use as a tool.

The premise: instead of forcing the network to grow an internal copy of the
lexicon, compile the lexicon into a module whose lexical weights are *frozen*
and *plug it in* as an input the network learns to query -- like handing a USB
drive to a computer. The compiled lexicon (a DAWG transition table, see
:mod:`lexicon_compiler`) is held in non-trainable buffers; the thin query/read
adapters around it are ordinary parameters. Those adapters *are* "the network
learning to operate the tool": gradients teach it what to ask and how to read
the answer, while the lexicon itself never changes.

A module is selected by name (``--lexicon-module``) from
:data:`LEXICON_MODULE_REGISTRY`. Each obeys one interface:

    forward(lane_feats:   (M, 15, C),    # the network's per-cell lane features
            lane_letters: (M, 15, 26))   # the known board tiles on the lane
        -> LexiconOutput(cell_residual: (M, 15, C) | None,
                         tokens:        (M, T, C)  | None)

``cell_residual`` is added to the 15 lane cells; ``tokens`` are prepended to the
lane's transformer sequence (alongside the rack tokens). Crucially, a module is
queried with the network's *learned* representation, never with the ground-truth
answer -- otherwise it would solve the task for the wrong reason and make the
held-out-word generalization test meaningless.

Modules run on every lane, rows and columns, with shared weights (transpose
sharing), matching the lane transformer: a word is a word along either axis.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from scribblez.max_move_per_lane.lexicon_compiler import (
    N_LETTERS,
    CompiledLexicon,
    compile_kwg,
)


@dataclass
class LexiconOutput:
    """What a lexicon module contributes to a lane's transformer sequence.

    cell_residual: ``(M, 15, C)`` added to the per-cell lane features, or None.
    tokens: ``(M, T, C)`` extra tokens prepended to the lane sequence, or None.
    cell_signals: ``(M, 15, S)`` the module's raw, interpretable per-cell lexical
        readout (e.g. accept / continuation / alive), or None. The network does
        not consume this -- it is exposed for tests and dashboard visualization.
    """

    cell_residual: torch.Tensor | None = None
    tokens: torch.Tensor | None = None
    cell_signals: torch.Tensor | None = None


class LexiconModule(nn.Module):
    """Base class for compiled-lexicon modules.

    Subclasses set :attr:`n_tokens` (how many tokens they prepend per lane, so
    the lane transformer can size its positional table) and implement forward.
    """

    n_tokens: int = 0


LEXICON_MODULE_REGISTRY: dict[str, Callable[..., LexiconModule]] = {}


def register_module(
    name: str,
) -> Callable[[Callable[..., LexiconModule]], Callable[..., LexiconModule]]:
    def deco(factory: Callable[..., LexiconModule]) -> Callable[..., LexiconModule]:
        LEXICON_MODULE_REGISTRY[name] = factory
        return factory

    return deco


def available_modules() -> list[str]:
    """All selectable ``--lexicon-module`` names, including the ``none`` no-op."""
    return ["none"] + sorted(LEXICON_MODULE_REGISTRY)


def parse_module_opts(items: list[str]) -> dict[str, object]:
    """Parse repeated ``KEY=VALUE`` CLI options into a kwargs dict.

    Values are coerced to int, then float, else left as a string."""
    opts: dict[str, object] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"lexicon option must be KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        opts[key.strip()] = _coerce(value.strip())
    return opts


def _coerce(value: str) -> object:
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    return value


def resolve_lane_ffn_mult(
    mode: str, has_module: bool, starve_ffn: bool, replace_ffn_mult: int
) -> int | None:
    """The lane transformer's FFN-width override for the lexicon experiment, or
    None to keep the default width.

    In "replace" mode the lane FFN -- where an internal lexicon would be
    memorized -- is shrunk to ``replace_ffn_mult`` so word knowledge must come
    from the tool. That fires when a tool is plugged in, OR when ``starve_ffn`` is
    set: the starved-no-tool control -- a shrunk backbone with no tool, which
    should then FAIL to learn the lexicon, proving the FFN is genuinely starved.
    "add" mode never overrides (the tool augments a full internal store).
    """
    if mode == "replace" and (has_module or starve_ffn):
        return replace_ffn_mult
    return None


def build_lexicon_module(
    name: str, *, channels: int, kwg_path: str, **opts: object
) -> LexiconModule | None:
    """Construct the named lexicon module, or None for ``"none"``.

    The lexicon is compiled from ``kwg_path``, which MUST be the same lexicon the
    self-play labels are computed from, or the frozen module and the training
    targets describe different dictionaries."""
    if name == "none":
        return None
    if name not in LEXICON_MODULE_REGISTRY:
        raise KeyError(f"unknown lexicon module {name!r}; choices: {available_modules()}")
    compiled = compile_kwg(kwg_path)
    return LEXICON_MODULE_REGISTRY[name](channels=channels, compiled=compiled, **opts)


class _DawgLexicon(LexiconModule):
    """Base for modules that traverse the compiled DAWG.

    Holds the frozen transition / acceptance / letter-legality tables as buffers
    (the compiled lexicon, moved with the model but never trained). Subclasses
    add the trainable adapters and the forward pass.
    """

    def __init__(self, compiled: CompiledLexicon):
        super().__init__()
        self.root = compiled.root
        self.dead = compiled.dead_state
        exists = (compiled.next != compiled.dead_state) | compiled.accept
        self.register_buffer("next_tbl", torch.from_numpy(compiled.next.astype(np.int64)))
        self.register_buffer("accept_tbl", torch.from_numpy(compiled.accept.astype(np.float32)))
        self.register_buffer("exists_tbl", torch.from_numpy(exists.astype(np.float32)))


@register_module("soft_traversal")
class SoftTraversalLexicon(_DawgLexicon):
    """A differentiable forward-DAWG walk along the lane, queried by the network.

    Design
    ------
    The frozen DAWG is a transition table; this module walks it left-to-right
    along a lane while the network supplies the letters. At each cell the letter
    fed in is the known board tile if the cell is occupied, else a *soft* query
    distribution the network emits from its own lane features. A learned restart
    gate lets a fresh word begin at any cell (words are not anchored at cell 0),
    so one left-to-right pass covers plays starting anywhere.

    The traversal state is a distribution over DAWG nodes, kept tractable as a
    sparse top-K set (the lexicon has tens of thousands of nodes; a dense state
    is infeasible). Each step reads out, differentiably:
      * accept   -- probability the soft prefix ending here is a complete word;
      * cont     -- per-letter legality: which letters the lexicon permits next
                    given the soft prefix (a soft, network-derived cross-check);
      * alive    -- how much of the soft letter mass was a legal continuation.
    These per-cell signals become a cell residual; their pooled summary becomes
    two lane tokens.

    Frozen vs. trainable
    --------------------
    The transition/accept/exists tables are buffers (frozen lexicon). Trainable:
    the query head (what letter to ask about), the restart gate, and the readout
    projections (how to turn lexical answers into features). Learning these is
    the network learning to use the tool.

    Tradeoffs / suitability
    -----------------------
    Faithful to "a differentiable automaton": exact transitions, true gradients
    through the soft letters, de-assemblable lexicon. Approximations: top-K
    truncation drops low-probability branches, and equal target states from
    different (state, letter) pairs are not merged before the top-K (a mild
    over-spread). Cost is O(lane_len * K * 26) per lane and is the heaviest of
    the planned modules.

    Toy vs. general: the left-to-right, single-anchor walk is tailored to the
    per-lane toy task -- it does not model a play threading bidirectionally
    through existing tiles, nor cross-words on the perpendicular axis (those are
    left to the shared transformer). The *soft* state is, however, exactly the
    primitive a future win-probability model wants (a soft prefix = genuine
    uncertainty over the opponent's tiles / the bag), so the mechanism is meant
    to transfer even though this particular anchoring is toy-specific.

    Options: ``topk`` (tracked states, default 16).
    """

    def __init__(self, *, channels: int, compiled: CompiledLexicon, topk: int = 16):
        super().__init__(compiled)
        self.n_tokens = 2
        self.channels = channels
        self.topk = int(topk)

        # Trainable adapters (the network's "hands" on the tool).
        self.query = nn.Linear(channels, N_LETTERS)  # which letter to ask about
        self.restart = nn.Linear(channels, 1)  # may a new word start here?
        feat_dim = 1 + N_LETTERS + 1  # accept, cont(26), alive
        self.readout = nn.Linear(feat_dim, channels)
        self.token_proj = nn.Linear(2 * feat_dim, self.n_tokens * channels)

    def _letter_query(self, lane_feats: torch.Tensor) -> torch.Tensor:
        """Per-cell letter distribution used to probe empty cells -- ``(M, L, 26)``.
        Soft (a full softmax): the traversal state spreads over every word the
        query is compatible with, giving true (top-K truncated) gradients."""
        return torch.softmax(self.query(lane_feats), dim=-1)

    def forward(self, lane_feats: torch.Tensor, lane_letters: torch.Tensor) -> LexiconOutput:
        m, length, _ = lane_feats.shape
        k = self.topk
        device = lane_feats.device

        # Letter fed at each cell: board tile if occupied, else the queried letter.
        q = self._letter_query(lane_feats)  # (M, L, 26)
        occ = lane_letters.sum(-1, keepdim=True).clamp(max=1.0)  # (M, L, 1)
        letters_in = occ * lane_letters + (1.0 - occ) * q  # (M, L, 26)
        restart = torch.sigmoid(self.restart(lane_feats)).squeeze(-1)  # (M, L)

        # Sparse top-K state, initialized at the root with all the mass.
        val = lane_feats.new_zeros(m, k)
        val[:, 0] = 1.0
        idx = torch.full((m, k), self.dead, dtype=torch.long, device=device)
        idx[:, 0] = self.root
        root_col = torch.full((m, 1), self.root, dtype=torch.long, device=device)

        accept_steps, cont_steps, alive_steps = [], [], []
        for j in range(length):
            gate = restart[:, j : j + 1]  # (M, 1)
            cand_val = torch.cat([(1.0 - gate) * val, gate], dim=1)  # (M, K+1)
            cand_idx = torch.cat([idx, root_col], dim=1)  # (M, K+1)

            nxt = self.next_tbl[cand_idx]  # (M, K+1, 26)
            acc = self.accept_tbl[cand_idx]  # (M, K+1, 26)
            exists = self.exists_tbl[cand_idx]  # (M, K+1, 26)
            flow = cand_val.unsqueeze(-1) * letters_in[:, j].unsqueeze(1)  # (M, K+1, 26)

            cont_steps.append((cand_val.unsqueeze(-1) * exists).sum(1))  # (M, 26)
            accept_steps.append((flow * acc).sum(dim=(1, 2)))  # (M,)

            flat_idx = nxt.reshape(m, -1)
            flat_flow = flow.reshape(m, -1).masked_fill(flat_idx == self.dead, 0.0)
            top_flow, top_pos = flat_flow.topk(k, dim=1)
            idx = flat_idx.gather(1, top_pos)
            # Keep raw probability mass (no renormalization): the mass on a prefix
            # IS its soft probability, so accept reflects how likely the queried
            # word is and gradients reach the query head. The surviving mass
            # ("alive") doubles as a signal of how lexically legal the prefix is.
            val = top_flow
            alive_steps.append(val.sum(1))

        accept = torch.stack(accept_steps, dim=1).unsqueeze(-1)  # (M, L, 1)
        cont = torch.stack(cont_steps, dim=1)  # (M, L, 26)
        alive = torch.stack(alive_steps, dim=1).unsqueeze(-1)  # (M, L, 1)
        feat = torch.cat([accept, cont, alive], dim=-1)  # (M, L, feat_dim)

        cell_residual = self.readout(feat)  # (M, L, C)
        pooled = torch.cat([feat.mean(1), feat.amax(1)], dim=-1)  # (M, 2*feat_dim)
        tokens = self.token_proj(pooled).view(m, self.n_tokens, self.channels)
        return LexiconOutput(cell_residual=cell_residual, tokens=tokens, cell_signals=feat)


@register_module("straight_through")
class StraightThroughLexicon(SoftTraversalLexicon):
    """A DAWG walk with an EXACT (hard) forward pass and straight-through gradients.

    Identical to :class:`SoftTraversalLexicon` except the letter probed at each
    empty cell is the network's query **hardened to one-hot** (argmax) via a
    straight-through estimator: the forward pass commits to a single letter, so
    the traversal follows one exact path -- acceptance and continuation are crisp
    0/1 facts with no top-K truncation -- while the backward pass flows gradient
    to the query head as if the soft distribution had been used.

    Tradeoffs / suitability
    -----------------------
    Exact and cheap forward (a single path; the soft module's truncation can
    never drop the true branch), and the readout reflects the real lexicon, not a
    smeared approximation. The cost is a **biased** gradient: the straight-through
    estimator ignores how committing to argmax changes the path, so the query
    head is trained on an approximate signal. A good toy baseline and a clean
    contrast to ``soft_traversal`` (exact-but-biased vs. approximate-but-true
    gradients); as a transfer primitive it is weaker, since the hard commitment
    discards exactly the distributional softness a win-probability model wants.
    """

    def _letter_query(self, lane_feats: torch.Tensor) -> torch.Tensor:
        q = torch.softmax(self.query(lane_feats), dim=-1)
        hard = torch.zeros_like(q).scatter_(-1, q.argmax(-1, keepdim=True), 1.0)
        return hard + (q - q.detach())  # forward: one-hot; backward: soft (STE)


@register_module("oracle_crosscheck")
class OracleCrosscheckLexicon(_DawgLexicon):
    """DIAGNOSTIC ONLY -- a cheating upper bound, NOT a legitimate result.

    Feeds the network exact, board-derived lexical legality -- the very
    cross-check knowledge the input encoder deliberately withholds (which is the
    whole point of the experiment). It is NOT queried by the network's learned
    representation: it reads the answer straight off the board, so it defeats the
    experiment and would "pass" the held-out-word test for the wrong reason
    (board-derived legality covers held-out words for free). Its only honest use
    is as a ceiling -- how high lane accuracy can go given perfect per-cell
    lexical info -- and as a wiring check. Never compare it to the real
    generators as if it were one.

    Mechanism: a single left-to-right DAWG scan with a hard reset at each empty
    cell (a broken contiguous run). At every cell it reads, exactly, from the
    state of the contiguous run on its left:
      * run_word -- whether that left run is itself a complete word;
      * cont     -- the letters that legally extend the run (prefix-valid);
      * accept   -- the letters that, placed here, complete a word.
    This is the LEFT-context cross-check (a simplification of the full
    bidirectional cross-check; the perpendicular and right-context legality are
    left to the shared transformer). The scan has no learnable part -- only the
    readout that projects the exact signal into the trunk's feature space is
    trained, i.e. the network learns to USE a perfect (cheating) oracle.
    """

    def __init__(self, *, channels: int, compiled: CompiledLexicon):
        super().__init__(compiled)
        self.n_tokens = 2
        self.channels = channels
        feat_dim = 1 + N_LETTERS + N_LETTERS  # run_word, cont(26), accept(26)
        self.readout = nn.Linear(feat_dim, channels)
        self.token_proj = nn.Linear(2 * feat_dim, self.n_tokens * channels)

    def forward(self, lane_feats: torch.Tensor, lane_letters: torch.Tensor) -> LexiconOutput:
        m, length, _ = lane_feats.shape
        device = lane_feats.device
        occupied = lane_letters.sum(-1) > 0  # (M, L)
        letters = lane_letters.argmax(-1)  # (M, L); 0 where empty (unused there)

        state = torch.full((m,), self.root, dtype=torch.long, device=device)
        root = torch.full((m,), self.root, dtype=torch.long, device=device)
        run_word = lane_feats.new_zeros(m)
        zero = lane_feats.new_zeros(m)

        run_steps, cont_steps, accept_steps = [], [], []
        for c in range(length):
            run_steps.append(run_word)  # left run (ending before c) is a word
            cont_steps.append(self.exists_tbl[state])  # prefix-legal letters here
            accept_steps.append(self.accept_tbl[state])  # word-completing letters here
            occ_c = occupied[:, c]
            lc = letters[:, c]
            nxt = self.next_tbl[state, lc]
            acc = self.accept_tbl[state, lc]
            state = torch.where(occ_c, nxt, root)  # continue the run, or reset at a gap
            run_word = torch.where(occ_c, acc, zero)

        run = torch.stack(run_steps, dim=1).unsqueeze(-1)  # (M, L, 1)
        cont = torch.stack(cont_steps, dim=1)  # (M, L, 26)
        accept = torch.stack(accept_steps, dim=1)  # (M, L, 26)
        feat = torch.cat([run, cont, accept], dim=-1)  # (M, L, feat_dim)

        cell_residual = self.readout(feat)
        pooled = torch.cat([feat.mean(1), feat.amax(1)], dim=-1)
        tokens = self.token_proj(pooled).view(m, self.n_tokens, self.channels)
        return LexiconOutput(cell_residual=cell_residual, tokens=tokens, cell_signals=feat)


@register_module("kv_memory")
class KvMemoryLexicon(LexiconModule):
    """Frozen key-value lexicon memory with learned product-key addressing.

    Compiles the lexicon into a frozen value memory -- one slot per word, holding
    that word's letter bag (its letter-count distribution). The network forms a
    query from its pooled lane features plus the lane's existing tiles and
    retrieves a blend of the compatible words' letters via product-key attention
    (Lample et al.): two learned sub-key codebooks of size ~sqrt(N) make top-k
    retrieval over the ~170k-word memory cheap (2*sqrt(N) comparisons, not N).
    The retrieved letter hint becomes lane tokens.

    Frozen vs. trainable: the per-word value memory is a buffer (the compiled,
    de-assemblable lexicon). Trainable: the query projection, the two sub-key
    codebooks (HOW to address the memory), and the readout -- the network
    learning to look words up.

    Tradeoffs / suitability: retrieval, NOT an automaton. Clean gradients
    (softmax over the retrieved slots) and tractable, but it returns a
    lane-level letter *hint*, not exact per-cell legality, and the addressing
    keys are learned rather than a structural encoding of the words, so the
    "compiled lexicon" lives only in the frozen values. The coarsest,
    fuzziest generator -- a differentiable lexicon lookup, weakest on exactness.
    Options: ``knn`` (retrieved slots, default 32), ``key_dim`` (per-codebook
    query width, default 32).
    """

    def __init__(
        self, *, channels: int, compiled: CompiledLexicon, knn: int = 32, key_dim: int = 32
    ):
        super().__init__()
        self.n_tokens = 2
        self.channels = channels
        self.key_dim = int(key_dim)

        words = compiled.words()
        codebook = int(np.ceil(np.sqrt(max(len(words), 1))))
        self.codebook = codebook
        self.knn = min(int(knn), codebook)

        # One frozen value slot per word: its letter-count distribution.
        value = np.zeros((codebook * codebook, N_LETTERS), dtype=np.float32)
        for i, word in enumerate(words):
            for ch in word:
                value[i, ord(ch) - ord("A")] += 1.0
        np.divide(value, np.clip(value.sum(1, keepdims=True), 1.0, None), out=value)
        self.register_buffer("value_mem", torch.from_numpy(value))

        # Learned addressing: query from lane context, two sub-key codebooks.
        self.query = nn.Linear(channels + N_LETTERS, 2 * self.key_dim)
        self.subkeys1 = nn.Parameter(torch.randn(codebook, self.key_dim) * 0.02)
        self.subkeys2 = nn.Parameter(torch.randn(codebook, self.key_dim) * 0.02)
        self.readout = nn.Linear(N_LETTERS, self.n_tokens * channels)

    def forward(self, lane_feats: torch.Tensor, lane_letters: torch.Tensor) -> LexiconOutput:
        m = lane_feats.size(0)
        c, k = self.codebook, self.knn

        ctx = torch.cat([lane_feats.mean(1), lane_letters.sum(1)], dim=-1)  # (M, C + 26)
        q1, q2 = self.query(ctx).split(self.key_dim, dim=-1)
        s1, s2 = q1 @ self.subkeys1.t(), q2 @ self.subkeys2.t()  # (M, c) each

        # Product-key top-k: best k of each codebook, then the best k of the k*k
        # combined slots (slot index == i1 * c + i2).
        v1, i1 = s1.topk(k, dim=1)
        v2, i2 = s2.topk(k, dim=1)
        cand_score = (v1.unsqueeze(2) + v2.unsqueeze(1)).reshape(m, -1)  # (M, k*k)
        cand_slot = (i1.unsqueeze(2) * c + i2.unsqueeze(1)).reshape(m, -1)
        top_score, top_pos = cand_score.topk(k, dim=1)
        slots = cand_slot.gather(1, top_pos)  # (M, k)

        weights = torch.softmax(top_score, dim=1).unsqueeze(-1)  # (M, k, 1)
        retrieved = (weights * self.value_mem[slots]).sum(1)  # (M, 26)
        tokens = self.readout(retrieved).view(m, self.n_tokens, self.channels)
        return LexiconOutput(tokens=tokens)


def _node_depths(compiled: CompiledLexicon) -> np.ndarray:
    """Depth (path length from the root) of every state in a trie-shaped lexicon.
    A `from_words` lexicon is a trie, so each state has a unique depth."""
    depth = np.zeros(compiled.num_states, dtype=np.int64)
    seen = np.zeros(compiled.num_states, dtype=bool)
    seen[compiled.root] = True
    queue = deque([compiled.root])
    nxt = compiled.next
    while queue:
        s = queue.popleft()
        for letter in range(N_LETTERS):
            t = int(nxt[s, letter])
            if t != compiled.dead_state and not seen[t]:
                seen[t] = True
                depth[t] = depth[s] + 1
                queue.append(t)
    return depth


@register_module("anagram")
class AnagramLexicon(_DawgLexicon):
    """A differentiable anagram-search tool: longest word formable from a rack.

    Built for tasks where the input is an unordered multiset of tiles (a rack)
    rather than an ordered word -- the walk modules don't apply because there is
    no canonical order to walk. Two ideas make it work:

    Canonicalize by sorting. The lexicon is compiled over each word's letters
    *sorted* (``CAT`` -> ``ACT``), so a multiset has a single canonical key and
    anagrams collapse. The rack is fed sorted, so a used subset's letters are read
    in the same sorted order the lexicon expects.

    Search subsets by a soft skip/use walk. Over the sorted rack, at each position
    the state can either skip the tile (carry mass) or use it (advance the
    sorted-anagram DAWG by that letter). Summing both over all positions explores
    every subset; the state is a sparse top-K node distribution. A word completing
    on a "use" transition is binned by the node's depth, giving ``accept`` -- a
    per-length vector of how reachable a word of each length is from this rack.
    The longest length with mass is the answer; ``accept`` becomes lane tokens.

    Frozen: the sorted-anagram DAWG and node depths (buffers). Trainable: the
    readout. As with the walk tools, the rack is fully given, so the module
    surfaces achievability and the network learns to read it. Bingo (use all
    tiles) is just the top length bin, so this also covers the all-tiles case.

    Requires the rack fed in sorted order. Options: ``topk`` (tracked nodes,
    default 128 -- enough to be exact for a 7-tile rack), ``max_word_len`` (cap
    the compiled words; default all).
    """

    def __init__(
        self,
        *,
        channels: int,
        compiled: CompiledLexicon,
        topk: int = 128,
        max_word_len: int | None = None,
    ):
        words = compiled.words()
        if max_word_len is not None:
            words = [w for w in words if len(w) <= max_word_len]
        sorted_lexicon = CompiledLexicon.from_words(["".join(sorted(w)) for w in words])
        super().__init__(sorted_lexicon)
        self.n_tokens = 2
        self.channels = channels
        self.topk = int(topk)

        depth = _node_depths(sorted_lexicon)
        self.n_bins = int(depth.max()) + 2  # word lengths 0 .. max, with headroom
        self.register_buffer("depth", torch.from_numpy(depth))
        self.readout = nn.Linear(self.n_bins, self.n_tokens * channels)

    def forward(self, lane_feats: torch.Tensor, lane_letters: torch.Tensor) -> LexiconOutput:
        m, seqlen, _ = lane_feats.shape
        device = lane_feats.device
        letters = lane_letters.argmax(-1)  # (M, seqlen) -- the sorted rack letters
        k = self.topk

        idx = torch.full((m, k), self.dead, dtype=torch.long, device=device)
        idx[:, 0] = self.root
        val = lane_feats.new_zeros(m, k)
        val[:, 0] = 1.0
        accept = lane_feats.new_zeros(m, self.n_bins)

        for j in range(seqlen):
            ell = letters[:, j : j + 1].expand(m, k)
            adv_next = self.next_tbl[idx, ell]  # (M, K) state after using this tile
            adv_acc = self.accept_tbl[idx, ell]  # (M, K) does that complete a word?
            done_len = (self.depth[idx] + 1).clamp(max=self.n_bins - 1)  # word length if so
            accept = accept.scatter_add(1, done_len, val * adv_acc)

            # New state: skip (keep idx, val) plus use (advance, valid moves only).
            cand_idx = torch.cat([idx, adv_next], dim=1)  # (M, 2K)
            cand_val = torch.cat([val, val * (adv_next != self.dead).float()], dim=1)
            val, pos = cand_val.topk(k, dim=1)
            idx = cand_idx.gather(1, pos)

        tokens = self.readout(accept).view(m, self.n_tokens, self.channels)
        return LexiconOutput(tokens=tokens, cell_signals=accept.unsqueeze(1))
