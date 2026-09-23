"""Compiled-lexicon tools: frozen lexicon modules a network learns to query.

Rather than making a network memorize the lexicon, these modules hold a compiled
lexicon (the DAWG tables from :mod:`compiler`) in non-trainable buffers and let
the network query it. Only thin adapters around the tables train: what to ask,
and how to read the answer. docs/lexical_tools.md surveys the modules and the
experiments that use them.

A module is picked by name (``--lexicon-module``) from
:data:`LEXICON_MODULE_REGISTRY`, and all share one interface over a batch of M
sequences of length L (board lanes, words, or racks, depending on the host):

    forward(lane_feats:   (M, L, C),     # the host's per-cell features
            lane_letters: (M, L, 26))    # one-hot letters known at each cell
        -> LexiconOutput

The host adds ``cell_residual`` to its cell features and prepends ``tokens`` to
its transformer sequence. A module must be queried with the network's learned
features, never with the answer. Otherwise it solves the task for the network,
and the held-out-word test no longer measures whether the network learned to
use the tool.
"""

from __future__ import annotations

import argparse
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from scribblez.lexical_tool.compiler import (
    N_LETTERS,
    CompiledLexicon,
    compile_kwg,
    default_kwg_path,
)


@dataclass
class LexiconOutput:
    """What a lexicon module returns; any field may be None.

    cell_residual: ``(M, L, C)``, added to the host's per-cell features.
    tokens: ``(M, T, C)``, prepended to the host's sequence.
    cell_signals: ``(M, L, S)``, the module's raw per-cell readout before
        projection (for example accept / continuation / alive). The host does
        not consume it; tests use it to check the module's lexical answers.
    """

    cell_residual: torch.Tensor | None = None
    tokens: torch.Tensor | None = None
    cell_signals: torch.Tensor | None = None


class LexiconModule(nn.Module):
    """Base class for compiled-lexicon modules.

    Subclasses set :attr:`n_tokens`, the number of tokens they prepend, which the
    host needs to size its positional embedding, and implement forward.
    """

    n_tokens: int = 0


LexiconModuleGenerator = Callable[..., LexiconModule]
LEXICON_MODULE_REGISTRY: dict[str, LexiconModuleGenerator] = {}


def register_module(
    name: str,
) -> Callable[[LexiconModuleGenerator], LexiconModuleGenerator]:
    def deco(factory: LexiconModuleGenerator) -> LexiconModuleGenerator:
        LEXICON_MODULE_REGISTRY[name] = factory
        return factory

    return deco


def available_modules() -> list[str]:
    """All selectable ``--lexicon-module`` names, including the ``none`` no-op."""
    return ["none"] + sorted(LEXICON_MODULE_REGISTRY)


# Shown in --lexicon-module's help.
_MODULE_BLURBS = {
    "none": "no tool (baseline)",
    "soft_traversal": "soft left-to-right DAWG walk (word membership)",
    "straight_through": "DAWG walk with straight-through argmax (crisp, biased grad)",
    "kv_memory": "attention over lexicon letter-bags (anagram-lossy)",
    "anagram": "subset/anagram lookup (rack -> formable words)",
    "oracle_crosscheck": "exact board-derived legality (cheating diagnostic ceiling)",
}


@dataclass
class LexiconArgs:
    """The lexicon-tool options, with their CLI flags and the model wiring they imply.

    Trainers share this instead of each defining the flags and building the module.
    """

    module: str = "none"
    opt: list[str] = field(default_factory=list)
    mode: str = "replace"
    replace_ffn_mult: int = 1
    starve_ffn: bool = False

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser):
        """Register the tool's CLI options under a "compiled-lexicon tool" group."""
        g = parser.add_argument_group(
            "compiled-lexicon tool",
            description="A frozen, compiled lexicon (a DAWG) plugged into the model as a "
            "tool the network learns to query. The lexicon never trains; small adapters "
            "learn what to ask and how to read the answer, so the model can use the lexicon "
            "without memorizing it. See docs/lexical_tools.md.",
        )
        module_help = "; ".join(
            f"{n}: {b}" for n, b in _MODULE_BLURBS.items() if n in available_modules()
        )
        g.add_argument(
            "--lexicon-module",
            type=str,
            default="none",
            choices=available_modules(),
            help=f"Which compiled-lexicon tool to plug in (classes in "
            f"scribblez/lexical_tool/modules.py). {module_help}.",
        )
        g.add_argument(
            "--lexicon-opt",
            action="append",
            default=[],
            metavar="KEY=VALUE",
            help="Module option passed to the selected module's constructor; repeatable. "
            "For example, --lexicon-opt topk=32 sets soft_traversal's beam width. Each "
            "module's docstring lists its options.",
        )
        g.add_argument(
            "--lexicon-mode",
            type=str,
            default="replace",
            choices=["add", "replace"],
            help="How the tool relates to a transformer host; ignored by conv-trunk models. "
            "'add': the tool supplements the host's full-width FFN. 'replace': shrink the "
            "FFN (attention keeps full width) so word knowledge has to come from the tool.",
        )
        g.add_argument(
            "--lexicon-replace-ffn-mult",
            type=int,
            default=1,
            help="Transformer FFN width multiple under --lexicon-mode replace (0 is nearly "
            "attention-only). Pick it by sweeping down until a model without the tool can no "
            "longer learn the lexicon.",
        )
        g.add_argument(
            "--lexicon-starve-ffn",
            action="store_true",
            help="Apply the replace-mode FFN shrink even with --lexicon-module none. This is "
            "the control run: same starved model, no tool.",
        )

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> LexiconArgs:
        return cls(
            module=args.lexicon_module,
            opt=args.lexicon_opt,
            mode=args.lexicon_mode,
            replace_ffn_mult=args.lexicon_replace_ffn_mult,
            starve_ffn=args.lexicon_starve_ffn,
        )

    def build(self, channels: int, kwg_path: str | None = None):
        """Build the selected module, or None for ``none``. It is compiled from
        `kwg_path`, defaulting to `default_kwg_path()`."""
        return build_lexicon_module(
            self.module,
            channels=channels,
            kwg_path=kwg_path or default_kwg_path(),
            **parse_module_opts(self.opt),
        )

    def lane_ffn_mult(self, has_module: bool) -> int | None:
        """The host FFN width multiple these options imply, or None for the default."""
        return resolve_lane_ffn_mult(self.mode, has_module, self.starve_ffn, self.replace_ffn_mult)


def parse_module_opts(items: list[str]) -> dict[str, object]:
    """Parse ``KEY=VALUE`` options into kwargs, converting each value to int or
    float where it parses as one."""
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
    """The host transformer's FFN width multiple, or None to keep its default.

    The FFN is where a transformer would memorize the lexicon. "replace" mode
    shrinks it to ``replace_ffn_mult`` so word knowledge has to come from the
    tool. The shrink also applies without a tool when ``starve_ffn`` is set: that
    control run should fail to learn the lexicon, showing that the shrunk FFN
    really cannot memorize it. "add" mode keeps the full FFN.
    """
    if mode == "replace" and (has_module or starve_ffn):
        return replace_ffn_mult
    return None


def build_lexicon_module(
    name: str, *, channels: int, kwg_path: str, **opts: object
) -> LexiconModule | None:
    """Construct the named lexicon module, or None for ``"none"``.

    ``kwg_path`` must be the lexicon the training labels were computed with, or
    the tool and the targets disagree about which words exist."""
    if name == "none":
        return None
    if name not in LEXICON_MODULE_REGISTRY:
        raise KeyError(f"unknown lexicon module {name!r}; choices: {available_modules()}")
    compiled = compile_kwg(kwg_path)
    return LEXICON_MODULE_REGISTRY[name](channels=channels, compiled=compiled, **opts)


class _DawgLexicon(LexiconModule):
    """Base for modules that walk the compiled DAWG.

    Holds the DAWG tables as buffers, so they move with the model but never
    train. ``exists_tbl`` marks every (state, letter) that is a real arc: one
    that leads on, completes a word, or both.
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
    """A differentiable left-to-right DAWG walk, with letters chosen by the network.

    At each cell the walk consumes the known letter if there is one, and
    otherwise a soft letter distribution the network produces from its own
    features. A learned restart gate lets a new word begin at any cell. The walk
    state is a probability distribution over DAWG states, kept sparse as the top
    K, since a real lexicon has far too many states for a dense distribution.
    Each cell reads out:

      * accept -- probability that the soft prefix ending here is a word;
      * cont   -- which letters the lexicon allows next after the soft prefix;
      * alive  -- how much probability mass is still on valid prefixes.

    These become a per-cell residual, and their pooled summary two tokens. The
    trainable parts are the letter query, the restart gate and the readouts.

    Transitions are exact and gradients reach the letter query through the soft
    letters. The approximation is the top-K truncation, which drops unlikely
    branches; mass reaching the same state along two paths is also not merged
    before truncation. The walk is left-to-right from a single start, so it does
    not see a play that extends through existing tiles in both directions, nor
    words on the perpendicular axis; the host has to handle those. Cost is
    O(L * K * 26) per sequence, the highest of the modules.

    Options: ``topk`` (states kept, default 16).
    """

    def __init__(self, *, channels: int, compiled: CompiledLexicon, topk: int = 16):
        super().__init__(compiled)
        self.n_tokens = 2
        self.channels = channels
        self.topk = int(topk)

        self.query = nn.Linear(channels, N_LETTERS)  # which letter to ask about
        self.restart = nn.Linear(channels, 1)  # may a new word start here?
        feat_dim = 1 + N_LETTERS + 1  # accept, cont(26), alive
        self.readout = nn.Linear(feat_dim, channels)
        self.token_proj = nn.Linear(2 * feat_dim, self.n_tokens * channels)

    def _letter_query(self, lane_feats: torch.Tensor) -> torch.Tensor:
        """Letter distribution (M, L, 26) fed to the walk at empty cells."""
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
            # No renormalization: the mass left on a prefix is its probability, so
            # accept measures how likely the queried word is, and the total
            # surviving mass (alive) measures how valid the soft prefix is.
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
    """:class:`SoftTraversalLexicon` with a one-hot letter query and
    straight-through gradients.

    The forward pass commits to the argmax letter at each empty cell, so the walk
    follows a single path and its readouts are exact 0/1 answers from the real
    lexicon, with nothing lost to truncation. The backward pass sends gradient to
    the query as if the soft distribution had been used. That gradient is biased,
    since it ignores how the argmax choice changes the path. The pair gives an
    exact-but-biased versus approximate-but-unbiased comparison.
    """

    def _letter_query(self, lane_feats: torch.Tensor) -> torch.Tensor:
        q = torch.softmax(self.query(lane_feats), dim=-1)
        hard = torch.zeros_like(q).scatter_(-1, q.argmax(-1, keepdim=True), 1.0)
        return hard + (q - q.detach())  # value is one-hot; gradient is q's


@register_module("oracle_crosscheck")
class OracleCrosscheckLexicon(_DawgLexicon):
    """Diagnostic ceiling that cheats: exact lexical facts read off the board.

    It computes the letter-legality information the input encoder deliberately
    withholds, directly from the board rather than from the network's features.
    It therefore passes the held-out-word test for free and is not a result. Use
    it only to measure how well a model could do given perfect per-cell lexical
    information, and as a wiring check.

    It scans each sequence left to right, restarting at every empty cell, and at
    each cell reports, for the run of tiles immediately to its left:

      * run_word -- whether the run is a word;
      * cont     -- the letters that extend the run to a valid prefix;
      * accept   -- the letters that complete a word.

    Only left context is covered; right-hand and perpendicular words are left to
    the host. The only trained part is the readout.
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
        letters = lane_letters.argmax(-1)  # (M, L); meaningless at empty cells

        state = torch.full((m,), self.root, dtype=torch.long, device=device)
        root = torch.full((m,), self.root, dtype=torch.long, device=device)
        run_word = lane_feats.new_zeros(m)
        zero = lane_feats.new_zeros(m)

        run_steps, cont_steps, accept_steps = [], [], []
        for c in range(length):
            run_steps.append(run_word)
            cont_steps.append(self.exists_tbl[state])
            accept_steps.append(self.accept_tbl[state])
            occ_c = occupied[:, c]
            lc = letters[:, c]
            nxt = self.next_tbl[state, lc]
            acc = self.accept_tbl[state, lc]
            state = torch.where(occ_c, nxt, root)  # an empty cell ends the run
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
    """Frozen word memory with learned product-key addressing (Lample et al.).

    Each word gets one frozen value slot holding its normalized letter counts.
    The network builds a query from its pooled features and the sequence's known
    letters, and retrieves a softmax blend of the top-k slots. Addressing uses
    two learned sub-key codebooks of about sqrt(N) keys each, so retrieval costs
    about 2*sqrt(N) comparisons instead of N. The blend becomes two tokens.

    Only the values are the lexicon; the keys are learned, not derived from the
    words. The output is a sequence-level letter hint, not per-cell legality, and
    letter counts cannot tell anagrams apart. This is the least exact module.

    Options: ``knn`` (slots retrieved, default 32), ``key_dim`` (width of each
    half-query, default 32).
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

        # Slots beyond the word count stay all-zero.
        value = np.zeros((codebook * codebook, N_LETTERS), dtype=np.float32)
        for i, word in enumerate(words):
            for ch in word:
                value[i, ord(ch) - ord("A")] += 1.0
        np.divide(value, np.clip(value.sum(1, keepdims=True), 1.0, None), out=value)
        self.register_buffer("value_mem", torch.from_numpy(value))

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

        # Top k of each codebook, then the top k of their k*k sums. Slot (i1, i2)
        # is value row i1 * c + i2.
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
    """Distance from the root of every state. Well defined only for a trie, such
    as a `from_words` lexicon, where each state has a single path from the root."""
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
    """Anagram search: which word lengths can be formed from a rack.

    For inputs that are an unordered set of tiles, where there is no letter order
    for a DAWG walk to follow. The lexicon is recompiled over each word's letters
    in sorted order (``CAT`` becomes ``ACT``), so all anagrams share one path.
    The rack must be fed sorted too; any subset of it is then read in the order
    this lexicon expects.

    The walk goes over the sorted rack, and at each tile the state either skips it
    or uses it to advance. Together these cover every subset of the rack. The
    state is a top-K set of trie states. Whenever a use step completes a word,
    its mass is added to the bin for that word's length. The result, ``accept``,
    says how reachable a word of each length is; the longest nonzero bin is the
    answer. Only the readout into two tokens trains.

    Options: ``topk`` (states kept, default 128; a 7-tile rack has at most 2^7 =
    128 subsets, so the default is exact), ``max_word_len`` (drop longer words
    from the compiled lexicon; default keep all).
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
        self.n_bins = int(depth.max()) + 2  # word lengths 0 .. max, plus one spare bin
        self.register_buffer("depth", torch.from_numpy(depth))
        self.readout = nn.Linear(self.n_bins, self.n_tokens * channels)

    def forward(self, lane_feats: torch.Tensor, lane_letters: torch.Tensor) -> LexiconOutput:
        m, seqlen, _ = lane_feats.shape
        device = lane_feats.device
        letters = lane_letters.argmax(-1)  # (M, seqlen) sorted rack letters
        k = self.topk

        idx = torch.full((m, k), self.dead, dtype=torch.long, device=device)
        idx[:, 0] = self.root
        val = lane_feats.new_zeros(m, k)
        val[:, 0] = 1.0
        accept = lane_feats.new_zeros(m, self.n_bins)

        for j in range(seqlen):
            ell = letters[:, j : j + 1].expand(m, k)
            adv_next = self.next_tbl[idx, ell]  # (M, K) state after using this tile
            adv_acc = self.accept_tbl[idx, ell]  # (M, K) whether that completes a word
            done_len = (self.depth[idx] + 1).clamp(max=self.n_bins - 1)
            accept = accept.scatter_add(1, done_len, val * adv_acc)

            # Skip keeps (idx, val); use advances, dropping mass with no valid arc.
            cand_idx = torch.cat([idx, adv_next], dim=1)  # (M, 2K)
            cand_val = torch.cat([val, val * (adv_next != self.dead).float()], dim=1)
            val, pos = cand_val.topk(k, dim=1)
            idx = cand_idx.gather(1, pos)

        tokens = self.readout(accept).view(m, self.n_tokens, self.channels)
        return LexiconOutput(tokens=tokens, cell_signals=accept.unsqueeze(1))
