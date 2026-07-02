# Lexicon tools: the compiled-lexicon modules

A *lexicon tool* is a frozen, compiled form of the lexicon plugged into a network
as an input it learns to query — the recurring premise of the lexical-NN track
(see `word_validity_experiments.md` and `rack_best_experiments.md` for results).
The compiled lexicon lives in non-trainable buffers; only thin adapters around it
train, and those adapters *are* "the network learning to use the tool." The
central finding across experiments: **a tool helps only when its structure
matches the shape of the task** — so there is no single universal tool, but a
small catalog, each a different access pattern over the same lexicon.

Code: `scribblez/lexical_tool/{compiler,modules}.py`.

## The registry interface

Registry modules implement one interface, selected by name
(`build_lexicon_module` / `available_modules`; CLI via `LexiconArgs`):

```
forward(lane_feats:   (M, L, C),    # the network's per-cell features for a sequence
        lane_letters: (M, L, 26))   # the known tiles on that sequence
    -> LexiconOutput(cell_residual: (M, L, C) | None,   # added to the cells
                     tokens:        (M, T, C) | None,    # prepended to the sequence
                     cell_signals:  (M, L, S) | None)    # interpretable readout (tests/UI)
```

**Frozen vs. trainable.** The compiled lexicon tensors are registered buffers
(never trained, no weight decay); the query/read adapters are ordinary
parameters. A tool is queried with the network's *learned* representation, never
the ground-truth answer.

**Replace vs. add (`resolve_lane_ffn_mult`).** A host can either *add* a tool
alongside its own internal lexical capacity, or *replace* that capacity by
shrinking the host transformer's FFN (where an internal lexicon would be
memorized) while keeping attention intact. `--lexicon-mode replace` +
`--lexicon-replace-ffn-mult` control the shrink; `--lexicon-starve-ffn` applies
the shrink with no tool attached (the control that shows the win is the tool, not
capacity).

## Catalog

| name | structure | access pattern | fits |
| --- | --- | --- | --- |
| `none` | — | no tool (baseline) | — |
| `soft_traversal` | forward DAWG | soft left-to-right walk of a given sequence; per-cell accept + continuation | ordered validation (word-validity) |
| `straight_through` | forward DAWG | as above but hard argmax forward + straight-through gradient | ordered validation (exact forward) |
| `oracle_crosscheck` | forward DAWG | **diagnostic cheat** — exact board-derived legality, not NN-queried | ceiling / plumbing only |
| `kv_memory` | frozen word bags | product-key attention over per-word letter bags | (weak; retrieval, lossy) |
| `anagram` | sorted-anagram DAWG | soft skip/use subset walk over the sorted rack; per-length reachability | order-invariant search (rack longest-length) |

- **`soft_traversal`** — walks the forward DAWG one cell at a time using the
  network's soft letter query, tracking a sparse top-K node state; reads out
  acceptance and the legal-continuation mask per cell. The DAWG lives in frozen
  transition tables. Wins word-validity; useless on an unordered rack.
- **`straight_through`** — identical, but the per-cell letter is hardened to
  one-hot in the forward pass (exact single path, no top-K smear) with gradients
  passed through as if soft. Exact-but-biased vs. `soft_traversal`'s
  approximate-but-true gradients.
- **`oracle_crosscheck`** — reads exact, board-derived cross-check legality
  rather than being queried by the network. It defeats the tool-use premise (it
  is handed the answer), so it is only ever a ceiling / wiring check, never a
  legitimate result.
- **`kv_memory`** — compiles the lexicon into a frozen key-value memory of
  per-word letter bags and retrieves via product-key attention. Order-invariant
  but the bag is anagram-invariant and lossy, so it sits near the no-tool
  baseline in practice.
- **`anagram`** — compiles the lexicon over each word's *sorted* letters (`CAT` →
  `ACT`), so a multiset has one canonical key and anagrams collapse. A soft
  skip/use walk over the sorted rack sums over all subsets, and a word completing
  on a "use" transition is binned by node depth (= word length, unique because
  the sorted lexicon is a trie), giving a per-length reachability vector. Wins the
  order-invariant rack task; requires the rack fed sorted.

## The forward-DAWG constraint (generation)

Ordered *generation* (rack-best score+word) does not use the registry interface —
it needs a per-step constraint inside an autoregressive decode, not a one-shot
`LexiconOutput`. Instead the forward DAWG is used directly as a **constraint
oracle**: at each decode step it supplies the valid-continuation mask (letters
that keep a real word prefix) and whether the prefix is a complete word, while
the rack supplies the availability mask. With hard masks every complete decode is
a valid rack-word, so the network only learns to reach maximal length. See
`scribblez/rack_best/model.py`.

## The theme

`soft_traversal` (ordered) and `anagram` (order-invariant) trade places across
word-validity and rack-length: each wins the task its structure fits and does
nothing on the other. So engineering a lexicon tool for a new task is choosing
the access pattern (and the compiled structure — forward DAWG, sorted-anagram
DAWG, and eventually a GADDAG for board-anchored access) that its shape calls
for.
