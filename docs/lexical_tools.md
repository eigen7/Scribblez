# Lexicon tools: the compiled-lexicon modules

A *lexicon tool* is a frozen, compiled form of the lexicon plugged into a
network as an input the network learns to query. It is the common premise of
the lexical-NN experiments
([word_validity_experiments.md](word_validity_experiments.md),
[rack_best_experiments.md](rack_best_experiments.md), and the
`lexicon_module` param of the max_move_per_lane workload,
[lexical_nn.md](lexical_nn.md)). The compiled lexicon lives in non-trainable
buffers. Only thin adapters around it train, and those adapters are the
network learning to use the tool.

The central finding across the experiments: **a tool helps only when its
structure matches the shape of the task.** So there is no universal tool, only
a small catalog, each entry a different access pattern over the same lexicon.

Code: `py/scribblez/lexical_tool/{compiler,modules}.py`.

## The registry interface

Registry modules implement one interface and are selected by name
(`build_lexicon_module`, `available_modules`; CLI flags via `LexiconArgs`):

```
forward(lane_feats:   (M, L, C),    # the network's per-cell features for a sequence
        lane_letters: (M, L, 26))   # the known tiles on that sequence
    -> LexiconOutput(cell_residual: (M, L, C) | None,   # added to the cells
                     tokens:        (M, T, C) | None,    # prepended to the sequence
                     cell_signals:  (M, L, S) | None)    # interpretable readout (tests/UI)
```

**Frozen vs. trainable.** The compiled lexicon tensors are registered buffers
(never trained, no weight decay); the query and read adapters are ordinary
parameters. A tool is queried with the network's *learned* representation,
never with the ground-truth answer.

**Replace vs. add** (`resolve_lane_ffn_mult`). A host can *add* a tool
alongside its own internal lexical capacity, or *replace* that capacity by
shrinking the host transformer's FFN (where an internal lexicon would be
memorized) while keeping attention intact. `--lexicon-mode replace` and
`--lexicon-replace-ffn-mult` control the shrink. `--lexicon-starve-ffn`
applies the same shrink with no tool attached: it is the control that shows a
win comes from the tool, not from capacity.

## Catalog

| name | structure | access pattern | fits |
| --- | --- | --- | --- |
| `none` | — | no tool (baseline) | — |
| `soft_traversal` | forward DAWG | soft left-to-right walk of a given sequence; per-cell accept + continuation | ordered validation (word-validity) |
| `straight_through` | forward DAWG | as above but hard argmax forward + straight-through gradient | ordered validation (exact forward) |
| `oracle_crosscheck` | forward DAWG | **diagnostic cheat** — exact board-derived legality, not NN-queried | ceiling / plumbing only |
| `kv_memory` | frozen word bags | product-key attention over per-word letter bags | (weak; retrieval, lossy) |
| `anagram` | sorted-anagram DAWG | soft skip/use subset walk over the sorted rack; per-length reachability | order-invariant search (rack longest-length) |

Notes:

- `straight_through` has an exact forward pass but biased gradients;
  `soft_traversal` is approximate but has true gradients.
- `oracle_crosscheck` is handed the answer rather than queried, so it is only
  ever a ceiling or wiring check, never a legitimate result.
- `kv_memory`'s letter bags are anagram-invariant and lossy, so it sits near
  the no-tool baseline.
- `anagram` collapses anagrams onto one canonical sorted key, so the rack must
  be fed sorted.

## The forward-DAWG constraint (generation)

Ordered *generation* (rack-best's score-plus-word version) does not use the
registry interface: it needs a per-step constraint inside an autoregressive
decode, not a one-shot `LexiconOutput`. The forward DAWG is used directly as a
**constraint oracle**. At each decode step it supplies the valid-continuation
mask (letters that extend a real word prefix) and whether the prefix is a
complete word, while the rack supplies the availability mask. With hard masks
every complete decode is a valid rack word, so the network only has to learn
to reach maximal length. See `py/scribblez/rack_best/model.py`.

## The theme

`soft_traversal` (ordered) and `anagram` (order-invariant) trade places
between word-validity and rack-length: each wins the task its structure fits
and does nothing on the other. Engineering a lexicon tool for a new task is
therefore choosing the access pattern its shape calls for, and the compiled
structure that provides it: a forward DAWG, a sorted-anagram DAWG, or, for
board-anchored access, a GADDAG (not yet built).
