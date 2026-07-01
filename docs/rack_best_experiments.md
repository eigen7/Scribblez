# Rack-best: a frozen lexicon tool only helps if its structure fits the task

This is the second experiment in the lexical-NN track, one rung harder than
word-validity (`word_validity_experiments.md`) and a step toward
max-move-per-lane (`lexical_nn.md`). Read together, the two experiments make a
single point: **plugging a frozen, compiled lexicon into a network helps only
when the tool's structure matches the shape of the task.**

**Task.** Given a 7-tile rack (an unordered multiset of letters, no blanks),
predict the **length of the longest word formable from a subset** of the rack
(classes 0, 2…7; length 1 never occurs).

**Why it is harder than word-validity.** Word-validity handed the network a word
*in order* — one DAWG walk validates it, which is why the walk tools aced it. A
rack has no order, and the answer requires searching *subsets* and *orderings*
(anagram search). A plain walk of "the given string" no longer applies. That is
the specific new capability this task isolates, with no board geometry.

## Result

Single GPU, ~1 minute per run after the dataset is cached, identical protocol
(below), seed 0. 300k distinct racks, 10% held out. Held-out accuracy is exact
match on the longest length.

| `--lexicon-module` | tool kind | params | held-out exact | within-1 |
| --- | --- | --- | --- | --- |
| `none` | — (full FFN baseline) | 418,696 | 0.656 | 0.985 |
| `none --lexicon-starve-ffn` | — (shrunk FFN, no tool) | 221,320 | 0.655 | 0.988 |
| `soft_traversal` | DAWG **walk** (ordered) | 243,363 | 0.652 | 0.988 |
| `anagram` | subset search (order-invariant) | 227,720 | **1.000** | 1.000 |

Reference points: the label is concentrated (length 5 is 43% of racks, 4–6 is
91%), so a majority-class guess scores ~0.43 and everyone's within-1 is ~0.98.
The no-tool baseline reaches ~0.66 from surface features (letter balance
correlates with having a long word) but cannot be exact.

What to read off this table:

1. **The order-invariant tool nails it.** `anagram` reaches ~100% exact on
   held-out racks — with *fewer* parameters than the baseline — because it
   searches subsets of the rack against the lexicon and the network learns to
   read the achievable-lengths signal.
2. **The wrong-shaped tool does nothing.** `soft_traversal`, the DAWG walk that
   scored 1.000 on word-validity, is no better than the baseline here (0.652):
   a rack has no order to walk, so validating "the sorted rack as a word" tells
   the network almost nothing. This is the predicted failure, and it is the
   whole point.
3. **The win is structure, not capacity.** The starved control
   (`none --lexicon-starve-ffn`, same FFN shrink as the tool runs, no tool)
   matches the full baseline (0.655 vs 0.656). Shrinking the FFN is neutral; the
   jump to exactness comes only from a *structurally matched* tool.

## The pair (why both experiments together matter)

|  | word-validity (ordered word) | rack-best (unordered rack) |
| --- | --- | --- |
| `soft_traversal` (DAWG walk) | **1.000** | 0.652 |
| `anagram` (subset search) | — | **1.000** |
| `none` baseline | ~0.78 | 0.656 |

The two tools trade places. `soft_traversal` wins the ordered task and loses the
unordered one; `anagram` is built for the unordered one. So a differentiable
lexicon module is not a generic "give the network the dictionary" — it is a
*specific search primitive*, and the engineering question for a new task is
which primitive its structure calls for. This is the cheap loop the toy tasks
are for: find a task the existing tools fumble, build the tool whose structure
fits, measure the recovery.

## Components

- **`anagram` module** — `scribblez/max_move_per_lane/lexicon_modules.py`. Two
  ideas: (1) **canonicalize by sorting** — compile the lexicon over each word's
  letters *sorted* (`CAT` → `ACT`), so a multiset has one key and anagrams
  collapse; the rack is fed sorted. (2) **search subsets by a soft skip/use
  walk** — over the sorted rack, each position either skips the tile (carry
  mass) or uses it (advance the sorted-anagram DAWG by that letter); summed over
  positions this explores every subset, with a sparse top-K node state. A word
  completing on a "use" transition is binned by node depth (= word length,
  unique because the sorted lexicon is a trie), giving a per-length
  achievability vector that becomes lane tokens. Frozen: the sorted-anagram DAWG
  and node depths; trainable: the readout. Bingo (use all seven tiles) is just
  the top length bin.
- **Data + labels** — `scribblez/rack_best/data.py`. Racks are 7 tiles drawn
  without replacement from the standard Scrabble bag (blanks excluded). The
  label is computed exactly: for k from 7 down, test every k-subset's sorted
  tuple against the real lexicon's set of sorted k-letter word-bags; the first
  hit is the longest length. Racks are deduplicated so held-out racks are
  genuinely unseen.
- **Model + trainer** — `scribblez/rack_best/model.py` and
  `scripts/rack_best/train.py`. The host is a small transformer over the
  length-7 sorted rack with a prepended CLS token driving an 8-class head; the
  optional lexicon tool (compiled from the real lexicon) feeds tokens.
  `--lexicon-mode replace` shrinks the host FFN so word knowledge must come from
  the tool. The labeled dataset is cached on disk (label generation is the slow
  part), so a sweep across modules is cheap to repeat.

## Reproduce

Prerequisites: `NWL23.kwg` in `<mount>/lexica/` (installed by
`setup_wizard.py`). The real lexicon is used both for the labels and, when a
tool is attached, for the frozen module. All commands run inside the container.

The first run builds and caches the labeled dataset (~1 minute for 300k racks,
under `<mount>/cache/rack_best/`); later runs load it instantly.

Protocol used for the table: `--max-steps 2000 --eval-every 1000 --seed 0`, all
other hyperparameters at their defaults (300k racks, channels 128, 2 layers, 4
heads, FFN mult 4, batch 512, lr 1e-3, weight decay 1e-4, held-out 10%).

```
./py/scripts/rack_best/train.py --lexicon-module none                       # baseline
./py/scripts/rack_best/train.py --lexicon-module none --lexicon-starve-ffn  # control
./py/scripts/rack_best/train.py --lexicon-module soft_traversal             # walk tool (weak here)
./py/scripts/rack_best/train.py --lexicon-module anagram                    # the matched tool
```

Each run prints the label distribution, the parameter count, and per-epoch
`train` / `holdout` exact and within-1 accuracy. Drop `--max-steps`/`--eval-every`
to train the full `--epochs` (default 10).

## Notes

- **No blanks (yet).** Blanks make a tile a wildcard, which adds a second skill —
  deciding what each blank represents — and breaks the sorted-walk `anagram`
  module (a wildcard has no fixed place in the sort). They are deferred to a
  focused follow-on with a letter-type-DP module; the label pipeline already has
  the pieces (a reduced-bag index) to support them.
- **`oracle_crosscheck`** is a board-derived cheat for the lane task and does not
  apply here. `kv_memory` (letter-bag retrieval) is order-invariant and could be
  run, but its bag is lossy; it is expected to sit near the baseline, as on
  word-validity.
