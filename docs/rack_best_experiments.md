# Rack-best: finding the best word in a 7-tile rack

The second experiment in the lexical-NN track, between word-validity
([word_validity_experiments.md](word_validity_experiments.md)) and
max-move-per-lane ([lexical_nn.md](lexical_nn.md)). It was studied in two
versions of increasing difficulty:

1. **Score only:** predict the *length* of the longest word formable from the
   rack. An unordered-multiset classification task.
2. **Score and word:** emit an actual longest *word*. Ordered generation.

Both take a 7-tile rack (no blanks) drawn from the standard Scrabble bag; the
lexicon tools they use are catalogued in [lexical_tools.md](lexical_tools.md).

**Findings.** The order-invariant `anagram` tool solves version 1 exactly,
while the ordered DAWG walk that solved word-validity does nothing there: a
frozen lexicon tool helps only when its structure matches the task. In
version 2 the forward-DAWG constraint helps generation, but modestly (0.65 vs.
0.48 validity), because the *ordering* requirement is genuinely harder and a
generative model partly learns word structure without the tool.

---

## 1. Score-only: longest length

**Task.** Given a 7-tile rack (an unordered multiset), predict the length of the
longest word formable from a subset (classes 0, 2…7; length 1 never occurs).

**Why it needs anagram search.** Word-validity hands the network a word *in
order*, and one DAWG walk validates it. A rack has no order, and the answer
requires searching *subsets*, so walking the given string no longer applies.
Subset search is the new capability this version isolates, with no board
geometry.

**Result** (300k racks, 10% held out, seed 0; held-out exact match on the longest
length):

| tool | kind | params | held-out exact | within-1 |
| --- | --- | --- | --- | --- |
| `none` | full-FFN baseline | 418,696 | 0.656 | 0.985 |
| `none --lexicon-starve-ffn` | shrunk FFN, no tool | 221,320 | 0.655 | 0.988 |
| `soft_traversal` | ordered DAWG walk | 243,363 | 0.652 | 0.988 |
| `anagram` | order-invariant subset search | 227,720 | **1.000** | 1.000 |

Reference points: the label is concentrated (length 5 is 43% of racks, 4–6 is
91%), so a majority guess scores ~0.43 and everyone's within-1 is ~0.98. The
no-tool baseline reaches ~0.66 from surface features but cannot be exact.

Readings:

- The order-invariant tool solves it (~100% exact, with fewer parameters than
  the baseline).
- The wrong-shaped tool does nothing. `soft_traversal`, perfect on
  word-validity, matches the baseline here: a rack has no order to walk.
- The win is structure, not capacity: the starved control matches the full
  baseline (0.655 vs. 0.656).

**The pair with word-validity.** The two tools trade places:

|  | word-validity (ordered word) | rack-best (unordered rack) |
| --- | --- | --- |
| `soft_traversal` (ordered walk) | **1.000** | 0.652 |
| `anagram` (subset search) | — | **1.000** |
| `none` baseline | ~0.78 | 0.656 |

A lexicon tool is not "the dictionary in a box" but a *specific search
primitive*, and the engineering question for a task is which primitive its
shape calls for.

These numbers come from the longest-length version of the rack-best code,
which was later repurposed for version 2. Reproducing them exactly requires
that earlier revision from git history; `py/scripts/rack_best/train.py` today
runs version 2.

---

## 2. Score + word: ordered generation

**Task.** Emit a valid *spelling* (`CAT`, not `ACT`) of a maximal-length word
from the rack. The training target is the lexicographically smallest longest
word; racks with no formable word are dropped.

**Why ordering is the real step up.** The length (and the tile multiset) is
essentially the version-1 problem in disguise. Producing a valid spelling
requires order, which lives in the **forward DAWG**; the sorted-anagram
structure `anagram` uses deliberately discards it. So this version goes back
to the forward DAWG, now as a per-step **constraint** inside an
autoregressive decoder (see "The forward-DAWG constraint" in
[lexical_tools.md](lexical_tools.md), and `py/scribblez/rack_best/model.py`).

A decoder-only transformer reads the rack, then generates letters. At each
step the forward DAWG masks the logits to valid word prefixes and the rack
masks them to available tiles. With the constraint **on**, every complete
decode is a valid rack word, so the network only has to learn to reach the
maximal length. With it **off** (`--no-dawg`), the decoder must have learned
the lexicon itself.

**Result** (300k racks, 10% held out, seed 0, 5000 steps; not converged, both
still climbing). Greedy decoding. `valid`: a real, formable word of any length
(what the tool guarantees). `valid-longest`: also of maximal length. `exact`:
equals the canonical target.

| config | valid | valid-longest | exact |
| --- | --- | --- | --- |
| forward-DAWG constraint | **0.65** | **0.40** | 0.32 |
| `--no-dawg` baseline | 0.48 | 0.32 | 0.26 |

Readings:

1. **The tool helps, but modestly.** On the discriminative tasks the no-tool
   baseline could not get past surface features, but a *generative* model
   partly learns word structure from the 265k training examples: the baseline
   reaches 0.48 validity and 0.32 valid-longest on held-out racks on its own.
   The tool's gap is real but smaller.
2. **Greedy decoding caps the tool's validity below 100%** (0.65, not ~1.0).
   Greedy can walk into a dead-end prefix that cannot be completed with the
   remaining tiles; the tool guarantees validity only *if the decode
   completes*. Beam search should push this toward 1.0 by exploring
   completable paths.

**Reproduce.** `NWL23.kwg` must be in `<mount>/lexica/`. The first run caches
the labeled dataset under `<mount>/cache/rack_best/`.

```
./py/scripts/rack_best/train.py                # forward-DAWG constrained
./py/scripts/rack_best/train.py --no-dawg      # baseline (no lexicon tool)
```

Each run prints the max-length distribution, parameter count, and per-epoch
holdout `valid` / `valid-longest` / `exact`. Defaults: 300k racks, channels 128,
3 layers, 4 heads, FFN mult 4, batch 512, lr 1e-3, 10 epochs, 10% held out.

**Not yet done.**

- **Blanks.** A blank is a wildcard; under the forward-DAWG constraint it would
  mean "any letter valid at this step, at the cost of one blank."
- **Beam search and recall.** Greedy is myopic; beam decoding would raise
  validity and allow a multi-answer view (recall over *all* longest words).
