# Rack-best: finding the best word in a 7-tile rack

The second experiment in the lexical-NN track, between word-validity
(`word_validity_experiments.md`) and max-move-per-lane (`lexical_nn.md`). It has
been studied in two versions of increasing difficulty:

1. **Score-only** — predict the *length* of the longest word formable from the
   rack. An unordered-multiset classification task.
2. **Score + word** — emit an actual longest *word spelling*. Ordered generation.

Both take a 7-tile rack (no blanks) drawn from the standard Scrabble bag; the
lexicon tools they use are catalogued in `lexical_tools.md`. Together they refine
the track's thesis: a frozen lexicon tool helps only when its structure matches
the task, and the *ordering* requirement in version 2 is what makes it genuinely
harder than version 1.

---

## 1. Score-only: longest length

**Task.** Given a 7-tile rack (an unordered multiset), predict the length of the
longest word formable from a subset (classes 0, 2…7; length 1 never occurs).

**Why it needs anagram search.** Word-validity handed the network a word *in
order* — one DAWG walk validates it. A rack has no order, and the answer requires
searching *subsets*. A plain walk of "the given string" no longer applies; that
is the specific new capability this version isolates, with no board geometry.

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

Readings: the order-invariant tool nails it (~100% exact, fewer parameters
than the baseline); the wrong-shaped tool does nothing (`soft_traversal`,
1.000 on word-validity, matches the baseline here — a rack has no order to
walk); and the win is structure, not capacity (the starved control matches
the full baseline, 0.655 vs 0.656).

**The pair with word-validity.** The two tools trade places:

|  | word-validity (ordered word) | rack-best (unordered rack) |
| --- | --- | --- |
| `soft_traversal` (ordered walk) | **1.000** | 0.652 |
| `anagram` (subset search) | — | **1.000** |
| `none` baseline | ~0.78 | 0.656 |

A lexicon tool is not "the dictionary in a box" — it is a *specific search
primitive*, and the engineering question for a task is which primitive its shape
calls for.

*Note on reproduction: this result is from rack-best's initial longest-length
implementation. The code was subsequently repurposed for version 2 below, so
reproducing these exact numbers requires that earlier revision; the finding is
what carries forward.*

---

## 2. Score + word: ordered generation

**Task.** Emit a valid *word spelling* (`CAT`, not `ACT`) of maximal length from
the rack. The training target is the lexicographically-smallest longest word;
racks with no formable word are dropped.

**Why ordering is the real step up.** The length (and the tile multiset) is
essentially the version-1 problem in disguise; producing a valid *spelling*
requires order, which lives in the **forward DAWG** — the sorted-anagram structure
`anagram` uses deliberately threw order away. So this version shifts the tool back
to the forward DAWG, now as a per-step **constraint** inside an autoregressive
decoder (see `lexical_tools.md` → "The forward-DAWG constraint", and
`scribblez/rack_best/model.py`).

A decoder-only transformer reads the rack, then generates letters; at each step
the forward DAWG masks the logits to valid word prefixes and the rack masks them
to available tiles. With the constraint **on**, every complete decode is a valid
rack-word, so the network only has to learn to reach the maximal length; **off**
(`--no-dawg`), the decoder must have learned the lexicon itself.

**Result** (300k racks, 10% held out, seed 0, 5000 steps — not converged, both
still climbing). Greedy decode; `valid` = a real formable word of any length (the
tool's clean guarantee), `valid-longest` = also of maximal length, `exact` =
equals the canonical target.

| config | valid | valid-longest | exact |
| --- | --- | --- | --- |
| forward-DAWG constraint | **0.65** | **0.40** | 0.32 |
| `--no-dawg` baseline | 0.48 | 0.32 | 0.26 |

Readings (this is a **more nuanced** result than version 1):

1. **The tool helps, but modestly.** Unlike the discriminative tasks — where the
   no-tool model was stuck at chance — a *generation* model partially learns word
   structure from the 265k training examples, so the baseline reaches 0.48
   validity / 0.32 valid-longest on held-out racks on its own. The tool's gap is
   real but smaller.
2. **Greedy caps the tool's validity below 100%** (0.65, not ~1.0): greedy can
   walk into a dead-end prefix that cannot complete with the remaining tiles. The
   tool guarantees validity only *if the decode completes*. Beam search should
   push this toward 1.0 by exploring completable paths.

**Reproduce** (current code). `NWL23.kwg` must be in `<mount>/lexica/`. The first
run caches the labeled dataset (under `<mount>/cache/rack_best/`).

```
./py/scripts/rack_best/train.py                # forward-DAWG constrained
./py/scripts/rack_best/train.py --no-dawg      # baseline (no lexicon tool)
```

Each run prints the max-length distribution, parameter count, and per-epoch
holdout `valid` / `valid-longest` / `exact`. Defaults: 300k racks, channels 128,
3 layers, 4 heads, FFN mult 4, batch 512, lr 1e-3, 10 epochs, 10% held out.

**Notes.**

- **No blanks yet.** A blank is a wildcard; with the forward-DAWG constraint it
  is "any letter available this step, at the cost of one blank." Deferred.
- **Beam search / recall.** Greedy is myopic; beam decoding would raise validity
  and give the multi-answer view (recall over *all* longest words). Deferred.
