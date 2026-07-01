# Word-validity: a clean test of "compiled lexicon as a differentiable tool"

This is the smallest experiment in the lexical-NN track (see `rack_best_experiments.md`
for the next rung and `lexical_nn.md` for the larger per-lane task). It asks one
question: can a neural network learn to *use* a frozen, compiled lexicon as a tool, and
thereby generalize to words it never saw in training? The rack-best follow-up sharpens
the lesson — that a tool only helps when its *structure* fits the task.

**Task.** Classify a word (2–15 letters) as lexicon-legal (label 1) or a phony
(label 0).

**Why it is a tool-use test, not a memorization test.** The negatives are not random
strings — they are a *phony lexicon* generated to match the real lexicon's character
statistics (see "Phony lexicon" below). `YOP` looks like a word; `QVF` does not. Because
real and phony words are drawn from the same letter statistics, surface features cannot
reliably separate them. A model can score well on words it has *seen* by memorizing, but
the only way to score well on a **held-out** word is to actually look it up. So held-out
accuracy measures whether the network learned to use the lexicon.

## Result

Single GPU, ~1–2 minutes per run, identical protocol (below), seed 0:

| `--lexicon-module` | tool kind | params | held-out acc | (real / phony) |
| --- | --- | --- | --- | --- |
| `none` | — (full FFN baseline) | 418,817 | 0.785 | 0.868 / 0.702 |
| `none --lexicon-starve-ffn` | — (shrunk FFN, no tool) | 221,441 | 0.788 | 0.861 / 0.716 |
| `kv_memory` | retrieval | 264,705 | 0.787 | 0.859 / 0.716 |
| `oracle_crosscheck` | board-derived cheat | 256,001 | 0.998 | 0.999 / 0.997 |
| `soft_traversal` | differentiable DAWG walk | 243,484 | 1.000 | 1.000 / 1.000 |
| `straight_through` | DAWG walk, straight-through | 243,484 | 1.000 | 1.000 / 0.999 |

Three things to read off this table:

1. **The DAWG-walk tools generalize ~perfectly.** `soft_traversal` and
   `straight_through` reach ~100% on held-out words — words never seen in training —
   because the tool, compiled from the *real* lexicon, validates any real word and
   rejects any phony, and the network learns to read that signal.
2. **The win is the tool, not capacity.** The starved control
   (`none --lexicon-starve-ffn`: same FFN shrink as the tool runs, but no tool) matches
   the full baseline (0.788 vs 0.785). Shrinking the transformer's FFN is neutral here;
   the jump to ~100% comes only when a working tool is attached — and the tool runs do it
   with *fewer* parameters than the baseline.
3. **Not every tool conveys membership.** `kv_memory` is no better than the baseline. It
   retrieves the nearest lexicon words' *letter bags*, which are anagram-invariant and
   lossy, and it returns one coarse lane-level hint — not enough to tell a real word from
   a same-letters phony. A useful negative result: a differentiable lexicon module only
   helps if what it exposes actually distinguishes members from non-members.

`oracle_crosscheck` is a diagnostic, not a real method: it reads exact board-derived
legality rather than being queried by the network, so it cannot be compared to the others
as a legitimate generator. It is here only as a near-perfect ceiling.

**Caveat on the 78% baseline.** A no-tool model is *not* at chance (50%). With order-3
phonies a small transformer reaches ~78% by exploiting order-4+ regularities the
order-3 character model did not reproduce, and at this size it is not memorizing (train
≈ held-out). Raising the phony generator's `--order` pushes that surface ceiling toward
chance and makes the tool's advantage starker; order 3 is the default used here.

## Components

- **Phony lexicon** — `py/tools/generate_phony_lexicon.py`. Fits an order-k character
  Markov model (start/end padded, so short words generate) on the real words, then for
  each length aims for the same count the real lexicon has: it *enumerates and ranks* the
  short lengths (whole space small enough to score every candidate, so the short tail
  fills exactly) and *samples with a give-up bound* the long lengths. Rejects real words
  and duplicates. Writes a word list and a real `.kwg` (via `write_kwg`, the DAWG-only
  inverse of `compile_kwg` in `scribblez/lexical_tool/compiler.py`). The
  committed artifact is `phonies/PHONY-NWL23.kwg` (155,660 words, 0 overlap with NWL23).
- **Lexicon modules** — `scribblez/lexical_tool/modules.py`. Frozen,
  compiled-lexicon `nn.Module`s selected by name (`soft_traversal`, `straight_through`,
  `oracle_crosscheck`, `kv_memory`); each documents its mechanism and trade-offs. The same
  registry serves the per-lane task.
- **Model + trainer** — `scribblez/word_validity/model.py` and
  `scripts/word_validity/train.py`. The host is a small transformer over the padded word
  with a prepended CLS token driving a binary head; the lexicon tool (compiled from the
  *real* lexicon) feeds a per-cell residual plus a couple of tokens. `--lexicon-mode
  replace` shrinks the host FFN so word knowledge must come from the tool; `none` keeps
  the full FFN.

## Reproduce

Prerequisites: the lexica live in `<mount>/lexica/`. `NWL23.kwg` is installed by
`setup_wizard.py` (downloaded from Woogles); `PHONY-NWL23.kwg` is committed under
`phonies/` and copied into the mount by `setup_wizard.py` (`install_phony_lexica`). All
commands run inside the container.

1. (Optional) regenerate the phony lexicon — already committed, so only needed to change
   it (e.g. a higher Markov order):

   ```
   ./py/tools/generate_phony_lexicon.py \
       --out-txt phonies/PHONY-NWL23.txt --out-kwg phonies/PHONY-NWL23.kwg
   # then re-copy into the mount, or re-run setup_wizard.py on the host
   ```

2. (Optional) inspect the phony lexicon next to the real one:

   ```
   ./py/tools/explore_lexicon.py /workspace/mount/lexica/PHONY-NWL23.kwg --length 2 --limit 0
   ./py/tools/explore_lexicon.py /workspace/mount/lexica/PHONY-NWL23.kwg --pattern 'QU.' --sample 15
   ```

3. Run each configuration. Protocol used for the table: `--max-steps 1800
   --eval-every 1800 --seed 0`, all other hyperparameters at their defaults (channels 128,
   2 layers, 4 heads, FFN mult 4, batch 512, lr 1e-3, weight decay 1e-4, held-out 10%).

   ```
   ./py/scripts/word_validity/train.py --lexicon-module none                       # baseline
   ./py/scripts/word_validity/train.py --lexicon-module none --lexicon-starve-ffn  # control
   ./py/scripts/word_validity/train.py --lexicon-module soft_traversal
   ./py/scripts/word_validity/train.py --lexicon-module straight_through
   ./py/scripts/word_validity/train.py --lexicon-module oracle_crosscheck
   ./py/scripts/word_validity/train.py --lexicon-module kv_memory
   ```

   Each run prints, per epoch, `train` and `holdout` accuracy with the held-out figure
   split into `real` and `phony`. Drop `--max-steps`/`--eval-every` to train the full
   `--epochs` (default 10) and watch the held-out accuracy converge.
