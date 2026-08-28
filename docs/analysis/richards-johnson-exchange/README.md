# Nigel Richards's EELLT exchange

**Richards vs. Johnson — 2025 World Cup, Round 11, move 22**

One of the most talked-about plays of the 2025 World Cup. Facing expert Mike
Johnson and up by 53, Nigel Richards passed over every scoring play — including
the engine's clear favorite, **ALLEE** — and instead **exchanged five tiles**,
throwing back **EELLT** and keeping only **AN**. No engine setting approves of
it (Macondo ranks it anywhere from 12th to 40th). This is an attempt to explain
why it may nonetheless be the winning play.

The board images below are rendered by this repo's own board renderer (the
manual GCG tool's `--dump-gcg` state dump feeding the `?tool=render` harness —
see [Reproducing the images](#reproducing-the-images)).

Credit for the analysis goes to Will Anderson and his fellow commentators in
his YouTube breakdown of this game ([search: "Will Anderson Nigel Richards
exchange"](https://www.youtube.com/results?search_query=will+anderson+nigel+richards+exchange));
the game itself is [annotated on
cross-tables](https://www.cross-tables.com/annotated.php?u=55430).

## The position

![The critical position at move 22](images/critical-position.png)

- **Nigel Richards: 379, on turn. Mike Johnson: 326.** Nigel leads by 53.
- **Nigel's rack: A E E L L N T.**
- **7 tiles remain in the bag**, so there are **14 tiles unseen** from Nigel's
  perspective (7 in the bag + 7 on Mike's rack).
- The unseen pool is **extremely bingo-prone**: every tile worth 4+ points is
  already on the board. Only the **C** and **D** are worth more than a point,
  and both are strong bingo letters.
- The engine's clear favorite is **ALLEE** (a tree-lined walkway) for 16.

Nigel exchanged **EELLT**, keeping **AN**.

## Reading the position first

The move only makes sense on top of two reads that Nigel makes *before* he
weighs any play. Both come from Mike's last two turns — **ZONES** (28) then
**WOE** (6).

### 1. Mike is very likely about to bingo

An expert who plays a one-tile move like WOE is almost always holding a
bingo-prone rack for next turn, and here the unseen tiles make that conclusion
even stronger. WOE also reads as a deliberate **S-hook setup** (WOE → WOES) on
a board that otherwise has none. So Nigel should expect a bingo, most likely:

- **hooking an S onto WOE → WOES** (the obvious lane WOE just created), or
- down the **V of VIRGA** — less likely, but higher-scoring and dangerous,
  giving Nigel little counterplay, or
- alongside the **Y of JUDY** — least likely, since it needs the last **A**.

### 2. The C is very likely still in the bag

If Mike had held the **C** on his ZONES turn, he would have played **CONE (37)**
and kept his S, not ZONES. He had no good C play on WOE either. So the C is
almost certainly **still in the bag** — meaning **Nigel is very likely to draw
it**, and Mike can only have it if he happened to draw it on a one-tile
replenishment.

*(Minor reads in the same vein: Mike probably doesn't hold two T's — WATT would
have been an equally good setup that also sheds a duplicate — and is a shade
less likely to hold the R, or he'd have played ZONER to set up his S even more
powerfully.)*

With those two reads in hand, the logic of the exchange falls out.

## The core logic of the exchange

When you are winning by a large margin, the decision that matters is the
worst case: **"how could I lose this game?"** Here the loss scenario is clear —
Mike bingos, and then gets out before Nigel can catch up. Every candidate has
to be judged against that.

### Playing a word — even ALLEE — helps Mike's timing

Playing a scoring word draws Nigel back up to 7 and leaves only **2 tiles in
the bag**. After Mike's expected bingo, Mike is then left with just **2 tiles**
and **goes out immediately** — Nigel gets only one more turn, and the game
simplifies into exactly the race Mike wants. Extending the lead doesn't help if
it hands Mike the tempo.

### The exchange satisfies two opposing goals at once

Nigel needs to do two contradictory things:

- **Play as *few* tiles as possible** — keep **7 tiles in the bag** so that
  when Mike bingos he is stuck with a **full 7-tile rack** and *cannot* go out,
  buying Nigel the extra turns he needs to erase a ~20-point post-bingo
  deficit.
- **Play as *many* tiles as possible** — draw **5 fresh tiles** to chase the
  **C** that is almost certainly in the bag.

The only move that does both is an **exchange**. (Note that ALLEE, a 5-tile
play, *does* fish for the C by drawing 5 — but it empties the bag to 2 and loses
the timing. The exchange keeps the C-fishing while winning the timing battle.)

### It sabotages Mike's endgame rack

Whatever Nigel throws back, Mike is guaranteed to end up holding it: with only
14 unseen tiles, Mike collects the leftovers in the endgame. By throwing
**EELLT**, Nigel guarantees Mike a rack with **duplicate E's and duplicate L's**
at a minimum (triplicates are possible on further draws) — a stiff, inflexible
rack that plays badly after a bingo and lets Nigel claw back the deficit over
two turns.

### Keep AN, not EN

Of the two tiles Nigel keeps, **AN** is much better than EN:

- With **AN** in reserve, **9 of the 13** other unseen letters give Nigel a
  4-letter C-play hooking **CLOSE**; keeping **EN** drops that to **5 of 13**.
- The **A** also plays alongside the **Y of JUDY**, giving Nigel an out there.
  The **E** does not — which is likely part of why he threw the E back.

### The line it aims for

Mike bingos (~70, Nigel down ~20) → Nigel scores **25-30 along the bottom**, or
better, more by reaching the triple-word with a **CLOSE hook** → Mike, stuck
with his junk EELLT-based rack, plays something small → Nigel goes out with a
**3-letter word** such as an **A\_ play hooking the Y of JUDY**.

## Why not ALLEE — the worst case

Even when Nigel draws the C after ALLEE, it doesn't guarantee the game.
Drawing the C is only about **5/7** after ALLEE (and that already assumes the C
is in the bag), and even with it Nigel doesn't always have the high-scoring
CLOSE hook. In one line Mike bingos in reply, Nigel is left with **CEINNTT** —
which cannot reach the triple with the CLOSE hook — and Mike's leftover **AD**
gives him unblockable outplays and the win. This is only an *illustration* that
the C is not a guarantee, not the main argument; the main argument is the
timing above.

## The YE\_ and double-S danger

Playing a word *now* (leaving two in the bag) also exposes Nigel to two-tile
setups off the **YE** — **YETI, YEET, YETT, YEAN** — that stand Mike's last S
up for nearly 100 points. Worse, that creates **two different S-lanes** for a
bingo ending in S — **WOES** on one side and **YE\_S** on the other — and Nigel
**cannot block both**. Exchanging rather than playing a word is what lets Nigel
answer this, again through timing: after an exchange he can immediately block or
make a high C-play with one tile left in the bag.

## What the engines say — and what happened

No engine approves of the exchange, because the decisive factors are
pre-endgame *timing and blocking dynamics* that current engines don't evaluate.
As Will Anderson's panel put it, this is a "human position" — and they suspect
future engines will vindicate it.

What actually happened is just one of countless ways the game could have gone,
and is **not** in itself a vindication of the play:

![Nigel drew the C and played CLAN](images/clan.png)

Mike did *not* connect on a bingo and played out; Nigel drew the C and played
**CLAN** — which makes **CLOSE** across row 12, exactly the hook the whole plan
was built around — leaving one tile in the bag. Mike then bingoed **RELISTEN**,
and Nigel answered with his own out-bingo **ENTAILER** (hooking the R) to win
495-412.

![Nigel goes out to win](images/final.png)

## Reproducing the images

The board images are generated entirely by this repo:

```bash
cd docs/analysis/richards-johnson-exchange

# 1. Dump per-ply front-end state JSON straight from the GCG (no server):
../../../target/engine/manual_gcg_tool --dump-gcg game.gcg \
  --dump-plies 21,24,26 --dump-out states

# 2. Rasterize the states to PNGs via the ?tool=render harness (headless Chromium):
node ../../../web/scripts/render_boards.mjs render.json
```

The manifest (`render.json`) names each state, its output PNG, an optional
`caption`, and an optional `hideRacks` (e.g. `[0]` blanks Mike's rack to unseen
"?" tiles for the Nigel-POV figure).
