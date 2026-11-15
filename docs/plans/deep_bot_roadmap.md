# DeepBot: the order of the remaining work

**Status: proposed 2026-11-15, not built.** This orders what remains of
[deep_bot.md](deep_bot.md)'s build order, and replaces the order of its steps
11 to 13. Its measurements inform the next step; no pass or kill thresholds
are fixed in advance.

**Decision.** Mature the agent whose sims use mechanical replies before the
one that searches Bob's reply, then make Bob's reply a learned policy.

- **First, the `hasty` agent,** which is Macondo's BestBot in outline with
  our parts in it: brought level with BestBot on the last turn, its parts
  measured one at a time, and advanced by [generations](deep_bot_generations.md).
- **Then the policy as Bob's reply** inside the sims: a transformer, trained
  on the pools, which should learn to block a setup because the pools' labels
  reward the block.
- **Bob's subtree, the evidence store, steering and the learned budget
  wait** until those are mature.

## Why

- **The `hasty` agent is already the cheap one and the generating one.** It
  plays a game against HastyBot in 1.8 seconds of one laptop GPU and wins 62%
  of 300; against BestBot it stands 28 to 27 over 55 games, which says only
  that it is not far off. Bob's subtree costs seconds a turn.
- **The subtree's reply is too noisy to train or measure against today.** At
  20 racks a second run of the subtree agrees with the first at 45% of racks,
  no more often than HastyBot's move does ([build order](deep_bot.md#build-order),
  step 10).
- **What the subtree is for is Bob's informed reply:** above all, that he
  blocks a setup. Under the `hasty` reply Bob never blocks on purpose, so the
  agent overvalues a setup against an opponent who would. A policy that
  blocks gives the sims the same correction for one network call.
- **Nothing carries what one candidate's sims learn to another** in the
  `hasty` agent: the candidates share their deals, which pairs their noise,
  and the screen's ranking, and no more. The store was to do that. Until it
  is built, what transfers is what the networks learn across positions.

## The agent beside BestBot

| | BestBot | DeepBot, `hasty` reply | Switch in DeepBot today |
|---|---|---|---|
| Candidates | static equity's top 40, or 80 late | every legal move but the pass, screened by 100 raced rollouts, the best 10 (32 late) kept | `--screen-rollouts 0` |
| Opponent's rack | drawn uniformly | a leave from the leave inference model, the rest drawn | omit the model |
| Rollout | 5 plies, or to the end late | 2 plies | none: the kernel's depth is fixed |
| Value at the rollout's end | Macondo's win table | the leaf network | none |
| Stopping and the pick | Macondo's rule every 128 iterations; the highest mean win probability | racing in rounds; the likeliest best | `--race 0` |
| A play that empties the bag | with one tile in the bag, the turn is solved exactly | the endgame solver makes Bob's reply | `--endgame-reply-budget 0`; `--peg 1` solves the turn |
| Rollouts run on | the CPU | the GPU | |

Depth and the leaf are tangled: the leaf network is what lets the rollout
stop at two plies. The last turn is measured: at the 149 blind spots with one
tile in the bag, the `hasty` agent's picks won 0.255 of the bag's draws with
HastyBot making Bob's reply to a play that empties the bag, where the best
play wins 0.346, and win 0.324 with the endgame solver making it.

## The order

Each step names what it measures.

1. **An exact last turn under the `hasty` reply.** Built 2026-11-15: the
   endgame solver makes Bob's reply to a candidate that empties the bag, as
   it does under `subtree` ([When Alice's move empties the
   bag](deep_bot.md#when-alices-move-empties-the-bag), with its numbers).
   `deepbot-endgame --peg 1`, BestBot's pre-endgame solver, is the exact and
   slower alternative for the turn with one tile in the bag. *Measure:* the
   picks at the 149 positions against the pre-endgame solver's ranks
   (`blind_spot_picks.py --peg-ranks`), and the seconds a turn, for both.
2. **The parts, one at a time, with today's switches.** The agent against
   itself with one part off: the leave inference model, the screen, the
   race. A game costs 3.3 seconds of the GPU, so 2,000 mirrored games, a
   standard error near 1.1 points, are under two hours a match. Also here:
   why the agent wins 62% of its games against HastyBot at a mean spread
   near zero. *Measure:* each match's win rate and spread.
3. **Generations under the `hasty` reply:**
   [deep_bot_generations.md](deep_bot_generations.md), its seven steps.
   *Measure:* as that plan lists, the matches against HastyBot and release 0
   among them.
4. **The policy as a transformer.** Offline, on the `policy_set` workload,
   so it can overlap step 3. See [The policy](#the-policy). *Measure:* its
   cross-entropy on its own training pools against the floor; held out, how
   often its top move is the pool's best and its regret, against static
   equity's, overall and on the turns where the pool's best is not equity's
   top move; its replies at known setups with Bob to move; and PolicyBot
   against HastyBot.
5. **The policy as Bob's reply:** a `policy` reply model, in the sims of the
   screen's survivors. See [The policy reply](#the-policy-reply). *Measure:*
   policy evaluations a second on the device and the seconds a turn; the
   agent against the `hasty` agent; and the picks at the known positions
   where the answer turns on Bob's reply.
6. **Generations under the `policy` reply.** The generating agent takes it
   up, and the policy each release carries is then played. *Measure:* the
   loop's matches, with a release of the `hasty` line as a second fixed
   opponent.

When time allows, in no fixed order:

- **Depth and the leaf as switches.** Step 5 breaks the rollout into stages
  with a network call between them, after which a rollout of more plies and
  a win-table leaf are parameters. They complete step 2's table.
- **The luck correction** over kept match games
  ([the matches](deep_bot_generations.md#the-matches)). Step 2's matches look
  for effects of a point or two, where it would pay soonest.
- **A convolutional trunk under the policy,** on step 4's readings and step
  5's speed.
- **A match against BestBot** on a release, on idle cores.

Waiting, with their designs as [deep_bot.md](deep_bot.md) has them: Bob's
subtree in the generating agent and the root correction head (the end of its
step 11), the store input (step 12), steering and the learned budget (step
13), and the `alice-subtree` reply. The subtree stays in the agent as it is,
and reads the known positions beside the agent being matured.

## The policy

Today's policy (`policy_set`, form 3) is a convolutional board trunk, 128
channels by 8 blocks, with each candidate embedded against the board tokens,
attending into them once, and read by a linear logit. It ties static equity:
PolicyBot went 208 to 189 against HastyBot. It underfits its own labels, a
training cross-entropy of 2.40 against a floor of 1.59 where equity's softmax
scores 3.10, so more pools alone will not fix it. The shelved greedy form
could not rank near-top moves at all, with labels whose split-half
reliability was 0.82 to 0.88.

A block is a near-top distinction that binds three things: what the opponent
probably holds, one square, and a candidate's footprint. That is a lookup by
content, which attention does directly. Where the project has measured
outputs tied to squares, the transformer trunk beat the convolutional one at
every placement head (`position_eval`, 2026-09-04, at equal generations);
the two tied on the leaf network's win probability
([results](../deep_bot_leaf_eval_results.md)), a scalar that does not see
such cases.

So the policy becomes a transformer first, and a convolutional trunk is
tried under it later:

- **The board trunk** is `position_eval`'s transformer trunk, with its 27
  tile-supply tokens, behind the `policy_set` workload's trunk param.
- **The candidates are tokens** that attend to the board and supply tokens
  through several layers, in place of one cross-attention. How a candidate
  is presented is the open design question; the afterstate, the board with
  the move played, is the first form to try.
- **The opponent-rack input** stays the chooser's belief about the
  opponent's rack, so that after a setup the belief's tell (she kept the F)
  is in front of the model.
- **The labels** are the pools': `q`, the chance the move is the sim's best.
  A pool is labeled with the opponent replying by HastyBot, so at a position
  after a setup the move that blocks it scores well: the rollouts in which
  it is not blocked have the opponent cash it. Setups are rare in HastyBot's
  games, and commoner in DeepBot's, which the loop's pools come from.

The subtree's reply is not the reference for whether the policy blocks: it
is too noisy at the racks it runs. The pools are, and the known setups.

## The policy reply

Under `--reply policy`, Bob's reply in a sim is the policy's top move among
static equity's top `K` at his rack, and Alice then replies by HastyBot and
the leaf network values the end state, as now. Blocking is Bob's decision,
and HastyBot's move is what cashes a setup, so Alice's ply stays mechanical.

- **Only the survivors' sims use it.** The screen keeps the `hasty` reply:
  it covers hundreds of moves, and an unblocked setup looks better under it,
  so the screen errs toward keeping the moves the policy reply then judges.
  The survivors' sims are 10 candidates by up to 1,000 rollouts, each with
  its own rack for Bob: at most 10,000 policy positions a turn, 32,000 late.
- **Bob's belief about Alice's rack** is one leave inference query per
  candidate, since it reads only what her move shows.
- **The inputs are encoded on the device.** PolicyBot spends about a
  millisecond of CPU a decision on them, which 10,000 decisions a turn
  cannot afford. The rollout kernel's cross-check code and the leaf
  network's device encoder are the parts to build from, and the encoding
  stays the one training uses.
- **The kernel runs in stages:** Bob's top `K` by equity, the policy call,
  his move applied, Alice's reply, the leaf. `K` is read from step 4: the
  smallest at which the policy's top move over the whole pool is almost
  always among equity's top `K`.

The cost is unmeasured. The smallest transformer timed on this GPU, a
board-only trunk of width 64 and 4 layers, ran 62,000 positions a second, at
which a turn's 10,000 are a sixth of a second; the policy is wider and
carries candidate tokens, so step 5 measures it before anything is sized.
