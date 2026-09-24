# Plan: a tag queue over a machine pool

**Status: proposed, not yet plan-reviewed.** Nothing here is built.

**Decision.** The dashboard gets one **machine pool** and one ordered
**tag queue**. A pool machine runs at most one queued tag at a time. When
a machine is free, the dashboard places the first queued tag that fits on
it: it adds the machine to the tag, creates the tag's slots from its
workload's **layout**, and starts them. When every slot of the tag has
finished, the dashboard releases the machine to the next tag. Every
workload must define an **end condition**, so every tag eventually
releases its machine.

## Why

Tags are placed by hand today. The operator adds a machine (rented, or
registered by ssh) to one tag, adds its slots, starts them, and later
notices the tag has finished and does the same for the next one. A
campaign of twelve 10-hour runs on three machines, one of them rented,
means a human awake at every hand-off. There were two more problems on
the first day of the tuning campaign:

- A tag's trainer was put on asus-laptop, whose GPU has 3.7 GiB. The
  transformer trainer needs about 10 GiB at batch 256 without activation
  checkpointing, and about 4.1 GiB with it. It crashed with CUDA
  out-of-memory. Hand placement has no check that the machine fits the tag.
- AWS quota allows one g6.2xlarge today. As more quota arrives, the number
  of machines grows, and hand placement becomes more work.

## Design

### 1. Every workload has an end condition

A tag whose roles never finish would hold its machine forever and stall
the queue. This is also a cost risk: a rented machine only idle-stops once
its slots have finished (#270, #271). So the rule applies to every
workload, not just to queued tags:

- **Every role reaches a terminal state on its own.** A trainer exits at
  its budget. Generators are finished by the scheduler once the trainer is
  done or the store is full. Match eval is finished once nothing is owed and
  the trainer is done (#271).
- **The end parameters cannot be "never".** `param()` gains a `minimum`,
  enforced by validation, by argparse and by the form. Each end parameter
  declares `minimum=1` and loses its "0 = run until paused" meaning.
- **A registry test pins the contract.** Each `WorkloadSpec` names its end
  parameters (`end_params`), and a test checks that every registered
  workload declares at least one, each with a minimum of at least 1.

Changes per workload:

| workload | end condition today | change |
|---|---|---|
| position_eval | `max_rows`, default 0 (never) | default 24,000,000 (~300 generations at the default window); minimum 1 |
| max_move_per_lane | `max_rows`, default 0 | a positive default (to be measured); minimum 1 |
| move_set_eval | `target_pairs` 600, `train_epochs` 20; both allow 0 | minimum 1 on both |
| evidence_trajectories | `target_pairs` default 0, so generators run until paused; `train_epochs` 20 | a positive `target_pairs` default; minimum 1 on both. The "no pair for 15 minutes means final" rule goes away with the 0 case. |
| blind_spots | `target_positions` 100, allows 0 | minimum 1 |
| kill_test | none: generators cycle forever | new `target_pairs`, with the scheduler finishing the generators at the target, the way blind_spots does |
| match_arms | `pairs_per_arm`, finite | already conforms |

**A tag is complete when every one of its slots has finished.** That is
the release signal.

**Existing tags.** Stored params keep their explicit 0s. The minimum is
enforced when a task is created and when it is enqueued, not when a stored
task is loaded, so archived and running tags keep loading
([tag-param migration](../../py/scripts/migrate_tag_params.py) is not
needed). A stored tag with a 0 can be edited before it is enqueued.

### 2. The pool

`pool.json` under the mount root, next to the workload tag trees, holds one
entry per machine the queue may use:

- **localhost**: this machine, running local slots.
- **registered machines** (asus-laptop): ssh host, key and arch, entered
  once here rather than per tag. A tag it is placed on gets the same
  `MachineRecord` that `add_machine` writes today.
- **rental capacity**: an instance type, spot or on-demand, and a
  **count cap**, e.g. "up to 1 × g6.2xlarge, spot". The queue rents up to
  the cap when a queued tag is waiting. The operator sets the cap to what
  their AWS quota allows. A refusal from the provider (quota, capacity)
  appears on the pool entry and is retried with backoff.

Each entry records the facts eligibility needs: vCPUs, GPU count and **GPU
memory**. For a rented type these come from the catalog, which gains a
numeric `gpu_memory_gb`. For localhost and registered machines, one
`nvidia-smi` query runs when the entry is added. An entry can override the
generator thread count its layout would pick. There is no on/off switch:
per the operator, the laptops are always available, and the operator
manages that by hand for now.

Pool machines stay usable by hand-placed tags. A pool machine counts as
**busy** while any unfinished task has slots on it, whether the queue put
them there or not. So the queue never double-books a laptop the operator is
also using by hand.

### 3. Layouts and eligibility

A queued tag has no slots until it is placed, because its slots depend on
the machine. Each workload therefore supplies a **layout**:
`layout(params, machine) -> [(role, threads)]`. For position_eval the
layout is:

- a trainer;
- a generator with the machine's vCPUs minus the trainer's reserve (its
  data-loader workers and the main process);
- match eval, if `match_every_generations > 0`.

Each layout also states each GPU role's **minimum GPU memory**, computed
from the params. A machine is eligible for a tag only if it meets every
role's requirement. For the transformer trunk at batch 256 the measured
peaks give the requirement: about 10 GiB with checkpointing off, about
4.1 GiB with it on. The requirement adds headroom on top of the peak. The
conv trunk's requirement is to be measured.

A queue entry can also narrow eligibility to named pool machines. The
default is "any eligible machine".

### 4. The queue

`queue.json` is an ordered list of `{workload, tag, machines}`, FIFO, and
the operator can reorder it. A tag enters the queue from the create form
("Create & enqueue") or from its task view ("Enqueue"). Enqueueing checks
the end-condition minimum and that the tag has no slots yet.

Each reconcile pass places tags. It visits free pool machines in a fixed
order, and gives each the first queued tag it is eligible for. Placing a
tag:

1. adds the machine to the task (renting one first, for a capacity entry);
2. creates slots from the layout and sets them running;
3. removes the tag from the queue and records the placement in the pool
   entry (`assigned: workload/tag`).

The existing machinery then applies unchanged: bundle deploy (the task's
bundle archs already grow to cover each new machine's arch), where each
slot's output goes, and ssh collection.

### 5. Release and hand-over

When a placed tag is complete, the dashboard:

1. removes its slots;
2. detaches the machine from the task, moving the machine's spend into the
   task's `retired_spend`;
3. frees the pool entry.

The tag's data, checkpoints and dashboard records stay where they are.

A **running rented machine is handed to the next queued tag directly**
instead of being stopped and a fresh one rented. That skips boot and
worker-image pulls, and the images already on its disk carry over. The
instance's owner tag moves with it, so the orphan listing stays correct.
If no queued tag fits, the idle rule stops the machine, and a capacity
entry's instance is **terminated** after the idle timeout. A stopped
instance still pays for its disk, and the pool will rent again on demand.

A tag stays on its machine until it completes. Moving a running tag to a
different machine is left out, even though the trainer can already restore
from the bucket.

A tag whose slot fails (the out-of-memory case) stays placed and shows the
failure. The operator can **requeue** it: its slots are removed, the
machine is released, and the tag returns to the head of the queue. It can
also be narrowed or excluded from the failing machine at the same time.

### 6. Dashboard

A **Queue** view shows:

- the pool: each entry's kind, GPU, current tag and state, with add, edit
  and remove;
- the queue: drag to reorder, edit eligibility, remove;
- a rental-capacity line with the cap, what is rented, and the cumulative
  spend.

The create form gains "Create & enqueue". The task view gains "Enqueue"
and "Requeue".

## Alternatives

- **Chaining** ("start B on A's machines when A finishes"). This is
  trivial, but it fixes each tag's machine in advance, which is the
  manual pattern that motivated this plan. The pool makes machines
  interchangeable.
- **Packing several tags onto one machine** by CPU, GPU and memory. The
  cloud-machines review rejected a general machine model because packing
  had no present value, and it still has none: a position_eval tag
  saturates its machine's GPU with the trainer and its CPUs with the
  generator. The pool shares machines over time, never at the same moment.
- **An external scheduler (Slurm, Ray, Kubernetes).** It would duplicate
  the dashboard's slot, bundle and machine lifecycle, and it could not
  honour its per-kind delivery rules. That is a large dependency for a
  queue of a dozen tags.

## Landing

| PR | Content |
|---|---|
| 1 | End-condition contract: `param(minimum=)`, `WorkloadSpec.end_params` plus the registry test, the per-workload changes above (kill_test's target and scheduler finish), and create-time validation. Independent of the rest, and useful on its own. |
| 2 | Pool: `pool.json`, localhost and registered entries, the GPU-memory probe, the catalog's `gpu_memory_gb`, the pool UI, and busy detection. Tasks can also add a pool machine by hand from the task view. |
| 3 | Layouts, eligibility, and the queue with placement, release and requeue, for localhost and registered machines. The laptops can then run the campaign unattended. |
| 4 | Rental capacity: renting up to the cap on demand, hand-over of running instances, and termination when idle. |

## Open questions for review

- evidence_trajectories' `target_pairs` default: the corpus size its runs
  actually use.
- A tag that wants more than one machine (a rented trainer plus extra
  generator machines) has no layout here; hand-added slots still work
  alongside a placement. Is that enough?
- Should the queue be global or per workload? It is global here, with
  eligibility doing the filtering, since the machines are shared.
