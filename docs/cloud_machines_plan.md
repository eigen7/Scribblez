# Plan: rented machines -- AWS instances as ssh machines

Status: plan-reviewed, not yet implemented. Follows
[cloud_training_plan.md](cloud_training_plan.md), whose trainer I/O
contract and bucket legs (3a-3c) this plan keeps whole; it replaces that
plan's Runpod cloud slot.

## Why

Runpod pods have properties that make them the wrong long-term shape for
the worker fleet, independently of any bug:

- A stopped pod is pinned to its host, bills for its disk, and cannot
  start again once the host has filled -- so a pause has to be a
  stop-and-replace, and any error in that path rents a fresh pod.
- Whole instance flavors go out of stock for hours at a time.
- A host's CPU family is unknown until a pod lands on it, so a bundle may
  run the generic build for a day before anyone notices.
- A pod cannot run Docker, so colocating a generator with the trainer
  needs a process supervisor and a control plane routed through the
  bucket, which lags by construction.

An AWS instance has none of these properties: it stops and starts on any
host (its disk survives, and costs cents while stopped), its CPU family
follows from the instance type, and it is a machine -- ssh-reachable,
running Docker -- which is exactly what the `ssh` kind already drives:
pause is a Docker pause, a scheduler gate parks a container, redeploy
replaces a container once it has drained, and liveness is a probe over the
link rather than a listing through a vendor API.

The dashboard owns the instance lifecycle rather than the operator. That
is what lets it enforce the properties that keep money safe: every
instance is tagged and listed each pass, an idle machine is stopped, an
unknown instance is shown rather than silently billed, and spend is per
machine and rolls into the task's total when the machine goes.

## What changes, in one sentence

A task can **rent a machine**: an AWS instance launched from a stock GPU
image, reached over ssh, hosting that task's slots of the existing `ssh`
kind -- and the `cloud` kind, Runpod and the pod-side bootstrap go.

## Concepts

- **Machine.** A record in the task's task.json (`task.machines`): a
  machine is rented by one task for its own slots, and the record lives
  and dies with the task the way its slots do. Fields: `name`, `provider`
  (`aws` | `manual`), `host` (`user@address`, what ssh is given),
  `identity_file` and `known_hosts_file` (both under
  `/workspace/mount/cloud/machines/<name>/` for a rented machine; a manual
  one may leave both unset and use the container's own identity), `arch`,
  `gpu`, and for a rented machine `instance_id`, `instance_type`,
  `region`, `cost_per_hr`, `launched_at`, `spend`. A **manual** machine is
  one the operator launched or owns and registered by name and host.
  Slots on today's host-string form (`WorkerRecord.host`) stay as they
  are; nothing is migrated.
- **Slot.** The `ssh` kind, unchanged in behavior. A slot on a machine
  carries `machine` (the name) instead of `host`; the add-worker form's
  host text box becomes a selector over the task's machines plus the free
  host string.
- **One ssh helper.** `WorkerManager._ssh(task, w) -> SshMachine` is the
  only place an `SshMachine` is built, resolving a slot's machine to host,
  identity file and known-hosts file. `SshMachine` takes those three; a
  host-string slot passes the host alone and keeps today's defaults.
- **Host keys.** A rented machine's key is unknown at launch, and AWS
  reuses addresses, so the global known_hosts is wrong twice over. Each
  rented machine gets its own known-hosts file with
  `StrictHostKeyChecking=accept-new`, created empty at launch; a relaunch
  is a new machine name, so a reused address never meets a stale key.
- **Provider.** `py/cloud/providers/aws.py` behind a small protocol in
  `base.py`: `catalog()`, `launch(request) -> instance`, `describe(ids)`,
  `stop`, `start`, `terminate`, and `refusal(error)` (the operator-facing
  sentence for a refused launch). The protocol is the seam a second
  provider plugs into as one module plus a credentials section; it is
  sized for AWS and the manual case, and corrected by the second provider
  rather than designed for it.
- **Machine state**, observed each reconcile pass from one `describe` per
  provider (cached for `OBSERVATION_TTL_SECONDS`) plus the ssh probes the
  slots already do: `launching` (instance requested; becomes `up` when ssh
  answers, cloud-init has finished -- a marker file its script writes last
  -- and both worker images are present), `up`, `stopping` / `stopped`
  (suspended; disk kept), `gone` (terminated). A manual machine is `up`
  or `unreachable`.

## Lifecycle rules

- **Machines before slots.** The reconcile pass gains a machine step ahead
  of the per-slot loop: observe each machine, apply the idle policy, and
  start a stopped instance that a slot wants. Slot enforcement on a
  machine that is not `up` is skipped (its probes would all read
  `unreachable`), and `set_worker_state` routes a Start on a slot of a
  stopped machine into the machine step rather than probing a container
  that cannot answer.
- **Launch.** From the task's add-worker form ("rent a machine": instance
  type from the catalog). The provider launches on the region's current
  AWS Deep Learning Base GPU AMI (Ubuntu; NVIDIA driver, Docker and the
  container toolkit preinstalled, looked up through its public SSM
  parameter, so there is no image of ours to build), with our key pair and
  security group, tagged `scribblez=<task>/<machine>`, and a cloud-init
  script that logs Docker in to the image registry with a read-only token
  from the credentials file, pulls both worker images, and writes the
  ready marker. The pull happens on the machine during `launching`, not on
  the controller's blocking thread at the first slot start; at slot start
  it is a no-op.
- **Slots on it.** Exactly the ssh kind: the container per slot, the
  bundle by the machine's arch, gates as pauses, redeploy as drained
  replacement, results collected over ssh -- except the trainer, below.
  `_check_role` consults the machine's catalog entry for slots on a
  rented machine: a GPU role on a CPU type, or a second GPU role on a
  one-GPU type, is refused at add time rather than at `docker run` on the
  remote. A GPU machine hosts the trainer and a generator on its vCPUs.
- **Finished slots.** A container that exits 0 has reached its role's
  terminal condition (the trainer's `max_rows`, a generator's cycle cap);
  reconcile marks the slot `finished` (desired state `paused`, shown as
  such) instead of restarting it. The worker entrypoint's SIGTERM flush
  exits non-zero so a machine reboot is not mistaken for completion.
- **Idle policy.** A machine none of whose containers has been observed
  running for `IDLE_STOP_SECONDS` (10 minutes), with no slot in a pending
  start (restart backoff), is stopped. A gated generator is a paused
  container and a finished trainer an exited one, so a run that ends stops
  its machine; a scheduler gate on a machine whose trainer is running does
  not. A Start on a slot of a stopped machine starts the instance (its
  address may change; `describe` refreshes `host`) and proceeds when it is
  `up`. Stopped costs the volume only (a 60 GB gp3 root: about $5 a
  month, prorated).
- **Terminate** is explicit: Remove on the machine's row in the workers
  table. On an `up` or `stopped` machine it is refused while a slot is
  running or still holding output (the rule for removing an ssh slot); on
  a `gone` machine the slots are removable unconditionally, and the dialog
  says their undelivered output went with the disk. Removing a machine
  removes its slots. On terminate the machine's spend rolls into
  `task.retired_spend`, as a removed slot's does.
- **Orphans.** Each pass lists the region's instances tagged `scribblez`;
  one no task's machines name is listed on the Overview as an orphan with
  its type, uptime and a Terminate button. It is not terminated
  automatically: a task.json restored from an older copy must not kill a
  running experiment, and one click is cheap.
- **Spend.** Per machine: `cost_per_hr` from the catalog row, accrued
  while `launching` or `up`, kept across dashboard restarts. The workers
  table shows each machine's rate, uptime and spend as a row above its
  slots; the task's spend line adds its machines, and its retired spend
  after they go.
- **Refusals.** A launch AWS refuses -- the on-demand vCPU quota for the G
  family (zero on a new account), no capacity for the type in the zone, a
  bad credential -- is shown as a sentence saying what to do, with the
  console link for a quota, under the existing launch backoff.

## The trainer on an ssh machine

The trainer is the one role with inputs. Generations reach it, and its
exports, checkpoint and records leave it, through its sink. On an ssh
machine that is the R2 sink: R2 egress is free, the machine has a
datacenter link, and the controller's bucket legs (publish,
`--trainer-outputs` sync, controls push) already exist. `train` gains
`ssh` in its kinds and loses `cloud`. Nothing in the trainer changes;
`cloud_sync` and the ingest tick do not know which machine the records
came from.

The controller changes one predicate, in four places. Today
`BUCKET_KINDS = ("cloud",)` gates the sync watcher's existence
(`_ensure_sync`), the `--trainer-outputs` flag and the controls push
(`_bucket_trainer`), and the scheduler's publish and mirror hooks
(`_make_publish`, `_make_mirror`). All four become one predicate, **a
remote trainer**: a slot of a non-local kind whose role has `ingest`.
Written in terms of roles rather than kinds, it survives the deletion of
`cloud`. The ssh container's environment gets `SCZ_SINK=r2` for a role
with `ingest` (every other role keeps `local`, collected over ssh), and
`_collect_ssh` skips such a slot, whose outputs are not on the machine to
collect.

## One-time setup (the operator, once)

1. An AWS account. In IAM, a user `scribblez` with an access key and one
   policy: `ec2:RunInstances, DescribeInstances, DescribeInstanceTypes,
   StartInstances, StopInstances, TerminateInstances, CreateTags,
   CreateKeyPair, DescribeKeyPairs, CreateSecurityGroup,
   AuthorizeSecurityGroupIngress, DescribeSecurityGroups` and
   `ssm:GetParameter` (the AMI lookup). The key goes in the credentials
   file under a new `aws` section with the region.
2. Quotas, on day one, since the grant takes a day or two: "Running
   On-Demand G and VT instances" (zero on a new account; request 64
   vCPUs), and "All G and VT Spot Instance Requests" at the same time.
3. A budget alert in the AWS console: belt-and-braces against any bug of
   ours.
4. A read-only Docker Hub access token for the worker image repo, in a new
   `registry.pull_token` (with `registry.username`) in the credentials
   file, validated by the check script; it is what a rented machine's
   cloud-init logs in with.
5. `py/scripts/aws_setup.py` (idempotent): creates the key pair (private
   key to `/workspace/mount/cloud/aws/`), the security group (port 22 from
   anywhere; key-only auth), and verifies the AMI lookup and the quota
   values, reporting what is still missing.

## Catalog

Curated, in code, with the on-demand list price beside the type (dated;
changed by PR when AWS reprices -- the spend figure is an estimate), and
the CPU family that picks the bundle arch:

| Type | vCPU | GPU | Family (arch) |
|---|---|---|---|
| c7a.4xlarge / 8xlarge | 16 / 32 | -- | Genoa (`znver4`) |
| g6.2xlarge / 4xlarge / 8xlarge | 8 / 16 / 32 | L4 24 GB | Milan (`znver3`) |

`znver3` joins `SUPPORTED_ARCHS`. Other types (g5's A10G, g6e's L40S)
are one row each when a run wants them.

## Not in this slice

- Spot (m3b, below): the price at launch from the instance's actual zone,
  the `gone`-by-interruption state, offering it only for interruptible
  roles.
- A custom AMI with the images baked in (cloud-init's pull is a few
  minutes of `launching` per fresh machine; a stopped machine keeps them).
- Automatic relaunch after a spot interruption; automatic placement or an
  autoscaler for generators; machines shared across tasks.
- A trainer chunk cache on the machine (its companion generator's chunks
  go to the controller over ssh and come back through the bucket; ~12 MB a
  generation, hidden by `open_ahead`).
- Any provider beyond AWS.

## Sequence

0. **Day one (operator):** the account, IAM user, both quota requests, the
   budget alert, the Docker Hub token. Host-side: rebuild the dev image
   with boto3 (an m3 prerequisite).
1. **PR m1 -- machines on the ssh kind.** `task.machines` and the
   `machine` field on ssh slots; `SshMachine(host, identity_file,
   known_hosts_file)` built only by `_ssh(task, w)`; the machine row in
   the workers table and the machine selector in the ssh form; "register
   a machine" (manual: name, host, key) on the form; the GPU fit check for
   slots on a machine with a known type; the machine step of the reconcile
   pass for manual machines (`up` / `unreachable`); the `finished` slot
   state and the entrypoint's exit codes. No provider yet.
2. **PR m2 -- the trainer on ssh.** The remote-trainer predicate at its
   four sites, the per-role sink, `train` kinds, `_collect_ssh` skipping
   the trainer. Tests: a task with a generator and a trainer both of ssh
   kind and no `cloud` slot gets its sync watcher with
   `--trainer-outputs`, its publish and mirror hooks, and its controls
   push; a local-only task gets none.
3. **The premise, measured by hand (operator + agent).** After the quota
   grant: launch one g6.4xlarge from the console following a recipe
   written for the purpose (the Deep Learning Base GPU AMI, the key pair,
   port 22; prepare it as master_dashboard.md's "SSH worker machines"
   says), register it as a manual machine of a test tag, and run a trainer
   and a generator on it through m1+m2. Outputs: the trainer's step on an
   L4 against the 4090's (the transformer trainer is launch/bandwidth
   bound, and an L4 has about a third of a 4090's memory bandwidth), its
   wait fraction, the generator's per-chunk time on the shared cores, and
   the cost line. Whether the stock AMI runs the `-torch` image under
   `--gpus all` and whether the dev container reaches a public address
   with the ssh options we hardcode are learned here too, before the
   provider exists. Terminate it from the console afterwards.
4. **PR m3 -- the AWS provider.** boto3, the credentials sections and the
   check script, `aws_setup.py`, `providers/base.py` and `aws.py`, the
   catalog, launch with the cloud-init script and readiness marker,
   machine states and the idle policy in the reconcile pass, orphans,
   spend and its roll-up, refusals, the rent form. On-demand only.
   Verified by the first dashboard-rented run: the same trainer-plus-
   generator shape as step 3, now launched, idled and terminated by the
   dashboard.
5. **PR m3b -- spot.** Offered when every role being placed is
   interruptible; the price at launch from the instance's zone; `gone` by
   interruption, with the slots' removal rule above.
6. **PR m4 -- Runpod out.** After m3's run has trained a generation: the
   `cloud` kind, `runpod_api.py`, `cloud_fleet.py`'s pod commands, the pod
   paths in the worker image's bootstrap (the container flow stays; it is
   what ssh containers run), the Runpod credential section and check, the
   GPU/CPU pod forms, and the Runpod paragraphs of the docs.

m1 < m2 < 3 < m3 < m3b; m4 after m3's run.
