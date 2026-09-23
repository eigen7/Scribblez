# Plan: rented machines -- AWS instances as ssh machines

**Status: landed in full.** m1 machines on the ssh kind (#199), m2 the
trainer on ssh (#200), m3 the AWS provider (#201), m3b spot (#213), m4
Runpod removed (#214); slots on rented machines deliver through the bucket
(#211). The first dashboard-rented run trained on 2026-09-16. The provider
lives in `py/cloud/providers/`, the machine lifecycle in
`py/scribblez/dashboard/workers.py`, and the operator's view in
[cloud_compute.md](../cloud_compute.md).

**Decision.** A task can **rent a machine**: an AWS instance, launched from a
stock GPU image, reached over ssh, hosting that task's slots of the existing
`ssh` worker kind. The `cloud` kind, Runpod and the pod-side bootstrap go.

This plan follows [cloud_training.md](cloud_training.md), keeping its
trainer I/O contract and bucket legs (3a to 3c) whole and replacing its
Runpod cloud slot.

## Why

Runpod pods had properties that made them the wrong long-term shape for the
worker fleet, independent of any bug:

- A stopped pod is pinned to its host, bills for its disk, and cannot start
  again once the host has filled. A pause has to be stop-and-replace, and any
  error in that path rents a fresh pod.
- Whole instance flavors go out of stock for hours at a time.
- A host's CPU family is unknown until a pod lands on it, so a bundle may run
  the generic build for a day before anyone notices.
- A pod cannot run Docker, so colocating a generator with the trainer needs a
  process supervisor and a control plane routed through the bucket, which
  lags by construction.

An AWS instance has none of these. It stops and starts on any host (its disk
survives and costs cents while stopped), its CPU family follows from the
instance type, and it is a machine, ssh-reachable and running Docker, which is
exactly what the `ssh` kind already drives: pause is a Docker pause, a
scheduler gate parks a container, redeploy replaces a container once it has
drained, and liveness is a probe over the link rather than a listing through a
vendor API.

The dashboard, not the operator, owns the instance lifecycle. That is what
lets it enforce the properties that keep money safe: every instance is tagged
and listed on every pass, an idle machine is stopped, an unknown instance is
shown rather than silently billed, and spend is tracked per machine and rolls
into the task's total when the machine goes.

## Concepts

- **Machine.** A record in the task's `task.json` (`task.machines`). A
  machine is rented by one task for its own slots, and the record lives and
  dies with the task, as its slots do. Fields: `name`, `provider` (`aws` or
  `manual`), `host` (`user@address`, what ssh is given), `identity_file` and
  `known_hosts_file`, `arch`, `gpu`, and for a rented machine `instance_id`,
  `instance_type`, `region`, `cost_per_hr`, `launched_at` and `spend`. A
  rented machine's key material lives under
  `/workspace/mount/cloud/machines/<workload>/<tag>/<name>/` (machine names
  are unique only within a task). A **manual** machine is one the operator
  launched or owns and registered by name and host; it may leave the key
  fields unset and use the container's own identity. Slots addressed by a
  bare host string (`WorkerRecord.host`) keep working unchanged.
- **Slot.** The `ssh` kind, unchanged in behavior. A slot on a machine carries
  `machine` (the name) instead of `host`; the add-worker form offers the
  task's machines as well as a free host string.
- **One ssh helper.** `_ssh_machine(task, w)` in `dashboard/workers.py` is the
  only place an `SshMachine` is built. It resolves a slot's machine to host,
  identity file and known-hosts file; a bare-host slot passes the host alone.
- **Host keys.** A rented machine's host key is unknown at launch, and AWS
  reuses addresses, so the global known_hosts is wrong twice over. Each rented
  machine gets its own known-hosts file, created empty at launch and used with
  `StrictHostKeyChecking=accept-new`. A relaunch is a new machine name, so a
  reused address never meets a stale key.
- **Provider.** `py/cloud/providers/aws.py`, behind a small protocol in
  `base.py`: `catalog()`, `launch(request)`, `describe()`, `stop`, `start`,
  `terminate`, and `refusal(error, type_id)` (the operator-facing sentence
  for a refused launch). The protocol is the seam a second provider would plug
  into as one module plus a credentials section. It is sized for AWS and the
  manual case, to be corrected by the second provider rather than designed
  for it in advance.
- **Machine state**, observed each reconcile pass from one `describe` per
  provider (cached for `OBSERVATION_TTL_SECONDS`) plus the ssh probes the
  slots already make:
  - `launching`: the instance is requested. It becomes `up` when ssh answers,
    cloud-init has finished (its script writes a ready marker last), and both
    worker images are present.
  - `up`.
  - `stopping` / `stopped`: suspended, disk kept.
  - `gone`: terminated.

  A manual machine is `up` or `unreachable`.

## Lifecycle rules

- **Machines before slots.** The reconcile pass runs a machine step ahead of
  the per-slot loop: observe each machine, apply the idle policy, and start a
  stopped instance that a slot wants. Slot enforcement on a machine that is
  not `up` is skipped (its probes would all read `unreachable`), and a Start
  on a slot of a stopped machine is routed to the machine step rather than
  probing a container that cannot answer.
- **Launch.** From the task's add-worker form ("rent a machine", with the
  instance type chosen from the catalog). The provider launches on the
  region's current AWS Deep Learning Base GPU AMI (Ubuntu, with the NVIDIA
  driver, Docker and the container toolkit preinstalled), found through its
  public SSM parameter, so there is no image of ours to build. The instance
  gets our key pair and security group, the tag `scribblez=<task>/<machine>`,
  and a cloud-init script that logs Docker in to the image registry with a
  read-only token from the credentials file, pulls both worker images, and
  writes the ready marker. The pull happens on the machine while it is
  `launching`, not on the controller's blocking thread at the first slot
  start.
- **Slots on it** behave exactly like the ssh kind: a container per slot, the
  bundle chosen by the machine's arch, gates as pauses, redeploy as a drained
  replacement. Two differences: slots on a rented machine deliver through the
  bucket (below), and the role check at add time consults the machine's
  catalog entry, so a GPU role on a CPU type, or a second GPU role on a
  one-GPU type, is refused when the slot is added rather than at `docker run`
  on the remote. A GPU machine can host the trainer and a generator on its
  vCPUs.
- **Finished slots.** A container that exits 0 has reached its role's
  terminal condition (the trainer's `max_rows`, a generator's cycle cap).
  Reconcile marks the slot finished (desired state `paused`, shown as such)
  instead of restarting it. The worker entrypoint exits non-zero after a
  SIGTERM flush, so a machine reboot is not mistaken for completion.
- **Idle policy.** A machine on which no container has been observed running
  for `IDLE_STOP_SECONDS` (10 minutes), and with no slot in a pending start
  (restart backoff), is stopped. A gated generator is a paused container and
  a finished trainer an exited one, so a run that ends stops its machine; a
  scheduler gate does not, while the machine's trainer is running. A Start on
  a slot of a stopped machine starts the instance (its address may change;
  `describe` refreshes `host`) and proceeds once it is `up`. A stopped
  machine costs only its volume: a 100 GB gp3 root (the image's snapshot is
  75 GB), about $8 a month, prorated.
- **Terminate** is explicit: Remove on the machine's row in the workers
  table. On an `up` or `stopped` machine it is refused while a slot is
  running or still holding output (the rule for removing an ssh slot). On a
  `gone` machine its slots can be removed unconditionally, and the dialog says
  their undelivered output went with the disk. Removing a machine removes its
  slots, and its spend rolls into `task.retired_spend`, as a removed slot's
  does.
- **Orphans.** Each pass lists the region's instances tagged `scribblez`. One
  that no task's machine names is shown on the Overview as an orphan, with
  its type, uptime and a Terminate button. It is never terminated
  automatically: a `task.json` restored from an older copy must not kill a
  running experiment, and one click is cheap.
- **Spend.** Per machine: `cost_per_hr` from the catalog (on-demand) or the
  zone's spot price at launch, accrued while `launching` or `up`, and kept
  across dashboard restarts. The workers table shows each machine's rate,
  uptime and spend in a row above its slots; the task's spend line adds its
  machines and, after they go, its retired spend.
- **Refusals.** A launch AWS refuses (the on-demand vCPU quota for the G
  family, zero on a new account; no capacity for the type in the zone; a bad
  credential) is shown as a sentence saying what to do, with the console link
  for a quota, under the existing launch backoff.

## Where a slot delivers

The trainer is the one role with inputs: generations reach it, and its
exports, checkpoint and records leave it, through its sink. On a rented
machine that sink is the bucket (R2): R2 egress is free, the machine has a
datacenter link, and the controller's bucket legs from cloud_training.md
(publish, the `--trainer-outputs` sync, the controls push) already existed.
The `train` role runs on the `ssh` kind; nothing in the trainer changed, and
neither `cloud_sync` nor the ingest tick knows which machine the records came
from.

One predicate, `_slot_sink(spec, task, w)`, decides where a slot delivers:

- **Through the bucket** for a slot whose role has an `ingest` hook (a
  trainer), wherever it runs; and for every slot on a rented machine, since
  collecting over ssh would haul each chunk to the controller and publish it
  back up from a home uplink.
- **Over the control link** (collected over ssh) for any other slot on the
  operator's own machine.
- **Locally** for a local slot.

Everything the controller does for a bucket-delivering slot keys off this
predicate, not off the worker kind: whether the sync watcher runs, the
`--trainer-outputs` flag and the controls push, and the scheduler's publish
and mirror hooks. The container's `SCZ_SINK` comes from the same predicate,
and ssh collection skips a bucket-delivering slot, whose outputs are not on
the machine to collect. Written in terms of roles and machines rather than
kinds, the predicate survived the deletion of `cloud`.

## One-time setup (the operator, once)

1. **An AWS account.** In IAM, a user `scribblez` with an access key and the
   policy below, pasted as is into the console's policy editor. It is exactly
   what `py/cloud/providers/aws.py` and `py/scripts/aws_setup.py` call, plus
   the right to create the EC2 Spot service-linked role, which AWS creates on
   the account's first spot request and refuses to when the caller lacks the
   right. A missing action does not always fail loudly: a spot price the user
   may not read is silently replaced by the catalog's on-demand rate.

   ```json
   {
     "Version": "2012-10-17",
     "Statement": [
       {
         "Effect": "Allow",
         "Action": [
           "ec2:RunInstances",
           "ec2:DescribeInstances",
           "ec2:StartInstances",
           "ec2:StopInstances",
           "ec2:TerminateInstances",
           "ec2:CreateTags",
           "ec2:CreateKeyPair",
           "ec2:DescribeKeyPairs",
           "ec2:CreateSecurityGroup",
           "ec2:AuthorizeSecurityGroupIngress",
           "ec2:DescribeSecurityGroups",
           "ec2:DescribeSpotPriceHistory",
           "ec2:CancelSpotInstanceRequests",
           "ssm:GetParameter",
           "servicequotas:GetServiceQuota",
           "servicequotas:ListRequestedServiceQuotaChangeHistory"
         ],
         "Resource": "*"
       },
       {
         "Effect": "Allow",
         "Action": "iam:CreateServiceLinkedRole",
         "Resource": "*",
         "Condition": {
           "StringEquals": { "iam:AWSServiceName": "spot.amazonaws.com" }
         }
       }
     ]
   }
   ```

   The key goes in the credentials file under an `aws` section, with the
   region.
2. **Quotas, on day one**, since a grant takes a day or two: "Running
   On-Demand G and VT instances" (zero on a new account; request 64 vCPUs) and
   "All G and VT Spot Instance Requests".
3. **A budget alert** in the AWS console, as a backstop against any bug of
   ours.
4. **A read-only Docker Hub access token** for the worker image repo, as
   `registry.pull_token` (with `registry.username`) in the credentials file,
   validated by `py/scripts/cloud_check_credentials.py`. A rented machine's
   cloud-init logs in with it.
5. **`py/scripts/aws_setup.py`** (idempotent): creates the key pair (private
   key under `/workspace/mount/cloud/aws/`) and the security group (port 22
   from anywhere, key-only auth), checks the AMI lookup and the quota values,
   and reports what is still missing.

## Catalog

Curated in code (`CATALOG` in `aws.py`), with each type's on-demand list
price beside it (dated, and changed by PR when AWS reprices, so the spend
figure is an estimate) and the CPU family that picks the bundle arch:

| Type | vCPU | GPU | Family (arch) |
|---|---|---|---|
| c7a.xlarge / 2xlarge / 4xlarge / 8xlarge | 4 / 8 / 16 / 32 | -- | Genoa (`znver4`) |
| g6.2xlarge / 4xlarge / 8xlarge | 8 / 16 / 32 | L4 24 GB | Milan (`znver3`) |

Bundles are built for the archs a task's machines report. Other types (g5's
A10G, g6e's L40S) are one row each when a run wants them.

At the first run the account's on-demand G-family quota was 8 vCPUs (a
request for 64 was an open support case), so it trained on a g6.2xlarge with
a generator on a separate c7a; the 16-vCPU sizes that colocate both waited on
the grant. **Measured on that run (2026-09-16):** the L4 trains about 560
rows/s against a 4090's 800, and a full-window generation consumes about 120
games/s, which a c7a.xlarge's 4 vCPUs supply at about 45 games/s each.

## Spot

Spot landed in m3b as a *persistent, stop-on-interruption* request. The
instance's disk survives an interruption, AWS restarts it when capacity
returns, and stop, start and the idle policy work as for on-demand, so the
machine model is unchanged and spot is offered for any role. A trainer loses
at most its in-flight generation. The rate charged is the zone's spot price
at launch, and the rent form shows current rates.

## Not built

- A custom AMI with the images baked in (cloud-init's pull costs a few
  minutes of `launching` per fresh machine; a stopped machine keeps them).
- Automatic relaunch after a spot interruption; automatic placement or an
  autoscaler for generators; machines shared across tasks.
- A trainer chunk cache on the machine. A companion generator's chunks go to
  the controller and come back through the bucket: about 12 MB a generation,
  hidden by `open_ahead`.
- Any provider beyond AWS.

## Sequence, as executed

0. **Day one (operator):** the account, the IAM user, both quota requests,
   the budget alert, the Docker Hub token. Host side: rebuild the dev image
   with boto3 (an m3 prerequisite).
1. **m1, machines on the ssh kind.** `task.machines` and the `machine` field
   on ssh slots; `SshMachine(host, identity_file, known_hosts_file)`, built
   only through the one helper; the machine row in the workers table and the
   machine selector in the ssh form; "register a machine" (manual: name, host,
   key); the GPU fit check for slots on a machine of known type; the machine
   step of the reconcile pass for manual machines (`up` / `unreachable`); the
   finished slot state and the entrypoint's exit codes. No provider.
2. **m2, the trainer on ssh.** The delivery predicate at its four sites, the
   per-slot sink, the `train` role's kinds, ssh collection skipping the
   trainer. Tests: a task with a generator and a trainer, both ssh and no
   `cloud` slot, gets its sync watcher with `--trainer-outputs`, its publish
   and mirror hooks, and its controls push; a local-only task gets none of
   them.
3. **m3, the AWS provider.** boto3, the credentials sections and the check
   script, `aws_setup.py`, `providers/base.py` and `aws.py`, the catalog,
   launch with the cloud-init script and ready marker, machine states and the
   idle policy in the reconcile pass, orphans, spend and its roll-up,
   refusals, the rent form. On-demand only. Verified by the first
   dashboard-rented run, which also measured the premise: a trainer on a
   g6.2xlarge and a generator on a c7a, launched, idled and terminated by the
   dashboard; the trainer's step on an L4 against a 4090's (the transformer
   trainer is launch- and bandwidth-bound, and an L4 has about a third of a
   4090's memory bandwidth); its wait fraction and cost; whether the stock AMI
   runs the `-torch` image under `--gpus all`; and whether the dev container
   reaches a public address with the ssh options it uses.
4. **m3b, spot** (above).
5. **m4, Runpod out**, after m3's run had trained a generation: the `cloud`
   kind, the Runpod API client, the pod commands of the fleet script, the pod
   paths in the worker image's bootstrap (the container flow stayed; it is
   what ssh containers run), the Runpod credential section and check, the GPU
   and CPU pod forms, and the Runpod paragraphs of the docs.

Order: m1, m2, m3, m3b; m4 after m3's run.
