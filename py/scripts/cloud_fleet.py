#!/usr/bin/env python3
"""Launch and manage the Runpod CPU worker fleet.

    up      launch N workers running a workload role against a tag
    status  list fleet pods (and, with --tag, the tag's bucket progress)
    down    terminate fleet pods

Workers run the worker image (build_and_push_worker_image.py) and pull their
code from the bundle named by --bundle (see cloud_push_binaries.py); "latest"
resolves to a concrete bundle_id at launch so every pod in one `up` runs
identical code even if newer bundles are pushed meanwhile. The workload, its
role, and its parameter flags come from the workload registry
(scribblez/workloads/); pods are rented interruptible when the role tolerates
preemption.

Pods are named scz-<tag>-<suffix>, which is how status/down recognize fleet
pods; pods created any other way are never touched. The master dashboard
manages workers through the same pod_create_spec, so pods are identical
however launched.

Usage:
    ./py/scripts/cloud_fleet.py up -n 4 --vcpus 16 -t hello
    ./py/scripts/cloud_fleet.py status [-t hello]
    ./py/scripts/cloud_fleet.py down [-t hello | --all] [-y]
"""

import argparse
import secrets
import sys
from dataclasses import dataclass

from cloud.bundles import resolve_bundle_id
from cloud.credentials import CloudCredentials, load_credentials
from cloud.r2 import bucket_path, rclone
from cloud.runpod_api import RunpodClient
from scribblez import params as params_mod
from scribblez import workloads
from util.argparse_ext import ArgumentDefaultsHelpFormatter

POD_NAME_PREFIX = "scz-"


def fleet_pods(client: RunpodClient, tag: str | None) -> list[dict]:
    prefix = POD_NAME_PREFIX + (f"{tag}-" if tag else "")
    return [p for p in client.list_pods() if p.get("name", "").startswith(prefix)]


def r2_env(creds: CloudCredentials) -> dict[str, str]:
    return {
        "R2_ACCOUNT_ID": creds.r2.account_id,
        "R2_ACCESS_KEY_ID": creds.r2.access_key_id,
        "R2_SECRET_ACCESS_KEY": creds.r2.secret_access_key,
        "R2_BUCKET": creds.r2.bucket,
    }


@dataclass(frozen=True)
class CpuResources:
    """A CPU pod's hardware: a Runpod CPU flavor id and a vCPU count."""

    vcpus: int
    flavor: str


@dataclass(frozen=True)
class GpuResources:
    """A GPU pod's hardware: a Runpod gpuTypeId and how many of them."""

    gpu_type_id: str
    gpu_count: int


def _compute_fields(resources: CpuResources | GpuResources) -> dict:
    """The compute-selection part of a POST /pods body: a CPU flavor + vCPU
    count, or a GPU type + count."""
    if isinstance(resources, GpuResources):
        return {
            "computeType": "GPU",
            "gpuTypeIds": [resources.gpu_type_id],
            "gpuCount": resources.gpu_count,
        }
    return {"computeType": "CPU", "cpuFlavorIds": [resources.flavor], "vcpuCount": resources.vcpus}


def pod_create_spec(
    creds: CloudCredentials,
    spec: workloads.WorkloadSpec,
    tag: str,
    params,
    *,
    role: str,
    bundle_id: str,
    resources: CpuResources | GpuResources,
    container_disk_gb: int = 20,
) -> tuple[str, dict]:
    """(pod name, POST /pods body) for one cloud worker. Shared by this CLI and
    the dashboard's WorkerManager so pods are identical however launched."""
    name = f"{POD_NAME_PREFIX}{tag}-{secrets.token_hex(3)}"
    env = {
        **r2_env(creds),
        **spec.worker_env(tag, params, role),
        "SCZ_BUNDLE": bundle_id,
        "SCZ_WORKER_ID": name,
    }
    return name, {
        "name": name,
        "imageName": creds.registry.worker_image,
        **_compute_fields(resources),
        "containerDiskInGb": container_disk_gb,
        "containerRegistryAuthId": creds.runpod.container_registry_auth_id,
        "interruptible": spec.role(role).interruptible,
        "env": env,
    }


def cmd_up(creds: CloudCredentials, client: RunpodClient, args) -> int:
    spec = workloads.get(args.workload)
    params = params_mod.from_args(spec.params_cls, args)
    bundle_id = resolve_bundle_id(creds.r2, args.bundle)
    print(f"Launching {args.num_workers} worker(s) on bundle {bundle_id} ...")
    for _ in range(args.num_workers):
        name, body = pod_create_spec(
            creds,
            spec,
            args.tag,
            params,
            role=args.role,
            bundle_id=bundle_id,
            resources=CpuResources(vcpus=args.vcpus, flavor=args.flavor),
            container_disk_gb=args.container_disk_gb,
        )
        pod = client.create_pod(body)
        cost = pod.get("costPerHr")
        print(f"  {name}  id={pod.get('id')}  ${cost}/hr")
    print(f"Fleet is launching. Watch with: cloud_fleet.py status -t {args.tag}")
    return 0


def tag_bucket_stats(creds: CloudCredentials, spec: workloads.WorkloadSpec, tag: str):
    for sub in spec.sync_data_dirs:
        res = rclone(creds.r2, "lsf", bucket_path(creds.r2, spec.name, tag, sub), capture=True)
        names = res.stdout.split()
        if sub == "slogs":
            pairs = sum(1 for n in names if n.endswith(".sobs"))
            print(f"\nBucket tag '{tag}' {sub}/: {pairs} complete pair(s)")
        else:
            print(f"\nBucket tag '{tag}' {sub}/: {len(names)} file(s)")


def cmd_status(creds: CloudCredentials, client: RunpodClient, args) -> int:
    pods = fleet_pods(client, args.tag)
    if not pods:
        print("No fleet pods.")
    else:
        total_cost = 0.0
        for p in pods:
            cost = p.get("costPerHr") or 0.0
            total_cost += cost
            print(
                f"  {p.get('name'):<28} id={p.get('id')}  "
                f"{p.get('desiredStatus'):<8} {p.get('vcpuCount')} vcpu  ${cost}/hr"
            )
        print(f"{len(pods)} pod(s), ${total_cost:.3f}/hr total")
    if args.tag:
        tag_bucket_stats(creds, workloads.get(args.workload), args.tag)
    return 0


def cmd_down(creds: CloudCredentials, client: RunpodClient, args) -> int:
    if not args.tag and not args.all:
        print("down: pass -t <tag> to terminate one tag's pods, or --all for every scz- pod")
        return 1
    pods = fleet_pods(client, args.tag)
    if not pods:
        print("No matching fleet pods.")
        return 0
    for p in pods:
        print(f"  {p.get('name')}  id={p.get('id')}  {p.get('desiredStatus')}")
    if not args.yes and input(f"Terminate these {len(pods)} pod(s)? [y/N] ").lower() != "y":
        print("Aborted.")
        return 1
    for p in pods:
        client.delete_pod(p["id"])
        print(f"  terminated {p.get('name')}")
    return 0


def main() -> int:
    # --workload picks the params dataclass whose flags `up` accepts, so it is
    # pre-parsed before the full parser is built.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--workload", choices=sorted(workloads.WORKLOADS), default="kill_test")
    known, _ = pre.parse_known_args()
    spec = workloads.get(known.workload)
    cloud_roles = [r.name for r in spec.roles if "cloud" in r.kinds]

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--workload",
        choices=sorted(workloads.WORKLOADS),
        default="kill_test",
        help="workload the workers run (registry: scribblez/workloads/)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    up = sub.add_parser("up", help="launch workers", formatter_class=ArgumentDefaultsHelpFormatter)
    up.add_argument("-n", "--num-workers", type=int, default=1)
    up.add_argument("-t", "--tag", required=True, help="run tag (the workload's data tag)")
    up.add_argument("--role", choices=cloud_roles, default=cloud_roles[0], help="worker role")
    up.add_argument("--vcpus", type=int, default=16, help="vCPUs per worker pod")
    up.add_argument("--flavor", default="cpu3c", help="Runpod CPU flavor id")
    up.add_argument("--container-disk-gb", type=int, default=20)
    up.add_argument("--bundle", default="latest", help='bundle_id or "latest"')
    params_mod.add_arguments(up, spec.params_cls)
    up.set_defaults(func=cmd_up)

    status = sub.add_parser("status", help="list fleet pods")
    status.add_argument(
        "-t", "--tag", default=None, help="restrict to one tag; also show bucket progress"
    )
    status.set_defaults(func=cmd_status)

    down = sub.add_parser("down", help="terminate fleet pods")
    down.add_argument("-t", "--tag", default=None)
    down.add_argument("--all", action="store_true", help="terminate every scz- pod")
    down.add_argument("-y", "--yes", action="store_true", help="skip confirmation")
    down.set_defaults(func=cmd_down)

    args = parser.parse_args()
    creds = load_credentials()
    client = RunpodClient(creds.runpod.api_key)
    return args.func(creds, client, args)


if __name__ == "__main__":
    sys.exit(main())
