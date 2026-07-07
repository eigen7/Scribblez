#!/usr/bin/env python3
"""Launch and manage the Runpod CPU worker fleet.

    up      launch N workers running a workload against a tag
    status  list fleet pods (and, with --tag, the tag's bucket progress)
    down    terminate fleet pods

Workers run the worker image (build_and_push_worker_image.py) and pull their code from
the bundle named by --bundle (see cloud_push_binaries.py); "latest" resolves
to a concrete bundle_id at launch so every pod in one `up` runs identical
code even if newer bundles are pushed meanwhile.

Pods are named scz-<tag>-<suffix>, which is how status/down recognize fleet
pods; pods created any other way are never touched.

Usage:
    ./py/scripts/cloud_fleet.py up -n 4 --vcpus 16 -t hello
    ./py/scripts/cloud_fleet.py status [-t hello]
    ./py/scripts/cloud_fleet.py down [-t hello | --all] [-y]
"""

import argparse
import secrets
import sys

from cloud.bundles import resolve_bundle_id
from cloud.credentials import CloudCredentials, load_credentials
from cloud.r2 import bucket_path, rclone
from cloud.runpod_api import RunpodClient
from scripts.generate_kill_test_data import KillTestParams
from util.argparse_ext import ArgumentDefaultsHelpFormatter

POD_NAME_PREFIX = "scz-"


def fleet_pods(client: RunpodClient, tag: str | None) -> list[dict]:
    prefix = POD_NAME_PREFIX + (f"{tag}-" if tag else "")
    return [p for p in client.list_pods() if p.get("name", "").startswith(prefix)]


def worker_env(creds: CloudCredentials, args, bundle_id: str, name: str) -> dict[str, str]:
    env = {
        "R2_ACCOUNT_ID": creds.r2.account_id,
        "R2_ACCESS_KEY_ID": creds.r2.access_key_id,
        "R2_SECRET_ACCESS_KEY": creds.r2.secret_access_key,
        "R2_BUCKET": creds.r2.bucket,
        "SCZ_BUNDLE": bundle_id,
        "SCZ_WORKLOAD": "kill_test",
        "SCZ_TAG": args.tag,
        "SCZ_GAMES_PER_BATCH": str(args.games_per_batch),
        "SCZ_ROLLOUTS": str(args.rollouts),
        "SCZ_TOP_K": str(args.top_k),
        "SCZ_POSITIONS_PER_GAME": str(args.positions_per_game),
        "SCZ_WORKER_ID": name,
    }
    if args.open_leaves:
        env["SCZ_OPEN_LEAVES"] = "1"
    return env


def cmd_up(creds: CloudCredentials, client: RunpodClient, args) -> int:
    bundle_id = resolve_bundle_id(creds.r2, args.bundle)
    print(f"Launching {args.num_workers} worker(s) on bundle {bundle_id} ...")
    for _ in range(args.num_workers):
        name = f"{POD_NAME_PREFIX}{args.tag}-{secrets.token_hex(3)}"
        pod = client.create_pod(
            {
                "name": name,
                "imageName": creds.registry.worker_image,
                "computeType": "CPU",
                "cpuFlavorIds": [args.flavor],
                "vcpuCount": args.vcpus,
                "containerDiskInGb": args.container_disk_gb,
                "containerRegistryAuthId": creds.runpod.container_registry_auth_id,
                "env": worker_env(creds, args, bundle_id, name),
            }
        )
        cost = pod.get("costPerHr")
        print(f"  {name}  id={pod.get('id')}  ${cost}/hr")
    print(f"Fleet is launching. Watch with: cloud_fleet.py status -t {args.tag}")
    return 0


def tag_bucket_stats(creds: CloudCredentials, tag: str):
    res = rclone(creds.r2, "lsf", bucket_path(creds.r2, "kill_test", tag, "slogs"), capture=True)
    names = res.stdout.split()
    pairs = sum(1 for n in names if n.endswith(".sobs"))
    print(f"\nBucket tag '{tag}': {pairs} complete pair(s)")


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
        tag_bucket_stats(creds, args.tag)
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
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    up = sub.add_parser("up", help="launch workers", formatter_class=ArgumentDefaultsHelpFormatter)
    up.add_argument("-n", "--num-workers", type=int, default=1)
    up.add_argument("-t", "--tag", required=True, help="run tag (kill_test data tag)")
    up.add_argument("--vcpus", type=int, default=16, help="vCPUs per worker pod")
    up.add_argument("--flavor", default="cpu3c", help="Runpod CPU flavor id")
    up.add_argument("--container-disk-gb", type=int, default=20)
    up.add_argument("--bundle", default="latest", help='bundle_id or "latest"')
    up.add_argument("--games-per-batch", type=int, default=KillTestParams.games_per_batch)
    up.add_argument("--rollouts", type=int, default=KillTestParams.rollouts)
    up.add_argument("--top-k", type=int, default=KillTestParams.top_k)
    up.add_argument("--positions-per-game", type=int, default=KillTestParams.positions_per_game)
    up.add_argument("--open-leaves", action="store_true")
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
