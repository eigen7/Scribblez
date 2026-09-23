#!/usr/bin/env python3
"""Prepare an AWS account for the dashboard's rented machines. Idempotent.

Creates the ssh key pair (private key saved under /workspace/mount/cloud/aws/)
and the security group admitting ssh, and looks up the Deep Learning GPU AMI.
The provider repeats this before every launch anyway; running it here up front
also reports the account's vCPU quotas and pending quota requests, because a
quota of 0 is the usual reason a first launch fails. See
docs/plans/cloud_machines.md.

Usage:
    ./py/scripts/aws_setup.py
"""

from cloud.credentials import CredentialsError, load_credentials
from cloud.providers.aws import AwsProvider
from cloud.providers.base import ProviderError


def main() -> int:
    try:
        creds = load_credentials()
    except CredentialsError as e:
        print(e)
        return 1
    provider = AwsProvider(creds.aws, creds.registry)
    try:
        print(f"identity: {provider.identity()} in {provider.region}")
        provider.prepare()
        print(f"key pair: {provider.identity_file}")
        print(f"security group: {provider._security_group_id()}")
        print(f"AMI: {provider.ami()}")
        for name, value in provider.quotas().items():
            print(f"quota {name}: {value:g}")
        for quota, wanted, status in provider.quota_requests():
            print(f"request {quota}: {wanted:g} ({status})")
        instances = provider.describe()
        print(f"instances tagged ours: {len(instances)}")
        for inst in instances.values():
            print(f"  {inst.id} {inst.type_id} {inst.state} owner={inst.owner}")
    except ProviderError as e:
        print(f"FAIL {e}: {e.detail}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
