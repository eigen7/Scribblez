#!/usr/bin/env python3
"""The one-time AWS setup the dashboard's provider needs, idempotent
(docs/plans/cloud_machines.md): the key pair (private key saved under
/workspace/mount/cloud/aws/), the security group admitting ssh, and the
Deep Learning GPU AMI lookup -- then a report of the account's vCPU quotas
and any pending quota requests, since a quota of 0 is the usual reason a
first launch fails.

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
