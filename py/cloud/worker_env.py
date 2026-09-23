"""The container environment for a worker that runs from a bundle.

The dashboard (scribblez/dashboard/workers.py) passes this to every ssh-slot
container it creates. The image's bootstrap reads SCZ_BUNDLE to fetch the
bundle; the worker entrypoint reads the rest (see its docstring for the list).
"""

from scribblez import workloads

from cloud.credentials import CloudCredentials


def r2_env(creds: CloudCredentials) -> dict[str, str]:
    return {
        "R2_ACCOUNT_ID": creds.r2.account_id,
        "R2_ACCESS_KEY_ID": creds.r2.access_key_id,
        "R2_SECRET_ACCESS_KEY": creds.r2.secret_access_key,
        "R2_BUCKET": creds.r2.bucket,
    }


def bundle_worker_env(
    creds: CloudCredentials,
    spec: workloads.WorkloadSpec,
    tag: str,
    params,
    *,
    role: str,
    bundle_id: str,
    worker_id: str,
) -> dict[str, str]:
    """Bucket credentials, the workload's SCZ_* definition, the bundle to run,
    and the slot's identity. The kind is set explicitly because the worker
    cannot infer it from its sink, which may be local or the bucket."""
    return {
        **r2_env(creds),
        **spec.worker_env(tag, params, role),
        "SCZ_BUNDLE": bundle_id,
        "SCZ_WORKER_ID": worker_id,
        "SCZ_WORKER_KIND": "ssh",
    }
