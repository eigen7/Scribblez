"""The environment a worker that boots the image + bundle flow is started
with: an ssh machine's container (dashboard/workers.py), and any launcher of
the same image."""

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
    kind: str,
) -> dict[str, str]:
    """The full environment for a worker that boots the image + bundle flow:
    bucket credentials, the workload's SCZ_* definition, the bundle to run,
    and the slot identity. `kind` travels because the worker cannot infer it
    from its surroundings; the launcher knows what it started."""
    return {
        **r2_env(creds),
        **spec.worker_env(tag, params, role),
        "SCZ_BUNDLE": bundle_id,
        "SCZ_WORKER_ID": worker_id,
        "SCZ_WORKER_KIND": kind,
    }
