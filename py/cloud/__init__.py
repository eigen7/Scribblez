"""Remote workers: the machines, containers and bucket they run through.

Everything that gets a workload running somewhere other than the dev
container: renting AWS machines (providers/), driving Docker on a machine over
ssh (ssh_machine, ssh_transfer), shipping code as bundles (bundles,
runtime_abi), and moving results through the R2 bucket (r2, sinks). The
dashboard and the py/scripts/cloud_* and aws_setup CLIs are the callers;
worker_entrypoint is what runs inside each worker container. Design:
docs/plans/cloud_machines.md.
"""
