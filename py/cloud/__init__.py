"""Remote workers: the machines and containers they run in.

Everything that gets a workload running somewhere other than the dev
container: renting AWS machines (providers/), driving Docker on a machine over
ssh (ssh_machine), shipping code as bundles (bundles, runtime_abi), and moving
results and state between the controller and the containers (ssh_transfer,
sinks). The
dashboard and the py/scripts/cloud_* and aws_setup CLIs are the callers;
worker_entrypoint is what runs inside each worker container. Design:
docs/plans/cloud_machines.md.
"""
