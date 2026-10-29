"""The policy-set model (docs/plans/deep_bot_v0.md, "Workload 2: policy_set"):
its labeled pools and its trainer. Light, so the workload registry can import
it without the engine."""

# The suffix of a .pool sidecar (pools.py reads them).
POOL_SUFFIX = ".pool"

# The model's forms (slot.py, greedy.py, policy.py; forms.py registers them),
# by name. Light, so the workload's params can name a `train_<form>` flag for
# each without loading the engine.
FORM_SLOT = "slot"
FORM_GREEDY = "greedy"
FORM_POLICY = "policy"
FORM_NAMES = (FORM_SLOT, FORM_GREEDY, FORM_POLICY)
