"""The policy-set model (docs/plans/deep_bot_v0.md, "Workload 2: policy_set"):
its labeled pools and its trainer. Light, so the workload registry can import
it without the engine."""

# The suffix of a .pool sidecar (pools.py reads them).
POOL_SUFFIX = ".pool"

# The model's forms (slot.py, greedy.py, policy.py; forms.py registers them),
# by name: light, so the workload registry can offer them as the `forms`
# param's choices. "both" trains the slot and greedy forms side by side, "all"
# every form.
FORM_SLOT = "slot"
FORM_GREEDY = "greedy"
FORM_POLICY = "policy"
FORM_BOTH = "both"
FORM_ALL = "all"
FORM_CHOICES = (FORM_SLOT, FORM_GREEDY, FORM_POLICY, FORM_BOTH, FORM_ALL)
