"""The deep_bot workload's trainer side (docs/plans/deep_bot_generations.md):
one trainer over the leave inference model, the leaf network and the policy,
publishing the three together as a release."""

# The three models, by the names their files, record keys and params carry.
LEAVE = "leave"
LEAF = "leaf"
POLICY = "policy"
KINDS = (LEAVE, LEAF, POLICY)
