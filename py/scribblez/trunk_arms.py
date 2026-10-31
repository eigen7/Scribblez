"""The choices of a workload's `trunk` param, and the help text of its
`normalize_scalar` param.

These live in a torch-free module because the dashboard process, which
builds its task-creation form from the workloads' params dataclasses, must not
import torch. spatial_trunk.py builds the tower each name selects.
(generational/optimizer_arms.py follows the same pattern.)
"""

TRUNK_CONV = "conv"
TRUNK_TRANSFORMER = "transformer"
TRUNKS = (TRUNK_CONV, TRUNK_TRANSFORMER)

# The help text of a workload's `normalize_scalar` param.
NORMALIZE_SCALAR_HELP = (
    "layer-normalize the board trunk's scalar projection (spatial_trunk.SpatialTrunk) so its "
    "scale cannot outgrow the board's features; off only in tags trained before it existed"
)
