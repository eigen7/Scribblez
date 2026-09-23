"""The choices of a workload's `trunk` param.

These names live in a torch-free module because the dashboard process, which
builds its task-creation form from the workloads' params dataclasses, must not
import torch. spatial_trunk.py builds the tower each name selects.
(generational/optimizer_arms.py follows the same pattern.)
"""

TRUNK_CONV = "conv"
TRUNK_TRANSFORMER = "transformer"
TRUNKS = (TRUNK_CONV, TRUNK_TRANSFORMER)
