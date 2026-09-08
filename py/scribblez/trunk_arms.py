"""The trunk-tower arms' names: the choices of a workload's `trunk` param.

Two layers need this vocabulary and only one of them may import torch: the
model layer (spatial_trunk.py, which builds the tower) and the parameter layer
(a workload's params dataclass, whose schema drives the dashboard's
task-creation form). The dashboard process is deliberately torch-free, so the
names live here rather than in spatial_trunk.py (the optimizer_arms.py
precedent).
"""

TRUNK_CONV = "conv"
TRUNK_TRANSFORMER = "transformer"
TRUNKS = (TRUNK_CONV, TRUNK_TRANSFORMER)
