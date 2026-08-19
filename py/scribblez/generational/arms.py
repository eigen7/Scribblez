"""The optimizer arms' names, and the parameter defaults each one implies.

Two layers need this vocabulary and only one of them may import torch: the
training layer (generational/optim.py, which builds the optimizers) and the
parameter layer (a workload's params dataclass, whose schema drives the
dashboard's task-creation form). The dashboard process is deliberately
torch-free, so the shared names live here rather than in optim.py.

ARM_DEFAULTS carries what a choice implies for the *other* parameters: the two
arms want different learning rates, because `lr` means different things to them
-- the peak of a schedule that decays away from it under `wsd`, and the
constant an averaged iterate is taken around under `schedule_free`. Declaring
that here is what lets the form re-seed the rate when the arm changes, instead
of leaving the operator to remember which number goes with which arm.
"""

OPTIMIZER_WSD = "wsd"
OPTIMIZER_SCHEDULE_FREE = "schedule_free"

# Choice -> the defaults it implies for other parameters.
ARM_DEFAULTS = {
    OPTIMIZER_WSD: {"lr": 1e-3},
    OPTIMIZER_SCHEDULE_FREE: {"lr": 2.5e-3},
}
