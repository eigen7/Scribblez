"""The optimizer arms' names, and the learning rate each one defaults to.

Two layers need this vocabulary and only one of them may import torch: the
training layer (generational/optim.py, which builds the optimizers) and the
parameter layer (a workload's params dataclass, whose schema drives the
dashboard's task-creation form). The dashboard process is deliberately
torch-free, so the shared names live here rather than in optim.py.

DEFAULT_LR exists because `lr` means different things to the arms -- the
peak of a schedule that decays away from it under `wsd` and `muon`, the
constant an averaged iterate is taken around under `schedule_free` -- so one
default cannot serve all of them. A task leaving `lr` at 0 gets its arm's rate.
"""

OPTIMIZER_WSD = "wsd"
OPTIMIZER_MUON = "muon"
OPTIMIZER_SCHEDULE_FREE = "schedule_free"
OPTIMIZERS = (OPTIMIZER_WSD, OPTIMIZER_MUON, OPTIMIZER_SCHEDULE_FREE)

# The rate each arm runs at when the task does not name one. Muon's updates are
# scaled to AdamW's RMS (generational/muon.py), so it takes the WSD arm's peak.
DEFAULT_LR = {
    OPTIMIZER_WSD: 1e-3,
    OPTIMIZER_MUON: 1e-3,
    OPTIMIZER_SCHEDULE_FREE: 2.5e-3,
}
