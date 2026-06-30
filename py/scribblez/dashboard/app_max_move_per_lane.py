"""Bokeh dashboard for the max-move-per-lane model.

Served via ``bokeh serve app_max_move_per_lane.py --args --mount-root <dir>``
(the max-move-per-lane trainer / scripts/dashboard.py spawn it). The task has no
held-out probe/calibration eval, so it shows only the tabs it actually
populates: Loss (per-lane loss components + move/score/has-move accuracy) and
Performance.
"""

from scribblez.dashboard.shell import LossTab, PerformanceTab, run
from scribblez.paths import MAX_MOVE_PER_LANE

run(
    [
        LossTab(),
        PerformanceTab(),
    ],
    task=MAX_MOVE_PER_LANE,
)
