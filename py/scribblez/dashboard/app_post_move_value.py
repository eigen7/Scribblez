"""Bokeh dashboard for the post-move value model.

Served via ``bokeh serve app_post_move_value.py --args --mount-root <dir>`` (the
post-move trainer / scripts/dashboard.py spawn it). Tabs: Loss, Positions,
Calibration, Training, Streaming.
"""

from scribblez.dashboard.post_move_tabs import LOSS, CalibrationTab, PositionsTab, TrainingTab
from scribblez.dashboard.shell import LossTab, StreamingTab, run
from scribblez.paths import POST_MOVE_VALUE

run(
    [
        LossTab(fallback_groups=LOSS),
        PositionsTab(),
        CalibrationTab(),
        TrainingTab(),
        StreamingTab(),
    ],
    task=POST_MOVE_VALUE,
)
