"""The leaf_bot_generation workload's trainer and measure sides
(workloads/leaf_bot_generation.py): one generation step of DeepBot, its leave
inference model and leaf network trained on DeepBot's own self-play."""

from scribblez.deep_bot import LEAF, LEAVE
from scribblez.deep_bot.release import Layout

WORKLOAD = "leaf_bot_generation"
# The two models, by the names their files, record keys and params carry.
KINDS = (LEAVE, LEAF)
# A release holds the two exports and nothing else: the measure slots analyze
# the known positions, not the trainer.
LAYOUT = Layout(KINDS)
