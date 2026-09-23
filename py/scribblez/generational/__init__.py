"""Shared machinery of the dashboard's training workloads.

The generational pipeline proper (position_eval, max_move_per_lane):
generation directories and their manifests (lifecycle), the controller-side
scheduler that fills and paces them (scheduler), and the sliding training
window. Also used by every trainer, generational or not: the rolling
checkpoint (checkpoint), the optimizer arms and LR schedule (optim, controls),
and the record stream to the dashboard (records, train_ingest).

See docs/generational_training.md for the design.
"""
