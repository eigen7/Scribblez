"""Generational training pipeline.

Generation lifecycle (directory/manifest bookkeeping, sliding window, restart
reconciliation) and the rows-clock learning-rate control that the generational
orchestrator is built on. See docs/generational_training.md for the design.
"""
