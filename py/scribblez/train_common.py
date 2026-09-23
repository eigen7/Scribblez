"""Small helpers shared by the task-specific trainers."""

from datetime import datetime


def timed_print(msg: str):
    """Print `msg` prefixed with a millisecond-resolution local timestamp."""
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} {msg}")
