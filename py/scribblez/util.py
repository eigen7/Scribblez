"""Small, generic, dependency-free helpers shared across scripts and modules.

Anything here is pure and domain-agnostic: it knows nothing about Scribblez,
models, tags, or training. Helpers tied to a specific subsystem belong in that
subsystem's module instead.
"""


def fmt_duration(secs: float) -> str:
    """Compact duration: "1h05m", "5m12s", or "42s"."""
    s = int(secs + 0.5)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"
