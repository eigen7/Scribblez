"""Shared runtime pieces for role runners, the loops workers execute.

The worker entrypoint (py/cloud/worker_entrypoint.py) owns process concerns
(env parsing, sink construction, the SIGTERM handler) and hands a
WorkerContext to the role's runner. Runners use two helpers from here:
WorkerStopped, which the SIGTERM handler raises to unwind any loop, and
WorkerStats, the per-worker record behind the dashboard's Stats tab.
"""

import time


class WorkerStopped(Exception):
    """Raised out of a runner's loop by the entrypoint's SIGTERM handler."""


# Per-cycle timing samples retained in the published stats record; enough for
# recent-throughput estimates without unbounded growth.
RECENT_SAMPLES = 50

# Cumulative-count points retained for the whole run: once the history holds
# this many, every other point is dropped, so a long run keeps a coarser but
# complete timeline at bounded size.
HISTORY_POINTS = 1000


def thin_history(history: list) -> list:
    """Halve `history` when it reaches the cap, keeping its latest point."""
    if len(history) < HISTORY_POINTS:
        return history
    return history[-1::-2][::-1]


def stats_rel_path(worker_id: str) -> str:
    """Where a worker's stats record lives, relative to the tag root."""
    return f"stats/{worker_id}.json"


class WorkerStats:
    """The per-worker stats record, published after every cycle: cumulative
    counters, a bounded window of per-cycle samples carrying the role's timing
    phases (RoleSpec.stats), and a [time, units_total] history. The Stats tab
    derives throughput and bottleneck breakdowns from it.

    The counters belong to the slot, not the process: a worker resumes the
    totals it last published, so the scheduler's pacing gate, which stops and
    restarts generators many times an hour, reads as a pause rather than lost
    work. The history is resumed too, and thinned rather than capped, so it
    covers the slot's whole run. The sample window starts empty: it measures
    the current rate, which samples from before a gap would only blur.
    """

    def __init__(self, ctx):
        self._sink = ctx.sink
        prior = ctx.sink.read_json(stats_rel_path(ctx.worker_id)) or {}
        now = time.time()
        self._record = {
            "worker_id": ctx.worker_id,
            "kind": ctx.kind,
            "role": ctx.role.name,
            "threads": ctx.threads,
            "started_at": prior.get("started_at", now),  # the slot's first start
            "updated_at": now,
            "units_total": prior.get("units_total", 0),
            "cycles_total": prior.get("cycles_total", 0),
            "recent": [],
            "history": prior.get("history", []),
            **ctx.provenance,
        }

    def cycle_done(self, phases: dict[str, float], units: int, nbytes: int):
        """Record one completed cycle (`phases` keyed as in the role's
        StatsSpec) and publish the updated record through the sink."""
        r = self._record
        r["cycles_total"] += 1
        r["units_total"] += units
        r["updated_at"] = time.time()
        r["recent"].append(
            {
                "t": r["updated_at"],
                "units": units,
                "units_total": r["units_total"],
                "bytes": nbytes,
                **{k: round(v, 3) for k, v in phases.items()},
            }
        )
        del r["recent"][:-RECENT_SAMPLES]
        r["history"] = thin_history(r["history"] + [[r["updated_at"], r["units_total"]]])
        self._sink.push_json(stats_rel_path(r["worker_id"]), r)
