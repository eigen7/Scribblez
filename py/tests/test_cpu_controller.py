"""Test the generational trainer's live CPU-thread controller."""

import torch
from scribblez.dashboard import db
from scribblez.generational.controls import (
    CONTROL_DATALOADER_WORKERS,
    CONTROL_TORCH_THREADS,
    CpuController,
)


def test_cpu_controller_serves_threads_and_logs_changes(tmp_path):
    conn = db.connect(tmp_path / "dashboard.db")
    default_torch = torch.get_num_threads()
    db.init_control(
        conn,
        {
            CONTROL_DATALOADER_WORKERS: 4,
            CONTROL_TORCH_THREADS: default_torch,
        },
    )
    cpu = CpuController(conn)
    assert cpu.dataloader_workers == 4
    assert torch.get_num_threads() == default_torch  # applied, unchanged here

    # Construction logs no events; a later change does, at the given rows-clock.
    assert db.read_control_events(conn) == []
    db.write_control(conn, CONTROL_DATALOADER_WORKERS, 2)
    cpu.refresh(500)
    assert cpu.dataloader_workers == 2
    events = {(e["name"], e["positions"], e["value"]) for e in db.read_control_events(conn)}
    assert (CONTROL_DATALOADER_WORKERS, 500, 2.0) in events
