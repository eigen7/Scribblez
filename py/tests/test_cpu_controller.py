"""Test the generational trainer's live CPU-thread controller."""

import torch
from scribblez.generational.controls import (
    CONTROL_DATALOADER_WORKERS,
    CONTROL_TORCH_THREADS,
    CpuController,
)


class _Recorder:
    def __init__(self):
        self.events = []

    def control_event(self, positions, name, value):
        self.events.append((name, positions, float(value)))


def test_cpu_controller_serves_threads_and_logs_changes():
    recorder = _Recorder()
    controls = {}  # nothing set by the operator yet: the defaults apply
    default_torch = torch.get_num_threads()
    cpu = CpuController(recorder, lambda: controls)
    assert cpu.dataloader_workers == 4
    assert torch.get_num_threads() == default_torch  # applied, unchanged here

    # Construction logs no events; a later change does, at the given rows-clock.
    assert recorder.events == []
    controls[CONTROL_DATALOADER_WORKERS] = 2
    cpu.refresh(500)
    assert cpu.dataloader_workers == 2
    assert recorder.events == [(CONTROL_DATALOADER_WORKERS, 500, 2.0)]
    # A control the file carries as a float (json) is applied as a count.
    controls[CONTROL_TORCH_THREADS] = float(default_torch)
    cpu.refresh(600)
    assert recorder.events == [(CONTROL_DATALOADER_WORKERS, 500, 2.0)]
