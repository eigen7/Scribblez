import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import TaskView from '../components/master/TaskView';
import type { Workload } from '../components/master/MasterApp';

// The Workers card's Remove controls: disabled, with the server's reason as the
// tooltip, wherever the server would refuse the removal (workers.py
// remove_worker) -- so a refused click is not a silent no-op.

const getJSON = vi.fn();
vi.mock('../lib/api', () => ({
  getJSON: (...a: unknown[]) => getJSON(...a),
  postJSON: vi.fn(),
}));
// The workload and Stats tabs pull in bokehjs, which needs a real canvas.
vi.mock('../workloads', () => ({ WORKLOAD_TABS: {} }));
vi.mock('../components/master/StatsTab', () => ({ default: () => null }));

const workload: Workload = {
  name: 'deep_bot_leaf_eval', title: 'Leaf eval',
  roles: [], primary_params: [], params: [], profiles: {}, default_profile: '',
};

const worker = (worker_id: string, state: string) => ({
  worker_id, role: 'generate', kind: 'ssh', desired_state: 'paused', state,
  observed_running: false, threads: null, host: 'ubuntu@10.0.0.1', machine: 'aws',
  bundle_id: null, undelivered: null, launched: true, sweeps_home: false,
});

const machine = (state: string) => ({
  name: 'aws', provider: 'aws', host: 'ubuntu@10.0.0.1', gpu_count: 1,
  instance_type: 'g6.2xlarge', instance_id: 'i-0', spot: false, cost_per_hr: 1,
  spend: 0, state, slots: ['ssh-0', 'ssh-1'], pool: false,
});

const setup = async (machineState: string, workerState: string) => {
  getJSON.mockResolvedValue({
    workload: workload.name, tag: 'v3', has_task: true, params: null, profile: '',
    profile_diff: [], created_at: 0, state: 'paused', progress: [], failure: null,
    gates: {}, data_dir: '/d', workers: [worker('ssh-0', workerState), worker('ssh-1', workerState)],
    machines: [machine(machineState)], spend: 0, queued: null, bundle_id: null,
    bundle_drift: false,
  });
  render(<TaskView workload={workload} tag="v3" />);
  await waitFor(() => screen.getByText('Remove all'));
};

const button = (label: string) => screen.getAllByText(label)[0].closest('button') as HTMLButtonElement;
const tooltip = (label: string) => button(label).closest('span[title]')?.getAttribute('title');

describe('the Workers card’s Remove controls', () => {
  beforeEach(() => getJSON.mockReset());

  it('point a stopped rental at removing its machine', async () => {
    await setup('stopped', 'unreachable');
    expect(button('Remove all').disabled).toBe(true);
    expect(tooltip('Remove all')).toMatch(/^ssh-0: machine aws is stopped; remove the machine/);
    expect(button('Remove').disabled).toBe(true);
    expect(tooltip('Remove')).toMatch(/^machine aws is stopped/);
  });

  it('say to wait while the rental is stopping', async () => {
    await setup('stopping', 'unreachable');
    expect(button('Remove all').disabled).toBe(true);
    expect(tooltip('Remove all')).toMatch(/machine aws is stopping/);
  });

  it('refuse a slot on a host that does not answer', async () => {
    await setup('up', 'unreachable');
    expect(tooltip('Remove all')).toMatch(/ubuntu@10\.0\.0\.1 is unreachable/);
  });

  it('remove the slots of a gone rental, whose containers went with its disk', async () => {
    await setup('gone', 'unreachable');
    expect(button('Remove all').disabled).toBe(false);
    expect(button('Remove').disabled).toBe(false);
  });

  it('remove paused slots on a machine that is up', async () => {
    await setup('up', 'paused');
    expect(button('Remove all').disabled).toBe(false);
  });
});
