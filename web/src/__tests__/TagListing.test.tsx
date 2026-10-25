import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { HomePage, type Workload } from '../components/master/MasterApp';

// What the tag listing shows when it cannot list a workload's tags, and that a
// switch of workload never leaves the previous one's rows under the new heading.

const getJSON = vi.fn();
vi.mock('../lib/api', () => ({
  getJSON: (...a: unknown[]) => getJSON(...a),
  postJSON: vi.fn(),
}));
vi.mock('../components/master/TaskView', () => ({ default: () => null }));

const workload = (name: string): Workload => ({
  name, title: name, roles: [], primary_params: [], params: [], profiles: {}, default_profile: '',
});

const row = (tag: string, params_error: string | null = null) => ({
  tag, has_task: true, created_at: 0, workers: 0, active_workers: 0,
  state: params_error ? 'invalid params' : 'idle', progress: [] as [string, string | number][],
  last_active: 0, pace: null, disk_bytes: 0, params_error,
});

describe('the tag listing', () => {
  beforeEach(() => { getJSON.mockReset(); });

  it('shows a failed fetch instead of the previous workload’s rows', async () => {
    getJSON.mockResolvedValueOnce({ tags: [row('tune-a')] });
    const { rerender } = render(<HomePage workload={workload('position_eval')} onOpen={() => {}} />);
    await waitFor(() => screen.getByText('tune-a'));

    getJSON.mockRejectedValue(new Error('unknown parameter'));
    rerender(<HomePage workload={workload('policy_set')} onOpen={() => {}} />);
    expect(screen.queryByText('tune-a')).toBeNull();
    await waitFor(() => screen.getByText(/Could not list tags: .*unknown parameter/));
  });

  it('shows a tag’s params error on its row', async () => {
    getJSON.mockResolvedValue({ tags: [row('ok'), row('stale', "unknown parameter 'x'")] });
    render(<HomePage workload={workload('policy_set')} onOpen={() => {}} />);
    await waitFor(() => screen.getByText("unknown parameter 'x'"));
    expect(screen.getByText('invalid params')).toBeTruthy();
    expect(screen.getByText('ok')).toBeTruthy();
  });
});
