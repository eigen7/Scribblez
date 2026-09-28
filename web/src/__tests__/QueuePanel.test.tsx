import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { QueuePanel } from '../components/master/QueuePanel';
import type { Role, Workload } from '../components/master/MasterApp';

// The Workers card's queue panel: what the queue would start for the tag,
// before and after it is enqueued, with Enqueue or Dequeue beside it.

const getJSON = vi.fn();
const postJSON = vi.fn();
vi.mock('../lib/api', () => ({
  getJSON: (...a: unknown[]) => getJSON(...a),
  postJSON: (...a: unknown[]) => postJSON(...a),
}));
vi.mock('../components/master/TaskView', () => ({ default: () => null }));

const role = (name: string, title: string): Role => ({
  name, title, singleton: false, kinds: ['local', 'ssh'], gpu: false, stats: null,
});
const workload: Workload = {
  name: 'position_eval', title: 'Train position evaluation',
  roles: [role('train', 'Trainer'), role('generate', 'Generator'), role('match_eval', 'Match eval')],
  primary_params: [], params: [], profiles: {}, default_profile: '',
};

const plan = {
  roles: ['train', 'generate'],
  machines: [
    { machine: 'localhost', slots: [{ role: 'train', threads: null }, { role: 'generate', threads: 28 }], gpu_gb: 14.03, refusal: null },
    { machine: 'asus', slots: [{ role: 'train', threads: null }, { role: 'generate', threads: 12 }], gpu_gb: 14.03, refusal: 'needs 14.0 GiB of GPU memory, has 3.7' },
  ],
};

describe('the queue panel', () => {
  beforeEach(() => {
    getJSON.mockReset();
    postJSON.mockReset();
  });

  it('shows the roles and per-machine slots before enqueueing, and enqueues', async () => {
    getJSON.mockResolvedValue(plan);
    postJSON.mockResolvedValue({ queued: true, warnings: [] });
    const onChanged = vi.fn();
    render(<QueuePanel workload={workload} tag="tune-a" queued={null} onChanged={onChanged} />);
    const panel = await screen.findByTestId('queue-panel');
    expect(panel.textContent).toContain('would run Trainer + Generator');
    expect(panel.textContent).toContain('localhost: Trainer, Generator (28 threads); 14.0 GiB GPU');
    expect(panel.textContent).toContain('asus: Trainer, Generator (12 threads); 14.0 GiB GPU — cannot take it: needs 14.0 GiB');
    expect(getJSON).toHaveBeenCalledWith('/api/queue/plan?workload=position_eval&tag=tune-a');
    fireEvent.click(screen.getByText('Enqueue'));
    await waitFor(() => expect(onChanged).toHaveBeenCalled());
    expect(postJSON).toHaveBeenCalledWith('/api/queue/enqueue', { workload: 'position_eval', tag: 'tune-a' });
  });

  it('shows its place once queued, and dequeues from the same spot', async () => {
    getJSON.mockResolvedValue(plan);
    postJSON.mockResolvedValue({ ok: true });
    const onChanged = vi.fn();
    render(<QueuePanel workload={workload} tag="tune-a" queued={2} onChanged={onChanged} />);
    const panel = await screen.findByTestId('queue-panel');
    expect(panel.textContent).toContain('queued #2');
    expect(panel.textContent).toContain('will run Trainer + Generator');
    expect(screen.queryByText('Enqueue')).toBeNull();
    fireEvent.click(screen.getByText('Dequeue'));
    await waitFor(() => expect(onChanged).toHaveBeenCalled());
    expect(postJSON).toHaveBeenCalledWith('/api/queue/action', {
      workload: 'position_eval', tag: 'tune-a', action: 'dequeue',
    });
  });

  it('says so when the pool is empty, and hides for an unqueueable workload', async () => {
    getJSON.mockResolvedValue({ roles: ['train', 'generate'], machines: [] });
    const { unmount } = render(<QueuePanel workload={workload} tag="t" queued={null} onChanged={() => {}} />);
    expect((await screen.findByTestId('queue-panel')).textContent).toContain('no machines or rental capacity');
    unmount();
    getJSON.mockRejectedValue(new Error('position_eval tags are not queueable yet (no layout)'));
    render(<QueuePanel workload={workload} tag="t" queued={null} onChanged={() => {}} />);
    await waitFor(() => expect(getJSON).toHaveBeenCalledTimes(2));
    expect(screen.queryByTestId('queue-panel')).toBeNull();
  });
});
