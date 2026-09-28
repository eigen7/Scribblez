import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import PoolView, { enqueueTag, type PoolMachine, type QueueRow } from '../components/master/PoolView';

// The pool page: each machine's hardware and why it is or is not free, and the
// add form's request.

const getJSON = vi.fn();
const postJSON = vi.fn();
vi.mock('../lib/api', () => ({
  getJSON: (...a: unknown[]) => getJSON(...a),
  postJSON: (...a: unknown[]) => postJSON(...a),
}));
// PoolView shares MasterApp's Button, and MasterApp pulls in the task view's
// charts, which need a canvas; as in NewTagForm.test.tsx.
vi.mock('../components/master/TaskView', () => ({ default: () => null }));

const machine = (over: Partial<PoolMachine>): PoolMachine => ({
  name: 'asus', kind: 'ssh', machine: { host: 'asus-laptop', identity_file: null },
  aliases: [], hardware: { vcpus: 12, gpu_count: 1, gpu_memory_gb: 4 },
  gpu_reserve_gb: 0, generator_threads: null, lease: null, occupants: [], state: 'free',
  ...over,
});

// Answer the page's two polls: the pool, and the queue.
const serve = (machines: PoolMachine[], entries: QueueRow[] = []) =>
  getJSON.mockImplementation((url: string) =>
    Promise.resolve(url === '/api/queue' ? { entries } : { machines }));

const row = (over: Partial<QueueRow>): QueueRow => ({
  workload: 'position_eval', tag: 'tune-a', machines: [], memory_override_gb: null,
  bundle: 'none', end_condition: true, refusals: {}, ...over,
});

describe('PoolView', () => {
  beforeEach(() => {
    getJSON.mockReset();
    postJSON.mockReset();
  });

  it('shows each machine with its GPU and what holds it', async () => {
    serve([
      machine({}),
      machine({ name: 'localhost', kind: 'local', machine: null, occupants: ['move_set_eval/x/local-0'], state: 'busy' }),
      machine({ name: 'l4', lease: { workload: 'position_eval', tag: 'tune-wsd', phase: 'running', since: 0, reason: '' }, state: 'leased' }),
    ]);
    render(<PoolView />);
    await waitFor(() => expect(screen.getByTestId('pool-state-asus').textContent).toBe('free'));
    expect(screen.getByTestId('pool-state-localhost').textContent).toBe('busy: move_set_eval/x/local-0');
    expect(screen.getByTestId('pool-state-l4').textContent).toBe('position_eval/tune-wsd (running)');
    expect(screen.getAllByText('1 × 4.0 GiB').length).toBe(3);
  });

  it('adds a registered machine with its aliases and reserve', async () => {
    serve([]);
    postJSON.mockResolvedValue({ name: 'asus' });
    render(<PoolView />);
    fireEvent.change(screen.getByLabelText('pool name'), { target: { value: 'asus' } });
    fireEvent.change(screen.getByLabelText('pool host'), { target: { value: 'asus-laptop' } });
    fireEvent.change(screen.getByLabelText('pool aliases'), { target: { value: 'dshin@asus-laptop' } });
    fireEvent.click(screen.getByText('Add to pool'));
    await waitFor(() => expect(postJSON).toHaveBeenCalledWith('/api/pool/machines', {
      name: 'asus', host: 'asus-laptop', aliases: 'dshin@asus-laptop', gpu_reserve_gb: '0',
    }));
  });
});

describe('the queue', () => {
  beforeEach(() => {
    getJSON.mockReset();
    postJSON.mockReset();
  });

  it('lists entries in order, marking endless ones and saying why they wait', async () => {
    serve([machine({})], [
      row({ tag: 'tune-a', refusals: { asus: 'needs 14.0 GiB of GPU memory, has 4.0' } }),
      row({ tag: 'endless', end_condition: false }),
    ]);
    render(<PoolView />);
    await waitFor(() => expect(screen.getByTestId('queue-tune-a').textContent).toBe('position_eval/tune-a'));
    expect(screen.getByTestId('queue-endless').textContent).toBe('position_eval/endless ∞');
    expect(screen.getByText('asus: needs 14.0 GiB of GPU memory, has 4.0')).toBeTruthy();
  });

  it('enqueues after the operator confirms the warnings', async () => {
    postJSON
      .mockResolvedValueOnce({ queued: false, warnings: ['no end condition: position_eval/x'] })
      .mockResolvedValueOnce({ queued: true, warnings: [] });
    const confirm = vi.spyOn(window, 'confirm').mockReturnValue(true);
    expect(await enqueueTag('position_eval', 'x')).toBe(true);
    expect(confirm.mock.calls[0][0]).toContain('no end condition');
    expect(postJSON).toHaveBeenLastCalledWith('/api/queue/enqueue', { workload: 'position_eval', tag: 'x', confirm: true });

    postJSON.mockReset();
    postJSON.mockResolvedValueOnce({ queued: false, warnings: ['w'] });
    confirm.mockReturnValue(false);
    expect(await enqueueTag('position_eval', 'x')).toBe(false);
    expect(postJSON).toHaveBeenCalledTimes(1);
    confirm.mockRestore();
  });
});

describe('PoolView row actions', () => {
  beforeEach(() => {
    getJSON.mockReset();
    postJSON.mockReset();
  });

  it('saves an edit with the edited fields', async () => {
    serve([machine({})]);
    postJSON.mockResolvedValue({ ok: true });
    render(<PoolView />);
    await waitFor(() => screen.getByTestId('pool-state-asus'));
    fireEvent.click(screen.getByText('Edit'));
    fireEvent.change(screen.getByLabelText('asus reserve'), { target: { value: '1.5' } });
    fireEvent.change(screen.getByLabelText('asus threads'), { target: { value: '6' } });
    fireEvent.click(screen.getByText('Save'));
    await waitFor(() => expect(postJSON).toHaveBeenCalledWith('/api/pool/machine_action', {
      name: 'asus', action: 'edit', aliases: '', gpu_reserve_gb: '1.5', generator_threads: '6',
    }));
  });

  it('refuses Remove on a leased machine and shows who else is on it', async () => {
    serve([machine({
      lease: { workload: 'position_eval', tag: 'a', phase: 'running', since: 0, reason: '' },
      occupants: ['position_eval/hand/local-0'], state: 'leased',
    })]);
    render(<PoolView />);
    await waitFor(() => expect(screen.getByTestId('pool-state-asus').textContent).toBe(
      'position_eval/a (running); also position_eval/hand/local-0',
    ));
    expect((screen.getByText('Remove').closest('button') as HTMLButtonElement).disabled).toBe(true);
  });
});
