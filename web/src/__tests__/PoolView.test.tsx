import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import PoolView, { type PoolMachine } from '../components/master/PoolView';

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

describe('PoolView', () => {
  beforeEach(() => {
    getJSON.mockReset();
    postJSON.mockReset();
  });

  it('shows each machine with its GPU and what holds it', async () => {
    getJSON.mockResolvedValue({ machines: [
      machine({}),
      machine({ name: 'localhost', kind: 'local', machine: null, occupants: ['move_set_eval/x/local-0'], state: 'busy' }),
      machine({ name: 'l4', lease: { workload: 'position_eval', tag: 'tune-wsd', phase: 'running', since: 0 }, state: 'leased' }),
    ] });
    render(<PoolView />);
    await waitFor(() => expect(screen.getByTestId('pool-state-asus').textContent).toBe('free'));
    expect(screen.getByTestId('pool-state-localhost').textContent).toBe('busy: move_set_eval/x/local-0');
    expect(screen.getByTestId('pool-state-l4').textContent).toBe('position_eval/tune-wsd (running)');
    expect(screen.getAllByText('1 × 4.0 GiB').length).toBe(3);
  });

  it('adds a registered machine with its aliases and reserve', async () => {
    getJSON.mockResolvedValue({ machines: [] });
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
